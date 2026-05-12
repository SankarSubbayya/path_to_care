// POST /api/triage — accepts a narrative + (optional) image, calls the
// MI300X-hosted vLLM endpoint, parses the structured JSON the model emits,
// applies cardinal-rule rewrites + the rule-based safety net, and returns
// a single TriageResult JSON consumed by the patient/clinician/audit views.

import { NextRequest, NextResponse } from "next/server";
import {
  enforceCardinalRule,
  crossCheckRedFlags,
} from "@/lib/cardinal-rule";
import { DEFAULT_VILLAGE } from "@/lib/village";
import type { TriageResult, ConditionGuess, SoapFields, Urgency, ToolInvocation } from "@/lib/types";

// JS half of the camera_capture MCP (mcp/camera_capture/server.py).
// Decodes the uploaded frame, reads PNG/JPEG dimensions from headers, and
// returns a ToolInvocation that the audit tab can render. Kept inline so the
// edge runtime doesn't need a Python sidecar — the Python module is the
// canonical reference impl, this mirrors its shape.
function readImageDims(buf: Buffer): { width: number; height: number; mime: string } {
  // PNG: 8-byte signature, then IHDR with width/height as big-endian uint32s
  if (
    buf.length >= 24 &&
    buf[0] === 0x89 && buf[1] === 0x50 && buf[2] === 0x4e && buf[3] === 0x47
  ) {
    return {
      width: buf.readUInt32BE(16),
      height: buf.readUInt32BE(20),
      mime: "image/png",
    };
  }
  // JPEG: scan SOF0/SOF2 markers for dimensions.
  if (buf.length >= 4 && buf[0] === 0xff && buf[1] === 0xd8) {
    let i = 2;
    while (i < buf.length - 9) {
      if (buf[i] !== 0xff) { i += 1; continue; }
      const marker = buf[i + 1];
      const segLen = buf.readUInt16BE(i + 2);
      // SOF markers (0xC0..0xCF except 0xC4, 0xC8, 0xCC)
      if (marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc) {
        const h = buf.readUInt16BE(i + 5);
        const w = buf.readUInt16BE(i + 7);
        return { width: w, height: h, mime: "image/jpeg" };
      }
      i += 2 + segLen;
    }
    return { width: 0, height: 0, mime: "image/jpeg" };
  }
  return { width: 0, height: 0, mime: "application/octet-stream" };
}

const VLLM_BASE_URL =
  process.env.PTC_VLLM_GEMMA4_URL ?? "http://165.245.137.117:8000/v1";
const VLLM_API_KEY = process.env.PTC_VLLM_API_KEY ?? "ptc-demo-2026-amd";
const VLLM_MODEL =
  process.env.PTC_VLLM_GEMMA4_MODEL_ID ?? "google/gemma-4-31B-it";

const SYSTEM_PROMPT = `You are a triage decision-support assistant for a rural community health worker in the Global South. The patient has sent a phone photo (if attached) and a typed narrative. You DO NOT diagnose. Your job is to produce decision-support output.

Output a single JSON object with these keys exactly. No prose before or after the JSON.

{
  "image_top3": [
    {"condition": "...", "confidence": 0.0-1.0},
    {"condition": "...", "confidence": 0.0-1.0},
    {"condition": "...", "confidence": 0.0-1.0}
  ],
  "soap": {
    "chief_complaint": "<one short phrase>",
    "hpi": "<1-3 sentences of history>",
    "duration": "<e.g. '2 days'>",
    "associated_symptoms": ["..."],
    "past_medical_history": ["none stated"],
    "medications": ["none stated"],
    "vitals": {},
    "exam_findings": ["..."],
    "red_flags": ["..."],
    "patient_concerns": ["..."]
  },
  "urgency": "red" | "yellow" | "green",
  "reasoning": "<2-4 sentences. Start with 'Signs suggest...'. Cite specific red flags. Never say 'you have X' or 'this is X'>",
  "red_flags_noted": ["..."],
  "patient_framing": "<1-2 sentences in plain language for the patient. Frame as cost-benefit relative to their wage and distance.>"
}

Hard rules (cardinal):
- image_top3 must always have exactly 3 entries with descending confidence.
- Never produce single-class output. Never say "you have X". Use "signs suggest", "consistent with", "the appearance is".
- urgency:
    red    = immediate same-day care needed
    yellow = clinical evaluation in 1-2 days
    green  = monitor at home, return if it worsens
- patient_framing must reference the cost / time tradeoff using numbers from VILLAGE_CONTEXT.

VILLAGE_CONTEXT:
${DEFAULT_VILLAGE.blurb}`;

interface VllmContentPart {
  type: "text" | "image_url";
  text?: string;
  image_url?: { url: string };
}

function extractJsonObject(text: string): unknown | null {
  // Greedy match for the first balanced {...} block.
  const start = text.indexOf("{");
  if (start < 0) return null;
  let depth = 0;
  let inStr = false;
  let escape = false;
  for (let i = start; i < text.length; i++) {
    const c = text[i];
    if (inStr) {
      if (escape) { escape = false; continue; }
      if (c === "\\") { escape = true; continue; }
      if (c === '"') inStr = false;
      continue;
    }
    if (c === '"') { inStr = true; continue; }
    if (c === "{") depth += 1;
    else if (c === "}") {
      depth -= 1;
      if (depth === 0) {
        const blob = text.slice(start, i + 1);
        try { return JSON.parse(blob); } catch { return null; }
      }
    }
  }
  return null;
}

function normalizeUrgency(u: unknown): Urgency {
  if (typeof u !== "string") return "green";
  const l = u.trim().toLowerCase();
  if (l === "red" || l === "yellow" || l === "green") return l;
  return "green";
}

function clipConfidence(c: unknown): number {
  const n = typeof c === "number" ? c : Number(c);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(1, n));
}

function normalizeTop3(arr: unknown): ConditionGuess[] {
  if (!Array.isArray(arr)) return [];
  return arr.slice(0, 3).map((x) => {
    const obj = (x ?? {}) as Record<string, unknown>;
    return {
      condition: String(obj.condition ?? "unknown") || "unknown",
      confidence: clipConfidence(obj.confidence),
    };
  });
}

function strList(x: unknown): string[] {
  if (!Array.isArray(x)) return [];
  return x.map((s) => String(s)).filter((s) => s.length > 0);
}

function normalizeSoap(s: unknown): SoapFields {
  const o = (s ?? {}) as Record<string, unknown>;
  return {
    chief_complaint: typeof o.chief_complaint === "string" ? o.chief_complaint : "",
    hpi: typeof o.hpi === "string" ? o.hpi : "",
    duration: typeof o.duration === "string" ? o.duration : "",
    associated_symptoms: strList(o.associated_symptoms),
    past_medical_history: strList(o.past_medical_history),
    medications: strList(o.medications),
    vitals: (o.vitals && typeof o.vitals === "object" ? (o.vitals as Record<string, string | number>) : {}),
    exam_findings: strList(o.exam_findings),
    red_flags: strList(o.red_flags),
    patient_concerns: strList(o.patient_concerns),
  };
}

export async function POST(req: NextRequest): Promise<NextResponse> {
  const t0 = Date.now();
  try {
    const form = await req.formData();
    const narrative = String(form.get("narrative") ?? "").trim();
    const imageDescription = String(form.get("image_description") ?? "").trim();
    const imageFile = form.get("image");

    if (!narrative && !imageDescription) {
      return NextResponse.json(
        { error: "Either narrative or image_description is required." },
        { status: 400 },
      );
    }

    // Build user content (text + optional image data URL)
    const userParts: VllmContentPart[] = [];
    const toolInvocations: ToolInvocation[] = [];
    let imageBytes = 0;
    const imageSource = String(form.get("image_source") ?? "none");
    if (imageFile && imageFile instanceof File && imageFile.size > 0) {
      const buf = Buffer.from(await imageFile.arrayBuffer());
      imageBytes = buf.length;
      const mime = imageFile.type || "image/png";
      const dataUrl = `data:${mime};base64,${buf.toString("base64")}`;
      userParts.push({ type: "image_url", image_url: { url: dataUrl } });

      // camera_capture MCP invocation — runs whether the frame came from
      // getUserMedia (image_source=camera) or the file picker (image_source=upload).
      // The browser's CameraCapture component is the producer; this tool entry
      // is the consumer that the audit tab renders.
      const dims = readImageDims(buf);
      toolInvocations.push({
        name: "camera_capture",
        ok: dims.width > 0 && dims.height > 0,
        meta: {
          source: imageSource,
          mime: dims.mime,
          width: dims.width,
          height: dims.height,
          bytes_in: buf.length,
          file_name: imageFile.name,
        },
      });
    }
    const userText =
      `Patient narrative: ${narrative}\n\n` +
      (imageDescription ? `Photo description (text proxy if no image attached): ${imageDescription}\n\n` : "") +
      `Now produce the JSON.`;
    userParts.push({ type: "text", text: userText });

    const body = {
      model: VLLM_MODEL,
      messages: [
        { role: "system", content: [{ type: "text", text: SYSTEM_PROMPT }] },
        { role: "user", content: userParts },
      ],
      max_tokens: 4096,
      temperature: 0,
    };

    const vllmResp = await fetch(`${VLLM_BASE_URL}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${VLLM_API_KEY}`,
      },
      body: JSON.stringify(body),
      // Reasonable upstream timeout.
      signal: AbortSignal.timeout(120_000),
    });

    if (!vllmResp.ok) {
      const errText = await vllmResp.text();
      return NextResponse.json(
        { error: `vLLM ${vllmResp.status}: ${errText.slice(0, 400)}` },
        { status: 502 },
      );
    }

    const vllmJson = (await vllmResp.json()) as {
      choices: Array<{ message: { content: string } }>;
    };
    const raw = vllmJson.choices?.[0]?.message?.content ?? "";

    // AI Studio's Gemma 4 emits a <thought>...</thought> reasoning block before
    // the JSON. It eats max_tokens and can contain stray braces that would
    // mislead extractJsonObject. Strip it before any further processing.
    const dethought = raw.replace(/<thought>[\s\S]*?<\/thought>\s*/g, "");

    // Cardinal-rule rewrite on the entire raw output BEFORE parsing — rewrites
    // both reasoning and patient_framing in one pass. Rewriting JSON values
    // is safe since the patterns target prose, not field names.
    const { text: cleanText, rewrites } = enforceCardinalRule(dethought);

    const parsed = extractJsonObject(cleanText) as Record<string, unknown> | null;

    if (!parsed) {
      return NextResponse.json(
        {
          error: "Model output did not contain a parsable JSON object.",
          raw_model_output: cleanText,
        },
        { status: 502 },
      );
    }

    const top3 = normalizeTop3(parsed.image_top3);
    const soap = normalizeSoap(parsed.soap);
    let urgency = normalizeUrgency(parsed.urgency);
    const reasoning = String(parsed.reasoning ?? "");
    const red_flags_noted = strList(parsed.red_flags_noted);
    const patient_framing = String(parsed.patient_framing ?? "");

    // Rule-based safety net: if model says green but cross-check finds 2+ red-flag
    // keywords in narrative, escalate to yellow. Cardinal rule: never under-triage.
    const cross = crossCheckRedFlags(narrative, imageDescription);
    let safety_escalation = false;
    if (urgency === "green" && cross.length >= 2) {
      urgency = "yellow";
      safety_escalation = true;
    }

    const result: TriageResult = {
      image_top3: top3,
      soap,
      urgency,
      reasoning,
      red_flags_noted,
      patient_framing,
      village: DEFAULT_VILLAGE,
      raw_model_output: cleanText,
      parse_ok: true,
      cardinal_rule_rewrites: rewrites,
      safety_escalation,
      cross_check_red_flags: cross,
      wall_seconds: (Date.now() - t0) / 1000,
      tool_invocations: toolInvocations,
    };

    return NextResponse.json(result, {
      headers: {
        "x-image-bytes": String(imageBytes),
      },
    });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return NextResponse.json({ error: `triage handler: ${msg}` }, { status: 500 });
  }
}
