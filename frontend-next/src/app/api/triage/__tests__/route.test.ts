// Integration tests for the /api/triage Next.js route handler. The
// upstream vLLM call is mocked via vi.stubGlobal('fetch', ...) so these
// tests are fast (no network) and deterministic. They exercise:
//
//   - JSON extraction from the model's prose
//   - urgency normalization (case, invalid value → green)
//   - top-3 capping + confidence clipping
//   - cardinal-rule rewrites on the model's prose
//   - safety net: green + ≥2 red-flag keywords → escalate to yellow
//   - vLLM upstream error → 502 with helpful message
//   - missing input → 400

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Helper: build a multipart FormData body the way the real client does.
function makeForm(narrative: string, imageDescription = ""): FormData {
  const fd = new FormData();
  fd.append("narrative", narrative);
  if (imageDescription) fd.append("image_description", imageDescription);
  return fd;
}

function makeReq(body: FormData): Request {
  return new Request("http://localhost/api/triage", { method: "POST", body });
}

function vllmReply(content: string, status = 200): Response {
  return new Response(
    JSON.stringify({ choices: [{ message: { content } }] }),
    { status, headers: { "Content-Type": "application/json" } }
  );
}

// Fully-formed structured response from the model.
const MODEL_RED_RESPONSE = JSON.stringify({
  image_top3: [
    { condition: "cellulitis", confidence: 0.85 },
    { condition: "tetanus", confidence: 0.10 },
    { condition: "necrotizing fasciitis", confidence: 0.05 },
  ],
  soap: {
    chief_complaint: "swollen foot from rusty nail",
    hpi: "Two days post-injury, fever, shivering, redness ascending leg.",
    duration: "2 days",
    associated_symptoms: ["fever", "shivering"],
    past_medical_history: ["none stated"],
    medications: ["none stated"],
    vitals: {},
    exam_findings: ["spreading erythema", "edema"],
    red_flags: ["spreading infection", "fever", "rigors"],
    patient_concerns: ["cannot afford to miss harvest"],
  },
  urgency: "red",
  reasoning:
    "Signs suggest a rapidly spreading infection with systemic involvement. " +
    "Fever and shivering point to early sepsis. Same-day care indicated.",
  red_flags_noted: ["spreading erythema", "fever", "rigors"],
  patient_framing:
    "Going to the PHC today is cheaper than waiting until you cannot work at all.",
});

let routeMod: typeof import("@/app/api/triage/route");

beforeEach(async () => {
  // Reload the module so each test gets a fresh fetch mock.
  vi.resetModules();
  routeMod = await import("@/app/api/triage/route");
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("POST /api/triage — happy path", () => {
  it("returns urgency=red with a parsed model response", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => vllmReply(MODEL_RED_RESPONSE)));
    const fd = makeForm("I cut my foot on a rusty nail. Fever, shivering. Redness going up leg.",
                       "Lower leg with redness extending proximally.");
    const resp = await routeMod.POST(makeReq(fd));
    expect(resp.status).toBe(200);
    const body = await resp.json();
    expect(body.urgency).toBe("red");
    expect(body.parse_ok).toBe(true);
    expect(body.image_top3).toHaveLength(3);
    expect(body.image_top3[0].condition).toBe("cellulitis");
    expect(body.image_top3[0].confidence).toBeCloseTo(0.85, 2);
    expect(body.soap.chief_complaint).toMatch(/swollen|nail|foot/);
  });

  it("clips confidence to [0,1] and caps top3 at length 3", async () => {
    const overReply = JSON.stringify({
      image_top3: [
        { condition: "a", confidence: 1.5 },
        { condition: "b", confidence: -0.2 },
        { condition: "c", confidence: 0.4 },
        { condition: "extra", confidence: 0.1 },
      ],
      soap: {},
      urgency: "yellow",
      reasoning: "Signs suggest mild irritation.",
      red_flags_noted: [],
      patient_framing: "Watch and wait.",
    });
    vi.stubGlobal("fetch", vi.fn(async () => vllmReply(overReply)));
    const resp = await routeMod.POST(makeReq(makeForm("itchy spot")));
    const body = await resp.json();
    expect(body.image_top3).toHaveLength(3);
    expect(body.image_top3[0].confidence).toBe(1.0);
    expect(body.image_top3[1].confidence).toBe(0.0);
  });

  it("rewrites diagnostic phrasing in reasoning before responding", async () => {
    const dirty = JSON.stringify({
      image_top3: [
        { condition: "x", confidence: 0.5 },
        { condition: "y", confidence: 0.3 },
        { condition: "z", confidence: 0.2 },
      ],
      soap: {},
      urgency: "yellow",
      reasoning: "You have a fever and the diagnosis is impetigo.",
      red_flags_noted: [],
      patient_framing: "Go to clinic tomorrow.",
    });
    vi.stubGlobal("fetch", vi.fn(async () => vllmReply(dirty)));
    const resp = await routeMod.POST(makeReq(makeForm("rash 2 days")));
    const body = await resp.json();
    expect(body.cardinal_rule_rewrites).toBeGreaterThan(0);
    expect(body.reasoning).not.toContain("you have");
    expect(body.reasoning).not.toContain("diagnosis is");
    expect(body.reasoning.toLowerCase()).toContain("signs suggest");
  });
});

describe("POST /api/triage — safety net", () => {
  it("escalates green→yellow when ≥2 red-flag keywords appear in narrative", async () => {
    const greenReply = JSON.stringify({
      image_top3: [
        { condition: "x", confidence: 0.5 },
        { condition: "y", confidence: 0.3 },
        { condition: "z", confidence: 0.2 },
      ],
      soap: {},
      urgency: "green",
      reasoning: "Signs suggest a mild self-limited reaction.",
      red_flags_noted: [],
      patient_framing: "Monitor at home.",
    });
    vi.stubGlobal("fetch", vi.fn(async () => vllmReply(greenReply)));
    // Narrative contains keywords: 'shivering' + 'rigors' + 'gangrene' → 3 ≥ 2
    const fd = makeForm(
      "Foot is dark and turning to gangrene. I have rigors and shivering.",
      "Dusky discoloration of the toe."
    );
    const resp = await routeMod.POST(makeReq(fd));
    const body = await resp.json();
    expect(body.urgency).toBe("yellow");
    expect(body.safety_escalation).toBe(true);
    expect(body.cross_check_red_flags.length).toBeGreaterThanOrEqual(2);
  });

  it("does NOT escalate when red flags are absent", async () => {
    const greenReply = JSON.stringify({
      image_top3: [
        { condition: "x", confidence: 0.5 },
        { condition: "y", confidence: 0.3 },
        { condition: "z", confidence: 0.2 },
      ],
      soap: {},
      urgency: "green",
      reasoning: "Signs suggest a mild rash.",
      red_flags_noted: [],
      patient_framing: "Monitor at home.",
    });
    vi.stubGlobal("fetch", vi.fn(async () => vllmReply(greenReply)));
    const resp = await routeMod.POST(makeReq(makeForm("Mild dry patch on elbow for many years.")));
    const body = await resp.json();
    expect(body.urgency).toBe("green");
    expect(body.safety_escalation).toBe(false);
  });
});

describe("POST /api/triage — error paths", () => {
  it("returns 400 when neither narrative nor image_description provided", async () => {
    vi.stubGlobal("fetch", vi.fn());
    const resp = await routeMod.POST(makeReq(makeForm("")));
    expect(resp.status).toBe(400);
  });

  it("returns 502 when vLLM upstream errors", async () => {
    vi.stubGlobal("fetch", vi.fn(async () =>
      new Response("internal model failure", { status: 500 })
    ));
    const resp = await routeMod.POST(makeReq(makeForm("rash 2 days")));
    expect(resp.status).toBe(502);
    const body = await resp.json();
    expect(body.error).toContain("vLLM 500");
  });

  it("returns 502 when the model output has no JSON", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => vllmReply("Sorry I cannot help.")));
    const resp = await routeMod.POST(makeReq(makeForm("rash 2 days")));
    expect(resp.status).toBe(502);
    const body = await resp.json();
    expect(body.error).toContain("did not contain a parsable JSON");
  });
});

describe("POST /api/triage — urgency normalization", () => {
  it("accepts uppercase urgency", async () => {
    const reply = JSON.stringify({
      image_top3: [
        { condition: "x", confidence: 0.5 },
        { condition: "y", confidence: 0.3 },
        { condition: "z", confidence: 0.2 },
      ],
      soap: {},
      urgency: "RED",
      reasoning: "Signs suggest danger.",
      red_flags_noted: [],
      patient_framing: "Go now.",
    });
    vi.stubGlobal("fetch", vi.fn(async () => vllmReply(reply)));
    const resp = await routeMod.POST(makeReq(makeForm("test")));
    const body = await resp.json();
    expect(body.urgency).toBe("red");
  });

  it("falls back to green for invalid urgency value", async () => {
    const reply = JSON.stringify({
      image_top3: [
        { condition: "x", confidence: 0.5 },
        { condition: "y", confidence: 0.3 },
        { condition: "z", confidence: 0.2 },
      ],
      soap: {},
      urgency: "extreme",
      reasoning: "Signs suggest something.",
      red_flags_noted: [],
      patient_framing: "Whatever.",
    });
    vi.stubGlobal("fetch", vi.fn(async () => vllmReply(reply)));
    const resp = await routeMod.POST(makeReq(makeForm("test")));
    const body = await resp.json();
    expect(body.urgency).toBe("green");
  });
});
