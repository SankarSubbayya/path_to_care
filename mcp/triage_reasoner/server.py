"""Triage Reasoner MCP: image_top3 + soap + village context → urgency.

Calls Gemma 4 (same loaded model as the image classifier MCP) with a prompt
that fuses (a) top-3 image conditions, (b) SOAP-extracted fields, (c) village
practical-urgency framing. Returns one of {red, yellow, green} plus a brief
reasoning trace. Cardinal-rule rewriter applied to the reasoning output.

Loads a LoRA adapter on top of the base Gemma 4 if `adapter_path` is provided.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from core.cardinal_rule import enforce
from core.llm import gemma4, gemma4_attach_adapter, chat_multimodal


_VALID = {"red", "yellow", "green"}


TRIAGE_PROMPT = """You are a triage decision-support assistant for a rural community health worker. You DO NOT diagnose. Your job is to recommend an urgency level.

Possible levels:
- RED:    needs immediate care (today, not tomorrow). Examples: spreading infection with fever, snakebite, anaphylaxis, deep burn, suspected sepsis.
- YELLOW: needs clinical evaluation in 1-2 days. Examples: localized infections, contagious skin conditions, herpes zoster.
- GREEN:  monitor at home, return if it worsens. Examples: heat rash, mosquito bites, mild sunburn, healing scratches.

Inputs:
1. Top-3 candidate conditions from the image classifier (with confidences).
2. Structured SOAP fields from the patient's narrative.
3. Village practical-urgency context (distance, cost, harvest pressure).

Your output, in this exact format:
URGENCY: <red|yellow|green>
REASONING: <2-4 sentences explaining the level. Use "signs suggest" not "you have". Reference specific red flags from the SOAP fields if present.>
RED_FLAGS_NOTED: <comma-separated list of red flags you considered, or "none">
PATIENT_FRAMING: <1-2 sentences the community health worker can read aloud, in plain language, framed as cost-benefit.>

Inputs follow.

IMAGE_TOP3:
{image_top3}

SOAP_FIELDS:
{soap_fields}

VILLAGE_CONTEXT:
{village_context}

Respond now."""


@dataclass
class TriageResult:
    urgency: str       # 'red' | 'yellow' | 'green'
    reasoning: str
    red_flags_noted: list[str]
    patient_framing: str
    raw: str
    parse_ok: bool


def triage(*, image_top3: list[dict], soap_fields: dict, village_context: str,
           adapter_path: str | None = None, case_id: str | None = None) -> TriageResult:
    handle = gemma4_attach_adapter(adapter_path) if adapter_path else gemma4()
    prompt = TRIAGE_PROMPT.format(
        image_top3=_format_top3(image_top3),
        soap_fields=_format_soap(soap_fields),
        village_context=village_context,
    )
    raw = chat_multimodal(handle, prompt, image=None, max_new_tokens=384)
    raw = enforce(raw, case_id=case_id)
    return _parse(raw)


def _format_top3(top3: list[dict]) -> str:
    if not top3:
        return "(no candidates produced)"
    return "\n".join(
        f"- {c.get('condition', 'unknown')} (confidence {float(c.get('confidence', 0.0)):.2f})"
        for c in top3
    )


def _format_soap(soap: dict) -> str:
    if not soap:
        return "(soap extraction failed)"
    fields = []
    for k in ("chief_complaint", "hpi", "duration", "associated_symptoms",
              "past_medical_history", "medications", "vitals", "exam_findings",
              "red_flags", "patient_concerns"):
        if k in soap:
            fields.append(f"{k}: {soap[k]}")
    return "\n".join(fields)


_URGENCY_RE = re.compile(r"URGENCY\s*:\s*(red|yellow|green)\b", re.IGNORECASE)
_REASONING_RE = re.compile(r"REASONING\s*:\s*(.+?)(?:\nRED_FLAGS_NOTED|\nPATIENT_FRAMING|\Z)", re.IGNORECASE | re.DOTALL)
_RED_FLAGS_RE = re.compile(r"RED_FLAGS_NOTED\s*:\s*(.+?)(?:\nPATIENT_FRAMING|\Z)", re.IGNORECASE | re.DOTALL)
_FRAMING_RE = re.compile(r"PATIENT_FRAMING\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


def _parse(raw: str) -> TriageResult:
    urgency = "green"  # safest "I am not sure" default for the parse-failure case
    reasoning = ""
    red_flags: list[str] = []
    framing = ""
    parse_ok = False

    m = _URGENCY_RE.search(raw)
    if m:
        urgency = m.group(1).lower()
        parse_ok = True
    if (m := _REASONING_RE.search(raw)):
        reasoning = m.group(1).strip()
    if (m := _RED_FLAGS_RE.search(raw)):
        rf = m.group(1).strip()
        if rf and rf.lower() != "none":
            red_flags = [s.strip() for s in rf.split(",") if s.strip()]
    if (m := _FRAMING_RE.search(raw)):
        framing = m.group(1).strip()

    if urgency not in _VALID:
        urgency = "green"
        parse_ok = False
    return TriageResult(
        urgency=urgency,
        reasoning=reasoning,
        red_flags_noted=red_flags,
        patient_framing=framing,
        raw=raw,
        parse_ok=parse_ok,
    )
