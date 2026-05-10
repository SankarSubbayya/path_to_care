"""SOAP Extractor MCP: narrative → structured SOAP fields.

Calls Qwen-2.5-7B-Instruct with a structured prompt to produce a SOAP-style
record. DSPy `BootstrapFewShot` optimization is a stretch for the 24h build;
the baseline uses a hand-engineered prompt. Cardinal-rule rewriter applied.

Fields returned (per docs/ARCHITECTURE.md):
  chief_complaint, hpi, past_medical_history, medications, vitals, exam,
  red_flags, patient_concerns
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from core.cardinal_rule import enforce
from core.llm import qwen, chat_text


SOAP_PROMPT = """You are a triage scribe for a rural clinic. Extract SOAP-style structured fields from the patient narrative below. You DO NOT diagnose. Output ONLY a JSON object with these keys:

- chief_complaint (one short phrase)
- hpi (history of present illness; 1-3 sentences)
- duration (e.g., "2 days", "1 week", "unknown")
- associated_symptoms (list of strings; explicitly mark negations as "no <symptom>")
- past_medical_history (list of strings; "none stated" if absent)
- medications (list of strings; "none stated" if absent)
- vitals (object with keys among: temperature_c, heart_rate, breathing — only fields explicitly mentioned)
- exam_findings (list of objective findings the patient describes — e.g., "swelling", "redness extending up leg")
- red_flags (list of clinically concerning features mentioned in the narrative)
- patient_concerns (list of barriers/asks the patient mentions — e.g., "cannot afford to miss work")

Hard rules:
- Do NOT invent fields the patient did not mention. If absent, use "none stated" or [].
- Negations: if the patient says "no fever", record "no fever" — never just "fever".
- Do not write any diagnosis or condition name in any field.

Patient narrative:
{narrative}

JSON output (one object, no preamble):"""


@dataclass
class SoapResult:
    fields: dict
    raw: str
    parse_ok: bool


def extract(narrative: str, case_id: str | None = None) -> SoapResult:
    handle = qwen()
    raw = chat_text(handle, SOAP_PROMPT.format(narrative=narrative), max_new_tokens=512)
    raw = enforce(raw, case_id=case_id)
    fields, ok = _parse(raw)
    if not ok:
        fields = {
            "chief_complaint": "extraction_failed",
            "hpi": narrative[:200],
            "duration": "unknown",
            "associated_symptoms": [],
            "past_medical_history": ["none stated"],
            "medications": ["none stated"],
            "vitals": {},
            "exam_findings": [],
            "red_flags": [],
            "patient_concerns": [],
        }
    return SoapResult(fields=fields, raw=raw, parse_ok=ok)


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse(text: str) -> tuple[dict, bool]:
    m = _JSON_OBJ_RE.search(text)
    if not m:
        return {}, False
    try:
        data = json.loads(m.group(0))
        if isinstance(data, dict):
            return data, True
    except json.JSONDecodeError:
        pass
    return {}, False
