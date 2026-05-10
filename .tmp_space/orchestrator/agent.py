"""DSPy-style orchestrator that wires the 4 MCPs into the 7-stage Rajan flow.

For the 24-hour build the orchestrator is a plain Python coordinator with the
same shape DSPy would impose — clean module boundaries, structured I/O.
DSPy's ReAct + signatures are a stretch goal layered on top once the baseline
runs. The function `run_case` is the single entry point used by both the
eval harness (`harness/run.py`) and the Gradio Space (`frontend/app.py`).

Flow (matches docs/ARCHITECTURE.md "7 stages, the Rajan dialogue"):
  1. Image intake          -> image_classifier MCP (top-3 + confidence)
  2. Subjective intake     -> soap_extractor MCP (narrative -> structured fields)
  3. Red-flag detection    -> rule check on SOAP fields + image_top3
  4. Triage + context      -> village_context MCP -> triage_reasoner MCP
  5. Persuasion framing    -> already produced by triage_reasoner (PATIENT_FRAMING)
  6. Provider summary      -> assembled SOAP + triage as the pre-visit note
  7. Engineering reflection -> structured trace returned with the result

Optional `adapter_path` argument lets the runner swap in a LoRA-tuned triage
reasoner without changing call sites.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

from mcp.image_classifier.server import classify, ClassificationResult
from mcp.soap_extractor.server import extract, SoapResult
from mcp.triage_reasoner.server import triage, TriageResult
from mcp.village_context.server import get_context, practical_urgency_blurb


# Hard-coded red-flag keywords used to cross-check the model's output. If a
# narrative or image_description contains any of these terms but the triage
# reasoner returned green, the orchestrator escalates to yellow and notes the
# disagreement (a simple safety net; the LoRA-tuned model should learn this).
_RED_FLAG_KEYWORDS = {
    "tetanus", "trismus", "snake", "snakebite", "fang",
    "anaphylax", "throat tight", "lips swelling",
    "necrotizing", "crepitus", "pain disproportionate", "out of proportion",
    "gangrene", "black", "altered mental",
    "spreading erythema", "lymphangitic", "red streaks",
    "rigors", "shivering", "fever 39", "fever 40",
}


@dataclass
class CaseTrace:
    case_id: str
    image_top3: list[dict]
    image_parse_ok: bool
    soap_fields: dict
    soap_parse_ok: bool
    cross_check_red_flags: list[str]
    village_blurb: str
    urgency: str
    reasoning: str
    red_flags_noted: list[str]
    patient_framing: str
    triage_parse_ok: bool
    safety_escalation: bool   # True if our cross-check forced an upgrade

    def to_dict(self) -> dict:
        return asdict(self)


def cross_check(narrative: str, image_description: str) -> list[str]:
    """Return any red-flag keywords found in the inputs."""
    blob = f"{narrative}\n{image_description}".lower()
    return [kw for kw in _RED_FLAG_KEYWORDS if kw in blob]


def run_case(case: dict, *, adapter_path: Optional[str] = None) -> CaseTrace:
    """Run one case through the 7-stage flow. Pure-Python; uses in-process MCPs."""
    case_id = case["case_id"]
    narrative = case["narrative"]
    image_description = case.get("image_description", "")

    # Stage 1: image intake -> top-3
    img: ClassificationResult = classify(image_description, image=None, case_id=case_id)

    # Stage 2: subjective intake -> SOAP
    soap: SoapResult = extract(narrative, case_id=case_id)

    # Stage 3: red-flag rule check
    flagged = cross_check(narrative, image_description)

    # Stage 4: village context + triage
    blurb = practical_urgency_blurb(case)
    tr: TriageResult = triage(
        image_top3=img.top3,
        soap_fields=soap.fields,
        village_context=blurb,
        adapter_path=adapter_path,
        case_id=case_id,
    )

    # Safety net: if the model said green but rule-based check found ≥2 red
    # flags, escalate to yellow. The cardinal rule is "do not under-triage."
    safety_escalation = False
    final_urgency = tr.urgency
    if tr.urgency == "green" and len(flagged) >= 2:
        final_urgency = "yellow"
        safety_escalation = True

    return CaseTrace(
        case_id=case_id,
        image_top3=img.top3,
        image_parse_ok=img.parse_ok,
        soap_fields=soap.fields,
        soap_parse_ok=soap.parse_ok,
        cross_check_red_flags=flagged,
        village_blurb=blurb,
        urgency=final_urgency,
        reasoning=tr.reasoning,
        red_flags_noted=tr.red_flags_noted,
        patient_framing=tr.patient_framing,
        triage_parse_ok=tr.parse_ok,
        safety_escalation=safety_escalation,
    )
