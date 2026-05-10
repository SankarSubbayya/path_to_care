"""Orchestrator unit tests — exercise cross_check() and run_case() with the
four MCP calls monkey-patched. This isolates the orchestrator's logic from
the LLMs so we can pin the safety-net escalation behavior in fast tests."""
from __future__ import annotations

import pytest

import orchestrator.agent as agent_mod
from orchestrator.agent import cross_check, run_case
from mcp.image_classifier.server import ClassificationResult
from mcp.soap_extractor.server import SoapResult
from mcp.triage_reasoner.server import TriageResult


# --- cross_check ----------------------------------------------------------

@pytest.mark.parametrize(
    "narrative,image_desc,expect_at_least_n",
    [
        ("I have fever 39 and shivering", "redness extending up leg", 2),
        ("snake bit me, fang marks visible", "swelling rapid", 1),
        ("just a small scratch healing well", "minor abrasion no signs", 0),
        ("trismus and rigors", "wound discoloration", 2),
    ],
)
def test_cross_check_finds_red_flags(narrative, image_desc, expect_at_least_n):
    flags = cross_check(narrative, image_desc)
    assert len(flags) >= expect_at_least_n


def test_cross_check_returns_empty_for_clean_text():
    assert cross_check(
        "Mild dry skin patches that have been there for years.",
        "Localized scaling, no inflammation.",
    ) == []


# --- run_case with monkey-patched MCPs ------------------------------------

def _stub_classify(image_top3):
    def _f(image_description, image=None, case_id=None):
        return ClassificationResult(top3=image_top3, raw="(stub)", parse_ok=True)
    return _f


def _stub_extract(soap_fields):
    def _f(narrative, case_id=None):
        return SoapResult(fields=soap_fields, raw="(stub)", parse_ok=True)
    return _f


def _stub_triage(urgency, reasoning="stub reasoning", red_flags=None, framing="stub framing"):
    def _f(*, image_top3, soap_fields, village_context, adapter_path=None, case_id=None):
        return TriageResult(
            urgency=urgency,
            reasoning=reasoning,
            red_flags_noted=red_flags or [],
            patient_framing=framing,
            raw="(stub)",
            parse_ok=True,
        )
    return _f


def _patch_all(monkeypatch, *, urgency, image_top3=None, soap_fields=None, red_flags=None):
    if image_top3 is None:
        image_top3 = [{"condition": "cellulitis", "confidence": 0.7}]
    if soap_fields is None:
        soap_fields = {"chief_complaint": "stub", "associated_symptoms": []}
    monkeypatch.setattr(agent_mod, "classify", _stub_classify(image_top3))
    monkeypatch.setattr(agent_mod, "extract", _stub_extract(soap_fields))
    monkeypatch.setattr(agent_mod, "triage", _stub_triage(urgency, red_flags=red_flags))


CLEAN_CASE = {
    "case_id": "TEST-CLEAN-01",
    "narrative": "Mild dry skin patches for years.",
    "image_description": "Localized scaling, no inflammation.",
    "ground_truth_urgency": "green",
    "village_context": {
        "distance_to_clinic_km": 18,
        "patient_daily_wage_inr": 350,
        "transport_cost_round_trip_inr": 180,
        "harvest_active": False,
    },
}

DANGEROUS_CASE = {
    "case_id": "TEST-DANGER-01",
    # Multiple red-flag keywords on purpose
    "narrative": "Snake bit me, fang punctures, leg swelling fast, trismus and rigors with fever 39.",
    "image_description": "Spreading erythema, dusky discoloration, crepitus.",
    "ground_truth_urgency": "red",
    "village_context": {
        "distance_to_clinic_km": 18,
        "patient_daily_wage_inr": 350,
        "transport_cost_round_trip_inr": 180,
        "harvest_active": True,
    },
}


def test_run_case_returns_model_urgency_when_consistent(monkeypatch):
    _patch_all(monkeypatch, urgency="green")
    trace = run_case(CLEAN_CASE)
    assert trace.urgency == "green"
    assert trace.safety_escalation is False
    assert trace.case_id == "TEST-CLEAN-01"


def test_safety_net_escalates_green_to_yellow_when_red_flags_present(monkeypatch):
    """If the model says 'green' but the rule-based cross-check finds >=2
    red-flag keywords in the narrative, the orchestrator MUST escalate to
    yellow. Cardinal rule: never under-triage."""
    _patch_all(monkeypatch, urgency="green")
    trace = run_case(DANGEROUS_CASE)
    assert trace.urgency == "yellow"          # escalated
    assert trace.safety_escalation is True
    assert len(trace.cross_check_red_flags) >= 2


def test_safety_net_does_not_escalate_red(monkeypatch):
    """If the model already says red, the safety net should leave it alone."""
    _patch_all(monkeypatch, urgency="red")
    trace = run_case(DANGEROUS_CASE)
    assert trace.urgency == "red"
    assert trace.safety_escalation is False


def test_safety_net_does_not_escalate_yellow(monkeypatch):
    """The safety net only escalates from green; yellow stays put even if
    red flags exist (the model already chose a non-home-care level)."""
    _patch_all(monkeypatch, urgency="yellow")
    trace = run_case(DANGEROUS_CASE)
    assert trace.urgency == "yellow"
    assert trace.safety_escalation is False


def test_run_case_passes_case_id_through(monkeypatch):
    _patch_all(monkeypatch, urgency="green")
    trace = run_case({**CLEAN_CASE, "case_id": "P2C-CUSTOM-99"})
    assert trace.case_id == "P2C-CUSTOM-99"


def test_run_case_with_clean_inputs_no_false_escalation(monkeypatch):
    """Clean narrative + image: cross_check finds no red flags, no escalation."""
    _patch_all(monkeypatch, urgency="green")
    trace = run_case(CLEAN_CASE)
    assert trace.cross_check_red_flags == []
    assert trace.safety_escalation is False
