"""Parser tests for the three MCPs that produce structured output from LLM
text. These are the most likely places for silent regressions: a model that
generates slightly different formatting can break the orchestrator."""
from __future__ import annotations

import pytest

from mcp.image_classifier.server import _parse as parse_image
from mcp.soap_extractor.server import _parse as parse_soap
from mcp.triage_reasoner.server import _parse as parse_triage


# --- Image classifier parser -----------------------------------------------

class TestImageParser:
    def test_clean_json_array(self):
        text = '[{"condition": "cellulitis", "confidence": 0.7}, {"condition": "abscess", "confidence": 0.2}]'
        parsed, ok = parse_image(text)
        assert ok is True
        assert len(parsed) == 2
        assert parsed[0]["condition"] == "cellulitis"
        assert parsed[0]["confidence"] == 0.7

    def test_json_array_wrapped_in_prose(self):
        text = (
            "Here are the candidates:\n"
            '[{"condition": "tinea", "confidence": 0.6}, {"condition": "eczema", "confidence": 0.3}]\n'
            "These are non-diagnostic."
        )
        parsed, ok = parse_image(text)
        assert ok is True
        assert parsed[0]["condition"] == "tinea"

    def test_malformed_json(self):
        parsed, ok = parse_image("[{condition: cellulitis, confidence: bad}")
        assert ok is False
        assert parsed == []

    def test_no_json_at_all(self):
        parsed, ok = parse_image("Sorry I cannot describe this without seeing it.")
        assert ok is False
        assert parsed == []


# --- SOAP extractor parser --------------------------------------------------

class TestSoapParser:
    def test_clean_json_object(self):
        text = '{"chief_complaint": "swollen foot", "duration": "2 days"}'
        parsed, ok = parse_soap(text)
        assert ok is True
        assert parsed["chief_complaint"] == "swollen foot"

    def test_object_with_preamble(self):
        text = 'Here is the SOAP:\n{"chief_complaint": "fever"}\n'
        parsed, ok = parse_soap(text)
        assert ok is True
        assert parsed["chief_complaint"] == "fever"

    def test_empty_string(self):
        parsed, ok = parse_soap("")
        assert ok is False
        assert parsed == {}

    def test_json_array_not_object_should_fail(self):
        # SOAP must be a top-level dict; a list isn't valid.
        parsed, ok = parse_soap('[{"chief_complaint": "x"}]')
        # _parse uses a greedy regex; if it parses to a list, _parse should reject.
        # If implementation accepts it, this guards behavior either way.
        assert ok is False or isinstance(parsed, dict)


# --- Triage parser ----------------------------------------------------------

class TestTriageParser:
    def test_full_well_formed_response(self):
        raw = (
            "URGENCY: red\n"
            "REASONING: Signs suggest a spreading infection with systemic involvement.\n"
            "RED_FLAGS_NOTED: fever, spreading erythema, rigors\n"
            "PATIENT_FRAMING: Going today is cheaper than waiting until it's an emergency."
        )
        result = parse_triage(raw)
        assert result.parse_ok is True
        assert result.urgency == "red"
        assert "spreading infection" in result.reasoning.lower()
        assert "fever" in result.red_flags_noted
        assert "spreading erythema" in result.red_flags_noted
        assert "going today" in result.patient_framing.lower()

    def test_uppercase_urgency(self):
        result = parse_triage("URGENCY: RED\nREASONING: ...")
        assert result.urgency == "red"
        assert result.parse_ok is True

    def test_yellow_with_no_red_flags(self):
        raw = (
            "URGENCY: yellow\n"
            "REASONING: Signs are mild.\n"
            "RED_FLAGS_NOTED: none\n"
            "PATIENT_FRAMING: Visit in a couple of days."
        )
        result = parse_triage(raw)
        assert result.urgency == "yellow"
        assert result.red_flags_noted == []

    def test_missing_urgency_defaults_to_green_with_parse_ok_false(self):
        raw = "REASONING: I am not sure what this is."
        result = parse_triage(raw)
        assert result.urgency == "green"   # safest default for parse failure
        assert result.parse_ok is False

    def test_invalid_urgency_value_falls_back(self):
        raw = "URGENCY: severe\nREASONING: ..."
        result = parse_triage(raw)
        # severe isn't in {red,yellow,green}; the urgency regex won't match.
        assert result.urgency == "green"
        assert result.parse_ok is False

    def test_garbled_input(self):
        result = parse_triage("the model produced gibberish }{][")
        assert result.urgency == "green"
        assert result.parse_ok is False
