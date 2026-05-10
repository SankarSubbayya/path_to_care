"""Cardinal-rule rewriter tests — the safety net documented in
core/cardinal_rule.py and verified live against case Y09 during eval."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from core.cardinal_rule import enforce, violations


# --- Pattern coverage ------------------------------------------------------

@pytest.mark.parametrize(
    "diagnostic_text,expected_substring",
    [
        ("you have cellulitis",          "signs suggest"),
        ("You have a fever",              "signs suggest"),
        ("YOU   HAVE  high fever",        "signs suggest"),     # whitespace + case
        ("the diagnosis is sepsis",       "a possibility is"),
        ("Diagnosis is contact dermatitis", "a possibility is"),
        ("diagnosed with impetigo",       "showing signs of"),
        ("I confirm the rash",            "appearances suggest"),
        ("This is definitely an infection", "consistent with"),
        ("This is clearly cellulitis",    "consistent with"),
    ],
)
def test_enforce_rewrites_diagnostic_phrases(diagnostic_text, expected_substring):
    out = enforce(diagnostic_text)
    assert expected_substring in out.lower()


@pytest.mark.parametrize(
    "soft_text,expected",
    [
        ("This is definitely cellulitis", "likely"),    # 'definitely' soft-cleaned
        ("certainly an infection",         "likely"),   # 'certainly' soft-cleaned
    ],
)
def test_enforce_soft_cleanups(soft_text, expected):
    assert expected in enforce(soft_text).lower()


def test_enforce_leaves_clean_text_unchanged():
    text = "Signs suggest cellulitis. The patient should be evaluated within 24 hours."
    assert enforce(text) == text


def test_enforce_idempotent_on_clean_output():
    once = enforce("you have cellulitis")
    twice = enforce(once)
    # Second pass shouldn't introduce further changes — once rewritten, the
    # output is already cardinal-rule clean.
    assert twice == once


# --- Violation detection ---------------------------------------------------

def test_violations_finds_all_diagnostic_phrases():
    text = "You have a fever and the diagnosis is cellulitis."
    matched = violations(text)
    assert any("you" in p.lower() and "have" in p for p in matched)
    assert any("diagnosis" in p.lower() for p in matched)


def test_violations_empty_for_clean_text():
    assert violations("Signs suggest infection. Recommend clinical evaluation.") == []


# --- Logging -----------------------------------------------------------------

def test_enforce_logs_rewrites_to_logfile(tmp_path, monkeypatch):
    import core.cardinal_rule as cr
    log_path = tmp_path / "rewrites.log"
    monkeypatch.setattr(cr, "LOG_PATH", str(log_path))
    enforce("you have cellulitis", case_id="TEST-CASE-01")
    content = log_path.read_text()
    assert "TEST-CASE-01" in content
    assert "you have cellulitis" in content   # 'before' line
    assert "signs suggest" in content         # 'after' line


def test_enforce_does_not_log_when_no_rewrite_happens(tmp_path, monkeypatch):
    import core.cardinal_rule as cr
    log_path = tmp_path / "rewrites.log"
    monkeypatch.setattr(cr, "LOG_PATH", str(log_path))
    enforce("Signs suggest infection.", case_id="TEST-CASE-02")
    assert not log_path.exists()


# --- Real case captured live during eval -----------------------------------

def test_y09_case_rewrite_matches_what_happened_live():
    """Reproduces the actual rewrite that fired during eval on case P2C-Y09:
    `... and you have a fever, ...` → `... and signs suggest a fever, ...`.
    This test pins behavior so refactoring core/cardinal_rule.py can't
    silently regress the safety net."""
    before = (
        "These spots are spreading and you have a fever, so it is important "
        "to see a nurse in the next day or two."
    )
    after = enforce(before, case_id="P2C-Y09")
    assert "you have a fever" not in after
    assert "signs suggest a fever" in after
