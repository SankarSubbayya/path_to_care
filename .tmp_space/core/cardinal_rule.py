"""Programmatic enforcement of the cardinal rule (docs/PROJECT.md):
'The system never produces diagnostic statements.'

Prompts can drift; this is the safety net. Every MCP that emits patient-facing
text passes through `enforce()` before returning.

Rewrites diagnostic phrases to non-diagnostic equivalents:
  'you have X'      -> 'signs are consistent with X'
  'this is X'       -> 'the appearance suggests X'
  'diagnosis is X'  -> 'a possibility is X'

Rewrites are logged to logs/cardinal_rule_rewrites.log so the audit trail
shows when the model drifted.
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = "logs/cardinal_rule_rewrites.log"

_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Order matters: more specific first.
    (re.compile(r"\bthe\s+diagnosis\s+is\b", re.IGNORECASE), "a possibility is"),
    (re.compile(r"\bdiagnosis\s+is\b", re.IGNORECASE), "a possibility is"),
    (re.compile(r"\bdiagnosed\s+with\b", re.IGNORECASE), "showing signs of"),
    (re.compile(r"\byou\s+have\b", re.IGNORECASE), "signs suggest"),
    (re.compile(r"\bI\s+confirm\b", re.IGNORECASE), "appearances suggest"),
    (re.compile(r"\bthis\s+is\s+(definitely|clearly)\b", re.IGNORECASE), r"signs are consistent with \1"),
    # Soft cleanups
    (re.compile(r"\bdefinitely\b", re.IGNORECASE), "likely"),
    (re.compile(r"\bcertainly\b", re.IGNORECASE), "likely"),
]


def enforce(text: str, *, case_id: str | None = None) -> str:
    """Rewrite diagnostic phrasing in `text`. Logs rewrites if any apply."""
    rewritten = text
    rewrites = []
    for pat, repl in _PATTERNS:
        new = pat.sub(repl, rewritten)
        if new != rewritten:
            rewrites.append(pat.pattern)
            rewritten = new
    if rewrites:
        _log(case_id, text, rewritten, rewrites)
    return rewritten


def _log(case_id: str | None, original: str, rewritten: str, patterns: list[str]) -> None:
    Path(os.path.dirname(LOG_PATH)).mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(LOG_PATH, "a") as f:
        f.write(
            f"[{ts}] case_id={case_id} patterns={patterns}\n"
            f"  before: {original!r}\n"
            f"  after:  {rewritten!r}\n"
        )


def violations(text: str) -> list[str]:
    """Return list of diagnostic-phrase patterns matched. Used by tests."""
    return [pat.pattern for pat, _ in _PATTERNS if pat.search(text)]
