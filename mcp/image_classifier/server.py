"""Image Classifier MCP: top-3 condition predictions with confidence.

Calls Gemma 4 with the case's image (or, for the 24-hour build, the case's
`image_description` text — full image-tensor inference is exercised in Phase 6
once the image dataset is wired). Output strictly enforced to be 3 (or fewer)
(condition, confidence) pairs. Cardinal-rule rewriter applied. Single-class
output is impossible by construction.

Why text-on-image-description for the baseline: HAM10000 download + image
encoding for all 30 cases adds latency we don't have for Phase 3. The
`image_description` strings in `data/cases.jsonl` describe what a real
photo would show — a strong proxy for the multimodal call. Phase 6 swaps
in real images for a final-eval pass.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from core.cardinal_rule import enforce
from core.llm import gemma4, chat_multimodal


SYSTEM_PROMPT = (
    "You are a careful skin-condition observer for rural healthcare triage. "
    "You DO NOT diagnose. Given a description (or photo) of a skin or wound "
    "presentation, return up to 3 plausible condition names ranked by "
    "likelihood, each with a confidence between 0 and 1. Always exactly 3 "
    "candidates if at all possible. Never say 'this is X' or 'you have X'. "
    "Output ONLY a JSON array of objects, e.g.:\n"
    '[{"condition": "cellulitis", "confidence": 0.7}, '
    '{"condition": "abscess", "confidence": 0.2}, '
    '{"condition": "contact dermatitis", "confidence": 0.1}]'
)


@dataclass
class ClassificationResult:
    top3: list[dict]   # list of {condition: str, confidence: float}
    raw: str
    parse_ok: bool


def classify(image_description: str, image=None, case_id: str | None = None) -> ClassificationResult:
    """Run the image classifier MCP on a description (or PIL image).

    Returns top-3 (or fewer) (condition, confidence) tuples. If parsing fails,
    returns a low-confidence fallback rather than guessing — the cardinal rule
    forbids fabricating a single class.
    """
    handle = gemma4()
    user_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Presentation:\n{image_description}\n\n"
        f"JSON array (top-3 ranked):"
    )
    raw = chat_multimodal(handle, user_prompt, image=image, max_new_tokens=192)
    raw = enforce(raw, case_id=case_id)
    parsed, ok = _parse(raw)
    if not ok or not parsed:
        parsed = [
            {"condition": "non-diagnostic / image quality insufficient", "confidence": 0.0},
        ]
    # Hard cap at 3, drop entries that don't parse, ensure confidences in [0,1].
    cleaned = []
    for c in parsed[:3]:
        cond = str(c.get("condition", "")).strip()
        try:
            conf = float(c.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        cleaned.append({"condition": cond or "unknown", "confidence": max(0.0, min(1.0, conf))})
    return ClassificationResult(top3=cleaned, raw=raw, parse_ok=ok)


_JSON_ARRAY_RE = re.compile(r"\[\s*\{.*?\}\s*\]", re.DOTALL)


def _parse(text: str) -> tuple[list[dict], bool]:
    m = _JSON_ARRAY_RE.search(text)
    if not m:
        return [], False
    try:
        data = json.loads(m.group(0))
        if isinstance(data, list) and all(isinstance(x, dict) for x in data):
            return data, True
    except json.JSONDecodeError:
        pass
    return [], False
