"""Triage reward function from docs/EVALUATION.md.

R(predicted, ground_truth) =
    1.0 if exact match
    0.5 if adjacent level (Yellow when truth is Red, etc.)
    0.0 if off by 2+

Levels are ordered Green < Yellow < Red so adjacency is a 1-step distance.
"""
from __future__ import annotations

URGENCY_ORDER = {"green": 0, "yellow": 1, "red": 2}


def normalize(level: str) -> str:
    s = level.strip().lower()
    if s not in URGENCY_ORDER:
        raise ValueError(f"unknown urgency level: {level!r}; expected one of green/yellow/red")
    return s


def reward(predicted: str, ground_truth: str) -> float:
    p = URGENCY_ORDER[normalize(predicted)]
    g = URGENCY_ORDER[normalize(ground_truth)]
    d = abs(p - g)
    if d == 0:
        return 1.0
    if d == 1:
        return 0.5
    return 0.0


def is_false_negative_red_to_green(predicted: str, ground_truth: str) -> bool:
    """The most-dangerous failure mode per docs/EVALUATION.md: predicting Green
    when truth is Red. Tracked separately from average reward."""
    return normalize(ground_truth) == "red" and normalize(predicted) == "green"
