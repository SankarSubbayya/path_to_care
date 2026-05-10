"""Reward function unit tests — pytest port of the original
harness/test_reward.py runner."""
from __future__ import annotations

import pytest

from harness.reward import reward, is_false_negative_red_to_green, normalize


@pytest.mark.parametrize(
    "predicted,truth,expected",
    [
        ("red", "red", 1.0),
        ("yellow", "yellow", 1.0),
        ("green", "green", 1.0),
        ("yellow", "red", 0.5),
        ("red", "yellow", 0.5),
        ("green", "yellow", 0.5),
        ("yellow", "green", 0.5),
        ("green", "red", 0.0),
        ("red", "green", 0.0),
    ],
)
def test_reward_levels(predicted, truth, expected):
    assert reward(predicted, truth) == expected


@pytest.mark.parametrize(
    "predicted,truth,expected",
    [
        ("Red", "RED", 1.0),
        (" yellow ", "red", 0.5),
        ("GREEN", "Green", 1.0),
    ],
)
def test_reward_case_insensitivity_and_whitespace(predicted, truth, expected):
    assert reward(predicted, truth) == expected


@pytest.mark.parametrize(
    "predicted,truth,expected",
    [
        ("green", "red", True),
        ("yellow", "red", False),
        ("green", "green", False),
        ("red", "green", False),
        ("green", "yellow", False),
    ],
)
def test_false_negative_red_to_green(predicted, truth, expected):
    assert is_false_negative_red_to_green(predicted, truth) is expected


def test_normalize_rejects_unknown():
    with pytest.raises(ValueError):
        normalize("orange")


def test_reward_rejects_unknown():
    with pytest.raises(ValueError):
        reward("red", "orange")
