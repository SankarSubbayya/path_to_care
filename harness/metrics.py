"""Eval metrics for Path to Care.

Computes per-case scores and aggregate metrics matching docs/EVALUATION.md
reporting template:
  - mean reward (R = 1.0 / 0.5 / 0.0)
  - exact-match accuracy
  - within-1-level accuracy
  - confusion matrix (3x3)
  - false-negative Red→Green rate (cardinal safety metric)
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Iterable

from collections import defaultdict
from typing import Callable

from harness.reward import reward, is_false_negative_red_to_green, normalize, URGENCY_ORDER


@dataclass
class CaseScore:
    case_id: str
    predicted: str
    ground_truth: str
    reward: float
    exact: bool
    within_one: bool
    fn_red_to_green: bool


@dataclass
class AggregateMetrics:
    n: int
    mean_reward: float
    exact_match_rate: float
    within_one_rate: float
    fn_red_to_green_rate: float  # (predictions of green when truth is red) / (total truth-red cases)
    confusion: dict  # {(truth, pred): count}
    per_case: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        # tuple keys aren't JSON-serializable; flatten to nested dict
        d["confusion"] = {
            t: {p: self.confusion.get((t, p), 0) for p in URGENCY_ORDER}
            for t in URGENCY_ORDER
        }
        return d


def score_case(case_id: str, predicted: str, ground_truth: str) -> CaseScore:
    p = normalize(predicted)
    g = normalize(ground_truth)
    r = reward(p, g)
    return CaseScore(
        case_id=case_id,
        predicted=p,
        ground_truth=g,
        reward=r,
        exact=(r == 1.0),
        within_one=(r >= 0.5),
        fn_red_to_green=is_false_negative_red_to_green(p, g),
    )


def aggregate(scores: Iterable[CaseScore]) -> AggregateMetrics:
    scores = list(scores)
    n = len(scores)
    if n == 0:
        return AggregateMetrics(0, 0.0, 0.0, 0.0, 0.0, {}, [])

    confusion: dict = {}
    for s in scores:
        confusion[(s.ground_truth, s.predicted)] = confusion.get((s.ground_truth, s.predicted), 0) + 1

    n_truth_red = sum(1 for s in scores if s.ground_truth == "red")
    n_fn_red_green = sum(1 for s in scores if s.fn_red_to_green)
    fn_rate = n_fn_red_green / n_truth_red if n_truth_red > 0 else 0.0

    return AggregateMetrics(
        n=n,
        mean_reward=sum(s.reward for s in scores) / n,
        exact_match_rate=sum(1 for s in scores if s.exact) / n,
        within_one_rate=sum(1 for s in scores if s.within_one) / n,
        fn_red_to_green_rate=fn_rate,
        confusion=confusion,
        per_case=[asdict(s) for s in scores],
    )


def stratified_aggregate(
    scores: Iterable[CaseScore],
    cases_by_id: dict[str, dict],
    key_fn: Callable[[dict], str | tuple[str, ...]],
) -> dict:
    """Group scores by `key_fn(case_dict)` and compute per-stratum metrics.

    `cases_by_id` maps case_id -> the original case dict from cases.jsonl.
    `key_fn` is a function that takes the case dict and returns a string (or
    tuple) bucket label. E.g.:

      stratified_aggregate(scores, cases_by_id, lambda c: c['ground_truth_urgency'])
        -> {'red': {...metrics...}, 'yellow': {...}, 'green': {...}}

      stratified_aggregate(scores, cases_by_id,
                          lambda c: 'perturbed' if c.get('adversarial_features') else 'clean')
        -> {'perturbed': {...}, 'clean': {...}}

    Returns a dict of stratum -> AggregateMetrics.to_dict() (sans per_case to
    keep the report compact)."""
    buckets: dict[str, list[CaseScore]] = defaultdict(list)
    for s in scores:
        case = cases_by_id.get(s.case_id)
        if case is None:
            continue
        bucket = key_fn(case)
        # tuple keys -> string for JSON friendliness
        if isinstance(bucket, tuple):
            bucket = "+".join(map(str, bucket))
        buckets[bucket].append(s)

    out = {}
    for bucket, bucket_scores in buckets.items():
        agg = aggregate(bucket_scores)
        d = agg.to_dict()
        d.pop("per_case", None)  # keep stratified report compact
        out[bucket] = d
    return out
