"""Stratified eval breakdown — reads results/baseline_metrics.json (+ optional
tuned_metrics.json) plus data/cases.jsonl, and prints / writes per-stratum
metrics tables.

Strata reported (per docs/EVALUATION.md and CLAUDE.md):
  1. by ground-truth urgency (Red / Yellow / Green) — already in the confusion
     matrix; the per-bucket reward is what's new here.
  2. by adversarial perturbation (clean vs. perturbed; and per-perturbation
     subtype: contradicted_narrative, off_distribution_image, dialect noise).
  3. by red-flag severity (cases with >=3 red flags vs. <3).

The skin-tone-stratified breakdown (light/medium/dark per Fitzpatrick) is
NOT computed here: the synthetic test set carries no Fitzpatrick metadata.
That's documented as v2 work in docs/PLAN.md.

Output:
  evidence/stratified_report.txt — human-readable table
  results/stratified.json        — machine-readable structure (one entry per
                                   metrics-source file passed in)

Usage:
  .venv/bin/python scripts/stratified_report.py results/baseline_metrics.json
  .venv/bin/python scripts/stratified_report.py results/baseline_metrics.json results/tuned_metrics.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from harness.metrics import CaseScore, stratified_aggregate

EVIDENCE = "evidence/stratified_report.txt"
OUT_JSON = "results/stratified.json"


def _load_cases(path: str = "data/cases.jsonl") -> dict[str, dict]:
    return {json.loads(l)["case_id"]: json.loads(l) for l in open(path) if l.strip()}


def _scores_from_metrics(path: str) -> list[CaseScore]:
    data = json.load(open(path))
    out = []
    for s in data["metrics"].get("per_case", []):
        out.append(CaseScore(
            case_id=s["case_id"],
            predicted=s["predicted"],
            ground_truth=s["ground_truth"],
            reward=s["reward"],
            exact=s["exact"],
            within_one=s["within_one"],
            fn_red_to_green=s["fn_red_to_green"],
        ))
    return out


def _key_urgency(case: dict) -> str:
    return case["ground_truth_urgency"]


def _key_perturbation(case: dict) -> str:
    return "perturbed" if case.get("adversarial_features") else "clean"


def _key_perturbation_subtype(case: dict) -> str:
    feats = case.get("adversarial_features") or []
    # Pick the first specifically named subtype if any; else "other_or_clean".
    for f in feats:
        if "contradict" in f:
            return "contradicted_narrative"
        if "off_distribution" in f or "blurred" in f:
            return "off_distribution_image"
        if "dialect" in f:
            return "dialect_noise"
    return "clean_or_minor"


def _key_red_flag_severity(case: dict) -> str:
    n = len(case.get("red_flags") or [])
    if n >= 3:
        return "high (>=3 red flags)"
    if n >= 1:
        return "low (1-2 red flags)"
    return "none"


STRATIFICATIONS = [
    ("by ground-truth urgency", _key_urgency),
    ("by perturbation (clean vs. perturbed)", _key_perturbation),
    ("by perturbation subtype", _key_perturbation_subtype),
    ("by red-flag severity", _key_red_flag_severity),
]


def _format_table(metric_dicts: dict[str, dict]) -> list[str]:
    rows = []
    rows.append(f"  {'bucket':<32s} {'n':>4s}  {'mean_R':>7s}  {'exact':>6s}  {'within1':>7s}  {'fn_R->G':>7s}")
    for bucket, m in sorted(metric_dicts.items()):
        rows.append(
            f"  {bucket:<32s} {m['n']:>4d}  {m['mean_reward']:>7.3f}  "
            f"{m['exact_match_rate']:>6.3f}  {m['within_one_rate']:>7.3f}  "
            f"{m['fn_red_to_green_rate']:>7.3f}"
        )
    return rows


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: stratified_report.py <metrics.json> [<metrics2.json> ...]", file=sys.stderr)
        return 1

    cases_by_id = _load_cases()
    Path("evidence").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    out_lines: list[str] = []
    json_out: dict = {}

    for metrics_path in sys.argv[1:]:
        scores = _scores_from_metrics(metrics_path)
        out_lines.append(f"=== {metrics_path} ({len(scores)} cases) ===")
        json_out[metrics_path] = {}
        for label, key_fn in STRATIFICATIONS:
            stratum = stratified_aggregate(scores, cases_by_id, key_fn)
            json_out[metrics_path][label] = stratum
            out_lines.append(f"\n-- {label} --")
            out_lines.extend(_format_table(stratum))
        out_lines.append("")

    out_lines.append(
        "Note: skin-tone (Fitzpatrick) stratification is not reported — the "
        "synthetic test set carries no skin-tone labels. See docs/PLAN.md "
        "v2 for the HAM10000 + Fitzpatrick expansion plan."
    )

    text = "\n".join(out_lines) + "\n"
    print(text)
    with open(EVIDENCE, "w") as f:
        f.write(text)
    with open(OUT_JSON, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"wrote {EVIDENCE}")
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
