"""Eval harness: runs the orchestrator against `data/cases.jsonl`, scores
predictions against ground-truth urgency, writes a metrics JSON.

Usage (always via the venv):
  .venv/bin/python -m harness.run --out results/baseline_metrics.json
  .venv/bin/python -m harness.run --adapter adapters/triage-gemma4-lora --out results/tuned_metrics.json

The output JSON is the canonical evidence for the `baseline_metrics_recorded`
and `tuned_metrics_recorded` features in test-results.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from harness.metrics import score_case, aggregate
from orchestrator.agent import run_case


def load_cases(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="data/cases.jsonl")
    ap.add_argument("--out", required=True, help="results JSON output path")
    ap.add_argument("--adapter", default=None, help="optional LoRA adapter path for the triage reasoner")
    ap.add_argument("--limit", type=int, default=None, help="run only the first N cases (for fast smoke)")
    args = ap.parse_args()

    cases = load_cases(args.cases)
    if args.limit:
        cases = cases[: args.limit]

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    log_path = args.out.replace(".json", ".log")

    print(f"== eval start ==")
    print(f"  cases:   {len(cases)}")
    print(f"  adapter: {args.adapter or '(none / zero-shot baseline)'}")
    print(f"  out:     {args.out}")
    print(f"  log:     {log_path}")

    scores = []
    traces = []
    t_total = time.time()
    log_f = open(log_path, "w")
    for i, case in enumerate(cases, 1):
        t0 = time.time()
        try:
            trace = run_case(case, adapter_path=args.adapter)
            score = score_case(case["case_id"], trace.urgency, case["ground_truth_urgency"])
            scores.append(score)
            traces.append(trace.to_dict())
            elapsed = time.time() - t0
            line = (
                f"[{i:02d}/{len(cases)}] {case['case_id']} "
                f"truth={case['ground_truth_urgency']} pred={score.predicted} "
                f"R={score.reward:.1f} t={elapsed:.1f}s "
                f"esc={trace.safety_escalation}"
            )
            print(line)
            log_f.write(line + "\n")
            log_f.flush()
        except Exception as e:
            line = f"[{i:02d}/{len(cases)}] {case['case_id']} ERROR {type(e).__name__}: {e}"
            print(line)
            log_f.write(line + "\n")
            log_f.flush()
            traces.append({"case_id": case["case_id"], "error": str(e)})
    log_f.close()

    agg = aggregate(scores)
    output = {
        "_meta": {
            "cases_path": args.cases,
            "adapter": args.adapter,
            "n_cases_attempted": len(cases),
            "n_cases_scored": len(scores),
            "wall_seconds": round(time.time() - t_total, 1),
        },
        "metrics": agg.to_dict(),
        "traces": traces,
    }
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"== eval done ==")
    print(f"  mean_reward:           {agg.mean_reward:.3f}")
    print(f"  exact_match_rate:      {agg.exact_match_rate:.3f}")
    print(f"  within_one_rate:       {agg.within_one_rate:.3f}")
    print(f"  fn_red_to_green_rate:  {agg.fn_red_to_green_rate:.3f}  (lower is safer)")
    print(f"  wall_seconds:          {output['_meta']['wall_seconds']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
