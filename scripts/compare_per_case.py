"""Side-by-side per-case comparison: zero-shot baseline vs. LoRA-tuned.

Reads results/baseline_metrics.json and results/tuned_metrics.json, prints a
table of (case_id, truth, base_pred, tuned_pred, reasoning_changed). Use
`--full <case_id>` to dump the complete reasoning + framing text for one case
in both modes. Useful for pitch/demo when explaining what the LoRA actually
shifts (urgency stays the same on this saturated set, reasoning style shifts
toward the training template on a couple of Red cases).
"""
from __future__ import annotations

import argparse
import json
import sys


def _load(path: str) -> dict[str, dict]:
    data = json.load(open(path))
    out = {}
    # Per-case scores live under metrics.per_case
    for c in data["metrics"].get("per_case", []):
        out[c["case_id"]] = c
    # Reasoning/framing live under traces (one entry per case_id)
    for t in data.get("traces", []):
        cid = t.get("case_id")
        if cid and cid in out:
            out[cid]["reasoning"] = t.get("reasoning", "")
            out[cid]["patient_framing"] = t.get("patient_framing", "")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="results/baseline_metrics.json")
    ap.add_argument("--tuned", default="results/tuned_metrics.json")
    ap.add_argument("--full", help="dump full reasoning+framing for this case_id")
    args = ap.parse_args()

    base = _load(args.baseline)
    tuned = _load(args.tuned)

    if args.full:
        cid = args.full
        if cid not in base or cid not in tuned:
            print(f"case {cid} not found", file=sys.stderr)
            return 1
        b, t = base[cid], tuned[cid]
        print(f"== {cid} ==")
        print(f"  truth:        {b['ground_truth']}")
        print(f"  base.pred:    {b['predicted']}   (R={b['reward']})")
        print(f"  tuned.pred:   {t['predicted']}   (R={t['reward']})")
        print()
        print("--- base.reasoning ---")
        print(b.get("reasoning", "(missing)"))
        print()
        print("--- tuned.reasoning ---")
        print(t.get("reasoning", "(missing)"))
        print()
        print("--- base.patient_framing ---")
        print(b.get("patient_framing", "(missing)"))
        print()
        print("--- tuned.patient_framing ---")
        print(t.get("patient_framing", "(missing)"))
        return 0

    # Table view
    print(f"{'case_id':<10s} {'truth':<7s} {'base':<7s} {'tuned':<7s} reason_chg framing_chg")
    print("-" * 60)
    n_reason_diff = 0
    n_pred_diff = 0
    for cid in sorted(base.keys()):
        if cid not in tuned:
            continue
        b, t = base[cid], tuned[cid]
        rc = "yes" if b.get("reasoning") != t.get("reasoning") else "no"
        fc = "yes" if b.get("patient_framing") != t.get("patient_framing") else "no"
        if rc == "yes":
            n_reason_diff += 1
        if b["predicted"] != t["predicted"]:
            n_pred_diff += 1
        marker = " ← LoRA shift" if rc == "yes" else ""
        print(f"{cid:<10s} {b['ground_truth']:<7s} {b['predicted']:<7s} {t['predicted']:<7s} "
              f"{rc:<10s} {fc:<11s}{marker}")
    print()
    print(f"summary: {n_pred_diff}/{len(base)} cases differ in urgency, "
          f"{n_reason_diff}/{len(base)} cases differ in reasoning text.")
    print()
    print("Use `--full <case_id>` to see the complete reasoning + framing text for a case.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
