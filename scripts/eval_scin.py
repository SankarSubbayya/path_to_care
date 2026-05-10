"""SCIN holdout eval: send each prompt directly to a vLLM endpoint, parse the
URGENCY line out of the response, score against ground_truth_urgency.

Skips the orchestrator pre-pass because data/scin/holdout_prompts.jsonl already
contains the fully-formed triage-reasoner prompts (built by
scripts/build_scin_prompts.py from real SCIN dermatologist labels + symptoms +
Fitzpatrick metadata).

Outputs:
  results/scin_<run-name>.json   per-case scores + aggregate metrics
  results/scin_<run-name>.log    per-case progress

Stratifies by ground-truth urgency, Fitzpatrick bucket, and condition.

Usage:
  PTC_VLLM_API_KEY=ptc-demo-2026-amd \
  .venv/bin/python scripts/eval_scin.py \
    --base-url http://localhost:8000/v1 \
    --model google/gemma-4-31B-it \
    --in data/scin/holdout_prompts.jsonl \
    --out results/scin_baseline.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from harness.reward import reward, is_false_negative_red_to_green, normalize


_URGENCY_RE = re.compile(r"URGENCY\s*:\s*(red|yellow|green)\b", re.IGNORECASE)


def parse_urgency(text: str) -> tuple[str, bool]:
    """Return (urgency, parse_ok). If unparseable, default to 'green' so the
    safety bias is toward over-triage by the holdout (the cardinal rule)."""
    m = _URGENCY_RE.search(text or "")
    if m:
        return m.group(1).lower(), True
    return "green", False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("PTC_VLLM_GEMMA4_URL", "http://localhost:8000/v1"))
    ap.add_argument("--api-key",  default=os.environ.get("PTC_VLLM_API_KEY", "ptc-demo-2026-amd"))
    ap.add_argument("--model",    default=os.environ.get("PTC_VLLM_GEMMA4_MODEL_ID", "google/gemma-4-31B-it"))
    ap.add_argument("--in", dest="in_path", default="data/scin/holdout_prompts.jsonl")
    ap.add_argument("--out", required=True, help="path to write results JSON")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=192)
    args = ap.parse_args()

    from openai import OpenAI

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    log_path = args.out.replace(".json", ".log")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    rows = [json.loads(l) for l in open(args.in_path) if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    print(f"== SCIN eval ==")
    print(f"  base_url: {args.base_url}")
    print(f"  model:    {args.model}")
    print(f"  rows:     {len(rows)}")
    print(f"  out:      {args.out}")

    log_f = open(log_path, "w")
    per_case = []
    t_total = time.time()

    for i, r in enumerate(rows, 1):
        t0 = time.time()
        truth = r["ground_truth_urgency"]
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": r["prompt"]}],
                max_tokens=args.max_tokens,
                temperature=0,
            )
            raw = (resp.choices[0].message.content or "").strip()
            pred, parse_ok = parse_urgency(raw)
            R = reward(pred, truth)
            line = (
                f"[{i:03d}/{len(rows)}] {r['case_id']:24s} truth={truth:7s} "
                f"pred={pred:7s} R={R:.1f} cond={r.get('condition','?'):24s} "
                f"fst={r.get('fitzpatrick_bucket','?'):8s} t={time.time()-t0:.1f}s"
                + ("" if parse_ok else " [PARSE-FAIL]")
            )
            print(line); log_f.write(line + "\n"); log_f.flush()
            per_case.append({
                "case_id": r["case_id"], "ground_truth": normalize(truth),
                "predicted": pred, "reward": R, "parse_ok": parse_ok,
                "raw": raw, "condition": r.get("condition"),
                "fitzpatrick_bucket": r.get("fitzpatrick_bucket"),
            })
        except Exception as e:
            line = f"[{i:03d}/{len(rows)}] {r['case_id']} ERROR {type(e).__name__}: {e}"
            print(line); log_f.write(line + "\n"); log_f.flush()
            per_case.append({"case_id": r["case_id"], "ground_truth": normalize(truth),
                             "predicted": "green", "reward": 0.0, "parse_ok": False,
                             "raw": "", "condition": r.get("condition"),
                             "fitzpatrick_bucket": r.get("fitzpatrick_bucket"),
                             "error": str(e)})
    log_f.close()

    n = len(per_case)
    n_red = sum(1 for c in per_case if c["ground_truth"] == "red")
    fn_rg = sum(1 for c in per_case
                if is_false_negative_red_to_green(c["predicted"], c["ground_truth"]))
    confusion = Counter()
    for c in per_case:
        confusion[(c["ground_truth"], c["predicted"])] += 1

    # Stratification by Fitzpatrick bucket
    by_fst: dict[str, list] = defaultdict(list)
    for c in per_case:
        by_fst[c["fitzpatrick_bucket"] or "unknown"].append(c)
    fst_metrics = {}
    for bucket, scores in by_fst.items():
        n_b = len(scores)
        n_red_b = sum(1 for c in scores if c["ground_truth"] == "red")
        fn_rg_b = sum(1 for c in scores
                      if is_false_negative_red_to_green(c["predicted"], c["ground_truth"]))
        fst_metrics[bucket] = {
            "n": n_b,
            "mean_reward": sum(c["reward"] for c in scores) / max(n_b, 1),
            "exact_match_rate": sum(1 for c in scores if c["reward"] == 1.0) / max(n_b, 1),
            "within_one_rate": sum(1 for c in scores if c["reward"] >= 0.5) / max(n_b, 1),
            "fn_red_to_green_rate": fn_rg_b / max(n_red_b, 1) if n_red_b else 0.0,
            "n_red_in_bucket": n_red_b,
        }

    # Stratification by condition
    by_cond: dict[str, list] = defaultdict(list)
    for c in per_case:
        by_cond[c["condition"] or "?"].append(c)
    cond_metrics = {}
    for cond, scores in by_cond.items():
        n_c = len(scores)
        cond_metrics[cond] = {
            "n": n_c,
            "mean_reward": sum(c["reward"] for c in scores) / max(n_c, 1),
            "exact_match_rate": sum(1 for c in scores if c["reward"] == 1.0) / max(n_c, 1),
        }

    out = {
        "_meta": {
            "in": args.in_path, "out": args.out, "model": args.model,
            "base_url": args.base_url, "n": n,
            "wall_seconds": round(time.time() - t_total, 1),
        },
        "metrics": {
            "n": n,
            "mean_reward": sum(c["reward"] for c in per_case) / max(n, 1),
            "exact_match_rate": sum(1 for c in per_case if c["reward"] == 1.0) / max(n, 1),
            "within_one_rate": sum(1 for c in per_case if c["reward"] >= 0.5) / max(n, 1),
            "fn_red_to_green_rate": fn_rg / max(n_red, 1) if n_red else 0.0,
            "parse_fail_rate": sum(1 for c in per_case if not c["parse_ok"]) / max(n, 1),
            "confusion": {f"{t}->{p}": v for (t, p), v in confusion.items()},
            "by_fitzpatrick": fst_metrics,
            "by_condition": cond_metrics,
        },
        "per_case": per_case,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    m = out["metrics"]
    print()
    print(f"== eval done ==")
    print(f"  mean_reward:           {m['mean_reward']:.3f}")
    print(f"  exact_match_rate:      {m['exact_match_rate']:.3f}")
    print(f"  within_one_rate:       {m['within_one_rate']:.3f}")
    print(f"  fn_red_to_green_rate:  {m['fn_red_to_green_rate']:.3f}")
    print(f"  parse_fail_rate:       {m['parse_fail_rate']:.3f}")
    print(f"  wall_seconds:          {out['_meta']['wall_seconds']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
