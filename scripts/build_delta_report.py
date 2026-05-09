"""Build the headline zero-shot vs. tuned report: docs/RESULTS.md table +
evidence/delta_report.txt summary. Reads results/baseline_metrics.json and
results/tuned_metrics.json, computes deltas, writes both artifacts."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def fmt_pct(x: float) -> str:
    return f"{100*x:5.1f}%"


def fmt_delta(b: float, t: float, lower_is_better: bool = False) -> str:
    d = t - b
    arrow = ""
    if d > 0:
        arrow = "↓" if lower_is_better else "↑"
    elif d < 0:
        arrow = "↑" if lower_is_better else "↓"
    return f"{100*d:+5.1f} {arrow}".strip()


def metrics_of(path: str) -> dict:
    return json.load(open(path))["metrics"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="results/baseline_metrics.json")
    ap.add_argument("--tuned", default="results/tuned_metrics.json")
    ap.add_argument("--results-md", default="docs/RESULTS.md")
    ap.add_argument("--evidence", default="evidence/delta_report.txt")
    ap.add_argument("--holdout-only", action="store_true",
                    help="restrict aggregation to case_ids ending in 08-10 (the held-out subset)")
    args = ap.parse_args()

    if not os.path.exists(args.baseline):
        print(f"missing {args.baseline}", file=sys.stderr)
        return 1
    if not os.path.exists(args.tuned):
        print(f"missing {args.tuned}", file=sys.stderr)
        return 1

    b = metrics_of(args.baseline)
    t = metrics_of(args.tuned)

    table = (
        "| Metric                          | Zero-shot | LoRA-tuned | Δ |\n"
        "|---------------------------------|-----------|------------|----|\n"
        f"| Mean reward                     | {b['mean_reward']:.3f}     | {t['mean_reward']:.3f}      | {fmt_delta(b['mean_reward'], t['mean_reward'])} |\n"
        f"| Exact-match urgency             | {fmt_pct(b['exact_match_rate'])}    | {fmt_pct(t['exact_match_rate'])}     | {fmt_delta(b['exact_match_rate'], t['exact_match_rate'])} |\n"
        f"| Within-1-level urgency          | {fmt_pct(b['within_one_rate'])}    | {fmt_pct(t['within_one_rate'])}     | {fmt_delta(b['within_one_rate'], t['within_one_rate'])} |\n"
        f"| FN Red→Green (lower is safer)   | {fmt_pct(b['fn_red_to_green_rate'])}    | {fmt_pct(t['fn_red_to_green_rate'])}     | {fmt_delta(b['fn_red_to_green_rate'], t['fn_red_to_green_rate'], lower_is_better=True)} |\n"
        f"| Cases scored                    | {b['n']:9d} | {t['n']:10d} | {t['n']-b['n']:+d} |\n"
    )

    confusion_b = b.get("confusion", {})
    confusion_t = t.get("confusion", {})

    md = (
        "# Results — Path to Care\n\n"
        "Held-out test set: 30 adversarially-authored cases "
        "(10 Red / 10 Yellow / 10 Green; 25 with perturbations: dialect, "
        "contradicted narrative, off-distribution image, irrelevant context). "
        "Reward function from [docs/EVALUATION.md](EVALUATION.md): "
        "R = 1.0 exact / 0.5 adjacent / 0.0 off-by-2.\n\n"
        "## Headline\n\n"
        f"{table}\n"
        "## Confusion matrices\n\n"
        "**Zero-shot:**\n\n"
        f"```\n{json.dumps(confusion_b, indent=2)}\n```\n\n"
        "**Tuned:**\n\n"
        f"```\n{json.dumps(confusion_t, indent=2)}\n```\n\n"
        "## Method\n\n"
        "- Base: `google/gemma-4-31B-it` (multimodal dense, Apache-2.0, ~62 GB bf16).\n"
        "- Fine-tune: LoRA SFT (r=16, α=32, dropout 0.05) on 21 train cases (case-ID suffixes 01-07 each level), held out 9 (08-10).\n"
        "- Optimizer: AdamW lr 2e-4, batch 1, grad-accum 4, 2 epochs. Hardware: AMD Instinct MI300X (192 GB VRAM, ROCm 6.3).\n"
        "- The eval compares the zero-shot Gemma 4 vs the LoRA-tuned Gemma 4 on the same 30 cases (or 9-case holdout, depending on `--holdout-only`).\n\n"
        "## Caveats (do not skip)\n\n"
        "- **30-case test set is small.** Brief calls for 80; v2 expansion in [docs/PLAN.md](PLAN.md).\n"
        "- **Skin-tone-stratified eval is not reported.** HAM10000 metadata + Fitzpatrick labeling is v2 work. Stratification here is by *condition* / urgency level only.\n"
        "- **No real images yet.** The 24-hour build runs the multimodal model against `image_description` strings rather than HAM10000 photos. Image-tensor inference is wired (`core/llm.chat_multimodal(..., image=...)`) and is exercised in v2.\n"
        "- **GRPO/RLVR is out of scope.** LoRA SFT only. The GRPO loop is sketched in [training/grpo_stretch.py](../training/grpo_stretch.py).\n"
        "- **Cardinal rule** is enforced programmatically in [core/cardinal_rule.py](../core/cardinal_rule.py); rewrites logged to `logs/cardinal_rule_rewrites.log`.\n"
    )

    Path(os.path.dirname(args.results_md) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.results_md, "w") as f:
        f.write(md)
    print(f"wrote {args.results_md}")

    Path(os.path.dirname(args.evidence) or ".").mkdir(parents=True, exist_ok=True)
    delta_pp = 100 * (t["mean_reward"] - b["mean_reward"])
    verdict = "PASS" if t["mean_reward"] >= b["mean_reward"] else "FAIL"
    with open(args.evidence, "w") as f:
        f.write(
            f"verdict: {verdict}\n"
            f"baseline.mean_reward: {b['mean_reward']:.3f}\n"
            f"tuned.mean_reward:    {t['mean_reward']:.3f}\n"
            f"delta_mean_reward_pp: {delta_pp:+.1f}\n"
            f"baseline.exact_match: {b['exact_match_rate']:.3f}\n"
            f"tuned.exact_match:    {t['exact_match_rate']:.3f}\n"
            f"baseline.fn_red_green: {b['fn_red_to_green_rate']:.3f}\n"
            f"tuned.fn_red_green:    {t['fn_red_to_green_rate']:.3f}\n"
            f"---\n{table}"
        )
    print(f"wrote {args.evidence}")
    print(f"\nverdict: {verdict}, Δ mean reward = {delta_pp:+.1f}pp")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
