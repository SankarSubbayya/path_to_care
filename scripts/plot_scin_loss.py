"""Plot the SCIN multimodal LoRA loss curve from logs/scin_lora_train.jsonl.

The JsonlLossLogger callback in scripts/lora_dx_multimodal.py writes one line
per logged training step (or eval) — train rows have `loss`, eval rows have
`eval_loss`. Both are plotted against optimizer step.
"""
import json, os, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> int:
    log_path = sys.argv[1] if len(sys.argv) > 1 else "logs/scin_lora_train.jsonl"
    out_png = sys.argv[2] if len(sys.argv) > 2 else "docs/figures/scin_lora_loss.png"
    Path(os.path.dirname(out_png)).mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(log_path) if l.strip()]
    train = [r for r in rows if "loss" in r and "eval_loss" not in r]
    evals = [r for r in rows if "eval_loss" in r]

    if not train:
        print(f"no train rows in {log_path}")
        return 1

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot([r["step"] for r in train], [r["loss"] for r in train],
            "-o", label=f"train_loss (last {train[-1]['loss']:.2f}, n={len(train)})",
            color="#1f77b4", lw=1.4, markersize=3)

    if evals:
        ax.plot([r["step"] for r in evals], [r["eval_loss"] for r in evals],
                "-s", label=f"eval_loss (last {evals[-1]['eval_loss']:.2f}, n={len(evals)})",
                color="#ff7f0e", lw=2)

    # Note: mean_token_accuracy is intentionally NOT plotted. It's a token-level
    # next-token-prediction metric reported by SFTTrainer; it rose to ~95% on
    # earlier failed runs while actual classification accuracy regressed by 11
    # points. Plotting it here would overstate how well the model is "learning"
    # the diagnosis task, which is a classification problem, not a token-LM one.

    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(
        "SCIN top-16 multimodal LoRA on Gemma 4 31B-it (AMD MI300X, ROCm 6.3)\n"
        f"target_modules: language_model self_attn q/k/v/o, r=8, alpha=16  |  "
        f"steps: {train[-1]['step']}  |  result: top-1 28.0% → 35.0% (+7.0 pp)"
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()
    print(f"wrote {out_png}  (train={len(train)}, eval={len(evals)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
