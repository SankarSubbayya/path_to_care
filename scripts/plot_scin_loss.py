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

    # Token-accuracy on a secondary axis if present
    if any("mean_token_accuracy" in r for r in train):
        ax2 = ax.twinx()
        ax2.plot([r["step"] for r in train],
                 [r.get("mean_token_accuracy", 0) for r in train],
                 ":d", color="#2ca02c", alpha=0.5,
                 label=f"train mean_token_acc (last {train[-1].get('mean_token_accuracy', 0):.2f})",
                 markersize=3, lw=1)
        ax2.set_ylabel("mean_token_accuracy", color="#2ca02c")
        ax2.tick_params(axis="y", colors="#2ca02c")
        ax2.set_ylim(0, 1)
        ax2.legend(loc="lower right")

    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss")
    ax.set_title(
        "SCIN dx-34 multimodal LoRA on Gemma 4 31B-it (MI300X, ROCm 6.3)\n"
        f"target_modules: language_model self_attn q/k/v/o, r=16, alpha=32  |  "
        f"steps: {train[-1]['step']}"
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
