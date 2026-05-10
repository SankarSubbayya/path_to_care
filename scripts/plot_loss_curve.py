"""Plot the LoRA SFT loss curve from logs/lora_train.log to a PNG.

Reads lines of the form `[HH:MM:SS] epoch=N step=K avg_loss_in_step=L` and
plots step vs. loss with epoch boundaries marked. Output:
docs/figures/lora_loss_curve.png.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LINE_RE = re.compile(
    r"epoch=(?P<epoch>\d+)\s+step=(?P<step>\d+)\s+avg_loss_in_step=(?P<loss>[\d.]+)"
)


def parse(path: str) -> list[tuple[int, int, float]]:
    rows = []
    for line in open(path):
        m = LINE_RE.search(line)
        if m:
            rows.append((int(m["epoch"]), int(m["step"]), float(m["loss"])))
    return rows


def main() -> int:
    log_path = sys.argv[1] if len(sys.argv) > 1 else "logs/lora_train.log"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "docs/figures/lora_loss_curve.png"

    rows = parse(log_path)
    if not rows:
        print(f"no loss-curve lines parsed from {log_path}", file=sys.stderr)
        return 1

    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)

    epochs = [r[0] for r in rows]
    steps = [r[1] for r in rows]
    losses = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(steps, losses, marker="o", linewidth=2, color="#c0392b", label="avg loss / accumulation step")
    ax.set_xlabel("Optimizer step (effective batch size = 4, grad_accum)")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Path to Care — LoRA SFT on Gemma 4 31B-it (AMD MI300X, ROCm 6.3)")
    ax.grid(True, alpha=0.3)

    # Epoch boundary lines
    boundaries: list[int] = []
    for i in range(1, len(rows)):
        if epochs[i] != epochs[i - 1]:
            boundaries.append((steps[i - 1] + steps[i]) / 2)
    for x in boundaries:
        ax.axvline(x=x, color="#7f8c8d", linestyle="--", alpha=0.5)
    if boundaries:
        ax.text(boundaries[0], max(losses) * 0.95, " epoch 0 → 1",
                color="#7f8c8d", fontsize=9, va="top")

    # Annotate first/last
    ax.annotate(f"start: {losses[0]:.2f}",
                xy=(steps[0], losses[0]),
                xytext=(steps[0] + 0.3, losses[0] + 0.3),
                fontsize=9, color="#34495e")
    ax.annotate(f"end: {losses[-1]:.2f}",
                xy=(steps[-1], losses[-1]),
                xytext=(steps[-1] - 1.6, losses[-1] - 0.45),
                fontsize=9, color="#34495e")

    # Footer with training facts
    delta = losses[0] - losses[-1]
    pct = 100 * delta / max(losses[0], 1e-9)
    fig.text(
        0.5, -0.02,
        f"21 train rows · 2 epochs · 10 optimizer steps · 32 s wall · "
        f"loss {losses[0]:.2f} → {losses[-1]:.2f} ({-delta:+.2f}, −{pct:.0f}%) · "
        f"45 M trainable params (0.14% of 31 B)",
        ha="center", fontsize=8, color="#555"
    )
    ax.legend(loc="upper right")

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}  (start={losses[0]:.3f}, end={losses[-1]:.3f}, delta={-delta:+.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
