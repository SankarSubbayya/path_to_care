"""Plot the SCIN multimodal LoRA loss curve from logs/lora_dx_multimodal.log."""
import os, re, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> int:
    log_path = sys.argv[1] if len(sys.argv) > 1 else "logs/lora_dx_multimodal.log"
    out_png = sys.argv[2] if len(sys.argv) > 2 else "docs/figures/scin_lora_loss.png"
    Path(os.path.dirname(out_png)).mkdir(parents=True, exist_ok=True)

    train_steps, train_loss = [], []
    eval_steps,  eval_loss  = [], []

    # SFTTrainer prints lines like:
    #  {'loss': '7.616', ..., 'epoch': '0.3333'}
    #  {'eval_loss': '3.119', ..., 'epoch': '1'}
    pat_train = re.compile(r"\{'loss'\s*:\s*'([\d.]+)'.*?'epoch'\s*:\s*'([\d.]+)'")
    pat_eval  = re.compile(r"\{'eval_loss'\s*:\s*'([\d.]+)'.*?'epoch'\s*:\s*'([\d.]+)'")
    step = 0
    eval_step = 0
    for line in open(log_path):
        m = pat_train.search(line)
        if m:
            step += 1
            train_steps.append(step)
            train_loss.append(float(m.group(1)))
            continue
        m = pat_eval.search(line)
        if m:
            eval_steps.append(step)
            eval_loss.append(float(m.group(1)))

    if not train_loss:
        print("no loss lines found — log may not be ready yet")
        return 1

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_steps, train_loss, "-o", label=f"train_loss (final {train_loss[-1]:.2f})",
            color="#1f77b4", linewidth=1.5, markersize=3)
    if eval_loss:
        ax.plot(eval_steps, eval_loss, "-s", label=f"eval_loss (final {eval_loss[-1]:.2f})",
                color="#ff7f0e", linewidth=2)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss")
    ax.set_title("SCIN dx-34 multimodal LoRA on Gemma 4 31B-it (MI300X)\n"
                 f"target_modules: language_model self_attn q/k/v/o, r=16, alpha=32")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()
    print(f"wrote {out_png}  ({len(train_loss)} train pts, {len(eval_loss)} eval pts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
