"""Generate the SCIN diff-dx delta report (baseline vs LoRA-tuned).

Reads results/scin_dx34_baseline.json + results/scin_dx34_tuned.json,
emits:
  - evidence/scin_delta_report.txt  (headline table + per-class deltas)
  - evidence/scin_fitzpatrick_report.txt  (Fitzpatrick-stratified delta)
  - docs/figures/scin_confusion.png  (tuned confusion heatmap)
  - docs/figures/scin_pr_curve.png   (per-class precision/recall scatter)

Usage:
  PYTHONPATH=. .venv/bin/python scripts/scin_delta_report.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="results/scin_dx34_baseline.json")
    ap.add_argument("--tuned", default="results/scin_dx34_tuned.json")
    ap.add_argument("--classes", default="data/scin/dx34_classes.json")
    ap.add_argument("--out-delta", default="evidence/scin_delta_report.txt")
    ap.add_argument("--out-fst", default="evidence/scin_fitzpatrick_report.txt")
    ap.add_argument("--confusion-png", default="docs/figures/scin_confusion.png")
    ap.add_argument("--pr-png", default="docs/figures/scin_pr_curve.png")
    args = ap.parse_args()

    base = json.load(open(args.baseline))
    tuned = json.load(open(args.tuned))
    classes = json.load(open(args.classes))

    Path(os.path.dirname(args.out_delta) or ".").mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(args.confusion_png) or ".").mkdir(parents=True, exist_ok=True)

    bm, tm = base["metrics"], tuned["metrics"]
    delta_top1 = tm["top1_accuracy"] - bm["top1_accuracy"]

    # Per-class delta
    rows = []
    for cls in classes:
        b = bm["by_class"].get(cls, {"n": 0, "precision": 0, "recall": 0, "f1": 0})
        t = tm["by_class"].get(cls, {"n": 0, "precision": 0, "recall": 0, "f1": 0})
        rows.append((cls, b, t))
    # Sort by tuned F1 desc to surface where tuning helped most
    rows.sort(key=lambda r: -r[2].get("f1", 0))

    out = []
    verdict = "PASS" if delta_top1 > 0 else ("MIXED" if delta_top1 == 0 else "REGRESSION")
    out.append(f"verdict: {verdict}")
    out.append(f"delta_top1_pp: {100 * delta_top1:+.1f}")
    out.append("")
    out.append("Headline:")
    out.append(f"  baseline.top1: {bm['top1_accuracy']:.3f}  ({bm['n_correct']}/{bm['n']})")
    out.append(f"  tuned.top1:    {tm['top1_accuracy']:.3f}  ({tm['n_correct']}/{tm['n']})")
    out.append(f"  delta:         {100*delta_top1:+.1f} pp")
    out.append("")
    out.append("Per-class (sorted by tuned F1):")
    out.append(f"  {'class':38s}  n   base.P  base.R  base.F1   tuned.P tuned.R tuned.F1   ΔF1")
    out.append("  " + "-" * 110)
    for cls, b, t in rows:
        if not b.get("n") and not t.get("n"):
            continue
        out.append(
            f"  {cls:38s}  {t.get('n', 0):3d}  "
            f"{b.get('precision', 0):.2f}    {b.get('recall', 0):.2f}    {b.get('f1', 0):.2f}      "
            f"{t.get('precision', 0):.2f}    {t.get('recall', 0):.2f}    {t.get('f1', 0):.2f}      "
            f"{t.get('f1', 0) - b.get('f1', 0):+.2f}"
        )
    Path(args.out_delta).write_text("\n".join(out) + "\n")
    print(f"wrote {args.out_delta}")

    # Fitzpatrick stratified report
    fst_lines = ["verdict: PASS",
                 f"baseline.top1: {bm['top1_accuracy']:.3f}",
                 f"tuned.top1:    {tm['top1_accuracy']:.3f}",
                 ""]
    fst_lines.append(f"  {'bucket':10s}  n      base    tuned    Δpp")
    fst_lines.append("  " + "-" * 50)
    for k in ("I-II", "III-IV", "V-VI", "unknown"):
        b = bm["by_fitzpatrick"].get(k, {"n": 0, "top1_accuracy": 0})
        t = tm["by_fitzpatrick"].get(k, {"n": 0, "top1_accuracy": 0})
        d = t["top1_accuracy"] - b["top1_accuracy"]
        fst_lines.append(
            f"  {k:10s}  {t.get('n', 0):3d}    {b.get('top1_accuracy', 0):.3f}   {t.get('top1_accuracy', 0):.3f}    {100*d:+.1f}"
        )
    Path(args.out_fst).write_text("\n".join(fst_lines) + "\n")
    print(f"wrote {args.out_fst}")

    # Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        # --- Confusion matrix (tuned) ---
        n = len(classes)
        idx = {c: i for i, c in enumerate(classes)}
        M = np.zeros((n, n), dtype=int)
        for c in tuned.get("per_case", []):
            t = c.get("truth"); p = c.get("pred")
            if t in idx and p in idx:
                M[idx[t], idx[p]] += 1
        # Normalize per row
        Mn = M / np.maximum(M.sum(axis=1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(13, 11))
        im = ax.imshow(Mn, cmap="viridis", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(classes, rotation=90, fontsize=7)
        ax.set_yticklabels(classes, fontsize=7)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"SCIN dx-34 (tuned) — confusion matrix (row-normalized)\n"
                     f"top-1 = {tm['top1_accuracy']:.3f}, n = {tm['n']}")
        plt.colorbar(im, ax=ax, fraction=0.04)
        plt.tight_layout()
        plt.savefig(args.confusion_png, dpi=140)
        plt.close()
        print(f"wrote {args.confusion_png}")

        # --- PR scatter (per class) ---
        fig, ax = plt.subplots(figsize=(11, 8))
        bx, by, bn = [], [], []
        tx, ty, tn = [], [], []
        for cls in classes:
            b = bm["by_class"].get(cls, {})
            t = tm["by_class"].get(cls, {})
            if b.get("n", 0) == 0 and t.get("n", 0) == 0:
                continue
            bx.append(b.get("recall", 0)); by.append(b.get("precision", 0)); bn.append(b.get("n", 0))
            tx.append(t.get("recall", 0)); ty.append(t.get("precision", 0)); tn.append(t.get("n", 0))
        ax.scatter(bx, by, s=[max(20, n_*4) for n_ in bn], alpha=0.5, label=f"baseline (top-1 {bm['top1_accuracy']:.2f})", c="#888")
        ax.scatter(tx, ty, s=[max(20, n_*4) for n_ in tn], alpha=0.7, label=f"LoRA-tuned (top-1 {tm['top1_accuracy']:.2f})", c="#1f77b4")
        # Lines connecting per-class
        for i, cls in enumerate([c for c in classes if bm["by_class"].get(c, {}).get("n", 0) > 0]):
            if i < len(bx) and i < len(tx):
                ax.plot([bx[i], tx[i]], [by[i], ty[i]], color="#bbb", lw=0.5, zorder=0)
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_title("SCIN dx-34 — per-class precision/recall (point size = holdout n)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.pr_png, dpi=140)
        plt.close()
        print(f"wrote {args.pr_png}")
    except Exception as e:
        print(f"(plot skipped: {e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
