"""In-process eval for SCIN dx-34: runs baseline (no adapter) AND tuned
(with the LoRA adapter attached via peft) on the SAME holdout subset, in
one process so we only load the 31B model once.

Why in-process: vLLM ROCm 0.20.1 silently doesn't apply Gemma 4 LoRA at
serving time, even when --enable-lora --lora-modules are configured and
the adapter loads cleanly. peft+transformers in-process applies it
correctly (verified live; outputs differ from base on 2/3 sample cases).

Outputs:
  results/scin_dx34_baseline_inproc.json
  results/scin_dx34_tuned_inproc.json   (same schema as scripts/eval_scin_dx.py)

Usage:
  PYTHONPATH=. .venv/bin/python scripts/eval_scin_dx_inproc.py \
      --in data/scin/dx34_trl_holdout.jsonl \
      --classes data/scin/dx34_classes.json \
      --adapter adapters/scin-dx34-gemma4-lora \
      --limit 100
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


_URGENCY_RE = re.compile(r"")  # placeholder; not used
_RESPONSE_TRIM = re.compile(r"^[\s\W]+|[\s\W]+$")


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _match_class(response: str, classes: list[str]) -> str | None:
    r = _normalize(response)
    if not r:
        return None
    norm_classes = [(c, _normalize(c)) for c in classes]
    for c, n in norm_classes:
        if n == r:
            return c
    by_len = sorted(norm_classes, key=lambda x: -len(x[1]))
    for c, n in by_len:
        if n in r:
            return c
    r_toks = set(re.findall(r"[a-z0-9]+", r))
    if not r_toks:
        return None
    best = (0.0, None)
    for c, n in norm_classes:
        toks = set(re.findall(r"[a-z0-9]+", n))
        if not toks:
            continue
        overlap = len(r_toks & toks) / len(toks)
        if overlap > best[0]:
            best = (overlap, c)
    return best[1] if best[0] >= 0.6 else None


def aggregate(per_case: list[dict], classes: list[str]) -> dict:
    n = len(per_case)
    n_correct = sum(1 for c in per_case if c["correct"])
    confusion = Counter()
    by_class = defaultdict(list)
    by_fst = defaultdict(list)
    for c in per_case:
        confusion[(c["truth"], c["pred"] or "?")] += 1
        by_class[c["truth"]].append(c)
        by_fst[c["fitzpatrick_bucket"]].append(c)

    pcm = {}
    for cls in classes:
        cls_cases = by_class.get(cls, [])
        tp = sum(1 for c in cls_cases if c["pred"] == cls)
        fn = sum(1 for c in cls_cases if c["pred"] != cls)
        fp = sum(1 for c in per_case if c["pred"] == cls and c["truth"] != cls)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        pcm[cls] = {"n": len(cls_cases), "tp": tp, "fp": fp, "fn": fn,
                    "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)}

    fst_metrics = {}
    for bucket, cases_in in by_fst.items():
        nb = len(cases_in)
        nbc = sum(1 for c in cases_in if c["correct"])
        fst_metrics[bucket] = {"n": nb, "top1_accuracy": nbc / max(nb, 1)}

    return {
        "n": n,
        "top1_accuracy": n_correct / max(n, 1),
        "n_correct": n_correct,
        "by_class": pcm,
        "by_fitzpatrick": fst_metrics,
        "confusion_pairs": [{"truth": t, "pred": p, "count": v}
                            for (t, p), v in confusion.most_common()],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/scin/dx34_trl_holdout.jsonl")
    ap.add_argument("--classes", default="data/scin/dx34_classes.json")
    ap.add_argument("--adapter", default="adapters/scin-dx34-gemma4-lora")
    ap.add_argument("--baseline-out", default="results/scin_dx34_baseline_inproc.json")
    ap.add_argument("--tuned-out", default="results/scin_dx34_tuned_inproc.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=24)
    args = ap.parse_args()

    Path(os.path.dirname(args.baseline_out) or ".").mkdir(parents=True, exist_ok=True)

    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import PeftModel

    classes = json.load(open(args.classes))
    rows = [json.loads(l) for l in open(args.in_path) if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"loading processor + base model (~30s)...")
    t0 = time.time()
    proc = AutoProcessor.from_pretrained("google/gemma-4-31B-it")
    model = AutoModelForImageTextToText.from_pretrained(
        "google/gemma-4-31B-it", dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    print(f"  loaded in {time.time() - t0:.1f}s")

    def gen(case: dict) -> tuple[str, str]:
        img = Image.open(case["image_path_local"]).convert("RGB")
        msgs = []
        for m in case["messages"]:
            if m["role"] == "assistant":
                continue
            cont = []
            for p in m["content"]:
                if p["type"] == "text":
                    cont.append({"type": "text", "text": p["text"]})
                elif p["type"] == "image":
                    cont.append({"type": "image", "image": img})
            msgs.append({"role": m["role"], "content": cont})
        text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        inputs = proc(text=[text], images=[[img]], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False)
        raw = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        return raw

    def run_pass(label: str) -> dict:
        per_case = []
        t_pass = time.time()
        for i, r in enumerate(rows, 1):
            t0 = time.time()
            try:
                raw = gen(r)
                pred = _match_class(raw, classes)
                correct = (pred == r["condition"])
                per_case.append({
                    "case_id": r["case_id"], "truth": r["condition"], "pred": pred,
                    "correct": correct, "raw": raw,
                    "fitzpatrick_bucket": r.get("fitzpatrick_bucket") or "unknown",
                })
                if i % 5 == 0 or i == len(rows):
                    n_correct = sum(1 for c in per_case if c["correct"])
                    print(f"  [{label}] [{i:03d}/{len(rows)}] {r['case_id'][:30]:30s} "
                          f"truth={r['condition'][:25]:25s} pred={(pred or '?')[:25]:25s} "
                          f"{'✓' if correct else '✗'} acc={n_correct}/{i}={n_correct/i:.3f} "
                          f"({time.time()-t0:.1f}s/case, {(time.time()-t_pass)/60:.1f} min wall)")
            except Exception as e:
                per_case.append({
                    "case_id": r["case_id"], "truth": r["condition"], "pred": None,
                    "correct": False, "raw": "", "fitzpatrick_bucket": r.get("fitzpatrick_bucket") or "unknown",
                    "error": str(e),
                })
                print(f"  [{label}] {r['case_id']} ERROR {type(e).__name__}: {e}")
        return {
            "_meta": {"model": "google/gemma-4-31B-it" + (f" + {label} adapter" if label == "tuned" else ""),
                       "in": args.in_path, "n": len(rows),
                       "wall_seconds": round(time.time() - t_pass, 1)},
            "metrics": aggregate(per_case, classes),
            "per_case": per_case,
        }

    print()
    print("=== BASELINE pass (no adapter) ===")
    base_out = run_pass("base")
    with open(args.baseline_out, "w") as f:
        json.dump(base_out, f, indent=2)
    print(f"  wrote {args.baseline_out}  top1={base_out['metrics']['top1_accuracy']:.3f}")

    print()
    print("=== Attaching LoRA adapter ===")
    t0 = time.time()
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    print(f"  attached in {time.time() - t0:.1f}s")

    print()
    print("=== TUNED pass (with adapter) ===")
    tuned_out = run_pass("tuned")
    with open(args.tuned_out, "w") as f:
        json.dump(tuned_out, f, indent=2)
    print(f"  wrote {args.tuned_out}  top1={tuned_out['metrics']['top1_accuracy']:.3f}")

    delta_pp = 100 * (tuned_out["metrics"]["top1_accuracy"] - base_out["metrics"]["top1_accuracy"])
    print()
    print(f"DELTA: {delta_pp:+.1f} pp (baseline={base_out['metrics']['top1_accuracy']:.3f}, "
          f"tuned={tuned_out['metrics']['top1_accuracy']:.3f}, n={tuned_out['_meta']['n']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
