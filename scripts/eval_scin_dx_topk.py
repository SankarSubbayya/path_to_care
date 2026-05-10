"""In-process top-k eval for SCIN dx-34. Runs baseline (no adapter) AND tuned
(with LoRA) on the same holdout subset; one model load.

Two metrics, both stricter than top-1-on-primary:

  - **top-3 set match**: case scores 1 if any of the model's predicted classes
    is in `dermatologist_skin_condition_on_label_name` (the dermatologist-
    labeled set). Matches the SCIN paper's eval protocol.
  - **top-1 primary match**: argmax of the model's predicted distribution
    matches `primary_condition`. Same as our old metric, kept for comparability.

Stratified by Fitzpatrick I-II / III-IV / V-VI / unknown. Per-class precision/
recall against the primary.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/eval_scin_dx_topk.py \
      --in data/scin/dx34_trl_topk_holdout.jsonl \
      --classes data/scin/dx34_classes.json \
      --adapter adapters/scin-dx34-gemma4-topk-lora \
      --baseline-out results/scin_dx34_topk_baseline.json \
      --tuned-out    results/scin_dx34_topk_tuned.json \
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


# ----- response parsing -----------------------------------------------------

_PRED_LINE_RE = re.compile(
    r"([A-Za-z][A-Za-z0-9 ,/\-]+?)\s*\(\s*([0-9.]+)\s*\)"
)


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def parse_topk(response: str, classes: list[str], k: int = 3) -> list[tuple[str, float]]:
    """Parse 'Eczema (0.5); Insect Bite (0.3); ...' from the model output. Return
    [(class_name, weight)] up to k, mapped to known classes via case-insensitive
    longest-substring match. Falls back to single-class match if the format is
    bare ('Eczema') with no weight."""
    norm_classes = [(c, _normalize(c)) for c in classes]
    out: list[tuple[str, float]] = []
    seen: set[str] = set()
    for piece in _PRED_LINE_RE.finditer(response):
        cand = _normalize(piece.group(1))
        try:
            w = float(piece.group(2))
        except Exception:
            continue
        # Match cand to a known class
        match = None
        # exact match
        for c, n in norm_classes:
            if n == cand:
                match = c
                break
        if match is None:
            # longest substring containment
            best = sorted(norm_classes, key=lambda x: -len(x[1]))
            for c, n in best:
                if n in cand or cand in n:
                    match = c
                    break
        if match and match not in seen:
            seen.add(match)
            out.append((match, w))
            if len(out) >= k:
                break
    if out:
        return out

    # Fallback: bare class name with no weight
    r = _normalize(response)
    if not r:
        return []
    by_len = sorted(norm_classes, key=lambda x: -len(x[1]))
    for c, n in by_len:
        if n == r or n in r:
            return [(c, 1.0)]
    return []


# ----- aggregate ------------------------------------------------------------

def aggregate(per_case: list[dict], classes: list[str]) -> dict:
    n = len(per_case)
    n_top1 = sum(1 for c in per_case if c["top1_correct"])
    n_top3 = sum(1 for c in per_case if c["top3_correct"])

    by_class = defaultdict(list)
    by_fst = defaultdict(list)
    confusion = Counter()
    for c in per_case:
        by_class[c["truth_primary"]].append(c)
        by_fst[c["fitzpatrick_bucket"]].append(c)
        confusion[(c["truth_primary"], c["pred_top1"] or "?")] += 1

    pcm = {}
    for cls in classes:
        cls_cases = by_class.get(cls, [])
        tp = sum(1 for c in cls_cases if c["pred_top1"] == cls)
        fn = sum(1 for c in cls_cases if c["pred_top1"] != cls)
        fp = sum(1 for c in per_case if c["pred_top1"] == cls and c["truth_primary"] != cls)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        # also top-3 recall for this class: how often did we include it in top-3?
        top3_rec_n = sum(1 for c in cls_cases
                         if cls in {p[0] for p in c.get("pred_topk", [])})
        top3_rec = top3_rec_n / max(len(cls_cases), 1)
        pcm[cls] = {
            "n": len(cls_cases), "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
            "top3_recall": round(top3_rec, 3),
        }

    fst_metrics = {}
    for bucket, cases_in in by_fst.items():
        nb = len(cases_in)
        nbc1 = sum(1 for c in cases_in if c["top1_correct"])
        nbc3 = sum(1 for c in cases_in if c["top3_correct"])
        fst_metrics[bucket] = {
            "n": nb,
            "top1_accuracy": nbc1 / max(nb, 1),
            "top3_set_match": nbc3 / max(nb, 1),
        }

    return {
        "n": n,
        "top1_accuracy": n_top1 / max(n, 1),
        "top3_set_match": n_top3 / max(n, 1),
        "n_top1_correct": n_top1,
        "n_top3_correct": n_top3,
        "by_class": pcm,
        "by_fitzpatrick": fst_metrics,
        "confusion_pairs": [{"truth": t, "pred": p, "count": v}
                            for (t, p), v in confusion.most_common()],
    }


# ----- main -----------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/scin/dx34_trl_topk_holdout.jsonl")
    ap.add_argument("--classes", default="data/scin/dx34_classes.json")
    ap.add_argument("--adapter", default="adapters/scin-dx34-gemma4-topk-lora")
    ap.add_argument("--baseline-out", default="results/scin_dx34_topk_baseline.json")
    ap.add_argument("--tuned-out", default="results/scin_dx34_topk_tuned.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=80)
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
    print(f"  base loaded in {time.time() - t0:.1f}s")

    def gen(case: dict) -> str:
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
        return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    def run_pass(label: str) -> dict:
        per_case = []
        t_pass = time.time()
        for i, r in enumerate(rows, 1):
            t0 = time.time()
            try:
                raw = gen(r)
                topk = parse_topk(raw, classes, k=3)
                pred_top1 = topk[0][0] if topk else None
                pred_set = {c for c, _ in topk}
                truth_primary = r["primary_condition"]
                # top-3 match: any predicted class is in the dermatologist set
                truth_set = set(r.get("weighted_label", {}).keys()) | {truth_primary}
                top1_correct = (pred_top1 == truth_primary)
                top3_correct = bool(pred_set & truth_set)
                per_case.append({
                    "case_id": r["case_id"],
                    "truth_primary": truth_primary,
                    "truth_set": sorted(truth_set),
                    "pred_top1": pred_top1,
                    "pred_topk": topk,
                    "top1_correct": top1_correct,
                    "top3_correct": top3_correct,
                    "raw": raw,
                    "fitzpatrick_bucket": r.get("fitzpatrick_bucket") or "unknown",
                })
                if i % 5 == 0 or i == len(rows):
                    n1 = sum(1 for c in per_case if c["top1_correct"])
                    n3 = sum(1 for c in per_case if c["top3_correct"])
                    print(f"  [{label}] [{i:03d}/{len(rows)}] {r['case_id'][:30]:30s} "
                          f"truth={truth_primary[:25]:25s} pred1={(pred_top1 or '?')[:25]:25s} "
                          f"{'V' if top1_correct else 'X'}1 {'V' if top3_correct else 'X'}3 "
                          f"acc1={n1/i:.3f} acc3={n3/i:.3f} ({time.time()-t0:.1f}s/case)")
            except Exception as e:
                per_case.append({"case_id": r["case_id"], "truth_primary": r["primary_condition"],
                                 "truth_set": [r["primary_condition"]], "pred_top1": None,
                                 "pred_topk": [], "top1_correct": False, "top3_correct": False,
                                 "raw": "", "fitzpatrick_bucket": r.get("fitzpatrick_bucket") or "unknown",
                                 "error": str(e)})
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
    bm = base_out["metrics"]
    print(f"  wrote {args.baseline_out}")
    print(f"  baseline.top1={bm['top1_accuracy']:.3f}  top3_set_match={bm['top3_set_match']:.3f}")

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
    tm = tuned_out["metrics"]
    print(f"  wrote {args.tuned_out}")
    print(f"  tuned.top1={tm['top1_accuracy']:.3f}  top3_set_match={tm['top3_set_match']:.3f}")

    print()
    print(f"DELTA top1:  {100*(tm['top1_accuracy']-bm['top1_accuracy']):+.1f} pp  "
          f"({bm['top1_accuracy']:.3f} -> {tm['top1_accuracy']:.3f})")
    print(f"DELTA top3:  {100*(tm['top3_set_match']-bm['top3_set_match']):+.1f} pp  "
          f"({bm['top3_set_match']:.3f} -> {tm['top3_set_match']:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
