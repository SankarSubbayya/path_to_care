"""Baseline-only top-k eval for MedGemma 27B-it on the SCIN dx-34 holdout.

Loads MedGemma once, runs zero-shot inference, scores top-1 + top-3 set-match
+ Fitzpatrick stratified — same metrics as scripts/eval_scin_dx_topk.py.
For comparing **medical-domain pretrained** Gemma vs general-pretrained
Gemma 4 31B, both at zero-shot (no fine-tune).

Reuses parse_topk + aggregate from eval_scin_dx_topk.py.

Usage:
  HF_TOKEN=hf_... PYTHONPATH=. .venv/bin/python scripts/eval_medgemma_baseline.py \
      --in data/scin/dx34_trl_topk_holdout.jsonl \
      --classes data/scin/dx34_classes.json \
      --out results/scin_dx34_topk_medgemma.json \
      --limit 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "scripts")
from eval_scin_dx_topk import parse_topk, aggregate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/medgemma-27b-it")
    ap.add_argument("--in", dest="in_path", default="data/scin/dx34_trl_topk_holdout.jsonl")
    ap.add_argument("--classes", default="data/scin/dx34_classes.json")
    ap.add_argument("--out", default="results/scin_dx34_topk_medgemma.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=80)
    args = ap.parse_args()

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    log_path = args.out.replace(".json", ".log")

    if not os.environ.get("HF_TOKEN"):
        print("ERROR: HF_TOKEN env var required (MedGemma is gated).")
        return 1

    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText

    classes = json.load(open(args.classes))
    rows = [json.loads(l) for l in open(args.in_path) if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    print(f"loading {args.model} (~30-90 s; first run also downloads ~54 GB)...")
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(args.model, token=os.environ["HF_TOKEN"])
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda",
        token=os.environ["HF_TOKEN"],
    )
    model.eval()
    print(f"  loaded in {time.time() - t0:.1f}s")

    log_f = open(log_path, "w")
    per_case = []
    t_pass = time.time()

    for i, r in enumerate(rows, 1):
        t0 = time.time()
        try:
            img = Image.open(r["image_path_local"]).convert("RGB")
            msgs = []
            for m in r["messages"]:
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
            topk = parse_topk(raw, classes, k=3)
            pred_top1 = topk[0][0] if topk else None
            pred_set = {c for c, _ in topk}
            truth_primary = r["primary_condition"]
            truth_set = set((r.get("weighted_label") or {}).keys()) | {truth_primary}
            top1_correct = (pred_top1 == truth_primary)
            top3_correct = bool(pred_set & truth_set)
            per_case.append({
                "case_id": r["case_id"], "truth_primary": truth_primary,
                "truth_set": sorted(truth_set), "pred_top1": pred_top1,
                "pred_topk": topk, "top1_correct": top1_correct, "top3_correct": top3_correct,
                "raw": raw, "fitzpatrick_bucket": r.get("fitzpatrick_bucket") or "unknown",
            })
            if i % 5 == 0 or i == len(rows):
                n1 = sum(1 for c in per_case if c["top1_correct"])
                n3 = sum(1 for c in per_case if c["top3_correct"])
                line = (f"[{i:03d}/{len(rows)}] {r['case_id'][:30]:30s} "
                        f"truth={truth_primary[:25]:25s} pred1={(pred_top1 or '?')[:25]:25s} "
                        f"{'V' if top1_correct else 'X'}1 {'V' if top3_correct else 'X'}3 "
                        f"acc1={n1/i:.3f} acc3={n3/i:.3f} ({time.time()-t0:.1f}s/case)")
                print(line); log_f.write(line + "\n"); log_f.flush()
        except Exception as e:
            line = f"[{i:03d}/{len(rows)}] {r['case_id']} ERROR {type(e).__name__}: {e}"
            print(line); log_f.write(line + "\n"); log_f.flush()
            per_case.append({"case_id": r["case_id"], "truth_primary": r["primary_condition"],
                             "truth_set": [r["primary_condition"]], "pred_top1": None,
                             "pred_topk": [], "top1_correct": False, "top3_correct": False,
                             "raw": "", "fitzpatrick_bucket": r.get("fitzpatrick_bucket") or "unknown",
                             "error": str(e)})
    log_f.close()

    output = {
        "_meta": {"model": args.model, "in": args.in_path, "n": len(rows),
                  "wall_seconds": round(time.time() - t_pass, 1)},
        "metrics": aggregate(per_case, classes),
        "per_case": per_case,
    }
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)

    m = output["metrics"]
    print()
    print(f"== {args.model} baseline ==")
    print(f"  top1_accuracy:  {m['top1_accuracy']:.3f} ({m['n_top1_correct']}/{m['n']})")
    print(f"  top3_set_match: {m['top3_set_match']:.3f} ({m['n_top3_correct']}/{m['n']})")
    print(f"  by Fitzpatrick:")
    for k, v in m["by_fitzpatrick"].items():
        print(f"    {k:8s}  n={v['n']:3d}  top1={v['top1_accuracy']:.3f}  top3={v['top3_set_match']:.3f}")
    print(f"  wall: {output['_meta']['wall_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
