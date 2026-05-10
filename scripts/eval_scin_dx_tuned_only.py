"""Run ONLY the tuned (LoRA-attached) pass in-process. Baseline already done."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# Reuse helpers from the dual-pass script
sys.path.insert(0, "scripts")
from eval_scin_dx_inproc import _match_class, aggregate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/scin/dx34_trl_holdout.jsonl")
    ap.add_argument("--classes", default="data/scin/dx34_classes.json")
    ap.add_argument("--adapter", default="adapters/scin-dx34-gemma4-lora")
    ap.add_argument("--out", default="results/scin_dx34_tuned_inproc.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=24)
    args = ap.parse_args()

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)

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

    print(f"attaching LoRA from {args.adapter} ...")
    t0 = time.time()
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    print(f"  attached in {time.time() - t0:.1f}s")

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
            pred = _match_class(raw, classes)
            correct = (pred == r["condition"])
            per_case.append({
                "case_id": r["case_id"], "truth": r["condition"], "pred": pred,
                "correct": correct, "raw": raw,
                "fitzpatrick_bucket": r.get("fitzpatrick_bucket") or "unknown",
            })
            if i % 5 == 0 or i == len(rows):
                n_correct = sum(1 for c in per_case if c["correct"])
                print(f"  [tuned] [{i:03d}/{len(rows)}] {r['case_id'][:30]:30s} "
                      f"truth={r['condition'][:25]:25s} pred={(pred or '?')[:25]:25s} "
                      f"{'V' if correct else 'X'} acc={n_correct}/{i}={n_correct/i:.3f} "
                      f"({time.time()-t0:.1f}s/case, {(time.time()-t_pass)/60:.1f} min wall)")
        except Exception as e:
            per_case.append({"case_id": r["case_id"], "truth": r["condition"], "pred": None,
                             "correct": False, "raw": "",
                             "fitzpatrick_bucket": r.get("fitzpatrick_bucket") or "unknown",
                             "error": str(e)})

    output = {
        "_meta": {"model": "google/gemma-4-31B-it + scin-dx34-gemma4-lora",
                  "in": args.in_path, "n": len(rows),
                  "wall_seconds": round(time.time() - t_pass, 1)},
        "metrics": aggregate(per_case, classes),
        "per_case": per_case,
    }
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  wrote {args.out}  top1={output['metrics']['top1_accuracy']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
