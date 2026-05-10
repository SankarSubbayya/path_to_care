"""Step 4 / 6: image+text differential-diagnosis eval against vLLM.

For each holdout chat (data/scin/dx34_trl_holdout.jsonl):
  - Load the image as base64 data-URL.
  - Send [system, user(image_url + text)] to vLLM via OpenAI multimodal API.
  - Parse the response to find one of the 34 known classes (case-insensitive
    longest-match, with a tolerant fuzzy fallback).
  - Score top-1 accuracy.

Stratifies by ground-truth class and Fitzpatrick bucket. Saves a confusion
matrix and per-class precision/recall.

Use --adapter to route through a vLLM `--enable-lora` deployment (sets the
served model name to the LoRA name registered on the vLLM container).

Output:
  results/<out_name>.json  (metrics + per-case + confusion + by_fst)
  logs/<out_name>.log      (per-case progress)

Usage:
  PYTHONPATH=. .venv/bin/python scripts/eval_scin_dx.py \
      --in data/scin/dx34_trl_holdout.jsonl \
      --classes data/scin/dx34_classes.json \
      --out results/scin_dx34_baseline.json
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


def _data_url(path: str) -> str:
    with open(path, "rb") as f:
        b = f.read()
    return f"data:image/png;base64,{base64.b64encode(b).decode('ascii')}"


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _match_class(response: str, classes: list[str]) -> str | None:
    """Find the class the model emitted. Strategies, in order:
    1) Exact case-insensitive match of full response
    2) Longest class name that appears as a substring in the response
    3) Fuzzy: response token-set overlap with a class name >= 0.6"""
    r = _normalize(response)
    if not r:
        return None
    norm_classes = [(c, _normalize(c)) for c in classes]
    for c, n in norm_classes:
        if n == r:
            return c
    # longest substring match
    by_len = sorted(norm_classes, key=lambda x: -len(x[1]))
    for c, n in by_len:
        if n in r:
            return c
    # token-set overlap
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/scin/dx34_trl_holdout.jsonl")
    ap.add_argument("--classes", default="data/scin/dx34_classes.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default=os.environ.get("PTC_VLLM_GEMMA4_URL", "http://localhost:8000/v1"))
    ap.add_argument("--api-key",  default=os.environ.get("PTC_VLLM_API_KEY", "ptc-demo-2026-amd"))
    ap.add_argument("--model",    default=os.environ.get("PTC_VLLM_GEMMA4_MODEL_ID", "google/gemma-4-31B-it"))
    ap.add_argument("--adapter",  default=None,
                    help="vLLM LoRA name; routes to that adapter (container must have --enable-lora)")
    ap.add_argument("--limit",    type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()

    if args.adapter:
        args.model = args.adapter  # vLLM LoRA endpoint convention

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    log_path = args.out.replace(".json", ".log")

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    classes = json.load(open(args.classes))
    rows = [json.loads(l) for l in open(args.in_path) if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    print(f"== SCIN dx eval ==")
    print(f"  base_url: {args.base_url}")
    print(f"  model:    {args.model}")
    print(f"  classes:  {len(classes)}")
    print(f"  rows:     {len(rows)}")
    print(f"  out:      {args.out}")

    log_f = open(log_path, "w")
    per_case = []
    t_total = time.time()

    for i, r in enumerate(rows, 1):
        truth = r["condition"]
        fst = r.get("fitzpatrick_bucket") or "unknown"
        # Reconstruct OpenAI multimodal messages from the chat.
        messages = []
        for msg in r["messages"]:
            if msg["role"] == "assistant":
                continue  # don't send the gold answer to the model
            content = []
            for part in msg["content"]:
                if part["type"] == "text":
                    content.append({"type": "text", "text": part["text"]})
                elif part["type"] == "image":
                    content.append({"type": "image_url",
                                    "image_url": {"url": _data_url(part["image"])}})
            messages.append({"role": msg["role"], "content": content})

        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=0,
            )
            raw = (resp.choices[0].message.content or "").strip()
            pred = _match_class(raw, classes)
            correct = (pred == truth)
            line = (f"[{i:03d}/{len(rows)}] {r['case_id']:35s} "
                    f"truth={truth:30s} pred={(pred or '?'):30s} "
                    f"{'✓' if correct else '✗'} "
                    f"fst={fst:8s} t={time.time()-t0:.1f}s "
                    f"raw={raw[:80]!r}")
            print(line); log_f.write(line + "\n"); log_f.flush()
            per_case.append({
                "case_id": r["case_id"], "truth": truth, "pred": pred,
                "correct": correct, "raw": raw, "fitzpatrick_bucket": fst,
            })
        except Exception as e:
            line = f"[{i:03d}/{len(rows)}] {r['case_id']} ERROR {type(e).__name__}: {e}"
            print(line); log_f.write(line + "\n"); log_f.flush()
            per_case.append({"case_id": r["case_id"], "truth": truth,
                             "pred": None, "correct": False, "raw": "",
                             "fitzpatrick_bucket": fst, "error": str(e)})
    log_f.close()

    # Aggregate
    n = len(per_case)
    n_correct = sum(1 for c in per_case if c["correct"])
    confusion = Counter()
    by_class: dict[str, list] = defaultdict(list)
    by_fst: dict[str, list] = defaultdict(list)
    for c in per_case:
        confusion[(c["truth"], c["pred"] or "?")] += 1
        by_class[c["truth"]].append(c)
        by_fst[c["fitzpatrick_bucket"]].append(c)

    per_class_metrics = {}
    for cls in classes:
        cls_cases = by_class.get(cls, [])
        tp = sum(1 for c in cls_cases if c["pred"] == cls)
        fn = sum(1 for c in cls_cases if c["pred"] != cls)
        fp = sum(1 for c in per_case if c["pred"] == cls and c["truth"] != cls)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class_metrics[cls] = {
            "n": len(cls_cases), "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
        }

    fst_metrics = {}
    for bucket, cases_in in by_fst.items():
        nb = len(cases_in)
        nbc = sum(1 for c in cases_in if c["correct"])
        fst_metrics[bucket] = {"n": nb, "top1_accuracy": nbc / max(nb, 1)}

    out = {
        "_meta": {
            "in": args.in_path, "out": args.out, "model": args.model,
            "base_url": args.base_url, "n": n,
            "wall_seconds": round(time.time() - t_total, 1),
        },
        "metrics": {
            "n": n,
            "top1_accuracy": n_correct / max(n, 1),
            "n_correct": n_correct,
            "by_class": per_class_metrics,
            "by_fitzpatrick": fst_metrics,
            "confusion_pairs": [{"truth": t, "pred": p, "count": v}
                                for (t, p), v in confusion.most_common()],
        },
        "per_case": per_case,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print()
    print(f"== eval done ==")
    print(f"  top1_accuracy:  {n_correct/max(n,1):.3f} ({n_correct}/{n})")
    print(f"  by_fitzpatrick: " + ", ".join(f"{k}={v['top1_accuracy']:.3f} (n={v['n']})"
                                            for k, v in fst_metrics.items()))
    print(f"  wall_seconds:   {out['_meta']['wall_seconds']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
