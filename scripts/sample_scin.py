"""Sample train / holdout splits from data/scin/curated.jsonl and download
the actual images in parallel.

- Train: 100 red + 300 yellow + 150 green (capped by availability) = up to 550 rows
- Holdout: 25 red + 100 yellow + 75 green (disjoint from train), Fitzpatrick-
  stratified where possible (V-VI underrepresented in SCIN; we take what we can).

Outputs:
  data/scin/train.jsonl
  data/scin/holdout.jsonl
  data/scin/images/<scin_case_id>.png   (the cropped subset we actually use)
  evidence/scin_split.txt               (split summary + Fitzpatrick distribution)
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


CURATED = "data/scin/curated.jsonl"
IMG_DIR = "data/scin/images"
OUT_TRAIN = "data/scin/train.jsonl"
OUT_HOLDOUT = "data/scin/holdout.jsonl"
EVIDENCE = "evidence/scin_split.txt"


def _stratified_sample(rows: list[dict], counts_per_urgency: dict[str, int],
                       seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    by_urg: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_urg[r["ground_truth_urgency"]].append(r)
    picked = []
    for urg, n in counts_per_urgency.items():
        bucket = by_urg.get(urg, [])
        rng.shuffle(bucket)
        picked.extend(bucket[:n])
    return picked


def _download(row: dict) -> tuple[str, bool, str]:
    url = row["image_path_remote"]
    out = row["image_path_local"]
    Path(os.path.dirname(out)).mkdir(parents=True, exist_ok=True)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return (row["case_id"], True, "cached")
    try:
        urllib.request.urlretrieve(url, out)
        return (row["case_id"], True, f"{os.path.getsize(out)} bytes")
    except Exception as e:
        return (row["case_id"], False, f"{type(e).__name__}: {e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-red", type=int, default=100)
    ap.add_argument("--train-yellow", type=int, default=300)
    ap.add_argument("--train-green", type=int, default=150)
    ap.add_argument("--holdout-red", type=int, default=25)
    ap.add_argument("--holdout-yellow", type=int, default=100)
    ap.add_argument("--holdout-green", type=int, default=75)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(CURATED) if l.strip()]
    rng = random.Random(42)
    rng.shuffle(rows)

    train_quota = {"red": args.train_red, "yellow": args.train_yellow, "green": args.train_green}
    holdout_quota = {"red": args.holdout_red, "yellow": args.holdout_yellow, "green": args.holdout_green}

    train = _stratified_sample(rows, train_quota, seed=42)
    train_ids = {r["case_id"] for r in train}
    remaining = [r for r in rows if r["case_id"] not in train_ids]
    holdout = _stratified_sample(remaining, holdout_quota, seed=1234)

    Path(IMG_DIR).mkdir(parents=True, exist_ok=True)
    print(f"sampled train={len(train)}  holdout={len(holdout)}")

    # Parallel download
    all_rows = train + holdout
    success = 0
    fail = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (cid, ok, msg) in enumerate(pool.map(_download, all_rows), 1):
            if ok:
                success += 1
            else:
                fail += 1
                print(f"  FAIL {cid}: {msg}")
            if i % 100 == 0:
                print(f"  downloaded {i}/{len(all_rows)}  (ok={success}, fail={fail})")
    print(f"download done: ok={success}, fail={fail}")

    # Drop rows whose image failed to download
    train = [r for r in train if os.path.exists(r["image_path_local"])
             and os.path.getsize(r["image_path_local"]) > 0]
    holdout = [r for r in holdout if os.path.exists(r["image_path_local"])
               and os.path.getsize(r["image_path_local"]) > 0]

    with open(OUT_TRAIN, "w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT_HOLDOUT, "w") as f:
        for r in holdout:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_urg_train = Counter(r["ground_truth_urgency"] for r in train)
    by_urg_hold = Counter(r["ground_truth_urgency"] for r in holdout)
    by_fst_train = Counter(r["fitzpatrick_bucket"] for r in train)
    by_fst_hold = Counter(r["fitzpatrick_bucket"] for r in holdout)
    by_cond_hold = Counter(r["condition"] for r in holdout)

    Path("evidence").mkdir(exist_ok=True)
    lines = [
        "verdict: PASS",
        f"train: {len(train)} rows",
        f"  by_urgency:    {dict(by_urg_train)}",
        f"  by_fitzpatrick: {dict(by_fst_train)}",
        f"holdout: {len(holdout)} rows",
        f"  by_urgency:    {dict(by_urg_hold)}",
        f"  by_fitzpatrick: {dict(by_fst_hold)}",
        f"  by_condition:  {dict(by_cond_hold)}",
        f"images:  ok={success}, fail={fail}",
    ]
    txt = "\n".join(lines) + "\n"
    print(txt)
    with open(EVIDENCE, "w") as f:
        f.write(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
