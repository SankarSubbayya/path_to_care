"""Step 1: Sample train/holdout splits from data/scin/dx_coarse_curated.jsonl
(16 coarse categories), stratified by class AND Fitzpatrick bucket, and
download any missing images from the public GCS bucket.

Decisions:
  - Keep only classes with >= MIN_CASES (default 30) → 12 classes.
  - Per class: 80% train / 20% holdout, capped at TRAIN_CAP / HOLDOUT_CAP.
  - Within each (class, split) bucket, draw cases stratified by Fitzpatrick
    (I-II / III-IV / V-VI / unknown) when feasible.
  - Deterministic via seed=42.
  - Images already on disk at data/scin/images/<remote_basename> are kept;
    the rest are pulled in parallel from
    https://storage.googleapis.com/dx-scin-public-data/<image_remote_path>.

Outputs:
  data/scin/dx_train.jsonl   (one row per sampled training case)
  data/scin/dx_holdout.jsonl (one row per sampled holdout case)
  data/scin/images/*.png     (only the ones we actually need)

Run:  PYTHONPATH=. .venv/bin/python scripts/sample_scin_dx.py
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

GCS_BASE = "https://storage.googleapis.com/dx-scin-public-data"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/scin/dx_coarse_curated.jsonl")
    ap.add_argument("--train-out", default="data/scin/dx_train.jsonl")
    ap.add_argument("--holdout-out", default="data/scin/dx_holdout.jsonl")
    ap.add_argument("--images-dir", default="data/scin/images")
    ap.add_argument("--min-cases", type=int, default=30,
                    help="drop classes with fewer than this many cases")
    ap.add_argument("--train-cap", type=int, default=80,
                    help="cap per-class training samples")
    ap.add_argument("--holdout-cap", type=int, default=25,
                    help="cap per-class holdout samples")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--class-field", default="coarse_category",
                    help="JSONL field that holds the class label "
                         "('coarse_category' for dx_coarse_curated, "
                         "'condition' for dx_curated)")
    ap.add_argument("--limit-download", type=int, default=None,
                    help="for smoke tests: cap number of new image downloads")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    rows = [json.loads(l) for l in open(args.in_path) if l.strip()]
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[r[args.class_field]].append(r)

    keep_classes = [c for c, lst in by_class.items() if len(lst) >= args.min_cases]
    keep_classes.sort()
    print(f"  total classes in curated:    {len(by_class)}")
    print(f"  classes with >= {args.min_cases} cases: {len(keep_classes)}")

    # Per-class stratified sampling
    train_rows, holdout_rows = [], []
    train_dist: Counter = Counter()
    holdout_dist: Counter = Counter()
    train_fst_per_class: dict[str, Counter] = defaultdict(Counter)
    holdout_fst_per_class: dict[str, Counter] = defaultdict(Counter)

    for cls in keep_classes:
        lst = by_class[cls]
        # bucket by Fitzpatrick
        by_fst: dict[str, list[dict]] = defaultdict(list)
        for r in lst:
            by_fst[r.get("fitzpatrick_bucket") or "unknown"].append(r)
        for bucket in by_fst.values():
            rng.shuffle(bucket)

        # decide per-class train/holdout sizes
        n = len(lst)
        n_train = min(args.train_cap, int(round(n * args.train_frac)))
        n_holdout = min(args.holdout_cap, n - n_train)

        # round-robin draw across Fitzpatrick buckets
        fst_order = ["I-II", "III-IV", "V-VI", "unknown"]
        cls_train, cls_holdout = [], []
        idx = {b: 0 for b in fst_order}

        def _next(bucket: str) -> dict | None:
            i = idx[bucket]
            arr = by_fst.get(bucket, [])
            if i < len(arr):
                idx[bucket] += 1
                return arr[i]
            return None

        # First fill train
        rr = 0
        while len(cls_train) < n_train:
            placed = False
            for _ in range(len(fst_order)):
                b = fst_order[rr % len(fst_order)]; rr += 1
                r = _next(b)
                if r is not None:
                    cls_train.append((b, r)); placed = True
                    if len(cls_train) >= n_train: break
            if not placed:
                break

        # Then fill holdout
        while len(cls_holdout) < n_holdout:
            placed = False
            for _ in range(len(fst_order)):
                b = fst_order[rr % len(fst_order)]; rr += 1
                r = _next(b)
                if r is not None:
                    cls_holdout.append((b, r)); placed = True
                    if len(cls_holdout) >= n_holdout: break
            if not placed:
                break

        for b, r in cls_train:
            train_rows.append(r)
            train_dist[cls] += 1
            train_fst_per_class[cls][b] += 1
        for b, r in cls_holdout:
            holdout_rows.append(r)
            holdout_dist[cls] += 1
            holdout_fst_per_class[cls][b] += 1

    Path(os.path.dirname(args.train_out)).mkdir(parents=True, exist_ok=True)
    Path(args.images_dir).mkdir(parents=True, exist_ok=True)

    # Write splits
    with open(args.train_out, "w") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.holdout_out, "w") as f:
        for r in holdout_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Compute which images need downloading
    need: list[tuple[str, str]] = []     # (remote_path, local_path)
    skip = 0
    for r in train_rows + holdout_rows:
        local = r["image_path_local"]
        if os.path.exists(local) and os.path.getsize(local) > 1000:
            skip += 1
            continue
        need.append((r["image_remote_path"], local))
    if args.limit_download is not None:
        need = need[: args.limit_download]

    print()
    print(f"== samples ==")
    print(f"  train: {len(train_rows)} rows across {len(train_dist)} classes")
    print(f"  hold : {len(holdout_rows)} rows across {len(holdout_dist)} classes")
    print()
    print(f'  {"coarse_category":35s}  train(I-II/III-IV/V-VI/unk)   hold(I-II/III-IV/V-VI/unk)')
    print("  " + "-" * 110)
    for cls in keep_classes:
        t = train_fst_per_class[cls]
        h = holdout_fst_per_class[cls]
        ts = f'{t.get("I-II",0):2d}/{t.get("III-IV",0):2d}/{t.get("V-VI",0):2d}/{t.get("unknown",0):2d}'
        hs = f'{h.get("I-II",0):2d}/{h.get("III-IV",0):2d}/{h.get("V-VI",0):2d}/{h.get("unknown",0):2d}'
        print(f"  {cls:35s}  {train_dist[cls]:3d} ({ts})        {holdout_dist[cls]:3d} ({hs})")
    print()
    print(f"== image download ==")
    print(f"  already on disk: {skip}")
    print(f"  to download:     {len(need)}")

    if not need:
        print("  nothing to fetch.")
        return 0

    def _fetch(item):
        remote, local = item
        url = f"{GCS_BASE}/{remote}"
        try:
            urllib.request.urlretrieve(url, local)
            return (local, True, "")
        except Exception as e:
            return (local, False, str(e))

    t0 = time.time()
    ok = 0; fail = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (local, success, err) in enumerate(ex.map(_fetch, need), 1):
            if success: ok += 1
            else: fail += 1
            if i % 50 == 0 or i == len(need):
                print(f"    downloaded {i}/{len(need)}  ok={ok}  fail={fail}  ({time.time()-t0:.1f}s)")

    print(f"  done: ok={ok} fail={fail} in {time.time()-t0:.1f}s")
    print(f"  wrote {args.train_out} ({len(train_rows)}) and {args.holdout_out} ({len(holdout_rows)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
