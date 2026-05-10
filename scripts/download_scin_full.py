"""Download all images referenced in data/scin/scin_cases.csv (up to 3 per
case, deduped). Skip files already on disk. Parallel. ~10 GB / ~10380 images.

Usage:  PYTHONPATH=. .venv/bin/python scripts/download_scin_full.py
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import os
import sys
import time
import urllib.request
from pathlib import Path

GCS_BASE = "https://storage.googleapis.com/dx-scin-public-data"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="data/scin/scin_cases.csv")
    ap.add_argument("--images-dir", default="data/scin/images")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None,
                    help="for testing: cap number of fetches")
    args = ap.parse_args()

    Path(args.images_dir).mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(args.cases)))
    refs: set[str] = set()
    for r in rows:
        for k in ("image_1_path", "image_2_path", "image_3_path"):
            v = (r.get(k) or "").strip()
            if v:
                refs.add(v)
    print(f"cases: {len(rows)}, unique images: {len(refs)}")

    need: list[tuple[str, str]] = []
    skip = 0
    for remote in refs:
        local = os.path.join(args.images_dir, remote.split("/")[-1])
        if os.path.exists(local) and os.path.getsize(local) > 1000:
            skip += 1
            continue
        need.append((remote, local))
    if args.limit is not None:
        need = need[: args.limit]
    print(f"already on disk: {skip}")
    print(f"to fetch:        {len(need)}")
    if not need:
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
    ok = 0; fail = 0; fails: list[str] = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (local, success, err) in enumerate(ex.map(_fetch, need), 1):
            if success: ok += 1
            else:
                fail += 1
                fails.append(f"{local}: {err}")
            if i % 200 == 0 or i == len(need):
                rate = i / max(time.time() - t0, 0.1)
                print(f"  {i:5d}/{len(need)}  ok={ok}  fail={fail}  ({rate:.0f}/s)")

    elapsed = time.time() - t0
    print()
    print(f"done in {elapsed:.1f}s  ok={ok}  fail={fail}")
    if fails:
        print("first 10 failures:")
        for line in fails[:10]:
            print(f"  {line}")
    total_bytes = sum(os.path.getsize(os.path.join(args.images_dir, p))
                      for p in os.listdir(args.images_dir)
                      if p.endswith(".png"))
    print(f"images dir total: {total_bytes / (1024**3):.2f} GB across {len(os.listdir(args.images_dir))} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
