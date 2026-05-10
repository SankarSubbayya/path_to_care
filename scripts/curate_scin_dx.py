"""Differential-diagnosis curation of SCIN.

Different from scripts/curate_scin.py (which mapped SCIN conditions to R/Y/G
triage urgency). This one keeps the **clinical condition** as the prediction
target and includes the long tail (≥ MIN_CASES per class) so fine-tuning has
somewhere to go beyond what Gemma 4 31B already knows from pretraining.

Output: data/scin/dx_curated.jsonl with one row per case:
  case_id, scin_case_id, condition (target), differentials (full label list),
  fitzpatrick_bucket, body_parts, symptoms (textual list), duration,
  image_paths (relative, served by GCS).

Stats printed at the end:
  - total cases kept
  - per-class counts
  - per-Fitzpatrick counts (per class as well)

Default min_cases = 20 → ~34 classes, ~2400 cases.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from ast import literal_eval
from collections import Counter, defaultdict
from pathlib import Path


def fst_bucket(*labels: str) -> str:
    """Map FST labels to {I-II, III-IV, V-VI, unknown}. Uses the first labeled."""
    for lab in labels:
        if not lab:
            continue
        l = lab.upper().replace(" ", "")
        if l in ("FST1", "FST2"): return "I-II"
        if l in ("FST3", "FST4"): return "III-IV"
        if l in ("FST5", "FST6"): return "V-VI"
    return "unknown"


def case_symptoms(row: dict) -> list[str]:
    flags = []
    map_ = {
        "condition_symptoms_bothersome_appearance": "bothersome appearance",
        "condition_symptoms_bleeding": "bleeding",
        "condition_symptoms_increasing_size": "increasing in size",
        "condition_symptoms_darkening": "darkening",
        "condition_symptoms_itching": "itching",
        "condition_symptoms_burning": "burning",
        "condition_symptoms_pain": "pain",
        "other_symptoms_fever": "fever",
        "other_symptoms_chills": "chills",
        "other_symptoms_fatigue": "fatigue",
        "other_symptoms_joint_pain": "joint pain",
        "other_symptoms_mouth_sores": "mouth sores",
        "other_symptoms_shortness_of_breath": "shortness of breath",
    }
    for k, v in map_.items():
        if row.get(k) == "YES":
            flags.append(v)
    if not flags:
        flags = ["no relevant symptoms"]
    return flags


def case_body_parts(row: dict) -> list[str]:
    parts = []
    map_ = {
        "body_parts_head_or_neck": "head/neck",
        "body_parts_arm": "arm",
        "body_parts_palm": "palm",
        "body_parts_back_of_hand": "back of hand",
        "body_parts_torso_front": "torso (front)",
        "body_parts_torso_back": "torso (back)",
        "body_parts_genitalia_or_groin": "genitalia/groin",
        "body_parts_buttocks": "buttocks",
        "body_parts_leg": "leg",
        "body_parts_foot_top_or_side": "foot top/side",
        "body_parts_foot_sole": "foot sole",
        "body_parts_other": "other",
    }
    for k, v in map_.items():
        if row.get(k) == "YES":
            parts.append(v)
    return parts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="data/scin/scin_cases.csv")
    ap.add_argument("--labels", default="data/scin/scin_labels.csv")
    ap.add_argument("--out", default="data/scin/dx_curated.jsonl")
    ap.add_argument("--min-cases", type=int, default=20)
    args = ap.parse_args()

    cases = list(csv.DictReader(open(args.cases)))
    labels = {r["case_id"]: r for r in csv.DictReader(open(args.labels))}

    # First pass: pick which conditions to keep
    primary_counts: Counter = Counter()
    rows = []
    for c in cases:
        cid = c["case_id"]
        lab = labels.get(cid)
        if not lab:
            continue
        wsc = lab.get("weighted_skin_condition_label", "")
        if not wsc or wsc in ('""', "{}"):
            continue
        try:
            d = literal_eval(wsc)
        except Exception:
            continue
        if not isinstance(d, dict) or not d:
            continue
        primary, conf = max(d.items(), key=lambda kv: kv[1])
        primary_counts[primary] += 1
        rows.append((c, lab, primary, d))

    keep = {cond for cond, n in primary_counts.items() if n >= args.min_cases}

    # Second pass: emit curated rows
    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    by_class: Counter = Counter()
    by_fst: Counter = Counter()
    by_class_fst: dict[str, Counter] = defaultdict(Counter)
    n_total = 0

    with open(args.out, "w") as f:
        for c, lab, primary, weighted in rows:
            if primary not in keep:
                continue
            cid = c["case_id"]
            img = c.get("image_1_path") or ""
            if not img:
                continue
            differentials = sorted(weighted.items(), key=lambda kv: -kv[1])
            differentials = [{"condition": k, "weight": round(v, 2)} for k, v in differentials]
            fst = fst_bucket(
                lab.get("dermatologist_fitzpatrick_skin_type_label_1", ""),
                lab.get("dermatologist_fitzpatrick_skin_type_label_2", ""),
                lab.get("dermatologist_fitzpatrick_skin_type_label_3", ""),
                c.get("fitzpatrick_skin_type", ""),
            )
            row = {
                "case_id": "SCIN-" + cid,
                "scin_case_id": cid,
                "condition": primary,
                "differentials": differentials,
                "fitzpatrick_bucket": fst,
                "body_parts": case_body_parts(c),
                "symptoms": case_symptoms(c),
                "duration": c.get("condition_duration", ""),
                "age_group": c.get("age_group", ""),
                "sex_at_birth": c.get("sex_at_birth", ""),
                "image_remote_path": img,                       # dataset/images/<id>.png on GCS
                "image_path_local": f"data/scin/images/{img.split('/')[-1]}",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            by_class[primary] += 1
            by_fst[fst] += 1
            by_class_fst[primary][fst] += 1
            n_total += 1

    print(f"verdict: PASS")
    print(f"min_cases: {args.min_cases}")
    print(f"total kept: {n_total} ({len(by_class)} classes)")
    print(f"by_fitzpatrick: {dict(by_fst)}")
    print()
    print("classes (n, then Fitzpatrick distribution):")
    for cond, n in by_class.most_common():
        fst_str = ", ".join(f"{k}={v}" for k, v in by_class_fst[cond].most_common())
        print(f"  {n:4d}  {cond:38s}  [{fst_str}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
