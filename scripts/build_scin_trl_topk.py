"""Build TRL chat-format JSONL with **multi-condition top-k** targets that
preserve SCIN's `weighted_skin_condition_label` probability distribution.

Each output row is one chat:

  [
    {"role": "system",    "content": [{"type": "text", "text": <task + class list + format>}]},
    {"role": "user",      "content": [{"type": "image", "image": <local path>},
                                      {"type": "text",  "text": <symptoms+body+fst>}]},
    {"role": "assistant", "content": [{"type": "text", "text":
                          "Eczema (0.41); Inflicted skin lesions (0.41); Irritant Contact Dermatitis (0.18)"}]}
  ]

For cases with only one labeled condition, the assistant target is just that
condition with weight 1.00.

Why: training a single-label target string ("Eczema") punishes the model for
emitting valid alternates that the dermatologist labelers picked. Multi-
condition targets preserve the full diff-dx signal and align with the
project's cardinal rule (top-3 with confidence, never single-class).

Inputs (must already exist):
  data/scin/dx34_train.jsonl, dx34_holdout.jsonl  (Fitzpatrick-stratified
                                                   splits from
                                                   scripts/sample_scin_dx.py)
  data/scin/scin_labels.csv                       (full weighted_skin_condition_label dict)

Outputs:
  data/scin/dx34_trl_topk_train.jsonl
  data/scin/dx34_trl_topk_holdout.jsonl
  data/scin/dx34_classes.json     (re-emitted for convenience; same as before)

Run:  PYTHONPATH=. .venv/bin/python scripts/build_scin_trl_topk.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from ast import literal_eval
from pathlib import Path


def build_user_text(row: dict) -> str:
    """Same as scripts/build_scin_trl.py: SCIN metadata only, no condition leakage."""
    bits = []
    bp = row.get("body_parts") or []
    if bp:
        bits.append(f"Body parts: {', '.join(bp)}.")
    sym = row.get("symptoms") or []
    if sym:
        bits.append(f"Reported symptoms: {', '.join(sym)}.")
    dur = (row.get("duration") or "").replace("_", " ").lower().strip()
    if dur and dur != "one day":
        bits.append(f"Duration: {dur}.")
    fst = row.get("fitzpatrick_bucket") or ""
    if fst and fst != "unknown":
        bits.append(f"Fitzpatrick skin type: {fst}.")
    sex = (row.get("sex_at_birth") or "").lower()
    if sex and sex not in ("other_or_unspecified", "prefer_not_to_answer", ""):
        bits.append(f"Sex at birth: {sex}.")
    age = (row.get("age_group") or "").replace("_", " ").lower()
    if age and "unknown" not in age:
        bits.append(f"Age: {age}.")
    if not bits:
        bits.append("(no additional metadata)")
    return " ".join(bits)


def build_system_text(class_list: list[str]) -> str:
    return (
        "You are a dermatology classifier for rural-healthcare triage decision support. "
        "Given a photograph of a skin condition and brief patient context, identify the "
        "most likely conditions. The valid condition list (34 classes):\n"
        + "\n".join(f"  - {c}" for c in class_list)
        + "\n\nOutput format: a semicolon-separated top-3 list with each condition's "
          "confidence in parentheses, sorted by confidence descending. Confidences should "
          "sum to 1.0 (or close). Use ONLY conditions from the list above. Example:\n"
          "  Eczema (0.55); Allergic Contact Dermatitis (0.30); Insect Bite (0.15)\n"
          "Do not diagnose, do not explain — output the formatted list only."
    )


def format_target(weighted: dict[str, float], known_classes: set[str]) -> str:
    """Take SCIN weighted_skin_condition_label and format the assistant target.
    Filter to known classes only (drop conditions our 34-class set doesn't cover).
    Renormalize the kept weights to sum to 1.0 (so the format is internally consistent)."""
    items = [(c, float(w)) for c, w in weighted.items() if c in known_classes]
    items.sort(key=lambda kv: -kv[1])
    if not items:
        return ""  # caller will skip
    total = sum(w for _, w in items)
    if total <= 0:
        return ""
    items = [(c, w / total) for c, w in items]
    # Cap at top-3 to match the project's cardinal rule.
    items = items[:3]
    # Round to 2 decimals; renormalize again to preserve sum=1 after rounding.
    items = [(c, round(w, 2)) for c, w in items]
    s = sum(w for _, w in items)
    if s != 1.0 and items:
        # Push residual into the first (most confident) entry so the string adds to 1.00.
        c0, w0 = items[0]
        items[0] = (c0, round(w0 + (1.0 - s), 2))
    return "; ".join(f"{c} ({w:.2f})" for c, w in items)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-in", default="data/scin/dx34_train.jsonl")
    ap.add_argument("--holdout-in", default="data/scin/dx34_holdout.jsonl")
    ap.add_argument("--scin-labels", default="data/scin/scin_labels.csv")
    ap.add_argument("--train-out", default="data/scin/dx34_trl_topk_train.jsonl")
    ap.add_argument("--holdout-out", default="data/scin/dx34_trl_topk_holdout.jsonl")
    ap.add_argument("--classes-out", default="data/scin/dx34_classes.json")
    args = ap.parse_args()

    train_rows = [json.loads(l) for l in open(args.train_in) if l.strip()]
    hold_rows = [json.loads(l) for l in open(args.holdout_in) if l.strip()]

    classes = sorted({r["condition"] for r in train_rows} | {r["condition"] for r in hold_rows})
    known = set(classes)
    Path(os.path.dirname(args.classes_out) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.classes_out, "w") as f:
        json.dump(classes, f, indent=2, ensure_ascii=False)
    print(f"  classes:    {len(classes)} -> {args.classes_out}")

    # Pull weighted labels from the full SCIN labels CSV
    print("  loading weighted labels from SCIN labels CSV ...")
    weighted_by_id: dict[str, dict] = {}
    for r in csv.DictReader(open(args.scin_labels)):
        s = r.get("weighted_skin_condition_label", "")
        if not s or s in ('""', "{}"):
            continue
        try:
            d = literal_eval(s)
        except Exception:
            continue
        if isinstance(d, dict) and d:
            weighted_by_id[r["case_id"]] = d
    print(f"  weighted labels for {len(weighted_by_id)} cases")

    system_text = build_system_text(classes)

    n_train_dropped = n_train_kept = 0
    n_hold_dropped = n_hold_kept = 0

    def emit(rows: list[dict], out_path: str, label: str) -> tuple[int, int]:
        kept = dropped = 0
        with open(out_path, "w") as f:
            for r in rows:
                # Use the SCIN case_id (without our SCIN- prefix) to look up weighted
                scin_id = r["scin_case_id"]
                weighted = weighted_by_id.get(scin_id) or {r["condition"]: 1.0}
                target = format_target(weighted, known)
                if not target:
                    dropped += 1
                    continue
                if not os.path.exists(r["image_path_local"]):
                    dropped += 1
                    continue
                chat = [
                    {"role": "system", "content": [{"type": "text", "text": system_text}]},
                    {"role": "user", "content": [
                        {"type": "image", "image": r["image_path_local"]},
                        {"type": "text", "text": build_user_text(r)},
                    ]},
                    {"role": "assistant", "content": [{"type": "text", "text": target}]},
                ]
                f.write(json.dumps({
                    "messages": chat,
                    "case_id": r["case_id"],
                    "primary_condition": r["condition"],
                    "weighted_label": weighted,
                    "fitzpatrick_bucket": r.get("fitzpatrick_bucket"),
                    "image_path_local": r["image_path_local"],
                    "target": target,
                }, ensure_ascii=False) + "\n")
                kept += 1
        print(f"  {label} rows: kept={kept} dropped={dropped} -> {out_path}")
        return kept, dropped

    tk, td = emit(train_rows, args.train_out, "train")
    hk, hd = emit(hold_rows, args.holdout_out, "hold")

    # Sample the diff-dx target distribution
    print()
    print("Sample target strings (first 5 train):")
    with open(args.train_out) as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            r = json.loads(line)
            print(f"  {r['case_id']:35s} primary={r['primary_condition']:30s} -> {r['target']!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
