"""Step 2: Convert dx34_{train,holdout}.jsonl into TRL chat-format JSONL.

Each output row is one chat:

  [
    {"role": "system",    "content": [{"type": "text", "text": <task + class list>}]},
    {"role": "user",      "content": [{"type": "image", "image": <local path>},
                                      {"type": "text",  "text": <symptoms+body+fst>}]},
    {"role": "assistant", "content": [{"type": "text", "text": <condition>}]}
  ]

The collate function in scripts/lora_dx_multimodal.py (Step 3) loads each
image via PIL, runs it through the model's processor with the chat template,
and masks pad + image tokens with -100 so the loss is computed only on the
assistant's condition string.

Outputs:
  data/scin/dx34_trl_train.jsonl
  data/scin/dx34_trl_holdout.jsonl
  data/scin/dx34_classes.json     (sorted list of class names — used at eval)

Run:  PYTHONPATH=. .venv/bin/python scripts/build_scin_trl.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def build_user_text(row: dict) -> str:
    """Concise structured patient context. We deliberately do NOT include
    a free-text 'narrative' field that mentions the condition — that would
    leak the answer. Just the SCIN metadata."""
    bits = []
    bp = row.get("body_parts") or []
    if bp:
        bits.append(f"Body parts: {', '.join(bp)}.")
    sym = row.get("symptoms") or []
    if sym:
        bits.append(f"Reported symptoms: {', '.join(sym)}.")
    dur = (row.get("duration") or "").replace("_", " ").lower().strip()
    if dur and dur != "one day":  # 'one day' often means 'unknown' in SCIN encoding
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
        "single most likely condition from this list:\n"
        + "\n".join(f"  - {c}" for c in class_list)
        + "\n\nRespond with the condition name exactly as written in the list above. "
          "Do not diagnose or explain — output the condition name only."
    )


def build_chat(row: dict, system_text: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_text}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": row["image_path_local"]},
                {"type": "text", "text": build_user_text(row)},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": row["condition"]}],
        },
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-in", default="data/scin/dx34_train.jsonl")
    ap.add_argument("--holdout-in", default="data/scin/dx34_holdout.jsonl")
    ap.add_argument("--train-out", default="data/scin/dx34_trl_train.jsonl")
    ap.add_argument("--holdout-out", default="data/scin/dx34_trl_holdout.jsonl")
    ap.add_argument("--classes-out", default="data/scin/dx34_classes.json")
    args = ap.parse_args()

    train_rows = [json.loads(l) for l in open(args.train_in) if l.strip()]
    hold_rows = [json.loads(l) for l in open(args.holdout_in) if l.strip()]

    # The class set is the union of conditions present in train + holdout.
    classes = sorted({r["condition"] for r in train_rows} | {r["condition"] for r in hold_rows})
    Path(os.path.dirname(args.classes_out)).mkdir(parents=True, exist_ok=True)
    with open(args.classes_out, "w") as f:
        json.dump(classes, f, indent=2, ensure_ascii=False)
    print(f"  classes:    {len(classes)} -> {args.classes_out}")

    system_text = build_system_text(classes)

    n_train_dropped = 0
    n_hold_dropped = 0
    with open(args.train_out, "w") as f:
        for r in train_rows:
            if not os.path.exists(r["image_path_local"]):
                n_train_dropped += 1
                continue
            chat = build_chat(r, system_text)
            f.write(json.dumps({
                "messages": chat,
                "case_id": r["case_id"],
                "condition": r["condition"],
                "fitzpatrick_bucket": r.get("fitzpatrick_bucket"),
                "image_path_local": r["image_path_local"],
            }, ensure_ascii=False) + "\n")
    with open(args.holdout_out, "w") as f:
        for r in hold_rows:
            if not os.path.exists(r["image_path_local"]):
                n_hold_dropped += 1
                continue
            chat = build_chat(r, system_text)
            f.write(json.dumps({
                "messages": chat,
                "case_id": r["case_id"],
                "condition": r["condition"],
                "fitzpatrick_bucket": r.get("fitzpatrick_bucket"),
                "image_path_local": r["image_path_local"],
            }, ensure_ascii=False) + "\n")

    print(f"  train rows: {len(train_rows) - n_train_dropped} ({n_train_dropped} dropped — image missing) -> {args.train_out}")
    print(f"  hold rows:  {len(hold_rows) - n_hold_dropped} ({n_hold_dropped} dropped) -> {args.holdout_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
