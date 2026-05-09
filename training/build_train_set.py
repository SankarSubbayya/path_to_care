"""Build the LoRA SFT training set from baseline traces + ground truth.

The training prompt is exactly what the triage reasoner sees at inference
(`TRIAGE_PROMPT.format(image_top3=..., soap_fields=..., village_context=...)`).
We reuse the zero-shot trace for image_top3 and soap_fields rather than
re-running the upstream MCPs — this keeps "input distribution at train time"
identical to "input distribution at inference time."

The target is a hand-constructed ideal triage answer in the structured
URGENCY/REASONING/RED_FLAGS_NOTED/PATIENT_FRAMING format, using the case's
ground-truth urgency and red_flags.

Train/holdout split: cases with case_id suffix 01-07 → train; 08-10 → eval.
For each urgency level we get 7 train + 3 holdout = 21 train, 9 holdout.

Usage:
  .venv/bin/python -m training.build_train_set \
      --traces results/baseline_metrics.json \
      --train-out data/train.jsonl \
      --holdout-out data/holdout.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from mcp.triage_reasoner.server import TRIAGE_PROMPT
from mcp.village_context.server import practical_urgency_blurb


def make_target(case: dict) -> str:
    """Construct the ideal assistant response for a case."""
    urgency = case["ground_truth_urgency"]
    red_flags = case.get("red_flags", [])
    narrative = case["narrative"]

    # Reasoning: 1-2 sentences anchored to the red flags. Cardinal-rule-clean
    # ("signs suggest", not "you have").
    if urgency == "red":
        if red_flags:
            reasoning = (
                f"Signs suggest a serious condition requiring same-day clinical evaluation. "
                f"Red flags noted: {', '.join(red_flags[:3])}. Delaying care risks rapid deterioration."
            )
        else:
            reasoning = (
                "Signs suggest a serious presentation requiring immediate clinical evaluation. "
                "Multiple concerning features make same-day care necessary."
            )
        framing = (
            "I know the trip costs time and money, but this is the kind of problem that gets much "
            "worse very fast. Going today saves both your health and the much bigger costs of "
            "waiting until it's an emergency."
        )
    elif urgency == "yellow":
        if red_flags:
            reasoning = (
                f"Signs suggest a condition that needs clinical evaluation in 1-2 days. "
                f"Specific concerns: {', '.join(red_flags[:3])}. Not an emergency, but should not be ignored."
            )
        else:
            reasoning = (
                "Signs suggest a condition that should be seen by a clinician within 1-2 days. "
                "Not an emergency tonight, but evaluation will let proper treatment start."
            )
        framing = (
            "This isn't an emergency, but you should see the clinic in the next day or two. "
            "Going to the morning PHC clinic and being back by lunch is a reasonable plan."
        )
    else:  # green
        reasoning = (
            "Signs suggest a self-limited condition that can be monitored at home. "
            "If symptoms worsen — spreading redness, fever, severe pain — return immediately."
        )
        framing = (
            "This looks like something that should resolve on its own with simple home care. "
            "Watch for any of these warning signs and come back if they appear."
        )

    rf_str = ", ".join(red_flags) if red_flags else "none"
    return (
        f"URGENCY: {urgency}\n"
        f"REASONING: {reasoning}\n"
        f"RED_FLAGS_NOTED: {rf_str}\n"
        f"PATIENT_FRAMING: {framing}"
    )


def make_prompt(case: dict, image_top3: list[dict], soap_fields: dict) -> str:
    """The user prompt as seen by the triage reasoner at inference time."""
    image_top3_str = (
        "\n".join(
            f"- {c.get('condition', 'unknown')} (confidence {float(c.get('confidence', 0.0)):.2f})"
            for c in image_top3
        )
        if image_top3
        else "(no candidates produced)"
    )
    soap_str = "\n".join(
        f"{k}: {soap_fields[k]}"
        for k in (
            "chief_complaint", "hpi", "duration", "associated_symptoms",
            "past_medical_history", "medications", "vitals", "exam_findings",
            "red_flags", "patient_concerns",
        )
        if k in soap_fields
    )
    blurb = practical_urgency_blurb(case)
    return TRIAGE_PROMPT.format(
        image_top3=image_top3_str,
        soap_fields=soap_str,
        village_context=blurb,
    )


_HOLDOUT_SUFFIXES = {"08", "09", "10"}


def is_holdout(case_id: str) -> bool:
    # case_ids look like 'P2C-R01' / 'P2C-Y08' / 'P2C-G10' — letter then 2 digits.
    m = re.search(r"(\d{2})$", case_id)
    return bool(m and m.group(1) in _HOLDOUT_SUFFIXES)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True, help="results/baseline_metrics.json from harness.run")
    ap.add_argument("--cases", default="data/cases.jsonl")
    ap.add_argument("--train-out", default="data/train.jsonl")
    ap.add_argument("--holdout-out", default="data/holdout.jsonl")
    args = ap.parse_args()

    cases = {json.loads(line)["case_id"]: json.loads(line) for line in open(args.cases) if line.strip()}
    baseline = json.load(open(args.traces))
    traces = {t["case_id"]: t for t in baseline["traces"] if "case_id" in t and "image_top3" in t}

    Path(os.path.dirname(args.train_out) or ".").mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(args.holdout_out) or ".").mkdir(parents=True, exist_ok=True)

    train_n = 0
    holdout_n = 0
    with open(args.train_out, "w") as ft, open(args.holdout_out, "w") as fh:
        for cid, case in cases.items():
            tr = traces.get(cid)
            if not tr or "image_top3" not in tr or "soap_fields" not in tr:
                # baseline failed for this case; skip — can't construct a prompt
                continue
            prompt = make_prompt(case, tr["image_top3"], tr["soap_fields"])
            target = make_target(case)
            row = {"case_id": cid, "prompt": prompt, "target": target,
                   "ground_truth_urgency": case["ground_truth_urgency"]}
            if is_holdout(cid):
                fh.write(json.dumps(row) + "\n")
                holdout_n += 1
            else:
                ft.write(json.dumps(row) + "\n")
                train_n += 1

    print(f"wrote {train_n} train rows -> {args.train_out}")
    print(f"wrote {holdout_n} holdout rows -> {args.holdout_out}")
    return 0


if __name__ == "__main__":
    import os
    sys.exit(main())
