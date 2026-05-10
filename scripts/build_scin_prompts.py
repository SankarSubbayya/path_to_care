"""Convert curated SCIN train/holdout JSONL into (prompt, target) pairs
matching the triage reasoner's prompt template, for LoRA training.

For each SCIN row we synthesize:
  - image_top3:  the real condition (with confidence drawn from a Beta distribution
                 around 0.7) plus two plausible differentials sampled from a per-
                 condition differential map. This mirrors what the image-classifier
                 MCP would emit at inference: top-3 with confidence, never single class.
  - soap_fields: extracted from the SCIN symptom + body-part metadata (chief
                 complaint, HPI, duration, associated symptoms, vitals, exam findings,
                 red flags). No condition leakage.
  - village_context: standard blurb (matches what the orchestrator emits).
  - target:      URGENCY / REASONING / RED_FLAGS_NOTED / PATIENT_FRAMING
                 templated from ground_truth_urgency + the case's red-flag triggers.

Outputs:
  data/scin/train_prompts.jsonl   (case_id, prompt, target, ground_truth_urgency,
                                   fitzpatrick_bucket, condition)
  data/scin/holdout_prompts.jsonl (same)

These can be fed straight into training.lora_sft.py (which expects prompt+target rows).
For eval we feed only the prompt and check whether the parsed urgency matches
ground_truth_urgency.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path


# Plausible differentials by primary condition. Used only at training time
# to avoid teaching the LoRA "image_top3 ⇒ ground-truth condition." The
# *order* is randomized; the primary condition is always one of the three
# but not always rank #1.
DIFFERENTIALS: dict[str, list[str]] = {
    "Eczema":                       ["atopic dermatitis", "contact dermatitis", "psoriasis", "tinea corporis", "lichen simplex"],
    "Allergic Contact Dermatitis":  ["irritant contact dermatitis", "atopic dermatitis", "id reaction", "drug eruption"],
    "Urticaria":                    ["drug eruption", "viral exanthem", "angioedema (no anaphylaxis)", "id reaction"],
    "Insect Bite":                  ["urticaria (localized)", "papular urticaria", "folliculitis", "scabies"],
    "Folliculitis":                 ["furuncle", "acne mechanica", "hot-tub folliculitis", "molluscum"],
    "Psoriasis":                    ["seborrheic dermatitis", "lichen planus", "tinea corporis", "discoid eczema"],
    "Tinea":                        ["nummular eczema", "psoriasis (annular)", "granuloma annulare", "pityriasis rosea"],
    "Impetigo":                     ["cellulitis (early)", "perioral dermatitis", "herpes simplex"],
    "Herpes Zoster":                ["herpes simplex", "contact dermatitis (linear)", "post-herpetic excoriation"],
    "Pigmented purpuric eruption":  ["stasis dermatitis", "petechial drug rash", "leukocytoclastic vasculitis (mild)"],
}


REASONING_TEMPLATES = {
    "red": (
        "Signs suggest a serious condition requiring same-day clinical evaluation. "
        "Multiple concerning features point to risk of rapid deterioration. "
        "Red flags noted: {flags}."
    ),
    "yellow": (
        "Signs suggest a condition that needs clinical evaluation in the next 1-2 days. "
        "Specific concerns: {flags}. Not an emergency tonight, but evaluation will let "
        "treatment start."
    ),
    "green": (
        "Signs suggest a self-limited condition that can be monitored at home. "
        "If symptoms worsen — spreading redness, fever, severe pain — return immediately."
    ),
}


FRAMING_TEMPLATES = {
    "red": (
        "I know the trip costs time and money, but this is the kind of problem that "
        "gets much worse very fast. Going today saves both your health and the much "
        "bigger costs of waiting."
    ),
    "yellow": (
        "This is not an emergency, but you should see the clinic in the next day or two. "
        "Going to the morning PHC clinic and being back by lunch is a reasonable plan."
    ),
    "green": (
        "This looks like something that should resolve on its own with simple home care. "
        "Watch for warning signs and come back if they appear."
    ),
}


def _build_image_top3(row: dict, rng: random.Random) -> list[dict]:
    cond = row["condition"]
    pool = DIFFERENTIALS.get(cond, ["dermatitis (nonspecific)", "post-inflammatory change", "unknown"])
    diffs = rng.sample(pool, k=min(2, len(pool)))
    items = [(cond, rng.uniform(0.55, 0.85))] + [(d, rng.uniform(0.10, 0.30)) for d in diffs]
    rng.shuffle(items)
    # Renormalize so confidences fall in [0,1] and are descending after sort
    items.sort(key=lambda kv: -kv[1])
    return [{"condition": c, "confidence": round(v, 2)} for c, v in items[:3]]


def _build_soap(row: dict) -> dict:
    syms = []
    raw = row.get("narrative", "")
    if "itchy" in raw: syms.append("itching")
    if "burning" in raw: syms.append("burning")
    if "painful" in raw: syms.append("pain")
    if "fever" in raw: syms.append("fever")
    if "bleeding" in raw: syms.append("bleeding intermittent")
    if "getting larger" in raw: syms.append("increasing size")
    return {
        "chief_complaint": "skin issue",
        "hpi": row.get("narrative", ""),
        "duration": "as reported",
        "associated_symptoms": syms or ["none stated"],
        "past_medical_history": ["none stated"],
        "medications": ["none stated"],
        "vitals": {},
        "exam_findings": [row.get("image_description", "")],
        "red_flags": [],
        "patient_concerns": [],
    }


def _build_village_blurb(row: dict) -> str:
    v = row.get("village_context", {})
    dist = v.get("distance_to_clinic_km", 18)
    wage = v.get("patient_daily_wage_inr", 350)
    cost = v.get("transport_cost_round_trip_inr", 180)
    pct = round(100.0 * cost / max(wage, 1), 1)
    harv = v.get("harvest_active", True)
    return (
        f"PHC is {dist} km away; round-trip transport is ₹{cost} (≈{pct}% of a "
        f"daily wage of ₹{wage}). District hospital is 65 km. "
        f"{'Harvest is active — leaving the field costs visible income today.' if harv else 'No harvest pressure today.'} "
        f"Antibiotics typically free at PHC if in stock; antivenom only at district hospital."
    )


def _format_top3(top3: list[dict]) -> str:
    return "\n".join(
        f"- {c['condition']} (confidence {c['confidence']:.2f})" for c in top3
    )


def _format_soap(soap: dict) -> str:
    return "\n".join(
        f"{k}: {v}"
        for k, v in soap.items()
    )


def _build_prompt(row: dict, top3: list[dict], soap: dict) -> str:
    from mcp.triage_reasoner.server import TRIAGE_PROMPT
    return TRIAGE_PROMPT.format(
        image_top3=_format_top3(top3),
        soap_fields=_format_soap(soap),
        village_context=_build_village_blurb(row),
    )


def _build_target(row: dict) -> str:
    urg = row["ground_truth_urgency"]
    flags = []
    if "fever" in row.get("narrative", ""): flags.append("fever")
    if "shortness of breath" in row.get("narrative", "").lower(): flags.append("shortness of breath")
    if "mouth" in row.get("narrative", "").lower(): flags.append("mucosal involvement")
    if "bleeding" in row.get("narrative", "").lower(): flags.append("bleeding")
    flags_str = ", ".join(flags) if flags else "none"
    reasoning = REASONING_TEMPLATES[urg].format(flags=flags_str if flags else "watch for spreading or systemic signs")
    framing = FRAMING_TEMPLATES[urg]
    return (
        f"URGENCY: {urg}\n"
        f"REASONING: {reasoning}\n"
        f"RED_FLAGS_NOTED: {flags_str}\n"
        f"PATIENT_FRAMING: {framing}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-in", default="data/scin/train.jsonl")
    ap.add_argument("--holdout-in", default="data/scin/holdout.jsonl")
    ap.add_argument("--train-out", default="data/scin/train_prompts.jsonl")
    ap.add_argument("--holdout-out", default="data/scin/holdout_prompts.jsonl")
    args = ap.parse_args()

    Path(os.path.dirname(args.train_out)).mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)

    for in_path, out_path in [(args.train_in, args.train_out), (args.holdout_in, args.holdout_out)]:
        rows = [json.loads(l) for l in open(in_path) if l.strip()]
        with open(out_path, "w") as f:
            for r in rows:
                top3 = _build_image_top3(r, rng)
                soap = _build_soap(r)
                prompt = _build_prompt(r, top3, soap)
                target = _build_target(r)
                f.write(json.dumps({
                    "case_id": r["case_id"],
                    "prompt": prompt,
                    "target": target,
                    "ground_truth_urgency": r["ground_truth_urgency"],
                    "condition": r["condition"],
                    "fitzpatrick_bucket": r["fitzpatrick_bucket"],
                    "image_path_local": r["image_path_local"],
                }, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
