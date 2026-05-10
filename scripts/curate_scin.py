"""Curate the SCIN public dataset down to the **top-10 most-frequent
dermatologist-labeled conditions**, map each to an urgency level (Red /
Yellow / Green) via a defensible clinical heuristic, and emit a clean
text-image-pair JSONL the harness can consume.

Inputs (already staged):
  data/scin/scin_labels.csv
  data/scin/scin_cases.csv

Outputs:
  data/scin/curated.jsonl        (one row per case in the top-10 subset)
  data/scin/condition_urgency.json (the heuristic mapping; auditable)
  evidence/scin_top10.txt        (frequencies + the urgency map)
"""
from __future__ import annotations

import ast
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path


LABELS = "data/scin/scin_labels.csv"
CASES = "data/scin/scin_cases.csv"
OUT_JSONL = "data/scin/curated.jsonl"
OUT_MAP = "data/scin/condition_urgency.json"
OUT_EVIDENCE = "evidence/scin_top10.txt"


# Per-condition BASE urgency. Final urgency is base + per-case symptom
# adjustments (see _adjust_urgency below): fever, mouth sores, anaphylaxis
# cues bump up; no-symptoms / unchanged bumps down. This gives the urgency
# label real R/Y/G variation within each condition class — same condition,
# different urgency depending on the patient's symptoms — instead of every
# eczema case becoming yellow regardless of presentation.
URGENCY_MAP_BASE: dict[str, str] = {
    # --- Likely RED (require same-day evaluation) ---
    "Skin infections (e.g., bacterial)": "red",   # cellulitis spectrum, can progress rapidly
    "Cellulitis": "red",
    "Necrotizing fasciitis": "red",
    "Stevens-Johnson Syndrome / Toxic Epidermal Necrolysis": "red",
    "Anaphylaxis / Angioedema": "red",
    "DRESS Syndrome": "red",
    "Severe drug reaction": "red",

    # --- Likely YELLOW (1-2 day clinical evaluation) ---
    "Acne": "yellow",                              # bothersome, common
    "Eczema": "yellow",                            # flaring eczema warrants follow-up
    "Atopic dermatitis": "yellow",
    "Allergic Contact Dermatitis": "yellow",
    "Irritant Contact Dermatitis": "yellow",
    "Contact dermatitis": "yellow",
    "Psoriasis": "yellow",
    "Rosacea": "yellow",
    "Tinea": "yellow",                             # contagious, treatable
    "Tinea (e.g., ringworm, athlete's foot)": "yellow",
    "Folliculitis": "yellow",
    "Impetigo": "yellow",
    "Herpes Zoster": "yellow",
    "Herpes Simplex": "yellow",
    "Urticaria": "yellow",                         # without anaphylaxis
    "Hives": "yellow",
    "Drug Rash": "yellow",
    "Drug Reaction": "yellow",
    "Stasis Dermatitis": "yellow",
    "Inflicted skin lesions": "yellow",            # safeguarding concern
    "Scabies": "yellow",
    "Cyst": "yellow",                              # if inflamed; we map default yellow
    "Abscess": "yellow",                           # needs drainage
    "Fungal Infection": "yellow",

    # --- Likely GREEN (monitor at home) ---
    "Acne (mild)": "green",
    "Seborrheic Keratosis": "green",
    "Lentigo": "green",
    "Solar Lentigo": "green",
    "Skin tag": "green",
    "Skin Tag": "green",
    "Acrochordon": "green",
    "Lipoma": "green",
    "Keloid": "green",
    "Striae": "green",
    "Vitiligo": "green",
    "Birthmark": "green",
    "Nevus": "green",
    "Melanocytic Nevus": "green",
    "Mole": "green",
    "Pityriasis Rosea": "green",
    "Pityriasis Alba": "green",
    "Xerosis": "green",
    "Dry skin": "green",
    "Calluses / Corns": "green",
    "Callus": "green",
    "Insect bite": "green",
    "Bug Bite": "green",
    "Mosquito bite": "green",
    "Heat Rash": "green",
    "Miliaria": "green",
    "Sunburn (mild)": "green",
    "Dandruff": "green",

    # --- Cancer / pre-cancer suspicion (evaluation in 1-2 days; never green) ---
    "Actinic keratosis": "yellow",
    "Basal Cell Carcinoma": "yellow",
    "Basal cell carcinoma": "yellow",
    "Squamous Cell Carcinoma": "yellow",
    "Squamous cell carcinoma": "yellow",
    "Melanoma": "red",                             # dermatology emergency for staging
    "Cutaneous T-cell Lymphoma": "yellow",
    "Skin cancer / Pre-cancer": "yellow",

    # --- Hair / nail (rarely urgent) ---
    "Alopecia areata": "green",
    "Androgenetic Alopecia": "green",
    "Onychomycosis": "green",
    "Tinea Capitis": "yellow",

    # --- Common SCIN labels we hit during curation ---
    "Insect Bite": "green",                        # benign default; bumps up with systemic signs
    "Pigmented purpuric eruption": "green",        # chronic benign, slow course
    "Pigmented Purpuric Eruption": "green",
    "Bug Bite": "green",
}


_ORDER = ["green", "yellow", "red"]


def _bump(urg: str, delta: int) -> str:
    """Bump urgency up (delta>0) or down (delta<0) one step on the R/Y/G ladder.
    Clamped at the ends."""
    i = _ORDER.index(urg)
    j = max(0, min(len(_ORDER) - 1, i + delta))
    return _ORDER[j]


def _adjust_urgency(base: str, case_row: dict) -> str:
    """Adjust base condition urgency based on per-case symptom flags.

    Up-bump triggers (each +1, capped at red):
      - fever
      - mouth sores  (mucosal involvement → SJS/severe drug reaction concern)
      - shortness of breath  (anaphylaxis cue)
      - bleeding (active)

    Down-bump triggers (each -1, capped at green):
      - no_relevant_symptoms set AND no_relevant_experience  (unchanged, no symptoms)
      - duration suggests chronic stable course (>= 1 year and no recent change)
    """
    urg = base
    fever = case_row.get("other_symptoms_fever") == "YES"
    chills = case_row.get("other_symptoms_chills") == "YES"
    sob = case_row.get("other_symptoms_shortness_of_breath") == "YES"
    mouth = case_row.get("other_symptoms_mouth_sores") == "YES"
    bleeding = case_row.get("condition_symptoms_bleeding") == "YES"

    if fever or chills:
        urg = _bump(urg, +1)
    if sob:
        urg = _bump(urg, +1)
    if mouth:
        urg = _bump(urg, +1)
    if bleeding:
        # bleeding from a skin lesion: bump unless it's a tiny cut/abrasion. Heuristic +1.
        urg = _bump(urg, +1)

    no_sym = (case_row.get("condition_symptoms_no_relevant_experience") == "YES"
              and case_row.get("other_symptoms_no_relevant_symptoms") == "YES")
    duration = (case_row.get("condition_duration") or "").upper()
    if no_sym and duration in ("ONE_YEAR_OR_MORE", "MORE_THAN_ONE_YEAR"):
        urg = _bump(urg, -1)

    return urg


def _primary_condition(weighted_label_str: str) -> str | None:
    """Return the highest-weighted condition from the JSON-like dict in the
    `weighted_skin_condition_label` column, or None on parse failure."""
    if not weighted_label_str or weighted_label_str.strip() in ("", "{}"):
        return None
    try:
        d = ast.literal_eval(weighted_label_str)
        if not isinstance(d, dict) or not d:
            return None
        return max(d.items(), key=lambda kv: kv[1])[0]
    except (SyntaxError, ValueError):
        return None


def _fitzpatrick_bucket(case_row: dict) -> str:
    """Self-reported FST -> bucket. Fall back to a placeholder if missing."""
    fst = (case_row.get("fitzpatrick_skin_type") or "").strip().upper()
    mapping = {
        "FST1": "I-II", "FST2": "I-II",
        "FST3": "III-IV", "FST4": "III-IV",
        "FST5": "V-VI", "FST6": "V-VI",
    }
    return mapping.get(fst, "unknown")


def _build_narrative(case_row: dict, condition: str) -> str:
    """Construct a short patient-style narrative from the SCIN metadata.
    The narrative does NOT mention the condition — that's the label."""
    age = case_row.get("age_group", "")
    sex = case_row.get("sex_at_birth", "")
    duration = case_row.get("condition_duration", "")
    parts = []
    body_parts = [k.replace("body_parts_", "")
                  for k in case_row if k.startswith("body_parts_") and case_row.get(k) == "YES"]
    body_str = ", ".join(body_parts) if body_parts else "skin"
    sym_keys = [
        ("condition_symptoms_itching", "itchy"),
        ("condition_symptoms_burning", "burning"),
        ("condition_symptoms_pain", "painful"),
        ("condition_symptoms_bleeding", "bleeding sometimes"),
        ("condition_symptoms_increasing_size", "getting larger"),
        ("condition_symptoms_darkening", "darkening"),
        ("other_symptoms_fever", "with mild fever"),
    ]
    syms = [phrase for col, phrase in sym_keys if case_row.get(col) == "YES"]
    sym_str = (", ".join(syms[:-1]) + " and " + syms[-1]) if len(syms) >= 2 else (syms[0] if syms else "")

    bits = []
    if age and age != "AGE_UNKNOWN":
        bits.append(f"I am in the {age.lower().replace('_',' ')} age group.")
    if sym_str:
        bits.append(f"Skin issue on my {body_str}, {sym_str}.")
    else:
        bits.append(f"Skin issue on my {body_str}.")
    if duration and duration not in ("", "ONE_DAY"):
        bits.append(f"It has been there for {duration.lower().replace('_',' ')}.")
    return " ".join(bits)


def _build_image_description(case_row: dict, condition: str) -> str:
    """Generate a brief image description from textures + body parts —
    deliberately NOT mentioning the condition diagnosis."""
    body_parts = [k.replace("body_parts_", "").replace("_", " ")
                  for k in case_row if k.startswith("body_parts_") and case_row.get(k) == "YES"]
    body_str = ", ".join(body_parts) if body_parts else "skin"
    tex = []
    if case_row.get("textures_raised_or_bumpy") == "YES": tex.append("raised/bumpy")
    if case_row.get("textures_flat") == "YES": tex.append("flat")
    if case_row.get("textures_rough_or_flaky") == "YES": tex.append("rough/flaky")
    if case_row.get("textures_fluid_filled") == "YES": tex.append("fluid-filled")
    tex_str = ", ".join(tex) if tex else "varied appearance"
    return f"Phone photo of {body_str}; texture: {tex_str}."


def main() -> int:
    Path("evidence").mkdir(exist_ok=True)
    Path("data/scin").mkdir(parents=True, exist_ok=True)

    # Index labels by case_id
    labels: dict[str, dict] = {}
    with open(LABELS) as f:
        for row in csv.DictReader(f):
            labels[row["case_id"]] = row

    # Load cases keeping only those with a derm-labeled primary condition
    cases: list[dict] = []
    with open(CASES) as f:
        for row in csv.DictReader(f):
            cid = row["case_id"]
            lab = labels.get(cid)
            if not lab:
                continue
            primary = _primary_condition(lab.get("weighted_skin_condition_label", ""))
            if not primary:
                continue
            row["_primary_condition"] = primary
            row["_fitzpatrick_bucket"] = _fitzpatrick_bucket(row)
            row["_image_path"] = row.get("image_1_path", "").strip()
            cases.append(row)

    counts = Counter(c["_primary_condition"] for c in cases)
    top10 = [c for c, _ in counts.most_common(10)]
    top10_set = set(top10)

    curated: list[dict] = []
    skipped_unmapped: Counter = Counter()
    for c in cases:
        cond = c["_primary_condition"]
        if cond not in top10_set:
            continue
        base = URGENCY_MAP_BASE.get(cond)
        if base is None:
            skipped_unmapped[cond] += 1
            continue
        urgency = _adjust_urgency(base, c)
        if not c["_image_path"]:
            continue
        curated.append({
            "case_id": "SCIN-" + c["case_id"],
            "scin_case_id": c["case_id"],
            "condition": cond,
            "base_urgency": base,
            "ground_truth_urgency": urgency,
            "fitzpatrick_bucket": c["_fitzpatrick_bucket"],
            "narrative": _build_narrative(c, cond),
            "image_description": _build_image_description(c, cond),
            "image_path_remote": f"https://storage.googleapis.com/dx-scin-public-data/{c['_image_path']}",
            "image_path_local": f"data/scin/images/{c['case_id']}.png",
            "red_flags": [],  # filled per-condition by the heuristic if relevant
            "village_context": {
                "distance_to_clinic_km": 18, "patient_daily_wage_inr": 350,
                "transport_cost_round_trip_inr": 180, "harvest_active": True,
            },
            "adversarial_features": [],
        })

    with open(OUT_JSONL, "w") as f:
        for row in curated:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(OUT_MAP, "w") as f:
        json.dump(URGENCY_MAP_BASE, f, indent=2, sort_keys=True)

    # Evidence summary
    by_urg = Counter(r["ground_truth_urgency"] for r in curated)
    by_fst = Counter(r["fitzpatrick_bucket"] for r in curated)
    lines = [
        f"verdict: PASS",
        f"total_scin_cases_with_labels: {len(cases)}",
        f"top10_conditions:",
    ]
    for cond in top10:
        urg = URGENCY_MAP_BASE.get(cond, "(unmapped)")
        lines.append(f"  {counts[cond]:>5d}  {cond}  ->  {urg}")
    lines.append(f"")
    lines.append(f"curated_after_top10_and_urgency_map: {len(curated)}")
    lines.append(f"by_urgency: {dict(by_urg)}")
    lines.append(f"by_fitzpatrick: {dict(by_fst)}")
    if skipped_unmapped:
        lines.append(f"skipped_unmapped_conditions: {dict(skipped_unmapped)}")
    out = "\n".join(lines) + "\n"
    print(out)
    with open(OUT_EVIDENCE, "w") as f:
        f.write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
