"""Adversarial test-case generator for Path to Care.

Produces 30 deterministic cases (10 Red, 10 Yellow, 10 Green) with realistic
rural-Indian dermatology presentations and adversarial perturbations baked in:
colloquial dialect, irrelevant context (harvest season, neighbor's opinion),
contradictory narrative fragments ("no fever" + "feverish"), off-distribution
image references (a wall, a cat), and blurred-image flags.

Output: data/cases.jsonl. Each line is one case object.

Why hand-crafted rather than LLM-generated for the 30 test cases:
- Deterministic, reproducible, auditable.
- Ground-truth urgency labels are stable (an LLM-author + LLM-grader is
  contaminated; here a human authored, the model is graded).
- 24-hour window: zero risk of generation failures.

The set is intentionally small (30) because every case has a hand-set
ground-truth urgency. See docs/EVALUATION.md for the reward and stratification
spec. The 80-case set in the brief is the v2 target.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

OUT = "data/cases.jsonl"

# 10 RED cases ---------------------------------------------------------------
RED_CASES = [
    {
        "case_id": "P2C-R01",
        "narrative": "I cut my foot on a rusty nail two days back when I was working in the field. Now my whole foot is swollen and red, the redness is going up my leg. I have fever since yesterday night, body shivering. Cannot keep weight on the foot.",
        "image_description": "Lower leg with poorly demarcated erythema extending proximally above the wound; warmth and edema; small puncture wound visible on plantar foot.",
        "red_flags": ["spreading erythema", "fever", "rigors", "wound is tetanus-prone"],
        "adversarial_features": ["dialect: 'two days back', 'body shivering'", "irrelevant context: field work"],
    },
    {
        "case_id": "P2C-R02",
        "narrative": "My child fell into the cooking fire. The skin on her arm is white in some places and very red in others, big blisters. She is crying and not eating. The arm is from elbow to wrist.",
        "image_description": "Forearm with mixed second- and possibly third-degree burns, blistering and pale areas suggesting deeper involvement; >5% body surface area in a child.",
        "red_flags": ["mixed-depth burn", "child", "large surface area", "pain disproportionate"],
        "adversarial_features": ["irrelevant context: cooking fire mentioned"],
    },
    {
        "case_id": "P2C-R03",
        "narrative": "Snake bit me one hour ago in the paddy field. Two punctures on my ankle. The leg is swelling fast and I feel dizzy. My gums are bleeding little bit when I spit.",
        "image_description": "Ankle with two fang puncture marks 1 cm apart, rapidly progressive edema, ecchymosis spreading proximally.",
        "red_flags": ["envenomation", "systemic bleeding", "rapid edema progression", "neurological symptoms"],
        "adversarial_features": ["irrelevant context: paddy field"],
    },
    {
        "case_id": "P2C-R04",
        "narrative": "My father has diabetes. Last week he had a small wound on his big toe. Today the toe is black and smells bad. He says no pain there but he is tired and confused.",
        "image_description": "Great toe with dry gangrenous changes, black eschar, surrounding cellulitis; absent pulses suspected.",
        "red_flags": ["diabetic foot", "tissue necrosis", "altered mental status", "absent pain (neuropathy + advanced)"],
        "adversarial_features": ["contradiction-trap: 'no pain' but altered mental status implies systemic illness"],
    },
    {
        "case_id": "P2C-R05",
        "narrative": "I scraped my hand on barbed wire three days ago, only small scratch. Today my jaw feels stiff, hard to open mouth. Hand wound looks small but skin around it is purple.",
        "image_description": "Hand with small healing-appearing wound but surrounding dusky skin; trismus reported clinically.",
        "red_flags": ["tetanus suspicion (trismus)", "wound discoloration", "delayed presentation"],
        "adversarial_features": ["misleading: 'only small scratch'"],
    },
    {
        "case_id": "P2C-R06",
        "narrative": "After taking new tablets for headache, my whole body has rashes that came in 30 minutes. My lips are swelling and my throat feels tight. Hard to breathe.",
        "image_description": "Generalized urticarial wheals across torso and limbs; angioedema of lips visible.",
        "red_flags": ["anaphylaxis", "airway compromise", "drug reaction"],
        "adversarial_features": ["irrelevant context: 'tablets for headache'"],
    },
    {
        "case_id": "P2C-R07",
        "narrative": "Big painful lump on my hand for a week. Now red lines going up to my armpit, and the armpit hurts. I have low fever.",
        "image_description": "Dorsum of hand with fluctuant abscess and surrounding cellulitis; visible lymphangitic streaking proximally to axilla.",
        "red_flags": ["lymphangitis", "regional lymphadenopathy", "systemic signs"],
        "adversarial_features": [],
    },
    {
        "case_id": "P2C-R08",
        "narrative": "After harvest work in the sun all day, big red patches on shoulders and back, blisters now. Old man, age 72. He is sleepy and not drinking water.",
        "image_description": "Extensive second-degree sunburn over upper back and shoulders with blistering; elderly patient.",
        "red_flags": ["large-area burn in elderly", "dehydration signs", "altered mental status"],
        "adversarial_features": ["irrelevant context: 'harvest work'"],
    },
    {
        "case_id": "P2C-R09",
        "narrative": "Pain in my thigh two days now, started small. Today the skin is dark and feels like crackling under it. Pain is much more than the redness looks. Fever 39.",
        "image_description": "Thigh with dusky discoloration disproportionate to surface erythema, palpable subcutaneous crepitus suggested.",
        "red_flags": ["pain out of proportion", "crepitus", "necrotizing fasciitis suspicion"],
        "adversarial_features": ["misleading: 'started small'"],
    },
    {
        "case_id": "P2C-R10",
        "narrative": "My son is 4 years. He has chickenpox spots since 5 days, now suddenly some spots are very red and warm with yellow pus, and he has high fever 40 again. He is sleepy.",
        "image_description": "Child with chickenpox lesions, several with secondary bacterial infection (purulent, surrounding erythema); systemically ill.",
        "red_flags": ["secondary bacterial infection of varicella", "high fever", "lethargy in child"],
        "adversarial_features": ["distractor: chickenpox is usually green-yellow; superinfection escalates"],
    },
]

# 10 YELLOW cases ------------------------------------------------------------
YELLOW_CASES = [
    {
        "case_id": "P2C-Y01",
        "narrative": "My daughter has yellow crusted sores around her mouth and nose since 3 days. They are spreading. No fever. Her brother is also getting same thing.",
        "image_description": "Honey-colored crusted lesions on perioral and nasal areas, classic impetigo distribution.",
        "red_flags": ["contagious", "spreading"],
        "adversarial_features": ["distractor: family member also affected"],
    },
    {
        "case_id": "P2C-Y02",
        "narrative": "Round patch on my chest, ring shape, itchy. Started small two weeks ago, now bigger than 2 rupees coin. Center is clear, edge is red.",
        "image_description": "Annular erythematous plaque with raised scaly border and central clearing; classic tinea corporis.",
        "red_flags": [],
        "adversarial_features": ["dialect: 'two rupees coin'"],
    },
    {
        "case_id": "P2C-Y03",
        "narrative": "I bought new soap last week. Now both hands are red, itchy, small bumps. No fever, no spread to other areas. Worse after washing.",
        "image_description": "Bilateral hand dermatitis with erythema and microvesicles in soap-contact distribution.",
        "red_flags": [],
        "adversarial_features": ["adversary: trigger-temporal pattern as distractor"],
    },
    {
        "case_id": "P2C-Y04",
        "narrative": "Small red area on my shin, warm, little bit tender, since 2 days. Size of my palm. No fever, no streaks going up. Walking is okay.",
        "image_description": "Localized erythema on pretibial shin, warm, sharply demarcated; no lymphangitic streaking.",
        "red_flags": ["watch for systemic progression"],
        "adversarial_features": [],
    },
    {
        "case_id": "P2C-Y05",
        "narrative": "Pimples on my face and back since teen years, but now some have become big and yellow with pus, painful. About 5-6 of them. No fever.",
        "image_description": "Inflammatory acne with several pustular and nodulocystic lesions on face and upper back.",
        "red_flags": [],
        "adversarial_features": ["distractor: chronic baseline condition"],
    },
    {
        "case_id": "P2C-Y06",
        "narrative": "Itching very bad at night, especially in finger webs and waist. My wife also itching. We see small lines and bumps.",
        "image_description": "Burrows visible in interdigital web spaces and waistline; classic scabies distribution.",
        "red_flags": ["household contact"],
        "adversarial_features": [],
    },
    {
        "case_id": "P2C-Y07",
        "narrative": "Long-time eczema patches on my elbows and behind knees. Now suddenly very red, weeping, since 4 days. More itchy than usual. No fever.",
        "image_description": "Atopic dermatitis flexural distribution with acute weeping eczematous changes; possible early secondary infection.",
        "red_flags": ["watch for secondary bacterial infection"],
        "adversarial_features": ["distractor: chronic baseline"],
    },
    {
        "case_id": "P2C-Y08",
        "narrative": "White scaly patches between my toes, peeling skin, itching, smell. Working in wet fields all day. No redness going up.",
        "image_description": "Tinea pedis interdigital with scaling and maceration.",
        "red_flags": ["risk of bacterial superinfection in diabetics"],
        "adversarial_features": ["irrelevant context: 'wet fields'"],
    },
    {
        "case_id": "P2C-Y09",
        "narrative": "Itchy spots all over body, started yesterday on chest. Adult, 35 years. Slight fever 38. Some spots are blistering. My nephew had chickenpox last week.",
        "image_description": "Generalized vesicular eruption in different stages, classic varicella in an adult.",
        "red_flags": ["adult chickenpox can be more severe", "watch for pneumonitis"],
        "adversarial_features": ["distractor: family contact"],
    },
    {
        "case_id": "P2C-Y10",
        "narrative": "Painful red rash on left side of my chest, in a band. Burning pain started 2 days before rash. Small blisters now.",
        "image_description": "Unilateral dermatomal vesicular eruption on T4-T5 distribution; classic herpes zoster.",
        "red_flags": ["antivirals within 72h ideal", "watch for ophthalmic involvement if face"],
        "adversarial_features": [],
    },
]

# 10 GREEN cases -------------------------------------------------------------
GREEN_CASES = [
    {
        "case_id": "P2C-G01",
        "narrative": "Tiny red bumps on neck and chest after I worked in the sun. Itchy little bit. No fever. They go little bit better at night.",
        "image_description": "Miliaria rubra (heat rash) — fine erythematous papules on covered areas.",
        "red_flags": [],
        "adversarial_features": [],
    },
    {
        "case_id": "P2C-G02",
        "narrative": "Small scratch on my arm from yesterday, when I was cutting vegetables. It is healing, scab forming. Little red around it but not spreading.",
        "image_description": "Healing superficial abrasion with scab formation, minimal surrounding erythema.",
        "red_flags": [],
        "adversarial_features": ["irrelevant context: 'cutting vegetables'"],
    },
    {
        "case_id": "P2C-G03",
        "narrative": "Many mosquito bites on legs from last evening, itchy. No spreading red lines, no pus, no fever.",
        "image_description": "Multiple discrete pruritic papules on lower legs, classic mosquito bite distribution.",
        "red_flags": [],
        "adversarial_features": [],
    },
    {
        "case_id": "P2C-G04",
        "narrative": "Red on my face after I was in the sun for one hour without scarf. Bit warm, no blister. Will it be fine if I put coconut oil?",
        "image_description": "Mild first-degree facial sunburn with diffuse erythema, no blistering.",
        "red_flags": [],
        "adversarial_features": ["dialect: 'will it be fine if I put coconut oil'"],
    },
    {
        "case_id": "P2C-G05",
        "narrative": "My skin is very dry, especially on my legs and arms. Some flaking. Itchy little bit. Worse in cold weather.",
        "image_description": "Generalized xerosis with fine scaling on extremities.",
        "red_flags": [],
        "adversarial_features": [],
    },
    {
        "case_id": "P2C-G06",
        "narrative": "Old scar from surgery 5 years back on my belly. Sometimes feels itchy, slightly red. No swelling, no pus, no fever.",
        "image_description": "Mature surgical scar with mild surrounding hyperpigmentation; no signs of inflammation.",
        "red_flags": [],
        "adversarial_features": ["distractor: longstanding lesion"],
    },
    {
        "case_id": "P2C-G07",
        "narrative": "Few pimples on my forehead since this week. No pus, just bumps. Teenage daughter. Worse before her periods.",
        "image_description": "A few comedones and small papules on forehead, mild teenage acne.",
        "red_flags": [],
        "adversarial_features": [],
    },
    {
        "case_id": "P2C-G08",
        "narrative": "I scraped my knee yesterday playing with my kid. Just a small graze. Cleaned it with water. Looks okay today.",
        "image_description": "Superficial abrasion on knee, clean, no signs of infection.",
        "red_flags": [],
        "adversarial_features": ["irrelevant context: 'playing with my kid'"],
    },
    {
        "case_id": "P2C-G09",
        "narrative": "My head has white flaky scales for many years, especially when I don't oil my hair. Itchy sometimes. No bald patches.",
        "image_description": "Diffuse fine white scaling on scalp, mild seborrheic dermatitis / dandruff.",
        "red_flags": [],
        "adversarial_features": ["distractor: chronic"],
    },
    {
        "case_id": "P2C-G10",
        "narrative": "Small dry rough patch on my elbow for many years. Sometimes I scratch and it gets red. No spreading. No bleeding. No fever.",
        "image_description": "Localized lichenified eczema patch on elbow, chronic and stable.",
        "red_flags": [],
        "adversarial_features": ["distractor: longstanding"],
    },
]

# Adversarial perturbations layered on top -----------------------------------
# Applied deterministically by case_id suffix so the test set is reproducible.

CONTRADICTIONS = {
    "P2C-R02": " I have no fever.",        # contradicts implicit "child crying, not eating"
    "P2C-Y04": " I think I had high fever yesterday.",  # contradicts "no fever"
    "P2C-G03": " But I feel hot all day.",  # mild contradiction
}
OFF_DISTRIBUTION_IMAGE = {
    "P2C-Y08": "PHOTO BLURRED — patient could not hold still.",
    "P2C-G05": "PHOTO OFF-TOPIC — image appears to show a wall, not skin.",
}
DIALECT_NOISE_PREFIX = {
    "P2C-R03": "Aiyo doctor, ",
    "P2C-Y01": "My neighbor said maybe it is contagion, ",
    "P2C-G04": "Auntie said apply turmeric, ",
}

VILLAGE_CONTEXT = {
    "distance_to_clinic_km": 18,
    "clinic_open_hours": "10:00-13:00 weekdays, mobile dispensary Wednesday",
    "harvest_active": True,
    "nearest_hospital_km": 65,
    "transport_cost_round_trip_inr": 180,
    "patient_daily_wage_inr": 350,
}


def _apply_perturbations(case: dict) -> dict:
    cid = case["case_id"]
    if cid in CONTRADICTIONS:
        case["narrative"] += CONTRADICTIONS[cid]
        case.setdefault("adversarial_features", []).append("contradicted_narrative")
    if cid in OFF_DISTRIBUTION_IMAGE:
        case["image_description"] = OFF_DISTRIBUTION_IMAGE[cid]
        case.setdefault("adversarial_features", []).append("off_distribution_or_blurred_image")
    if cid in DIALECT_NOISE_PREFIX:
        case["narrative"] = DIALECT_NOISE_PREFIX[cid] + case["narrative"]
        case.setdefault("adversarial_features", []).append("dialect_or_third_party_noise")
    return case


def build_cases() -> list[dict]:
    out = []
    for urgency, group in (("red", RED_CASES), ("yellow", YELLOW_CASES), ("green", GREEN_CASES)):
        for c in group:
            case = dict(c)  # shallow copy
            case["ground_truth_urgency"] = urgency
            case["village_context"] = dict(VILLAGE_CONTEXT)
            case["image_ref"] = f"data/images/{case['case_id']}.jpg"  # placeholder; image dataset filled later
            case = _apply_perturbations(case)
            out.append(case)
    return out


def write(path: str = OUT) -> int:
    cases = build_cases()
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"wrote {len(cases)} cases to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(write())
