"""30 new training cases targeting failure-mode patterns.

These are intended as TRAINING-ONLY cases (case_id prefix `P2C-T-`) — never
in the test set. Each case shares a *pattern* with the hard 30 / extra 20
test cases but uses different specifics to avoid leakage: different
anatomy, different demographic slot, different pathogen, different drug.

Pattern coverage (mirrors the failure modes documented in
docs/FINE_TUNING_DEMO.md):
  - High-risk patient + minor presentation (RED that looks GREEN)
  - Atypical / pain-out-of-proportion / immunocompromised
  - Drug eruptions of varying severity
  - Dramatic narrative + benign reality (GREEN that sounds RED)
  - Post-treatment non-response
  - Pediatric, pregnant, lactating, elderly variants
  - Mucosal involvement subtle
  - Time-since-onset + delayed envenomation/toxicity
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

OUT = "data/cases_train_v2.jsonl"


# 10 RED training cases — atypical / high-risk variants ---------------------
RED_CASES = [
    {"case_id": "P2C-T-R01",
     "narrative": "I am 55, kidney transplant 3 years back, on tacrolimus. Small skin spot on my arm 5 days, not painful, slightly raised. Today I feel feverish, 38.3.",
     "image_description": "Solitary erythematous papule on forearm in solid-organ transplant recipient with low-grade fever; possible disseminated infection.",
     "red_flags": ["transplant + atypical infection", "low threshold for opportunistic"]},

    {"case_id": "P2C-T-R02",
     "narrative": "Painful right side of perineum since 2 days, redness spreading to scrotum, fever 38.8. I have diabetes.",
     "image_description": "Perineal/scrotal cellulitis with rapid spread in a diabetic; concern for Fournier gangrene.",
     "red_flags": ["Fournier gangrene suspicion", "surgical emergency"]},

    {"case_id": "P2C-T-R03",
     "narrative": "Working in flood water, scratched leg on metal 3 days back. Now leg pain so severe I cannot sleep. Skin only mildly red, but feels like there is gas under it.",
     "image_description": "Lower leg with surface mild erythema, palpable subcutaneous crepitus, severe pain disproportionate to surface findings.",
     "red_flags": ["necrotizing fasciitis with crepitus", "surgical emergency"]},

    {"case_id": "P2C-T-R04",
     "narrative": "8-month-old fell from cot 2 days back. Today bruise spreading and not eating, very fussy. Bruise is purple over a wide area on the thigh.",
     "image_description": "Wide ecchymosis on thigh in infant beyond mechanism of injury; consider non-accidental trauma + coagulopathy.",
     "red_flags": ["bruise out of proportion in infant", "rule out NAT and bleeding disorder"]},

    {"case_id": "P2C-T-R05",
     "narrative": "I started seizure tablet 3 weeks back. Last 4 days fever, body aches, swollen face, rash starting on chest spreading.",
     "image_description": "Truncal morbilliform eruption with facial edema, fever, lymphadenopathy implied; DRESS syndrome.",
     "red_flags": ["DRESS (drug reaction with eosinophilia and systemic symptoms)", "stop drug, hospitalize"]},

    {"case_id": "P2C-T-R06",
     "narrative": "Snake bit my husband on the hand 8 hours ago. Initially fine, now drowsy, eye-lids drooping, having trouble swallowing.",
     "image_description": "Hand snakebite with delayed neurotoxic symptoms (ptosis, dysphagia).",
     "red_flags": ["neurotoxic envenomation", "antivenom + airway support"]},

    {"case_id": "P2C-T-R07",
     "narrative": "My son with chickenpox last week. Today new fever 40, one chickenpox spot on his thigh has become very large, red, and is draining yellow pus. He is pale and breathing fast.",
     "image_description": "Varicella with secondary bacterial infection progressing to toxic features in a child.",
     "red_flags": ["secondary bacterial sepsis from varicella", "pediatric emergency"]},

    {"case_id": "P2C-T-R08",
     "narrative": "I have HIV, CD4 was low last test. Painful blisters on my forehead in a stripe, going close to my eye. Eye is red and watery.",
     "image_description": "Herpes zoster ophthalmicus (V1) in immunosuppressed patient with ocular involvement.",
     "red_flags": ["zoster ophthalmicus + immunosuppression", "antivirals + ophthalmology"]},

    {"case_id": "P2C-T-R09",
     "narrative": "I am 4 months pregnant. Bee swarm attacked me, many stings. Now generalized rash, throat closing feeling, feeling faint.",
     "image_description": "Multiple sting sites, generalized urticaria, oropharyngeal angioedema, pre-syncope; anaphylaxis in pregnancy.",
     "red_flags": ["anaphylaxis in pregnancy", "epinephrine + ED"]},

    {"case_id": "P2C-T-R10",
     "narrative": "Skin on my back where the IV was inserted in hospital last week is now blackening and the surrounding area is hot and tender. I have fever 38.5.",
     "image_description": "Eschar at IV site with surrounding cellulitis and fever; possible necrotizing soft tissue infection.",
     "red_flags": ["IV-site necrosis + sepsis signs"]},
]


# 10 YELLOW training cases ---------------------------------------------------
YELLOW_CASES = [
    {"case_id": "P2C-T-Y01",
     "narrative": "Yellow crusted sores around my child's nostrils for 4 days. Spreading to lips. No fever. Other kids in school have similar.",
     "image_description": "Perioral honey-crusted lesions classic for impetigo; spreading; school contacts.",
     "red_flags": ["contagious impetigo", "topical/oral antibiotics"]},

    {"case_id": "P2C-T-Y02",
     "narrative": "Big red spreading patch on my calf since 2 days, painful, warm. No fever, no streaks, no pus. Cut from grass last week.",
     "image_description": "Lower-leg cellulitis without systemic signs in healthy adult.",
     "red_flags": ["cellulitis", "antibiotics + close follow-up"]},

    {"case_id": "P2C-T-Y03",
     "narrative": "Itchy ring-shaped red patches on my torso, growing slowly over 3 weeks. I have a cat at home. No fever.",
     "image_description": "Multiple annular scaling plaques with central clearing; tinea corporis.",
     "red_flags": ["dermatophyte infection", "topical antifungal"]},

    {"case_id": "P2C-T-Y04",
     "narrative": "My eczema patches have been spreading and getting darker, and now they crust and itch a lot more. I am 8 months pregnant.",
     "image_description": "Eczema flare with secondary infection signs in pregnancy; lichenified plaques with crusting.",
     "red_flags": ["pregnancy-safe management", "watch for systemic spread"]},

    {"case_id": "P2C-T-Y05",
     "narrative": "I had pneumonia last month, on antibiotic. Last 3 days mouth feels burning, white patches on tongue and inside cheeks I cannot wipe off.",
     "image_description": "Oral candidiasis (post-antibiotic), thrush plaques on buccal mucosa and tongue.",
     "red_flags": ["oral candidiasis", "antifungal lozenges, screen for immunosuppression"]},

    {"case_id": "P2C-T-Y06",
     "narrative": "Painful nodules along the side of my arm, like a line going up from a cat scratch on my hand 2 weeks ago. No fever.",
     "image_description": "Lymphangitic streaking with palpable nodules ('rosary' pattern); sporotrichosis-like or atypical mycobacterial.",
     "red_flags": ["nodular lymphangitis", "needs targeted antibiotic / antifungal"]},

    {"case_id": "P2C-T-Y07",
     "narrative": "Painful tooth on lower jaw for 1 week. Now jaw swelling on that side is firm, and small abscess forming under the chin. No high fever yet.",
     "image_description": "Submandibular cellulitis with early abscess from odontogenic source; risk of Ludwig.",
     "red_flags": ["dental infection spreading", "watch for airway compromise"]},

    {"case_id": "P2C-T-Y08",
     "narrative": "My right great toe nail is red, hot, and pus is at the corner. Painful for 4 days, now I cannot wear shoes.",
     "image_description": "Acute paronychia with purulence at lateral nail fold.",
     "red_flags": ["paronychia with abscess", "drain + topical/oral antibiotics"]},

    {"case_id": "P2C-T-Y09",
     "narrative": "Dark brown patches on my face since I had baby 6 months ago. Worse with sun. Not painful, not itchy. Embarrassing.",
     "image_description": "Symmetric facial hyperpigmentation in malar pattern; melasma post-pregnancy.",
     "red_flags": ["melasma", "sun protection + topical care"]},

    {"case_id": "P2C-T-Y10",
     "narrative": "I cracked the skin between my buttocks from sitting on tractor all day. Now small painful lump that drains pus sometimes.",
     "image_description": "Pilonidal sinus with intermittent purulent drainage.",
     "red_flags": ["pilonidal disease", "needs surgical evaluation, urgent if febrile"]},
]


# 10 GREEN training cases ----------------------------------------------------
GREEN_CASES = [
    {"case_id": "P2C-T-G01",
     "narrative": "My wedding ring left an itchy red mark on my finger after a day at the festival. I took the ring off, the mark is fading.",
     "image_description": "Annular contact dermatitis at ring site, resolving on removal.",
     "red_flags": []},

    {"case_id": "P2C-T-G02",
     "narrative": "I am 70. Brown spots on the back of my hands and forearms, more each year. Flat, painless, not itchy. Sun exposure all life.",
     "image_description": "Solar lentigines on dorsa of hands; benign age-related.",
     "red_flags": []},

    {"case_id": "P2C-T-G03",
     "narrative": "I was scratched by my son's toy 1 hour back, small graze on my shin. Bleeding stopped, cleaned with water. No spreading, no fever.",
     "image_description": "Superficial fresh abrasion, clean, no signs of infection.",
     "red_flags": []},

    {"case_id": "P2C-T-G04",
     "narrative": "Annoying dandruff for years. White flakes from scalp, no patches of bald, no redness on scalp. Worse when I miss oil bath.",
     "image_description": "Diffuse fine scalp scaling, mild seborrheic dermatitis.",
     "red_flags": []},

    {"case_id": "P2C-T-G05",
     "narrative": "I see a small soft lump on my baby's neck since 1 month, moves with skin, painless, not red. Baby is otherwise normal.",
     "image_description": "Small mobile non-tender subcutaneous lump in infant; likely lymph node or cyst, benign appearance.",
     "red_flags": []},

    {"case_id": "P2C-T-G06",
     "narrative": "Small skin tag in my armpit, soft, hangs by thin stalk. Sometimes catches on clothes. No bleeding, no growth in months.",
     "image_description": "Pedunculated soft fibroma in axilla.",
     "red_flags": []},

    {"case_id": "P2C-T-G07",
     "narrative": "I drank from a public glass and now my lip has a tiny cold sore tingling, came back like before. No fever.",
     "image_description": "Recurrent labial herpes simplex with prodromal tingling, single small vesicle.",
     "red_flags": []},

    {"case_id": "P2C-T-G08",
     "narrative": "I am pregnant 6 months. Itchy red lines on belly skin from stretching. No spread, baby moving fine, no fever.",
     "image_description": "Striae gravidarum with mild surrounding pruritus, no PUPPP papules, no jaundice.",
     "red_flags": []},

    {"case_id": "P2C-T-G09",
     "narrative": "Small painless lump on the back of my arm for many years, soft, moves under skin. I had it removed before, came back.",
     "image_description": "Recurrent benign-feeling subcutaneous lipoma.",
     "red_flags": []},

    {"case_id": "P2C-T-G10",
     "narrative": "Dry cracked lips, white tongue coating, mild discomfort while eating spicy. I ate too much fried food at festival yesterday.",
     "image_description": "Mild cheilitis and white tongue coating consistent with post-festival irritation.",
     "red_flags": []},
]


VILLAGE_CONTEXT = {
    "distance_to_clinic_km": 18,
    "clinic_open_hours": "10:00-13:00 weekdays, mobile dispensary Wednesday",
    "harvest_active": True,
    "nearest_hospital_km": 65,
    "transport_cost_round_trip_inr": 180,
    "patient_daily_wage_inr": 350,
}


def build_cases() -> list[dict]:
    out = []
    for urgency, group in (("red", RED_CASES), ("yellow", YELLOW_CASES), ("green", GREEN_CASES)):
        for c in group:
            case = dict(c)
            case["ground_truth_urgency"] = urgency
            case["village_context"] = dict(VILLAGE_CONTEXT)
            case["image_ref"] = f"data/images/{case['case_id']}.jpg"
            case.setdefault("adversarial_features", [])
            out.append(case)
    return out


def write(path: str = OUT) -> int:
    cases = build_cases()
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"wrote {len(cases)} training cases to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(write())
