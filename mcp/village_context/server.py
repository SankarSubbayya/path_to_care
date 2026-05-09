"""Village Context MCP: practical-urgency layer.

Reads the village knowledge JSON and returns the structured barriers and
cost-benefit data the triage reasoner needs to convert clinical urgency into
*practical* urgency. No LLM. Deterministic.

The schema mirrors the CareGraph structure described in docs/ARCHITECTURE.md
(Village → Clinics → Drugs, Transport, Seasonal calendar) but is a flat JSON
file rather than a Neo4j graph for the 24-hour build.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge.json")


@lru_cache(maxsize=1)
def _load() -> dict:
    with open(KB_PATH) as f:
        return json.load(f)


def get_context(case: dict | None = None) -> dict[str, Any]:
    """Return barriers + cost-benefit framing data for the triage reasoner.

    `case` may carry case-specific overrides (e.g., the case's own
    village_context field) — used during eval where the test set fixes
    distance/wage values per case for reproducibility. If `case` is None
    the village's static defaults are returned.
    """
    kb = _load()

    distance_km = case["village_context"]["distance_to_clinic_km"] if case and "village_context" in case else kb["primary_health_centre"]["distance_km"]
    daily_wage = case["village_context"]["patient_daily_wage_inr"] if case and "village_context" in case else kb["household_economics"]["patient_daily_wage_inr"]
    round_trip = case["village_context"]["transport_cost_round_trip_inr"] if case and "village_context" in case else kb["household_economics"]["average_round_trip_to_phc_inr"]
    harvest_active = case["village_context"]["harvest_active"] if case and "village_context" in case else (kb["seasonal_calendar"]["current_month"] in kb["seasonal_calendar"]["harvest_active_months"])

    # Practical-urgency framing strings the triage reasoner can compose into
    # patient-facing reasoning. Stay descriptive, not prescriptive.
    framing = {
        "phc_round_trip_cost_inr": round_trip,
        "phc_round_trip_as_pct_of_daily_wage": round(100.0 * round_trip / max(daily_wage, 1), 1),
        "lost_wages_if_full_day_inr": daily_wage,
        "harvest_active_today": harvest_active,
        "phc_distance_km": distance_km,
        "phc_open_now": _phc_open_now(kb),
        "district_hospital_km": kb["district_hospital"]["distance_km"],
        "drug_in_stock": {
            k: v["in_stock"] for k, v in kb["drug_availability_phc"].items()
        },
        "cultural_factors": kb["cultural_factors"],
    }
    return framing


def _phc_open_now(kb: dict) -> bool:
    # Naive: the synthetic KB doesn't carry a clock; assume PHC open during
    # weekday daytime. Eval cases don't depend on this; demo can override.
    return True


def practical_urgency_blurb(case: dict | None = None) -> str:
    """One-paragraph plain-language framing of barriers, for the triage prompt."""
    f = get_context(case)
    return (
        f"PHC is {f['phc_distance_km']} km away; round-trip transport is "
        f"₹{f['phc_round_trip_cost_inr']} (≈{f['phc_round_trip_as_pct_of_daily_wage']}% of a "
        f"daily wage of ₹{f['lost_wages_if_full_day_inr']}). "
        f"District hospital is {f['district_hospital_km']} km. "
        f"{'Harvest is active — leaving the field costs visible income today.' if f['harvest_active_today'] else 'No harvest pressure today.'} "
        f"Antibiotics typically free at PHC if in stock; antivenom only at district hospital."
    )
