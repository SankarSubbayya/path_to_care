"""Tests for the deterministic village context MCP — knowledge JSON lookups,
case-specific overrides, and the practical-urgency blurb."""
from __future__ import annotations

import pytest

from mcp.village_context.server import get_context, practical_urgency_blurb


def test_get_context_returns_expected_keys():
    ctx = get_context(case=None)
    expected_keys = {
        "phc_round_trip_cost_inr",
        "phc_round_trip_as_pct_of_daily_wage",
        "lost_wages_if_full_day_inr",
        "harvest_active_today",
        "phc_distance_km",
        "phc_open_now",
        "district_hospital_km",
        "drug_in_stock",
        "cultural_factors",
    }
    assert expected_keys.issubset(ctx.keys())


def test_get_context_honors_case_overrides():
    case = {
        "village_context": {
            "distance_to_clinic_km": 99,
            "patient_daily_wage_inr": 1234,
            "transport_cost_round_trip_inr": 555,
            "harvest_active": False,
        }
    }
    ctx = get_context(case=case)
    assert ctx["phc_distance_km"] == 99
    assert ctx["lost_wages_if_full_day_inr"] == 1234
    assert ctx["phc_round_trip_cost_inr"] == 555
    assert ctx["harvest_active_today"] is False


def test_round_trip_pct_calculation_is_accurate():
    case = {
        "village_context": {
            "distance_to_clinic_km": 18,
            "patient_daily_wage_inr": 1000,
            "transport_cost_round_trip_inr": 250,
            "harvest_active": False,
        }
    }
    ctx = get_context(case=case)
    assert ctx["phc_round_trip_as_pct_of_daily_wage"] == 25.0


def test_practical_urgency_blurb_mentions_distance_and_currency():
    blurb = practical_urgency_blurb(case=None)
    assert "km" in blurb.lower()
    assert "₹" in blurb or "INR" in blurb.upper() or "rupee" in blurb.lower()


def test_practical_urgency_blurb_reflects_harvest_state():
    case_harvest = {
        "village_context": {
            "distance_to_clinic_km": 18, "patient_daily_wage_inr": 350,
            "transport_cost_round_trip_inr": 180, "harvest_active": True,
        }
    }
    case_no_harvest = {
        "village_context": {
            "distance_to_clinic_km": 18, "patient_daily_wage_inr": 350,
            "transport_cost_round_trip_inr": 180, "harvest_active": False,
        }
    }
    assert "harvest" in practical_urgency_blurb(case_harvest).lower()
    assert "no harvest" in practical_urgency_blurb(case_no_harvest).lower()


def test_drug_in_stock_lookup_present():
    ctx = get_context(case=None)
    assert "amoxicillin_500mg_oral" in ctx["drug_in_stock"]
    # antivenom is documented as not in stock at PHC
    assert ctx["drug_in_stock"]["antivenom_polyvalent"] is False
