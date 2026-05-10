// Synthetic Tamil-Nadu-composite village context (mirrors
// mcp/village_context/knowledge.json defaults). For the demo we use the
// same numbers as the Python orchestrator so the live UI matches the eval.

import type { VillageContext } from "./types";

export const DEFAULT_VILLAGE: VillageContext = {
  phc_distance_km: 18,
  phc_round_trip_cost_inr: 180,
  phc_round_trip_as_pct_of_daily_wage: 51.4,
  lost_wages_if_full_day_inr: 350,
  harvest_active_today: true,
  district_hospital_km: 65,
  blurb:
    "PHC is 18 km away; round-trip transport is ₹180 (≈51.4% of a daily wage of ₹350). " +
    "District hospital is 65 km. Harvest is active — leaving the field costs visible income today. " +
    "Antibiotics typically free at PHC if in stock; antivenom only at district hospital.",
};
