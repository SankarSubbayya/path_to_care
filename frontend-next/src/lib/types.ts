// Shared types between API route and UI.

export type Urgency = "red" | "yellow" | "green";

export interface ConditionGuess {
  condition: string;
  confidence: number; // 0..1
}

export interface SoapFields {
  chief_complaint?: string;
  hpi?: string;
  duration?: string;
  associated_symptoms?: string[];
  past_medical_history?: string[];
  medications?: string[];
  vitals?: Record<string, string | number>;
  exam_findings?: string[];
  red_flags?: string[];
  patient_concerns?: string[];
}

export interface VillageContext {
  phc_distance_km: number;
  phc_round_trip_cost_inr: number;
  phc_round_trip_as_pct_of_daily_wage: number;
  lost_wages_if_full_day_inr: number;
  harvest_active_today: boolean;
  district_hospital_km: number;
  blurb: string;
}

export interface ToolInvocation {
  name: string;
  ok: boolean;
  meta: Record<string, unknown>;
}

export interface TriageResult {
  image_top3: ConditionGuess[];
  soap: SoapFields;
  urgency: Urgency;
  reasoning: string;
  red_flags_noted: string[];
  patient_framing: string;
  village: VillageContext;
  // Diagnostics
  raw_model_output: string;
  parse_ok: boolean;
  cardinal_rule_rewrites: number;
  safety_escalation: boolean;
  cross_check_red_flags: string[];
  wall_seconds: number;
  tool_invocations?: ToolInvocation[];
}

export interface TriageError {
  error: string;
}
