"""Gradio app for the Path to Care HF Space.

Two-tab layout: a **Patient view** (plain-language urgency, what-to-do, cost
framing) and a **Doctor view** (SOAP note + differential + red flags + the
patient narrative). Both views read from one orchestrator run.

Run locally:
  .venv/bin/python -m frontend.app

The Space deploy points `core.llm` at the MI300X-hosted vLLM container via
`PTC_INFERENCE=vllm` (set as a Space variable). The Space hardware tier
(`cpu-basic`) only runs Gradio + the OpenAI client — no models on the Space.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr


DISCLAIMER_PATIENT = (
    "**This is not a medical diagnosis.** This screen is decision-support — "
    "it suggests how soon to see a clinician and what to tell them. The "
    "doctor is the only one who can diagnose."
)

DISCLAIMER_DOCTOR = (
    "**Pre-visit decision-support.** Image-classifier output is *top-3 with "
    "confidence*, not a single class. All triage text passes through a "
    "code-level cardinal-rule rewriter (no diagnostic statements). See the "
    "[GitHub repo](https://github.com/SankarSubbayya/path_to_care) for the "
    "eval methodology, skin-tone bias caveats, and the v2 roadmap."
)


CANNED_RAJAN_NARRATIVE = (
    "I cut my foot on a rusty nail two days back when I was working in the "
    "field. Now my whole foot is swollen and red, the redness is going up my "
    "leg. I have fever since yesterday night, body shivering. Cannot keep "
    "weight on the foot."
)

CANNED_RAJAN_IMG = (
    "Lower leg with poorly demarcated erythema extending proximally above "
    "the wound; warmth and edema; small puncture wound visible on plantar foot."
)


URGENCY_VISUAL = {
    "red":    {"emoji": "🔴", "label": "RED — see a clinician TODAY",        "color": "#c0392b"},
    "yellow": {"emoji": "🟡", "label": "YELLOW — see a clinician in 1–2 days", "color": "#d39e00"},
    "green":  {"emoji": "🟢", "label": "GREEN — monitor at home",            "color": "#2e7d32"},
}


# ----- presentation helpers -------------------------------------------------

def _patient_view(trace) -> str:
    """Plain-language patient-facing summary as Markdown."""
    u = URGENCY_VISUAL.get(trace.urgency, URGENCY_VISUAL["green"])
    framing = trace.patient_framing.strip() or "(no specific guidance produced)"
    red_flag_md = ""
    if trace.red_flags_noted:
        items = "\n".join(f"- {rf}" for rf in trace.red_flags_noted[:5])
        red_flag_md = f"\n\n#### Watch for these signs\n{items}"
    safety_md = ""
    if getattr(trace, "safety_escalation", False):
        safety_md = (
            "\n\n> ⚠️ **Safety escalation.** The model first said *home care* but "
            "the rule-based check found multiple warning signs in your story. "
            "We recommend you see a clinic to be safe."
        )

    return (
        f"<div style='border-left:8px solid {u['color']};padding:12px 18px;"
        f"background:#fafafa;border-radius:6px;'>"
        f"<div style='font-size:1.6em;font-weight:600;color:{u['color']};'>"
        f"{u['emoji']} {u['label']}</div></div>\n\n"
        f"#### What to do\n{framing}"
        f"{red_flag_md}{safety_md}\n\n"
        f"---\n*{DISCLAIMER_PATIENT}*"
    )


def _doctor_view(trace) -> str:
    """Pre-visit SOAP-style summary for the clinician as Markdown."""
    u = URGENCY_VISUAL.get(trace.urgency, URGENCY_VISUAL["green"])

    top3_md = "\n".join(
        f"- **{c.get('condition','unknown')}** — confidence {float(c.get('confidence',0)):.2f}"
        for c in trace.image_top3
    ) or "*(image classifier produced no candidates)*"

    soap = trace.soap_fields or {}
    def _fmt(v):
        if isinstance(v, list): return ", ".join(map(str, v)) if v else "_none stated_"
        if isinstance(v, dict): return ", ".join(f"{k}={vv}" for k, vv in v.items()) if v else "_none stated_"
        return str(v) if v else "_none stated_"

    soap_lines = []
    for label, key in [
        ("Chief complaint",        "chief_complaint"),
        ("History of present illness", "hpi"),
        ("Duration",               "duration"),
        ("Associated symptoms",    "associated_symptoms"),
        ("Past medical history",   "past_medical_history"),
        ("Medications",            "medications"),
        ("Vitals",                 "vitals"),
        ("Exam findings",          "exam_findings"),
        ("Patient-reported red flags", "red_flags"),
        ("Patient concerns",       "patient_concerns"),
    ]:
        if key in soap:
            soap_lines.append(f"- **{label}:** {_fmt(soap[key])}")
    soap_md = "\n".join(soap_lines) if soap_lines else "*(SOAP extraction failed)*"

    rf_md = "\n".join(f"- {rf}" for rf in trace.red_flags_noted) or "_none flagged by triage reasoner_"
    cross_md = "\n".join(f"- {kw}" for kw in trace.cross_check_red_flags) or "_none_"

    safety_md = ""
    if getattr(trace, "safety_escalation", False):
        safety_md = (
            "\n\n> ⚠️ **Safety net escalated** the urgency from `green` → `yellow` "
            "because the rule-based cross-check found ≥2 red-flag keywords. "
            "Recommend confirming clinically rather than relying on the model alone."
        )

    return (
        f"### {u['emoji']} Triage urgency: **{trace.urgency.upper()}**{safety_md}\n\n"
        f"#### Top-3 differential (image classifier, top-3 + confidence — never single-class)\n{top3_md}\n\n"
        f"#### Pre-visit SOAP\n{soap_md}\n\n"
        f"#### Triage reasoning\n{trace.reasoning or '*(no reasoning)*'}\n\n"
        f"#### Red flags noted by triage reasoner\n{rf_md}\n\n"
        f"#### Rule-based cross-check (keyword scan of narrative + image description)\n{cross_md}\n\n"
        f"---\n{DISCLAIMER_DOCTOR}"
    )


def _village_view(trace) -> str:
    return (
        f"#### Practical-urgency context\n{trace.village_blurb}"
        if trace.village_blurb else "_no village context produced_"
    )


def _engineering_view(trace) -> str:
    """Tiny audit panel — folded behind a disclosure for transparency."""
    return (
        f"```\n"
        f"case_id:           {trace.case_id}\n"
        f"image_parse_ok:    {trace.image_parse_ok}\n"
        f"soap_parse_ok:     {trace.soap_parse_ok}\n"
        f"triage_parse_ok:   {trace.triage_parse_ok}\n"
        f"safety_escalation: {trace.safety_escalation}\n"
        f"```"
    )


# ----- orchestrator wrapping ------------------------------------------------

def run(narrative: str, image_description: str, adapter_path: str):
    """Single orchestrator pass. Returns:
       (patient_md, doctor_md, village_md, engineering_md)"""
    from orchestrator.agent import run_case

    case = {
        "case_id": "DEMO-RAJAN",
        "narrative": narrative,
        "image_description": image_description,
        "ground_truth_urgency": "unknown",
        "image_ref": None,
        "village_context": {
            "distance_to_clinic_km": 18,
            "patient_daily_wage_inr": 350,
            "transport_cost_round_trip_inr": 180,
            "harvest_active": True,
        },
    }
    adapter = (adapter_path or "").strip() or None
    trace = run_case(case, adapter_path=adapter)
    return (
        _patient_view(trace),
        _doctor_view(trace),
        _village_view(trace),
        _engineering_view(trace),
    )


# ----- UI --------------------------------------------------------------------

INTRO = """
# 🩺 Path to Care
**Multimodal, agentic decision-support for rural healthcare.** The system **never diagnoses.**
It (1) ranks plausible skin conditions, (2) assesses urgency Red / Yellow / Green,
(3) flags red signs, and (4) frames barriers (distance, cost, harvest pressure)
as cost-benefit framing for the patient.

Built for the AMD Developer Hackathon, May 2026 — Gemma 4 31B-it (vision + triage,
LoRA-tuned on MI300X) + Qwen-2.5-7B (SOAP extraction). See [GitHub](https://github.com/SankarSubbayya/path_to_care)
for the eval methodology and skin-tone bias caveats.

The example below is the **Rajan dialogue** — a Tamil Nadu farmer, foot wound, fever.
Edit it or paste your own narrative + image description, then click **Run triage**.
"""


with gr.Blocks(title="Path to Care — Rural healthcare triage decision support") as demo:
    gr.Markdown(INTRO)

    with gr.Row():
        with gr.Column(scale=2):
            narrative = gr.Textbox(
                label="Patient narrative (typed)",
                value=CANNED_RAJAN_NARRATIVE,
                lines=6,
            )
            image_desc = gr.Textbox(
                label="What the image shows (text proxy for the photo)",
                value=CANNED_RAJAN_IMG,
                lines=3,
            )
            adapter = gr.Textbox(
                label="LoRA adapter path (advanced — leave blank for zero-shot)",
                value="",
                placeholder="adapters/triage-gemma4-lora",
            )
            go = gr.Button("Run triage", variant="primary", size="lg")

        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab("👤 For the Patient"):
                    patient_out = gr.Markdown()
                    village_out = gr.Markdown()
                with gr.Tab("🩺 For the Clinician"):
                    doctor_out = gr.Markdown()
                with gr.Tab("⚙️ Audit trail"):
                    engineering_out = gr.Markdown()
                    gr.Markdown(
                        "Parser flags + safety-net status. Used for debugging "
                        "and for the cardinal-rule audit (logs go to "
                        "`logs/cardinal_rule_rewrites.log`)."
                    )

    go.click(
        fn=run,
        inputs=[narrative, image_desc, adapter],
        outputs=[patient_out, doctor_out, village_out, engineering_out],
    )

    gr.Markdown(
        "---\n"
        "**Cardinal rule:** the system never produces diagnostic statements. "
        "Image output is always top-3 with confidence, never single-class. "
        "All patient-facing text passes through a regex rewriter that "
        "replaces \"you have X\" with \"signs suggest X\". See "
        "[`core/cardinal_rule.py`](https://github.com/SankarSubbayya/path_to_care/blob/main/core/cardinal_rule.py)."
    )


if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("PTC_GRADIO_HOST", "0.0.0.0"),
        server_port=int(os.environ.get("PTC_GRADIO_PORT", "7860")),
    )
