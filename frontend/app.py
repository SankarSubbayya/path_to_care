"""Gradio app for the Path to Care HF Space.

Replays the Rajan dialogue (or any user-entered narrative + image) end-to-end
through the orchestrator. The user types a narrative + (optionally) attaches
an image; the app shows the four MCP outputs side-by-side: image top-3, SOAP
fields, village barriers, triage urgency + reasoning.

Run locally:
  .venv/bin/python -m frontend.app

For HF Space: this file is the Space's entry point (Gradio detects `demo`).
The Space runs the *same* code as local but you can pin a smaller model for
the demo (set PTC_GEMMA4_ID=google/gemma-4-E4B-it before launch) if Space
hardware is constrained.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr


# Cardinal-rule footer that goes below every triage output. Required by
# docs/PROJECT.md.
DISCLAIMER = (
    "**This is not a medical diagnosis.** This tool offers decision-support "
    "for community health workers and patients in low-resource settings. "
    "Final treatment decisions belong to a licensed clinician."
)


CANNED_RAJAN = (
    "I cut my foot on a rusty nail two days back when I was working in the "
    "field. Now my whole foot is swollen and red, the redness is going up my "
    "leg. I have fever since yesterday night, body shivering. Cannot keep "
    "weight on the foot."
)
CANNED_RAJAN_IMG_DESC = (
    "Lower leg with poorly demarcated erythema extending proximally above "
    "the wound; warmth and edema; small puncture wound visible on plantar foot."
)


def _format_top3(top3: list[dict]) -> str:
    if not top3:
        return "_(no candidates)_"
    return "\n".join(
        f"- **{c.get('condition', 'unknown')}** (confidence {float(c.get('confidence', 0.0)):.2f})"
        for c in top3
    )


def _format_soap(fields: dict) -> str:
    if not fields:
        return "_(SOAP extraction failed)_"
    out = []
    for k in ("chief_complaint", "hpi", "duration", "associated_symptoms",
              "past_medical_history", "medications", "vitals", "exam_findings",
              "red_flags", "patient_concerns"):
        if k in fields:
            out.append(f"**{k}:** {fields[k]}")
    return "\n\n".join(out)


def run(narrative: str, image_description: str, adapter_path: str) -> tuple[str, str, str, str, str]:
    """Run the orchestrator and return (top3_md, soap_md, village_md, triage_md, footer)."""
    # Lazy import: orchestrator pulls in torch + transformers.
    from orchestrator.agent import run_case

    case = {
        "case_id": "DEMO-RAJAN",
        "narrative": narrative,
        "image_description": image_description,
        "ground_truth_urgency": "unknown",  # demo; ground truth not used
        "image_ref": None,
        "village_context": {
            "distance_to_clinic_km": 18,
            "patient_daily_wage_inr": 350,
            "transport_cost_round_trip_inr": 180,
            "harvest_active": True,
        },
    }
    adapter = adapter_path.strip() or None
    trace = run_case(case, adapter_path=adapter)
    urgency_color = {"red": "🔴", "yellow": "🟡", "green": "🟢"}.get(trace.urgency, "⚪")
    triage_md = (
        f"## {urgency_color} **Urgency: {trace.urgency.upper()}**\n\n"
        f"### Reasoning\n{trace.reasoning}\n\n"
        f"### Red flags noted\n{', '.join(trace.red_flags_noted) if trace.red_flags_noted else '_none_'}\n\n"
        f"### Plain-language framing\n{trace.patient_framing}\n\n"
        + (f"_Safety net escalated this case from green to yellow_ " if trace.safety_escalation else "")
    )
    return (
        _format_top3(trace.image_top3),
        _format_soap(trace.soap_fields),
        trace.village_blurb,
        triage_md,
        DISCLAIMER,
    )


with gr.Blocks(title="Path to Care — Rural healthcare triage decision support") as demo:
    gr.Markdown(
        "# 🩺 Path to Care\n"
        "**Multimodal, agentic decision-support for rural healthcare.** "
        "The system **never diagnoses.** It (1) ranks plausible skin conditions, "
        "(2) assesses urgency Red / Yellow / Green, (3) flags red signs, and "
        "(4) frames barriers (distance, cost, harvest pressure).\n\n"
        "Built for the AMD Developer Hackathon, May 2026 — "
        "Gemma 4 31B-it (vision + triage, LoRA-tuned on MI300X) + Qwen-2.5-7B "
        "(SOAP extraction). See [GitHub](.) for the eval methodology and "
        "skin-tone bias caveats."
    )

    with gr.Row():
        with gr.Column():
            narrative = gr.Textbox(
                label="Patient narrative (typed)",
                value=CANNED_RAJAN,
                lines=6,
                placeholder="Describe the symptoms in the patient's own words...",
            )
            image_desc = gr.Textbox(
                label="What the image shows (text proxy for the photo)",
                value=CANNED_RAJAN_IMG_DESC,
                lines=3,
            )
            adapter = gr.Textbox(
                label="LoRA adapter path (optional — leave blank for zero-shot)",
                value="adapters/triage-gemma4-lora",
                placeholder="adapters/triage-gemma4-lora",
            )
            go = gr.Button("Run triage", variant="primary")

        with gr.Column():
            top3_out = gr.Markdown(label="Image classifier (top-3)")
            soap_out = gr.Markdown(label="SOAP fields")
            village_out = gr.Textbox(label="Village context", lines=3, interactive=False)
            triage_out = gr.Markdown(label="Triage")
            footer = gr.Markdown()

    go.click(
        fn=run,
        inputs=[narrative, image_desc, adapter],
        outputs=[top3_out, soap_out, village_out, triage_out, footer],
    )

    gr.Markdown(
        "---\n"
        "**Cardinal rule:** the system never produces diagnostic statements. "
        "Image output is always top-3 with confidence, never single-class. "
        "All patient-facing text passes through a code-level rewriter that "
        "replaces 'you have X' with 'signs suggest X'. See `core/cardinal_rule.py`."
    )


if __name__ == "__main__":
    demo.launch(server_name=os.environ.get("PTC_GRADIO_HOST", "0.0.0.0"),
                server_port=int(os.environ.get("PTC_GRADIO_PORT", "7860")))
