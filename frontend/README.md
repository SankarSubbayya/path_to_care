---
title: Path to Care
emoji: 🩺
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: "6.14"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Multimodal triage decision-support for rural healthcare. Never diagnoses.
---

# Path to Care

Multimodal, agentic decision-support system for rural healthcare. The system **never diagnoses**. Built for the AMD Developer Hackathon (May 2026).

The Space replays the Rajan dialogue: phone-photo description + typed narrative → top-3 conditions, structured SOAP, urgency Red/Yellow/Green with reasoning + cost-benefit framing.

**Source repo:** [github.com/sankara68/path-to-care](https://github.com/sankara68/path-to-care)
**Architecture:** Gemma 4 31B-it (vision + triage, LoRA on MI300X) + Qwen-2.5-7B (SOAP).
**Cardinal rule:** image → top-3 + confidence (never single-class). Triage outputs: "signs suggest" not "you have".

> ⚠️ This Space is the demo UI. The full eval (zero-shot vs LoRA-tuned, FN Red→Green = 0%) was run on a single AMD Instinct MI300X. See the source repo `docs/RESULTS.md` for numbers.

## Hardware note

If this Space is configured with a small CPU backend, it will set
`PTC_GEMMA4_ID=google/gemma-4-E4B-it` (4B dense fallback) for usable demo
latency. Real eval numbers come from the 31B model on MI300X — see the repo.
