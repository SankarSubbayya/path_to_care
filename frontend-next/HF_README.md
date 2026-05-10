---
title: Path to Care (Next.js)
emoji: 🩺
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: Triage decision-support for rural healthcare (Next.js)
---

# 🩺 Path to Care — Next.js frontend

Multimodal, agentic decision-support for rural healthcare. The system **never diagnoses.**

This Space is the React/Next.js UI for [Path to Care](https://github.com/SankarSubbayya/path_to_care). It SSRs three views (Patient · Clinician · Audit) and posts triage requests to a vLLM endpoint hosted on an AMD Instinct MI300X running `vllm/vllm-openai-rocm:v0.20.1` with Gemma 4 31B-it.

The free Space hardware (`cpu-basic`) only runs the Gradio + OpenAI-client surface. All model inference happens on the MI300X.

## Configuration

The Dockerfile exposes:

- `PTC_VLLM_GEMMA4_URL` (default `http://165.245.137.117:8000/v1`)
- `PTC_VLLM_GEMMA4_MODEL_ID` (default `google/gemma-4-31B-it`)
- `PTC_VLLM_API_KEY` — **set as a Space *Secret* not Variable** (default ships the demo key).

Override these in Space → Settings → Variables and secrets.

## Compared to the Gradio Space

| | Gradio (`sankara68/path-to-care`) | Next.js (this Space) |
|---|---|---|
| UI | Gradio Blocks tabs | App Router · Tailwind · TS |
| Server | Python | Node 20 · Next 16.2 |
| API | n/a — same Python process | `/api/triage` — multipart→vLLM proxy |
| Source | `frontend/` | `frontend-next/` |

## Cardinal rule (enforced in code)

Every model output passes through a regex rewriter (`src/lib/cardinal-rule.ts`) that replaces diagnostic phrasing (`you have X` → `signs suggest X`). A rule-based safety net escalates `green` to `yellow` if ≥2 red-flag keywords are detected in the narrative, regardless of the model's call. **Never under-triage.**
