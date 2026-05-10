# 🩺 Path to Care

**Multimodal, agentic decision-support for rural healthcare.** Runs on a phone. **Never diagnoses.** Built in 24 hours for the [AMD Developer Hackathon (May 2026)](https://lablab.ai/event/amd-developer-hackathon).

> A patient in a Tamil Nadu village cuts his foot on a rusty nail. Two days later it's swollen, red, the redness is going up his leg, fever, shivering. The PHC is 18 km away. Round-trip transport costs ₹180; he earns ₹350 on a good day. Harvest is active.
>
> The system gets a phone photo + a typed narrative. It does **four** things, and only these four:
>
> 1. **Ranks plausible conditions** — top-3 with confidence (never single class).
> 2. **Assesses urgency** — Red (today) / Yellow (1-2 days) / Green (monitor at home).
> 3. **Flags red signs** — features that demand professional evaluation.
> 4. **Frames barriers** — distance, cost, harvest pressure; cost-benefit reframing for the patient.
>
> The clinician diagnoses. The patient decides. The system informs both.

## Demo

- 🤗 [HuggingFace Space (Next.js / React)](https://sankara68-path-to-care-react.hf.space/) — Patient / Clinician / Audit tabs, camera capture, voice dictation, MCP tool invocations panel.
- 📄 [docs/SUBMISSION_REPORT.md](docs/SUBMISSION_REPORT.md) — full submission report (architecture, evals, fine-tuning results, what worked / what didn't).

## Headline numbers

Held-out eval: 30 adversarially-authored test cases (10 R / 10 Y / 10 G; 25 with perturbations: dialect, contradicted narrative, off-distribution image, irrelevant context). Reward function from [docs/EVALUATION.md](docs/EVALUATION.md): `R = 1.0 exact / 0.5 adjacent / 0.0 off-by-2`.

| Metric                         | Zero-shot Gemma 4 31B | LoRA-tuned Gemma 4 31B | Δ |
|--------------------------------|-----------------------|------------------------|---|
| Mean reward                    | 0.983                 | 0.983                  | +0.0 |
| Exact-match urgency            | 96.7% (29/30)         | 96.7% (29/30)          | +0.0 |
| Within-1-level urgency         | 100.0%                | 100.0%                 | +0.0 |
| **FN Red→Green** (lower safer) | 0.0% (0/10 Red cases) | 0.0% (0/10 Red cases)  | +0.0 |

**Read the result honestly:** the zero-shot baseline is essentially at the ceiling of this hand-crafted 30-case test set. The LoRA fine-tune **does not regress** the baseline (no false negatives, no Red-cases-misclassified-as-Green), and demonstrates that **LoRA SFT on Gemma 4 31B-it converges in 32 seconds on a single MI300X** (loss 3.90 → 0.58, 45M trainable params, 0.14% of base). See [docs/RESULTS.md](docs/RESULTS.md) for the full confusion matrix.

### Real fine-tuning win — SCIN top-16 dermatology classification

The 30-case urgency test set was at ceiling, so we ran a second fine-tune on a real-world classification task: Google's [SCIN dataset](https://github.com/google-research-datasets/scin) (10 K consumer dermatology photos with weighted multi-condition labels and Fitzpatrick skin-type metadata). Restricted to the 16 most-occurring conditions; trained with top-k probability targets ("Eczema (0.41); Inflicted skin lesions (0.41); …") so the loss matches the SCIN paper's set-match metric, not single-class. Held-out 100-case eval:

| Metric                       | Zero-shot Gemma 4 31B | + SCIN top-16 LoRA   | Δ           |
|------------------------------|-----------------------|----------------------|-------------|
| Top-1 primary-condition acc  | 28.0%                 | **35.0%**            | **+7.0 pp** |
| Top-3 set-match (SCIN paper) | 71.0%                 | 68.0%                | −3.0 pp     |

**~239 training steps, single MI300X, ~38 min, r=8 LoRA (90 MB adapter).** First positive delta after a sequence of negatives that taught us the lesson: per-class sample count matters more than total epochs. Loss curve in [docs/figures/scin_top16_lora_loss.png](docs/figures/scin_top16_lora_loss.png); full write-up in [docs/SCIN_DIFF_DX.md](docs/SCIN_DIFF_DX.md) and [docs/SUBMISSION_REPORT.md](docs/SUBMISSION_REPORT.md). Adapter served live alongside base via vLLM `--enable-lora` (model id `scin-top16` on the droplet); in-process inference path in [scripts/infer_scin_top16.py](scripts/infer_scin_top16.py).

The "false-negative Red→Green" rate — predicting *Green* when ground-truth is *Red* — is the **cardinal safety metric**: under-triage in this context can mean a patient stays home with sepsis. We report it separately because aggregate accuracy hides it.

**Cardinal-rule rewriter fired live during eval** ([logs/cardinal_rule_rewrites.log](logs/cardinal_rule_rewrites.log)): on case Y09 the model emitted "you have a fever" in patient framing; the rewriter changed it to "signs suggest a fever" before the orchestrator returned. Evidence the safety net works under real model drift.

## Architecture

```
            Patient (phone camera + text)
                       │
                       ▼
              Frontend (Gradio)
                       │
                       ▼
        ┌── Orchestrator (DSPy-style) ──┐
        │              │              │              │
        ▼              ▼              ▼              ▼
  Image Classifier  SOAP Extractor  Village Context  Triage Reasoner
   (Gemma 4 31B)   (Qwen 2.5-7B)    (JSON KB)       (Gemma 4 31B + LoRA)
```

- **Camera Capture MCP** — browser-side `getUserMedia` + canvas snapshot in [`frontend-next/src/components/CameraCapture.tsx`](frontend-next/src/components/CameraCapture.tsx); server-side ingestion + audit save in [`mcp/camera_capture/server.py`](mcp/camera_capture/server.py). Frontend also includes Web-Speech-API voice dictation ([`VoiceInput.tsx`](frontend-next/src/components/VoiceInput.tsx)) so the patient can either type or speak the narrative. Both surface as discrete tool invocations in the audit tab.
- **Image Classifier MCP** — top-3 conditions + confidence. Never a single class. JSON-validated; falls back to "non-diagnostic / image insufficient" rather than guess.
- **SOAP Extractor MCP** — narrative → chief complaint, HPI, duration, associated symptoms (with explicit negations), red flags, patient concerns. Hand-engineered prompt; DSPy `BootstrapFewShot` is v2.
- **Village Context MCP** — synthetic Tamil-Nadu-composite knowledge file: PHC distance/hours/services, drug-stock map, transport options, household economics, seasonal calendar.
- **Triage Reasoner MCP** — fuses image-top-3 + SOAP + village barriers → `{urgency, reasoning, red_flags_noted, patient_framing}`. LoRA-tuned for the urgency-calibration delta. Cardinal-rule rewriter on every output.

The image classifier and triage reasoner share **one loaded Gemma 4 31B** model in memory — two prompts against the same weights — so VRAM stays at ~77 GB.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full diagram and conversation flow, and [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md) for the model-selection audit (Gemma 3 → gated; Gemma 4 26B-A4B MoE → ROCm-incompat; Gemma 4 31B-it → final).

## Cardinal rule (non-negotiable)

The system **never produces diagnostic statements.** Always "signs suggest", never "you have." Image output is **always top-3 with confidence**, never single-class, never binary sick/healthy.

Enforcement is in code, not just prompts. [`core/cardinal_rule.py`](core/cardinal_rule.py) regex-rewrites diagnostic phrases on every model output and logs rewrites to `logs/cardinal_rule_rewrites.log`.

## How to run

```bash
# 0. Setup (uv venv + pyproject.toml)
make setup

# 1. Verify ROCm + GPU
make smoke-torch

# 2. Verify both models load and generate
make smoke-models

# 3. Generate the 30-case adversarial test set
make build-cases

# 4. Run the zero-shot baseline (~7 min on MI300X)
make baseline

# 5. Build the LoRA training/holdout split
make build-train

# 6. LoRA SFT on Gemma 4 31B-it (~30 min)
make train

# 7. Run the tuned eval
make tuned

# 8. Headline delta report
.venv/bin/python scripts/build_delta_report.py

# 9. Launch the Gradio demo locally
make frontend
```

Hardware target: **AMD Instinct MI300X** (192 GB VRAM, ROCm 6.3). Should also run on a single 80 GB H100 with `PTC_GEMMA4_ID=google/gemma-4-E4B-it` set.

## Track / prize coverage

| Track / prize | How it's hit |
|---|---|
| **Track 1 — Agents** | Multi-agent (5 MCP modules + orchestrator); cardinal-rule code-level enforcement; safety-net cross-checks. |
| **Track 2 — Fine-tuning on AMD GPUs** | LoRA SFT on Gemma 4 31B-it on a single MI300X via ROCm 6.3 + peft 0.19. |
| **Track 3 — Vision & Multimodal** | Gemma 4 multimodal (image + text → urgency); two MCPs share weights. |
| **Qwen prize** | Qwen-2.5-7B-Instruct does the SOAP extraction — meaningful structural contribution to the pipeline. |
| **HuggingFace prize** | Gradio Space + LoRA adapter pushed to HF Hub. |
| **Build-in-Public** | [docs/BIP_POST.md](docs/BIP_POST.md) — X/LinkedIn thread + ROCm/AMD Dev Cloud feedback writeup. |

## Caveats (do not skip)

- **30 cases is small.** The brief calls for 80; v2 expansion in [docs/PLAN.md](docs/PLAN.md).
- **No skin-tone-stratified eval.** Stratification by Fitzpatrick scale needs HAM10000 metadata work that's v2 scope. Stratification here is by *condition* and urgency level only.
- **No real images yet.** The 24-hour build runs the multimodal model against `image_description` text strings rather than HAM10000 photos. Image-tensor inference is wired ([`core/llm.chat_multimodal(..., image=...)`](core/llm.py)) and is exercised in v2.
- **GRPO/RLVR is out of scope** for the 24-hour build. LoRA SFT is the primary fine-tune. The GRPO loop is sketched in [`training/grpo_stretch.py`](training/grpo_stretch.py) for v2.
- **Synthetic village knowledge.** [`mcp/village_context/knowledge.json`](mcp/village_context/knowledge.json) is a composite of typical Tamil Nadu rural logistics, not a specific village's data. v2 calls for real village fieldwork — see [docs/PLAN.md](docs/PLAN.md) week 1.

## v1 → v2 roadmap

The submission is the 24-hour proof-of-concept. The 8-week roadmap to a field-deployable version is in [docs/PLAN.md](docs/PLAN.md): real-village fieldwork, expanded test set (80 cases), GRPO/RLVR triage tuning, vision-from-scratch fine-tune, physician review of 20-30 outputs, skin-tone-stratified bias audit, Tamil-language UX validation, and a 2-3-real-user pilot.

## Repo layout

See [evidence/repo_tree.txt](evidence/repo_tree.txt) for the full file list, or:

```
.
├── CLAUDE.md            # operational context for AI agents working on this repo
├── PROGRESS.md          # session-to-session handoff log
├── pyproject.toml       # uv-managed; torch ROCm wheels via [tool.uv.sources]
├── test-results.json    # evidence-gated feature checklist (~20 features)
├── core/                # shared LLM loader, cardinal-rule rewriter
├── mcp/                 # 5 MCP modules (in-process for v1; FastAPI shells in v2)
│   ├── camera_capture/   # browser snapshot ingest + audit save
│   ├── image_classifier/
│   ├── soap_extractor/
│   ├── village_context/
│   └── triage_reasoner/
├── frontend-next/       # Next.js 16 + React 19 UI deployed to HF Space
├── orchestrator/        # DSPy-style coordinator
├── adversary/           # 30-case adversarial test-set generator
├── harness/             # eval runner, reward fn, metrics
├── training/            # LoRA SFT (+ GRPO skeleton)
├── frontend/            # Gradio Space app
├── docs/                # architecture, plan, eval, compatibility, bip post
└── .claude/             # long-running-agents harness (hooks + evaluator)
```

## License

Apache-2.0. See [LICENSE](LICENSE) (or pyproject.toml).

## Acknowledgements

- AMD for the AMD Developer Cloud and the MI300X compute.
- Google for the Gemma 4 family (Apache-2.0).
- Alibaba for the Qwen 2.5 family (Apache-2.0).
- Anthropic for the [`cwc-long-running-agents`](https://github.com/anthropics/cwc-long-running-agents) harness pattern that kept this 24-hour build coherent across context resets.
