# CLAUDE.md — Path to Care

Operational context for Claude Code working in this repo. Read this first; the rest of the docs ([docs/](docs/)) are detail. Slug: `path_to_care`.

---

## What this project is (60 seconds)

Multimodal, agentic decision-support for rural healthcare in the Global South. Runs on a phone. **Never diagnoses.** It (1) ranks plausible skin conditions from an image, (2) assigns urgency Red/Yellow/Green, (3) flags red signs, (4) contextualizes barriers (distance, cost, harvest season). Output: decision support for the patient + structured pre-visit SOAP for the clinic doctor.

Submission target: **AMD Developer Hackathon** — Tracks 1 (Agents), 2 (Fine-Tuning on MI300X), 3 (Multimodal), plus Qwen + HF + Build-in-Public prizes.

The brief, architecture, full plan, eval framework, and asset map are in [docs/](docs/). Do not duplicate them here — link, don't copy.

---

## Cardinal rule (non-negotiable)

**The system never produces diagnostic statements.** Always "signs suggest infection," never "you have cellulitis." Image output is **top-3 with confidence**, never a single class label, never binary sick/healthy. If you write code that violates this, fix it before moving on. See [docs/PROJECT.md](docs/PROJECT.md).

---

## Submission deadline: mid-afternoon 2026-05-10

This is a **~24-hour build**, not a 3- or 8-week build. The operative plan is the 24-hour table below. Time is the binding constraint — every decision is "does this ship by tomorrow afternoon."

**The 8-week plan in [docs/PLAN.md](docs/PLAN.md) is the post-hackathon v1→v2 roadmap**, not active scope. The submission README will reference it as the path from "24-hour proof-of-concept" to "field-deployable system" — village fieldwork, GRPO/RLVR, vision fine-tune, physician review of 20-30 outputs, skin-tone-stratified bias audit. This signals seriousness about the real version without trying to build it tomorrow.

### What we have on this machine

- **AMD Instinct MI300X / 192 GB VRAM**, ROCm, gfx942. ✓
- 235 GB RAM, 624 GB disk. ✓
- **No Python ML stack pre-installed.** First task is environment.
- **None of the assets in [docs/ASSETS.md](docs/ASSETS.md) are present.** Those paths (`/Users/sankar/...`) are on a different machine. We are building greenfield.

### What we are not building (explicit cuts)

- ❌ **Dedicated vision fine-tune** (separate ResNet/EfficientNet/ViT). Replaced by **Gemma 3 12B-it native multimodal** — image classifier MCP runs Gemma zero-shot, triage MCP runs Gemma after LoRA SFT. One model serves both. Standalone vision fine-tune deferred to v2.
- ❌ **GRPO/RLVR** as the primary fine-tune. TRL on ROCm is unverified; if it fails at hour 18, we have no headline number. **Use LoRA SFT** instead. Ship the GRPO loop *code* as future work to preserve the RLVR narrative.
- ❌ **Skin-tone stratified eval at scale.** Stratify by *condition* in the eval table; document the skin-tone gap as a known limitation needing Fitzpatrick-labeled data.
- ❌ **Tamil UX.** English only. Mention as roadmap.
- ❌ **Voice / TTS / STT.** Out. Text + image only.
- ❌ **vLLM serving.** Use raw `transformers` — slower but reliable. vLLM-on-ROCm is its own debugging session.
- ❌ **Real village fieldwork, physician review of 20-30 outputs.** Not feasible in 24h. Replace with one synthetic case study (the Rajan dialogue) and a written caveat.

### What we are building

| Component | Approach | Hits |
|---|---|---|
| Orchestrator | DSPy ReAct + hierarchical multi-agent | Track 1 |
| Image classifier MCP | **Gemma 4 31B-it** (vision, dense) — top-3 conditions + confidence | Track 3 |
| SOAP extractor MCP | **Qwen-2.5-7B-Instruct** via DSPy `NarrativeToSOAP` signature | Track 1, Qwen |
| Village context MCP | JSON knowledge file (transport, clinics, seasons) | — |
| Triage reasoner MCP | **Gemma 4 31B-it + LoRA SFT** on MI300X (image + SOAP + context → urgency) | Track 2, Track 3 |
| Adversarial agent | Generates 30-case held-out test set with red-flag/contradiction/off-distribution variants | safety / credibility |
| Eval harness | Reward `R = 1.0 / 0.5 / 0.0`, exact-match %, false-negative Red→Green % | safety |
| Demo | Gradio Space replaying the Rajan dialogue | HF prize |
| BiP | 1 X thread + 1 ROCm-feedback writeup | BiP prize |

**Hybrid model rationale (final):** Gemma 4 31B-it is multimodal, dense, **not gated** (Apache-2.0). Image-classifier MCP and triage-reasoner MCP **share one loaded Gemma 4 model** (two prompts against the same weights). Qwen-2.5-7B-Instruct handles text-only SOAP extraction, preserving the Qwen prize. Total VRAM: ~62 GB (Gemma 4) + 15 GB (Qwen 2.5) + ~12 GB LoRA overhead = ~89 GB on a 192 GB MI300X. **Pivot trail:** Gemma 3 12B-it (gated 401) → Gemma 4 26B-A4B-it (MoE, hits `torch._grouped_mm` ROCm-incompat) → **Gemma 4 31B-it (dense, works)**. Remaining fallbacks if 31B has a problem: Gemma 4 E4B-it (4B dense) → Qwen2-VL-7B-Instruct. See [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md).

### Headline narrative (target)

> *Path to Care: 24-hr build, MI300X, multimodal triage decision-support. Qwen-2.5-7B zero-shot vs. LoRA-tuned: +X% urgency accuracy, false-negative Red→Green falls from Y% to Z% on a 30-case adversarially-authored test set.*

The adversarial agent is load-bearing for credibility — it's the answer to "is this just memorization." It authors the held-out test set; the model never sees it during training.

---

## Hour-by-hour plan

| Hours | Task | Output |
|---|---|---|
| 0–2 | ROCm PyTorch + transformers/peft/dspy/gradio install; repo skeleton; **Gemma 3 12B-it + Qwen-2.5-7B** smoke load | env verified |
| 2–4 | Eval harness + adversarial generator + 30-case test set (HAM10000 images, public, no application form) | `harness/run.py`, `data/cases.jsonl` |
| 4–6 | Zero-shot baseline: orchestrator + 4 MCP stubs end-to-end (Gemma vision + Qwen SOAP + JSON village + Gemma triage). **Headline number on the board first.** | baseline metrics |
| 6–7 | Stub HF Space deployed early — confirm push pipeline works end-to-end | live URL |
| 7–9 | **LoRA SFT on Gemma 3 12B-it** with 20-30 (image+SOAP+context → urgency+reasoning) pairs, ~1 epoch | `adapters/triage-gemma-lora/` |
| 9–14 | **Sleep ~5 hrs**, LoRA training runs unattended | tuned weights |
| 14–16 | Re-run harness with tuned Gemma; compute delta; build report table | `docs/RESULTS.md` |
| 16–19 | Gradio replay of Rajan dialogue; push tuned adapter + Space to HF | working demo |
| 19–21 | README (with v2 roadmap pointer), architecture diagram, BiP post (tag `@AIatAMD` / `@lablab`), open-source push | submission-ready |
| 21–24 | **Buffer for disaster** | margin |

### Risks ranked

1. **ROCm PyTorch install fails or is slow.** Mitigate: fall back to CPU for harness (use Qwen2.5-1.5B-Instruct), keep MI300X for the LoRA run only.
2. **Qwen 7B inference is slow on raw transformers.** Mitigate: use 1.5B for the harness loop; reserve 7B for final eval pass + demo.
3. **LoRA OOMs or diverges.** Mitigate: small batch + gradient checkpointing; fall back to "DSPy BootstrapFewShot only" — keeps a delta number, loses the GPU-tune story.
4. **HF Space push fails at hour 22.** Mitigate: deploy a stub Space at hour 6 to prove the pipeline; replace contents at hour 18.

---

## Repo layout (target)

```
path_to_care/
├── CLAUDE.md                  # this file
├── README.md                  # public-facing, written at hour 19+
├── docs/                      # design docs (already populated)
├── harness/                   # eval harness — scoring, reward fn, runners
│   ├── run.py
│   ├── reward.py
│   └── metrics.py
├── adversary/                 # adversarial test-case generator
│   └── generate.py
├── data/
│   ├── cases.jsonl            # 30 held-out test cases (adversary-authored)
│   ├── train.jsonl            # 20-30 LoRA training cases
│   └── images/                # HAM10000 subset
├── mcp/                       # 4 MCP servers (FastAPI)
│   ├── image_classifier/      # zero-shot CLIP
│   ├── soap_extractor/        # DSPy NarrativeToSOAP
│   ├── village_context/       # JSON knowledge
│   └── triage_reasoner/       # Qwen-2.5-7B + LoRA
├── orchestrator/              # DSPy ReAct orchestrator
├── training/                  # LoRA SFT script + (stretch) GRPO loop
├── frontend/                  # Gradio app for HF Space
├── adapters/                  # saved LoRA weights
└── results/                   # eval outputs, reports
```

Prefer flat over deep. Don't create subpackages until there are 3+ files in them.

---

## Python environment (uv + venv)

This project uses **`uv`** with a **`pyproject.toml`** and a local **`.venv`** — not system Python. Reasons: reproducible builds, clean dep graph, ROCm wheels routed via `[tool.uv.sources]` to avoid the "PyPI torch clobbers ROCm torch" footgun we hit early in Phase 1.

```bash
# First-time setup (one-shot):
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv missing
uv sync                                            # creates .venv and installs from pyproject.toml

# Run anything Python:
.venv/bin/python scripts/smoke_torch.py            # explicit
# or
source .venv/bin/activate && python scripts/smoke_torch.py
```

Notes:
- `pyproject.toml` pins `torch==2.9.1+rocm6.3` and routes `torch`, `torchvision`, `torchaudio`, `pytorch-triton-rocm` through `[tool.uv.sources] pytorch-rocm` → `https://download.pytorch.org/whl/rocm6.3`. uv will not silently pull a CPU/CUDA wheel from PyPI.
- Python is pinned to `>=3.12,<3.13` because the ROCm wheels for `torch 2.9.1` are cp312-only.
- HF model weights live in `~/.cache/huggingface` (venv-independent — survives a `rm -rf .venv`).

## Working defaults

- **Models:** **Gemma 4 31B-it** (`google/gemma-4-31B-it`, multimodal dense — image+text → urgency, top-3 conditions). **Qwen-2.5-7B-Instruct** (`Qwen/Qwen2.5-7B-Instruct`, text-only — SOAP extraction). Remaining fallbacks: Gemma 4 E4B-it → Qwen2-VL-7B-Instruct.
- **Frameworks:** PyTorch (ROCm), `transformers`, `peft`, `dspy-ai`, `gradio`, `fastapi`. **Avoid vLLM and TRL/GRPO for now** — they're stretch goals, not load-bearing.
- **Image dataset:** HAM10000 (public, no application form). Stanford Skin Dataset is a reach goal.
- **Test set size:** 30 cases (10 each Red/Yellow/Green), authored by the adversarial generator.
- **Training set size:** 20-30 cases, disjoint from the test set.
- **Reward:** `R(predicted, ground_truth) = 1.0` exact / `0.5` adjacent / `0.0` off-by-2+. See [docs/EVALUATION.md](docs/EVALUATION.md).
- **Latency target:** <30s end-to-end. Don't optimize past this until everything else works.

## Conventions

- **Don't read every paper.** 2-3 dermatology review articles max if any. Time is the constraint, not knowledge.
- **Don't add features beyond the table above** without flagging the time cost. Scope creep is the #1 risk.
- **Cite file paths with line numbers** in all status updates: `mcp/triage_reasoner/server.py:42`.
- **Always run the harness** after a code change that touches model behavior. Don't trust eyeball checks for triage outputs.
- **Commit early, commit often.** If the box dies, what's on disk is what's submitted.
- **Open-source from day one.** This repo is the artifact. Don't put credentials, API keys, or private data anywhere in it.

## What "done" looks like

- [ ] HF Space with the Rajan dialogue replayable end-to-end.
- [ ] README with architecture diagram + eval table (zero-shot vs. tuned, with deltas).
- [ ] Open-source repo pushed to GitHub.
- [ ] LoRA adapter weights on HF Hub.
- [ ] 1 BiP post (X thread or LinkedIn) tagging `@AIatAMD` and `@lablab`.
- [ ] 1 ROCm/AMD-Dev-Cloud feedback writeup.
- [ ] lablab submission filed.

If any of these are missing at hour 23, that's the buffer hour's job. Don't add new features in the buffer.

---

## Inference paths

We have two live inference paths, used for different purposes. Don't mix them mid-eval.

1. **In-process `transformers` (v1 evidence path)** — [core/llm.py](core/llm.py). `gemma4()` and `qwen()` load via `transformers.AutoModelFor*`, hold weights in memory, called from MCPs and the orchestrator directly. **This is what produced `results/baseline_metrics.json` and `results/tuned_metrics.json`.** Switching engines mid-eval would change KV cache + sampling implementations and invalidate the before/after delta.

2. **vLLM in Docker (production serving path)** — [docs/VLLM_SERVE.md](docs/VLLM_SERVE.md). `vllm/vllm-openai-rocm:v0.20.1` image, OpenAI-compatible API on `:8000` with `--api-key ptc-demo-2026-amd`. Use [scripts/vllm_serve.sh](scripts/vllm_serve.sh) to start (`bf16` or `fp8` mode). Verified live: weight load 39 s, model resident 61.9 GB on MI300X, `/v1/chat/completions` returns clean responses. Why Docker not pip: the PyPI vLLM wheel is CUDA-built and clobbers ROCm torch; the `wheels.vllm.ai/rocm/0.20.1/rocm721` index ships ROCm-7.2.1-built torch which is incompatible with our 6.2 driver. Docker bundles its own ROCm user-mode libs.

**Switching:** a `core/llm_vllm.py` shim (planned, v2) will make the orchestrator engine-agnostic via `PTC_INFERENCE=vllm` env var. Until then, the eval path is in-process and the Gradio Space points at the vLLM server.

## Continuous-development harness

This repo runs the [cwc-long-running-agents](https://github.com/anthropics/cwc-long-running-agents) harness, adapted for Path-to-Care evidence patterns. The 24-hour build has a sleep window and likely context resets — the harness keeps state durable and prevents "I claim it's done" lies.

- [.claude/settings.json](.claude/settings.json) — hook wiring (PreToolUse + Stop)
- [.claude/CLAUDE.md](.claude/CLAUDE.md) — long-running workflow conventions (one feature at a time, evidence-before-passing)
- [.claude/hooks/verify-gate.sh](.claude/hooks/verify-gate.sh) — denies writes to `test-results.json` until evidence has been opened
- [.claude/hooks/track-read.sh](.claude/hooks/track-read.sh) — logs evidence opens; gate consults this
- [.claude/hooks/commit-on-stop.sh](.claude/hooks/commit-on-stop.sh) — auto-commit at session end so crashes don't lose work
- [.claude/hooks/kill-switch.sh](.claude/hooks/kill-switch.sh) — `touch AGENT_STOP` halts; `rm AGENT_STOP` resumes
- [.claude/hooks/steer.sh](.claude/hooks/steer.sh) — write to `STEER.md` to redirect mid-run
- [.claude/agents/evaluator.md](.claude/agents/evaluator.md) — fresh-context PASS/NEEDS_WORK reviewer, run at phase boundaries
- [PROGRESS.md](PROGRESS.md) — live handoff log; read this first on every session
- [test-results.json](test-results.json) — ~20 features by phase, all evidence-gated

## Pointers

- [docs/PROJECT.md](docs/PROJECT.md) — full project description and cardinal rule.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — system diagram, MCP server specs, conversation flow.
- [docs/PLAN.md](docs/PLAN.md) — 8-week roadmap, **reference only**. The 24-hour table above is operative.
- [docs/EVALUATION.md](docs/EVALUATION.md) — 4-dimension eval framework, reward function, reporting template.
- [docs/ASSETS.md](docs/ASSETS.md) — code reuse map (note: assets live on a different machine; we are greenfield here).
- [docs/TRACK_MAPPING.md](docs/TRACK_MAPPING.md) — how the project hits each prize track.
