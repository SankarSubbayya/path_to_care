# PROGRESS.md

Handoff log between sessions. Every session ends by updating this; every session starts by reading it.

## Done

**Harness wiring (this session)**
- cwc-long-running-agents harness wired: `.claude/hooks/`, `.claude/agents/evaluator.md`, `.claude/CLAUDE.md`, `.claude/settings.json`. Adapted `verify-gate.sh` and `track-read.sh` for our evidence patterns.
- `PROGRESS.md`, `test-results.json` (~20 features), `evidence/`, `results/`, `logs/` directories.
- `.gitignore` updated.
- Initial commit: `5dba9e8`.

**Phase 1 — Environment + smoke (4/5 features passing)**
- ✅ `env_torch_rocm` — torch 2.5.1+rocm6.2 sees the MI300X (gfx942, 192 GB). Sample matmul on cuda:0 OK. `evidence/env_torch_check.txt`.
- ✅ `env_packages_installed` — transformers 5.8.0, peft 0.19.1, accelerate 1.13.0, datasets 4.8.5, dspy 3.2.1, gradio 6.14.0, fastapi 0.136.1, etc. `evidence/env_pip_freeze.txt`.
- ✅ `qwen25_7b_smoke` — Qwen-2.5-7B-Instruct loaded in 7.4s, generated coherent chief-complaint in 11.2s. `evidence/qwen_smoke.txt`.
- ⏳ `gemma4_26b_smoke` — running in background (large download).
- ⏳ `repo_skeleton_built` — orchestrator + MCP modules being scaffolded now.

**Phase 2 — Harness + adversary (3/3 features passing — DONE)**
- ✅ `reward_fn_unit_tests` — 14/14 reward function tests pass. `evidence/reward_unit_tests.txt`.
- ✅ `adversary_generates_30` — `data/cases.jsonl` has 30 cases (10 R / 10 Y / 10 G), 25 with adversarial perturbations. Hand-crafted dermatology presentations + dialect/contradiction/off-distribution-image perturbations.
- ✅ `test_set_balanced_RYG` — distribution verified.

## In progress

- **Phase 1 — closing out:**
  - ✅ `gemma4_31b_smoke` — Gemma 4 31B-it loaded in 65s, generated top-3 with confidence in 3.7s. Output: "Cellulitis: 0.7 / Localized skin infection: 0.6 / Contact dermatitis: 0.3". `evidence/gemma4_smoke.txt`.
  - ✅ uv venv + pyproject.toml — pinned `torch==2.9.1+rocm6.3` via `[tool.uv.sources]`. `.venv/bin/python scripts/smoke_torch.py` PASS.
  - ⏳ `repo_skeleton_built` — core/, mcp/{image,soap,village,triage}, orchestrator/, training/, frontend/ in place; harness/run.py + orchestrator/agent.py + training/lora_sft.py still pending.

- **Phase 3 prep:** scaffold harness/run.py, orchestrator/agent.py.

## Next

- **Phase 3 — Zero-shot baseline** (hours 4–6). Wire orchestrator to call the 4 MCPs, run on `data/cases.jsonl`, write `results/baseline_metrics.json`. **This is the headline number.**
- **Phase 4 — Stub HF Space** (hour 6–7). Push a placeholder Gradio Space early to validate the deploy pipeline.
- **Phase 5 — LoRA SFT on Gemma 4 26B-A4B-it** (hours 7–9). Train on 20-25 of the 30 cases (hold 5–10 out for eval).
- **Phase 6 — Re-eval + delta** (hours 14–16). Headline: zero-shot vs. tuned on the held-out cases.
- **Phase 7 — Demo + README + BiP + ship** (hours 16–21).

See [CLAUDE.md](CLAUDE.md) for the full 24-hour table.

## Notes

### Architecture decisions (final)

- **Models — hybrid Gemma 4 + Qwen** (see [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md)):
  - Image-classifier MCP + Triage-reasoner MCP: `google/gemma-4-26B-A4B-it` (multimodal MoE, 26B total / ~4B active, Apache-2.0). Both MCPs share the loaded model.
  - SOAP-extractor MCP: `Qwen/Qwen2.5-7B-Instruct` (text-only). Preserves Qwen prize.
  - Fallback chain if 26B-A4B doesn't behave: 31B dense → E4B → Qwen2-VL-7B.

- **Pivots taken (documented for honesty):**
  - Gemma 3 12B-it → 401 gated. Confirmed live (`evidence/gemma_smoke.txt`).
  - Briefly considered single-vendor Qwen2-VL pivot before discovering Gemma 4 is open. Reverted to Gemma 4 hybrid.
  - User asked "what about gemma4" — discovered Gemma 4 family on HF Hub via `HfApi.list_models`, all variants `gated=False`, license=apache-2.0.

- **MCP servers as in-process modules (not FastAPI servers).** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) calls for FastAPI MCP servers; for the 24-hour build the 4 MCPs are Python modules the orchestrator imports directly. This lets all 4 share one loaded Gemma 4 model (~52 GB), avoids spinning up 4 separate processes, and cuts eval latency. The architecture is preserved; the IPC layer is deferred to v2.

- **GRPO/RLVR is out — LoRA SFT is in.** TRL+ROCm is too risky for 24h. GRPO loop *code* will ship in `training/grpo_stretch.py` as future work to preserve the RLVR narrative.

### Compatibility gotchas (verified live)

- `pip install --ignore-installed transformers …` clobbered ROCm torch with PyPI `torch 2.11.0` (CPU/CUDA). Recovery: `pip install --force-reinstall --index-url https://download.pytorch.org/whl/rocm6.2 torch==2.5.1 …`. **Lesson:** never `--ignore-installed` for packages that pull torch transitively.
- transformers 5.x: `torch_dtype=` deprecated → `dtype=`; `apply_chat_template(..., return_tensors='pt')` unreliable → use `tokenize=False` then tokenize. See [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md).

### Evidence patterns

`evidence/*.txt`, `results/*.json`, `logs/*.log`, `adapters/*/adapter_config.json`, `data/cases.jsonl` — these are what the verify-gate accepts as proof. See [.claude/CLAUDE.md](.claude/CLAUDE.md).

### Hardware

MI300X / 192 GB VRAM, ROCm 6.2.41133, gfx942. Plenty of headroom for hybrid (Gemma 4 26B + Qwen 7B + LoRA = ~77 GB).

### No prior assets on this box

All `/Users/sankar/...` paths in [docs/ASSETS.md](docs/ASSETS.md) are on a different machine. We are greenfield.
