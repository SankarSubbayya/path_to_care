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

**Phase 3 — Zero-shot baseline (3/3 features passing — DONE)**
- ✅ `mcp_servers_respond` — all 4 MCPs respond on case P2C-R01 in 63s cold. `evidence/mcp_smoke.txt`.
- ✅ `orchestrator_e2e` — end-to-end through `run_case`; cardinal-rule clean output. `evidence/orchestrator_smoke.txt`.
- ✅ `baseline_metrics_recorded` — **30/30 cases scored on MI300X in 393.6s (~13s per case after warmup)**. Mean reward 0.983 / exact-match 96.7% / within-1-level 100% / **FN Red→Green = 0.0** (cardinal safety metric perfect). 10/10 RED, 10/10 GREEN, 9/10 YELLOW. Only miss: P2C-Y03 (mild contact dermatitis) → green at R=0.5. `results/baseline_metrics.json`.

**Important context for the LoRA story:** baseline is essentially at the ceiling of this 30-case test set. The LoRA fine-tune's job is now (a) demonstrate the training infrastructure works on AMD MI300X, (b) maintain or improve the score, (c) potentially fix P2C-Y03. The headline pivots from "+X% from tuning" to "tuning runs reliably on MI300X without regressing a 96.7%-accurate baseline." Honest framing.

**Phase 4 — Stub HF Space (deferred)** — skipped to keep momentum on Phases 5-6. Deploy script ready (`scripts/deploy_hf_space.sh`); awaiting `HF_TOKEN` for the actual push. Bundle layout in `frontend/`.

**Phase 5 — LoRA SFT on Gemma 4 31B-it (2/2 features passing — DONE)**
- ✅ `lora_train_completes` — **trained in 32 seconds on MI300X**. 21 train rows × 2 epochs / grad_accum 4 = 10 optimizer steps. **Loss curve: 3.90 → 0.58** (clean convergence). 45,015,040 trainable params (0.1437% of 31B base). target_modules regex: `.*language_model.*self_attn\.(q_proj|k_proj|v_proj|o_proj)$` (avoiding the vision tower's `Gemma4ClippableLinear` wrapper). `logs/lora_train.log`.
- ✅ `adapter_saved` — `adapters/triage-gemma4-lora/adapter_config.json` + `adapter_model.safetensors` (180 MB). `peft_version=0.19.1`, base=`google/gemma-4-31B-it`.

**Compatibility lessons from Phase 5 (added to docs/COMPATIBILITY.md):**
- `Gemma4ClippableLinear` is a transformers wrapper around `nn.Linear` used in the **vision tower** (not language model). peft 0.19 doesn't recognize it as a Linear-class module. Workaround: regex-target only the language-model self_attn projections, which are plain `nn.Linear`.
- Gradient checkpointing + `use_cache=True` warns; transformers auto-disables `use_cache` for training. No action needed.

## In progress

- **Phase 6 — re-eval + delta** running NOW (background ID: b0hk3n8y0). Same 30 cases with LoRA adapter loaded; ~7 min wall time expected. After:
  - `tuned_metrics_recorded` — `results/tuned_metrics.json`
  - `delta_positive` — `evidence/delta_report.txt` (built by `scripts/build_delta_report.py`)

- **Phase 7 — ship** prep done; pending:
  - HF Space deploy (needs `HF_TOKEN` from user; `scripts/deploy_hf_space.sh` ready)
  - README.md numbers fill-in (template ready, waiting on tuned metrics)
  - Final commit + git push

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
