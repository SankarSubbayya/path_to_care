# Compatibility & Risk Audit

Written 2026-05-09 as Phase 1 wraps. This document exists so future sessions don't re-discover compatibility gotchas the hard way. Updated whenever a compatibility decision is taken.

## Hardware ↔ runtime stack

| Layer | Version | Verified | Notes |
|---|---|---|---|
| GPU | AMD Instinct MI300X VF (gfx942), 192 GB HBM3 | ✓ `rocm-smi` | Single GPU |
| ROCm | 6.2.41133 | ✓ via `torch.version.hip` | Matches torch wheel |
| PyTorch | 2.5.1+rocm6.2 | ✓ smoke matmul on cuda:0 | `evidence/env_torch_check.txt` |
| Python | 3.12 (system) | ✓ | No venv (24h pragmatic) |

**Recovered failure (documented):** `pip install --ignore-installed transformers …` pulled in PyPI `torch 2.11.0` (CPU/CUDA), clobbering the ROCm 6.2 build. Recovery: `pip install --force-reinstall --index-url https://download.pytorch.org/whl/rocm6.2 torch==2.5.1 …`. Lesson: never `--ignore-installed` for packages that pull torch transitively.

## Python package matrix

| Package | Version | Why this version | Risk |
|---|---|---|---|
| torch | 2.5.1+rocm6.2 | Only ROCm 6.2 wheel available for AMD's index | none — pin |
| transformers | 5.8.0 | Pip resolved | **bleeding edge (5.x major)** — fall back to 4.50.x if model loads break |
| peft | 0.19.1 | Pip resolved | LoRA `target_modules` must be set per model architecture |
| accelerate | 1.13.0 | Pip resolved | none |
| datasets | 4.8.5 | Pip resolved | none |
| dspy | 3.2.1 | Pip resolved | LLM backend must be wired (litellm or custom dspy.LM) |
| gradio | 6.14.0 | Pip resolved | **bleeding edge (6.x major)** — Spaces may pin to 4.x or 5.x; verify before Phase 7 |
| fastapi / uvicorn | 0.136.1 / 0.46.0 | Pip resolved | none |
| open_clip_torch | 3.3.0 | Backup if multimodal Qwen plan fails | none |
| huggingface_hub | 1.14.0 | Pip resolved | **major-version 1.x** — token API may differ from 0.x; verify push_to_hub in Phase 7 |
| safetensors | 0.7.0 | Pip resolved | none |
| sentencepiece, protobuf, einops | latest | Pip resolved | none |

## Models — critical decision

**Original plan (CLAUDE.md):** Gemma 3 12B-it + Qwen-2.5-7B-Instruct hybrid.

**Discovery 1 — Gemma 3 is gated.** Smoke confirmed (`evidence/gemma_smoke.txt` → 401 OSError "gated repo"). We have no `HF_TOKEN` and the user told us not to ask.

**Discovery 2 — Gemma 4 exists, is open.** HF Hub query (`huggingface_hub.HfApi.list_models(author='google', search='gemma')`) returns the Gemma 4 family. Verified `gated=False`, `pipeline_tag=image-text-to-text`, `license=apache-2.0` on:

| Model | Type | Total params | Active params | bf16 VRAM | Use |
|---|---|---|---|---|---|
| `google/gemma-4-26B-A4B-it` | MoE | 26B | ~4B | ~52 GB | **primary** — fast inference + reasoning |
| `google/gemma-4-31B-it` | dense | 31B | 31B | ~62 GB | fallback if MoE LoRA breaks |
| `google/gemma-4-E4B-it` | dense | 4B | 4B | ~8 GB | fallback if VRAM/training time blow up |
| `google/gemma-4-E2B-it` | dense | 2B | 2B | ~4 GB | dev/CPU fallback |

**Discovery 3 — Gemma 4 26B-A4B (MoE) does NOT run on ROCm.** Smoke confirmed: model loads fine but `model.generate(...)` raises `RuntimeError: grouped gemm is not supported on ROCM` from `transformers/integrations/moe.py` calling `torch._grouped_mm`. The MoE forward path uses a CUDA-only kernel that has no ROCm equivalent in torch 2.9.1. **Implication:** any HF MoE model is currently incompatible with our stack. Pivot to dense.

**Discovery 4 — torch 2.5.1+rocm6.2 is too old for Gemma 4 modeling code.** Gemma 4's modeling file uses `or_mask_function`/`and_mask_function` introduced in torch 2.6. Upgraded to `torch 2.9.1+rocm6.3` from `https://download.pytorch.org/whl/rocm6.3`. ROCm 6.3 wheels work fine with our 6.2 driver runtime (the wheel bundles the 6.3 user-mode libs). Side benefit: Qwen 2.5 generation went 11.2s → 2.6s on the upgrade.

**Final architecture (hybrid restored, Gemma 4 dense not MoE):**

| Role | Model | License | VRAM (bf16) |
|---|---|---|---|
| Image classifier MCP (top-3 + confidence) | `google/gemma-4-31B-it` | Apache-2.0 | ~62 GB |
| Triage Reasoner MCP (image + SOAP + context → urgency) | **same** Gemma 4 31B-it (LoRA-tuned) | Apache-2.0 | shares weights |
| SOAP Extractor MCP (text → fields) | `Qwen/Qwen2.5-7B-Instruct` | Apache-2.0 | ~15 GB |
| Dev fallback | `Qwen/Qwen2.5-1.5B-Instruct` | Apache-2.0 | ~3 GB |

**Why share weights between image MCP and triage MCP:** two prompts against one loaded Gemma 4. Saves VRAM and load time.

**Total VRAM:** ~77 GB for inference. LoRA training overhead ~12 GB → still ~89 GB on a 192 GB MI300X. Comfortable margin.

**Why Gemma 4 31B-it (dense) is the pick:**
- 26B-A4B (MoE) blocked on ROCm — confirmed live.
- E4B (4B dense) would work but loses ~10× the parameter capacity for reasoning.
- 31B dense has known-good peft LoRA support (target_modules = `q_proj, k_proj, v_proj, o_proj` family).

**Why keep Qwen-2.5-7B for SOAP rather than letting Gemma 4 do everything:**
- Preserves Qwen prize eligibility (Qwen contributes meaningfully — full SOAP extraction).
- Hybrid keeps blast radius small: a Gemma 4 issue doesn't take down SOAP.
- Marginal VRAM cost (15 GB) is negligible on a 192 GB GPU.

## transformers 5.x API gotchas (verified live)

While running smoke tests, two breaking changes from 4.x → 5.x bit us:
- `torch_dtype=...` deprecated → use `dtype=...`
- `tokenizer.apply_chat_template(..., return_tensors='pt')` no longer reliable → call with `tokenize=False`, then tokenize separately

Fixed in [scripts/smoke_models.py](../scripts/smoke_models.py). Mention this in the BiP / ROCm-feedback writeup as a transformers 5.x ROCm-stack observation.

## vLLM on ROCm — Docker is the working path

**Final state (verified live):** Gemma 4 31B-it served by `vllm/vllm-openai-rocm:v0.20.1` Docker image, OpenAI-compatible API on `:8000` with `--api-key` auth. See [docs/VLLM_SERVE.md](VLLM_SERVE.md) for the operational playbook.

We hit two pip-install dead ends before landing on Docker:

1. **`uv pip install vllm`** (PyPI default) — resolves to a CUDA-built `vllm==0.20.1` whose dependency tree silently downgrades our `torch==2.9.1+rocm6.3` to a CPU `torch==2.11.0`. vLLM then fails: `OSError: libtorch_cuda.so: cannot open shared object file`.
2. **`uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/0.20.1/rocm721`** — the vLLM-published ROCm wheel index. Pulls a `torch==2.10.0+git8514f05` built against ROCm **7.2.1**. Our system driver is ROCm **6.2.x**. Torch import fails: `ImportError: undefined symbol: ncclCommShrink`. ROCm 7.2.1's RCCL exports a function our 6.2 driver's `librccl.so` doesn't. Driver mismatch.

**The working path:** the official `vllm/vllm-openai-rocm:v0.20.1` Docker image bundles its own ROCm user-mode libraries, isolated from the host's torch / driver mismatch. The image needs:

```
--group-add=video                  # /dev/dri group access
--cap-add=SYS_PTRACE               # ROCm RCCL debugger ops
--security-opt seccomp=unconfined  # ROCm syscalls Docker default-blocks
--device /dev/kfd                  # AMD Kernel Fusion Driver
--device /dev/dri                  # DRM render nodes
```

`--entrypoint /bin/bash` lets us pass our own `vllm serve …` invocation as the command, no image rebuild needed.

**Verified live (2026-05-09, 21:42 UTC):**
- Server up in ~90 s: weight load 39 s, model resident 61.9 GB, KV cache + cudagraph compile follows.
- `GET /v1/models` with `Authorization: Bearer ptc-demo-2026-amd` → `200 OK`. Without the header → `401`.
- Chat completion → "Plausible conditions include cellulitis, a localized skin infection, or an allergic reaction to a sting or bite." Cardinal-rule clean ("plausible conditions" not "you have").

**Concrete AMD product-feedback:** publish the ROCm vLLM wheel on `https://download.pytorch.org/whl/rocm6.3` so `uv pip install vllm` is sticky to the user's existing torch build, instead of silently replacing it with CPU. Every new ROCm vLLM user hits this. The Docker image works around it but adds an extra layer of indirection that not every workflow can accept.

**Why we keep the in-process `transformers` path for the eval numbers:** the v1 baseline + tuned eval runs through `core/llm.py`'s `transformers.generate`. Switching engines mid-eval would change KV-cache/sampling implementations and invalidate the before/after delta. vLLM is the **production serving path** (Gradio Space, post-submission demo, throughput); the eval is the credibility evidence. Both coexist via a planned `core/llm_vllm.py` swap layer. See [docs/VLLM_SERVE.md](VLLM_SERVE.md) "What this changes elsewhere."

## peft + Gemma 4: `Gemma4ClippableLinear` (verified live)

Gemma 4's **vision tower** wraps each projection in `Gemma4ClippableLinear` — a thin clipping wrapper around `torch.nn.Linear`. peft 0.19's `_create_new_module` validates the target is one of `(nn.Linear, nn.Embedding, ...)` and raises `ValueError: Target module Gemma4ClippableLinear(...) is not supported.`

The **language model** projections are plain `nn.Linear`, so the workaround is to regex-target only the language-model self-attention:

```python
LoraConfig(
    target_modules=r".*language_model.*self_attn\.(q_proj|k_proj|v_proj|o_proj)$",
    task_type="CAUSAL_LM",
    ...
)
```

Trade-off: we don't tune the vision tower. For triage-urgency calibration that's fine — the urgency decision is reasoning-driven once the vision encoder produces tokens. If we wanted to tune the vision tower we would need to teach peft to handle `Gemma4ClippableLinear` (subclass `LoraLayer` to wrap the inner `linear`). v2 work.

Loss curve from the tune (10 optimizer steps, 21 train rows, 2 epochs):
`3.90 → 3.27 → 2.78 → 2.21 → 1.67 → 1.48 → 1.00 → 1.11 → 0.83 → 0.58`. Clean convergence in 32 seconds on MI300X.

## Cardinal-rule enforcement (programmatic, not just prompt)

The cardinal rule (`docs/PROJECT.md`) says the system never produces diagnostic statements and the image classifier outputs top-3 with confidence. Prompts can drift. Therefore each MCP enforces the rule in code:

- **Image classifier MCP** validates that the model output parses to exactly 3 (condition, confidence) tuples with confidences in [0,1]. If parsing fails, return a fallback "low-confidence, please retake image" response, never a single class.
- **Triage Reasoner MCP** runs a regex post-filter: if the model output contains diagnostic phrases ("you have", "diagnosis is", "this is"), rewrite to "signs suggest" / "consistent with" before returning. Log rewrites to `logs/cardinal_rule_rewrites.log`.

These checks are part of the MCP code, not the model. They are tested in the eval harness.

## DSPy integration

DSPy needs an LLM backend. Options:

- **Local transformers via litellm proxy:** complex to wire in 24h.
- **Custom `dspy.LM`** that calls our Qwen2.5-7B via `transformers.pipeline`: 30 lines of code, robust.

**Decision:** custom `dspy.LM` wrapper for the SOAP extractor. The signature/optimizer story still holds — DSPy `BootstrapFewShot` works regardless of how the LM is called.

## HF Space deployment plan

**Reality:** free CPU Space can't run a 7B model in <60s. Free GPU Space (T4) is 16 GB — Qwen2-VL-7B in bf16 just fits but inference will be slow.

**Plan:**
- HF Space hosts the **Gradio UI** with a small Qwen-2.5-1.5B-Instruct model for live interaction. Demo is functional but not the full architecture.
- README publishes **real eval numbers** from the MI300X-hosted full stack (Qwen2-VL-7B + Qwen-2.5-7B + LoRA adapter).
- LoRA adapter weights pushed to HF Hub separately (`adapters/triage-qwen-vl-lora`) so reviewers can reproduce.
- Pre-recorded demo video shows the full Rajan dialogue against the MI300X-hosted version.

Alternative (if budget): HF Inference Endpoints. Out of scope unless explicit.

## Reproducibility checklist

- [ ] Test set is deterministic (`adversary/generate.py` is seedless but hand-coded — same output every run).
- [ ] Eval uses `do_sample=False`, `temperature=0`, fixed seed for any sampling step.
- [ ] LoRA training: fixed seed in `transformers.set_seed(42)` + `torch.manual_seed`.
- [ ] Model versions pinned in `requirements.txt` (Phase 7 deliverable).

## Sleep-window strategy

LoRA training is the only multi-hour task. It must:

1. Be one self-contained command (`python -m training.lora_sft`) that runs to completion without supervision.
2. Write progress to `logs/lora_train.log` continuously (so on resume we see how far it got).
3. Save checkpoints every N steps to `adapters/triage-qwen-vl-lora/checkpoint-*` so a crash doesn't lose all training.
4. Save final adapter to `adapters/triage-qwen-vl-lora/` (no checkpoint suffix) at end.

`commit-on-stop.sh` will commit the adapter_config.json and any new code, but **not** the .safetensors blobs (gitignored). Adapter weights are pushed to HF Hub instead, in Phase 7.

## Unverified items (to revisit before relying on them)

- transformers 5.8.0 actually loads Qwen2-VL — verify in Phase 1 smoke.
- peft 0.19.1 LoRA `target_modules` for Qwen2-VL — verify in Phase 5.
- gradio 6.14.0 deploys to HF Space without forcing a downgrade — verify in Phase 4 stub deploy.
- huggingface_hub 1.x `push_to_hub` API works as expected — verify in Phase 7.

## What this audit changes

1. **Model swap:** Gemma 3 12B-it → Qwen2-VL-7B-Instruct (vision + triage), single-vendor Qwen stack. Update [CLAUDE.md](../CLAUDE.md).
2. **Image classifier MCP shares weights with Triage MCP** (both call Qwen2-VL). Update orchestrator wiring.
3. **HF Space uses small Qwen-1.5B for live UI; real numbers come from MI300X.** Update [docs/PLAN.md](PLAN.md) "ship" phase.
4. **Cardinal rule has a code-level enforcer** in each MCP, tested in eval. Update MCP scaffolding.
5. **DSPy SOAP extractor wraps a custom `dspy.LM` calling local Qwen**, not a remote API. Update [orchestrator/](../orchestrator/).
