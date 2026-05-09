# Serving Gemma 4 with vLLM on MI300X (Docker)

Operational playbook for the **production inference path** on AMD ROCm. Pairs with [docs/COMPATIBILITY.md](COMPATIBILITY.md) (why we route this through Docker rather than installing vLLM in the venv) and [SUBMISSION.md](../SUBMISSION.md) (how this fits into the deploy flow).

## Why Docker (not pip install)

Tried `pip install vllm` and `pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/0.20.1/rocm721`:

1. **PyPI `vllm==0.20.1`** is CUDA-built — pulls a CPU/CUDA `torch==2.11.0` and silently downgrades the ROCm `torch==2.9.1+rocm6.3` we depend on for harness/training. vLLM then fails: `OSError: libtorch_cuda.so: cannot open shared object file`.
2. **The `wheels.vllm.ai/rocm/0.20.1/rocm721` index** ships `torch==2.10.0+git8514f05` built against ROCm **7.2.1**. Our system driver is ROCm **6.2.x**. Torch import fails with `ImportError: undefined symbol: ncclCommShrink` because `librccl.so` from 6.2.x doesn't export the 7.2.1 NCCL/RCCL API.

**Conclusion:** the only safe path on a ROCm 6.2 driver is the official Docker image, which bundles its own user-mode ROCm libs (independent of the host driver, as long as the host KFD driver speaks the right ABI). We use `vllm/vllm-openai-rocm:v0.20.1` (≈31 GB pull) per AMD's MI300X recipe.

This keeps **vLLM in Docker, the venv pure for the eval/training stack** — no torch reinstall churn.

## Run command (verbatim)

Wrapped in [scripts/vllm_serve.sh](../scripts/vllm_serve.sh). Reproduces the `docker run` we used.

```bash
docker run -d --name ptc-vllm \
  --group-add=video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device /dev/kfd \
  --device /dev/dri \
  -p 8000:8000 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -e HF_HOME=/root/.cache/huggingface \
  --entrypoint /bin/bash \
  vllm/vllm-openai-rocm:v0.20.1 \
  -c 'vllm serve google/gemma-4-31B-it \
        --host 0.0.0.0 --port 8000 \
        --api-key ptc-demo-2026-amd \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.9 \
        --dtype bfloat16'
```

**Why each flag:**

| Flag | Why |
|---|---|
| `-d --name ptc-vllm` | detached + named so subsequent `docker logs ptc-vllm` / `docker rm -f ptc-vllm` work |
| `--group-add=video` | grants the in-container user access to `/dev/dri` render-node group |
| `--cap-add=SYS_PTRACE` | required for some ROCm RCCL / debugger ops |
| `--security-opt seccomp=unconfined` | ROCm libs use syscalls Docker's default seccomp profile blocks |
| `--device /dev/kfd` | AMD Kernel Fusion Driver — the GPU compute device |
| `--device /dev/dri` | DRM render nodes (multiple GPUs, even though we have 1) |
| `-p 8000:8000` | OpenAI-compatible API port |
| `-v /root/.cache/huggingface:...` | re-uses the model weights we already downloaded for the venv path; saves a re-download |
| `-e HF_HOME=...` | tells transformers inside the container where the cache is |
| `--entrypoint /bin/bash` | overrides the image's default entrypoint so we can pass our own `vllm serve …` invocation as `bash -c '...'`. Lets us tweak args without rebuilding the image |

**vLLM serve flags:**

| Flag | Value | Why |
|---|---|---|
| `--api-key` | `ptc-demo-2026-amd` | requires `Authorization: Bearer …` on every request. Verified `401` without it. |
| `--max-model-len` | `8192` | enough for our prompts (~3000 token triage prompts + responses); larger ⇒ bigger KV cache footprint |
| `--gpu-memory-utilization` | `0.9` | reserves ~173 GB of MI300X's 192 GB. Drop to ≤0.5 to leave the GPU available for parallel work; see "GPU memory tuning" below |
| `--dtype` | `bfloat16` | Gemma 4 31B is bf16-native; FP8 is the next optimization (see below) |

vLLM auto-resolved a few decisions in our run:
- `Resolved architecture: Gemma4ForConditionalGeneration`
- Forced `TRITON_ATTN` backend because Gemma 4 has heterogeneous head dimensions (`head_dim=256, global_head_dim=512`) — a cuBLAS/MFMA-mixed path would have caused numerical divergence.
- Disabled `torch.compile`'s native GELU-tanh approximation on ROCm (instability noted by vLLM — falls back to `none` approximation).

## Connection details

```
Base URL:   http://<host>:8000/v1
API key:    ptc-demo-2026-amd
Model:      google/gemma-4-31B-it
```

OpenAI-compatible. Works with `openai`, `litellm`, `dspy.LM(...)`, anything.

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="ptc-demo-2026-amd")
resp = client.chat.completions.create(
    model="google/gemma-4-31B-it",
    messages=[{"role":"user","content":"signs of cellulitis"}],
    max_tokens=128, temperature=0,
)
print(resp.choices[0].message.content)
```

Verified live (2026-05-09 21:42 UTC):
- `GET /v1/models` with key ⇒ `200 OK` listing `google/gemma-4-31B-it`
- `GET /v1/models` without key ⇒ `401`
- Chat completion: `"Plausible conditions include cellulitis, a localized skin infection, or an allergic reaction to a sting or bite."` — cardinal-rule clean ("plausible conditions" not "you have"), 24 output tokens.

## GPU memory tuning

The bf16 baseline pinned 61.9 GB for weights + KV cache. To leave room for other work on the GPU:

1. **Lower `--gpu-memory-utilization`** to e.g. `0.4`. vLLM caps total reservation at 40% (~77 GB) instead of 90% (~173 GB). Cost: smaller KV cache, fewer concurrent requests.
2. **Lower `--max-model-len`** to e.g. `4096`. Cuts KV cache by half. Fine for our triage prompts (~3000 tokens).
3. **Switch to FP8 weights.** Cuts the model's resident memory roughly in half (~31 GB instead of ~62 GB). Two routes:
   - vLLM **online quantization** with `--quantization fp8` (vLLM down-converts bf16 weights at load).
   - **Pre-quantized model** if available on the Hub (`RedHatAI/Gemma-4-31B-it-FP8`, `neuralmagic/...-FP8`, etc.) — load directly with `--model <fp8-id>`. Faster startup, sometimes better accuracy than online quantization.

Search the Hub before starting:

```bash
.venv/bin/python -c "
from huggingface_hub import HfApi
api = HfApi()
hits = list(api.list_models(search='gemma-4-31B FP8', limit=10))
for m in hits: print(m.id)
"
```

Recommended low-mem profile (when ready):

```bash
docker run -d --name ptc-vllm-fp8 \
  --group-add=video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --device /dev/kfd --device /dev/dri \
  -p 8000:8000 -v /root/.cache/huggingface:/root/.cache/huggingface \
  -e HF_HOME=/root/.cache/huggingface \
  --entrypoint /bin/bash \
  vllm/vllm-openai-rocm:v0.20.1 \
  -c 'vllm serve google/gemma-4-31B-it \
        --host 0.0.0.0 --port 8000 \
        --api-key ptc-demo-2026-amd \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.4 \
        --quantization fp8 \
        --dtype bfloat16'
```

(or replace `google/gemma-4-31B-it` with a pre-quantized FP8 repo id and drop `--quantization fp8`).

Expected footprint: ~35–45 GB resident, freeing ~140 GB on the MI300X for parallel work.

## Lifecycle

```bash
# Inspect logs (live)
docker logs -f ptc-vllm

# Stop & remove
docker rm -f ptc-vllm

# Restart fresh with a different config
docker run -d --name ptc-vllm  ...new args...

# Resource check
rocm-smi --showmeminfo vram
```

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `401 Unauthorized` | missing or wrong `Authorization: Bearer …` | use the `--api-key` value |
| `404 model not found` | request body's `model` field doesn't match what vLLM serves | use the exact `id` from `GET /v1/models` |
| `Application startup` never appears, container alive >5 min | KV cache OOM at warmup, or cudagraph capture stuck | check `docker logs` for `OutOfMemoryError`; lower `--max-model-len` and/or `--gpu-memory-utilization` |
| `libtorch_cuda.so: cannot open shared object file` | tried to install vLLM via pip, not Docker | use Docker — see top of doc |
| `undefined symbol: ncclCommShrink` | tried `wheels.vllm.ai/rocm/0.20.1/rocm721` against a ROCm 6.x driver | use Docker |

## What this changes elsewhere in the project

- **[harness/run.py](../harness/run.py) and [orchestrator/agent.py](../orchestrator/agent.py)** currently call in-process `transformers.generate` via `core/llm.py`. A v2 swap-in is to add `core/llm_vllm.py` that talks to `http://localhost:8000/v1` instead — same `chat_text`/`chat_multimodal` signature, OpenAI-API call inside. Single env var (`PTC_INFERENCE=vllm` vs `transformers`) flips the path. Not done for v1 because the eval ran with the in-process path; switching backends mid-eval would invalidate baseline-vs-tuned comparability.
- **[frontend/app.py](../frontend/app.py)** can be pointed at the vLLM endpoint for the Space demo (Space hardware tier wouldn't run a 31B in process; pointing at a remote vLLM server is the practical demo path).
- **The LoRA adapter** (`adapters/triage-gemma4-lora/`) needs a vLLM-aware loader. vLLM supports LoRA via `--enable-lora --lora-modules triage=adapters/triage-gemma4-lora`; can be added when we wire the vLLM path through the orchestrator.

## Why we're keeping in-process transformers as the v1 path

The 24-hour submission's headline numbers (`results/baseline_metrics.json`, `results/tuned_metrics.json`) are produced by the in-process path. Re-running them through vLLM would change the inference engine — different KV-cache, different sampling implementation, possibly different numerical results. For a credible "before vs after" delta we keep one engine across both runs.

vLLM is the **production-grade serving path** for the demo and post-submission usage; it's not in the v1 critical path for the evidence numbers. Both paths coexist cleanly — see the env-var switch idea above.
