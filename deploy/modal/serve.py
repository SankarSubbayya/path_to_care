"""Modal serve for Path to Care — Gemma 4 31B-it + SCIN top-16 LoRA, OpenAI-compat.

This is the **fallback** inference path. Primary is the AMD MI300X droplet running
the same model via `vllm/vllm-openai-rocm:v0.20.1` Docker — see docs/VLLM_SERVE.md.
We keep AMD as the canonical path (the +7.0 pp SCIN top-16 result was produced on
it). Modal exists so a `PTC_VLLM_GEMMA4_URL` env-var swap on the HF Space flips
inference to NVIDIA H100 within ~60 s if the droplet hiccups during demo.

Architecture:
  - Modal serverless GPU — A100-80GB by default (see GPU constant below).
    bf16 Gemma 4 31B ≈ 62 GB resident; A100-80GB is the cheapest 80 GB
    Modal option and fits the same as H100 at roughly half the hourly rate.
    Upgrade to "H100" for ~2× faster latency if budget allows.
  - vLLM 0.6.x CUDA-built, `--enable-lora`, multimodal Gemma 4 vision tower included
  - LoRA mounted at startup from huggingface.co/sankara68/path-to-care-scin-top16-lora
  - Exposes OpenAI-compat endpoint at /v1/chat/completions and /v1/models
  - Authorization: Bearer ptc-demo-2026-amd (matches the droplet's key so swaps are zero-touch)

Deploy:
  # IMPORTANT: do NOT `uv add modal` into this project — Modal's protobuf<7
  # conflicts with our protobuf>=7 (transformers 5.x / gradio 5.x transitively).
  # Run Modal in an isolated env via uvx; the local CLI is just a deploy
  # client, real work happens in Modal-built Docker images.
  uvx modal token new
  uvx modal secret create ptc-hf HF_TOKEN=hf_xxx
  uvx modal deploy deploy/modal/serve.py

After deploy:
  Modal prints a URL like https://sankara68--ptc-gemma4-serve.modal.run
  Set on the HF Space (Settings → Variables and secrets):
    PTC_VLLM_GEMMA4_URL  →  https://sankara68--ptc-gemma4-serve.modal.run/v1
  Restart the Space.

Cost math on a $30 Modal credit (check modal.com/pricing for current rates):
  - A100-80GB ≈ $2/hr  → ~14 hours of GPU time
  - H100-80GB ≈ $5/hr  → ~6 hours

This config is set up for **pure wake-on-demand** — no pre-warming, no idle burn:
  - `min_containers` is intentionally NOT set; the container sleeps by default.
  - First request after sleep: ~60-90 s cold-start delay, billed as GPU seconds.
    On A100 that's ≈ $0.04 per cold-start cycle.
  - scaledown_window=300 → once awake, stays warm for 5 min so a sequential
    set of demo questions doesn't each pay cold-start.
  - Subsequent in-window requests: ~2-3 s each, ≈ $0.002 / request.
  - $30 ≈ 750 cold-starts on A100 — effectively unlimited demo headroom.
  - First-ever deploy spends ~15-20 min on image build + 62 GB weight pull
    on Modal CPU builders → essentially free against the GPU budget.

Note: vLLM 0.6.x serving of Gemma 4 multimodal LoRA has the same caveat as our
ROCm container (silent fallback to base on some adapter configs). The adapter
config in this project ships list-form target_modules (not regex) so vLLM can
parse it; if you see suspicious responses, log the model field — if it says
`google/gemma-4-31B-it` you got base; if `scin-top16` you got the LoRA.
"""
from __future__ import annotations

import os
import subprocess

import modal

APP_NAME = "ptc-gemma4-serve"

BASE_MODEL = "google/gemma-4-31B-it"
LORA_REPO = "sankara68/path-to-care-scin-top16-lora"
LORA_MODULE_NAME = "scin-top16"
API_KEY = "ptc-demo-2026-amd"
PORT = 8000

# Use vLLM's upstream CUDA Docker image — it bundles a vllm+transformers+CUDA
# combination tested together that supports Gemma 4 (`model_type: "gemma4"`).
# Earlier attempts that built from `nvidia/cuda:12.4` + pip-installed
# `vllm==0.6.3.post1` failed with `KeyError: 'gemma4'` because that pair
# predates the Gemma 4 architecture. The vllm/vllm-openai image is what
# the project canonically targets (we run vllm/vllm-openai-rocm:v0.20.1 on
# the AMD MI300X droplet; this is its NVIDIA twin).
#
# Note: we don't pre-pull weights at image-build time. An earlier version
# tried .run_function(_download_model, ...) but the build step stalled
# silently for >30 min. Pulling at serve-time means the first request is
# ~3-5 min while vLLM fetches Gemma from HF; subsequent requests in the
# 5-min warm window are 2-3 s each.
vllm_image = (
    modal.Image
    .from_registry("nvidia/cuda:12.4.0-runtime-ubuntu22.04", add_python="3.12")
    # `git` for HF clone, `gcc` because Triton / torch._inductor JIT-compiles
    # kernels at runtime and silently needs a C compiler on PATH.
    .apt_install("git", "gcc", "g++", "build-essential")
    # Unpinned vllm + transformers so the resolver picks the latest pair that
    # together support Gemma 4's `model_type: "gemma4"`. The earlier attempt
    # pinned vllm==0.6.3.post1 + transformers>=4.46 which predate gemma4 and
    # fail with `KeyError: 'gemma4'`. We also can't use vllm/vllm-openai
    # upstream image directly — Modal can't auto-detect its Python version.
    .pip_install(
        "vllm",
        "transformers",
        "huggingface_hub[hf_transfer]",
        "peft",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App(APP_NAME)


@app.function(
    image=vllm_image,
    gpu="A100-80GB",                  # A100-80GB ≈ $2/hr; bump to "H100" for ~2× speed
    timeout=60 * 60,                  # 1 h per request hard timeout
    scaledown_window=60 * 5,          # scale to zero after 5 min idle (don't burn credits)
    secrets=[modal.Secret.from_name("ptc-hf")],
    # min_containers=1,               # uncomment during demo window — A100 ≈ $2/h while warm
)
@modal.concurrent(max_inputs=20)
@modal.web_server(port=PORT, startup_timeout=60 * 15)   # 15 min — vLLM pulls 62 GB Gemma at first start
def serve() -> None:
    """Launch vLLM's OpenAI-compat server inside this container.

    The @modal.web_server decorator forwards HTTPS traffic to the container's
    internal port. vLLM speaks OpenAI on /v1/* so the HF Space's existing
    `fetch(`${PTC_VLLM_GEMMA4_URL}/chat/completions`)` works unchanged.
    """
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--model", BASE_MODEL,
        "--served-model-name", BASE_MODEL,
        "--api-key", API_KEY,
        "--dtype", "bfloat16",
        "--gpu-memory-utilization", "0.92",
        # Multimodal: allow up to 1 image per request (matches Gemma 4 default).
        # Latest vLLM parses this as JSON (older versions accepted `image=1`).
        "--limit-mm-per-prompt", '{"image": 1}',
        # LoRA — make it addressable as the `scin-top16` model id, same as the droplet
        "--enable-lora",
        "--max-loras", "1",
        "--max-lora-rank", "8",
        "--lora-modules", f"{LORA_MODULE_NAME}={LORA_REPO}",
        # Sane KV-cache defaults for a single 80GB GPU
        "--max-model-len", "8192",
        # Gemma 4 multimodal image encoder produces up to 2496 tokens per image
        # and chunked MM is auto-disabled for bidirectional-attention models,
        # so the per-step batched-token budget must accommodate one full image.
        # 4096 gives headroom over the 2496 minimum.
        "--max-num-batched-tokens", "4096",
    ]
    # Use Popen so the decorator can monitor the port coming up rather than
    # blocking on Python's call().
    subprocess.Popen(cmd)


@app.local_entrypoint()
def main() -> None:
    """`modal run deploy/modal/serve.py` — quick local sanity, prints what would deploy."""
    print(f"App:        {APP_NAME}")
    print(f"Base:       {BASE_MODEL}")
    print(f"LoRA repo:  {LORA_REPO}  (served as model id `{LORA_MODULE_NAME}`)")
    print(f"GPU:        A100-80GB  (≈ $2/hr; upgrade to H100 in @app.function for ~2× speed)")
    print(f"API key:    {API_KEY}  (matches the droplet — drop-in URL swap on HF Space)")
    print()
    print("Deploy:    modal deploy deploy/modal/serve.py")
    print("Then set:  PTC_VLLM_GEMMA4_URL = https://sankara68--ptc-gemma4-serve.modal.run/v1")
    print("           on the HF Space (Settings → Variables and secrets).")
