"""Baseten Truss model wrapping vLLM's OpenAI-compatible server.

Why this shape, rather than re-implementing /v1/chat/completions in `predict()`:
the HF Space already speaks OpenAI to our AMD droplet and to the Modal fallback.
Wrapping vLLM's own /v1 endpoint means the same `PTC_VLLM_GEMMA4_URL` env-var
swap on the HF Space flips inference targets with zero application code change.

`load()` starts the vLLM server in a subprocess and waits until /v1/models
returns 200. `predict()` proxies the incoming JSON straight through to vLLM
and returns its response unchanged.

Cold start expectations (single A100-80GB, fresh container):
  - vLLM init + model download from HF: ~4 min
  - bf16 load to VRAM: ~3-4 min
  - CUDA graph capture + Triton autotune: ~4-5 min
  - Total to first token: ~12-15 min (matches what we observed on Modal)
Subsequent requests in the same container: 1-3 s end-to-end.
"""
from __future__ import annotations

import os
import subprocess
import time
from typing import Any

import requests


BASE_MODEL = "google/gemma-4-31B-it"
LORA_REPO = "sankara68/path-to-care-scin-top16-lora"
LORA_MODULE_NAME = "scin-top16"
API_KEY = "ptc-demo-2026-amd"   # same as droplet + Modal so URL swaps are zero-touch
VLLM_PORT = 8000
VLLM_BASE = f"http://localhost:{VLLM_PORT}"


class Model:
    def __init__(self, **kwargs: Any) -> None:
        self._secrets = kwargs.get("secrets") or {}
        self._proc: subprocess.Popen | None = None

    def load(self) -> None:
        """Start vLLM and block until it's serving."""
        # Make Gemma 4's gating accept our request — both vLLM and HF Hub
        # consult HF_TOKEN at process start. The Baseten secret name is
        # `hf_access_token` per config.yaml.
        token = self._secrets.get("hf_access_token") or os.environ.get("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token

        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--host", "0.0.0.0",
            "--port", str(VLLM_PORT),
            "--model", BASE_MODEL,
            "--served-model-name", BASE_MODEL,
            "--api-key", API_KEY,
            "--dtype", "bfloat16",
            "--gpu-memory-utilization", "0.92",
            # Multimodal: 1 image per prompt (matches Gemma 4 default).
            # Latest vLLM parses this as JSON; older versions accepted `image=1`.
            "--limit-mm-per-prompt", '{"image": 1}',
            # Image encoder produces up to 2496 tokens per image; chunked MM is
            # auto-disabled for bidirectional-attention models, so the batched-
            # tokens budget must cover one full image.
            "--max-num-batched-tokens", "4096",
            # LoRA: addressable as model id `scin-top16`, mirrors the droplet.
            "--enable-lora",
            "--max-loras", "1",
            "--max-lora-rank", "8",
            "--lora-modules", f"{LORA_MODULE_NAME}={LORA_REPO}",
            "--max-model-len", "8192",
        ]
        print("[ptc] launching vLLM:", " ".join(cmd), flush=True)
        self._proc = subprocess.Popen(cmd)

        # Block until /v1/models returns 200. vLLM's startup is dominated by
        # weight pull + bf16 load + CUDA graph capture, so we give it up to
        # 20 minutes. If we hit that ceiling, the subprocess likely died —
        # let Baseten surface the failure rather than spin forever.
        deadline = time.time() + 60 * 20
        last_err: Exception | None = None
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"vLLM exited during startup with code {self._proc.returncode}"
                )
            try:
                r = requests.get(
                    f"{VLLM_BASE}/v1/models",
                    headers={"Authorization": f"Bearer {API_KEY}"},
                    timeout=5,
                )
                if r.status_code == 200:
                    print("[ptc] vLLM ready:", r.json(), flush=True)
                    return
                last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            except requests.RequestException as e:
                last_err = e
            time.sleep(5)
        raise RuntimeError(f"vLLM did not become ready within 20 min: {last_err}")

    def predict(self, request: dict[str, Any]) -> dict[str, Any]:
        """Proxy the request straight to vLLM's OpenAI-compatible endpoint.

        The HF Space sends a full OpenAI chat-completions payload. We forward
        it unchanged so Path-to-Care application code never has to know it's
        talking to Baseten vs. AMD vs. Modal.

        For non-/chat-completions endpoints (e.g. /v1/models), Truss routes
        them through Baseten's standard model paths and we don't see them
        here — that's fine; we don't need to handle /v1/models from inside
        predict because Baseten gives us a fronting URL.
        """
        # Baseten injects an `is_streaming` hint for streaming responses; we
        # honor it by streaming back if present.
        is_streaming = bool(request.pop("stream", False)) if "stream" in request else False
        try:
            resp = requests.post(
                f"{VLLM_BASE}/v1/chat/completions",
                json={**request, "stream": is_streaming},
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                timeout=300,
                stream=is_streaming,
            )
        except requests.RequestException as e:
            return {"error": f"vLLM proxy failed: {type(e).__name__}: {e}"}

        if is_streaming:
            # Stream raw chunks back via Truss's streaming generator interface.
            def _iter():
                for chunk in resp.iter_content(chunk_size=1024):
                    if chunk:
                        yield chunk
            return _iter()
        try:
            return resp.json()
        except ValueError:
            return {"error": f"vLLM returned non-JSON: {resp.text[:400]}"}
