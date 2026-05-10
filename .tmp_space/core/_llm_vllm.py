"""vLLM HTTP backend for Path to Care.

Same public surface as core/_llm_transformers.py:
  - LoadedLM dataclass (with model=None, processor_or_tok=OpenAI client)
  - gemma4(), qwen()
  - gemma4_attach_adapter(adapter_path)
  - chat_text(handle, prompt), chat_multimodal(handle, prompt, image)

Why both backends share a class+function shape: the MCPs, orchestrator, and
harness import from core.llm without caring which backend they hit. The
dispatcher in core/llm.py picks one at process start based on
`PTC_INFERENCE=transformers|vllm` (default `transformers`).

vLLM serves one model per container. To run the hybrid stack you need TWO
containers:
  - Gemma 4 31B-it on http://<host>:8000/v1   (PTC_VLLM_GEMMA4_URL)
  - Qwen 2.5-7B-Instruct on http://<host>:8001/v1 (PTC_VLLM_QWEN_URL)

For dev/demo, you can point both at the same Gemma container if you don't
need real Qwen SOAP extraction (Gemma is competent at that too) — set
both URLs to the same value and override PTC_VLLM_QWEN_MODEL_ID accordingly.

LoRA: vLLM serves adapters via `--enable-lora --lora-modules name=path`. To
use the path-to-care adapter, start the vLLM container with:
  --enable-lora --lora-modules triage=adapters/triage-gemma4-lora
…and `gemma4_attach_adapter` flips the served model id to "triage".
"""
from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass, field
from typing import Any, Optional


GEMMA4_BASE_URL = os.environ.get("PTC_VLLM_GEMMA4_URL", "http://localhost:8000/v1")
GEMMA4_MODEL_ID = os.environ.get("PTC_VLLM_GEMMA4_MODEL_ID", "google/gemma-4-31B-it")

QWEN_BASE_URL = os.environ.get("PTC_VLLM_QWEN_URL", "http://localhost:8001/v1")
QWEN_MODEL_ID = os.environ.get("PTC_VLLM_QWEN_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")

API_KEY = os.environ.get("PTC_VLLM_API_KEY", "ptc-demo-2026-amd")

# vLLM LoRA endpoint convention: served model id changes to the LoRA name.
# Default empty — only route to a LoRA name when the vLLM container was
# explicitly started with `--enable-lora --lora-modules <name>=<path>`.
# Otherwise asking for a non-existent model id returns 404 and the demo
# blows up. To opt in: set PTC_VLLM_LORA_NAME=triage on the client.
LORA_NAME = os.environ.get("PTC_VLLM_LORA_NAME", "")


@dataclass
class LoadedLM:
    """Mirrors the dataclass in core/_llm_transformers.py.
    For the vLLM backend `model` is a model id string (used in API calls);
    `processor_or_tok` is an OpenAI client; `is_multimodal` flags vision support."""
    model: str
    processor_or_tok: Any
    is_multimodal: bool


def _client(base_url: str):
    # Lazy-import openai so import-time cost is paid only when vLLM is selected.
    from openai import OpenAI
    return OpenAI(base_url=base_url, api_key=API_KEY)


def gemma4() -> LoadedLM:
    return LoadedLM(
        model=GEMMA4_MODEL_ID,
        processor_or_tok=_client(GEMMA4_BASE_URL),
        is_multimodal=True,
    )


def qwen() -> LoadedLM:
    return LoadedLM(
        model=QWEN_MODEL_ID,
        processor_or_tok=_client(QWEN_BASE_URL),
        is_multimodal=False,
    )


def gemma4_attach_adapter(adapter_path: str) -> LoadedLM:
    """Switch the served model id to a LoRA name only when explicitly
    configured (`PTC_VLLM_LORA_NAME=<name>`). The vLLM container must have
    been launched with `--enable-lora --lora-modules <name>=<path>`.

    If `PTC_VLLM_LORA_NAME` is empty (the default), the adapter request is
    a no-op and we serve the base model. This keeps the live demo working
    against a base-model-only vLLM container; the LoRA delta lives in the
    eval numbers (results/tuned_metrics.json), not the demo."""
    handle = gemma4()
    if LORA_NAME:
        handle.model = LORA_NAME
    return handle


def _image_to_data_url(image) -> str:
    """Convert a PIL.Image to a data: URL the OpenAI/vLLM image_url field accepts."""
    buf = io.BytesIO()
    fmt = (image.format or "PNG").upper()
    image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"


def chat_text(handle: LoadedLM, prompt: str, max_new_tokens: int = 256) -> str:
    client = handle.processor_or_tok
    resp = client.chat.completions.create(
        model=handle.model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens,
        temperature=0,
    )
    return (resp.choices[0].message.content or "").strip()


def chat_multimodal(handle: LoadedLM, prompt: str, image=None, max_new_tokens: int = 256) -> str:
    """OpenAI-compatible content list. `image` may be None or a PIL.Image."""
    if image is None:
        # Fall through to the text-only path; vLLM accepts text-only on
        # multimodal endpoints when the content is plain string OR a single
        # text part. Use the text-only call for cleanliness.
        return chat_text(handle, prompt, max_new_tokens=max_new_tokens)

    content = [
        {"type": "image_url", "image_url": {"url": _image_to_data_url(image)}},
        {"type": "text", "text": prompt},
    ]
    client = handle.processor_or_tok
    resp = client.chat.completions.create(
        model=handle.model,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_new_tokens,
        temperature=0,
    )
    return (resp.choices[0].message.content or "").strip()
