"""Inference backend dispatcher for Path to Care.

Two backends share the same public surface so MCPs, the orchestrator, and the
harness can do `from core.llm import gemma4, qwen, chat_text, chat_multimodal,
gemma4_attach_adapter, LoadedLM` without caring which backend is in use.

Selected at process start by `PTC_INFERENCE`:
  - `transformers` (default) — in-process via the venv. This is the path the
    Phase 3 baseline and Phase 6 tuned eval ran on, so the
    `results/baseline_metrics.json` and `results/tuned_metrics.json` numbers
    are in this backend.
  - `vllm` — OpenAI-compatible HTTP calls to a local vLLM container. See
    docs/VLLM_SERVE.md for the Docker run command. Use for the Gradio Space,
    post-submission demos, and load tests.

Switching backends mid-eval invalidates a before/after delta because the two
engines have different KV-cache and sampling implementations. Always run a
single eval with one backend.
"""
from __future__ import annotations

import os

INFERENCE_BACKEND = os.environ.get("PTC_INFERENCE", "transformers").lower()

if INFERENCE_BACKEND == "vllm":
    from core._llm_vllm import (  # noqa: F401  (public re-exports)
        LoadedLM,
        gemma4,
        qwen,
        gemma4_attach_adapter,
        chat_text,
        chat_multimodal,
    )
elif INFERENCE_BACKEND == "transformers":
    from core._llm_transformers import (  # noqa: F401
        LoadedLM,
        gemma4,
        qwen,
        gemma4_attach_adapter,
        chat_text,
        chat_multimodal,
    )
else:
    raise RuntimeError(
        f"Unknown PTC_INFERENCE={INFERENCE_BACKEND!r}. "
        "Use 'transformers' (in-process, default) or 'vllm' (HTTP)."
    )

# Compatibility re-export: pre-dispatcher code occasionally references
# `core.llm.GEMMA4_ID` or `core.llm.QWEN_ID`. Keep them populated from
# whichever backend was selected.
if INFERENCE_BACKEND == "vllm":
    from core._llm_vllm import GEMMA4_MODEL_ID as GEMMA4_ID  # noqa: F401
    from core._llm_vllm import QWEN_MODEL_ID as QWEN_ID  # noqa: F401
else:
    from core._llm_transformers import GEMMA4_ID, QWEN_ID  # noqa: F401
