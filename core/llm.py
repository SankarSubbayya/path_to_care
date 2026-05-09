"""Shared model loaders for Path to Care.

Two model handles, lazy-loaded the first time they're called and cached for
the lifetime of the process:

  - `gemma4()`  — `google/gemma-4-26B-A4B-it`, multimodal MoE. Backs the
                  image-classifier MCP and the triage-reasoner MCP. Same
                  loaded weights, two prompts.
  - `qwen()`    — `Qwen/Qwen2.5-7B-Instruct`, text-only. Backs the
                  SOAP-extractor MCP and the DSPy LM wrapper.

Why share: with 4 MCPs each loading their own copy of a 26B model, we'd OOM
even on a 192 GB MI300X (and load times would dominate the eval). Single
loaded copy = ~67 GB across both models, leaves ~120 GB headroom for LoRA
training overhead.

The triage-reasoner MCP later loads a LoRA adapter on top of the Gemma 4
weights via `gemma4_with_adapter(adapter_path)` — that wraps the existing
loaded model with `peft.PeftModel.from_pretrained` rather than re-loading.

See [docs/COMPATIBILITY.md] for the full model-selection audit (Gemma 3 was
gated; pivoted to Gemma 4 26B-A4B-it, the open multimodal MoE).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

GEMMA4_ID = os.environ.get("PTC_GEMMA4_ID", "google/gemma-4-31B-it")  # dense; MoE 26B-A4B hits grouped_mm-on-ROCm
QWEN_ID = os.environ.get("PTC_QWEN_ID", "Qwen/Qwen2.5-7B-Instruct")


@dataclass
class LoadedLM:
    model: object         # transformers model
    processor_or_tok: object  # AutoProcessor (multimodal) or AutoTokenizer (text)
    is_multimodal: bool


_GEMMA4: Optional[LoadedLM] = None
_QWEN: Optional[LoadedLM] = None


def gemma4() -> LoadedLM:
    """Load Gemma 4 26B-A4B-it on first call; return cached handle thereafter."""
    global _GEMMA4
    if _GEMMA4 is None:
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        processor = AutoProcessor.from_pretrained(GEMMA4_ID)
        model = AutoModelForImageTextToText.from_pretrained(
            GEMMA4_ID, dtype=torch.bfloat16, device_map="cuda"
        )
        model.eval()
        _GEMMA4 = LoadedLM(model=model, processor_or_tok=processor, is_multimodal=True)
    return _GEMMA4


def gemma4_attach_adapter(adapter_path: str) -> LoadedLM:
    """Wrap the loaded Gemma 4 with a LoRA adapter from `adapter_path` for
    the triage-reasoner MCP. Idempotent: calling twice is fine."""
    global _GEMMA4
    base = gemma4()
    from peft import PeftModel
    adapted = PeftModel.from_pretrained(base.model, adapter_path)
    adapted.eval()
    _GEMMA4 = LoadedLM(model=adapted, processor_or_tok=base.processor_or_tok, is_multimodal=True)
    return _GEMMA4


def qwen() -> LoadedLM:
    """Load Qwen-2.5-7B-Instruct on first call; return cached handle."""
    global _QWEN
    if _QWEN is None:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(QWEN_ID)
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_ID, dtype=torch.bfloat16, device_map="cuda"
        )
        model.eval()
        _QWEN = LoadedLM(model=model, processor_or_tok=tok, is_multimodal=False)
    return _QWEN


def chat_text(handle: LoadedLM, prompt: str, max_new_tokens: int = 256) -> str:
    """Run a single user-turn chat completion against a text-only handle."""
    import torch
    tok = handle.processor_or_tok
    text_prompt = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tok(text_prompt, return_tensors="pt").to(handle.model.device)
    with torch.no_grad():
        out = handle.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def chat_multimodal(handle: LoadedLM, prompt: str, image=None, max_new_tokens: int = 256) -> str:
    """Run a single user-turn chat completion against a multimodal handle.
    `image` may be None (text-only call against a multimodal model) or a PIL.Image."""
    import torch
    processor = handle.processor_or_tok
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    images = [image] if image is not None else None
    inputs = processor(text=[text_prompt], images=images, return_tensors="pt").to(handle.model.device)
    with torch.no_grad():
        out = handle.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
