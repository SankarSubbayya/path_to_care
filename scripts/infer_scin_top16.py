"""Inference utility for the SCIN top-16 LoRA fine-tuned Gemma 4 31B-it.

Loads base + adapter once. Exposes:
  - predict(image_path_or_pil, symptoms_text=...) → list[(condition, confidence)]
  - CLI: --image <path> [--symptoms "..."] [--fitzpatrick "I-II"] [--body-parts "leg"]

Default predicts top-3 conditions ranked by the model's emitted confidences.
The system prompt is identical to training (16 SCIN conditions; output format
'Condition (0.XX); ...').

Why this exists: the project's vLLM ROCm 0.20.1 container does not reliably
apply Gemma 4 multimodal LoRA at serving time (silent fallback to base —
documented in docs/SCIN_DIFF_DX.md). In-process peft is the path that
actually applies the adapter (verified by the +7.0 pp top-1 eval delta on
the 100-row holdout).

Usage:
  PYTHONPATH=. .venv/bin/python scripts/infer_scin_top16.py \
      --image data/scin/images/-1380203193145709903.png \
      --symptoms "itchy red patch for 1 week" \
      --fitzpatrick "I-II" \
      --body-parts "arm"

  Or import:
      from scripts.infer_scin_top16 import load, predict
      handle = load()
      predict(handle, "data/scin/images/foo.png", symptoms_text="itchy")
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Re-use the trained system prompt + class list
sys.path.insert(0, "scripts")
from build_scin_trl_topk import build_system_text  # noqa: E402
from eval_scin_dx_topk import parse_topk  # noqa: E402


@dataclass
class Handle:
    model: object
    processor: object
    classes: list[str]
    system_text: str


def load(adapter: str = "adapters/scin-top16-gemma4-lora",
         classes_path: str = "data/scin/top16_classes_full.json",
         base_id: str = "google/gemma-4-31B-it") -> Handle:
    """Load base + adapter once. ~30-60 s for the 31B model on MI300X."""
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import PeftModel

    classes = json.load(open(classes_path))
    print(f"loading {base_id} ...", flush=True)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(base_id)
    model = AutoModelForImageTextToText.from_pretrained(
        base_id, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    print(f"  base loaded in {time.time() - t0:.1f}s", flush=True)

    print(f"attaching LoRA from {adapter} ...", flush=True)
    t0 = time.time()
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    print(f"  attached in {time.time() - t0:.1f}s", flush=True)

    return Handle(
        model=model,
        processor=processor,
        classes=classes,
        system_text=build_system_text(classes),
    )


def _build_user_text(*, symptoms_text: str | None, body_parts: str | None,
                     duration: str | None, fitzpatrick: str | None,
                     sex: str | None, age: str | None) -> str:
    bits = []
    if body_parts:
        bits.append(f"Body parts: {body_parts}.")
    if symptoms_text:
        bits.append(f"Reported symptoms: {symptoms_text}.")
    if duration:
        bits.append(f"Duration: {duration}.")
    if fitzpatrick:
        bits.append(f"Fitzpatrick skin type: {fitzpatrick}.")
    if sex:
        bits.append(f"Sex at birth: {sex}.")
    if age:
        bits.append(f"Age: {age}.")
    if not bits:
        bits.append("(no additional metadata)")
    return " ".join(bits)


def predict(handle: Handle, image,
            symptoms_text: str | None = None,
            body_parts: str | None = None,
            duration: str | None = None,
            fitzpatrick: str | None = None,
            sex: str | None = None,
            age: str | None = None,
            max_new_tokens: int = 80) -> tuple[list[tuple[str, float]], str]:
    """Predict top-3 conditions for an image+context.

    `image` may be a path string or a PIL.Image. Returns
    (parsed_topk, raw_response_text).
    """
    import torch
    from PIL import Image

    img = image if hasattr(image, "size") else Image.open(image).convert("RGB")
    user_text = _build_user_text(
        symptoms_text=symptoms_text, body_parts=body_parts,
        duration=duration, fitzpatrick=fitzpatrick, sex=sex, age=age,
    )
    msgs = [
        {"role": "system", "content": [{"type": "text", "text": handle.system_text}]},
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": user_text},
        ]},
    ]
    text = handle.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inputs = handle.processor(text=[text], images=[[img]], return_tensors="pt").to(handle.model.device)
    with torch.no_grad():
        out = handle.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    raw = handle.processor.batch_decode(
        out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0].strip()
    return parse_topk(raw, handle.classes, k=3), raw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="path to the skin image (PNG/JPG)")
    ap.add_argument("--symptoms", default=None)
    ap.add_argument("--body-parts", default=None)
    ap.add_argument("--duration", default=None)
    ap.add_argument("--fitzpatrick", default=None, help="I-II / III-IV / V-VI")
    ap.add_argument("--sex", default=None)
    ap.add_argument("--age", default=None)
    ap.add_argument("--adapter", default="adapters/scin-top16-gemma4-lora")
    ap.add_argument("--classes", default="data/scin/top16_classes_full.json")
    args = ap.parse_args()

    handle = load(adapter=args.adapter, classes_path=args.classes)
    topk, raw = predict(
        handle, args.image,
        symptoms_text=args.symptoms,
        body_parts=args.body_parts,
        duration=args.duration,
        fitzpatrick=args.fitzpatrick,
        sex=args.sex,
        age=args.age,
    )
    print()
    print("=== prediction ===")
    print(f"  raw: {raw!r}")
    print(f"  parsed top-{len(topk)}:")
    for c, w in topk:
        print(f"    {c:35s}  {w:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
