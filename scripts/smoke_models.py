"""Phase 1 smoke test: load Qwen-2.5-7B-Instruct (SOAP) and Gemma 4 26B-A4B-it
(vision + triage). Writes evidence/qwen_smoke.txt and evidence/gemma4_smoke.txt.

See docs/COMPATIBILITY.md for the model-selection audit. Pivot path:
  Gemma 3 12B-it (gated, 401)  →  Qwen2-VL-7B (single-vendor pivot)
  →  Gemma 4 26B-A4B-it (open, multimodal MoE — final pick).

transformers 5.x notes (different from 4.x):
- pass `dtype=...` not `torch_dtype=...`
- `apply_chat_template(..., return_tensors='pt')` is unreliable; instead
  apply with `tokenize=False`, then call the tokenizer separately.

Fallbacks if Gemma 4 26B-A4B-it doesn't load: 31B dense → E4B → Qwen2-VL-7B.
"""
import argparse
import os
import sys
import time
from pathlib import Path

QWEN_TEXT_ID = "Qwen/Qwen2.5-7B-Instruct"
GEMMA4_ID = "google/gemma-4-31B-it"  # 26B-A4B (MoE) hits grouped_mm-on-ROCm bug; 31B-it is dense and works

QWEN_TEXT_PROMPT = "Extract the chief complaint from this narrative as one short phrase: 'My foot hurts and is swollen, started two days ago after I cut it on a rusty nail.'"
GEMMA4_PROMPT = "List up to three possible skin/wound conditions for a swollen, reddened ankle with a small puncture wound, each with a confidence between 0 and 1. Do not diagnose. Output one per line as 'condition: confidence'."


def smoke_text(model_id: str, prompt: str, evidence_path: str) -> int:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    Path(os.path.dirname(evidence_path)).mkdir(parents=True, exist_ok=True)
    lines = [f"model: {model_id}", f"prompt: {prompt!r}"]
    try:
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
        load_s = time.time() - t0

        text_prompt = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = tok(text_prompt, return_tensors="pt").to("cuda")
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=96, do_sample=False)
        gen_s = time.time() - t0
        text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        lines = ["verdict: PASS"] + lines + [
            f"load_seconds: {load_s:.1f}",
            f"generate_seconds: {gen_s:.2f}",
            f"output: {text!r}",
        ]
    except Exception as e:
        import traceback
        lines = ["verdict: FAIL"] + lines + [f"exception: {type(e).__name__}: {e}", traceback.format_exc()]

    with open(evidence_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if lines[0].endswith("PASS") else 1


def smoke_multimodal(model_id: str, prompt: str, evidence_path: str) -> int:
    """Multimodal model smoke: text-only prompt for now (no image yet). Image
    inference is exercised in the image-classifier MCP smoke later. Goal here
    is just to confirm the model loads and generates at all on ROCm."""
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    Path(os.path.dirname(evidence_path)).mkdir(parents=True, exist_ok=True)
    lines = [f"model: {model_id}", f"prompt: {prompt!r}"]
    try:
        t0 = time.time()
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
        load_s = time.time() - t0

        # Text-only conversation (no image) — confirms the model class works.
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[text_prompt], return_tensors="pt").to("cuda")
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        gen_s = time.time() - t0
        gen_ids = out[:, inputs["input_ids"].shape[1]:]
        text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

        lines = ["verdict: PASS"] + lines + [
            f"load_seconds: {load_s:.1f}",
            f"generate_seconds: {gen_s:.2f}",
            f"output: {text!r}",
        ]
    except Exception as e:
        import traceback
        lines = ["verdict: FAIL"] + lines + [f"exception: {type(e).__name__}: {e}", traceback.format_exc()]

    with open(evidence_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if lines[0].endswith("PASS") else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=("text", "gemma4", "both"), default="both")
    args = ap.parse_args()

    rcs = []
    if args.which in ("text", "both"):
        rcs.append(smoke_text(QWEN_TEXT_ID, QWEN_TEXT_PROMPT, "evidence/qwen_smoke.txt"))
    if args.which in ("gemma4", "both"):
        rcs.append(smoke_multimodal(GEMMA4_ID, GEMMA4_PROMPT, "evidence/gemma4_smoke.txt"))
    return 0 if all(rc == 0 for rc in rcs) else 1


if __name__ == "__main__":
    sys.exit(main())
