"""Step 3 / 5: Multimodal LoRA SFT on Gemma 4 31B-it for SCIN 34-class
differential diagnosis.

Recipe ported from /shared-docker/fine_tuning_lora_qwen2vl.ipynb (AMD-modified
TRL + PEFT cookbook). Differences for Gemma 4:

  - Loaded via AutoProcessor + AutoModelForImageTextToText (Gemma 4 31B
    classes register against these auto classes).
  - LoRA target_modules restricted to **language-model self-attention**
    (q/k/v/o) via regex — avoids the vision tower's Gemma4ClippableLinear
    wrapper which peft 0.19 doesn't recognize as a Linear (verified live
    in Phase 5; documented in docs/COMPATIBILITY.md).
  - bf16 throughout. The Qwen2-VL notebook uses .half() + fp16=True; on our
    ROCm 6.3 + torch 2.9.1 stack the in-process Phase 5 LoRA already ran
    cleanly in bf16, so we keep bf16 here.
  - Image-token IDs auto-detected from the processor at runtime (Gemma 4
    has its own special tokens, not Qwen2-VL's [151652, 151653, 151655]).
  - Reads the chat-format JSONL produced by scripts/build_scin_trl.py.

Inputs:
  data/scin/dx34_trl_train.jsonl    (1203 chats with image+text → condition)
  data/scin/dx34_trl_holdout.jsonl  (336 chats; used for periodic eval)

Outputs:
  adapters/scin-dx34-gemma4-lora/   (adapter_config.json + .safetensors)
  logs/lora_dx_multimodal.log       (loss curve, written by Trainer)

Usage:
  PYTHONPATH=. .venv/bin/python scripts/lora_dx_multimodal.py \
      --epochs 3 --batch-size 1 --grad-accum 8 --lr 2e-4 \
      --lora-r 16 --lora-alpha 32 \
      --output adapters/scin-dx34-gemma4-lora
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


# ----------------------------------------------------------------------------
# Helpers used by the data pipeline
# ----------------------------------------------------------------------------

def _image_token_ids(processor) -> list[int]:
    """Return all token IDs the chat template uses to represent image content.
    These are masked to -100 in labels so the loss skips image positions."""
    ids: list[int] = []

    # Gemma 4 / Gemma 3 processors expose `image_token` (single token).
    if hasattr(processor, "image_token"):
        try:
            tid = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
            if isinstance(tid, int) and tid >= 0:
                ids.append(tid)
        except Exception:
            pass

    # Some processors expose `image_token_id` directly.
    if hasattr(processor, "image_token_id"):
        try:
            tid = int(processor.image_token_id)
            if tid >= 0:
                ids.append(tid)
        except Exception:
            pass

    # Try a list of common special tokens used by image-aware chat templates.
    for tok in ("<image>", "<image_soft_token>", "<start_of_image>",
                "<end_of_image>", "<|image_pad|>", "<|vision_start|>",
                "<|vision_end|>", "<|image|>", "<|vision_pad|>"):
        try:
            tid = processor.tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid != processor.tokenizer.unk_token_id and tid >= 0:
                ids.append(tid)
        except Exception:
            pass

    return sorted(set(ids))


def _images_from_messages(messages: list[dict]) -> list:
    """Pull PIL images out of one chat's message structure."""
    from PIL import Image
    images = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                src = part.get("image")
                if isinstance(src, str):
                    images.append(Image.open(src).convert("RGB"))
                else:
                    images.append(src)
    return images


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/scin/dx34_trl_train.jsonl")
    ap.add_argument("--eval", dest="eval_path", default="data/scin/dx34_trl_holdout.jsonl")
    ap.add_argument("--output", default="adapters/scin-dx34-gemma4-lora")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--logging-steps", type=int, default=1)
    ap.add_argument("--eval-steps", type=int, default=25)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--limit-train", type=int, default=None,
                    help="for smoke testing: cap training rows")
    ap.add_argument("--limit-eval", type=int, default=None)
    args = ap.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

    # Lazy import: heavy stuff only after --help works.
    import torch
    from transformers import (
        AutoProcessor,
        AutoModelForImageTextToText,
        TrainerCallback,
        set_seed,
    )
    from peft import LoraConfig, get_peft_model
    from trl import SFTConfig, SFTTrainer

    # Real-time per-step logger so we can plot the loss curve while training
    # is still running, instead of waiting for trainer_state.json at the
    # first checkpoint save.
    class JsonlLossLogger(TrainerCallback):
        def __init__(self, path: str):
            import os, time as _t
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self.path = path
            # truncate existing
            with open(path, "w") as f:
                f.write("")
            self.t0 = _t.time()

        def on_log(self, args, state, control, logs=None, **kwargs):
            import json, time as _t
            if not logs:
                return
            row = {
                "step": int(state.global_step),
                "epoch": float(logs.get("epoch") or state.epoch or 0),
                "wall_seconds": round(_t.time() - self.t0, 1),
                **{k: v for k, v in logs.items() if not isinstance(v, (dict, list))},
            }
            with open(self.path, "a") as f:
                f.write(json.dumps(row) + "\n")

    set_seed(args.seed)
    torch.manual_seed(args.seed)

    # We deliberately import GEMMA4_ID from the in-process backend so the
    # model id is consistent with the rest of the project (Phase 5 adapter).
    from core._llm_transformers import GEMMA4_ID

    print("== multimodal LoRA SFT ==")
    print(f"  base:      {GEMMA4_ID}")
    print(f"  train:     {args.train}")
    print(f"  eval:      {args.eval_path}")
    print(f"  output:    {args.output}")
    print(f"  epochs:    {args.epochs}, bs={args.batch_size}, grad_accum={args.grad_accum}, lr={args.lr}")
    print(f"  lora:      r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print()

    # --- Load model + processor in bf16 ---
    print("loading processor + model (this takes ~60-90s for Gemma 4 31B)...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(GEMMA4_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        GEMMA4_ID,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    print(f"  loaded in {time.time() - t0:.1f}s")

    # Discover image token IDs from this specific processor.
    img_tok_ids = _image_token_ids(processor)
    print(f"  image_token_ids: {img_tok_ids}  (these will be -100 in labels)")

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # --- LoRA: language-model attention only (Gemma4ClippableLinear wraps
    # the *vision* tower's projections, which peft 0.19 doesn't accept).
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=r".*language_model.*self_attn\.(q_proj|k_proj|v_proj|o_proj)$",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # --- Data ---
    def load_jsonl(path: str, limit: int | None) -> list[dict]:
        out = []
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if limit is not None and len(out) >= limit:
                break
        return out

    train_rows = load_jsonl(args.train, args.limit_train)
    eval_rows = load_jsonl(args.eval_path, args.limit_eval)
    print(f"  train rows: {len(train_rows)}")
    print(f"  eval  rows: {len(eval_rows)}")

    # --- collate_fn (this is what does the heavy lifting per batch) ---
    def collate_fn(examples: list[dict]) -> dict:
        from PIL import Image  # noqa: F401  (used inside _images_from_messages)
        # The chat template emits image placeholder tokens; processor() then
        # replaces them with the actual image embeddings/patches.
        msg_lists = [ex["messages"] for ex in examples]
        texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in msg_lists
        ]
        image_lists = [_images_from_messages(m) for m in msg_lists]

        # Some models prefer flat list of images; some want list-of-lists.
        # AutoProcessor handles either; we pass list-of-lists so it can match
        # multiple images per message in the future.
        batch = processor(
            text=texts,
            images=image_lists,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_seq_len,
        )

        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        for tid in img_tok_ids:
            labels[labels == tid] = -100
        batch["labels"] = labels
        return batch

    # --- SFTConfig (mirrors the AMD notebook's settings) ---
    sft_config = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        learning_rate=args.lr,
        lr_scheduler_type="constant",
        warmup_ratio=0.03,
        max_grad_norm=0.3,
        # bf16 on ROCm; this matched our successful Phase 5 LoRA run.
        bf16=True,
        fp16=False,
        tf32=False,
        # Logging / eval / save cadence — eval is on the holdout chats.
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["none"],
        logging_dir="logs",
        # Crucial flags for vision SFT (per the AMD notebook):
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        seed=args.seed,
    )

    # --- Trainer ---
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_rows,
        eval_dataset=eval_rows,
        data_collator=collate_fn,
        # Tokenizer/processor handle is read off the model in trl 1.4+
    )
    trainer.add_callback(JsonlLossLogger("logs/scin_lora_train.jsonl"))

    print("starting training...")
    t0 = time.time()
    trainer.train()
    print(f"trained in {time.time() - t0:.1f}s")

    print(f"saving adapter to {args.output} ...")
    trainer.save_model(args.output)
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
