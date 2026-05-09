"""LoRA SFT for Gemma 4 31B-it triage reasoner.

Trains on `data/train.jsonl` (built by training.build_train_set), saves the
adapter to `adapters/triage-gemma4-lora/`, logs to `logs/lora_train.log`.

This script is designed to run unattended through the sleep window per
[CLAUDE.md] Phase 5. It writes its progress continuously so a crash doesn't
lose information about how far it got.

Usage:
  .venv/bin/python -m training.lora_sft \
      --train data/train.jsonl \
      --output adapters/triage-gemma4-lora \
      --epochs 2 --batch-size 1 --lr 2e-4

Stretch goal (not enabled by default): GRPO/RLVR via TRL. Skeleton in
training/grpo_stretch.py — not used unless `--method grpo` is passed and
TRL imports work on ROCm.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from core.llm import GEMMA4_ID


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--output", default="adapters/triage-gemma4-lora")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

    # Lazy imports so the script can fail fast on missing torch without
    # holding the import for users running --help.
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText, set_seed
    from peft import LoraConfig, get_peft_model

    set_seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"== LoRA SFT ==")
    print(f"  base:    {GEMMA4_ID}")
    print(f"  train:   {args.train}")
    print(f"  output:  {args.output}")
    print(f"  epochs:  {args.epochs}, bs={args.batch_size}, grad_accum={args.grad_accum}, lr={args.lr}")
    print(f"  lora:    r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")

    processor = AutoProcessor.from_pretrained(GEMMA4_ID)
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    model = AutoModelForImageTextToText.from_pretrained(
        GEMMA4_ID, dtype=torch.bfloat16, device_map="cuda"
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        # Target attention projections in language layers; safe across
        # dense decoder transformers. Avoid touching vision tower.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # Build the training set as token tensors. We mask the prompt portion so
    # the loss is computed only on the assistant response.
    rows = load_jsonl(args.train)
    print(f"  train_rows: {len(rows)}")

    def encode(row: dict) -> dict:
        # Frame as a chat: user prompt, assistant target. Compute labels with
        # prompt tokens set to -100 (ignored).
        messages_user = [{"role": "user", "content": [{"type": "text", "text": row["prompt"]}]}]
        prompt_text = processor.apply_chat_template(messages_user, add_generation_prompt=True, tokenize=False)
        full_text = prompt_text + row["target"] + tok.eos_token
        full = tok(full_text, return_tensors="pt", truncation=True, max_length=args.max_seq_len)
        prompt_only = tok(prompt_text, return_tensors="pt", truncation=True, max_length=args.max_seq_len)
        input_ids = full["input_ids"][0]
        labels = input_ids.clone()
        labels[: prompt_only["input_ids"].shape[1]] = -100
        return {"input_ids": input_ids, "attention_mask": full["attention_mask"][0], "labels": labels}

    encoded = [encode(r) for r in rows]

    # Custom training loop — Trainer would also work but this keeps us close
    # to the metal so a ROCm-specific oddity is easier to debug.
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    log_path = "logs/lora_train.log"
    log_f = open(log_path, "w")

    def log(msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        log_f.write(line + "\n")
        log_f.flush()

    log(f"start: epochs={args.epochs}, rows={len(encoded)}, effective_bs={args.batch_size * args.grad_accum}")
    model.train()
    step = 0
    accum_loss = 0.0
    optim.zero_grad()
    for epoch in range(args.epochs):
        for i, ex in enumerate(encoded):
            input_ids = ex["input_ids"].unsqueeze(0).to(model.device)
            attn = ex["attention_mask"].unsqueeze(0).to(model.device)
            labels = ex["labels"].unsqueeze(0).to(model.device)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss / args.grad_accum
            loss.backward()
            accum_loss += loss.item()
            if (i + 1) % args.grad_accum == 0:
                optim.step()
                optim.zero_grad()
                step += 1
                log(f"epoch={epoch} step={step} avg_loss_in_step={accum_loss:.4f}")
                accum_loss = 0.0
        log(f"=== epoch {epoch} done ===")

    log("saving adapter ...")
    model.save_pretrained(args.output)
    log(f"adapter saved to {args.output}")
    log_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
