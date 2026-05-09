# PROGRESS.md

Handoff log between sessions. Every session ends by updating this; every session starts by reading it.

## Done

(nothing yet — harness wired 2026-05-09)

## In progress

- **Phase 1 — Environment + skeleton + smoke loads** (hours 0–2 of the 24-hour plan)
  - Wire the long-running-agents harness ✓ (this session)
  - Install ROCm PyTorch + transformers + peft + dspy + gradio
  - Scaffold repo skeleton (harness/, adversary/, mcp/, orchestrator/, training/, frontend/)
  - Smoke-load Gemma 3 12B-it on MI300X (`evidence/gemma_smoke.txt`)
  - Smoke-load Qwen-2.5-7B-Instruct (`evidence/qwen_smoke.txt`)

## Next

- **Phase 2 — Eval harness + adversary + 30-case test set** (hours 2–4)
  - Reward function (1.0 / 0.5 / 0.0) with unit tests
  - Adversarial generator producing 30 balanced (10 R / 10 Y / 10 G) cases
  - Test set committed at `data/cases.jsonl`

- **Phase 3 — Zero-shot baseline** (hours 4–6)
- **Phase 4 — Stub HF Space** (hours 6–7)
- **Phase 5 — LoRA SFT on Gemma 3 12B-it** (hours 7–9, training runs through sleep)
- **Phase 6 — Re-eval + delta** (hours 14–16)
- **Phase 7 — Demo + README + BiP + ship** (hours 16–21)

See [CLAUDE.md](CLAUDE.md) for the full 24-hour table.

## Notes

- **Hardware:** MI300X / 192 GB VRAM, ROCm, gfx942. Plenty of headroom for hybrid (Gemma 12B + Qwen 7B + LoRA).
- **No prior assets on this box.** All `/Users/sankar/...` paths in [docs/ASSETS.md](docs/ASSETS.md) are on a different machine. We are greenfield.
- **vLLM is out of scope.** Use raw `transformers`. vLLM-on-ROCm debugging is its own day.
- **GRPO/RLVR is out of scope.** LoRA SFT is the primary fine-tune. GRPO loop ships as future-work code only.
- **Vision fine-tune from scratch is out of scope.** Gemma 3 12B-it native multimodal handles vision.
- **Skin-tone stratification is out of scope.** Stratify by *condition*; document the gap.
- **Evidence patterns** for the verify-gate: `evidence/*.txt`, `results/*.json`, `logs/*.log`, `adapters/*/adapter_config.json`, `data/cases.jsonl`. See [.claude/CLAUDE.md](.claude/CLAUDE.md).
