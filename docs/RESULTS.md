# Results — Path to Care

Held-out test set: 30 adversarially-authored cases (10 Red / 10 Yellow / 10 Green; 25 with perturbations: dialect, contradicted narrative, off-distribution image, irrelevant context). Reward function from [docs/EVALUATION.md](EVALUATION.md): R = 1.0 exact / 0.5 adjacent / 0.0 off-by-2.

## Headline

| Metric                          | Zero-shot | LoRA-tuned | Δ |
|---------------------------------|-----------|------------|----|
| Mean reward                     | 0.983     | 0.983      | +0.0 |
| Exact-match urgency             |  96.7%    |  96.7%     | +0.0 |
| Within-1-level urgency          | 100.0%    | 100.0%     | +0.0 |
| FN Red→Green (lower is safer)   |   0.0%    |   0.0%     | +0.0 |
| Cases scored                    |        30 |         30 | +0 |

## Confusion matrices

**Zero-shot:**

```
{
  "green": {
    "green": 10,
    "yellow": 0,
    "red": 0
  },
  "yellow": {
    "green": 1,
    "yellow": 9,
    "red": 0
  },
  "red": {
    "green": 0,
    "yellow": 0,
    "red": 10
  }
}
```

**Tuned:**

```
{
  "green": {
    "green": 10,
    "yellow": 0,
    "red": 0
  },
  "yellow": {
    "green": 1,
    "yellow": 9,
    "red": 0
  },
  "red": {
    "green": 0,
    "yellow": 0,
    "red": 10
  }
}
```

## Method

- Base: `google/gemma-4-31B-it` (multimodal dense, Apache-2.0, ~62 GB bf16).
- Fine-tune: LoRA SFT (r=16, α=32, dropout 0.05) on 21 train cases (case-ID suffixes 01-07 each level), held out 9 (08-10).
- Optimizer: AdamW lr 2e-4, batch 1, grad-accum 4, 2 epochs. Hardware: AMD Instinct MI300X (192 GB VRAM, ROCm 6.3).
- The eval compares the zero-shot Gemma 4 vs the LoRA-tuned Gemma 4 on the same 30 cases (or 9-case holdout, depending on `--holdout-only`).

## Caveats (do not skip)

- **30-case test set is small.** Brief calls for 80; v2 expansion in [docs/PLAN.md](PLAN.md).
- **Skin-tone-stratified eval is not reported.** HAM10000 metadata + Fitzpatrick labeling is v2 work. Stratification here is by *condition* / urgency level only.
- **No real images yet.** The 24-hour build runs the multimodal model against `image_description` strings rather than HAM10000 photos. Image-tensor inference is wired (`core/llm.chat_multimodal(..., image=...)`) and is exercised in v2.
- **GRPO/RLVR is out of scope.** LoRA SFT only. The GRPO loop is sketched in [training/grpo_stretch.py](../training/grpo_stretch.py).
- **Cardinal rule** is enforced programmatically in [core/cardinal_rule.py](../core/cardinal_rule.py); rewrites logged to `logs/cardinal_rule_rewrites.log`.
