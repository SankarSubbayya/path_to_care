# SCIN differential-diagnosis fine-tune — design + lessons

## What we're trying to demonstrate

That **fine-tuning Gemma 4 31B-it on SCIN dermatology images measurably improves differential-diagnosis quality** on a held-out subset, on a single AMD MI300X (Track 2 deliverable).

## What went wrong on the first attempt (top-1 single-label SFT)

We trained the LoRA to emit the highest-weighted condition from `weighted_skin_condition_label` as a bare class string (e.g., `"Eczema"`), with standard cross-entropy on the assistant's tokens.

Results on a 100-row in-process holdout:
- baseline top-1: **24.0%**
- LoRA-tuned top-1: **13.0%** (regression, −11 pp)

The training loss looked clean (0.04 plateau, eval_loss tracked train), but **token-level eval loss did not capture the actual classification task**. The model collapsed onto a few classes (`Hypersensitivity`, `Folliculitis`, `Pityriasis rosea`) and stopped emitting the most common training classes (`Eczema` 14→0 predictions, `Allergic Contact Dermatitis` 11→0, `Tinea` 8→0).

Per-class F1 movement (top movers):
| class | base.F1 | tuned.F1 | Δ |
|---|---|---|---|
| Eczema | 0.53 | 0.00 | **−0.53** |
| Allergic Contact Dermatitis | 0.32 | 0.00 | **−0.32** |
| Drug Rash | 0.27 | 0.00 | −0.27 |
| Acne | 0.40 | 0.67 | +0.27 (over-predicted) |

**Diagnosis: classic small-data LoRA mode collapse.** With ~35 train samples per class and lr=2e-4, r=16, 3 epochs, the LoRA over-fit the chat-format target distribution rather than the image→class mapping.

**Compounding problem**: SCIN data is intrinsically multi-label. `weighted_skin_condition_label` for a given case is a **probability distribution over conditions** that the up-to-three dermatologist labelers chose, weighted by their per-label confidence (1–5 scale). Example case schema:

```json
{
  "weighted_skin_condition_label": {
    "Inflicted skin lesions": 0.41,
    "Eczema": 0.41,
    "Irritant Contact Dermatitis": 0.18
  },
  "dermatologist_skin_condition_on_label_name": [
    "Inflicted skin lesions", "Eczema", "Irritant Contact Dermatitis"
  ],
  "dermatologist_skin_condition_confidence": [4, 4, 3]
}
```

By taking only `argmax(weighted)` as the training target, **we threw away both the probability distribution AND the alternates**. Training said "Eczema is right, Inflicted skin lesions is wrong, Irritant Contact Dermatitis is wrong" — which directly punishes valid clinical alternatives. **This also violates Path to Care's own cardinal rule** (`docs/PROJECT.md`, `core/cardinal_rule.py`): the project must always emit top-3 with confidence, never single-class.

## The corrected experiment

**Use the full SCIN diff-dx distribution as the training target.**

### New training target string

For the case above, the assistant target becomes:
```
Inflicted skin lesions (0.41); Eczema (0.41); Irritant Contact Dermatitis (0.18)
```

Sorted by weight descending. Standard cross-entropy on these tokens then teaches the model to:
1. Output the full diff-dx (top-3 with confidence) — matches our cardinal rule
2. Distribute probability mass across dermatologist-supported alternates rather than collapse to one
3. Respect relative weights (high-weight conditions appear first)

### New eval metrics

Two metrics, both stricter than what we had:

1. **Top-3 set match** — what the SCIN paper itself reports. The case scores 1 if **any** of the model's predicted conditions appears in `dermatologist_skin_condition_on_label_name`, else 0. Because SCIN intrinsically has multi-label disagreement, top-3 set-match is the honest target.
2. **Top-1 primary match** — argmax(weighted) match. Same as our old metric, kept for comparability.

Stratified by Fitzpatrick I-II / III-IV / V-VI / unknown.

### Hyperparameter changes (informed by mode-collapse failure)

| Param | Old | New | Why |
|---|---|---|---|
| `lr` | 2e-4 | **1e-4** | Slower learning; avoid early collapse |
| `lora_r` | 16 | **8** | Half the capacity; less room to memorize noise |
| `lora_alpha` | 32 | **16** | Maintains alpha/r ratio |
| `epochs` | 3 | **2** | Loss plateaued by step ~30 in the failed run; 3 epochs added no signal but allowed continued drift |
| `target_modules` | `q,k,v,o` (regex on language model) | unchanged | Avoid `Gemma4ClippableLinear` in vision tower |
| `dtype` | bf16 | unchanged | Working ROCm config |

`metric_for_best_model` stays `eval_loss` because we don't have an in-trainer top-3 metric — but the post-training eval is what we'll trust, not the trainer's eval_loss.

## Schema of the new TRL training row

```json
{
  "messages": [
    {"role": "system", "content": [{"type": "text", "text": "You are a dermatology classifier... Output the most likely top-3 conditions with confidence, e.g.: Eczema (0.6); Allergic Contact Dermatitis (0.3); Insect Bite (0.1)"}]},
    {"role": "user", "content": [
      {"type": "image", "image": "data/scin/images/<id>.png"},
      {"type": "text", "text": "Body parts: leg. Symptoms: itching. Fitzpatrick: III-IV. ..."}
    ]},
    {"role": "assistant", "content": [{"type": "text", "text": "Eczema (0.41); Inflicted skin lesions (0.41); Irritant Contact Dermatitis (0.18)"}]}
  ],
  "case_id": "SCIN-<id>",
  "primary_condition": "Eczema",
  "weighted_label": {"Eczema": 0.41, ...},
  "fitzpatrick_bucket": "III-IV",
  "image_path_local": "data/scin/images/<id>.png"
}
```

## Files

- `scripts/build_scin_trl_topk.py` — builds the new TRL JSONL with multi-condition targets from `weighted_skin_condition_label`.
- `scripts/eval_scin_dx_topk.py` — eval with top-3 set-match + top-1-primary-match + Fitzpatrick stratified.
- `scripts/lora_dx_multimodal.py` — unchanged training script, called with new hyperparameters and the new TRL data.

## Outputs

- `data/scin/dx34_trl_topk_train.jsonl` — new training data
- `data/scin/dx34_trl_topk_holdout.jsonl` — new holdout data
- `adapters/scin-dx34-gemma4-topk-lora/` — new adapter
- `logs/scin_lora_topk_train.jsonl` — per-step loss
- `results/scin_dx34_topk_baseline.json` — zero-shot eval with topk targets/metrics
- `results/scin_dx34_topk_tuned.json` — LoRA-tuned eval
- `evidence/scin_topk_delta.txt` — headline + per-class deltas
- `evidence/scin_topk_fitzpatrick.txt` — stratified
- `docs/figures/scin_lora_topk_loss.png` — loss curve
- `docs/figures/scin_topk_confusion.png` — top-1 confusion
- `docs/figures/scin_topk_pr.png` — per-class precision/recall

## Why this is the right experiment to run *now*

1. Uses **all** of SCIN's signal (the dermatologist confidence distribution we paid for and curated).
2. Matches the **SCIN paper's own evaluation** (top-3 set-match), so our numbers are comparable to published baselines.
3. Matches the **Path to Care cardinal rule** (top-3 + confidence, never single class).
4. Addresses the **specific failure mode** we observed (mode collapse on hard single-label targets).
5. Doesn't require switching base model (avoids 3+ hr of MedGemma debugging).
6. Likely to either succeed (clean delta) or fail informatively (we'll know whether the issue was the target format vs the model capacity).

## What would still fail this

- If 34 classes with ~35 samples/class is fundamentally too few for SCIN's image complexity, even soft-label training won't fix it. Backup plan: run the same recipe at 16 coarse classes (~175 samples/class).
- If the SCIN dermatologist labels are noisy enough that the "right" answer is genuinely indeterminate, we'd see a baseline near top-3 set-match ≈ 50–60% with no fine-tune lift. That would be a SCIN-dataset finding, not a Path-to-Care failure.
