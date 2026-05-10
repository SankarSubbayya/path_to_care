# Path to Care — Detailed Submission Report

**AMD Developer Hackathon, May 2026**
GitHub: https://github.com/SankarSubbayya/path_to_care
HF Space: https://huggingface.co/spaces/sankara68/path-to-care-react
LoRA adapter (Phase-5 triage): https://huggingface.co/sankara68/path-to-care-triage-gemma4-lora
LoRA adapter (Phase-8 SCIN top-16): `adapters/scin-top16-gemma4-lora/` (90 MB safetensors; LoRA weights only — base Gemma 4 31B-it pulled separately from HF)

## 1. Project description

Path to Care is a multimodal, agentic decision-support system for rural healthcare in the Global South. It runs on a phone. It **never diagnoses** — it (1) ranks plausible skin conditions from an image as **top-3 with confidence**, (2) assigns an urgency level Red/Yellow/Green, (3) flags red signs that require professional evaluation, and (4) frames the decision as cost-benefit (distance, harvest pressure, transport cost, drug stock) so the patient can decide whether to spend half a day's wage on a clinic trip.

The cardinal rule — `core/cardinal_rule.py` — enforces this in *code*, not just prompts: every patient-facing string passes through a regex post-filter that rewrites diagnostic phrases ("you have cellulitis") into non-diagnostic equivalents ("signs are consistent with cellulitis"). Verified live during eval (case Y09 of `data/cases.jsonl` had "you have a fever" rewritten to "signs suggest a fever").

## 2. Architecture

```
            Patient (phone photo + typed narrative)
                       │
                       ▼
              Frontend — Gradio Space (HF) or React + FastAPI
                       │
                       ▼
        ┌── Orchestrator (DSPy-style ReAct) ──┐
        │              │              │              │
        ▼              ▼              ▼              ▼
  Image Classifier  SOAP Extractor  Village Context  Triage Reasoner
   (Gemma 4 31B)   (Qwen 2.5-7B)    (JSON KB)       (Gemma 4 31B + LoRA)
        │              │              │              │
        └────── shared loaded model ─────────────────┘
                       │
                       ▼
        Cardinal-rule rewriter (regex post-filter on every output)
                       │
                       ▼
        Top-3 conditions (with confidence) +
        Urgency (R/Y/G) + reasoning + cost-benefit framing + SOAP note
```

The image-classifier MCP and triage-reasoner MCP **share one loaded Gemma 4 31B model** (two prompts, one set of weights). SOAP extraction goes through Qwen-2.5-7B-Instruct. Village context is a deterministic JSON knowledge file, not an LLM call.

## 3. Hardware & runtime

- **AMD Instinct MI300X**, 192 GB HBM3, ROCm 6.3, gfx942
- **PyTorch 2.9.1+rocm6.3**, transformers 5.8, peft 0.19, dspy-ai 3.2, gradio 5.49, trl 1.4
- **uv-managed venv** with `[tool.uv.sources]` pinning torch / torchvision / torchaudio / pytorch-triton-rocm to the AMD ROCm 6.3 wheel index — prevents the "PyPI torch clobbers ROCm torch" footgun documented in `docs/COMPATIBILITY.md`
- **Production inference path**: `vllm/vllm-openai-rocm:v0.20.1` Docker image, AMD's recipe at https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html, OpenAI-compatible API on `:8000`
- **Inference dispatcher**: `core/llm.py` switches between in-process `transformers.generate` and vLLM HTTP via `PTC_INFERENCE=transformers|vllm` — same MCP code targets either backend

## 4. Track / prize coverage

| Track / prize | Deliverable |
|---|---|
| **Track 1 — Agents** | DSPy-style orchestrator + 5 MCP modules; cardinal-rule code-level enforcement; safety-net cross-checks; both in-process and HTTP backends |
| **Track 2 — Fine-tuning on AMD GPUs** | LoRA SFT on Gemma 4 31B-it on a single MI300X via ROCm 6.3 + peft 0.19. Two adapters: triage (Phase 5) + SCIN top-16 (Phase 8, +7.0 pp top-1) |
| **Track 3 — Vision & Multimodal** | Gemma 4 multimodal (image + text → urgency); two MCPs share weights to halve VRAM |
| **Qwen prize** | Qwen-2.5-7B-Instruct does the SOAP extraction — meaningful structural contribution to the pipeline |
| **HuggingFace prize** | Live Gradio Space + LoRA adapters on HF Hub |
| **Build-in-Public** | `docs/BIP_POST.md` — X/LinkedIn thread + ROCm/AMD-Cloud feedback writeup with concrete asks (vLLM ROCm wheel index; vLLM Gemma-4 multimodal-LoRA serving) |

## 5. Phase-5 LoRA (triage urgency on hand-built 30-case set)

**Setup**: 21 train + 9 holdout (suffix-stratified), targets are the structured `URGENCY: ... / REASONING: ... / RED_FLAGS_NOTED: ... / PATIENT_FRAMING: ...` block, lr=2e-4, r=16, 2 epochs. Wall **32 seconds**. Loss 3.90 → 0.58.

**Result on 30 cases**: 96.7% / 0.983 mean reward / **FN Red→Green = 0%**. The LoRA *matches* baseline — base Gemma 4 31B was already at the ceiling of this hand-built test set. The deliverable is the *training infrastructure* (clean LoRA on AMD MI300X) and the safety-metric preservation (no under-triage), not a delta number.

## 6. Phase-8 SCIN differential-diagnosis experiments

The credibility experiment. Used Google/Stanford SCIN (Skin Condition Image Network) — 5034 cases, dermatologist-labeled with confidence-weighted multi-label probability distributions, Fitzpatrick I-II / III-IV / V-VI metadata.

### 6.1 Negative result: 34 fine-grained classes (~35 train rows/class)

| Run | Hyperparams | Top-1 |
|---|---|---|
| Baseline (Gemma 4 31B-it) | n/a | 24.0% (8.2× chance) |
| LoRA, hard single-label | lr=2e-4, r=16, 3 ep | 13.0% (**−11.0 pp regression**) |
| LoRA, top-k probability | lr=1e-4, r=8, 1 ep | 21.0% (**−3.0 pp regression**) |

**Why it failed**: ~35 training rows per class is below the threshold where LoRA learns image-conditional class assignment vs. memorizes which classes to emit. The training loss collapsed to 0.04 by step 30 and stayed flat — token-level eval loss (0.045) lied because it rewarded chat-format memorization, not classification. Mode-collapse pattern: model stopped emitting common classes (Eczema 14→0 predictions, Allergic Contact Dermatitis 11→0) and over-emitted rare ones (Hypersensitivity 0→15). This finding **matches the boundary identified in https://github.com/SankarSubbayya/patient_advocacy_agent**: contrastive learning at 211 SCIN classes (~31 imgs/class) similarly degraded vs. vanilla SigLIP.

### 6.2 MedGemma 27B-it: medical pretraining doesn't move the needle

Tested whether swapping to `google/medgemma-27b-it` (medical-domain pretrained) lifts the baseline.

| Model | Top-1 | Top-3 set-match |
|---|---|---|
| Gemma 4 31B-it | 26.0% | 71.0% |
| MedGemma 27B-it | 23.0% | 69.0% |

Within noise on n=100. Both models cap around 70% top-3 set-match — that is the **ceiling of the base prior on this task**. The lever is not the model.

### 6.3 Positive result: top-16 most-occurring classes (~60 train rows/class)

Same 16 fine-grained SCIN conditions (top-16 by frequency, NOT coarse-merged), confidence-weighted multi-label targets, lr=1e-4, r=8, 1 epoch. Wall: 11.5 min training, 25.4 min eval.

| Metric | Baseline | Tuned | Δ |
|---|---|---|---|
| **Top-1 accuracy** | 28.0% | **35.0%** | **+7.0 pp** ✅ |
| Top-3 set-match | 71.0% | 68.0% | −3.0 pp |

**By Fitzpatrick:**

| Bucket | n | Δtop-1 | Δtop-3 |
|---|---|---|---|
| I-II | 38 | +5.3 pp | −15.8 pp |
| **III-IV** (largest) | 54 | **+9.3 pp** | **+7.4 pp** |
| V-VI | 8 | +0.0 pp | −12.5 pp |

**Per-class winners (top-1 F1):**

- Folliculitis: 0.37 → 0.52 (+0.15) — top-3 recall 0.29 → 0.65
- Acne: 0.50 → 0.63 (+0.13)
- Drug Rash: 0.44 → 0.56 (+0.12) — top-3 recall 0.33 → 0.67
- Eczema: 0.37 → 0.44 (+0.07)

**The trade-off** is honest: the LoRA makes the model more committed to its top-1 pick. Aggregate top-3 set-match drops 3 pp because some probability mass that was correctly distributed across alternate dermatologist labels is now consolidated into the most-confident pick. For the III-IV majority Fitzpatrick bucket both metrics win; for I-II it's a top-1-for-top-3 trade.

### 6.4 Loss curve comparison

Three runs preserved at:
- `docs/figures/scin_lora_loss.png` — failed 34-class single-label
- `docs/figures/scin_lora_topk_loss.png` — 34-class top-k targets (1 epoch, still regression)
- `docs/figures/scin_top16_lora_loss.png` — top-16 top-k targets (the working run)

The healthy run is qualitatively different: loss declines monotonically over 239 steps (train 6.6 → 0.07, eval 0.087 → 0.068), train-eval gap stays ≤ 0.02 throughout. The failed runs hit their loss floor at step 30 and flatlined.

### 6.5 Reproducibility commands

```bash
# 1. Curate top-16 from SCIN (download from gs://dx-scin-public-data already done)
PYTHONPATH=. .venv/bin/python -c "
import csv, json
from ast import literal_eval
from collections import Counter
labels = list(csv.DictReader(open('data/scin/scin_labels.csv')))
ctr = Counter()
for r in labels:
    s = r.get('weighted_skin_condition_label','')
    if not s or s in ('\"\"','{}'): continue
    try: d = literal_eval(s)
    except: continue
    if isinstance(d, dict) and d:
        ctr[max(d.items(), key=lambda kv: kv[1])[0]] += 1
top16 = [c for c,_ in ctr.most_common(16)]
json.dump(top16, open('data/scin/top16_classes.json','w'), indent=2)
"

# 2. Filter + sample stratified splits
PYTHONPATH=. .venv/bin/python scripts/sample_scin_dx.py \
  --in data/scin/top16_curated.jsonl \
  --train-out data/scin/top16_train.jsonl \
  --holdout-out data/scin/top16_holdout.jsonl \
  --class-field condition --min-cases 30 --train-cap 80 --holdout-cap 25

# 3. Build TRL chat data with top-k probability targets
PYTHONPATH=. .venv/bin/python scripts/build_scin_trl_topk.py \
  --train-in data/scin/top16_train.jsonl \
  --holdout-in data/scin/top16_holdout.jsonl \
  --train-out data/scin/top16_trl_topk_train.jsonl \
  --holdout-out data/scin/top16_trl_topk_holdout.jsonl \
  --classes-out data/scin/top16_classes_full.json

# 4. Train LoRA
PYTHONPATH=. .venv/bin/python scripts/lora_dx_multimodal.py \
  --train data/scin/top16_trl_topk_train.jsonl \
  --eval  data/scin/top16_trl_topk_holdout.jsonl \
  --output adapters/scin-top16-gemma4-lora \
  --epochs 1 --batch-size 1 --grad-accum 4 --lr 1e-4 \
  --lora-r 8 --lora-alpha 16

# 5. Eval (baseline + tuned in one process)
PYTHONPATH=. .venv/bin/python scripts/eval_scin_dx_topk.py \
  --in data/scin/top16_trl_topk_holdout.jsonl \
  --classes data/scin/top16_classes_full.json \
  --adapter adapters/scin-top16-gemma4-lora \
  --baseline-out results/scin_top16_topk_baseline.json \
  --tuned-out    results/scin_top16_topk_tuned.json \
  --limit 100
```

End-to-end on a single MI300X: ~45 min including model download.

## 7. Inference paths for the SCIN top-16 adapter

### 7.1 Path A — in-process Python (always works)

```bash
PYTHONPATH=. .venv/bin/python scripts/infer_scin_top16.py \
  --image data/scin/images/<id>.png \
  --symptoms "itchy red patch for 1 week" \
  --fitzpatrick I-II \
  --body-parts arm
```

Or in code:
```python
from scripts.infer_scin_top16 import load, predict
handle = load()  # ~30 s
topk, raw = predict(handle, "image.png", symptoms_text="itchy", fitzpatrick="III-IV")
# → [("Eczema", 0.55), ("Allergic Contact Dermatitis", 0.30), ("Insect Bite", 0.15)]
```

### 7.2 Path B — vLLM served LoRA on the AMD droplet

`bash scripts/vllm_serve.sh` and `--enable-lora --lora-modules scin-top16=/adapters/scin-top16-gemma4-lora` give an OpenAI-compatible HTTP API. Verified live (see logs of `ptc-vllm` container). Note the *Path B caveat* documented in `docs/COMPATIBILITY.md`: vLLM ROCm 0.20.1 + Gemma 4 multimodal LoRA had a silent-fallback issue with the 34-class adapter; the 16-class case is being verified — if it doesn't apply, in-process Path A is the canonical inference path until vLLM ROCm fixes Gemma 4 LoRA serving.

## 8. Honest caveats

- **Holdout n=100** for the SCIN comparison. Wide-ish confidence intervals on the per-class numbers; the +7.0 pp aggregate top-1 lift is the only finding I'd call confidently real.
- **Eval set is a single random sample** of the SCIN top-16 holdout. We didn't run k-fold; +7.0 pp could be 4–10 pp on a different fold.
- **No clinician validation** — labels are SCIN's own dermatologist confidence-weighted distributions, never reviewed by us. The cardinal rule prevents the system from claiming diagnoses regardless.
- **No real fieldwork**. The 30-case Phase-5 set is hand-built. The Phase-8 SCIN set is consumer-uploaded photos with known dataset bias toward US contributors.
- **Skin-tone Fitzpatrick V-VI is intrinsically sparse in SCIN** (8/100 holdout). Per-Fitzpatrick numbers on V-VI are illustrative, not statistically robust.
- **vLLM Gemma 4 multimodal LoRA serving** is not stable at ROCm 0.20.1; in-process peft is the documented fallback.

## 9. v1 → v2 roadmap

- Push the 16-class scheme + corrected hyperparams to the full 80-case SCIN test set with proper k-fold cross-validation
- Real-village fieldwork in Thiruvallur or Ranipet for the village-context layer
- Physician review of 20–30 outputs (single-blind)
- Skin-tone-stratified bias audit at scale (HAM10000 + DDI to complement SCIN's I-II/III-IV/V-VI)
- GRPO/RLVR triage tuning once TRL+ROCm is verified end-to-end
- Stand up vLLM serving with the LoRA adapter once vLLM Gemma 4 multimodal LoRA application is fixed upstream

## 10. What gets shipped

- Open-source code: https://github.com/SankarSubbayya/path_to_care
- Live demo: https://huggingface.co/spaces/sankara68/path-to-care-react (Next.js · Patient / Clinician / Audit tabs · camera capture · voice dictation)
- Two LoRA adapters (Phase 5 triage + Phase 8 SCIN top-16); `adapters/` directory committed
- This report, `docs/SCIN_DIFF_DX.md` design rationale, `docs/COMPATIBILITY.md` model-selection trail, `docs/EVALUATION.md` reward + stratification framework
- 68 unit + integration tests (`pytest tests/` — verified passing)
- BiP / ROCm feedback writeup at `docs/BIP_POST.md`
