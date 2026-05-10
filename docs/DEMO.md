# Live demo runbook — Path to Care

Three things to show, in this order. Each is verified live and copy-pasteable.

## 0. One-line all-in-one

```bash
bash scripts/demo.sh
```

That runs sections 1, 2, and 3 in sequence. Each section also runs standalone:

```bash
bash scripts/demo.sh vllm    # vLLM on MI300X
bash scripts/demo.sh rajan   # orchestrator + 4 MCPs
bash scripts/demo.sh space   # HF Space probe
```

## 1. vLLM Gemma 4 31B-it on MI300X (Track 2 + Track 3)

The headline AMD piece. Show that:
- Docker container is running the AMD-published `vllm/vllm-openai-rocm:v0.20.1`
- API key auth works (with key → 200; without → 401)
- Gemma 4 31B-it (multimodal, dense, 62 GB resident) generates top-3 with confidence in <1 s

```bash
docker ps --filter name=ptc-vllm
curl http://localhost:8000/v1/models   -H "Authorization: Bearer ptc-demo-2026-amd"
curl http://localhost:8000/v1/models                                    # 401
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer ptc-demo-2026-amd" \
  -H "Content-Type: application/json" \
  -d '{
    "model":"google/gemma-4-31B-it",
    "messages":[{"role":"user","content":"List three plausible skin conditions for a swollen reddened ankle with a small puncture wound. Confidences 0-1. Do not diagnose."}],
    "max_tokens":80,"temperature":0
  }'
```

Expected response (verified): `"Cellulitis [0.8] / Staphylococcal skin infection [0.6] / Contact dermatitis [0.3]"` — top-3 + confidence, no diagnostic phrases.

**Talking points:**
- 192 GB MI300X / ROCm 6.3 / `gfx942`.
- 39 s weight load → 62 GB resident → first token in ~1 s.
- vLLM Docker is the AMD-blessed install path; the PyPI `pip install vllm` wheel is CUDA-only and clobbers ROCm torch (documented in `docs/COMPATIBILITY.md`).

## 2. Rajan dialogue end-to-end (Track 1)

The orchestrator runs the 7-stage flow. Show that all four MCPs cooperate and the **cardinal rule is enforced in code**, not just the prompt.

```bash
# Default (in-process transformers — same backend the eval ran on)
.venv/bin/python scripts/orchestrator_smoke.py

# Or via vLLM (faster, OpenAI-compatible)
PTC_INFERENCE=vllm .venv/bin/python scripts/orchestrator_smoke.py
```

Each run writes `evidence/orchestrator_smoke.txt`. Read it during the demo:

```
verdict: PASS
case_id: P2C-R01
truth:   red
pred:    red
image_top3: [
  {"condition":"cellulitis", "confidence":0.8},
  {"condition":"deep tissue infection", "confidence":0.15},
  {"condition":"stasis dermatitis", "confidence":0.05}
]
village_blurb: PHC is 18 km away; round-trip transport is ₹180
              (≈51.4% of a daily wage of ₹350). District hospital is 65 km.
              Harvest is active — leaving the field costs visible income today.
reasoning: 'Signs suggest a rapidly spreading infection... Immediate clinical
           intervention is required to prevent permanent limb damage.'
patient_framing: 'While I know the harvest is important and the trip is
                expensive, this infection is spreading fast... If we treat it
                today, we can save your foot and get you back to the fields
                much sooner than if we wait.'
```

**Talking points:**
- Image classifier outputs top-3 + confidence; never single-class. Validation in code: bad parse → low-confidence "non-diagnostic" fallback, not a guess.
- Cardinal-rule rewriter (`core/cardinal_rule.py`) regex-rewrites diagnostic phrases. Verified live: case Y09 said "you have a fever" — rewriter changed it to "signs suggest a fever". Logged to `logs/cardinal_rule_rewrites.log`.
- Village context = barriers (cost, distance, harvest pressure) — converts clinical urgency into *practical* urgency framed as cost-benefit.

## 3. HF Space (HuggingFace prize)

Live URL: **https://huggingface.co/spaces/sankara68/path-to-care**

- Open the Space in a browser. Show the Rajan dialogue prefilled.
- Click "Run triage". Free Space hardware uses Gemma 4 **E4B** + Qwen 2.5-**1.5B** (set via Space variables `PTC_GEMMA4_ID`, `PTC_QWEN_ID`) so it fits.
- For the full 31B numbers, point reviewers at `docs/RESULTS.md` (run on MI300X).

**Backstop if the Space is slow:** show the recorded MI300X output (in `evidence/orchestrator_smoke.txt` or `results/baseline_metrics.json`'s first trace) and the GitHub repo. The Space is the demo surface; the **headline numbers** live in the repo.

## 4. Headline numbers

```
                              Zero-shot   LoRA-tuned   Δ
Mean reward                    0.983       0.983       +0.0
Exact-match urgency            96.7%       96.7%       +0.0
Within-1-level urgency        100.0%      100.0%       +0.0
FN Red→Green (lower safer)      0.0%        0.0%        +0.0
```

**Honest framing**: zero-shot Gemma 4 31B is essentially at the ceiling of this hand-crafted 30-case test set. The LoRA-SFT loop converges in **32 seconds** on MI300X (loss 3.90 → 0.58, 45M trainable params, 0.14% of base) and **does not regress** the baseline (no false negatives, FN Red→Green stays 0%). The submission story is the **infrastructure** — 24-hour multimodal-agent build on AMD — not a tuning delta. Real delta numbers want the 80-case set + skin-tone stratification scoped for v2.

Stratified breakdown (in `evidence/stratified_report.txt`):
- By urgency: Red 1.000, Green 1.000, Yellow 0.950 (the lone miss is P2C-Y03, mild contact dermatitis)
- By perturbation: clean 1.000, perturbed 0.980 — model handles perturbations as well as clean cases
- By red-flag severity: high (≥3) 1.000, low 1.000, none 0.962

## 5. Architecture in one slide

```
            Patient (phone camera + text)
                       │
                       ▼
              Frontend (Gradio)  ──→  HF Space (sankara68/path-to-care)
                       │
                       ▼
        ┌── Orchestrator (DSPy-style) ──┐
        │              │              │              │
        ▼              ▼              ▼              ▼
  Image Classifier  SOAP Extractor  Village Context  Triage Reasoner
   (Gemma 4 31B)   (Qwen 2.5-7B)    (JSON KB)       (Gemma 4 31B + LoRA)
        │              │              │              │
        └──────────────┴──── shared loaded model ────┘
                       │
                       ▼
        Cardinal-rule rewriter (regex post-filter)
                       │
                       ▼
        Urgency (R/Y/G) + reasoning + cost-benefit framing
```

`docs/VLLM_SERVE.md` covers the production path (Docker, AMD-blessed). `docs/COMPATIBILITY.md` covers the model-selection trail (Gemma 3 gated → Gemma 4 26B-A4B MoE blocked on ROCm → Gemma 4 31B dense, works).

## 6. If something breaks during the demo

| Symptom | Quick recovery |
|---|---|
| `curl ... /v1/models` returns nothing | `docker ps -a` to see if container exited; restart with `bash scripts/vllm_serve.sh` |
| Gradio Space is loading models for the first time and is slow | Show the local `make frontend` instead, or jump to section 4 (numbers) |
| `orchestrator_smoke.py` is slow on cold start | It loads ~77 GB of weights; show `evidence/orchestrator_smoke.txt` from a prior run |
| `pytest` reports failures | `.venv/bin/pytest tests/ -q` — last verified 68/68 passing (`evidence/pytest.txt`) |
