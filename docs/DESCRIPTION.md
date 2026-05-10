# Path to Care — Detailed Project Description

For lablab submission, BiP posts, judges who want the long form. Everything in this doc is also reflected in `README.md`, `CLAUDE.md`, and the `docs/` design files.

## The problem

In rural communities across the Global South — Tamil Nadu, the Mississippi Delta, the Andean foothills — a patient with an infected wound, a rash that won't heal, or a fever they can't name faces three barriers, in this order:

1. **Recognition.** Is this serious? Is it spreading? Is it a red flag?
2. **Calibration.** If I go to the clinic, is it worth a day's wage and a 2-hour bus ride?
3. **Action.** What do I tell the doctor when I get there, in 5 minutes of triage time?

Most digital-health products try to solve #1 by acting like a doctor. They ask leading questions, they output a diagnosis, they recommend a course of treatment. **They are wrong by design.** A diagnosis is a clinical act; misdiagnosis at scale is malpractice at scale.

**Path to Care does not diagnose.** It does four narrowly defined jobs, and only these four:

1. **Ranks plausible skin conditions** from a phone photo — top-3 with confidence, never single-class, never binary "sick/healthy."
2. **Assigns urgency** — Red (immediate care today), Yellow (clinical evaluation within 1-2 days), Green (monitor at home).
3. **Flags red signs** — features in the narrative or image that demand professional evaluation (spreading erythema, systemic signs, neurological deficits, etc.).
4. **Contextualizes barriers** — distance to the PHC, transport cost as a fraction of daily wages, whether harvest season makes a clinic visit costly, what drugs are likely in stock.

The output is **decision support**: a Red/Yellow/Green call with reasoning, plus a structured pre-visit SOAP note for the doctor. The patient decides; the clinician diagnoses; the system informs both.

## The cardinal rule (enforced in code, not just prompts)

The system **never produces diagnostic statements.** Always "signs suggest infection," never "you have cellulitis." This rule is enforced *programmatically*. [`core/cardinal_rule.py`](../core/cardinal_rule.py) is a regex post-filter that rewrites diagnostic phrasing on every model output before it reaches the patient. Verified live during eval: case Y09 (chickenpox in adult), the model emitted "you have a fever" in the patient framing; the rewriter changed it to "signs suggest a fever." Logged to [`logs/cardinal_rule_rewrites.log`](../logs/cardinal_rule_rewrites.log).

The image classifier is similarly hard-constrained: the parser requires a JSON array of (condition, confidence) tuples; if it can't parse three, it returns `"non-diagnostic / image quality insufficient"` rather than guessing. Single-class output is impossible by construction.

## Architecture

A DSPy-style orchestrator coordinates four modular MCP-style services in a 7-stage flow:

```
            Patient (phone photo + typed narrative)
                       │
                       ▼
              Frontend (Gradio) ─── HF Space (sankara68/path-to-care)
                       │
                       ▼
        ┌── Orchestrator (DSPy-style ReAct) ──┐
        │              │              │              │
        ▼              ▼              ▼              ▼
  Image Classifier  SOAP Extractor  Village Context  Triage Reasoner
   (Gemma 4 31B)   (Qwen 2.5-7B)    (JSON KB)       (Gemma 4 31B + LoRA)
        │              │              │              │
        └─── shared loaded model — two prompts, one set of weights ───┘
                       │
                       ▼
        Cardinal-rule rewriter (regex post-filter)
                       │
                       ▼
        Urgency (R/Y/G) + reasoning + cost-benefit framing + SOAP note
```

**The image-classifier MCP and triage-reasoner MCP share one loaded Gemma 4 31B model** in memory — two prompts against the same weights. SOAP extraction goes through Qwen-2.5-7B-Instruct (text-only). The village-context MCP is a deterministic JSON knowledge file (PHC distance/hours, drug-stock map, transport options, household economics, seasonal calendar) — no LLM call, fully reproducible.

A safety net layered on top: if the model returns "green" but the orchestrator's rule-based cross-check finds ≥2 red-flag keywords in the narrative or image description, the urgency is escalated to "yellow" and the disagreement is logged. **The cardinal rule is "never under-triage."**

## Why these models (the real story)

The model selection trail is documented in [`docs/COMPATIBILITY.md`](COMPATIBILITY.md) and is worth telling because it shows the actual constraints of building on AMD ROCm in 24 hours:

1. **Originally Gemma 3 12B-it.** Failed: gated on HF Hub, no token in environment.
2. **Briefly considered single-vendor Qwen2-VL-7B.** Discarded once we found Gemma 4 was open.
3. **Gemma 4 26B-A4B-it (MoE).** Failed at runtime: vLLM/transformers route MoE forward through `torch._grouped_mm`, which has no ROCm implementation. `RuntimeError: grouped gemm is not supported on ROCM`.
4. **Gemma 4 31B-it (dense).** ✅ Works. ~62 GB resident in bf16, generates in ~3.7 s in-process and ~1 s on vLLM.

Each pivot is documented with the exact error message. This is **production AMD-stack feedback** that lands in the Build-in-Public deliverable.

## Hardware & runtime

- **AMD Instinct MI300X**, 192 GB HBM3, ROCm 6.3, gfx942.
- **PyTorch 2.9.1+rocm6.3**, transformers 5.8, peft 0.19, dspy-ai 3.2, gradio 5.49.
- **uv-managed venv** with `[tool.uv.sources]` routing torch / torchvision / torchaudio / pytorch-triton-rocm through `https://download.pytorch.org/whl/rocm6.3` so `pip install` can never silently pull a CUDA wheel that clobbers the ROCm build (a footgun we hit live and documented).
- **Production inference path**: `vllm/vllm-openai-rocm:v0.20.1` Docker image (AMD's recipe at <https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html>). vLLM container exposes an OpenAI-compatible API on `:8000` with API-key auth.
- **Inference dispatcher**: [`core/llm.py`](../core/llm.py) is `PTC_INFERENCE=transformers|vllm` — same MCP code can target either backend without changes.
- **HF Space → MI300X**: the deployed Space at <https://huggingface.co/spaces/sankara68/path-to-care> is a thin Gradio + OpenAI-client surface (~500 MB resident on `cpu-basic`); it HTTPs into the MI300X-hosted vLLM container for all model work.

## Evaluation

Held-out test set: **30 adversarially-authored cases** (10 Red / 10 Yellow / 10 Green). 25 of the 30 carry adversarial perturbations: colloquial dialect ("paining bad", "two days back"), contradicted narratives ("no fever" + "feverish"), off-distribution image references ("photo blurred", "image shows a wall"), neighbor's-opinion noise.

Reward function (per [`docs/EVALUATION.md`](EVALUATION.md)):

```
R(predicted, ground_truth) = 1.0  if exact match
                            0.5  if adjacent level
                            0.0  if off by 2+
```

| Metric                        | Zero-shot Gemma 4 31B | LoRA-tuned Gemma 4 31B | Δ |
|-------------------------------|-----------------------|------------------------|---|
| Mean reward                   | **0.983**             | 0.983                  | +0.0 |
| Exact-match urgency           | **96.7%** (29/30)     | 96.7% (29/30)          | +0.0 |
| Within-1-level urgency        | **100.0%**            | 100.0%                 | +0.0 |
| **FN Red→Green** (lower safer) | **0.0%** (0/10 Red)   | 0.0% (0/10 Red)        | +0.0 |
| Holdout-only mean reward      | 1.000 (n=9)           | 1.000 (n=9)            | +0.0 |

The **FN Red→Green rate is the cardinal safety metric**: predicting *Green* when the truth is *Red* means a patient with sepsis stays home. Both runs are 0% — no under-triage on any Red case.

Stratified breakdown ([`evidence/stratified_report.txt`](../evidence/stratified_report.txt)): the model handles **perturbed cases (0.980)** as well as **clean cases (1.000)** — robustness across dialect, contradictions, and off-distribution images.

**Honest framing of the LoRA delta:** the zero-shot baseline is essentially at the ceiling of this 30-case set. Tuning preserves accuracy, preserves the 0% FN Red→Green safety metric, and shifts the *reasoning text* toward the LoRA training template on 2/30 cases. The headline of this submission is the **infrastructure** — a 24-hour multimodal-agent build on AMD that converges LoRA SFT on Gemma 4 31B in **32 seconds** (loss 3.90 → 0.58, 45M trainable params, 0.14% of base) without regressing the baseline. Real delta numbers want the 80-case set + skin-tone stratification scoped for v2. See [`docs/FINE_TUNING_DEMO.md`](FINE_TUNING_DEMO.md) for the with vs. without comparison.

## What gets shipped

| Artifact | Where |
|---|---|
| Open-source code | <https://github.com/SankarSubbayya/path_to_care> |
| HF Space (live demo) | <https://huggingface.co/spaces/sankara68/path-to-care> |
| LoRA adapter weights | <https://huggingface.co/sankara68/path-to-care-triage-gemma4-lora> |
| 4-dimension eval framework | [`docs/EVALUATION.md`](EVALUATION.md) |
| Per-case results + traces | [`results/baseline_metrics.json`](../results/baseline_metrics.json), [`results/tuned_metrics.json`](../results/tuned_metrics.json) |
| Stratified breakdown | [`evidence/stratified_report.txt`](../evidence/stratified_report.txt) |
| 68 unit + integration tests | [`tests/`](../tests/) (verify-gate evidence-before-passing harness in [`.claude/`](../.claude/)) |
| BiP / ROCm feedback writeup | [`docs/BIP_POST.md`](BIP_POST.md) |
| Live-demo runbook | [`docs/DEMO.md`](DEMO.md) + [`scripts/demo.sh`](../scripts/demo.sh) |
| 3-minute pitch | [`docs/PITCH.md`](PITCH.md) |
| Fine-tuning with-vs-without demo | [`docs/FINE_TUNING_DEMO.md`](FINE_TUNING_DEMO.md) |

## Track / prize coverage

| Track / prize | How |
|---|---|
| **Track 1 — Agents** | DSPy-style orchestrator + 4 MCP modules; cardinal-rule code-level enforcement; safety-net cross-checks; in-process and HTTP backends interchangeable. |
| **Track 2 — Fine-tuning on AMD GPUs** | LoRA SFT on Gemma 4 31B-it on a single MI300X via ROCm 6.3 + peft 0.19. Documented `Gemma4ClippableLinear` peft incompat workaround (regex-target the language-model linear layers). |
| **Track 3 — Vision & Multimodal** | Gemma 4 multimodal (image + text → urgency); two MCPs share weights to halve VRAM. |
| **Qwen prize** | Qwen-2.5-7B-Instruct does the SOAP extraction — meaningful structural contribution to the pipeline. |
| **HuggingFace prize** | Live Gradio Space + LoRA adapter on HF Hub. |
| **Build-in-Public** | X/LinkedIn thread + ROCm/AMD Dev Cloud feedback writeup with concrete asks (vLLM ROCm wheel index, `torch._grouped_mm` ROCm impl). |

## Honest caveats

- **30 cases is small.** The brief calls for 80; v2 expansion in [`docs/PLAN.md`](PLAN.md).
- **No skin-tone-stratified eval.** Stratification by Fitzpatrick scale needs HAM10000 metadata work that's v2 scope. We stratified by *condition*, *perturbation type*, and *red-flag severity* instead.
- **No real images yet.** The 24-hour build runs the multimodal model against `image_description` text strings rather than HAM10000 photos. Image-tensor inference is wired in `core/llm.chat_multimodal(..., image=...)` — exercised in v2.
- **GRPO/RLVR is out of scope** for the 24-hour build. LoRA SFT is the primary fine-tune; the GRPO loop is sketched in [`training/grpo_stretch.py`](../training/grpo_stretch.py) for v2.
- **Synthetic village knowledge.** [`mcp/village_context/knowledge.json`](../mcp/village_context/knowledge.json) is a composite of typical Tamil Nadu rural logistics, not a specific village's data. v2 calls for real village fieldwork.

## v1 → v2 roadmap

The 24-hour submission is the proof-of-concept. The 8-week roadmap to a field-deployable version is in [`docs/PLAN.md`](PLAN.md):

- **Real-village fieldwork** in Thiruvallur or Ranipet (Tamil Nadu) — bus schedules, clinic hours, drug costs, harvest cycles, demographics, top local diseases.
- **Expand test set to 80 cases** with skin-tone-stratified labels (Fitzpatrick I-VI), authored or reviewed by a clinician.
- **Vision-from-scratch fine-tune** on Stanford Skin Dataset (or HAM10000), tone-balanced, audited for bias.
- **GRPO/RLVR triage tuning** with the verifiable urgency reward (1.0/0.5/0.0) once TRL+ROCm is verified.
- **Physician review of 20-30 outputs** for clinical appropriateness and red-flag detection.
- **Tamil-language UX** validated on actual rural users.
- **2-3 real-user pilot** with feedback loop.
