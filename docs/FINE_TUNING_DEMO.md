# Fine-tuning demo: with vs. without

Honest framing first, then concrete demo steps.

## The honest framing

On our 30-case adversarial test set, **the LoRA-tuned model and the zero-shot baseline produce identical urgency calls on every case**:

| Metric                        | Zero-shot Gemma 4 31B | LoRA-tuned Gemma 4 31B | Δ |
|-------------------------------|-----------------------|------------------------|---|
| Mean reward                   | 0.983                 | 0.983                  | +0.0 |
| Exact-match urgency           | 96.7% (29/30)         | 96.7% (29/30)          | +0.0 |
| Within-1-level urgency        | 100.0%                | 100.0%                 | +0.0 |
| **FN Red→Green (cardinal)**   | **0.0% (0/10 Red)**   | **0.0% (0/10 Red)**    | **+0.0** |
| Holdout-only mean reward (n=9)| 1.000                 | 1.000                  | +0.0 |

The base Gemma 4 31B-it is **at the ceiling of this hand-crafted test set**. There's no room to improve on Red→Red, Green→Green, or 9 of the 10 Yellow cases. The single miss (P2C-Y03, mild contact dermatitis from a new soap, predicted Green when ground truth is Yellow) is also wrong in the tuned model — even though Y03 is in the *training* split, two epochs of LoRA on 21 examples didn't override the base model's prior on a genuinely borderline case.

**So why fine-tune at all? Three reasons, all visible:**

1. **Track 2 deliverable: a working AMD-stack fine-tune loop.** Our LoRA SFT converges in **32 seconds on a single MI300X**: loss curve 3.90 → 3.27 → 2.78 → 2.21 → 1.67 → 1.48 → 1.00 → 1.11 → 0.83 → 0.58. 45M trainable params (0.14% of the 31B base). Adapter saves cleanly to `adapters/triage-gemma4-lora/`. This is reproducible and forms the basis for v2 work.

2. **Tuning preserves safety.** It would have been very easy to LoRA-train a model that nudges urgencies down (because tuning on a small set shifts toward the ground-truth distribution) and increases the FN Red→Green rate. **Our tuned model's FN Red→Green rate stays at 0.0%.** Demonstrating "fine-tune doesn't break safety" is itself the deliverable on a 30-case set this saturated.

3. **The reasoning text *does* shift.** Two of 30 cases have meaningfully different `reasoning` and `patient_framing` text between baseline and tuned. The tuned output mirrors our LoRA training template ("Red flags noted: ...", "Delaying care risks rapid deterioration"). The LoRA learned the *style* of the training data even though the urgency calls were already correct.

This is what fine-tuning a near-ceiling-saturated model on a small set looks like, honestly. The story for the eval delta is the v2 roadmap (80-case set, harder ambiguous cases, skin-tone stratification, physician-reviewed labels).

## Concrete with vs. without comparison

Run this script to see per-case differences:

```bash
.venv/bin/python scripts/compare_per_case.py
```

It prints a side-by-side table: case_id / truth / base prediction / tuned prediction / reasoning-text-changed flag. On our run:

```
case_id    truth   base    tuned   reasoning_changed
P2C-R01    red     red     red     yes  ← LoRA-style "Red flags noted: ..." phrasing
P2C-R02    red     red     red     no
...
P2C-Y03    yellow  green   green   no   ← lone miss, both wrong (genuinely borderline)
P2C-Y09    yellow  yellow  yellow  no   ← cardinal-rule rewriter fired here on baseline
...
P2C-G10    green   green   green   no
```

Add `--full <case_id>` to dump the complete reasoning + framing text for a single case in both modes.

## Live demo on the HF Space (with vs. without)

The Space is wired to the MI300X-hosted vLLM container. Two ways to demonstrate:

**Mode A: Same Space, two clicks** — leave the "LoRA adapter path" textbox blank for the first run (zero-shot). Click "Run triage". Then put `adapters/triage-gemma4-lora` back in and click again. *Caveat:* the live vLLM container was not started with `--enable-lora`, so this currently has no effect — see Mode B.

**Mode B: Restart vLLM with `--enable-lora` (5 minutes)** — to make the live demo show a real LoRA delta:

```bash
docker rm -f ptc-vllm
# Mount the adapter into the container
docker run -d --name ptc-vllm \
  --group-add=video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --device /dev/kfd --device /dev/dri \
  -p 8000:8000 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v /root/path_to_care/adapters:/adapters \
  -e HF_HOME=/root/.cache/huggingface \
  --entrypoint /bin/bash \
  vllm/vllm-openai-rocm:v0.20.1 \
  -c 'vllm serve google/gemma-4-31B-it \
        --host 0.0.0.0 --port 8000 \
        --api-key ptc-demo-2026-amd \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.9 \
        --enable-lora --max-lora-rank 16 \
        --lora-modules triage=/adapters/triage-gemma4-lora'

# Then on the HF Space side:
HF_TOKEN=hf_... .venv/bin/python -c "
from huggingface_hub import HfApi
HfApi().add_space_variable('sankara68/path-to-care', 'PTC_VLLM_LORA_NAME', 'triage')
"
```

Now the Space's "LoRA adapter path" textbox is meaningful: blank → base Gemma 4, any value → LoRA-routed via vLLM `--lora-modules triage`. Compare the two responses live.

## What gets compared

For each case the demo shows:

- **Top-3 image candidates** — usually identical, sometimes different ordering
- **SOAP fields** — almost always identical (Qwen does this, no LoRA)
- **Village context** — always identical (deterministic JSON lookup)
- **Urgency** — identical in our 30-case set (base is at ceiling)
- **Reasoning** — *this is where the LoRA shows*; tuned tends toward "Red flags noted: ...", "Delaying care risks ..."
- **Patient framing** — same shift in 2 of 30 cases

## Files

- `results/baseline_metrics.json` — full zero-shot per-case scores + traces
- `results/tuned_metrics.json` — full tuned per-case scores + traces
- `evidence/delta_report.txt` — headline-table delta
- `evidence/stratified_report.txt` — by urgency, by perturbation, by red-flag severity
- `logs/lora_train.log` — the 32-second loss curve
- `adapters/triage-gemma4-lora/adapter_config.json` — proof of training run
- `adapters/triage-gemma4-lora/adapter_model.safetensors` — 180 MB trained adapter
- HF Hub: <https://huggingface.co/sankara68/path-to-care-triage-gemma4-lora>
