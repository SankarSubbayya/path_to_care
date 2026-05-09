# Build-in-Public post — draft

Two-deliverable BiP submission per [README.md](../README.md) Track requirements.

---

## Post 1 — X / LinkedIn thread (draft)

> 🧵 Built **Path to Care** in 24 hours for the @AIatAMD Developer Hackathon: a multimodal triage decision-support agent for rural healthcare. Never diagnoses — ranks possibilities, assesses urgency, frames barriers. Tag: @lablab
>
> 1/ The brief: a community health worker in a Tamil Nadu village. A patient with a swollen, infected foot. The system has to (a) identify red flags from a phone photo + narrative, (b) recommend Red/Yellow/Green urgency, (c) frame the trade-off in cost-of-time terms (₹180 round-trip vs. one day's wage of ₹350).
>
> 2/ Stack on AMD MI300X (192 GB VRAM, ROCm 6.3): **Gemma 4 31B-it** (multimodal, vision + triage) + **Qwen-2.5-7B-Instruct** (SOAP extraction). DSPy-style orchestrator wires four MCP-style modules. LoRA SFT on Gemma 4 produced the headline delta. transformers 5.x, peft 0.19, dspy 3.2.
>
> 3/ Hardest 24-hour decisions:
> - Gemma 3 → 401 (gated, no token).
> - Gemma 4 26B-A4B (MoE) → loads fine but `model.generate` hits `torch._grouped_mm` which has no ROCm path. Tried the dense **31B variant**, it works.
> - Pinned the ROCm wheel index in `pyproject.toml` `[tool.uv.sources]` after `pip install --ignore-installed transformers` clobbered our ROCm torch with a CPU build. Lesson learned the hard way.
>
> 4/ Cardinal-rule enforcement is in code, not just prompts. A regex post-filter rewrites "you have X" → "signs suggest X" on every model output. Logged. The image classifier never returns a single class — top-3 with confidence or a "non-diagnostic, please retake" fallback.
>
> 5/ Adversarial test set: 30 hand-crafted cases (10 R / 10 Y / 10 G) with perturbations — colloquial dialect, "no fever / I have fever" contradictions, off-distribution image references, neighbor's-opinion noise. The held-out 9 cases (suffix 08-10 each level) are the credibility load-bearer.
>
> 6/ Headline: zero-shot Gemma 4 31B → LoRA-tuned, **mean reward Δ = TODO_PP** on the held-out 9. False-negative Red→Green rate **TODO%** (cardinal safety metric — Green-when-actually-Red is the most dangerous failure). Real numbers in the README.
>
> 7/ Ship: HF Space (Gradio) replays the Rajan dialogue end-to-end. LoRA adapter on @huggingface Hub. Open-source repo + technical walkthrough. Build-in-Public + ROCm feedback writeup linked. AMD Dev Cloud is good — installer wheels for ROCm 6.3 work cleanly via uv `[tool.uv.sources]`.

## Post 2 — ROCm / AMD Developer Cloud feedback writeup

### What worked

- **MI300X via ROCm 6.3** — torch 2.9.1+rocm6.3 wheel, MI300X visible to `torch.cuda.is_available()` in <30 seconds, sample matmul on `cuda:0`.
- **`pip install` + the AMD wheel index** — `https://download.pytorch.org/whl/rocm6.3` resolves cleanly when fed to `[tool.uv.sources]` in `pyproject.toml`. uv routes `torch`, `torchvision`, `torchaudio`, `pytorch-triton-rocm` through it without complaint.
- **HBM3 throughput** — Gemma 4 31B-it bf16 inference (~62 GB resident) generates at ~3.7 seconds per 128-token completion on the cold path, ~3 seconds warm. Loaded the model once, served two MCP roles (image classifier + triage reasoner) from it. 192 GB headroom meant we never had to think about VRAM.

### What needed a workaround

- **`pytorch-triton-rocm` resolution in uv.** uv could not auto-route the transitive dependency through the AMD wheel index until we declared it explicitly in `pyproject.toml` and added a matching `[tool.uv.sources]` entry. Cost ~20 minutes of debugging. A note in the AMD/PyTorch docs about uv's strict-transitive behavior would help.
- **MoE on ROCm.** Gemma 4 26B-A4B-it loads fine on ROCm 6.3 / torch 2.9.1, but `model.generate(...)` raises `RuntimeError: grouped gemm is not supported on ROCM` from `transformers/integrations/moe.py` calling `torch._grouped_mm`. Currently any MoE that goes through that path is unusable on AMD. **Concrete ask:** a `torch._grouped_mm` ROCm implementation, or a transformers fallback path that doesn't require it.
- **Old wheels for ROCm 6.2.** The latest torch on the rocm6.2 wheel index is 2.5.1 (released Oct 2024). Gemma 4's modeling code requires `torch >= 2.6`. We had to upgrade the wheel index to ROCm 6.3 to get torch 2.9. Worked, but worth noting that the wheel index tier behind production (rocm6.2) lags the model-code requirements by about a year.

- **vLLM via pip on ROCm is broken; via Docker it works perfectly.** Three install paths tried:
  - `pip install vllm` (PyPI default): CUDA-built; pulls CPU `torch==2.11.0` over our ROCm 2.9.1. Fails: `libtorch_cuda.so: cannot open shared object file`.
  - `pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/0.20.1/rocm721` (vLLM's ROCm wheel index): pulls `torch==2.10.0+git…` built against ROCm **7.2.1**. Our driver is ROCm 6.2.x. Fails: `undefined symbol: ncclCommShrink`. ROCm 7.2.1 ships an RCCL API our 6.2 driver doesn't have.
  - `docker run vllm/vllm-openai-rocm:v0.20.1` (per AMD's MI300X recipe): **works**. Image bundles its own ROCm user-mode libs; isolated from host driver. Server up in 90 s, OpenAI-compatible, `--api-key` auth honored, `/v1/chat/completions` returns clean output. **Concrete ask:** publish the ROCm vLLM wheel on the same `download.pytorch.org/whl/rocm6.3` index that already hosts ROCm torch wheels — so `pip install vllm` is sticky to the user's existing torch build. Every new ROCm vLLM user runs this same gauntlet. The fix is one wheel publish.
- **`torch_dtype` deprecation in transformers 5.x.** Not an AMD thing per se but worth flagging: transformers 5.x deprecated `torch_dtype=` in favor of `dtype=`, and `tokenizer.apply_chat_template(..., return_tensors='pt')` no longer reliably tokenizes — must call with `tokenize=False` then tokenize separately. AMD-stack users encountering these will think it's a ROCm bug; it's a transformers 5.x change.

### What I did not have to debug

- ROCm driver compatibility — the rocm6.3 wheel runs against the rocm6.2 system driver without issue.
- bf16 — works on `gfx942` natively, no fp16 fallback needed.
- HF model downloads — fast, ~50 MB/s on the AMD Developer Cloud instance.

### Feedback summary

Positive: 24-hour multimodal-LLM hackathon submission shipped end-to-end on a single MI300X. The headroom (192 GB) made the architecture decisions pleasant rather than constrained.

Two specific gaps worth closing for the next cohort: (1) MoE forward-path support (`torch._grouped_mm`) on ROCm, and (2) clearer documentation for uv's transitive-dependency routing through wheel indices. Both are concrete, fixable, and would reduce the "first hour spent on environment instead of model work" tax that hits every new ROCm user.
