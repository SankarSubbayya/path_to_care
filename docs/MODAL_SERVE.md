# Modal serve — Path to Care fallback inference path

The **AMD MI300X droplet** running `vllm/vllm-openai-rocm:v0.20.1` (see [VLLM_SERVE.md](VLLM_SERVE.md)) remains the **primary and canonical** inference path. The +7.0 pp SCIN top-16 win was produced there. This Modal deployment exists as a hot fallback that the HF Space can swap to via a single env-var change if the droplet hiccups.

## Architecture

```
HF Space (Next.js)                            HF Space (Next.js)
        │                                              │
        │ PTC_VLLM_GEMMA4_URL=                         │ PTC_VLLM_GEMMA4_URL=
        │ http://165.245.137.117:8000/v1               │ https://sankara68--ptc-gemma4-serve.modal.run/v1
        ▼                                              ▼
AMD MI300X · ROCm 6.3                          NVIDIA H100-80GB · CUDA 12.4
vllm/vllm-openai-rocm:0.20.1                   vllm 0.6.3.post1
google/gemma-4-31B-it + scin-top16             google/gemma-4-31B-it + scin-top16

      PRIMARY (canonical)                          FALLBACK (warm)
```

Same model, same LoRA, same OpenAI-compat surface, same API key. **The application code does not know which backend it's talking to** — the swap is a pure URL change in the Space settings.

## One-time setup

**Important: do NOT `uv add modal` into this project.** Modal's `protobuf<7`
pin conflicts with our `protobuf>=7` transitive requirement (from
`transformers 5.x` / `gradio 5.x`). The clean fix is to run Modal in an
isolated env. `uvx` does this for free — Modal's local CLI is just a deploy
client; the real work happens in Docker images Modal builds on its own side,
which never see your project venv.

```bash
# 1. Authenticate Modal (browser flow)
uvx modal token new

# 2. Register the HF token as a Modal Secret (the base Gemma 4 weights are
#    gated for some accounts; the LoRA repo is public, but using a token still
#    avoids rate-limited pulls on cold start)
uvx modal secret create ptc-hf HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

If `uvx` isn't available (very old `uv`), fall back to `pipx`:

```bash
pipx install modal
modal token new
modal secret create ptc-hf HF_TOKEN=hf_xxx
```

## Deploy

```bash
cd /root/path_to_care
uvx modal deploy deploy/modal/serve.py
```

First deploy takes ~15-20 min because Modal builds the CUDA image and pulls **both** the base model (~62 GB) and the LoRA (~90 MB) into the image cache. Subsequent deploys are seconds — only the diff of `serve.py` is shipped.

Output ends with a URL like:

```
✓ Created web function serve => https://sankara68--ptc-gemma4-serve.modal.run
```

That URL is your fallback. Append `/v1` for the OpenAI-compat base.

## Wire to the HF Space

In the Space settings (https://huggingface.co/spaces/sankara68/path-to-care-react/settings):

| Variable | Value |
|---|---|
| `PTC_VLLM_GEMMA4_URL` | `https://sankara68--ptc-gemma4-serve.modal.run/v1` |
| `PTC_VLLM_GEMMA4_MODEL_ID` | `google/gemma-4-31B-it` (unchanged) |
| `PTC_VLLM_API_KEY` | `ptc-demo-2026-amd` (unchanged — same key as the droplet) |

Restart the Space (Settings → "Factory reboot" or just trigger a rebuild). The API route in [frontend-next/src/app/api/triage/route.ts](../frontend-next/src/app/api/triage/route.ts) reads these env vars and will start hitting Modal.

To **flip back to AMD**, set `PTC_VLLM_GEMMA4_URL` back to `http://165.245.137.117:8000/v1` and restart.

## Smoke test the Modal endpoint directly

```bash
# /v1/models — should list base + LoRA
curl -sS https://sankara68--ptc-gemma4-serve.modal.run/v1/models \
     -H "Authorization: Bearer ptc-demo-2026-amd"

# /v1/chat/completions — base
curl -sS -X POST https://sankara68--ptc-gemma4-serve.modal.run/v1/chat/completions \
     -H "Authorization: Bearer ptc-demo-2026-amd" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "google/gemma-4-31B-it",
       "messages": [{"role": "user", "content": "Reply with a single word: ok"}],
       "max_tokens": 8
     }'

# /v1/chat/completions — LoRA
curl -sS -X POST https://sankara68--ptc-gemma4-serve.modal.run/v1/chat/completions \
     -H "Authorization: Bearer ptc-demo-2026-amd" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "scin-top16",
       "messages": [{"role": "user", "content": "Reply with a single word: ok"}],
       "max_tokens": 8
     }'
```

## Cost / scale-to-zero (pure cold-start workflow)

Default GPU is **A100-80GB** (cheapest 80 GB option that fits Gemma 4 31B bf16). Approximate hourly rates — check [modal.com/pricing](https://modal.com/pricing) for the current numbers:

| GPU | ≈ $/hr | $30 credit gets you | Latency |
|---|---|---|---|
| **A100-80GB (default)** | ~$2 | ~14 hours | ~2-3 s / req (after warm) |
| H100-80GB (optional) | ~$5 | ~6 hours | ~1-1.5 s / req (after warm) |

**This config is set up for pure "wake-up-on-demand" usage** — no pre-warming, no idle burn:

- `min_containers` is **not set** (commented out in `serve.py`). The container is asleep by default and only spins up when a request arrives.
- `scaledown_window=300` — once awake, the container stays warm for 5 min so sequential demo questions don't each pay cold-start. After 5 min idle it scales back to zero. Adjust to `60` if you want to be more aggressive.
- **First request after sleep: ~60-90 s cold-start delay.** The container has to load the 62 GB Gemma weights into VRAM and start the vLLM OpenAI server. That delay is billed (you pay for the cold-start time as GPU seconds) — at A100 rates, **one cold-start ≈ $0.04**.
- **Subsequent requests in the same 5-min warm window: ~2-3 s each, ≈ $0.002 / request.**
- First-ever deploy spends ~15-20 min on image build + 62 GB weight pull. That runs on Modal's CPU builders → effectively free against your GPU budget.

**Cost arithmetic for $30 with pure scale-to-zero:**

- $0.04 / cold-start × 750 cold-starts = $30. So you have effectively unlimited demo headroom for a hackathon.
- A pessimistic "10-minute demo with one cold-start + 10 requests": ~$0.06.
- The only way to burn credits unintentionally is to leave `min_containers=1` on. Don't.

**Watch live usage:** https://modal.com/apps — shows current container state, per-app spend, and lets you stop a container manually if you ever see it stuck warm.

## Caveat — vLLM Gemma 4 multimodal LoRA serving

We documented this for the ROCm container in [SCIN_DIFF_DX.md](SCIN_DIFF_DX.md) and it applies on CUDA too: in some vLLM versions, requesting `model=scin-top16` for multimodal (image+text) prompts silently falls back to the base model rather than applying the LoRA. The adapter config in `adapters/scin-top16-gemma4-lora/adapter_config.json` ships **list-form** `target_modules` (not regex) to maximize compatibility, but if you see degraded behavior:

1. Send a text-only request to `model=scin-top16` and a text-only request to `model=google/gemma-4-31B-it`. If outputs differ, the LoRA is firing on text but maybe not on multimodal.
2. The **canonical** SCIN top-16 evaluation path is the in-process `peft` approach in [`scripts/infer_scin_top16.py`](../scripts/infer_scin_top16.py) — that's where the +7 pp result came from.

vLLM serving (either AMD or Modal) is for **demo throughput and easy deployment**, not for re-validating the eval result.

## When to deploy Modal

| Situation | Action |
|---|---|
| Hackathon demo running smoothly on droplet | **Don't deploy.** AMD MI300X is the headline. |
| Droplet feels flaky <30 min before demo | Deploy Modal with `min_containers=1` as warm spare. Keep `PTC_VLLM_GEMMA4_URL` pointing at AMD. |
| Droplet dies mid-demo | Flip `PTC_VLLM_GEMMA4_URL` to Modal on the Space settings, restart Space. ~30 s downtime. |
| Post-hackathon (AMD credits expired) | Modal becomes primary; archive the droplet config to git. |

## Files

- [deploy/modal/serve.py](../deploy/modal/serve.py) — Modal app definition (CUDA image, H100-80GB, vLLM 0.6.3.post1 with `--enable-lora`)
- [adapters/scin-top16-gemma4-lora/](../adapters/scin-top16-gemma4-lora/) — local LoRA dir (weights also on HF Hub at `sankara68/path-to-care-scin-top16-lora`)
- [frontend-next/src/app/api/triage/route.ts](../frontend-next/src/app/api/triage/route.ts) — reads `PTC_VLLM_GEMMA4_URL` env var; no code change needed for the swap
