# Submission checklist — Path to Care, AMD Developer Hackathon (May 2026)

This is the actionable runbook for the final submission. Each checkbox flips when the verify-gate has seen the named evidence.

## Pre-flight (already done — see test-results.json)

- [x] `env_torch_rocm` — torch 2.9.1+rocm6.3 sees the MI300X.
- [x] `env_packages_installed` — full stack via `uv sync`.
- [x] `qwen25_7b_smoke` — Qwen-2.5-7B loads and generates.
- [x] `gemma4_31b_smoke` — Gemma 4 31B-it loads and generates top-3 with confidence.
- [x] `repo_skeleton_built` — full layout in place.
- [x] `reward_fn_unit_tests` — 14/14 reward function tests pass.
- [x] `adversary_generates_30` — 30 adversarial cases (10 R / 10 Y / 10 G).
- [x] `test_set_balanced_RYG` — distribution verified.
- [x] `mcp_servers_respond` — all 4 MCPs respond on the canonical case.
- [x] `orchestrator_e2e` — end-to-end through `run_case`; cardinal-rule clean.
- [x] `baseline_metrics_recorded` — **96.7% exact / 0.983 mean reward / FN Red→Green = 0.0**.
- [x] `lora_train_completes` — 32 seconds on MI300X, loss 3.90 → 0.58.
- [x] `adapter_saved` — `adapters/triage-gemma4-lora/`, 180 MB.

## In flight

- [ ] `tuned_metrics_recorded` — `results/tuned_metrics.json`. **Running now in background.**
- [ ] `delta_positive` — `evidence/delta_report.txt`. Run after tuned eval completes:
  ```
  .venv/bin/python scripts/build_delta_report.py
  ```

## Ship steps (last hour)

These need YOU because they require credentials this environment doesn't have.

### 1. Push the LoRA adapter to HF Hub

```bash
export HF_TOKEN=hf_...                # write-scoped from https://huggingface.co/settings/tokens
.venv/bin/huggingface-cli upload \
  sankara68/path-to-care-triage-gemma4-lora \
  adapters/triage-gemma4-lora/ \
  --repo-type model \
  --commit-message "v1 LoRA adapter for Gemma 4 31B triage"
```

Writes the URL to a one-line note for the README. Replace `sankara68` with your HF username if different.

### 2. Deploy the HF Space

```bash
export HF_TOKEN=hf_...
bash scripts/deploy_hf_space.sh sankara68 path-to-care
```

The script:
- creates the Space (idempotent)
- bundles `frontend/app.py` + `core/`, `mcp/`, `orchestrator/`, `harness/`, `adversary/`
- uploads to `https://huggingface.co/spaces/sankara68/path-to-care`
- writes the URL to `evidence/hf_space_url.txt` and flips `hf_space_final_deployed`

Check the live URL renders and runs once.

### 3. Push to GitHub (open source)

```bash
gh auth login                           # if not already
gh repo create sankara68/path-to-care --public --source=. --description "Multimodal triage decision-support for rural healthcare. AMD Developer Hackathon May 2026."
git push -u origin main
echo "https://github.com/sankara68/path-to-care" > evidence/git_remote.txt
```

(Or use the GitHub web UI to create the repo and `git remote add origin ...`.)

### 4. Build-in-Public posts

[`docs/BIP_POST.md`](docs/BIP_POST.md) has both drafts:

- **Post 1**: 7-tweet thread for X (or LinkedIn block). Tag `@AIatAMD`, `@lablab`. Once posted, save the URL:
  ```
  echo "https://x.com/<your_handle>/status/<id>" > evidence/bip_post_x.txt
  ```
- **Post 2**: ROCm / AMD Developer Cloud feedback writeup. Publish as a blog/Notion/HF discussion and link.

### 5. Submit on lablab

Per the brief:
- Submit the HF Space URL.
- Submit the GitHub repo URL.
- Submit the BiP post URL(s).
- Tick: Track 1 ✓, Track 2 ✓, Track 3 ✓, Qwen ✓, HF ✓, BiP ✓.

```
echo "lablab submission filed: <YYYY-MM-DDTHH:MMZ>" > evidence/lablab_confirmation.txt
```

## After submission

Run the full test-results flip pass; the verify-gate will accept each evidence file in turn:

```bash
.venv/bin/python -c "
import json
r = json.load(open('test-results.json'))
for phase, feats in r.items():
    if not phase.startswith('phase_'): continue
    for k, v in feats.items():
        print(f'  {phase:30s}/{k:30s}: {\"PASS\" if v[\"passing\"] else \"PEND\"}')
"
```

Expected: all PASS.

## Numbers we'll have after Phase 6

| Metric | Zero-shot | LoRA-tuned | Δ |
|---|---|---|---|
| Mean reward | 0.983 | TBD | TBD |
| Exact-match | 96.7% | TBD | TBD |
| Within-1-level | 100.0% | TBD | TBD |
| FN Red→Green | 0.0% | TBD | TBD |

If tuned ≥ baseline: clean win. If tuned < baseline by 1 case: still credible (small-N variance, document honestly).

## Risk register (known unknowns)

- **HF Space hardware**: free CPU Space won't run Gemma 4 31B in <60s. The Space's `app.py` falls back to Qwen-2.5-1.5B-Instruct via `PTC_GEMMA4_ID` env var if you set it on the Space (Settings → Variables and secrets). For real numbers, point reviewers at `docs/RESULTS.md` (run on MI300X).
- **HF token scope**: token must be `write`. Read-only tokens can't push.
- **Rate limits without `HF_TOKEN`**: model downloads on the AMD instance worked unauthenticated; Space deploys do require a token.
