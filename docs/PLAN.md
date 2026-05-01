# Plan

Two plans below: a thorough 8-week build for full deployment readiness, and a compressed 2-3 week hackathon-MVP variant that ships a defensible demo earlier. Pick the variant that matches the actual time window.

## Full 8-week plan

### Week 1 — Background and village selection

- Read 2-3 dermatology review articles. **Do not read textbooks.**
- Choose a real village. Research:
  - Transport (bus routes, fares, frequency, seasonal changes)
  - Healthcare infrastructure (PHC, hospital, private clinics, hours, services)
  - Seasonal rhythms (crop cycles, peak labor months)
  - Demographics and literacy
  - Top local diseases (malaria? snake bites? skin infections from agricultural work?)
- Audit the Stanford Skin Dataset; pick the curated subset.
- Set up MI300X instance on AMD Developer Cloud; verify ROCm / PyTorch.

### Weeks 2-3 — Vision model + SOAP extractor

- Fine-tune ResNet-50 / EfficientNet-B4 / ViT on the curated subset. 8-16 hours per training run on a recent GPU.
- Hyperparameter search, validation, ablation.
- Stratify accuracy by skin tone — publish numbers, not aggregates.
- Hand-label 50-100 (narrative → SOAP) pairs from the chosen village (or proxy population).
- Define DSPy `NarrativeToSOAP` signature; optimize with `BootstrapFewShot`.

### Week 4 — MCP servers + orchestrator

- Stand up the 4 MCP servers (Image, SOAP, Village Context, Triage placeholder).
- Wire them into a DSPy orchestrator using ReAct.
- End-to-end smoke test: phone image + typed narrative → SOAP summary.

### Week 5 — Triage Reasoner fine-tune (RLVR)

- Gather 50-100 labeled cases (narrative + image + ground-truth urgency).
- LoRA fine-tune Qwen-2.5 7B with GRPO using the brief's reward function.
- Compare zero-shot vs. tuned: report deltas.

### Week 6 — Integration, persuasion, frontend polish

- Wire Triage Reasoner into the orchestrator.
- Build the persuasion-framing DSPy signature (cost-benefit reframing).
- Tamil-language UI strings; readable on small screens with brightness/contrast considered.
- Pre-visit SOAP summary generator (SMS-friendly).

### Week 7 — Evaluation + field deployment

- Run the 4-dimension evaluation (see [EVALUATION.md](EVALUATION.md)).
- Physician review of 20-30 outputs.
- Deploy to 2-3 real users in the chosen village. Collect feedback.
- Red-team: blurry images, contradictory narratives, edge cases.

### Week 8 — Documentation and deployment readiness

- Write the technical walkthrough (Build-in-Public deliverable).
- Publish the HF Space.
- Bias audit report (skin tone, age, gender).
- Failure-mode documentation.
- Final demo recording.

## Compressed hackathon plan (3 weeks)

If the AMD hackathon window is tight, compress as follows. **Keep the architecture intact; reduce dataset size and fieldwork ambition.**

### Week 1 — Spin-up + dataset + zero-shot baseline

- AMD Dev Cloud instance, ROCm verified, vLLM serving Qwen-2.5 7B.
- Stanford Skin Dataset subset curated (~5 conditions, tone-balanced).
- Hand-label 30 narrative→SOAP pairs (smaller than the brief's 50-100).
- Zero-shot baseline: orchestrator + 4 MCP servers, no fine-tune yet.
- Score against the 80-case test set (use 30 for fine-tune, hold 50 for eval).

### Week 2 — Fine-tune both models

- Vision model fine-tune (one GPU-day).
- Qwen 7B LoRA + GRPO triage fine-tune (one GPU-day).
- Re-score: report delta from zero-shot. This is the headline number.
- DSPy `BootstrapFewShot` on SOAP extractor.

### Week 3 — Polish, deploy, document

- HF Space deployment with replayable Rajan dialogue.
- Persuasion framing via DSPy.
- Skin-tone-stratified accuracy report.
- 2 social posts (Build-in-Public deliverable).
- Technical walkthrough + open-source the repo.
- Submit to lablab.

**What gets cut in the compressed plan:**

- Real-village fieldwork (replaced with documented assumptions and one synthetic case study).
- Field deployment to real users (replaced with physician review of 10-15 outputs, not 20-30).
- Snake-bite extension and other extensibility demos.
- Multilingual UX beyond Tamil + English.
- Voice/TTS/STT — not in scope unless explicitly added.

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Stanford Skin Dataset access requires research-data form | Medium | Apply on day one. Fallback: HAM10000 (10k images, public, derm classifier-friendly). |
| GRPO fine-tune fails to converge with 50 cases | Medium | Have DSPy `BootstrapFewShot`-only baseline working first; RLVR is additive. |
| MI300X queue times on AMD Dev Cloud | Low-Medium | Reserve early. Have local-CPU dev path for non-training work. |
| Skin-tone bias is severe in pretrained checkpoint | High | This is the *expected* finding. Report it honestly; that's the point. |
| Physician review unavailable | Medium | Recruit a med student or use a clinical reviewer from an existing healthcare contact. |

## Definition of done (hackathon submission)

- HF Space with the Rajan dialogue replayable end-to-end.
- README with: architecture diagram, eval numbers (zero-shot vs. RLVR-tuned, stratified by skin tone), reproducibility instructions.
- One technical walkthrough post (blog or X thread) tagging `@AIatAMD` and `@lablab`.
- One feedback writeup on building with ROCm / AMD Dev Cloud.
- Open-source repo with model weights on HF Hub.
- Submitted on lablab.
