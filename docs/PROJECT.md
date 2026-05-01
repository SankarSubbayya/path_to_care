# Project: Path to Care

*Slug: `path_to_care`.*

## One-line description

A multimodal, agentic decision-support system for rural healthcare in the Global South — runs on a phone, never diagnoses, helps a patient understand whether they need a doctor and whether they can afford to go.

## Cardinal rule

**The system never diagnoses.** It does four things:

1. **Narrows possibilities** — ranks plausible skin conditions by likelihood from an image.
2. **Assesses urgency** — Red (immediate), Yellow (1-2 days), Green (monitor at home).
3. **Flags red signs** — features that demand professional evaluation.
4. **Contextualizes barriers** — distance, cost, harvest season, and what mitigations exist.

The output is decision support for the patient and a structured pre-visit summary for the clinic doctor. The doctor diagnoses; the patient decides; the system informs both.

## Why this project

| Reason | Detail |
|---|---|
| Three AMD tracks in one project | Multi-agent orchestration (Track 1) + LoRA fine-tune with GRPO/RLVR on MI300X (Track 2) + image + text + structured data (Track 3). |
| Three partner prizes accessible | Qwen-2.5 7B as base model (Qwen prize), HF Space deployment (HF prize), Build-in-Public-friendly story (BiP prize). |
| Ready-made evaluation | An 80-case dermatology test set with R = 1.0 / 0.5 / 0.0 urgency reward — measurable from day one. |
| 70% code reuse | Existing healthcare and agent infrastructure plugs in directly — see [ASSETS.md](ASSETS.md). |

## What gets built

- **Mobile/web frontend** — phone camera + text input. Tamil-language UI for the chosen village; chat-style interaction.
- **DSPy orchestrator** — coordinates the four MCP servers, handles conversation state.
- **4 MCP servers:**
  - Image Classifier — fine-tuned ResNet-50 / EfficientNet-B4 / ViT on Stanford Skin Dataset subset.
  - SOAP Extractor — DSPy `NarrativeToSOAP` signature, optimized via `BootstrapFewShot`.
  - Village Context — local logistics (bus schedules, clinic hours, costs, harvest seasons).
  - Triage Reasoner — Qwen-2.5 7B + LoRA, optimized with GRPO/RLVR for urgency calibration.
- **Pre-visit summary generator** — structured SOAP note for the clinic doctor (SMS + printable).
- **Evaluation suite** — 4 dimensions: technical, clinical, human, safety (see [EVALUATION.md](EVALUATION.md)).
- **HF Space demo** — replayable Rajan dialogue against zero-shot vs. RLVR-tuned models.

## Hard constraints from the brief

- Never produce diagnostic statements ("you have cellulitis"). Always: "signs suggest infection."
- **Top-3 image predictions with confidence** — never single class label, never binary sick/healthy.
- **Skin-tone-stratified evaluation** — separate accuracy reports for light/medium/dark tones. Aggregate is not acceptable.
- **Real village** — not a hypothetical proxy. Choose Tamil Nadu (Thiruvallur/Ranipet have public health data) or rural Appalachia / Mississippi Delta.
- **75-85% extraction accuracy** is the realistic target, paired with human-in-the-loop confirmation. 95% is not the goal.

## Out of scope (explicitly)

- Becoming a dermatologist. Read 2-3 review articles, not every paper.
- Manual decision trees for 50 conditions. Use the dataset; let the model learn.
- Diagnosing rare or atypical presentations. Defer to the doctor.
- Full deployment to thousands of users in 8 weeks. Deploy to 2-3 real users for feedback.
