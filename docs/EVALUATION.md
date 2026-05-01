# Evaluation

Four dimensions: **technical, clinical, human, safety.** None is optional.

## 1. Technical metrics

### Image classification

- Top-1, top-3, top-5 accuracy on a held-out test set.
- **Stratified by skin tone** (light / medium / dark) and **by condition**. Aggregate-only reporting is not acceptable per the brief.
- If the model underperforms on a subgroup, that is a published finding, not a deployment blocker to hide.

### SOAP extraction

- Field-level precision and recall: chief complaint, HPI, negations ("no fever"), medication list.
- Negation handling is a specific failure mode to test — a regex baseline will extract "fever: okay" from "I feel okay"; the DSPy version must not.

### Triage calibration

- Confusion matrix of predicted vs. ground-truth urgency.
- **False negatives (predicted Green when ground truth is Red) matter more than false positives.** Report them separately.
- Reward function:
  ```
  R(predicted, ground_truth) = 1.0  if exact match
                              0.5  if adjacent level
                              0.0  if off by 2+
  ```
- Average reward over the 80-case test set is the headline number.

### Latency

- End-to-end target: **< 30 seconds** from image capture to triage output.
- Break down: phone capture → MCP image classifier → SOAP extraction → triage reasoning.
- vLLM throughput on MI300X for Qwen 7B: report tokens/sec.

## 2. Clinical appropriateness

A physician (or med student / clinical reviewer) reviews 20-30 system outputs. Questions:

- Are the urgency assessments clinically accurate?
- Are red flags missed? Are there false alarms?
- **Cardinal-rule compliance:** does the system say "you have cellulitis" (bad) or "signs suggest infection" (good)?
- Are recommendations proportional? (Not over-referring; not missing serious signs.)

## 3. Human factors

- **Usability:** can a farmer unfamiliar with technology submit an image and narrative?
- **Trust:** do users believe the system's assessments? Do they follow recommendations? (Requires field testing over time — out of scope for a 3-week hackathon, in scope for the full 8-week plan.)
- **Accessibility:**
  - Text summaries readable at phone size and brightness.
  - Image capture works in low-light / outdoor conditions.
- **Language:** is the Tamil-language interface natural? Did we avoid medical jargon, or did we explain it?

## 4. Safety and adversarial

- **Red-teaming:** deliberately try to break the system.
  - Blurry images
  - Images of unrelated skin conditions (or non-skin: a cat, a wall)
  - Narratives with contradictions ("no fever" + "I have a high fever")
  - Edge cases: very old scars, burns, conditions the model was not trained on
- **Bias audit:** does the system behave differently for different skin tones, ages, genders? Stratify all results.
- **Failure modes:**
  - What happens if the image is blank?
  - What happens if the patient reports no symptoms?
  - What happens if language is ambiguous?
- Document failure modes and design fallbacks: *"I'm not confident in this image. Please take another, or contact your clinic directly."*

## Held-out test set

The brief calls for an 80-case dermatology test set. Sources:

- Stanford Skin Dataset (held-out subset of the curated training data).
- Synthesized cases inspired by the Rajan dialogue (cellulitis, impetigo, contact dermatitis, etc.).
- Optional expansion: snake-bite vignettes (per the previous-cohort extension noted in the brief).

For the Triage Reasoner specifically, **50-100 cases (narrative + image + ground-truth urgency)** are needed for the RLVR loop. These can overlap partially with the dermatology test set, but ground-truth urgency labels need to be supplied (binary Red/Yellow/Green) by a clinician.

## Reporting template

The hackathon README must publish:

```
                        Zero-shot   RLVR-tuned   Δ
Top-1 image accuracy     XX%         XX%         +X
Top-3 image accuracy     XX%         XX%         +X
  - light skin           XX%         XX%         +X
  - medium skin          XX%         XX%         +X
  - dark skin            XX%         XX%         +X
SOAP extraction P/R      X.XX/X.XX   X.XX/X.XX   +X
Triage exact match       XX%         XX%         +X
Triage within-1-level    XX%         XX%         +X
False-negative Red→Green XX%         XX%         -X
End-to-end latency p50   XX s        XX s        -X
```

The "delta from zero-shot" column is the credibility evidence: *"My system is +13% better, not because of magic, but because I tuned both the language and the weights."*
