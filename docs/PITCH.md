# 3-minute pitch — Path to Care

Spoken pace: ~450 words, ≈3:00. Slide cues in **bold**.

---

**[Slide 1 — Title: Path to Care + the Rajan dialogue thumbnail]**

A patient in a Tamil Nadu village cuts his foot on a rusty nail. Two days later it's swollen, the redness is climbing his leg, and he has fever and shivering. The nearest health centre is 18 km away — a ₹180 round-trip on a daily wage of ₹350. Harvest is active. Going to the clinic costs him half a day's income he doesn't have, and a long bus ride he doesn't want to take.

He pulls out his phone. **(00:25)**

**[Slide 2 — The four jobs]**

Path to Care is a multimodal, agentic decision-support system. It does four things, and only these four. It ranks plausible skin conditions from a phone photo — **always top three with confidence**, never single class. It assigns urgency: Red, Yellow, or Green. It flags red signs. And it converts clinical urgency into *practical* urgency, framed as cost-benefit: "I know the trip costs time and money, but this infection is spreading fast — going today saves much bigger costs of waiting." **(00:55)**

**[Slide 3 — Cardinal rule + safety net]**

The system **never diagnoses**. That rule is enforced in code, not just the prompt. Every model output passes through a regex rewriter that replaces "you have cellulitis" with "signs suggest cellulitis." During eval, this fired on a real case — model said "you have a fever," rewriter caught it, logged it. There's also a rule-based safety net: if the model says "green" but two or more red-flag keywords are in the narrative, we escalate to "yellow" automatically. We never under-triage. **(01:25)**

**[Slide 4 — Architecture + the AMD stack]**

Under the hood: a DSPy-style orchestrator wires four MCP services — image classifier, SOAP extractor, village context, triage reasoner. **The image classifier and triage reasoner share one loaded Gemma 4 31B model** — multimodal, dense, Apache-2.0 — running on a single AMD MI300X via ROCm 6.3. SOAP extraction runs on Qwen 2.5-7B. Production inference is via vLLM in Docker, AMD's own published recipe — what you're seeing on this Hugging Face Space is the Gradio UI calling the MI300X via HTTP. **(01:55)**

**[Slide 5 — The eval table + the honest framing]**

We built a 30-case adversarially-authored test set: 10 Red, 10 Yellow, 10 Green, 25 with perturbations — colloquial dialect, contradicted narratives, off-distribution images. Mean reward 0.983. Exact-match urgency 96.7%. **False-negative Red→Green rate: zero percent.** No Red case was ever down-triaged to Green — that's the cardinal safety metric. The LoRA fine-tune converged in **32 seconds** on the MI300X — loss 3.90 to 0.58, 45 million trainable parameters, 0.14% of the base. Tuning matched baseline accuracy and shifted the reasoning style toward our preferred phrasing. **(02:25)**

**[Slide 6 — What's next]**

This is a 24-hour build. The full version is in the v2 roadmap: 80 cases, skin-tone stratification, GRPO with verifiable urgency reward, real village fieldwork in Thiruvallur, physician review of 20-30 outputs. Open source on GitHub, adapter on Hugging Face, demo live on the Space. The cardinal rule travels with it: **the doctor diagnoses, the patient decides, the system informs both.** **(02:55)**

---

## Bullet outline (use as speaker notes)

- **Hook (0:00–0:25)**: Rajan, rusty nail, 18 km, ₹180, harvest pressure.
- **What it does (0:25–0:55)**: top-3 / RYG / red flags / barriers framed as cost-benefit.
- **Cardinal rule + safety net (0:55–1:25)**: regex rewriter (real example), rule-based escalation, "never under-triage."
- **Architecture (1:25–1:55)**: Gemma 4 31B + Qwen 2.5-7B, MI300X / ROCm 6.3, vLLM Docker on AMD's recipe, HF Space → MI300X HTTP.
- **Eval (1:55–2:25)**: 30 adversarial cases, 0.983 / 96.7%, **FN Red→Green = 0%**, LoRA in 32s.
- **Roadmap (2:25–2:55)**: 80 cases, skin-tone stratification, GRPO, fieldwork, physician review.

## Stretch hooks (if asked)

- "Why Gemma 4 31B and not the MoE 26B?" — MoE forward path uses `torch._grouped_mm`, no ROCm impl. Documented in `docs/COMPATIBILITY.md` as concrete AMD product feedback.
- "Why this is novel" — most digital-health products diagnose. We don't. The system is wrong by design when it tries to. The cardinal rule is the product.
- "What does the LoRA actually change?" — same urgency calls (base is at the eval ceiling), but reasoning text reformats toward our training style ("Red flags noted: ..."). See `docs/FINE_TUNING_DEMO.md`.
