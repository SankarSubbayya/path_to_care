# AMD Hackathon Track + Prize Mapping

Path to Care is intentionally designed to satisfy multiple AMD hackathon tracks and partner-prize criteria with a single artifact.

## AMD Tracks

### Track 1 — AI Agents & Agentic Workflows (beginner-friendly)

**Required:** sophisticated agentic system, beyond simple RAG. Coordinates agents or assists in complex tasks. LangChain / CrewAI / AutoGen + open models.

**How Path to Care satisfies:**

- 4 MCP servers (Image Classifier, SOAP Extractor, Village Context, Triage Reasoner) coordinated by a DSPy orchestrator using ReAct + hierarchical multi-agent patterns.
- Multi-turn conversation with clarifying questions, image analysis, contextual reasoning, persuasion framing.
- Not RAG-shaped — the orchestrator does meaningful planning and tool routing, not just retrieve-and-answer.

**Compute:** $100 in AMD Developer Cloud credits is sufficient for inference, light experimentation. The fine-tune work needs MI300X (Track 2 resource).

### Track 2 — Fine-Tuning on AMD GPUs (advanced)

**Required:** domain-specific fine-tuning of open-source models on ROCm. PyTorch + Hugging Face Optimum-AMD + vLLM serving.

**How Path to Care satisfies:**

- **Vision fine-tune:** ResNet-50 / EfficientNet-B4 / ViT on a curated Stanford Skin Dataset subset. ~8-16 hours on MI300X.
- **LLM fine-tune:** Qwen-2.5 7B with LoRA + GRPO on 50-100 labeled triage cases. Verifiable reward function.
- **Serving:** vLLM on MI300X for Qwen 7B inference; PyTorch for vision inference.
- **Story:** "fine-tuned on the rural healthcare dataset to bring triage calibration to underserved Tamil Nadu / Mississippi Delta villages."

**Compute:** MI300X access via AMD Developer Cloud. 192 GB HBM3 is enough headroom to run the fine-tune + a vision model + serving in one instance.

### Track 3 — Vision & Multimodal AI

**Required:** applications processing multiple data types (image, video, audio) using AMD GPU memory bandwidth. Multimodal models (Llama 3.2 Vision, Qwen-VL) optimized for ROCm.

**How Path to Care satisfies:**

- **Image:** smartphone photo of skin condition → vision model classification with top-3 ranked outputs.
- **Text:** patient narrative → SOAP-structured extraction; pre-visit summary for the clinic doctor.
- **Structured data:** village context (clinic hours, costs, transport, seasonality) integrated into reasoning.
- Multimodal here means **image + structured text + structured knowledge graph** — not voice.

**Optional upgrade:** swap the dedicated image classifier for **Qwen-VL** to handle image + text in a single model, demonstrating Qwen's multimodal capabilities and reducing the architecture to 3 MCP servers.

## Partner Prizes

### Hugging Face — "Most-liked Space wins"

**Deliverable:** publish a Space under the AMD Developer Hackathon HF Organization.

**Plan:**

- Gradio frontend with the **Rajan dialogue replayable end-to-end**.
- Side-by-side comparison: zero-shot Qwen 7B vs. RLVR-tuned Qwen 7B on the same case.
- Skin-tone-stratified accuracy table prominently displayed.
- "Try your own case" mode (upload an image, type a narrative).
- Tagged with healthcare, rural, multimodal, agentic, ROCm.

**Why it can win likes:** the demo tells a clear story (rural healthcare access), shows measurable improvement (delta numbers), and addresses bias openly (skin-tone reporting). Stories that combine impact + technical rigor + honest limitations get shared.

### Qwen — "Qwen contributes meaningfully to functionality"

**Deliverable:** Qwen models meaningfully integrated, called out in the submission.

**Plan:**

- **Qwen-2.5 7B** as the base for the Triage Reasoner (RLVR fine-tuned).
- **Qwen-2.5 7B** as the base for the SOAP Extractor (DSPy-optimized).
- *(Optional)* **Qwen-VL** as a unified vision-language replacement for the dedicated image classifier.
- Highlighted in the README and HF Space description: "Powered by Qwen-2.5 7B, fine-tuned on AMD MI300X with GRPO/RLVR for clinical triage calibration."

### Build in Public

**Deliverable:** 2 social posts + ROCm/AMD Dev Cloud feedback + open-source.

**Plan:**

- **Post 1 (mid-build):** "Fine-tuning Qwen-2.5 7B on AMD MI300X for rural healthcare triage. Here's what 192 GB of HBM3 gets you when you can run vision + LLM + serving in one instance." — tag `@AIatAMD`, `@lablab`, attach a screenshot of `rocm-smi`.
- **Post 2 (post-build):** "Path to Care: open-source. 70% accurate triage calibration on the WHO red/yellow/green scale. Tone-stratified accuracy report inside (we're 12 points lower on dark skin — we report it because that's the point)." — link to the HF Space.
- **Feedback writeup** — honest assessment of the AMD Dev Cloud onboarding, ROCm + PyTorch experience, vLLM-on-MI300X serving, gaps versus CUDA workflow.
- **Open source:** repo on GitHub, model weights on HF Hub, dataset prep scripts included.

## Prize stacking summary

| Prize | Track | Effort to qualify | Probability of winning |
|---|---|---|---|
| Track 1 (Agents) | Track 1 | Already qualifying by architecture | Competitive — many entries |
| Track 2 (Fine-Tuning) | Track 2 | Already qualifying by required RLVR loop | Competitive — fewer entries (Track 2 is harder) |
| Track 3 (Multimodal) | Track 3 | Already qualifying — image + text + structured data | Competitive |
| Qwen | Cross-track | Already using Qwen 7B as base | Strong, especially if Qwen-VL is added |
| Hugging Face | Cross-track | Need to publish Space + actively promote | Depends on community engagement |
| Build in Public | Cross-track | 2 posts + feedback + open-source | Strong — most entries don't bother |

A single project, multiple prize categories. That's the design intent.
