# AMD Developer Hackathon

Build the next generation of AI agents and high-performance applications, powered by AMD.

This hackathon is a space to explore, experiment, and create with **AMD Developer Cloud** and **ROCm** — no hardware, no complex setup, just access to powerful compute.

**Goal:** build an application, agent, or developer tool that feels real, works end-to-end, and shows what AMD's compute stack can unlock.

## Project: Path to Care

*Slug: `path_to_care`.*

A multimodal, agentic decision-support system for rural healthcare. See [docs/](docs/) for full project documentation:

- [docs/PROJECT.md](docs/PROJECT.md) — what Path to Care is, the cardinal rule, why this project, what gets built.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — system diagram, 4 MCP servers, conversation flow, serving stack.
- [docs/PLAN.md](docs/PLAN.md) — full 8-week plan + 3-week hackathon-MVP plan, risks, definition of done.
- [docs/EVALUATION.md](docs/EVALUATION.md) — technical, clinical, human, safety metrics. Reporting template.
- [docs/ASSETS.md](docs/ASSETS.md) — mapping of existing code (sentinel_health, CareGraph, virtual_consultation, CancerLLM, etc.) to project requirements.
- [docs/TRACK_MAPPING.md](docs/TRACK_MAPPING.md) — how the project hits Tracks 1/2/3 + Qwen + HF + Build-in-Public prizes.

---

## Tracks

### Track 1: AI Agents & Agentic Workflows *(beginner-friendly)*

- **Objective:** move beyond simple RAG to build sophisticated AI agentic systems and workloads.
- **What to build:** intelligent AI systems that automate workflows, coordinate agents, or assist users in complex tasks.
- **Tech stack:** LangChain, CrewAI, or AutoGen connecting to open-source models (Llama, DeepSeek, Mistral, Qwen).
- **Compute:** $100 in AMD Developer Cloud credits.

### Track 2: Fine-Tuning on AMD GPUs *(advanced / GPU-intensive)*

- **Objective:** leverage direct GPU access to fine-tune open-source models for high-impact domain specialization.
- **What to build:** domain-specific LLMs (Healthcare, Finance, Legal, Code) fine-tuned for accuracy and efficiency on ROCm.
- **Tech stack:** ROCm, PyTorch, Hugging Face Optimum-AMD, vLLM for serving.
- **Compute:** AMD Instinct MI300X instances via AMD Developer Cloud.

### Track 3: Vision & Multimodal AI

- **Objective:** build applications that process and understand multiple data types (images, video, audio) using the memory bandwidth of AMD GPUs.
- **What to build:** high-throughput industrial inspection, medical imaging analysis, or multimodal conversational assistants.
- **Tech stack:** multimodal models (Llama 3.2 Vision, Qwen-VL) optimized for ROCm.
- **Compute:** AMD Instinct MI300X instances via AMD Developer Cloud.

### Extra Challenge: Ship It + Build in Public

Can be combined with any track.

- **Objective:** document the building journey, share insights, and provide feedback on the AMD developer experience.
- **Requirements:**
  1. Share at least 2 technical updates on social media (tag `@lablab` on X or `lablab.ai` on LinkedIn, and `@AIatAMD` on X or `AMD Developer` on LinkedIn).
  2. Provide meaningful feedback about building with ROCm, AMD Developer Cloud, or APIs.
  3. Open-source the project or publish a technical walkthrough.
- **Reward:** dedicated prize pool for the best Build in Public stories and most valuable product feedback.

---

## Technology & Access

All development happens on cloud-accessible AMD GPUs — no need to own or manage hardware.

### AMD Developer Cloud

On-demand access to AMD Instinct GPUs. Spin up GPU environments in minutes.

**Typical uses:**
- Training and fine-tuning ML models
- Benchmarking AI workloads on AMD GPUs
- Prototyping AI systems before moving to on-prem

**Access:**
- $100 in AMD Developer Cloud credits for AMD AI Developer Program members
- Pay-as-you-go pricing available

**Docs:** Getting started guide · AMD Developer Cloud overview

### ROCm (Radeon Open Compute)

AMD's open-source GPU computing platform — the AMD equivalent of CUDA.

**Common uses:**
- Running PyTorch and TensorFlow on AMD GPUs
- Porting CUDA-based workloads to AMD hardware
- Executing high-performance AI/ML and HPC workloads

**Docs:** ROCm documentation · ROCm installation guide · ROCm GitHub

### Access Phasing

- **Online phase:** credits-based access for all participants.
- **On-site phase:** dedicated GPU access for selected finalists.

---

## Technology Partners

### Hugging Face

Model hub and deployment layer for the project.

1. Find a model on Hugging Face Hub.
2. Build or fine-tune it on AMD Developer Cloud.
3. Publish the completed project as a Hugging Face **Space** under the event organization.
4. Submit the Space link on lablab.

**Hugging Face category prize:** the Space with the most likes at the end of the hackathon wins. Once the Space is live, share it — the community votes with likes.

Join the AMD Developer Hackathon HF Organization to publish a Space under it.

### Qwen

Family of advanced AI models from Alibaba Cloud — strong reasoning, coding, and multilingual capabilities. Spans text, code, and multimodal use cases.

**Challenge:** incorporate Qwen models into the project across any track. Build a complete, end-to-end application where Qwen contributes meaningfully to functionality, performance, or intelligence. For example:

- AI agents or copilots
- Natural language interfaces
- Workflow / decision automation
- Multilingual or user-facing AI features

**Getting started:**
1. Explore Qwen models and capabilities.
2. Choose a model that fits the use case.
3. Integrate it into the project.
4. Highlight how Qwen is used in the final submission.

**Resources:** Qwen documentation · Models on Hugging Face · ModelScope
