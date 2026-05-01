# Existing Assets — What Plugs In

Mapping of Path to Care requirements to code already in `/Users/sankar/hackathons/` and `/Users/sankar/projects/`. The premise: ~70% of this project should be assembly, not greenfield.

## Asset map

| Brief requirement | Existing asset | Path | What's reusable |
|---|---|---|---|
| Triage Reasoner with WHO red/yellow/green | `sentinel_health` | `/Users/sankar/projects/sentinel_health` | WHO rule engine; 20 emergency vignettes; FastAPI scaffolding; rule-based safety layer for the cardinal rule |
| Village Context MCP (clinic hours, drug costs, distances) | `CareGraph` | `/Users/sankar/hackathons/CareGraph` | Neo4j schema with 10+ node types (Senior, Medication, Symptom, Condition, Doctor, Clinic); 500+ relationship patterns. Repurpose Senior → Patient, add Transport / SeasonalCalendar nodes |
| Image + chat frontend | `virtual_consultation` | `/Users/sankar/projects/virtual_consultation` | Camera + Gemini integration for dermatology — replace Gemini with the fine-tuned Stanford-trained classifier; reuse the camera-capture flow and chat scaffolding |
| LoRA fine-tune recipe | `CancerLLM` | `/Users/sankar/projects/CancerLLM` | LoRA on Mistral 7B with rank 8, alpha 16. Structurally identical to Qwen 7B LoRA; swap the base model and adjust target modules |
| Clinical evaluation suite | `sentinel_health` evaluation | `/Users/sankar/projects/sentinel_health` | 20 synthetic clinical vignettes (18/20 passing) — expand to 50-100 for Triage Reasoner training |
| Tamil rural-healthcare context | `rural-medical-kiosk-ui` | `/Users/sankar/projects/rural-medical-kiosk-ui` | UX patterns for low-literacy users; assumptions documented; can crib the language/literacy considerations |
| Multi-agent orchestration patterns | `agent_toolkit`, `agent_hackday`, `agentic_task` | `/Users/sankar/projects/agentic_task`, `/Users/sankar/hackathons/agent_toolkit`, `/Users/sankar/hackathons/agent_hackday` | CrewAI / OpenClaw orchestration patterns; tool composition; agent debugging/tracing |
| Memory / long-context for multi-turn conversations | `EverMemOS`, `midstream` | `/Users/sankar/hackathons/EverMemOS`, `/Users/sankar/hackathons/midstream` | Episodic memory and context compression — useful for the Intake Agent's clarifying-question loop |
| Domain corpora for fine-tune expansion | `CancerLLM` data | `/Users/sankar/projects/CancerLLM` | 2.7M clinical notes + 515K pathology reports — out of scope for Path to Care (skin, not oncology) but available if narrowing focus to a wound-care or dermatological-clinical-text expansion |

## Where to start

1. **Fork [sentinel_health](../../projects/sentinel_health/) → Triage Reasoner MCP.** It already has the rule engine and the vignette structure. Add the GRPO training loop on top.
2. **Extract [CareGraph](../CareGraph/) schema → Village Context MCP.** Strip the eldercare-specific bits, add transport + seasonal calendar nodes.
3. **Adapt [virtual_consultation](../../projects/virtual_consultation/) camera flow → Image intake.** Replace the Gemini call with the local fine-tuned classifier.
4. **Port [CancerLLM](../../projects/CancerLLM/) LoRA recipe → Qwen-2.5 7B fine-tune.** Same recipe, new base, different reward (urgency calibration instead of clinical-text generation).

## What's genuinely new (not reusable)

- **Fine-tuning the vision model** on Stanford Skin Dataset. New training run, new dataset, new bias-audit reports.
- **GRPO/RLVR loop.** None of the existing projects use GRPO. This is new code (`trl` library or DSPy's RL primitives).
- **DSPy `NarrativeToSOAP` signature.** New, but trivial — DSPy is a small surface area.
- **Real-village fieldwork.** Cannot be cribbed.
- **Skin-tone bias audit and tone-stratified eval reporting.** New.
- **Tamil-language UX validation.** Some prior work exists but no evaluated/deployed system.

## Voice / audio assets — explicitly out of scope

The user's [voice_agent](../../projects/voice_agent/) project has a working real-time STT/TTS pipeline. **It is not used for Path to Care baseline.** Text input + image is the default frontend assumption. Add voice only if a specific deployment population requires hands-free interaction and the user explicitly opts in.

## Net assessment

The risk that a hackathon project carries — *"can we actually ship something working in time?"* — is materially lower for Path to Care than for a greenfield project. Every architectural pillar matches a pillar that already exists in some form in the user's codebase.
