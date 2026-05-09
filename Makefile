.DEFAULT_GOAL := help
PY := .venv/bin/python

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS=":.*?## "}; {printf "%-20s %s\n", $$1, $$2}'

## --- environment ---

setup:  ## install uv venv + sync deps from pyproject.toml
	@which uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh)
	uv sync

smoke-torch:  ## verify ROCm torch sees the MI300X
	$(PY) scripts/smoke_torch.py

smoke-models:  ## verify Gemma 4 + Qwen 2.5 load and generate
	$(PY) scripts/smoke_models.py --which both

## --- harness ---

test-reward:  ## reward-function unit tests
	$(PY) -m harness.test_reward

build-cases:  ## (re)generate the 30-case adversarial test set
	$(PY) -m adversary.generate
	$(PY) -m adversary.check_distribution

## --- baseline / eval ---

baseline:  ## run zero-shot eval over data/cases.jsonl -> results/baseline_metrics.json
	$(PY) -m harness.run --out results/baseline_metrics.json

build-train:  ## turn baseline traces into data/train.jsonl + data/holdout.jsonl
	$(PY) -m training.build_train_set --traces results/baseline_metrics.json \
		--train-out data/train.jsonl --holdout-out data/holdout.jsonl

## --- training ---

train:  ## LoRA SFT on Gemma 4 31B-it -> adapters/triage-gemma4-lora/
	$(PY) -m training.lora_sft --train data/train.jsonl --output adapters/triage-gemma4-lora

## --- tuned eval ---

tuned:  ## run eval with the LoRA adapter -> results/tuned_metrics.json
	$(PY) -m harness.run --adapter adapters/triage-gemma4-lora --out results/tuned_metrics.json

## --- demo ---

frontend:  ## launch the Gradio app on localhost:7860
	$(PY) -m frontend.app

## --- ops ---

mcp-smoke:  ## run all 4 MCPs on the first case
	$(PY) scripts/mcp_smoke.py

orch-smoke:  ## run the orchestrator on the first case end-to-end
	$(PY) scripts/orchestrator_smoke.py

clean:  ## drop transient state (NOT model weights or .venv)
	rm -rf logs/*.log evidence/*.txt results/*.json results/*.log .claude/.evidence-reads
.PHONY: help setup smoke-torch smoke-models test-reward build-cases baseline build-train train tuned frontend mcp-smoke orch-smoke clean
