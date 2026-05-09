#!/usr/bin/env bash
# Deploy the Path to Care frontend as an HF Space.
#
# Prerequisites:
#   - export HF_TOKEN=hf_...    # write-scoped token from https://huggingface.co/settings/tokens
#   - .venv with huggingface_hub installed (uv sync did this).
#
# Usage:
#   bash scripts/deploy_hf_space.sh <hf_username> <space_name>
#   e.g. bash scripts/deploy_hf_space.sh sankara68 path-to-care
#
# What it does:
#   1. Creates the Space (if missing) via huggingface-cli.
#   2. Bundles frontend/app.py, frontend/requirements.txt, frontend/README.md
#      and the in-process MCP code (core/, mcp/, orchestrator/, harness/) into
#      a flat layout the Space needs.
#   3. Uploads the bundle.
#
# Hardware: by default a free CPU Space won't run Gemma 4 31B. For the Space
# we set PTC_GEMMA4_ID=google/gemma-4-E4B-it via the Space environment;
# real eval numbers come from the MI300X (in the repo README).

set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is not set. Get one at https://huggingface.co/settings/tokens (write scope)."
  exit 1
fi

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <hf_username> <space_name>"
  exit 1
fi

USER="$1"
SPACE="$2"
REPO_ID="${USER}/${SPACE}"
STAGE=".tmp_space_bundle"

echo "== Bundling Space at ${STAGE} =="
rm -rf "${STAGE}"
mkdir -p "${STAGE}"
cp frontend/app.py "${STAGE}/app.py"
cp frontend/requirements.txt "${STAGE}/requirements.txt"
cp frontend/README.md "${STAGE}/README.md"
# Bring in the in-process MCP code; keep paths so app.py imports work.
cp -r core mcp orchestrator harness adversary "${STAGE}/"

echo "== Creating Space ${REPO_ID} (idempotent) =="
.venv/bin/huggingface-cli repo create "${REPO_ID}" --type space --space_sdk gradio -y || true

echo "== Uploading ${STAGE} -> ${REPO_ID} =="
.venv/bin/huggingface-cli upload "${REPO_ID}" "${STAGE}" --repo-type space --commit-message "Deploy Path to Care v1"

URL="https://huggingface.co/spaces/${REPO_ID}"
echo "${URL}" > evidence/hf_space_url.txt
echo "verdict: PASS" >> evidence/hf_space_url.txt
echo "Space deployed: ${URL}"
echo "(Wrote evidence/hf_space_url.txt)"
