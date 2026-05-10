#!/usr/bin/env bash
# Live-demo script for Path to Care.
#
# Three things to show, in order:
#   1) vLLM container is alive on MI300X (ROCm 6.3, Gemma 4 31B-it)
#   2) The Rajan dialogue runs end-to-end through the orchestrator/MCPs
#      (top-3 image, SOAP, village context, triage with cost-benefit framing)
#   3) HF Space is deployed
#
# Usage:
#   bash scripts/demo.sh                       # all 3 sections
#   bash scripts/demo.sh vllm | rajan | space  # just one section
#
# Override the vLLM connection if needed:
#   VLLM_URL=http://other-host:8000/v1 \
#   VLLM_KEY=other-key \
#   VLLM_MODEL=different/model \
#   bash scripts/demo.sh

set -euo pipefail

VLLM_URL="${VLLM_URL:-http://localhost:8000/v1}"
VLLM_KEY="${VLLM_KEY:-ptc-demo-2026-amd}"
VLLM_MODEL="${VLLM_MODEL:-google/gemma-4-31B-it}"
HF_SPACE_URL="${HF_SPACE_URL:-https://huggingface.co/spaces/sankara68/path-to-care}"

SECTION="${1:-all}"

bar() { printf '\n\033[1m%s\033[0m\n%s\n' "$1" "$(printf '%.0s—' {1..72})"; }

# --- Section 1: vLLM on MI300X --------------------------------------------
section_vllm() {
  bar "1. vLLM serving Gemma 4 31B-it on MI300X (ROCm 6.3)"

  echo "$ docker ps --filter name=ptc-vllm"
  docker ps --filter name=ptc-vllm --format "  {{.Names}}: {{.Status}}  ports={{.Ports}}"
  echo

  echo "$ curl GET /v1/models  (with auth)"
  curl -s -m 5 "${VLLM_URL}/models" \
    -H "Authorization: Bearer ${VLLM_KEY}" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(' ', d['data'][0]['id'], 'max_model_len=' + str(d['data'][0]['max_model_len']))"
  echo

  echo "$ curl GET /v1/models  (no auth — should 401)"
  printf '  HTTP '
  curl -s -m 5 -o /dev/null -w "%{http_code}\n" "${VLLM_URL}/models"
  echo

  echo "$ curl POST /v1/chat/completions  (top-3 conditions, no diagnosis)"
  curl -s -m 60 "${VLLM_URL}/chat/completions" \
    -H "Authorization: Bearer ${VLLM_KEY}" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${VLLM_MODEL}\",
      \"messages\": [{
        \"role\":\"user\",
        \"content\":\"List up to three plausible skin/wound conditions for a swollen, reddened ankle with a small puncture wound, each with a confidence in [0,1]. Do not diagnose. Output one per line.\"
      }],
      \"max_tokens\": 80,
      \"temperature\": 0
    }" \
    | python3 -c "import sys,json; r=json.load(sys.stdin); print(' ', repr(r['choices'][0]['message']['content']))"
}

# --- Section 2: end-to-end Rajan dialogue ---------------------------------
section_rajan() {
  bar "2. Rajan dialogue end-to-end through orchestrator + 4 MCPs"

  echo "$ .venv/bin/python scripts/orchestrator_smoke.py"
  echo "  (uses the in-process transformers backend; full pipeline)"
  echo "  (~60s cold load + ~10s inference; or set PTC_INFERENCE=vllm to use the container)"
  echo
  if [ "${PTC_INFERENCE:-transformers}" = "vllm" ]; then
    echo "  Using PTC_INFERENCE=vllm — calls go to ${VLLM_URL}"
  fi
  PTC_INFERENCE="${PTC_INFERENCE:-transformers}" .venv/bin/python scripts/orchestrator_smoke.py 2>&1 \
    | grep -vE "(Loading weights|Fetching|Warning:.*HF_TOKEN)" \
    | tail -30
}

# --- Section 3: HF Space --------------------------------------------------
section_space() {
  bar "3. HF Space deploy (Gradio frontend on Hugging Face)"

  echo "$ curl ${HF_SPACE_URL}"
  printf '  HTTP '
  curl -s -m 10 -o /dev/null -w "%{http_code}\n" "${HF_SPACE_URL}"

  echo
  echo "  Open in a browser:"
  echo "    ${HF_SPACE_URL}"
  echo
  echo "  Click 'Run triage' on the prefilled Rajan dialogue. The Space is"
  echo "  configured (PTC_GEMMA4_ID, PTC_QWEN_ID) to use Gemma 4 E4B + Qwen 2.5-1.5B"
  echo "  so it fits free hardware. For full Gemma 4 31B numbers, see docs/RESULTS.md."
}

case "${SECTION}" in
  vllm)  section_vllm ;;
  rajan) section_rajan ;;
  space) section_space ;;
  all)   section_vllm; section_rajan; section_space ;;
  *)     echo "usage: $0 [vllm|rajan|space|all]" ; exit 1 ;;
esac

echo
echo "—— end of demo ——"
