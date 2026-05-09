#!/usr/bin/env bash
# Launch a Gemma 4 31B-it OpenAI-compatible vLLM server inside the AMD ROCm
# Docker image. See docs/VLLM_SERVE.md for the rationale (why Docker, why
# these flags) and for the FP8 / low-mem variant.
#
# Usage:
#   bash scripts/vllm_serve.sh                                  # bf16, 0.9 GPU mem
#   bash scripts/vllm_serve.sh fp8                              # online fp8, 0.4 GPU mem
#   API_KEY=my-key MAX_LEN=4096 bash scripts/vllm_serve.sh      # override
#
# After this script returns, the server takes ~90s to load weights and start
# serving. Watch with:
#   docker logs -f ptc-vllm
#
# Health probe:
#   curl http://localhost:8000/v1/models -H "Authorization: Bearer $API_KEY"

set -euo pipefail

MODE="${1:-bf16}"
NAME="${NAME:-ptc-vllm}"
IMAGE="${IMAGE:-vllm/vllm-openai-rocm:v0.20.1}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-google/gemma-4-31B-it}"
API_KEY="${API_KEY:-ptc-demo-2026-amd}"
HF_CACHE="${HF_CACHE:-/root/.cache/huggingface}"

case "$MODE" in
  bf16)
    MAX_LEN="${MAX_LEN:-8192}"
    GPU_UTIL="${GPU_UTIL:-0.9}"
    EXTRA_ARGS="--dtype bfloat16"
    ;;
  fp8)
    MAX_LEN="${MAX_LEN:-4096}"
    GPU_UTIL="${GPU_UTIL:-0.4}"
    EXTRA_ARGS="--quantization fp8 --dtype bfloat16"
    ;;
  *)
    echo "Usage: $0 [bf16|fp8]" ; exit 1 ;;
esac

# Stop any prior container with this name (idempotent).
docker rm -f "${NAME}" >/dev/null 2>&1 || true

echo "== Launching ${NAME} =="
echo "  image:     ${IMAGE}"
echo "  model:     ${MODEL}"
echo "  port:      ${PORT}"
echo "  api_key:   ${API_KEY}"
echo "  mode:      ${MODE} (max_len=${MAX_LEN}, gpu_util=${GPU_UTIL})"

docker run -d --name "${NAME}" \
  --group-add=video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device /dev/kfd \
  --device /dev/dri \
  -p "${PORT}:${PORT}" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -e HF_HOME=/root/.cache/huggingface \
  --entrypoint /bin/bash \
  "${IMAGE}" \
  -c "vllm serve ${MODEL} \
        --host 0.0.0.0 --port ${PORT} \
        --api-key ${API_KEY} \
        --max-model-len ${MAX_LEN} \
        --gpu-memory-utilization ${GPU_UTIL} \
        ${EXTRA_ARGS}"

echo
echo "Container started. Tail logs with:"
echo "  docker logs -f ${NAME}"
echo
echo "Wait ~90s for weights to load, then:"
echo "  curl http://localhost:${PORT}/v1/models -H 'Authorization: Bearer ${API_KEY}'"
