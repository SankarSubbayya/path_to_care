# HF Spaces Dockerfile for the Next.js Path to Care frontend.
# Multi-stage so the runtime image stays small (~250 MB) and contains no
# build toolchain. The Next.js app SSRs all triage requests; heavy model
# inference happens on the AMD MI300X via vLLM HTTP (env var below).

FROM node:20-bookworm-slim AS deps
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci --no-audit --no-fund

FROM node:20-bookworm-slim AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

FROM node:20-bookworm-slim AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
# HF Spaces injects PORT (typically 7860); fall back for local docker run.
ENV PORT=7860
ENV HOSTNAME=0.0.0.0

# Connection to the MI300X-hosted vLLM. Override at runtime; default is the
# droplet's public IP exposing vllm/vllm-openai-rocm:v0.20.1 with the demo key.
ENV PTC_VLLM_GEMMA4_URL="http://165.245.137.117:8000/v1"
ENV PTC_VLLM_GEMMA4_MODEL_ID="google/gemma-4-31B-it"
# PTC_VLLM_API_KEY should be set as an HF Space "secret" (not a "variable")
# so it isn't displayed in the Space UI. Default here is the demo key from
# scripts/vllm_serve.sh — replace before going to production.
ENV PTC_VLLM_API_KEY="ptc-demo-2026-amd"

# node:20-bookworm-slim ships with user `node` at UID 1000 — exactly what
# HF Spaces requires. No need to create our own.
COPY --from=builder --chown=node:node /app/.next ./.next
COPY --from=builder --chown=node:node /app/public ./public
COPY --from=builder --chown=node:node /app/node_modules ./node_modules
COPY --from=builder --chown=node:node /app/package.json ./package.json
COPY --from=builder --chown=node:node /app/next.config.ts ./next.config.ts

USER node
EXPOSE 7860

# `next start` reads PORT/HOSTNAME from env and serves the app.
CMD ["sh", "-c", "node node_modules/next/dist/bin/next start --port ${PORT} --hostname ${HOSTNAME}"]
