#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen3-VL-2B-Instruct}"
PORT="${PORT:-8001}"

vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --dtype auto \
  --max-model-len 8192

