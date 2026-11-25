#!/usr/bin/env bash
set -euo pipefail
ROOT="$(pwd -P)"
TOKENS="${1:-16}"
[[ "$TOKENS" =~ ^[0-9]+$ ]] && ((TOKENS >= 1 && TOKENS <= 256)) || { echo 'Token count 1..256 expected' >&2; exit 2; }
IMAGE="${INKLING_TORCH_IMAGE:-pytorch/pytorch:2.14.0-cuda13.0-cudnn9-runtime}"
docker run --rm --gpus '"device=0"' -v "$ROOT:/workspace:ro" -w /workspace \
  -e PYTHONPATH=/workspace:/workspace/tools/inkling2 \
  --entrypoint python "$IMAGE" \
  tools/inkling2/benchmark_batch_stage6c.py --tokens "$TOKENS" --device cuda:0 --repeats 20
