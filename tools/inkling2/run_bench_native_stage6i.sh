#!/usr/bin/env bash
set -euo pipefail
ROOT="$(pwd -P)"
LAYER="${1:-0}"; TOKENS="${2:-16}"; REPEATS="${3:-8}"
[[ "$LAYER" =~ ^[0-9]+$ ]] && ((LAYER<24)) || { echo 'layer must be 0..23' >&2; exit 2; }
[[ "$TOKENS" =~ ^[0-9]+$ ]] && ((TOKENS>=1 && TOKENS<=128)) || { echo 'tokens must be 1..128' >&2; exit 2; }
[[ "$REPEATS" =~ ^[0-9]+$ ]] && ((REPEATS>=1 && REPEATS<=100)) || { echo 'repeats 1..100' >&2; exit 2; }
ART="${INKLING_ARTIFACTS:-$HOME/llmLab/inkling2-artifacts}"
LIB="$ART/native6ghi/libinkling_native_stage6ghi.so"
[[ -f "$LIB" ]] || { echo "Missing native library $LIB" >&2; exit 2; }
IMAGE="${INKLING_HF_IMAGE:-ninfer:inkling2-hf-stage5p}"
docker run --rm --gpus '"device=0"' --user "$(id -u):$(id -g)" \
 -e HOME=/tmp -e PYTHONPATH=/workspace:/workspace/tools/inkling2 \
 -v "$ROOT:/workspace:ro" -v "$ART:/artifacts:ro" -w /workspace \
 --entrypoint python "$IMAGE" \
 tools/inkling2/benchmark_native_stage6i.py \
 --library /artifacts/native6ghi/libinkling_native_stage6ghi.so \
 --nonexpert /artifacts/remaining-full.ninfer --experts /artifacts/experts-full.ninfer \
 --layer "$LAYER" --tokens "$TOKENS" --repeats "$REPEATS" \
 --route-scale "${INKLING_ROUTE_SCALE:-8.0}"
