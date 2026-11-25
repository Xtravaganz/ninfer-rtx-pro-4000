#!/usr/bin/env bash
set -euo pipefail
ROOT="$(pwd -P)"
LAYER="${1:-0}"
[[ "$LAYER" =~ ^[0-9]+$ ]] && ((LAYER<24)) || { echo 'layer 0..23' >&2;exit 2; }
ART="${INKLING_ARTIFACTS:-$HOME/llmLab/inkling2-artifacts}"
LIB="$ART/native6jkl/libinkling_native_stage6jkl.so"
[[ -f "$LIB" ]] || { echo "Missing $LIB" >&2;exit 2; }
IMAGE="${INKLING_HF_IMAGE:-ninfer:inkling2-hf-stage5p}"
docker run --rm --gpus '"device=0"' --user "$(id -u):$(id -g)" \
 -e HOME=/tmp -e PYTHONPATH=/workspace:/workspace/tools/inkling2 \
 -v "$ROOT:/workspace:ro" -v "$ART:/artifacts:ro" -w /workspace \
 --entrypoint python "$IMAGE" tools/inkling2/benchmark_native_graph_stage6k.py \
 --nonexpert /artifacts/remaining-full.ninfer --experts /artifacts/experts-full.ninfer \
 --library /artifacts/native6jkl/libinkling_native_stage6jkl.so --layer "$LAYER"
