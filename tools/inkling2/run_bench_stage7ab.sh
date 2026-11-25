#!/usr/bin/env bash
set -euo pipefail
ROOT="$(pwd -P)"
ART="${INKLING_ARTIFACTS:-$HOME/llmLab/inkling2-artifacts}"
IMAGE="${INKLING_HF_IMAGE:-ninfer:inkling2-hf-stage5p}"
LAYER="${1:-0}"
[[ "$LAYER" =~ ^([0-9]|1[0-9]|2[0-3])$ ]] || { echo 'layer 0..23' >&2;exit 2; }
for f in remaining-full.ninfer experts-full.ninfer native6ghi/libinkling_native_stage6ghi.so native7ab/libinkling_native_stage7ab.so; do
 [[ -f "$ART/$f" ]] || { echo "Missing $ART/$f" >&2;exit 2; }
done
OPTIONS=()
[[ "${INKLING_STAGE7AB_GRAPH_TEST:-0}" == 1 ]] && OPTIONS+=(--graph)
docker run --rm --gpus '"device=0"' --user "$(id -u):$(id -g)" \
 -e HOME=/tmp -e PYTHONPATH=/workspace:/workspace/tools/inkling2 \
 -v "$ROOT:/workspace:ro" -v "$ART:/artifacts:ro" -w /workspace \
 --entrypoint python "$IMAGE" tools/inkling2/benchmark_native_stage7ab.py \
 --nonexpert /artifacts/remaining-full.ninfer --experts /artifacts/experts-full.ninfer \
 --base /artifacts/native6ghi/libinkling_native_stage6ghi.so \
 --library /artifacts/native7ab/libinkling_native_stage7ab.so --layer "$LAYER" \
 --samples "${INKLING_STAGE7AB_SAMPLES:-3}" --repeats "${INKLING_STAGE7AB_REPEATS:-25}" "${OPTIONS[@]}"
