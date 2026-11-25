#!/usr/bin/env bash
set -euo pipefail
ROOT="$(pwd -P)"
ART="${INKLING_ARTIFACTS:-$HOME/llmLab/inkling2-artifacts}"
LIB="$ART/native6jkl/libinkling_native_stage6jkl.so"
[[ -f "$LIB" ]] || { echo "Missing $LIB" >&2;exit 2; }
IMAGE="${INKLING_HF_IMAGE:-ninfer:inkling2-hf-stage5p}"
# Test both K=4 paths, including preserved state across a second call.
for mode in static dynamic;do
  FLAGS=(--channels 2304)
  if [[ "$mode" == dynamic ]];then FLAGS=(--channels 1024 --dynamic);fi
  docker run --rm --gpus '"device=0"' --user "$(id -u):$(id -g)" \
   -e HOME=/tmp -e PYTHONPATH=/workspace:/workspace/tools/inkling2 \
   -v "$ROOT:/workspace:ro" -v "$ART:/artifacts:ro" -w /workspace \
   --entrypoint python "$IMAGE" tools/inkling2/benchmark_native_conv_stage6l.py \
   --library /artifacts/native6jkl/libinkling_native_stage6jkl.so --tokens 16 "${FLAGS[@]}"
done
