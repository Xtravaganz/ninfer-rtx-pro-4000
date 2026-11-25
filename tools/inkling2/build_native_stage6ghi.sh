#!/usr/bin/env bash
set -euo pipefail
ROOT="$(pwd -P)"
for f in \
 src/models/inkling2/moe_top8_stage5e.h \
 src/models/inkling2/moe_top8_stage6e.cu \
 src/models/inkling2/native_resident_stage6gh.h \
 src/models/inkling2/native_resident_stage6gh.cu \
 src/models/inkling2/native_batch_moe_stage6i.cu; do
  [[ -f "$ROOT/$f" ]] || { echo "ERROR: Missing $f; install Stage6E/F + Stage6G/H/I" >&2; exit 2; }
done
ART="${INKLING_ARTIFACTS:-$HOME/llmLab/inkling2-artifacts}"
OUT="$ART/native6ghi"
mkdir -p "$OUT"
IMAGE="${INKLING_CUDA_DEVEL_IMAGE:-ninfer:inkling2-contract-cuda13.1.2}"
docker run --rm --gpus '"device=1"' --user "$(id -u):$(id -g)" \
 -v "$ROOT:/workspace:ro" -v "$OUT:/out" -w /workspace \
 --entrypoint bash "$IMAGE" -ceu '
  /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_120 \
    -Xcompiler=-fPIC -shared -I /workspace/src \
    /workspace/src/models/inkling2/moe_top8_stage6e.cu \
    /workspace/src/models/inkling2/native_resident_stage6gh.cu \
    /workspace/src/models/inkling2/native_batch_moe_stage6i.cu \
    -o /out/libinkling_native_stage6ghi.so
  ls -lh /out/libinkling_native_stage6ghi.so
'
echo "Stage6G/H/I native sm_120 library ready: $OUT/libinkling_native_stage6ghi.so"
