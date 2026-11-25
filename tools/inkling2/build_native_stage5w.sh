#!/usr/bin/env bash
# Build ONLY Stage5W corrected native MoE shared library with CUDA Toolkit.
set -euo pipefail
ROOT="$(pwd -P)"
[[ -f "$ROOT/src/models/inkling2/moe_top8_stage5e.h" ]] || { echo 'Missing Stage5E header; run from worktree' >&2; exit 2; }
[[ -f "$ROOT/src/models/inkling2/moe_top8_stage5w.cu" ]] || { echo 'Missing Stage5W CUDA source' >&2; exit 2; }
OUT="${INKLING_ARTIFACTS:-$HOME/llmLab/inkling2-artifacts}/native5w"
mkdir -p "$OUT"
IMAGE="${INKLING_CUDA_DEVEL_IMAGE:-ninfer:inkling2-contract-cuda13.1.2}"
docker run --rm --gpus '"device=1"' --user "$(id -u):$(id -g)" \
  -v "$ROOT:/workspace:ro" -v "$OUT:/out" -w /workspace \
  --entrypoint bash "$IMAGE" -ceu '
    /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_120 \
      -Xcompiler=-fPIC -shared -I /workspace/src \
      /workspace/src/models/inkling2/moe_top8_stage5w.cu \
      -o /out/libinkling_moe_stage5w.so
    ls -lh /out/libinkling_moe_stage5w.so
  '
echo "BUILT corrected Stage5W: $OUT/libinkling_moe_stage5w.so"
