#!/usr/bin/env bash
set -euo pipefail
ROOT="$(pwd -P)"
for f in src/models/inkling2/moe_top8_stage6e.cu src/models/inkling2/moe_top8_stage5e.h src/models/inkling2/batch_dispatch_stage6f.cpp src/models/inkling2/plan.cpp; do
  [[ -f "$ROOT/$f" ]] || { echo "Missing $f; run from installed worktree" >&2; exit 2; }
done
OUT="${INKLING_ARTIFACTS:-$HOME/llmLab/inkling2-artifacts}/native6ef"
mkdir -p "$OUT"
IMAGE="${INKLING_CUDA_DEVEL_IMAGE:-ninfer:inkling2-contract-cuda13.1.2}"
docker run --rm --gpus '"device=1"' --user "$(id -u):$(id -g)" \
  -v "$ROOT:/workspace:ro" -v "$OUT:/out" -w /workspace \
  --entrypoint bash "$IMAGE" -ceu '
    /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_120 \
      -Xcompiler=-fPIC -shared -I /workspace/src \
      /workspace/src/models/inkling2/moe_top8_stage6e.cu \
      -o /out/libinkling_moe_stage6e.so
    g++ -std=c++20 -O2 -fPIC -shared -I /workspace/src \
      /workspace/src/models/inkling2/batch_dispatch_stage6f.cpp \
      /workspace/src/models/inkling2/plan.cpp \
      -o /out/libinkling_batch_stage6f.so
    g++ -std=c++20 -O2 -I /workspace/src \
      /workspace/src/models/inkling2/batch_dispatch_stage6f.cpp \
      /workspace/src/models/inkling2/plan.cpp \
      /workspace/tests/models/inkling2/stage6f/test_batch_dispatch_stage6f.cpp \
      -o /out/test_inkling_batch_stage6f
    /out/test_inkling_batch_stage6f
    ls -lh /out/libinkling_moe_stage6e.so /out/libinkling_batch_stage6f.so
  '
echo "BUILT Stage6E native GPU pointer MoE + Stage6F host execution scheduler: $OUT"
