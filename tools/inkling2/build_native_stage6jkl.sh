#!/usr/bin/env bash
set -euo pipefail
ROOT="$(pwd -P)"
for file in \
 src/models/inkling2/moe_top8_stage5e.h \
 src/models/inkling2/moe_top8_stage6e.cu \
 src/models/inkling2/native_resident_stage6gh.h \
 src/models/inkling2/native_resident_stage6gh.cu \
 src/models/inkling2/native_batch_moe_stage6i.cu \
 src/models/inkling2/native_6h_bridge_stage6jkl_hotfix2.cu \
 src/models/inkling2/native_extensions_stage6jkl.cu; do
  [[ -f "$ROOT/$file" ]] || { echo "ERROR missing $file (6E + 6G/H/I + 6J/K/L required)" >&2; exit 2; }
done
ART="${INKLING_ARTIFACTS:-$HOME/llmLab/inkling2-artifacts}"
OUT="$ART/native6jkl"
mkdir -p "$OUT"
IMAGE="${INKLING_CUDA_DEVEL_IMAGE:-ninfer:inkling2-contract-cuda13.1.2}"
# Pin the host's physical GPU0. CUDA_VISIBLE_DEVICES inside the container is cuda:0.
docker run --rm --gpus '"device=0"' --user "$(id -u):$(id -g)" \
 -v "$ROOT:/workspace:ro" -v "$OUT:/out" -w /workspace \
 --entrypoint bash "$IMAGE" -ceu '
  /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_120 -Xcompiler=-fPIC \
    -shared -I /workspace/src \
    /workspace/src/models/inkling2/moe_top8_stage6e.cu \
    /workspace/src/models/inkling2/native_resident_stage6gh.cu \
    /workspace/src/models/inkling2/native_batch_moe_stage6i.cu \
    /workspace/src/models/inkling2/native_6h_bridge_stage6jkl_hotfix2.cu \
    /workspace/src/models/inkling2/native_extensions_stage6jkl.cu \
    -Xlinker -z -Xlinker defs \
    -o /out/libinkling_native_stage6jkl.so
  lib=/out/libinkling_native_stage6jkl.so
  # The graph extension requires the exact Stage 6G/H exports in this same DSO.
  # Detect stale/incomplete build inputs before Python ctypes starts.
  symbols="$(nm -D --defined-only "$lib" | sed -E "s/^.*[[:space:]]//")"
  for symbol in inkling6g_store_create inkling6g_upload_layer inkling6g_layer_view \
                inkling6h_run inkling6e_workspace_create inkling6i_workspace_create \
                inkling6k_graph_create inkling6k_capture inkling6k_replay \
                inkling6l_conv_create inkling6l_conv_apply; do
    if ! grep -Fxq "$symbol" <<<"$symbols"; then
      echo "ERROR: native6jkl missing exported symbol: $symbol" >&2
      nm -D "$lib" | grep -E "inkling6[eg hijkl]" | head -65 >&2 || true
      exit 21
    fi
  done
  echo "STAGE6JKL ABI EXPORTS: PASS"
  ls -lh "$lib"
'
echo "STAGE6JKL: device=0 native library: $OUT/libinkling_native_stage6jkl.so"
