#!/usr/bin/env bash
# Load the resulting library with exactly the same dynamic loader used by the
# benchmark and verify ctypes can see symbols in both the extension and its DSO.
set -euo pipefail
ART="${INKLING_ARTIFACTS:-$HOME/llmLab/inkling2-artifacts}"
LIB="$ART/native6jkl/libinkling_native_stage6jkl.so"
BASE="$ART/native6ghi/libinkling_native_stage6ghi.so"
[[ -f "$LIB" && -f "$BASE" ]] || { echo 'ERROR: build the split DSO first' >&2; exit 2; }
IMAGE="${INKLING_HF_IMAGE:-ninfer:inkling2-hf-stage5p}"
docker run --rm --gpus '"device=0"' --user "$(id -u):$(id -g)" \
  -v "$ART:/artifacts:ro" --entrypoint python "$IMAGE" -c '
import ctypes
p="/artifacts/native6jkl/libinkling_native_stage6jkl.so"
lib=ctypes.CDLL(p)
for name in (
  "inkling6e_run", "inkling6g_store_create", "inkling6h_run", "inkling6i_run",
  "inkling6k_graph_create", "inkling6k_capture", "inkling6k_replay",
  "inkling6l_conv_create", "inkling6l_conv_apply"):
    getattr(lib, name)
print("STAGE6JKL SPLIT CTYPES ABI: PASS (including dependency symbol inkling6h_run)")
'
