#!/usr/bin/env bash
# Stage 6J/K/L ABI split: link only the graph/conv extension against the proven
# resident Stage 6G/H/I DSO.  Never recompile or shadow 6H's strong C ABI.
set -euo pipefail
ROOT="$(pwd -P)"
SOURCE="$ROOT/src/models/inkling2/native_extensions_stage6jkl.cu"
HEADER="$ROOT/src/models/inkling2/native_resident_stage6gh.h"
ART="${INKLING_ARTIFACTS:-$HOME/llmLab/inkling2-artifacts}"
BASE="$ART/native6ghi/libinkling_native_stage6ghi.so"
OUT="$ART/native6jkl"
LIB="$OUT/libinkling_native_stage6jkl.so"
IMAGE="${INKLING_CUDA_DEVEL_IMAGE:-ninfer:inkling2-contract-cuda13.1.2}"

[[ -f "$SOURCE" && -f "$HEADER" ]] || {
  echo 'ERROR: Stage 6G/H/I and Stage 6J/K/L sources are required' >&2; exit 2;
}
[[ -f "$BASE" ]] || {
  echo "ERROR: Missing known-working Stage6G/H/I library: $BASE" >&2
  echo 'Do not rerun the original 6G/H/I builder: it may target a different GPU. Use your existing tested DSO.' >&2
  exit 2
}
[[ "$(id -u)" != 0 ]] || echo 'NOTE: executing as root; the output file will be root-owned.' >&2
mkdir -p "$OUT"

# Keep a broken previous build for comparison. It is never used at runtime.
# The output is replaced only after an ABI/link check passes.
docker run --rm --gpus '"device=0"' --user "$(id -u):$(id -g)" \
  -v "$ROOT:/workspace:ro" -v "$ART/native6ghi:/deps:ro" -v "$OUT:/out" \
  -w /workspace --entrypoint bash "$IMAGE" -ceu '
  base=/deps/libinkling_native_stage6ghi.so
  tmp=/out/.libinkling_native_stage6jkl.split.tmp.so
  trap '\''rm -f "$tmp"'\'' EXIT
  # This is the exact strong ABI used successfully by Stage 6H.
  nm -D --defined-only "$base" | grep -Eq " [TW] inkling6h_run$" || {
    echo "ERROR: original Stage6G/H/I DSO does not export inkling6h_run" >&2
    exit 22
  }
  /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_120 -Xcompiler=-fPIC \
    -shared -I /workspace/src \
    /workspace/src/models/inkling2/native_extensions_stage6jkl.cu \
    -L/deps -linkling_native_stage6ghi \
    -Xlinker -z -Xlinker defs \
    -Xlinker -rpath -Xlinker "\$ORIGIN/../native6ghi" \
    -o "$tmp"
  # Only Stage6K/L is compiled in this DSO; Stage6H stays in the dependency.
  for sym in inkling6k_graph_create inkling6k_graph_destroy \
      inkling6k_capture inkling6k_replay inkling6l_conv_create \
      inkling6l_conv_destroy inkling6l_conv_apply; do
    if ! nm -D --defined-only "$tmp" | grep -Eq " [TW] ${sym}$"; then
      echo "ERROR: missing native Stage6K/L export: $sym" >&2; exit 23
    fi
  done
  # Assert the extension references the actual 6H implementation rather than
  # accidentally defining another one.
  if nm -D --defined-only "$tmp" | grep -q " inkling6h_run$"; then
    echo "ERROR: Stage6J/K/L DSO unexpectedly redefines Stage6H" >&2; exit 24
  fi
  readelf -d "$tmp" | grep -q "\[libinkling_native_stage6ghi.so\]" || {
    echo "ERROR: link dependency on Stage6G/H/I is absent" >&2; exit 25
  }
  readelf -d "$tmp" | grep -Fq "\$ORIGIN/../native6ghi" || {
    echo "ERROR: relative runtime library lookup is absent" >&2; exit 26
  }
  lib=/out/libinkling_native_stage6jkl.so
  backup=/out/libinkling_native_stage6jkl.before_split.so
  if [[ -f "$lib" && ! -f "$backup" ]]; then cp -p "$lib" "$backup"; fi
  mv -f "$tmp" "$lib"
  echo "STAGE6JKL SPLIT ABI: PASS (Stage6H from native6ghi, Stage6K/L from native6jkl)"
  echo "STAGE6JKL CUDA TARGET: sm_120, physical device=0"
  nm -D "$lib" | grep -E "inkling6[hkl]_" || true
  ls -lh "$lib"
'
echo "STAGE6JKL SPLIT: $LIB"
echo 'NEXT: bash tools/inkling2/check_stage6jkl_split_abi.sh'
