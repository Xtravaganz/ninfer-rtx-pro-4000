#!/usr/bin/env bash
set -euo pipefail
ROOT="$(pwd -P)"
ART="${INKLING_ARTIFACTS:-$HOME/llmLab/inkling2-artifacts}"
BASE="$ART/native6ghi/libinkling_native_stage6ghi.so"
OUT="$ART/native7ab"
IMAGE="${INKLING_CUDA_DEVEL_IMAGE:-ninfer:inkling2-contract-cuda13.1.2}"
[[ -f "$BASE" ]] || { echo "Missing tested Stage6G/H/I DSO: $BASE" >&2; exit 2; }
[[ -f "$ROOT/src/models/inkling2/native_decode_stage7ab.cu" ]] || exit 2
mkdir -p "$OUT"
docker run --rm --gpus '"device=0"' --user "$(id -u):$(id -g)" \
 -v "$ROOT:/workspace:ro" -v "$ART/native6ghi:/deps:ro" -v "$OUT:/out" \
 -w /workspace --entrypoint bash "$IMAGE" -ceu '
 base=/deps/libinkling_native_stage6ghi.so
 tmp=/out/.libinkling_native_stage7ab.tmp.so
 trap '\''rm -f "$tmp"'\'' EXIT
 nm -D --defined-only "$base" > /tmp/stage7ab-base-symbols.txt
 grep -Eq " [TW] inkling6g_layer_view$" /tmp/stage7ab-base-symbols.txt
 grep -Eq " [TW] inkling6h_run$" /tmp/stage7ab-base-symbols.txt
 /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_120 -Xcompiler=-fPIC \
   -shared -I /workspace/src /workspace/src/models/inkling2/native_decode_stage7ab.cu \
   -L/deps -linkling_native_stage6ghi -Xlinker -z -Xlinker defs \
   -Xlinker -rpath -Xlinker "\$ORIGIN/../native6ghi" -o "$tmp"
 nm -D --defined-only "$tmp" > /tmp/stage7ab-symbols.txt
 for sym in inkling7ab_create inkling7ab_destroy inkling7ab_run inkling7ab_routes inkling7ab_reference_routes; do
   grep -Eq " [TW] ${sym}$" /tmp/stage7ab-symbols.txt
 done
 if grep -Eq " inkling6[eghikl]_" /tmp/stage7ab-symbols.txt; then echo "Unexpected Stage6 symbol redefinition" >&2; exit 2; fi
 readelf -d "$tmp" > /tmp/stage7ab-dynamic.txt
 grep -Fq "[libinkling_native_stage6ghi.so]" /tmp/stage7ab-dynamic.txt
 grep -Fq "\$ORIGIN/../native6ghi" /tmp/stage7ab-dynamic.txt
 mv -f "$tmp" /out/libinkling_native_stage7ab.so
 echo "STAGE7AB SPLIT ABI: PASS (Stage6G/H/I unchanged; sm_120, physical GPU 0)"
 ls -lh /out/libinkling_native_stage7ab.so
'
