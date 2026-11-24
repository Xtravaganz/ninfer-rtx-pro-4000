#!/usr/bin/env bash
# Run INSIDE the existing CUDA 13.1 developer container with GPU device=1.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

for cmd in nvcc cmake ninja ctest; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: $cmd missing; use ninfer:inkling2-contract-cuda13.1.2" >&2
    exit 1
  fi
done
BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/inkling2-stage3c-XXXXXXXX")"
trap 'rm -rf "$BUILD_DIR"' EXIT
printf 'Stage 3C native MXFP8 CUDA A/B build: sm_%s\n' "${INKLING_CUDA_ARCH:-120}"
cmake -S tests/models/inkling2/cuda_ab -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="${INKLING_CUDA_ARCH:-120}"
cmake --build "$BUILD_DIR" --parallel 4
ctest --test-dir "$BUILD_DIR" --output-on-failure
"$BUILD_DIR/ninfer_inkling2_mxfp8_gemv_ab_test" --bench
