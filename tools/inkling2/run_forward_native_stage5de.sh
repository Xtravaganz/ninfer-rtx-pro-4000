#!/usr/bin/env bash
# Stage 5D+5E: independently compile and run both CUDA acceptance targets.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
for program in nvcc cmake ninja ctest; do
  command -v "$program" >/dev/null || { echo "Missing $program (use CUDA 13.1.2 devel image)" >&2; exit 1; }
done
build_dir="$(mktemp -d "${TMPDIR:-/tmp}/inkling2-stage5de-XXXXXXXX")"
trap 'rm -rf "$build_dir"' EXIT
for stage in d e; do
  cmake -S "tests/models/inkling2/stage5${stage}_cuda" \
    -B "$build_dir/stage5${stage}" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${INKLING_CUDA_ARCH:-120}"
  cmake --build "$build_dir/stage5${stage}" --parallel "${INKLING_BUILD_JOBS:-4}"
  ctest --test-dir "$build_dir/stage5${stage}" --output-on-failure
  "$build_dir/stage5${stage}/ninfer_inkling2_stage5${stage}_test"
done
