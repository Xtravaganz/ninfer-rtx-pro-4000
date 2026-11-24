#!/usr/bin/env bash
# Run in an NVIDIA CUDA 13.1 devel image with a GPU.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
for cmd in nvcc cmake ninja ctest; do
    command -v "$cmd" >/dev/null 2>&1 || { echo "Missing $cmd; use the CUDA 13.1.2 developer image" >&2; exit 1; }
done
build_dir="$(mktemp -d "${TMPDIR:-/tmp}/inkling2-stage5b-XXXXXXXX")"
trap 'rm -rf "$build_dir"' EXIT
cmake -S tests/models/inkling2/stage5b_cuda -B "$build_dir" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${INKLING_CUDA_ARCH:-120}"
cmake --build "$build_dir" --parallel 4
ctest --test-dir "$build_dir" --output-on-failure
"$build_dir/ninfer_inkling2_forward_primitives_test"
