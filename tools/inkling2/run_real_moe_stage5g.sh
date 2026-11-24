#!/usr/bin/env bash
set -euo pipefail
# Run inside the existing CUDA-13.1 developer image, with the worktree mounted
# at /workspace and bundle artifact accessible via a second -v bind mount.
if [[ $# -ne 1 ]]; then
  echo "Usage: bash tools/inkling2/run_real_moe_stage5g.sh /input/bundle" >&2
  exit 2
fi
BUNDLE="$(realpath "$1")"
[[ -d "$BUNDLE" ]] || { echo "Missing bundle: $BUNDLE" >&2; exit 2; }
python3 tools/inkling2/prepare_real_moe_stage5f.py --verify --output "$BUNDLE"
BUILD_DIR="$(mktemp -d /tmp/inkling2-stage5g-XXXXXXXX)"
trap 'rm -rf "$BUILD_DIR"' EXIT
cmake -S tests/models/inkling2/stage5g_cuda -B "$BUILD_DIR" -G Ninja \
  -DNINFER_ROOT="$PWD" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build "$BUILD_DIR" --parallel 4
"$BUILD_DIR/ninfer_inkling2_real_moe_stage5g" "$BUNDLE"
