#!/usr/bin/env bash
set -euo pipefail
ROOT="$(pwd -P)"
ART="${INKLING_ARTIFACTS:-$HOME/llmLab/inkling2-artifacts}"
OUT="$ART/native7cd"
BASE="${INKLING_HF_IMAGE:-ninfer:inkling2-hf-stage5p}"
IMAGE="${INKLING_STAGE7CD_BUILD_IMAGE:-ninfer:inkling2-stage7cd-build}"
mkdir -p "$OUT"
# Build against exactly the runtime image's installed libtorch (no new PyTorch).
docker build --build-arg "BASE=$BASE" -f "$ROOT/tools/inkling2/Dockerfile.stage7cd_build" -t "$IMAGE" "$ROOT/tools/inkling2"
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e MAX_JOBS=2 \
 -v "$ROOT:/workspace:ro" -v "$OUT:/out" -w /workspace --entrypoint bash "$IMAGE" -ceu '
  tmp=$(mktemp -d /out/.build.XXXXXX); trap '\''rm -rf "$tmp"'\'' EXIT
  python tools/inkling2/build_stage7cd_extension.py "$tmp"
  python tools/inkling2/test_stage7cd_cpu.py "$tmp/inkling7cd_ops.so"
  mv -f "$tmp/inkling7cd_ops.so" /out/libinkling_stage7cd.so
 '
echo "STAGE7CD BUILD: $OUT/libinkling_stage7cd.so"
