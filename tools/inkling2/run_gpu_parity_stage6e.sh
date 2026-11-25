#!/usr/bin/env bash
set -euo pipefail
ROOT="$(pwd -P)"
ART="${INKLING_ARTIFACTS:-$HOME/llmLab/inkling2-artifacts}"
for f in "$ART/native5w/libinkling_moe_stage5w.so" "$ART/native6ef/libinkling_moe_stage6e.so"; do
  [[ -f "$f" ]] || { echo "Missing $f" >&2; exit 2; }
done
docker run --rm --gpus '"device=1"' --user "$(id -u):$(id -g)" \
  -e INKLING_STAGE5W_LIB=/artifacts/native5w/libinkling_moe_stage5w.so \
  -e INKLING_STAGE6E_LIB=/artifacts/native6ef/libinkling_moe_stage6e.so \
  -e PYTHONPATH=/workspace:/workspace/tools/inkling2 -e HOME=/tmp \
  -v "$ROOT:/workspace:ro" -v "$ART:/artifacts:ro" -w /workspace \
  --entrypoint python "${INKLING_HF_IMAGE:-ninfer:inkling2-hf-stage5p}" \
  -m unittest tools.inkling2.test_direct_cuda_stage6e -v
