#!/usr/bin/env bash
# Stage 5H/5I real-weight Torch CUDA smoke in a pre-existing PyTorch image.
# Usage: bash tools/inkling2/run_real_sliding_stage5hi.sh [layers:1..5] [token-id]
set -euo pipefail
ROOT="$(pwd -P)"
[[ -f "$ROOT/tools/inkling2/real_sliding_stage5h.py" && -f "$ROOT/tools/inkling2/run_real_sliding_stage5hi.py" ]] || {
  echo "ERROR: run from the Inkling worktree root" >&2; exit 2;
}
LAYERS="${1:-1}"
TOKEN="${2:-1000}"
[[ "$LAYERS" =~ ^[1-5]$ && "$TOKEN" =~ ^[0-9]+$ ]] || {
  echo 'ERROR: layers must be 1..5 and token a nonnegative integer' >&2; exit 2;
}
OUT="${INKLING_ARTIFACTS:-$HOME/llmLab/inkling2-artifacts}"
HFCACHE="${HF_HOME:-$HOME/.cache/huggingface}"
REPO="$HFCACHE/hub/models--thinkingmachines--Inkling-2-Mini"
[[ -f "$REPO/refs/main" ]] || { echo "ERROR: missing local HF snapshot reference $REPO/refs/main" >&2; exit 2; }
REV="$(cat "$REPO/refs/main")"
for path in "$OUT/remaining-full.ninfer" "$OUT/experts-full.ninfer" "$OUT/banks-full/manifest.json" "$REPO/snapshots/$REV/config.json"; do
  [[ -e "$path" ]] || { echo "ERROR: missing prerequisite $path" >&2; exit 2; }
done
IMAGE="${INKLING_TORCH_IMAGE:-pytorch/pytorch:2.14.0-cuda13.0-cudnn9-runtime}"
echo "Stage 5H/5I real checkpoint smoke; layers=0..$((LAYERS-1)); token=$TOKEN; model_revision=$REV"
docker run --rm --gpus '"device=1"' \
  -v "$ROOT:/workspace:ro" \
  -v "$OUT:/artifacts:ro" \
  -v "$HFCACHE:/hf:ro" \
  -w /workspace \
  --entrypoint python \
  "$IMAGE" tools/inkling2/run_real_sliding_stage5hi.py \
  --source "/hf/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/$REV" \
  --nonexpert /artifacts/remaining-full.ninfer \
  --experts /artifacts/experts-full.ninfer \
  --banks /artifacts/banks-full \
  --layers "$LAYERS" --token-id "$TOKEN" --device cuda:0
