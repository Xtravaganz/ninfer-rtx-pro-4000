#!/usr/bin/env bash
# Stage 5J/5K - position-zero one-token real NInfer-v3 first block.
# Usage: bash tools/inkling2/run_real_block_stage5jk.sh [layers:5|6] [token-id]
set -euo pipefail
ROOT="$(pwd -P)"
[[ -f "$ROOT/tools/inkling2/real_full_stage5j.py" ]] || { echo 'ERROR: launch from Inkling worktree root' >&2; exit 2; }
LAYERS="${1:-6}"
TOKEN="${2:-1000}"
[[ "$LAYERS" == 5 || "$LAYERS" == 6 ]] || { echo 'ERROR: layers must be 5 or 6' >&2; exit 2; }
[[ "$TOKEN" =~ ^[0-9]+$ ]] || { echo 'ERROR: token must be a nonnegative integer' >&2; exit 2; }
OUT="${INKLING_ARTIFACTS:-$HOME/llmLab/inkling2-artifacts}"
HFCACHE="${HF_HOME:-$HOME/.cache/huggingface}"
REPO="$HFCACHE/hub/models--thinkingmachines--Inkling-2-Mini"
[[ -f "$REPO/refs/main" ]] || { echo "ERROR: missing $REPO/refs/main" >&2; exit 2; }
REV="$(cat "$REPO/refs/main")"
for file in "$OUT/remaining-full.ninfer" "$OUT/experts-full.ninfer" "$OUT/banks-full/manifest.json" "$REPO/snapshots/$REV/config.json"; do
  [[ -e "$file" ]] || { echo "ERROR: missing $file" >&2; exit 2; }
done
IMAGE="${INKLING_TORCH_IMAGE:-pytorch/pytorch:2.14.0-cuda13.0-cudnn9-runtime}"
echo "Stage5J/K real first attention block; layers=0..$((LAYERS-1)); token=$TOKEN; model_revision=$REV"
docker run --rm --gpus '"device=1"' \
  -v "$ROOT:/workspace:ro" -v "$OUT:/artifacts:ro" -v "$HFCACHE:/hf:ro" \
  -w /workspace --entrypoint python "$IMAGE" \
  tools/inkling2/run_real_block_stage5jk.py \
  --source "/hf/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/$REV" \
  --nonexpert /artifacts/remaining-full.ninfer \
  --experts /artifacts/experts-full.ninfer \
  --banks /artifacts/banks-full \
  --layers "$LAYERS" --token-id "$TOKEN" --device cuda:0
