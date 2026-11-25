#!/usr/bin/env bash
# Original NInfer v3 real-weight core smoke: prelude 0..5 + core pass(es).
# Usage: bash tools/inkling2/run_real_core_stage5lm.sh [core-layers:6|12] [passes:1|2|3] [token-id]
set -euo pipefail
ROOT="$(pwd -P)"
[[ -f "$ROOT/tools/inkling2/real_core_stage5lm.py" ]] || { echo 'ERROR: invoke from Inkling worktree root' >&2; exit 2; }
LAYERS="${1:-6}"
PASSES="${2:-1}"
TOKEN="${3:-1000}"
[[ "$LAYERS" == 6 || "$LAYERS" == 12 ]] || { echo 'ERROR: core layers must be 6 or 12' >&2; exit 2; }
[[ "$PASSES" == 1 || "$PASSES" == 2 || "$PASSES" == 3 ]] || { echo 'ERROR: passes must be 1..3' >&2; exit 2; }
[[ "$LAYERS" == 12 || "$PASSES" == 1 ]] || { echo 'ERROR: multiple passes require 12 core layers' >&2; exit 2; }
[[ "$TOKEN" =~ ^[0-9]+$ ]] || { echo 'ERROR: invalid token ID' >&2; exit 2; }
OUT="${INKLING_ARTIFACTS:-$HOME/llmLab/inkling2-artifacts}"
HFCACHE="${HF_HOME:-$HOME/.cache/huggingface}"
REPO="$HFCACHE/hub/models--thinkingmachines--Inkling-2-Mini"
[[ -f "$REPO/refs/main" ]] || { echo 'ERROR: HF refs/main missing' >&2; exit 2; }
REV="$(cat "$REPO/refs/main")"
for f in "$OUT/remaining-full.ninfer" "$OUT/experts-full.ninfer" "$OUT/banks-full/manifest.json" "$REPO/snapshots/$REV/config.json"; do
  [[ -e "$f" ]] || { echo "ERROR: missing $f" >&2; exit 2; }
done
IMAGE="${INKLING_TORCH_IMAGE:-pytorch/pytorch:2.14.0-cuda13.0-cudnn9-runtime}"
echo "Stage5L/M real first core loops; physical=6..$((6+LAYERS-1)) passes=$PASSES token=$TOKEN revision=$REV"
docker run --rm --gpus '"device=1"' \
  -v "$ROOT:/workspace:ro" -v "$OUT:/artifacts:ro" -v "$HFCACHE:/hf:ro" \
  -w /workspace --entrypoint python "$IMAGE" \
  tools/inkling2/run_real_core_stage5lm.py \
  --source "/hf/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/$REV" \
  --nonexpert /artifacts/remaining-full.ninfer \
  --experts /artifacts/experts-full.ninfer \
  --banks /artifacts/banks-full \
  --core-layers "$LAYERS" --passes "$PASSES" --token-id "$TOKEN" --device cuda:0
