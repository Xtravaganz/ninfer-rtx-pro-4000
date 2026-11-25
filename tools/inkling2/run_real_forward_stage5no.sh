#!/usr/bin/env bash
# Usage: bash tools/inkling2/run_real_forward_stage5no.sh [hidden|greedy] [token-id]
set -euo pipefail
ROOT="$(pwd -P)"
[[ -f "$ROOT/tools/inkling2/real_postlude_stage5no.py" ]] || { echo 'ERROR: run from Inkling Worktree root' >&2;exit 2; }
MODE="${1:-hidden}"
TOKEN="${2:-1000}"
[[ "$MODE" == hidden || "$MODE" == greedy ]] || { echo 'ERROR: mode must be hidden|greedy' >&2;exit 2; }
[[ "$TOKEN" =~ ^[0-9]+$ ]] || { echo 'ERROR: invalid token' >&2;exit 2; }
OUT="${INKLING_ARTIFACTS:-$HOME/llmLab/inkling2-artifacts}"
HFCACHE="${HF_HOME:-$HOME/.cache/huggingface}"
REPO="$HFCACHE/hub/models--thinkingmachines--Inkling-2-Mini"
[[ -f "$REPO/refs/main" ]] || { echo 'ERROR: missing HF refs/main' >&2;exit 2; }
REV="$(cat "$REPO/refs/main")"
for f in "$OUT/remaining-full.ninfer" "$OUT/experts-full.ninfer" "$OUT/banks-full/manifest.json" "$REPO/snapshots/$REV/config.json"; do
 [[ -e "$f" ]] || { echo "ERROR: missing $f" >&2;exit 2; }
done
IMAGE="${INKLING_TORCH_IMAGE:-pytorch/pytorch:2.14.0-cuda13.0-cudnn9-runtime}"
echo "Stage5N/O original weights: mode=$MODE token=$TOKEN revision=$REV"
docker run --rm --gpus '"device=0"' \
 -v "$ROOT:/workspace:ro" -v "$OUT:/artifacts:ro" -v "$HFCACHE:/hf:ro" \
 -w /workspace --entrypoint python "$IMAGE" \
 tools/inkling2/run_real_forward_stage5no.py \
 --source "/hf/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/$REV" \
 --nonexpert /artifacts/remaining-full.ninfer \
 --experts /artifacts/experts-full.ninfer \
 --banks /artifacts/banks-full \
 --mode "$MODE" --token-id "$TOKEN" --device cuda:0
