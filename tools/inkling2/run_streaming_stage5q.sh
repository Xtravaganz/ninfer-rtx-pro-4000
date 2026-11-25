#!/usr/bin/env bash
# Stage5Q. Usage: bash tools/inkling2/run_streaming_stage5q.sh fixed 1000,1001
#              bash tools/inkling2/run_streaming_stage5q.sh greedy 1000 2
set -euo pipefail
ROOT="$(pwd -P)"
[[ -f "$ROOT/tools/inkling2/streaming_stage5q.py" ]] || { echo 'ERROR: run from Inkling worktree root' >&2;exit 2; }
MODE="${1:-fixed}"; INPUT="${2:-1000,1001}"; STEPS="${3:-2}"
ARTIFACTS="${INKLING_ARTIFACTS:-$HOME/llmLab/inkling2-artifacts}"
HF_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
REPO="$HF_DIR/hub/models--thinkingmachines--Inkling-2-Mini"
[[ -f "$REPO/refs/main" ]] || { echo 'ERROR: missing HF revision' >&2;exit 2; }
REV="$(cat "$REPO/refs/main")"
[[ "$REV" =~ ^[a-f0-9]{40}$ ]] || { echo 'ERROR: unexpected HF revision' >&2;exit 2; }
REPORTS="$ARTIFACTS/cache5q"
mkdir -p "$REPORTS"
IMAGE="${INKLING_TORCH_IMAGE:-pytorch/pytorch:2.14.0-cuda13.0-cudnn9-runtime}"
OPTIONS=()
case "$MODE" in
 fixed)
   [[ "$INPUT" =~ ^[0-9]+(,[0-9]+){0,2}$ ]] || { echo 'ERROR: fixed input requires 1..3 comma-separated token IDs' >&2;exit 2; }
   OPTIONS=(--tokens "$INPUT")
   BASENAME="fixed-${INPUT//,/_}"
   ;;
 greedy)
   [[ "$INPUT" =~ ^[0-9]+$ && "$STEPS" =~ ^[123]$ ]] || { echo 'ERROR: greedy seed and 1..3 steps required' >&2;exit 2; }
   OPTIONS=(--greedy --seed-token "$INPUT" --steps "$STEPS")
   BASENAME="greedy-${INPUT}-${STEPS}"
   ;;
 *) echo 'ERROR: mode must be fixed|greedy' >&2;exit 2;;
esac
for f in "$ARTIFACTS/remaining-full.ninfer" "$ARTIFACTS/experts-full.ninfer" \
  "$ARTIFACTS/banks-full/manifest.json"; do [[ -f "$f" ]] || { echo "ERROR: missing $f" >&2;exit 2; }; done
printf 'Stage5Q: %s position-aware token decoder; revision=%s; input=%s\n' "$MODE" "$REV" "$INPUT"
docker run --rm --gpus '"device=0"' --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$ROOT:/workspace:ro" -v "$ARTIFACTS:/artifacts:ro" -v "$HF_DIR:/hf:ro" \
  -v "$REPORTS:/reports" -w /workspace --entrypoint python "$IMAGE" \
  tools/inkling2/run_streaming_stage5q.py \
  --source "/hf/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/$REV" \
  --nonexpert /artifacts/remaining-full.ninfer --experts /artifacts/experts-full.ninfer \
  --banks /artifacts/banks-full --device cuda:0 --check-legacy \
  "${OPTIONS[@]}" --output "/reports/$BASENAME-$REV.json"
