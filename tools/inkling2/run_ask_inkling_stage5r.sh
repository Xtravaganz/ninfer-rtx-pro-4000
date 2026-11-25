#!/usr/bin/env bash
# Run from Inkling worktree: bash tools/inkling2/run_ask_inkling_stage5r.sh 'Was ist 2 + 2?' 12
# Preflight only: INKLING_TOKENIZE_ONLY=1 bash tools/inkling2/run_ask_inkling_stage5r.sh 'Hallo!'
set -euo pipefail
ROOT="$(pwd -P)"
[[ -f "$ROOT/tools/inkling2/ask_inkling_stage5r.py" && -f "$ROOT/tools/inkling2/run_streaming_stage5q.py" ]] || {
  echo 'ERROR: run this from the Inkling NInfer worktree with Stage5Q installed' >&2; exit 2; }
QUESTION="${1:-}"
[[ -n "$QUESTION" ]] || { echo 'Usage: bash tools/inkling2/run_ask_inkling_stage5r.sh "Frage" [new_tokens]' >&2; exit 2; }
MAX_NEW="${2:-12}"
[[ "$MAX_NEW" =~ ^[0-9]+$ ]] && (( MAX_NEW >= 1 && MAX_NEW <= 64 )) || {
  echo 'ERROR: new_tokens must be 1..64' >&2; exit 2; }
ARTIFACTS="${INKLING_ARTIFACTS:-$HOME/llmLab/inkling2-artifacts}"
HF_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
REPO="$HF_DIR/hub/models--thinkingmachines--Inkling-2-Mini"
[[ -f "$REPO/refs/main" ]] || { echo 'ERROR: HF snapshot reference missing' >&2; exit 2; }
REV="$(cat "$REPO/refs/main")"
[[ "$REV" =~ ^[0-9a-f]{40}$ ]] || { echo 'ERROR: invalid HF snapshot revision' >&2; exit 2; }
SOURCE="/hf/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/$REV"
IMAGE="${INKLING_HF_IMAGE:-ninfer:inkling2-hf-stage5p}"
OPTIONS=()
[[ "${INKLING_RAW_PROMPT:-0}" == 1 ]] && OPTIONS+=(--raw)
[[ "${INKLING_TOKENIZE_ONLY:-0}" == 1 ]] && OPTIONS+=(--tokenize-only)
[[ "${INKLING_TOKENIZE_ONLY:-0}" == 1 ]] || {
  for p in "$ARTIFACTS/remaining-full.ninfer" "$ARTIFACTS/experts-full.ninfer" "$ARTIFACTS/banks-full/manifest.json"; do
    [[ -f "$p" ]] || { echo "ERROR: missing $p" >&2; exit 2; }
  done
}
# HF image provides the tokenizer but we deliberately DO NOT instantiate its full model.
docker run --rm --gpus '"device=0"' --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v "$ROOT:/workspace:ro" -v "$HF_DIR:/hf:ro" -v "$ARTIFACTS:/artifacts:ro" \
  -w /workspace --entrypoint python "$IMAGE" \
  tools/inkling2/ask_inkling_stage5r.py \
  --source "$SOURCE" --nonexpert /artifacts/remaining-full.ninfer \
  --experts /artifacts/experts-full.ninfer --banks /artifacts/banks-full \
  --question "$QUESTION" --max-new-tokens "$MAX_NEW" --device cuda:0 "${OPTIONS[@]}"
