#!/usr/bin/env bash
set -euo pipefail
ROOT="$(pwd -P)"
[[ -f "$ROOT/tools/inkling2/gateup_audit_stage5st.py" ]] || { echo 'ERROR: run inside Inkling worktree' >&2; exit 2; }
MODE="${1:-audit}"
case "$MODE" in audit|compare|question) ;; *) echo 'Usage: bash tools/inkling2/run_gateup_stage5st.sh audit|compare|question [question] [new_tokens]' >&2; exit 2;; esac
HF_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
REPO="$HF_DIR/hub/models--thinkingmachines--Inkling-2-Mini"
[[ -f "$REPO/refs/main" ]] || { echo 'ERROR: missing HF snapshot refs/main' >&2; exit 2; }
REV="$(cat "$REPO/refs/main")"
[[ "$REV" =~ ^[0-9a-f]{40}$ ]] || { echo 'ERROR: invalid HF revision' >&2; exit 2; }
OUT="${INKLING_ARTIFACTS:-$HOME/llmLab/inkling2-artifacts}"
for file in "$OUT/remaining-full.ninfer" "$OUT/experts-full.ninfer" "$OUT/banks-full/manifest.json"; do
 [[ -f "$file" ]] || { echo "ERROR: missing $file" >&2; exit 2; }
done
IMAGE="${INKLING_HF_IMAGE:-ninfer:inkling2-hf-stage5p}"
ARGS=("$MODE" --source "/hf/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/$REV" --nonexpert /artifacts/remaining-full.ninfer --experts /artifacts/experts-full.ninfer --banks /artifacts/banks-full --device cuda:0)
if [[ "$MODE" == question ]]; then
  QUESTION="${2:-What is 2 + 2?}"
  NUM="${3:-6}"
  [[ "$NUM" =~ ^[0-9]+$ ]] && (( NUM>=1 && NUM<=64 )) || { echo 'ERROR: new_tokens 1..64' >&2; exit 2; }
  ARGS+=(--question "$QUESTION" --max-new-tokens "$NUM")
fi
if [[ "$MODE" == compare ]]; then
  ARGS+=(--token "${2:-1000}")
fi
# Existing image needs only tokenizer/torch: never loads full HF model.
docker run --rm --gpus '"device=0"' --user "$(id -u):$(id -g)" \
 -e PYTHONPATH=/workspace:/workspace/tools/inkling2 -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
 -v "$ROOT:/workspace:ro" -v "$HF_DIR:/hf:ro" -v "$OUT:/artifacts:ro" \
 -w /workspace --entrypoint python "$IMAGE" \
 tools/inkling2/gateup_audit_stage5st.py "${ARGS[@]}"
