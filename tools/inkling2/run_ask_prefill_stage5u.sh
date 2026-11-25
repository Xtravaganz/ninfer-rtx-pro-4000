#!/usr/bin/env bash
# Stage 5U, opt-in. Same checkpoint, tokenizer, corrected W13, 48-slot Torch path.
# Baseline: INKLING_PREFILL_MODE=baseline bash ... 'What is 2 + 2?' 2
# Cached:   INKLING_PREFILL_MODE=cached   bash ... 'What is 2 + 2?' 2
set -euo pipefail
ROOT="$(pwd -P)"
for file in tools/inkling2/ask_prefill_stage5u.py tools/inkling2/ask_inkling_stage5r.py tools/inkling2/gateup_audit_stage5st.py tools/inkling2/streaming_stage5q.py; do
  [[ -f "$ROOT/$file" ]] || { echo "ERROR: missing $file; run from Inkling worktree with stages 5Q/5R/5S/5U installed" >&2; exit 2; }
done
QUESTION="${1:-}"
[[ -n "$QUESTION" ]] || { echo 'Usage: bash tools/inkling2/run_ask_prefill_stage5u.sh "Frage" [max_new_tokens]' >&2; exit 2; }
MAX_NEW="${2:-3}"
MODE="${INKLING_PREFILL_MODE:-cached}"
[[ "$MODE" == baseline || "$MODE" == cached ]] || { echo 'ERROR: INKLING_PREFILL_MODE must be baseline or cached' >&2; exit 2; }
NONEXPERT_MIB="${INKLING_NONEXPERT_CACHE_MIB:-2048}"
EXPERT_MIB="${INKLING_EXPERT_CACHE_MIB:-4096}"
for entry in "$MAX_NEW" "$NONEXPERT_MIB" "$EXPERT_MIB"; do
  [[ "$entry" =~ ^[0-9]+$ ]] || { echo 'ERROR: token count/cache budgets must be integers' >&2; exit 2; }
done
(( MAX_NEW>=1 && MAX_NEW<=64 && NONEXPERT_MIB<=16384 && EXPERT_MIB<=16384 )) || { echo 'ERROR: token count or cache budget out of range' >&2; exit 2; }
HF_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
REPO="$HF_DIR/hub/models--thinkingmachines--Inkling-2-Mini"
[[ -f "$REPO/refs/main" ]] || { echo 'ERROR: Hugging Face snapshot ref missing' >&2; exit 2; }
REV="$(cat "$REPO/refs/main")"
[[ "$REV" =~ ^[0-9a-f]{40}$ ]] || { echo 'ERROR: invalid revision' >&2; exit 2; }
ARTIFACTS="${INKLING_ARTIFACTS:-$HOME/llmLab/inkling2-artifacts}"
OPTIONS=()
[[ "${INKLING_RAW_PROMPT:-0}" == 1 ]] && OPTIONS+=(--raw)
[[ "${INKLING_TOKENIZE_ONLY:-0}" == 1 ]] && OPTIONS+=(--tokenize-only)
[[ "${INKLING_PROFILE_EVERY_TOKEN:-0}" == 1 ]] && OPTIONS+=(--per-token-profile)
if [[ "${INKLING_TOKENIZE_ONLY:-0}" != 1 ]]; then
  for path in "$ARTIFACTS/remaining-full.ninfer" "$ARTIFACTS/experts-full.ninfer" "$ARTIFACTS/banks-full/manifest.json"; do
    [[ -f "$path" ]] || { echo "ERROR: missing $path" >&2; exit 2; }
  done
fi
IMAGE="${INKLING_HF_IMAGE:-ninfer:inkling2-hf-stage5p}"
docker run --rm --gpus '"device=0"' --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e PYTHONPATH=/workspace:/workspace/tools/inkling2 \
  -v "$ROOT:/workspace:ro" -v "$HF_DIR:/hf:ro" -v "$ARTIFACTS:/artifacts:ro" \
  -w /workspace --entrypoint python "$IMAGE" \
  tools/inkling2/ask_prefill_stage5u.py \
  --source "/hf/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/$REV" \
  --nonexpert /artifacts/remaining-full.ninfer --experts /artifacts/experts-full.ninfer \
  --banks /artifacts/banks-full --mode "$MODE" \
  --nonexpert-cache-mib "$NONEXPERT_MIB" --expert-cache-mib "$EXPERT_MIB" \
  --question "$QUESTION" --max-new-tokens "$MAX_NEW" --device cuda:0 "${OPTIONS[@]}"
