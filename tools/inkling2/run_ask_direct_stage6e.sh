#!/usr/bin/env bash
set -euo pipefail
ROOT="$(pwd -P)"
QUESTION="${1:-}"
[[ -n "$QUESTION" ]] || { echo 'Usage: bash tools/inkling2/run_ask_direct_stage6e.sh "Question" [max_new_tokens]' >&2; exit 2; }
MAX_NEW="${2:-2}"
[[ "$MAX_NEW" =~ ^[0-9]+$ ]] && (( MAX_NEW>=1 && MAX_NEW<=64 )) || { echo 'Invalid max_new_tokens' >&2; exit 2; }
HF_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
REF="$HF_DIR/hub/models--thinkingmachines--Inkling-2-Mini/refs/main"
[[ -f "$REF" ]] || { echo 'Missing HF main revision' >&2; exit 2; }
REV="$(cat "$REF")"
[[ "$REV" =~ ^[0-9a-f]{40}$ ]] || { echo 'Invalid HF revision' >&2; exit 2; }
ART="${INKLING_ARTIFACTS:-$HOME/llmLab/inkling2-artifacts}"
LIB="$ART/native6ef/libinkling_moe_stage6e.so"
[[ -f "$LIB" ]] || { echo "Missing Stage6E CUDA library: $LIB. Build with build_native_stage6ef.sh" >&2; exit 2; }
for f in "$ART/remaining-full.ninfer" "$ART/experts-full.ninfer" "$ART/banks-full/manifest.json"; do
  [[ -f "$f" ]] || { echo "Missing artifact $f" >&2; exit 2; }
done
OPTIONS=()
[[ "${INKLING_PROFILE_EVERY_TOKEN:-0}" == 1 ]] && OPTIONS+=(--per-token-profile)
[[ "${INKLING_NATIVE_VERIFY_FIRST:-1}" == 1 ]] && OPTIONS+=(--verify-first)
IMAGE="${INKLING_HF_IMAGE:-ninfer:inkling2-hf-stage5p}"
docker run --rm --gpus '"device=0"' --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e PYTHONPATH=/workspace:/workspace/tools/inkling2 \
  -v "$ROOT:/workspace:ro" -v "$HF_DIR:/hf:ro" -v "$ART:/artifacts:ro" \
  -w /workspace --entrypoint python "$IMAGE" \
  tools/inkling2/ask_direct_stage6e.py \
  --source "/hf/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/$REV" \
  --nonexpert /artifacts/remaining-full.ninfer --experts /artifacts/experts-full.ninfer \
  --banks /artifacts/banks-full --library /artifacts/native6ef/libinkling_moe_stage6e.so \
  --nonexpert-cache-mib "${INKLING_NONEXPERT_CACHE_MIB:-4096}" \
  --quantized-cache-mib "${INKLING_QUANT_CACHE_MIB:-8192}" \
  --question "$QUESTION" --max-new-tokens "$MAX_NEW" --device cuda:0 "${OPTIONS[@]}"
