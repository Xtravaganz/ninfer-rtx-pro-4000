#!/usr/bin/env bash
set -euo pipefail
ROOT="$(pwd -P)"
QUESTION="${1:-Hi}"
MAX_NEW="${2:-1}"
[[ "$MAX_NEW" =~ ^[0-9]+$ ]] && ((MAX_NEW >= 1 && MAX_NEW <= 64)) || { echo 'Invalid max_new_tokens' >&2; exit 2; }
HF_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
REF="$HF_DIR/hub/models--thinkingmachines--Inkling-2-Mini/refs/main"
[[ -f "$REF" ]] || { echo "Missing $REF" >&2; exit 2; }
REV="$(cat "$REF")"
[[ "$REV" =~ ^[0-9a-f]{40}$ ]] || exit 2
ART="${INKLING_ARTIFACTS:-$HOME/llmLab/inkling2-artifacts}"
LIB="$ART/native5w/libinkling_moe_stage5w.so"
for p in "$LIB" "$ART/remaining-full.ninfer" "$ART/experts-full.ninfer" "$ART/banks-full/manifest.json"; do
  [[ -f "$p" ]] || { echo "Missing $p" >&2; exit 2; }
done
OPTS=()
[[ "${INKLING_PROFILE_SYNC:-0}" == 1 ]] && OPTS+=(--sync-per-operation)
[[ "${INKLING_NATIVE_VERIFY_FIRST:-0}" == 1 ]] && OPTS+=(--verify-first)
IMAGE="${INKLING_HF_IMAGE:-ninfer:inkling2-hf-stage5p}"
docker run --rm --gpus '"device=0"' --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e PYTHONPATH=/workspace:/workspace/tools/inkling2 \
  -v "$ROOT:/workspace:ro" -v "$HF_DIR:/hf:ro" -v "$ART:/artifacts:ro" \
  -w /workspace --entrypoint python "$IMAGE" tools/inkling2/run_profile_stage6b.py \
  --source "/hf/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/$REV" \
  --nonexpert /artifacts/remaining-full.ninfer --experts /artifacts/experts-full.ninfer \
  --banks /artifacts/banks-full --library /artifacts/native5w/libinkling_moe_stage5w.so \
  --nonexpert-cache-mib "${INKLING_NONEXPERT_CACHE_MIB:-4096}" \
  --quantized-cache-mib "${INKLING_QUANT_CACHE_MIB:-8192}" \
  --question "$QUESTION" --max-new-tokens "$MAX_NEW" --device cuda:0 "${OPTS[@]}"
