#!/usr/bin/env bash
set -euo pipefail
ROOT="$(pwd -P)"
QUESTION="${1:-}"
[[ -n "$QUESTION" ]] || { echo 'Usage: bash tools/inkling2/run_ask_native_stage6jkl.sh "Question" [max_new_tokens]' >&2; exit 2; }
MAX_NEW="${2:-10}"
[[ "$MAX_NEW" =~ ^[0-9]+$ ]] && ((MAX_NEW>=1 && MAX_NEW<=64)) || { echo 'Invalid max_new_tokens' >&2;exit 2; }
HF_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
REF="$HF_DIR/hub/models--thinkingmachines--Inkling-2-Mini/refs/main"
[[ -f "$REF" ]] || { echo 'Missing Inkling HF main revision' >&2; exit 2; }
REV="$(cat "$REF")"
[[ "$REV" =~ ^[0-9a-f]{40}$ ]] || { echo 'Invalid HF revision' >&2; exit 2; }
ART="${INKLING_ARTIFACTS:-$HOME/llmLab/inkling2-artifacts}"
LIB="$ART/native6jkl/libinkling_native_stage6jkl.so"
[[ -f "$LIB" ]] || { echo "Missing $LIB: run build_native_stage6jkl.sh first" >&2;exit 2; }
for f in "$ART/remaining-full.ninfer" "$ART/experts-full.ninfer" "$ART/banks-full/manifest.json";do
 [[ -f "$f" ]] || { echo "Missing $f" >&2;exit 2; }
done
OPTIONS=()
[[ "${INKLING_NATIVE_VERIFY_FIRST:-1}" == 1 ]] && OPTIONS+=(--verify-first)
[[ "${INKLING_CUDA_GRAPH:-0}" == 1 ]] && OPTIONS+=(--graph)
[[ "${INKLING_NATIVE_CONV:-0}" == 1 ]] && OPTIONS+=(--native-conv)
[[ "${INKLING_PROFILE_EVERY_TOKEN:-0}" == 1 ]] && OPTIONS+=(--per-token-profile)
IMAGE="${INKLING_HF_IMAGE:-ninfer:inkling2-hf-stage5p}"
echo "STAGE6JKL: host_GPU=0 CUDA_DEVICE=cuda:0 graph=${INKLING_CUDA_GRAPH:-0} conv=${INKLING_NATIVE_CONV:-0}"
docker run --rm --gpus '"device=0"' --user "$(id -u):$(id -g)" \
 -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
 -e PYTHONPATH=/workspace:/workspace/tools/inkling2 \
 -v "$ROOT:/workspace:ro" -v "$HF_DIR:/hf:ro" -v "$ART:/artifacts:ro" \
 -w /workspace --entrypoint python "$IMAGE" \
 tools/inkling2/ask_native_stage6jkl.py \
 --source "/hf/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/$REV" \
 --nonexpert /artifacts/remaining-full.ninfer --experts /artifacts/experts-full.ninfer \
 --banks /artifacts/banks-full --library /artifacts/native6jkl/libinkling_native_stage6jkl.so \
 --nonexpert-cache-mib "${INKLING_NONEXPERT_CACHE_MIB:-4096}" \
 --question "$QUESTION" --max-new-tokens "$MAX_NEW" --device cuda:0 "${OPTIONS[@]}"
