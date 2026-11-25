#!/usr/bin/env bash
set -euo pipefail
ROOT="$(pwd -P)"
QUESTION="${1:-}"
MAX_NEW="${2:-10}"
[[ -n "$QUESTION" && "$MAX_NEW" =~ ^[0-9]+$ ]] && ((MAX_NEW>=1 && MAX_NEW<=64)) || { echo 'Usage: bash tools/inkling2/run_ask_native_stage7cd.sh "Question" [1..64]' >&2;exit 2; }
[[ "${INKLING_CUDA_GRAPH:-0}" == 0 ]] || { echo 'Stage7CD is eager only' >&2;exit 2; }
HF_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
REF="$HF_DIR/hub/models--thinkingmachines--Inkling-2-Mini/refs/main"
[[ -f "$REF" ]] || { echo 'Missing Inkling HF main revision' >&2;exit 2; }
REV="$(cat "$REF")"; [[ "$REV" =~ ^[0-9a-f]{40}$ ]] || exit 2
ART="${INKLING_ARTIFACTS:-$HOME/llmLab/inkling2-artifacts}"
for f in remaining-full.ninfer experts-full.ninfer banks-full/manifest.json native6ghi/libinkling_native_stage6ghi.so native6jkl/libinkling_native_stage6jkl.so native7ab/libinkling_native_stage7ab.so native7cd/libinkling_stage7cd.so; do
 [[ -f "$ART/$f" ]] || { echo "Missing $ART/$f" >&2;exit 2; }
done
OPTIONS=()
[[ "${INKLING_STAGE7CD_VERIFY:-0}" == 1 ]] && OPTIONS+=(--verify)
IMAGE="${INKLING_HF_IMAGE:-ninfer:inkling2-hf-stage5p}"
docker run --rm --gpus '"device=0"' --user "$(id -u):$(id -g)" \
 -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
 -e PYTHONPATH=/workspace:/workspace/tools/inkling2 \
 -v "$ROOT:/workspace:ro" -v "$HF_DIR:/hf:ro" -v "$ART:/artifacts:ro" \
 -w /workspace --entrypoint python "$IMAGE" tools/inkling2/ask_native_stage7cd.py \
 --source "/hf/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/$REV" \
 --nonexpert /artifacts/remaining-full.ninfer --experts /artifacts/experts-full.ninfer \
 --banks /artifacts/banks-full --library /artifacts/native6jkl/libinkling_native_stage6jkl.so \
 --fast-library /artifacts/native7ab/libinkling_native_stage7ab.so --executor /artifacts/native7cd/libinkling_stage7cd.so \
 --nonexpert-mib "${INKLING_NONEXPERT_CACHE_MIB:-4096}" --capacity "${INKLING_STAGE7CD_CAPACITY:-128}" \
 --mode "${INKLING_STAGE7CD_MODE:-cd}" --question "$QUESTION" --max-new-tokens "$MAX_NEW" "${OPTIONS[@]}"
