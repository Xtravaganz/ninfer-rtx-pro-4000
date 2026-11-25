#!/usr/bin/env bash
# Stage5P. Modes: student|hf|compare. Never edits NInfer model weights.
set -euo pipefail
ROOT="$(pwd -P)"
[[ -f "$ROOT/tools/inkling2/parity_stage5p.py" ]] || { echo 'ERROR: run from Inkling worktree root' >&2;exit 2; }
MODE="${1:-student}"; TOKEN="${2:-1000}"
[[ "$TOKEN" =~ ^[0-9]+$ ]] || { echo 'ERROR: token must be unsigned decimal' >&2;exit 2; }
ARTIFACTS="${INKLING_ARTIFACTS:-$HOME/llmLab/inkling2-artifacts}"
HF_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
REPO="$HF_DIR/hub/models--thinkingmachines--Inkling-2-Mini"
[[ -f "$REPO/refs/main" ]] || { echo 'ERROR: HF snapshot not found' >&2;exit 2; }
REV="$(cat "$REPO/refs/main")"
[[ "$REV" =~ ^[a-f0-9]{40}$ ]] || { echo 'ERROR: unexpected HF revision' >&2;exit 2; }
REPORTS="$ARTIFACTS/parity5p"
mkdir -p "$REPORTS"
IMAGE="${INKLING_TORCH_IMAGE:-pytorch/pytorch:2.14.0-cuda13.0-cudnn9-runtime}"
COMMON=(--rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$ROOT:/workspace:ro" -v "$REPORTS:/reports" -w /workspace --entrypoint python)
case "$MODE" in
  student)
    for x in "$ARTIFACTS/remaining-full.ninfer" "$ARTIFACTS/experts-full.ninfer" \
      "$ARTIFACTS/banks-full/manifest.json"; do
      [[ -f "$x" ]] || { echo "ERROR: missing $x" >&2;exit 2; }
    done
    echo "Capturing NInfer-v3 FP32 student, revision=$REV, token=$TOKEN"
    docker run "${COMMON[@]}" --gpus '"device=0"' \
      -v "$ARTIFACTS:/artifacts:ro" -v "$HF_DIR:/hf:ro" "$IMAGE" \
      tools/inkling2/parity_stage5p.py student \
      --source "/hf/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/$REV" \
      --nonexpert /artifacts/remaining-full.ninfer --experts /artifacts/experts-full.ninfer \
      --banks /artifacts/banks-full --device cuda:0 --token "$TOKEN" \
      --output "/reports/student-$REV-$TOKEN.npz"
    ;;
  hf)
    # NOT a fallback implementation: requires official Transformers to load the whole model.
    HF_IMAGE="${INKLING_HF_IMAGE:-ninfer:inkling2-hf-stage5p}"
    echo 'Official BF16 HF model may expand to >48 GiB. Ensure sufficient GPU+host RAM.'
    echo "Capturing official HF, revision=$REV, token=$TOKEN, image=$HF_IMAGE"
    docker run "${COMMON[@]}" --gpus all -v "$HF_DIR:/hf:ro" "$HF_IMAGE" \
      tools/inkling2/parity_stage5p.py hf \
      --source "/hf/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/$REV" \
      --device-map auto --token "$TOKEN" --output "/reports/hf-$REV-$TOKEN.npz"
    ;;
  compare)
    echo "Comparing same-revision independent NInfer-v3 and HF trace for token=$TOKEN"
    docker run "${COMMON[@]}" "$IMAGE" tools/inkling2/parity_stage5p.py compare \
      --student "/reports/student-$REV-$TOKEN.npz" \
      --teacher "/reports/hf-$REV-$TOKEN.npz" \
      --output "/reports/compare-$REV-$TOKEN.json"
    ;;
  *) echo 'ERROR: mode must be student|hf|compare' >&2;exit 2;;
esac
