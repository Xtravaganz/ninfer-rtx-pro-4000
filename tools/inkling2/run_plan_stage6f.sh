#!/usr/bin/env bash
set -euo pipefail
ART="${INKLING_ARTIFACTS:-$HOME/llmLab/inkling2-artifacts}"
LIB="$ART/native6ef/libinkling_batch_stage6f.so"
[[ -f "$LIB" ]] || { echo 'Build Stage6E/F first: tools/inkling2/build_native_stage6ef.sh' >&2; exit 2; }
python3 tools/inkling2/batch_plan_probe_stage6f.py --library "$LIB" --tokens "${1:-16}"
