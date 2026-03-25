#!/bin/bash
# Mode CCD: Confusion-aware Class Decomposition
# Step 1: Build confusion matrix + create decomposed LMDB
# Step 2: Train with extended charset
#
# Usage: MODEL=svtrv2 CHECKPOINT=<ckpt> bash scripts/run_CCD.sh [--steps=CCD,2]
#   --steps=2       (default) training only
#   --steps=CCD     step 1 (data prep)
#   --steps=CCD,2   everything
#   --steps=all     everything
#
# Required env vars:
#   MODEL       : svtrv2 | igtr | parseq | mdiff4str
#   MODEL_ID    : experiment name tag for output dirs (e.g. mdiff4str_B_BLC)
#   CHECKPOINT  : path to trained baseline checkpoint
# Optional env vars:
#   DATA_ROOT       : original LMDB root (default: ~/data/STR/openocr)
#   DDSTR_DATA_ROOT : output root for DDSTR data (default: ~/data/STR/ddstr)
#   MIN_RATE        : confusion min rate threshold (default: 0.001)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

declare -A STEP_ALIASES=( [CCD]="1" )
DEFAULT_STEPS="2"
source "$SCRIPT_DIR/_parse_steps.sh"

MODEL="${MODEL:-svtrv2}"
source "$SCRIPT_DIR/_model_config.sh"
source "$SCRIPT_DIR/_launch.sh"

DATA_ROOT="$(eval echo "${DATA_ROOT:-~/data/STR/openocr}")"
DDSTR_DATA_ROOT="$(eval echo "${DDSTR_DATA_ROOT:-~/data/STR/ddstr}")"
CHECKPOINT="${CHECKPOINT:?ERROR: CHECKPOINT is required}"
MODEL_ID="${MODEL_ID:?ERROR: MODEL_ID is required (e.g. mdiff4str_B_BLC)}"
MIN_RATE="${MIN_RATE:-0.001}"

CCD_OUTPUT="${DDSTR_DATA_ROOT}/CCD/${MODEL_ID}"
_LMDB_BASE="${CCD_OUTPUT}/Union14M-L-LMDB-Filtered"
_DATA_DIRS="['${_LMDB_BASE}/filter_train_challenging', '${_LMDB_BASE}/filter_train_hard', '${_LMDB_BASE}/filter_train_medium', '${_LMDB_BASE}/filter_train_normal', '${_LMDB_BASE}/filter_train_easy']"

if run_step 1; then
    echo "=== Step 1: Build confusion matrix & generate decomposed LMDB (${MODEL_ID}) ==="
    _T0=$SECONDS
    python tools/confusion_and_pl.py \
        -c "${CONFIG_BASELINE}" \
        --checkpoint "${CHECKPOINT}" \
        --data_root "${DATA_ROOT}" \
        --output_dir "${CCD_OUTPUT}" \
        --min_rate "${MIN_RATE}" \
        --pl_output_root "${CCD_OUTPUT}"
    elapsed 1
fi

if run_step 2; then
    echo ""
    echo "=== Step 2: Train with extended charset (${MODEL_ID} CCD) ==="
    _T0=$SECONDS
    train "configs/rec/ddstr/${CONFIG_PREFIX}_ccd.yml" \
        "${PASS_ARGS[@]}" \
        -o "Global.unicode_mapping=${CCD_OUTPUT}/unicode_mapping.json" \
           "Train.dataset.data_dir_list=${_DATA_DIRS}"
    elapsed 2
fi
