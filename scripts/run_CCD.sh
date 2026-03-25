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
MIN_RATE="${MIN_RATE:-0.001}"

CCD_OUTPUT="${DDSTR_DATA_ROOT}/CCD/${MODEL}"

if run_step 1; then
    echo "=== Step 1: Build confusion matrix & generate decomposed LMDB (${MODEL}) ==="
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
    echo "=== Step 2: Train with extended charset (${MODEL} CCD) ==="
    _T0=$SECONDS
    train "configs/rec/ddstr/${CONFIG_PREFIX}_ccd.yml" \
        "${PASS_ARGS[@]}"
    elapsed 2
fi
