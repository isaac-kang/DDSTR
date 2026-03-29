#!/bin/bash
# Baseline: Standard training (no SLD/CCD)
#
# Usage: MODEL=svtrv2 bash scripts/run_baseline.sh [extra train_rec.py args]
#
# Required env vars:
#   MODEL : svtrv2 | igtr | parseq | mdiff4str
# Optional env vars:
#   CUDA_VISIBLE_DEVICES : GPU(s) to use (default: 0)
#   NPROC                : override number of processes

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-svtrv2}"
source "$SCRIPT_DIR/_model_config.sh"
source "$SCRIPT_DIR/_launch.sh"

train "${CONFIG_BASELINE}" "$@"
