#!/bin/bash
# Mode CLD: Confidence-based Label Denoising
# Step 1: Extract error info  -- active env (ddstr)
# Step 2: LLM judge           -- vllm env (conda run)
# Step 3: Generate PL LMDB   -- active env (ddstr)
# Step 4: Train               -- active env (ddstr)
#
# Usage: MODEL=svtrv2 CHECKPOINT=<ckpt> bash scripts/run_CLD.sh [--steps=CLD,4]
#   --steps=4       (default) training only
#   --steps=CLD     steps 1-3 (data prep)
#   --steps=CLD,4   everything
#   --steps=all     everything
#
# Required env vars:
#   MODEL       : svtrv2 | igtr | parseq | mdiff4str
#   MODEL_ID    : experiment name tag for output dirs (e.g. mdiff4str_B_CLD)
#   CHECKPOINT  : path to trained baseline checkpoint
# Optional env vars:
#   LLM_ENV         : conda env with vllm (default: vllm)
#   LLM_MODEL_IDX   : LLM model index in llm_judge.py (default: 4 = Qwen3-8B)
#   DATA_ROOT       : original LMDB root (default: ~/data/STR/openocr)
#   DDSTR_DATA_ROOT : output root for DDSTR data (default: ~/data/STR/ddstr)
#   ALPHA           : STR/LLM fusion weight (default: 0.5)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

declare -A STEP_ALIASES=( [CLD]="123" )
DEFAULT_STEPS="4"
source "$SCRIPT_DIR/_parse_steps.sh"

MODEL="${MODEL:-svtrv2}"
source "$SCRIPT_DIR/_model_config.sh"
source "$SCRIPT_DIR/_launch.sh"

LLM_ENV="${LLM_ENV:-vllm}"
LLM_MODEL_IDX="${LLM_MODEL_IDX:-4}"  # Qwen3-8B
ALPHA="${ALPHA:-0.5}"
DATA_ROOT="$(eval echo "${DATA_ROOT:-~/data/STR/openocr}")"
DDSTR_DATA_ROOT="$(eval echo "${DDSTR_DATA_ROOT:-~/data/STR/ddstr}")"
CHECKPOINT="${CHECKPOINT:?ERROR: CHECKPOINT is required}"
MODEL_ID="${MODEL_ID:?ERROR: MODEL_ID is required (e.g. mdiff4str_B_CLD)}"

# Derive LLM model name for output path (e.g., index 4 → Qwen3-8B)
LLM_NAMES=( "Qwen3-0.6B" "Qwen3-1.7B" "Qwen3-4B" "Qwen3-4B-Instruct-2507" "Qwen3-8B" "Qwen3-14B" "Qwen3-30B-A3B" "Llama-3.2-1B" "Llama-3.2-3B" "Llama-3.1-8B" "gemma-3-1b" "gemma-3-4b" "gemma-3-12b" )
LLM_NAME="${LLM_NAMES[$LLM_MODEL_IDX]:-llm${LLM_MODEL_IDX}}"

ERROR_DIR="${DDSTR_DATA_ROOT}/error_info/${MODEL_ID}"
CLD_OUTPUT="${DDSTR_DATA_ROOT}/CLD/${MODEL_ID}__${LLM_NAME}_a${ALPHA}"

mkdir -p "${ERROR_DIR}"

_LMDB_BASE="${CLD_OUTPUT}/Union14M-L-LMDB-Filtered"
_DATA_DIRS="['${_LMDB_BASE}/filter_train_challenging', '${_LMDB_BASE}/filter_train_hard', '${_LMDB_BASE}/filter_train_medium', '${_LMDB_BASE}/filter_train_normal', '${_LMDB_BASE}/filter_train_easy']"

if run_step 1; then
    echo "=== Step 1: Extract error info (${MODEL_ID}) ==="
    _T0=$SECONDS
    python tools/denoise/extract_error_info.py \
        -c "${CONFIG_BASELINE}" \
        --checkpoint "${CHECKPOINT}" \
        --data_root "${DATA_ROOT}" \
        --output "${ERROR_DIR}/error_info.tsv"
    elapsed 1
fi

if run_step 2; then
    echo ""
    echo "=== Step 2: LLM judge (${LLM_NAME}) ==="
    _T0=$SECONDS
    # vllm env used separately due to numpy compatibility constraints
    conda run --no-capture-output -n "${LLM_ENV}" \
        python tools/denoise/llm_judge.py \
        "${ERROR_DIR}/error_info.tsv" "${LLM_MODEL_IDX}" \
        --alpha "${ALPHA}" \
        --output "${ERROR_DIR}/judge_results.tsv"
    elapsed 2
fi

if run_step 3; then
    echo ""
    echo "=== Step 3: Generate denoised LMDB ==="
    _T0=$SECONDS
    python tools/denoise/generate_pl.py \
        --error_info "${ERROR_DIR}/error_info.tsv" \
        --judge "${ERROR_DIR}/judge_results.tsv" \
        --data_root "${DATA_ROOT}" \
        --output_root "${CLD_OUTPUT}"
    elapsed 3
fi

if run_step 4; then
    echo ""
    echo "=== Step 4: Train on denoised data (${MODEL_ID} CLD) ==="
    _T0=$SECONDS
    train "configs/rec/ddstr/${CONFIG_PREFIX}_cld.yml" \
        "${PASS_ARGS[@]}" \
        -o "Train.dataset.data_dir_list=${_DATA_DIRS}"
    elapsed 4
fi
