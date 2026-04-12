#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_DIR}}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export VIDEO_BASE_DIR="${VIDEO_BASE_DIR:-${PROJECT_ROOT}/videos/douyin/upload_pack}"
export MINICPM_MODEL_DIR="${MINICPM_MODEL_DIR:-${PROJECT_ROOT}/models/openbmb/MiniCPM-o-2_6}"
export MINICPM_LORA_PATH="${MINICPM_LORA_PATH:-${PROJECT_ROOT}/runs/minicpm_o_lora_a100/v3-20260308-015640/checkpoint-1875}"
export MINICPM_TEST_CSV="${MINICPM_TEST_CSV:-${PROJECT_ROOT}/splits/split_v3/test_main.csv}"
export MINICPM_EVAL_DIR="${MINICPM_EVAL_DIR:-${PROJECT_ROOT}/outputs/scaling_eval}"

python "${PROJECT_ROOT}/src/minicpm/eval_minicpm_v3.py" "$@"
