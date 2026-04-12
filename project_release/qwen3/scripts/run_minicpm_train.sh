#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_DIR}}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MINICPM_MODEL_DIR="${MINICPM_MODEL_DIR:-${PROJECT_ROOT}/models/openbmb/MiniCPM-o-2_6}"
export MINICPM_TRAIN_DATA="${MINICPM_TRAIN_DATA:-${PROJECT_ROOT}/splits/split_v3/swift_train.jsonl}"
export MINICPM_OUTPUT_DIR="${MINICPM_OUTPUT_DIR:-${PROJECT_ROOT}/runs/minicpm_o_lora_a100}"

python "${PROJECT_ROOT}/src/minicpm/train_minicpm_v3.py" "$@"
