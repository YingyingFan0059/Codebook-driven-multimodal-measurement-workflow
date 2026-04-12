#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_DIR}}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export QWEN3_MODEL_DIR="${QWEN3_MODEL_DIR:-${PROJECT_ROOT}/models/qwen/Qwen3-Omni-30B-A3B-Instruct}"
export QWEN3_TRAIN_DATA="${QWEN3_TRAIN_DATA:-${PROJECT_ROOT}/splits/split_v3/swift_train_2200.jsonl}"
export QWEN3_OUTPUT_DIR="${QWEN3_OUTPUT_DIR:-${PROJECT_ROOT}/runs/qwen3_omni_lora_aligned_v2}"

python "${PROJECT_ROOT}/src/qwen3/train_qwen3_omni_v3.py" "$@"
