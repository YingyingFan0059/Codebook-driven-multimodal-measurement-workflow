#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_DIR}}"
export VIDEO_BASE_DIR="${VIDEO_BASE_DIR:-${PROJECT_ROOT}/videos/douyin/upload_pack}"
export QWEN3_TRAIN_CSV="${QWEN3_TRAIN_CSV:-${PROJECT_ROOT}/splits/split_v3/train_2200.csv}"
export QWEN3_TRAIN_DATA="${QWEN3_TRAIN_DATA:-${PROJECT_ROOT}/splits/split_v3/swift_train_2200.jsonl}"

python "${PROJECT_ROOT}/src/shared/gen_swift_train_2200.py" "$@"
