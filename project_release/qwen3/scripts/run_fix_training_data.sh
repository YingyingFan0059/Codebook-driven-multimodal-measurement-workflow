#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_DIR}}"
export QWEN3_SWIFT_TRAIN_INPUT="${QWEN3_SWIFT_TRAIN_INPUT:-${PROJECT_ROOT}/splits/split_v3/swift_train.jsonl}"
export QWEN3_SWIFT_TRAIN_FIXED="${QWEN3_SWIFT_TRAIN_FIXED:-${PROJECT_ROOT}/splits/split_v3/swift_train_v2.jsonl}"
export QWEN3_SWIFT_TRAIN_BACKUP="${QWEN3_SWIFT_TRAIN_BACKUP:-${PROJECT_ROOT}/splits/split_v3/swift_train_v1_backup.jsonl}"

python "${PROJECT_ROOT}/src/shared/fix_training_data.py" "$@"
