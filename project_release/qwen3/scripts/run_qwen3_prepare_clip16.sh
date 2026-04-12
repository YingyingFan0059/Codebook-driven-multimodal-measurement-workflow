#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_DIR}}"
export VIDEO_BASE_DIR="${VIDEO_BASE_DIR:-${PROJECT_ROOT}/videos/douyin/upload_pack}"
export AUDIO_BASE_DIR="${AUDIO_BASE_DIR:-${PROJECT_ROOT}/audios}"
export QWEN3_TEST_CSV="${QWEN3_TEST_CSV:-${PROJECT_ROOT}/splits/split_v3/test_main.csv}"
export QWEN3_EVAL_DIR="${QWEN3_EVAL_DIR:-${PROJECT_ROOT}/outputs/qwen3_omni_eval_clip16_wav}"

python "${PROJECT_ROOT}/src/qwen3/prepare_clip16_for_eval.py" "$@"
