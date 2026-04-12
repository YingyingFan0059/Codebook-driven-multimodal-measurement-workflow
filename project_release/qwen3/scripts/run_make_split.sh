#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_DIR}}"
export QWEN3_CODEBOOK_FILE="${QWEN3_CODEBOOK_FILE:-${PROJECT_ROOT}/codebook.xlsx}"
export QWEN3_SPLIT_DIR="${QWEN3_SPLIT_DIR:-${PROJECT_ROOT}/splits/split_v3}"

python "${PROJECT_ROOT}/src/shared/make_split_v3.py" "$@"
