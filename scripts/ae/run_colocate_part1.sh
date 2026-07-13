#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON:-/home/rtx/miniconda3/envs/smsched/bin/python}"

exec "${PYTHON_BIN}" "${ROOT_DIR}/scripts/ae/run_colocate_part1.py" "$@"
