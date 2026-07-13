#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON:-/home/rtx/miniconda3/envs/smsched/bin/python}"

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" scripts/ae/run_overhead_model_part4.py "$@"
