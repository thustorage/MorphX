#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-/home/rtx/miniconda3/envs/smsched/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi
export PATH="$(dirname -- "${PYTHON_BIN}"):${PATH}"

exec "${PYTHON_BIN}" "${ROOT_DIR}/scripts/ae/run_all.py" "$@"
