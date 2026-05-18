#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

PYTHON_BIN=${PYTHON_BIN:-python}
LOG_ROOT=${LOG_ROOT:-logs/h200_fastexplore}
OUTPUT_DIR=${OUTPUT_DIR:-${LOG_ROOT%/}/summary}

"${PYTHON_BIN}" examples/calvin/train_files/compare_h200_fastexplore.py \
  --log-root "${LOG_ROOT}" \
  --output-dir "${OUTPUT_DIR}"
