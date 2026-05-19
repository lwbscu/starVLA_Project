#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project}
CONDA_ROOT=${CONDA_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/26220216/miniconda3}
STAR_VLA_PYTHON=${STAR_VLA_PYTHON:-"${CONDA_ROOT}/envs/starVLA_qwen35/bin/python"}

DATASET_SCAN_ROOT=${DATASET_SCAN_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/three/dataset}
CALVIN_DD_ROOT=${CALVIN_DD_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_d_d}
CALVIN_DEBUG_ROOT=${CALVIN_DEBUG_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/four/calvin/dataset/calvin_debug_dataset}

DATASET_SCAN_OUT=${DATASET_SCAN_OUT:-"${PROJECT_ROOT}/logs/dataset_scan_$(date +%Y%m%d_%H%M%S)"}

MAX_DEPTH=${MAX_DEPTH:-6}
CHILD_MAX_DEPTH=${CHILD_MAX_DEPTH:-4}
MAX_COUNT_FILES=${MAX_COUNT_FILES:-200000}
MAX_COUNT_FILES_PER_CHILD=${MAX_COUNT_FILES_PER_CHILD:-50000}
MAX_LEROBOT_FILES=${MAX_LEROBOT_FILES:-200000}
CHILDREN_LIMIT=${CHILDREN_LIMIT:-120}
SAMPLE_PARQUET=${SAMPLE_PARQUET:-1}

cd "${PROJECT_ROOT}"
mkdir -p "${DATASET_SCAN_OUT}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "STAR_VLA_PYTHON=${STAR_VLA_PYTHON}"
echo "DATASET_SCAN_ROOT=${DATASET_SCAN_ROOT}"
echo "CALVIN_DD_ROOT=${CALVIN_DD_ROOT}"
echo "CALVIN_DEBUG_ROOT=${CALVIN_DEBUG_ROOT}"
echo "DATASET_SCAN_OUT=${DATASET_SCAN_OUT}"

args=(
  examples/calvin/scripts/inspect_dataset_layout.py
  "${DATASET_SCAN_ROOT}"
  "${CALVIN_DD_ROOT}"
  "${CALVIN_DEBUG_ROOT}"
  --progress
  --max-depth "${MAX_DEPTH}"
  --child-max-depth "${CHILD_MAX_DEPTH}"
  --max-count-files "${MAX_COUNT_FILES}"
  --max-count-files-per-child "${MAX_COUNT_FILES_PER_CHILD}"
  --max-lerobot-files "${MAX_LEROBOT_FILES}"
  --children-limit "${CHILDREN_LIMIT}"
  --json-out "${DATASET_SCAN_OUT}/dataset_layout.json"
)

if [[ "${SAMPLE_PARQUET}" == "1" ]]; then
  args+=(--sample-parquet)
fi

"${STAR_VLA_PYTHON}" "${args[@]}" > "${DATASET_SCAN_OUT}/dataset_layout.md"

echo "WROTE ${DATASET_SCAN_OUT}/dataset_layout.md"
echo "WROTE ${DATASET_SCAN_OUT}/dataset_layout.json"
sed -n '1,220p' "${DATASET_SCAN_OUT}/dataset_layout.md"
