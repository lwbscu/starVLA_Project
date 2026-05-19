#!/usr/bin/env bash
# Parallel PI-State rollout launcher for AWAC handoff.
#
# The policy server is a single-process inference server. Giving one server a
# comma-separated CUDA_VISIBLE_DEVICES list does not provide tensor parallelism;
# it mainly uses the first visible GPU. To actually occupy 8 H200 GPUs, this
# wrapper launches 8 independent rollout workers:
#   - Qwen3.5-0.8B: 1 worker on GPU 0
#   - Qwen3.5-4B:   2 workers on GPUs 1,2
#   - Qwen3.5-9B:   5 workers on GPUs 3,4,5,6,7
#
# Each worker gets a disjoint EVAL_SEQUENCE_INDEX_OFFSET range for the same
# model, writes into its own output subdirectory, and the wrapper merges
# per-worker summaries at the end.
set -euo pipefail

cd "$(dirname "$0")/../../.."

export PROJECT_ROOT=${PROJECT_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project}
export CONDA_ROOT=${CONDA_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/26220216/miniconda3}
export STAR_VLA_PYTHON=${STAR_VLA_PYTHON:-"${CONDA_ROOT}/envs/starVLA_qwen35/bin/python"}
export CALVIN_PYTHON=${CALVIN_PYTHON:-"${CONDA_ROOT}/envs/calvin/bin/python"}

export EVAL_BATCH_ROOT=${EVAL_BATCH_ROOT:-"${PROJECT_ROOT}/logs/20260519_h200_server1_pi_state_30k_v1"}
export ROLLOUT_OUTPUT_ROOT=${ROLLOUT_OUTPUT_ROOT:-"${PROJECT_ROOT}/results/rollout"}
export ROLLOUT_RUN_NAME=${ROLLOUT_RUN_NAME:-"pi_state_parallel_$(date +%Y%m%d_%H%M%S)"}
export ROLLOUT_RUN_ROOT="${ROLLOUT_OUTPUT_ROOT%/}/${ROLLOUT_RUN_NAME}"
export ROLLOUT_NUM_SEQUENCES_PER_MODEL=${ROLLOUT_NUM_SEQUENCES_PER_MODEL:-200}

export UNNORM_KEY=${UNNORM_KEY:-franka}
export POLICY_SERVER_START_TIMEOUT=${POLICY_SERVER_START_TIMEOUT:-600}
export REQUIRE_MP4=${REQUIRE_MP4:-0}
export DEBUG_MP4=${DEBUG_MP4:-0}
export WRITE_ROLLOUT_VIDEOS=${WRITE_ROLLOUT_VIDEOS:-0}
export ROLLOUT_PARQUET_PYTHON=${ROLLOUT_PARQUET_PYTHON:-"${STAR_VLA_PYTHON}"}

export H200_CALVIN_EVAL_DATASET_PATH=${H200_CALVIN_EVAL_DATASET_PATH:-"${PROJECT_ROOT}/calvin/dataset/calvin_debug_dataset"}
export EVAL_SEQUENCES_PATH=${EVAL_SEQUENCES_PATH:-examples/calvin/eval_files/eval_sequences.json}

if [[ ! -x "${STAR_VLA_PYTHON}" ]]; then
  echo "STAR_VLA_PYTHON is not executable: ${STAR_VLA_PYTHON}" >&2
  exit 2
fi
if [[ ! -x "${CALVIN_PYTHON}" ]]; then
  echo "CALVIN_PYTHON is not executable: ${CALVIN_PYTHON}" >&2
  exit 2
fi
if [[ -e "${ROLLOUT_RUN_ROOT}" && "${ALLOW_EXISTING_ROLLOUT_RUN_ROOT:-0}" != "1" ]]; then
  echo "ROLLOUT_RUN_ROOT already exists: ${ROLLOUT_RUN_ROOT}" >&2
  echo "Use a new ROLLOUT_RUN_NAME, or set ALLOW_EXISTING_ROLLOUT_RUN_ROOT=1 if you intentionally want to append." >&2
  exit 2
fi

mkdir -p "${ROLLOUT_RUN_ROOT}/terminal"

"${STAR_VLA_PYTHON}" - <<'PY'
from __future__ import annotations

import json
import os
import re
from pathlib import Path

batch_root = Path(os.environ["EVAL_BATCH_ROOT"])
eval_sequences_path = Path(os.environ["EVAL_SEQUENCES_PATH"])
total_sequences = int(os.environ["ROLLOUT_NUM_SEQUENCES_PER_MODEL"])
if total_sequences < 5:
    raise SystemExit("ROLLOUT_NUM_SEQUENCES_PER_MODEL must be >= 5 to keep all 9B workers non-empty.")
if not eval_sequences_path.is_file():
    raise SystemExit(f"EVAL_SEQUENCES_PATH not found: {eval_sequences_path}")
with eval_sequences_path.open("r", encoding="utf-8") as f:
    sequences = json.load(f)
if total_sequences > len(sequences):
    raise SystemExit(
        f"ROLLOUT_NUM_SEQUENCES_PER_MODEL={total_sequences} exceeds available eval sequences={len(sequences)}"
    )

expected_min_steps = {
    "server1_pi_state/qwen35_0p8b": 30000,
    "server1_pi_state/qwen35_4b": 30000,
    "server1_pi_state/qwen35_9b": 25000,
}
pattern = re.compile(r"steps_(\d+)_pytorch_model\.pt$")
errors = []
for job, min_step in expected_min_steps.items():
    job_root = batch_root / job
    if not job_root.is_dir():
        errors.append(f"{job}: missing job directory {job_root}")
        continue
    candidates = []
    for ckpt in job_root.rglob("steps_*_pytorch_model.pt"):
        match = pattern.search(ckpt.name)
        if match:
            candidates.append((int(match.group(1)), ckpt))
    if not candidates:
        errors.append(f"{job}: no steps_*_pytorch_model.pt checkpoint")
        continue
    step, ckpt = max(candidates, key=lambda item: item[0])
    if step < min_step:
        errors.append(f"{job}: latest step {step} < required minimum {min_step}; latest={ckpt}")
if errors:
    print("Parallel PI-State rollout preflight failed:")
    for error in errors:
        print(f"  - {error}")
    raise SystemExit(2)
print(
    "Parallel PI-State rollout preflight OK: "
    f"sequences_per_model={total_sequences}, eval_sequences={len(sequences)}"
)
PY

manifest_path="${ROLLOUT_RUN_ROOT}/manifest.tsv"
summary_path="${ROLLOUT_RUN_ROOT}/summary.tsv"
: > "${manifest_path}"
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
  "model_tag" "job" "shard_index" "shard_count" "gpu" "port" "offset" "count" \
  >> "${manifest_path}"

pids=()
labels=()
logs=()

launch_worker() {
  local model_tag=$1
  local job=$2
  local gpu=$3
  local port=$4
  local shard_index=$5
  local shard_count=$6

  local total=${ROLLOUT_NUM_SEQUENCES_PER_MODEL}
  local base=$((total / shard_count))
  local rem=$((total % shard_count))
  local count=${base}
  local offset
  if (( shard_index < rem )); then
    count=$((count + 1))
    offset=$((shard_index * count))
  else
    offset=$((rem * (base + 1) + (shard_index - rem) * base))
  fi
  if (( count <= 0 )); then
    echo "Internal shard error: ${model_tag} shard=${shard_index}/${shard_count} count=${count}" >&2
    exit 2
  fi

  local worker_root="${ROLLOUT_RUN_ROOT}/${model_tag}/shard_${shard_index}"
  local log_file="${ROLLOUT_RUN_ROOT}/terminal/${model_tag}_shard_${shard_index}.log"
  mkdir -p "${worker_root}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${model_tag}" "${job}" "${shard_index}" "${shard_count}" "${gpu}" "${port}" "${offset}" "${count}" \
    >> "${manifest_path}"

  (
    set -euo pipefail
    export PROJECT_ROOT CONDA_ROOT STAR_VLA_PYTHON CALVIN_PYTHON
    export EVAL_BATCH_ROOT
    export EVAL_OUTPUT_ROOT="${worker_root}"
    export EXPECTED_JOBS="${job}"
    export EVAL_GPU="${gpu}"
    export EVAL_PORT="${port}"
    export NUM_SEQUENCES="${count}"
    export EVAL_SEQUENCE_INDEX_OFFSET="${offset}"
    export FAIL_FAST=1
    export REQUIRE_MP4 DEBUG_MP4
    export SEND_STATE_TO_POLICY=1
    export WRITE_ROLLOUT_LEROBOT=1
    export WRITE_ROLLOUT_VIDEOS
    export ROLLOUT_PARQUET_PYTHON
    export UNNORM_KEY
    export POLICY_SERVER_START_TIMEOUT
    export H200_CALVIN_EVAL_DATASET_PATH
    export EVAL_SEQUENCES_PATH
    bash examples/calvin/eval_files/eval_h200_qwen_mix_latest.sh
  ) > "${log_file}" 2>&1 &

  pids+=("$!")
  labels+=("${model_tag}/shard_${shard_index}")
  logs+=("${log_file}")
  echo "LAUNCHED ${model_tag}/shard_${shard_index}: gpu=${gpu} port=${port} offset=${offset} count=${count} log=${log_file}"
}

launch_worker qwen35_0p8b server1_pi_state/qwen35_0p8b 0 6200 0 1
launch_worker qwen35_4b server1_pi_state/qwen35_4b 1 6210 0 2
launch_worker qwen35_4b server1_pi_state/qwen35_4b 2 6211 1 2
launch_worker qwen35_9b server1_pi_state/qwen35_9b 3 6220 0 5
launch_worker qwen35_9b server1_pi_state/qwen35_9b 4 6221 1 5
launch_worker qwen35_9b server1_pi_state/qwen35_9b 5 6222 2 5
launch_worker qwen35_9b server1_pi_state/qwen35_9b 6 6223 3 5
launch_worker qwen35_9b server1_pi_state/qwen35_9b 7 6224 4 5

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "DONE ${labels[$index]} log=${logs[$index]}"
  else
    worker_status=$?
    status=1
    echo "FAILED ${labels[$index]} status=${worker_status} log=${logs[$index]}" >&2
  fi
done

: > "${summary_path}"
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
  "model_tag" "shard_index" "model_id" "step" "ckpt_path" "results_json" "mp4_count" "status" \
  >> "${summary_path}"
while IFS=$'\t' read -r model_tag job shard_index shard_count gpu port offset count; do
  if [[ "${model_tag}" == "model_tag" ]]; then
    continue
  fi
  worker_summary="${ROLLOUT_RUN_ROOT}/${model_tag}/shard_${shard_index}/summary.tsv"
  if [[ ! -f "${worker_summary}" ]]; then
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${model_tag}" "${shard_index}" "" "" "" "" "0" "MISSING_SUMMARY" \
      >> "${summary_path}"
    continue
  fi
  tail -n +2 "${worker_summary}" | while IFS=$'\t' read -r model_id step ckpt_path results_json mp4_count row_status; do
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${model_tag}" "${shard_index}" "${model_id}" "${step}" "${ckpt_path}" "${results_json}" "${mp4_count}" "${row_status}" \
      >> "${summary_path}"
  done
done < "${manifest_path}"

echo "ROLLOUT_RUN_ROOT=${ROLLOUT_RUN_ROOT}"
echo "manifest=${manifest_path}"
echo "summary=${summary_path}"
echo "terminal_logs=${ROLLOUT_RUN_ROOT}/terminal"

exit "${status}"
