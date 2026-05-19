#!/usr/bin/env bash
# Parallel PI-State rollout launcher for AWAC handoff.
#
# The policy server is a single-process inference server. Giving one server a
# comma-separated CUDA_VISIBLE_DEVICES list does not provide tensor parallelism;
# it mainly uses the first visible GPU. To actually occupy 8 H200 GPUs, this
# wrapper launches 8 independent Qwen3.5-4B rollout workers by default,
# because the current PI-State rollout inspection shows 4B is the useful
# success-producing model for AWAC data collection.
# Override ROLLOUT_WORKER_LAYOUT to bias GPUs toward the model that actually
# produces useful rollouts. Each non-comment line is:
#   model_tag job gpu port
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

if [[ -z "${ROLLOUT_WORKER_LAYOUT:-}" ]]; then
ROLLOUT_WORKER_LAYOUT=$(cat <<'EOF'
qwen35_4b server1_pi_state/qwen35_4b 0 6210
qwen35_4b server1_pi_state/qwen35_4b 1 6211
qwen35_4b server1_pi_state/qwen35_4b 2 6212
qwen35_4b server1_pi_state/qwen35_4b 3 6213
qwen35_4b server1_pi_state/qwen35_4b 4 6214
qwen35_4b server1_pi_state/qwen35_4b 5 6215
qwen35_4b server1_pi_state/qwen35_4b 6 6216
qwen35_4b server1_pi_state/qwen35_4b 7 6217
EOF
)
fi
export ROLLOUT_WORKER_LAYOUT

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
if total_sequences < 1:
    raise SystemExit("ROLLOUT_NUM_SEQUENCES_PER_MODEL must be >= 1.")
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
layout_jobs: set[str] = set()
for raw_line in os.environ["ROLLOUT_WORKER_LAYOUT"].splitlines():
    line = raw_line.split("#", 1)[0].strip()
    if not line:
        continue
    parts = line.split()
    if len(parts) != 4:
        raise SystemExit(f"Bad ROLLOUT_WORKER_LAYOUT line, expected 4 fields: {raw_line!r}")
    _model_tag, job, _gpu, _port = parts
    if job not in expected_min_steps:
        raise SystemExit(f"Unknown rollout job in ROLLOUT_WORKER_LAYOUT: {job!r}")
    layout_jobs.add(job)
if not layout_jobs:
    raise SystemExit("ROLLOUT_WORKER_LAYOUT contains no workers.")

pattern = re.compile(r"steps_(\d+)_pytorch_model\.pt$")
errors = []
for job in sorted(layout_jobs):
    min_step = expected_min_steps[job]
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
    f"sequences_per_model={total_sequences}, jobs={len(layout_jobs)}, eval_sequences={len(sequences)}"
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
layout_rows=()
declare -A shard_counts=()
declare -A next_shard=()
declare -A used_gpus=()
declare -A used_ports=()

while IFS= read -r raw_line; do
  line="${raw_line%%#*}"
  if [[ "${line}" =~ ^[[:space:]]*$ ]]; then
    continue
  fi
  read -r model_tag job gpu port extra <<< "${line}"
  if [[ -z "${model_tag:-}" || -z "${job:-}" || -z "${gpu:-}" || -z "${port:-}" || -n "${extra:-}" ]]; then
    echo "Bad ROLLOUT_WORKER_LAYOUT line, expected: model_tag job gpu port" >&2
    echo "line=${raw_line}" >&2
    exit 2
  fi
  if [[ ! "${gpu}" =~ ^[0-9]+$ || ! "${port}" =~ ^[0-9]+$ ]]; then
    echo "Bad ROLLOUT_WORKER_LAYOUT gpu/port, both must be integers: ${raw_line}" >&2
    exit 2
  fi
  if [[ -n "${used_gpus[$gpu]:-}" ]]; then
    echo "Duplicate GPU in ROLLOUT_WORKER_LAYOUT: gpu=${gpu}" >&2
    exit 2
  fi
  if [[ -n "${used_ports[$port]:-}" ]]; then
    echo "Duplicate port in ROLLOUT_WORKER_LAYOUT: port=${port}" >&2
    exit 2
  fi
  used_gpus[$gpu]=1
  used_ports[$port]=1
  key="${model_tag}|${job}"
  shard_counts[$key]=$(( ${shard_counts[$key]:-0} + 1 ))
  layout_rows+=("${model_tag}"$'\t'"${job}"$'\t'"${gpu}"$'\t'"${port}")
done <<< "${ROLLOUT_WORKER_LAYOUT}"

if (( ${#layout_rows[@]} == 0 )); then
  echo "ROLLOUT_WORKER_LAYOUT contains no workers." >&2
  exit 2
fi

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

for row in "${layout_rows[@]}"; do
  IFS=$'\t' read -r model_tag job gpu port <<< "${row}"
  key="${model_tag}|${job}"
  shard_index=${next_shard[$key]:-0}
  shard_count=${shard_counts[$key]}
  next_shard[$key]=$(( shard_index + 1 ))
  launch_worker "${model_tag}" "${job}" "${gpu}" "${port}" "${shard_index}" "${shard_count}"
done

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
