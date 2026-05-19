#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export PROJECT_ROOT=${PROJECT_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project}
export CONDA_ROOT=${CONDA_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/26220216/miniconda3}
export STAR_VLA_PYTHON=${STAR_VLA_PYTHON:-"${CONDA_ROOT}/envs/starVLA_qwen35/bin/python"}
export CALVIN_PYTHON=${CALVIN_PYTHON:-"${CONDA_ROOT}/envs/calvin/bin/python"}

export EVAL_BATCH_ROOT=${EVAL_BATCH_ROOT:-"${PROJECT_ROOT}/logs/20260519_h200_qwen_mix_30k_v1"}
export EVAL_OUTPUT_ROOT=${EVAL_OUTPUT_ROOT:-"${PROJECT_ROOT}/logs/calvin_eval_latest_20260519_h200_qwen_mix_30k_v1"}
export EVAL_GPU=${EVAL_GPU:-0}
export EVAL_PORT=${EVAL_PORT:-6200}
export NUM_SEQUENCES=${NUM_SEQUENCES:-5}
export UNNORM_KEY=${UNNORM_KEY:-franka}
export POLICY_SERVER_START_TIMEOUT=${POLICY_SERVER_START_TIMEOUT:-600}
export FAIL_FAST=${FAIL_FAST:-0}
export REQUIRE_MP4=${REQUIRE_MP4:-1}
export SEND_STATE_TO_POLICY=${SEND_STATE_TO_POLICY:-0}
export WRITE_ROLLOUT_LEROBOT=${WRITE_ROLLOUT_LEROBOT:-0}
export WRITE_ROLLOUT_VIDEOS=${WRITE_ROLLOUT_VIDEOS:-0}

export H200_CALVIN_DATA_ROOT=${H200_CALVIN_DATA_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d}
export H200_CALVIN_DATA_NAME=${H200_CALVIN_DATA_NAME:-calvin_task_ABC_D}
export EVAL_SEQUENCES_PATH=${EVAL_SEQUENCES_PATH:-examples/calvin/eval_files/eval_sequences.json}

export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export GIT_PYTHON_REFRESH=${GIT_PYTHON_REFRESH:-quiet}
export CALVIN_ALLOW_OFFLINE_GIT_METADATA=${CALVIN_ALLOW_OFFLINE_GIT_METADATA:-1}
export CALVIN_FORCE_NO_EGL=${CALVIN_FORCE_NO_EGL:-1}
export NO_ALBUMENTATIONS_UPDATE=${NO_ALBUMENTATIONS_UPDATE:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

resolve_first_existing_dir() {
  local candidate
  for candidate in "$@"; do
    if [[ -d "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

CALVIN_ROOT=${CALVIN_ROOT:-$(resolve_first_existing_dir "${PROJECT_ROOT}/calvin" "${PROJECT_ROOT}/../calvin" || true)}
CALVIN_CONFIG_PATH=${CALVIN_CONFIG_PATH:-$(resolve_first_existing_dir "${PROJECT_ROOT}/calvin/calvin_models/conf" "${PROJECT_ROOT}/../calvin/calvin_models/conf" || true)}
CALVIN_ASSET_ROOT=${CALVIN_ASSET_ROOT:-$(resolve_first_existing_dir "${PROJECT_ROOT}/calvin/calvin_env/data" "${PROJECT_ROOT}/../calvin/calvin_env/data" || true)}
export CALVIN_ROOT CALVIN_CONFIG_PATH CALVIN_ASSET_ROOT

export PYTHONPATH="${PROJECT_ROOT}:${CALVIN_ROOT:-}:${CALVIN_ROOT:-}/calvin_models:${CALVIN_ROOT:-}/calvin_env:${PYTHONPATH:-}"

if [[ -z "${EXPECTED_JOBS:-}" ]]; then
EXPECTED_JOBS=$(cat <<'EOF'
server1_oft/qwen35_0p8b
server1_oft/qwen35_4b
server1_oft/qwen35_9b
server2_adapter/qwen35_0p8b
server2_adapter/qwen35_4b
server2_adapter/qwen35_9b
server3_lora/qwen35_0p8b
server3_lora/qwen35_4b
server3_lora/qwen35_9b
server4_pi/qwen35_0p8b
server4_pi/qwen35_4b
server4_pi/qwen35_9b
EOF
)
fi
export EXPECTED_JOBS

is_calvin_eval_dataset() {
  local candidate=$1
  [[ -f "${candidate}/validation/.hydra/merged_config.yaml" ]] \
    || [[ -f "${candidate}/training/.hydra/merged_config.yaml" ]] \
    || [[ -f "${candidate}/.hydra/merged_config.yaml" ]]
}

resolve_eval_dataset() {
  if [[ -n "${H200_CALVIN_EVAL_DATASET_PATH:-}" ]]; then
    echo "${H200_CALVIN_EVAL_DATASET_PATH}"
    return 0
  fi

  local candidate
  # Keep this order aligned with docs/datasets_anlazy.md:
  # training uses calvin_task_ABC_D (LeRobot), quick CALVIN visual eval uses
  # task_D_D because the shared H200 tree currently exposes D env config there.
  for candidate in \
    "${H200_CALVIN_DATA_ROOT%/}/task_D_D" \
    "${H200_CALVIN_DATA_ROOT%/}/task_ABC_D" \
    "${PROJECT_ROOT}/calvin/dataset/calvin_debug_dataset" \
    "${PROJECT_ROOT}/calvin/dataset/task_D_D" \
    "${PROJECT_ROOT}/calvin/dataset/task_ABC_D" \
    "${PROJECT_ROOT}/../calvin/dataset/calvin_debug_dataset" \
    "${PROJECT_ROOT}/../calvin/dataset/task_D_D" \
    "${PROJECT_ROOT}/../calvin/dataset/task_ABC_D" \
    "${H200_CALVIN_DATA_ROOT%/}/${H200_CALVIN_DATA_NAME}"
  do
    if [[ -d "${candidate}" ]] && is_calvin_eval_dataset "${candidate}"; then
      echo "${candidate}"
      return 0
    fi
  done

  echo "No CALVIN eval dataset found. Set H200_CALVIN_EVAL_DATASET_PATH explicitly." >&2
  return 2
}

verify_calvin_dependencies() {
  if [[ ! -x "${STAR_VLA_PYTHON}" ]]; then
    echo "STAR_VLA_PYTHON is not executable: ${STAR_VLA_PYTHON}" >&2
    return 2
  fi
  if [[ ! -x "${CALVIN_PYTHON}" ]]; then
    echo "CALVIN_PYTHON is not executable: ${CALVIN_PYTHON}" >&2
    return 2
  fi
  if [[ -z "${CALVIN_CONFIG_PATH}" || ! -d "${CALVIN_CONFIG_PATH}" ]]; then
    echo "CALVIN_CONFIG_PATH not found. Set CALVIN_CONFIG_PATH=/path/to/calvin_models/conf" >&2
    return 2
  fi
  if [[ ! -f "${CALVIN_CONFIG_PATH}/callbacks/rollout/tasks/new_playtable_tasks.yaml" ]]; then
    echo "Missing CALVIN task oracle config under CALVIN_CONFIG_PATH=${CALVIN_CONFIG_PATH}" >&2
    return 2
  fi
  if [[ -z "${CALVIN_ASSET_ROOT}" || ! -d "${CALVIN_ASSET_ROOT}" ]]; then
    echo "CALVIN_ASSET_ROOT not found. Set CALVIN_ASSET_ROOT=/path/to/calvin_env/data" >&2
    return 2
  fi
  if [[ ! -f "${CALVIN_ASSET_ROOT}/plane/plane.urdf" \
     && ! -f "${CALVIN_ASSET_ROOT}/franka_panda/panda_longer_finger.urdf" \
     && ! -f "${CALVIN_ASSET_ROOT}/calvin_table_D/urdf/calvin_table_D.urdf" ]]; then
    echo "CALVIN_ASSET_ROOT does not look like calvin_env/data: ${CALVIN_ASSET_ROOT}" >&2
    return 2
  fi
  if [[ ! -f "${EVAL_SEQUENCES_PATH}" ]]; then
    echo "EVAL_SEQUENCES_PATH not found: ${EVAL_SEQUENCES_PATH}" >&2
    return 2
  fi

  "${CALVIN_PYTHON}" - <<'PY'
import importlib
import os
import sys

required = [
    "cv2",
    "git",
    "hydra",
    "omegaconf",
    "pybullet",
    "calvin_agent",
    "calvin_env",
    "imageio",
    "tyro",
    "websockets",
    "msgpack",
    "jinja2",
    "termcolor",
    "tqdm",
    "numpy",
]
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")
if missing:
    print("CALVIN environment dependency check failed:", file=sys.stderr)
    for item in missing:
        print(f"  - {item}", file=sys.stderr)
    print("\nTypical fixes:", file=sys.stderr)
    print("  conda activate calvin", file=sys.stderr)
    print("  pip install tyro websockets msgpack jinja2 imageio imageio-ffmpeg GitPython termcolor tqdm opencv-python-headless==4.11.0.86", file=sys.stderr)
    print("  cd <calvin_repo> && sh install.sh", file=sys.stderr)
    raise SystemExit(2)
try:
    from imageio.plugins import ffmpeg
    print(f"imageio ffmpeg: {ffmpeg.get_exe()}")
except Exception as exc:
    print(f"imageio ffmpeg check failed: {exc}", file=sys.stderr)
    raise SystemExit(2) from exc
import cv2
print(f"CALVIN_PYTHON={sys.executable}")
print(f"cv2={cv2.__version__}")
print(f"PYOPENGL_PLATFORM={os.environ.get('PYOPENGL_PLATFORM')}")
print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL')}")
print(f"CALVIN_FORCE_NO_EGL={os.environ.get('CALVIN_FORCE_NO_EGL')}")
PY
}

latest_checkpoints() {
  "${STAR_VLA_PYTHON}" - <<'PY'
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

root = Path(os.environ["EVAL_BATCH_ROOT"])
expected = [line.strip() for line in os.environ["EXPECTED_JOBS"].splitlines() if line.strip()]
if not root.is_dir():
    raise SystemExit(f"EVAL_BATCH_ROOT not found: {root}")

missing = []
rows = []
pattern = re.compile(r"steps_(\d+)_pytorch_model\.pt$")
for item in expected:
    model_dir = root / item
    if not model_dir.is_dir():
        missing.append(f"{item}: model directory not found")
        continue
    candidates = []
    for ckpt in model_dir.rglob("steps_*_pytorch_model.pt"):
        match = pattern.search(ckpt.name)
        if not match:
            continue
        candidates.append((int(match.group(1)), ckpt.stat().st_mtime, ckpt))
    if not candidates:
        missing.append(f"{item}: no steps_*_pytorch_model.pt checkpoint found")
        continue
    step, _mtime, ckpt = max(candidates, key=lambda x: (x[0], x[1]))
    server, model = item.split("/", 1)
    model_id = f"{server}__{model}__steps_{step}"
    rows.append((server, model, step, model_id, ckpt))

if missing:
    print("Missing expected checkpoints:", file=sys.stderr)
    for item in missing:
        print(f"  - {item}", file=sys.stderr)
    raise SystemExit(3)

for server, model, step, model_id, ckpt in rows:
    print(f"{model_id}\t{ckpt}\t{step}\t{server}\t{model}")
PY
}

wait_for_policy_server() {
  local server_pid=$1
  local port=$2
  local log_file=$3
  local deadline=$((SECONDS + POLICY_SERVER_START_TIMEOUT))
  while (( SECONDS < deadline )); do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "Policy server wrapper exited before port ${port} became ready. See ${log_file}" >&2
      return 1
    fi
    if EVAL_PORT="${port}" "${STAR_VLA_PYTHON}" - <<'PY' >/dev/null 2>&1
import os
import socket
port = int(os.environ["EVAL_PORT"])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(0.5)
try:
    sock.connect(("127.0.0.1", port))
finally:
    sock.close()
PY
    then
      return 0
    fi
    sleep 2
  done
  echo "Timed out waiting for policy server on port ${port}. See ${log_file}" >&2
  return 1
}

cleanup_port() {
  "${STAR_VLA_PYTHON}" examples/calvin/train_files/cleanup_policy_server.py \
    --port "${EVAL_PORT}" \
    --timeout 10 \
    || return $?
}

run_one_eval() {
  local model_id=$1
  local ckpt_path=$2
  local step=$3
  local server_name=$4
  local model_name=$5
  local job_dir="${EVAL_OUTPUT_ROOT}/${server_name}/${model_name}/steps_${step}"
  local server_dir="${job_dir}/policy_server"
  local eval_dir="${job_dir}/eval"
  local server_log="${server_dir}/terminal/policy_server.log"
  local eval_log="${eval_dir}/terminal/eval.log"
  local server_pid
  local eval_status=0
  local cleanup_status=0
  local result_file=""
  local mp4_count=0

  mkdir -p "${server_dir}" "${eval_dir}/terminal" "${job_dir}/configs"
  cp "$0" "${job_dir}/configs/"
  {
    echo "model_id=${model_id}"
    echo "server=${server_name}"
    echo "model=${model_name}"
    echo "step=${step}"
    echo "ckpt_path=${ckpt_path}"
    echo "eval_port=${EVAL_PORT}"
    echo "eval_gpu=${EVAL_GPU}"
    echo "num_sequences=${NUM_SEQUENCES}"
    echo "send_state_to_policy=${SEND_STATE_TO_POLICY}"
    echo "dataset_path=${H200_CALVIN_EVAL_DATASET_PATH}"
    echo "calvin_config_path=${CALVIN_CONFIG_PATH}"
    echo "calvin_asset_root=${CALVIN_ASSET_ROOT}"
  } > "${job_dir}/env.log"

  echo "===== START eval ${model_id} ckpt=${ckpt_path} ====="
  cleanup_port

  CKPT_PATH="${ckpt_path}" \
  PORT="${EVAL_PORT}" \
  RUN_ID="policy_${model_id}" \
  LOG_DIR="${server_dir}" \
  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" \
  STAR_VLA_PYTHON="${STAR_VLA_PYTHON}" \
  bash examples/calvin/eval_files/run_policy_server_debug.sh &
  server_pid=$!

  if ! wait_for_policy_server "${server_pid}" "${EVAL_PORT}" "${server_log}"; then
    cleanup_port || true
    return 1
  fi

  local eval_args=(
    examples/calvin/eval_files/eval_calvin.py
    --args.pretrained-path "${ckpt_path}"
    --args.unnorm-key "${UNNORM_KEY}"
    --args.host 127.0.0.1
    --args.port "${EVAL_PORT}"
    --args.dataset-path "${H200_CALVIN_EVAL_DATASET_PATH}"
    --args.calvin-config-path "${CALVIN_CONFIG_PATH}"
    --args.eval-sequences-path "${EVAL_SEQUENCES_PATH}"
    --args.num-sequences "${NUM_SEQUENCES}"
    --args.eval-log-dir "${eval_dir}/mp4"
    --args.debug
  )
  if [[ "${SEND_STATE_TO_POLICY}" == "1" ]]; then
    eval_args+=(--args.send-state-to-policy)
  fi
  if [[ "${WRITE_ROLLOUT_LEROBOT}" == "1" ]]; then
    eval_args+=(--args.rollout-lerobot-dir "${eval_dir}/rollout_lerobot")
    if [[ "${WRITE_ROLLOUT_VIDEOS}" == "1" ]]; then
      eval_args+=(--args.rollout-lerobot-write-videos)
    fi
  fi

  set +e
  "${CALVIN_PYTHON}" "${eval_args[@]}" 2>&1 | tee "${eval_log}"
  eval_status=${PIPESTATUS[0]}
  cleanup_port
  cleanup_status=$?
  wait "${server_pid}" 2>/dev/null || true
  set -e

  if [[ "${cleanup_status}" -ne 0 ]]; then
    echo "Policy server cleanup failed for ${model_id}" >&2
    return "${cleanup_status}"
  fi
  if [[ "${eval_status}" -ne 0 ]]; then
    echo "Eval failed for ${model_id}. See ${eval_log}" >&2
    return "${eval_status}"
  fi

  result_file=$(find "${eval_dir}" -type f -name results.json -print -quit)
  if [[ -z "${result_file}" ]]; then
    echo "Eval finished but results.json was not produced for ${model_id}" >&2
    return 3
  fi
  mp4_count=$(find "${eval_dir}" -type f -name "*.mp4" | wc -l | tr -d " ")
  if [[ "${REQUIRE_MP4}" == "1" && "${mp4_count}" -lt 1 ]]; then
    echo "Eval finished but no mp4 was produced for ${model_id}" >&2
    return 3
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${model_id}" "${step}" "${ckpt_path}" "${result_file}" "${mp4_count}" "OK" >> "${EVAL_OUTPUT_ROOT}/summary.tsv"
  echo "===== DONE eval ${model_id} results=${result_file} mp4_count=${mp4_count} ====="
}

main() {
  mkdir -p "${EVAL_OUTPUT_ROOT}"
  H200_CALVIN_EVAL_DATASET_PATH=$(resolve_eval_dataset)
  export H200_CALVIN_EVAL_DATASET_PATH
  if ! is_calvin_eval_dataset "${H200_CALVIN_EVAL_DATASET_PATH}"; then
    echo "Eval dataset is invalid: ${H200_CALVIN_EVAL_DATASET_PATH}" >&2
    exit 2
  fi
  verify_calvin_dependencies
  : > "${EVAL_OUTPUT_ROOT}/summary.tsv"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "model_id" "step" "ckpt_path" "results_json" "mp4_count" "status" >> "${EVAL_OUTPUT_ROOT}/summary.tsv"

  local status=0
  while IFS=$'\t' read -r model_id ckpt_path step server_name model_name; do
    if ! run_one_eval "${model_id}" "${ckpt_path}" "${step}" "${server_name}" "${model_name}"; then
      status=1
      printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${model_id}" "${step}" "${ckpt_path}" "" "0" "FAILED" >> "${EVAL_OUTPUT_ROOT}/summary.tsv"
      if [[ "${FAIL_FAST}" == "1" ]]; then
        exit "${status}"
      fi
    fi
  done < <(latest_checkpoints)

  exit "${status}"
}

main "$@"
