#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
export CONDA_ROOT=${CONDA_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/26220216/miniconda3}
export STARVLA_ENV=${STARVLA_ENV:-starVLA_qwen35}
export H200_QWEN35_9B=${H200_QWEN35_9B:-./playground/Pretrained_models/Qwen3.5-9B}
export PATH="${CONDA_ROOT}/bin:${PATH}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${STARVLA_ENV}"

export STAR_VLA_PYTHON=${STAR_VLA_PYTHON:-"$(python -c 'import sys; print(sys.executable)')"}
export H200_CALVIN_DATA_ROOT=${H200_CALVIN_DATA_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d}
export H200_CALVIN_DATA_NAME=${H200_CALVIN_DATA_NAME:-calvin_task_ABC_D}
export H200_CALVIN_DATA_MIX=${H200_CALVIN_DATA_MIX:-calvin_abc_d_h200}
export LOG_ROOT=${LOG_ROOT:-logs/h200_fastexplore}
export ROUTE=${ROUTE:?Set ROUTE, e.g. p0_oft}
export TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3,4,5,6,7}
export NUM_PROCESSES=${NUM_PROCESSES:-8}
export MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29600}
export DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-16}
if [[ "${ROUTE}" == "p4_pi" ]]; then
  export CONFIG_YAML=${CONFIG_YAML:-examples/calvin/train_files/starvla_train_calvin_qwen35_pi_h200.yaml}
else
  export CONFIG_YAML=${CONFIG_YAML:-examples/calvin/train_files/starvla_train_calvin_qwen35_oft_h200.yaml}
fi
export ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2_route_validation.yaml}
export OBS_IMAGE_SIZE=${OBS_IMAGE_SIZE:-"[224,224]"}
export ACTION_HORIZON=${ACTION_HORIZON:-8}
export NUM_ACTIONS_CHUNK=${NUM_ACTIONS_CHUNK:-8}
export ACTION_QUERY_NUM=${ACTION_QUERY_NUM:-128}
export ADAPTER_HIDDEN_DIM=${ADAPTER_HIDDEN_DIM:-2048}
export LORA_R=${LORA_R:-64}
export LORA_ALPHA=${LORA_ALPHA:-128}
export LORA_DROPOUT=${LORA_DROPOUT:-0.05}
export PI_NUM_INFERENCE_TIMESTEPS=${PI_NUM_INFERENCE_TIMESTEPS:-4}
export PI_REPEATED_DIFFUSION_STEPS=${PI_REPEATED_DIFFUSION_STEPS:-2}
export PI_NUM_TARGET_VISION_TOKENS=${PI_NUM_TARGET_VISION_TOKENS:-32}
export EVAL_ENABLED=${EVAL_ENABLED:-1}
export EVAL_HOST=${EVAL_HOST:-127.0.0.1}
export EVAL_PORT=${EVAL_PORT:-5694}
export EVAL_GPU=${EVAL_GPU:-${TRAIN_GPUS%%,*}}
export EVAL_UNNORM_KEY=${EVAL_UNNORM_KEY:-franka}
export SMOKE_STEPS=${SMOKE_STEPS:-1000}
export FAST_STEPS=${FAST_STEPS:-10000}
export DECISION_STEPS=${DECISION_STEPS:-30000}
export SMOKE_SAVE_INTERVAL=${SMOKE_SAVE_INTERVAL:-${SMOKE_STEPS}}
export FAST_SAVE_INTERVAL=${FAST_SAVE_INTERVAL:-5000}
export DECISION_SAVE_INTERVAL=${DECISION_SAVE_INTERVAL:-10000}
export SMOKE_EVAL_SEQUENCES=${SMOKE_EVAL_SEQUENCES:-1}
export FAST_EVAL_SEQUENCES=${FAST_EVAL_SEQUENCES:-3}
export DECISION_EVAL_SEQUENCES=${DECISION_EVAL_SEQUENCES:-5}
export POLICY_SERVER_START_TIMEOUT=${POLICY_SERVER_START_TIMEOUT:-600}
export CALVIN_PYTHON=${CALVIN_PYTHON:-"${CONDA_ROOT}/envs/calvin/bin/python"}
export CALVIN_CONFIG_PATH=${CALVIN_CONFIG_PATH:-"${PROJECT_ROOT}/calvin/calvin_models/conf"}
export EVAL_SEQUENCES_PATH=${EVAL_SEQUENCES_PATH:-examples/calvin/eval_files/eval_sequences.json}
export GIT_PYTHON_REFRESH=${GIT_PYTHON_REFRESH:-quiet}
export CALVIN_ALLOW_OFFLINE_GIT_METADATA=${CALVIN_ALLOW_OFFLINE_GIT_METADATA:-1}
export CALVIN_FORCE_NO_EGL=${CALVIN_FORCE_NO_EGL:-1}

mkdir -p "${LOG_ROOT}/terminal" "${LOG_ROOT}/summary"

if [[ -z "${BASE_VLM:-}" ]]; then
  BASE_VLM="${H200_QWEN35_9B}"
  export BASE_VLM
fi

if [[ ! -f "${H200_QWEN35_9B}/config.json" ]]; then
  echo "Required Qwen3.5-9B weight not found: ${H200_QWEN35_9B}/config.json" >&2
  exit 2
fi

if [[ ! -f "${BASE_VLM}/config.json" ]]; then
  echo "BASE_VLM does not contain config.json: ${BASE_VLM}" >&2
  exit 2
fi

base_vlm_resolved=$(readlink -f "${BASE_VLM}")
qwen35_9b_resolved=$(readlink -f "${H200_QWEN35_9B}")
if [[ "${base_vlm_resolved}" != "${qwen35_9b_resolved}" ]]; then
  echo "H200 fast explore requires Qwen3.5-9B for every route." >&2
  echo "BASE_VLM=${BASE_VLM} -> ${base_vlm_resolved}" >&2
  echo "H200_QWEN35_9B=${H200_QWEN35_9B} -> ${qwen35_9b_resolved}" >&2
  exit 2
fi

if [[ ! -d "${H200_CALVIN_DATA_ROOT%/}/${H200_CALVIN_DATA_NAME}" ]]; then
  echo "CALVIN dataset not found: ${H200_CALVIN_DATA_ROOT%/}/${H200_CALVIN_DATA_NAME}" >&2
  exit 2
fi

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
  for candidate in \
    "${H200_CALVIN_DATA_ROOT%/}/task_ABC_D" \
    "${H200_CALVIN_DATA_ROOT%/}/task_D_D" \
    "${PROJECT_ROOT}/calvin/dataset/calvin_debug_dataset" \
    "${PROJECT_ROOT}/calvin/dataset/task_ABC_D" \
    "${PROJECT_ROOT}/calvin/dataset/task_D_D" \
    "${PROJECT_ROOT}/calvin/dataset/calvin_task_ABC_D" \
    "${H200_CALVIN_DATA_ROOT%/}/${H200_CALVIN_DATA_NAME}"
  do
    if [[ -d "${candidate}" ]] && is_calvin_eval_dataset "${candidate}"; then
      echo "${candidate}"
      return 0
    fi
  done

  echo "No CALVIN eval dataset found. Expected validation/.hydra, training/.hydra, or .hydra. Set H200_CALVIN_EVAL_DATASET_PATH explicitly." >&2
  return 2
}

if [[ "${EVAL_ENABLED}" == "1" ]]; then
  H200_CALVIN_EVAL_DATASET_PATH=$(resolve_eval_dataset)
  export H200_CALVIN_EVAL_DATASET_PATH
  if ! is_calvin_eval_dataset "${H200_CALVIN_EVAL_DATASET_PATH}"; then
    echo "Eval dataset must contain validation/.hydra, training/.hydra, or .hydra: ${H200_CALVIN_EVAL_DATASET_PATH}" >&2
    exit 2
  fi
  if [[ ! -x "${CALVIN_PYTHON}" ]]; then
    echo "CALVIN_PYTHON is not executable: ${CALVIN_PYTHON}" >&2
    exit 2
  fi
  if [[ ! -d "${CALVIN_CONFIG_PATH}" ]]; then
    echo "CALVIN_CONFIG_PATH not found: ${CALVIN_CONFIG_PATH}" >&2
    exit 2
  fi
  if [[ ! -f "${EVAL_SEQUENCES_PATH}" ]]; then
    echo "EVAL_SEQUENCES_PATH not found: ${EVAL_SEQUENCES_PATH}" >&2
    exit 2
  fi
  if ! "${CALVIN_PYTHON}" - <<'PY'
import os
import sys

try:
    import cv2
except ImportError as exc:
    print(
        f"CALVIN_PYTHON={sys.executable} cannot import cv2: {exc}. "
        "Fix the calvin env before running eval. For example: "
        f"{sys.executable} -m pip uninstall -y opencv-python opencv-contrib-python "
        "opencv-python-headless opencv-contrib-python-headless && "
        f"{sys.executable} -m pip install opencv-python-headless==4.11.0.86",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

print(f"cv2 import OK: {cv2.__version__}")
try:
    import git
except ImportError as exc:
    print(
        f"CALVIN_PYTHON={sys.executable} cannot import GitPython: {exc}",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc
print(
    "GitPython import OK: "
    f"refresh={os.environ.get('GIT_PYTHON_REFRESH')}, "
    f"offline_metadata={os.environ.get('CALVIN_ALLOW_OFFLINE_GIT_METADATA')}, "
    f"force_no_egl={os.environ.get('CALVIN_FORCE_NO_EGL')}, "
    f"git={os.environ.get('GIT_PYTHON_GIT_EXECUTABLE', '<unset>')}"
)
PY
  then
    exit 2
  fi
fi

"${STAR_VLA_PYTHON}" - <<'PY'
from transformers import Qwen3_5ForConditionalGeneration
print("Qwen3.5 import OK")
PY

PIPELINE_TS=${PIPELINE_TS:-$(date +"%Y%m%d_%H%M%S")}
PIPELINE_LOG="${LOG_ROOT}/terminal/${PIPELINE_TS}_${ROUTE}_pipeline.log"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

echo "PIPELINE_TS=${PIPELINE_TS}"
echo "ROUTE=${ROUTE}"
echo "TRAIN_GPUS=${TRAIN_GPUS}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT}"
echo "H200_QWEN35_9B=${H200_QWEN35_9B}"
echo "BASE_VLM=${BASE_VLM}"
echo "OBS_IMAGE_SIZE=${OBS_IMAGE_SIZE}"
echo "LOG_ROOT=${LOG_ROOT}"
echo "EVAL_ENABLED=${EVAL_ENABLED}"
echo "EVAL_PORT=${EVAL_PORT}"
echo "EVAL_GPU=${EVAL_GPU}"
echo "GIT_PYTHON_REFRESH=${GIT_PYTHON_REFRESH}"
echo "CALVIN_ALLOW_OFFLINE_GIT_METADATA=${CALVIN_ALLOW_OFFLINE_GIT_METADATA}"
echo "CALVIN_FORCE_NO_EGL=${CALVIN_FORCE_NO_EGL}"
echo "GIT_PYTHON_GIT_EXECUTABLE=${GIT_PYTHON_GIT_EXECUTABLE:-<unset>}"
echo "H200_CALVIN_EVAL_DATASET_PATH=${H200_CALVIN_EVAL_DATASET_PATH:-<disabled>}"
echo "SMOKE_STEPS=${SMOKE_STEPS}"
echo "FAST_STEPS=${FAST_STEPS}"
echo "DECISION_STEPS=${DECISION_STEPS}"

ensure_eval_port_free() {
  local port=$1

  EVAL_PORT_TO_CHECK="${port}" "${STAR_VLA_PYTHON}" - <<'PY'
import os
import socket
import sys

port = int(os.environ["EVAL_PORT_TO_CHECK"])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(("0.0.0.0", port))
except OSError as exc:
    print(
        f"EVAL_PORT={port} is already in use before policy server launch: {exc}. "
        "Use a fresh EVAL_PORT/P0_EVAL_PORT/P4_EVAL_PORT or stop the stale policy server.",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc
finally:
    sock.close()

print(f"EVAL_PORT={port} is available before policy server launch")
PY
}

wait_for_policy_server() {
  local server_pid=$1
  local server_log=$2
  local timeout=$3
  local waited=0

  while [[ "${waited}" -lt "${timeout}" ]]; do
    if [[ -f "${server_log}" ]] && grep -q "server listening" "${server_log}"; then
      return 0
    fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "Policy server exited before ready. Log: ${server_log}" >&2
      wait "${server_pid}" || true
      if [[ -f "${server_log}" ]]; then
        tail -n 120 "${server_log}" >&2 || true
      fi
      return 1
    fi
    sleep 2
    waited=$((waited + 2))
  done

  echo "Policy server did not become ready within ${timeout}s. Log: ${server_log}" >&2
  if [[ -f "${server_log}" ]]; then
    tail -n 120 "${server_log}" >&2 || true
  fi
  return 1
}

run_eval_stage() {
  local stage_name=$1
  local steps=$2
  local log_dir=$3
  local ckpt_path=$4
  local num_sequences=$5
  local server_dir="${log_dir}/server_${stage_name}"
  local eval_dir="${log_dir}/eval_${stage_name}_${num_sequences}seq"
  local server_log="${server_dir}/terminal/policy_server.log"
  local server_pid
  local eval_status
  local mp4_count
  local result_file

  echo "===== START eval stage=${stage_name} route=${ROUTE} steps=${steps} num_sequences=${num_sequences} ====="
  mkdir -p "${server_dir}" "${eval_dir}"
  ensure_eval_port_free "${EVAL_PORT}"

  CKPT_PATH="${ckpt_path}" \
  PORT="${EVAL_PORT}" \
  RUN_TS="${PIPELINE_TS}" \
  RUN_ID="server_${stage_name}_${ROUTE}_${steps}step" \
  LOG_DIR="${server_dir}" \
  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" \
  STAR_VLA_PYTHON="${STAR_VLA_PYTHON}" \
  bash examples/calvin/eval_files/run_policy_server_debug.sh &
  server_pid=$!

  if ! wait_for_policy_server "${server_pid}" "${server_log}" "${POLICY_SERVER_START_TIMEOUT}"; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
    return 1
  fi

  set +e
  CKPT_PATH="${ckpt_path}" \
  HOST="${EVAL_HOST}" \
  PORT="${EVAL_PORT}" \
  NUM_SEQUENCES="${num_sequences}" \
  UNNORM_KEY="${EVAL_UNNORM_KEY}" \
  RUN_TS="${PIPELINE_TS}" \
  RUN_ID="eval_${stage_name}_${ROUTE}_${steps}step" \
  LOG_DIR="${eval_dir}" \
  DATASET_PATH="${H200_CALVIN_EVAL_DATASET_PATH}" \
  CALVIN_CONFIG_PATH="${CALVIN_CONFIG_PATH}" \
  EVAL_SEQUENCES_PATH="${EVAL_SEQUENCES_PATH}" \
  CALVIN_PYTHON="${CALVIN_PYTHON}" \
  GIT_PYTHON_REFRESH="${GIT_PYTHON_REFRESH}" \
  CALVIN_ALLOW_OFFLINE_GIT_METADATA="${CALVIN_ALLOW_OFFLINE_GIT_METADATA}" \
  CALVIN_FORCE_NO_EGL="${CALVIN_FORCE_NO_EGL}" \
  GIT_PYTHON_GIT_EXECUTABLE="${GIT_PYTHON_GIT_EXECUTABLE:-}" \
  bash examples/calvin/eval_files/eval_calvin_debug.sh
  eval_status=$?
  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
  set -e

  if [[ "${eval_status}" -ne 0 ]]; then
    echo "Eval failed for stage=${stage_name}. See ${eval_dir}/terminal/eval.log" >&2
    return "${eval_status}"
  fi

  result_file=$(find "${eval_dir}" -type f -name "results.json" -print -quit)
  if [[ -z "${result_file}" ]]; then
    echo "Eval finished but results.json was not produced under ${eval_dir}" >&2
    return 3
  fi

  mp4_count=$(find "${eval_dir}" -type f -name "*.mp4" | wc -l | tr -d " ")
  if [[ "${mp4_count}" -lt 1 ]]; then
    echo "Eval finished but no mp4 was produced under ${eval_dir}" >&2
    return 3
  fi

  echo "===== DONE eval stage=${stage_name} route=${ROUTE} results=${result_file} mp4_count=${mp4_count} ====="
}

run_stage() {
  local stage_name=$1
  local steps=$2
  local save_interval=$3
  local workers=$4
  local eval_sequences=$5
  local run_id="${stage_name}_${ROUTE}_${steps}step"
  local log_dir="${LOG_ROOT}/${PIPELINE_TS}_${run_id}"
  local ckpt_path="${log_dir}/checkpoints/${run_id}/checkpoints/steps_${steps}_pytorch_model.pt"

  echo "===== START stage=${stage_name} route=${ROUTE} steps=${steps} ====="
  CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
  NUM_PROCESSES="${NUM_PROCESSES}" \
  MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT}" \
  BASE_VLM="${BASE_VLM}" \
  OBS_IMAGE_SIZE="${OBS_IMAGE_SIZE}" \
  ACTION_HORIZON="${ACTION_HORIZON}" \
  ACTION_QUERY_NUM="${ACTION_QUERY_NUM}" \
  NUM_ACTIONS_CHUNK="${NUM_ACTIONS_CHUNK}" \
  ADAPTER_HIDDEN_DIM="${ADAPTER_HIDDEN_DIM}" \
  LORA_R="${LORA_R}" \
  LORA_ALPHA="${LORA_ALPHA}" \
  LORA_DROPOUT="${LORA_DROPOUT}" \
  PI_NUM_INFERENCE_TIMESTEPS="${PI_NUM_INFERENCE_TIMESTEPS}" \
  PI_REPEATED_DIFFUSION_STEPS="${PI_REPEATED_DIFFUSION_STEPS}" \
  PI_NUM_TARGET_VISION_TOKENS="${PI_NUM_TARGET_VISION_TOKENS}" \
  CALVIN_DATA_ROOT="${H200_CALVIN_DATA_ROOT}" \
  CALVIN_DATA_NAME="${H200_CALVIN_DATA_NAME}" \
  CALVIN_DATA_MIX="${H200_CALVIN_DATA_MIX}" \
  CONFIG_YAML="${CONFIG_YAML}" \
  ACCELERATE_CONFIG="${ACCELERATE_CONFIG}" \
  ROUTE="${ROUTE}" \
  MAX_TRAIN_STEPS="${steps}" \
  SAVE_INTERVAL="${save_interval}" \
  EVAL_INTERVAL=1000000 \
  RUN_TS="${PIPELINE_TS}" \
  RUN_ID="${run_id}" \
  LOG_DIR="${log_dir}" \
  DATALOADER_NUM_WORKERS="${workers}" \
  STAR_VLA_PYTHON="${STAR_VLA_PYTHON}" \
  bash examples/calvin/train_files/run_route_validation_train.sh

  test -f "${ckpt_path}"
  mkdir -p "${log_dir}/metrics"
  "${STAR_VLA_PYTHON}" examples/calvin/eval_files/check_checkpoint_reload.py \
    --ckpt-path "${ckpt_path}" \
    --expected-action-chunk-size "${ACTION_HORIZON}" \
    --expected-unnorm-key franka \
    --output-json "${log_dir}/metrics/reload_check_steps_${steps}.json"

  if [[ "${EVAL_ENABLED}" == "1" ]]; then
    run_eval_stage "${stage_name}" "${steps}" "${log_dir}" "${ckpt_path}" "${eval_sequences}"
  fi

  echo "===== DONE stage=${stage_name} route=${ROUTE} ckpt=${ckpt_path} ====="
}

run_stage smoke1k "${SMOKE_STEPS}" "${SMOKE_SAVE_INTERVAL}" "${SMOKE_WORKERS:-8}" "${SMOKE_EVAL_SEQUENCES}"
run_stage fast10k "${FAST_STEPS}" "${FAST_SAVE_INTERVAL}" "${FAST_WORKERS:-${DATALOADER_NUM_WORKERS}}" "${FAST_EVAL_SEQUENCES}"
run_stage decision30k "${DECISION_STEPS}" "${DECISION_SAVE_INTERVAL}" "${DECISION_WORKERS:-${DATALOADER_NUM_WORKERS}}" "${DECISION_EVAL_SEQUENCES}"

echo "PIPELINE_DONE route=${ROUTE} log=${PIPELINE_LOG}"
