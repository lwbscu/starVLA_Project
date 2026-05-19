#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export PROJECT_ROOT=${PROJECT_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project}
export CONDA_ROOT=${CONDA_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/26220216/miniconda3}
export STAR_VLA_PYTHON=${STAR_VLA_PYTHON:-"${CONDA_ROOT}/envs/starVLA_qwen35/bin/python"}

export QWEN35_0P8B=${QWEN35_0P8B:-"${PROJECT_ROOT}/playground/Pretrained_models/Qwen3.5-0.8B"}
export QWEN35_4B=${QWEN35_4B:-"${PROJECT_ROOT}/playground/Pretrained_models/Qwen3.5-4B"}
export QWEN35_9B=${QWEN35_9B:-"${PROJECT_ROOT}/playground/Pretrained_models/Qwen3.5-9B"}

export H200_CALVIN_DATA_ROOT=${H200_CALVIN_DATA_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d}
export H200_CALVIN_DATA_NAME=${H200_CALVIN_DATA_NAME:-calvin_task_ABC_D}
export H200_CALVIN_DATA_MIX=${H200_CALVIN_DATA_MIX:-calvin_abc_d_h200}

export ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2_route_validation.yaml}
export CONFIG_YAML=${CONFIG_YAML:-examples/calvin/train_files/starvla_train_calvin_qwen35_pi_h200.yaml}

export WANDB_MODE=${WANDB_MODE:-disabled}
export NO_ALBUMENTATIONS_UPDATE=${NO_ALBUMENTATIONS_UPDATE:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

export ROUTE=p4_pi
export ROUTE_LABEL=pi_state
export SERVER_DIR=${SERVER_DIR:-server1_pi_state}
export TRAIN_VARIANT=${TRAIN_VARIANT:-proprio}

export MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-30000}
export SAVE_INTERVAL=${SAVE_INTERVAL:-10000}
export EVAL_INTERVAL=${EVAL_INTERVAL:-1000000}
export CHECKPOINT_KEEP_LATEST=${CHECKPOINT_KEEP_LATEST:-1}
export CHECKPOINT_KEEP_STEPS=${CHECKPOINT_KEEP_STEPS:-10000,20000,30000}
export LOGGING_FREQUENCY=${LOGGING_FREQUENCY:-10}
export USE_TENSORBOARD=${USE_TENSORBOARD:-true}

export OBS_IMAGE_SIZE=${OBS_IMAGE_SIZE:-"[224,224]"}
export ACTION_DIM=${ACTION_DIM:-7}
export ACTION_HORIZON=${ACTION_HORIZON:-8}
export NUM_ACTIONS_CHUNK=${NUM_ACTIONS_CHUNK:-8}
export ACTION_QUERY_NUM=${ACTION_QUERY_NUM:-128}
export ADAPTER_HIDDEN_DIM=${ADAPTER_HIDDEN_DIM:-auto}
export INCLUDE_STATE=true
export STATE_DIM=${STATE_DIM:-8}
export ADAPTER_USE_PROPRIO=${ADAPTER_USE_PROPRIO:-false}
export PI_STATE_DIM=${PI_STATE_DIM:-8}
export PI_NUM_INFERENCE_TIMESTEPS=${PI_NUM_INFERENCE_TIMESTEPS:-4}
export PI_REPEATED_DIFFUSION_STEPS=${PI_REPEATED_DIFFUSION_STEPS:-2}
export PI_NUM_TARGET_VISION_TOKENS=${PI_NUM_TARGET_VISION_TOKENS:-32}
export DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}
export H200_PI_STATE_MODELS=${H200_PI_STATE_MODELS:-"qwen35_0p8b qwen35_4b qwen35_9b"}

export BATCH_NAME=${BATCH_NAME:-$(date +"%Y%m%d")_h200_server1_pi_state_30k_v1}
export BATCH_ROOT=${BATCH_ROOT:-"${PROJECT_ROOT}/logs/${BATCH_NAME}"}

if [[ ! -x "${STAR_VLA_PYTHON}" ]]; then
  echo "STAR_VLA_PYTHON is not executable: ${STAR_VLA_PYTHON}" >&2
  exit 2
fi

for model_dir in "${QWEN35_0P8B}" "${QWEN35_4B}" "${QWEN35_9B}"; do
  if [[ ! -f "${model_dir}/config.json" ]]; then
    echo "Missing model config: ${model_dir}/config.json" >&2
    exit 2
  fi
done

CALVIN_DATASET_DIR="${H200_CALVIN_DATA_ROOT%/}/${H200_CALVIN_DATA_NAME}"
if [[ ! -f "${CALVIN_DATASET_DIR}/meta/info.json" || ! -f "${CALVIN_DATASET_DIR}/meta/modality.json" || ! -d "${CALVIN_DATASET_DIR}/data" ]]; then
  echo "CALVIN LeRobot dataset is incomplete: ${CALVIN_DATASET_DIR}" >&2
  exit 2
fi

mkdir -p "${BATCH_ROOT}/${SERVER_DIR}/terminal"

qwen_hidden_size() {
  "${STAR_VLA_PYTHON}" - "$1" <<'PY'
import json
import sys

model_dir = sys.argv[1]
with open(f"{model_dir}/config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)
text_cfg = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}
vision_cfg = cfg.get("vision_config") if isinstance(cfg.get("vision_config"), dict) else {}
hidden = cfg.get("hidden_size") or text_cfg.get("hidden_size") or vision_cfg.get("out_hidden_size")
if not isinstance(hidden, int) or hidden <= 0:
    raise SystemExit(f"cannot resolve hidden size from {model_dir}/config.json")
print(hidden)
PY
}

check_state_dim() {
  "${STAR_VLA_PYTHON}" - "${CALVIN_DATASET_DIR}/meta/info.json" "${STATE_DIM}" <<'PY'
import json
import sys

info_path = sys.argv[1]
expected_dim = int(sys.argv[2])
with open(info_path, "r", encoding="utf-8") as f:
    info = json.load(f)
state_feature = info.get("features", {}).get("state")
shape = state_feature.get("shape") if isinstance(state_feature, dict) else None
if not isinstance(shape, list) or not shape or int(shape[0]) != expected_dim:
    raise SystemExit(
        f"INCLUDE_STATE=true expects features.state.shape[0] == {expected_dim}, "
        f"got {shape} in {info_path}"
    )
print(f"state_dim_check=ok shape={shape}")
PY
}

check_state_dim

check_sample_parquet_contract() {
  "${STAR_VLA_PYTHON}" - "${CALVIN_DATASET_DIR}" "${STATE_DIM}" "${ACTION_DIM}" "${ACTION_HORIZON}" <<'PY'
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
state_dim = int(sys.argv[2])
action_dim = int(sys.argv[3])
horizon = int(sys.argv[4])

parquets = sorted((root / "data").glob("**/*.parquet"))
if not parquets:
    raise SystemExit(f"no parquet files under {root / 'data'}")

sample = parquets[0]
df = pd.read_parquet(sample)
missing = [name for name in ("state", "actions") if name not in df.columns]
if missing:
    raise SystemExit(f"sample parquet missing required PI-State columns {missing}: {sample}")
if len(df) < 2 * horizon:
    raise SystemExit(f"sample parquet too short for AWAC t+2H check: len={len(df)}, H={horizon}, file={sample}")

state = np.asarray(df["state"].iloc[0]).reshape(-1)
action = np.asarray(df["actions"].iloc[0]).reshape(-1)
if state.shape[0] != state_dim:
    raise SystemExit(f"state dim mismatch: expected {state_dim}, got {state.shape[0]}, file={sample}")
if action.shape[0] != action_dim:
    raise SystemExit(f"action dim mismatch: expected {action_dim}, got {action.shape[0]}, file={sample}")

print(f"sample_parquet_contract=ok file={sample} state_dim={state_dim} action_dim={action_dim} H={horizon}")
PY
}

check_sample_parquet_contract

declare -a H200_BATCH_PIDS=()
declare -a H200_BATCH_NAMES=()
declare -a H200_BATCH_LOG_DIRS=()

launch_qwen_size() {
  local model_tag=$1
  local base_vlm=$2
  local gpus=$3
  local num_processes=$4
  local main_process_port=$5
  local hidden_dim
  hidden_dim="$(qwen_hidden_size "${base_vlm}")"

  local run_id="${ROUTE_LABEL}_${model_tag}_${MAX_TRAIN_STEPS}step"
  local log_dir="${BATCH_ROOT}/${SERVER_DIR}/${model_tag}/${run_id}"
  local launcher_log="${log_dir}/terminal/launcher.log"
  local env_log="${log_dir}/terminal/env.log"

  mkdir -p "${log_dir}/terminal"
  echo "launch server=${SERVER_DIR} route=${ROUTE} model=${model_tag} gpus=${gpus} port=${main_process_port} hidden_dim=${hidden_dim} log_dir=${log_dir}"

  {
    echo "timestamp=$(date -Is)"
    echo "server=${SERVER_DIR}"
    echo "route=${ROUTE}"
    echo "route_label=${ROUTE_LABEL}"
    echo "model_tag=${model_tag}"
    echo "base_vlm=${base_vlm}"
    echo "qwen_hidden_dim=${hidden_dim}"
    echo "config_yaml=${CONFIG_YAML}"
    echo "cuda_visible_devices=${gpus}"
    echo "num_processes=${num_processes}"
    echo "main_process_port=${main_process_port}"
    echo "train_variant=${TRAIN_VARIANT}"
    echo "selected_models=${H200_PI_STATE_MODELS}"
    echo "include_state=${INCLUDE_STATE}"
    echo "state_dim=${STATE_DIM}"
    echo "pi_state_dim=${PI_STATE_DIM}"
    echo "action_dim=${ACTION_DIM}"
    echo "action_horizon=${ACTION_HORIZON}"
    echo "max_train_steps=${MAX_TRAIN_STEPS}"
    echo "save_interval=${SAVE_INTERVAL}"
    echo "checkpoint_keep_latest=${CHECKPOINT_KEEP_LATEST}"
    echo "checkpoint_keep_steps=${CHECKPOINT_KEEP_STEPS}"
    echo "tensorboard_log_dir=${log_dir}/tensorboard"
    echo "run_id=${run_id}"
    echo "log_dir=${log_dir}"
    echo "launcher_log=${launcher_log}"
  } > "${env_log}"

  (
    set +e
    echo "===== launcher start $(date -Is) ====="
    cat "${env_log}"
    echo "===== terminal stdout/stderr ====="
    ROUTE="${ROUTE}" \
    BASE_VLM="${base_vlm}" \
    CONFIG_YAML="${CONFIG_YAML}" \
    ADAPTER_HIDDEN_DIM="${hidden_dim}" \
    INCLUDE_STATE="${INCLUDE_STATE}" \
    STATE_DIM="${STATE_DIM}" \
    ADAPTER_USE_PROPRIO="${ADAPTER_USE_PROPRIO}" \
    PI_STATE_DIM="${PI_STATE_DIM}" \
    CUDA_VISIBLE_DEVICES="${gpus}" \
    NUM_PROCESSES="${num_processes}" \
    MAIN_PROCESS_PORT="${main_process_port}" \
    MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS}" \
    SAVE_INTERVAL="${SAVE_INTERVAL}" \
    EVAL_INTERVAL="${EVAL_INTERVAL}" \
    CHECKPOINT_KEEP_LATEST="${CHECKPOINT_KEEP_LATEST}" \
    CHECKPOINT_KEEP_STEPS="${CHECKPOINT_KEEP_STEPS}" \
    LOGGING_FREQUENCY="${LOGGING_FREQUENCY}" \
    USE_TENSORBOARD="${USE_TENSORBOARD}" \
    TENSORBOARD_LOG_DIR="${log_dir}/tensorboard" \
    RUN_TS="${BATCH_NAME}_${SERVER_DIR}_${model_tag}" \
    RUN_ID="${run_id}" \
    LOG_DIR="${log_dir}" \
    STAR_VLA_PYTHON="${STAR_VLA_PYTHON}" \
    DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS}" \
    H200_CALVIN_DATA_ROOT="${H200_CALVIN_DATA_ROOT}" \
    H200_CALVIN_DATA_NAME="${H200_CALVIN_DATA_NAME}" \
    H200_CALVIN_DATA_MIX="${H200_CALVIN_DATA_MIX}" \
    ACCELERATE_CONFIG="${ACCELERATE_CONFIG}" \
    bash examples/calvin/train_files/run_route_validation_train.sh
    status=$?
    echo "===== launcher end $(date -Is) status=${status} ====="
    exit "${status}"
  ) > "${launcher_log}" 2>&1 &

  local pid=$!
  echo "${pid}" > "${log_dir}/pid.txt"
  H200_BATCH_PIDS+=("${pid}")
  H200_BATCH_NAMES+=("${model_tag}")
  H200_BATCH_LOG_DIRS+=("${log_dir}")
}

should_launch_model() {
  local model_tag=$1
  [[ " ${H200_PI_STATE_MODELS} " == *" ${model_tag} "* ]]
}

if should_launch_model qwen35_0p8b; then
  launch_qwen_size qwen35_0p8b "${QWEN35_0P8B}" 0 1 33100
fi
if should_launch_model qwen35_4b; then
  launch_qwen_size qwen35_4b "${QWEN35_4B}" 1,2 2 33110
fi
if should_launch_model qwen35_9b; then
  launch_qwen_size qwen35_9b "${QWEN35_9B}" 3,4,5,6,7 5 33120
fi

if (( ${#H200_BATCH_PIDS[@]} == 0 )); then
  echo "No PI-State models selected. Set H200_PI_STATE_MODELS to one or more of: qwen35_0p8b qwen35_4b qwen35_9b" >&2
  exit 2
fi

status=0
for idx in "${!H200_BATCH_PIDS[@]}"; do
  pid="${H200_BATCH_PIDS[$idx]}"
  name="${H200_BATCH_NAMES[$idx]}"
  log_dir="${H200_BATCH_LOG_DIRS[$idx]}"
  if ! wait "${pid}"; then
    echo "FAILED model=${name} pid=${pid} log_dir=${log_dir}" >&2
    tail -n 160 "${log_dir}/terminal/launcher.log" >&2 || true
    status=1
  else
    echo "DONE model=${name} pid=${pid} log_dir=${log_dir}"
  fi
done

exit "${status}"
