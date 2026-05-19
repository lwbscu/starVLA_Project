#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export WANDB_MODE=${WANDB_MODE:-disabled}
export NO_ALBUMENTATIONS_UPDATE=${NO_ALBUMENTATIONS_UPDATE:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
CONDA_ROOT=${CONDA_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/26220216/miniconda3}
STAR_VLA_PYTHON=${STAR_VLA_PYTHON:-"${CONDA_ROOT}/envs/starVLA_qwen35/bin/python"}

ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2_route_validation.yaml}
CONFIG_YAML=${CONFIG_YAML:-examples/calvin/train_files/starvla_train_calvin_qwen35_pi_h200.yaml}

QWEN35_0P8B=${QWEN35_0P8B:-"${PROJECT_ROOT}/playground/Pretrained_models/Qwen3.5-0.8B"}
QWEN35_4B=${QWEN35_4B:-"${PROJECT_ROOT}/playground/Pretrained_models/Qwen3.5-4B"}
QWEN35_9B=${QWEN35_9B:-"${PROJECT_ROOT}/playground/Pretrained_models/Qwen3.5-9B"}

CROSS_DATA_ROOT=${CROSS_DATA_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/three/dataset}
CROSS_DATA_MIX=${CROSS_DATA_MIX:-h200_three_libero_only}
REQUIRE_MODALITY_JSON=${REQUIRE_MODALITY_JSON:-1}
SKIP_DATA_PREFLIGHT=${SKIP_DATA_PREFLIGHT:-0}
DRY_RUN=${DRY_RUN:-0}

MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-30000}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
EVAL_INTERVAL=${EVAL_INTERVAL:-1000000}
CHECKPOINT_KEEP_LATEST=${CHECKPOINT_KEEP_LATEST:-1}
CHECKPOINT_KEEP_STEPS=${CHECKPOINT_KEEP_STEPS:-10000,20000,30000}
LOGGING_FREQUENCY=${LOGGING_FREQUENCY:-10}
USE_TENSORBOARD=${USE_TENSORBOARD:-true}

OBS_IMAGE_SIZE=${OBS_IMAGE_SIZE:-"[224,224]"}
ACTION_DIM=${ACTION_DIM:-7}
ACTION_HORIZON=${ACTION_HORIZON:-8}
NUM_ACTIONS_CHUNK=${NUM_ACTIONS_CHUNK:-8}
ACTION_QUERY_NUM=${ACTION_QUERY_NUM:-128}
PI_NUM_INFERENCE_TIMESTEPS=${PI_NUM_INFERENCE_TIMESTEPS:-4}
PI_REPEATED_DIFFUSION_STEPS=${PI_REPEATED_DIFFUSION_STEPS:-2}
PI_NUM_TARGET_VISION_TOKENS=${PI_NUM_TARGET_VISION_TOKENS:-32}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
INCLUDE_STATE=${INCLUDE_STATE:-false}
STATE_DIM=${STATE_DIM:-8}
PI_STATE_DIM=${PI_STATE_DIM:-}

BATCH_NAME=${BATCH_NAME:-$(date +"%Y%m%d")_h200_three_crossdata_pi_${CROSS_DATA_MIX}}
BATCH_ROOT=${BATCH_ROOT:-"${PROJECT_ROOT}/logs/${BATCH_NAME}"}
SERVER_DIR=${SERVER_DIR:-crossdata_pi}
ROUTE_LABEL=${ROUTE_LABEL:-pi_crossdata}

normalize_bool_01() {
  local name=$1
  local value=$2
  case "${value}" in
    1|true|True|TRUE|yes|Yes|YES|on|On|ON) echo 1 ;;
    0|false|False|FALSE|no|No|NO|off|Off|OFF) echo 0 ;;
    *)
      echo "${name} must be boolean-like, got: ${value}" >&2
      return 2
      ;;
  esac
}

normalize_bool_tf() {
  local name=$1
  local value=$2
  case "${value}" in
    1|true|True|TRUE|yes|Yes|YES|on|On|ON) echo true ;;
    0|false|False|FALSE|no|No|NO|off|Off|OFF) echo false ;;
    *)
      echo "${name} must be boolean-like, got: ${value}" >&2
      return 2
      ;;
  esac
}

REQUIRE_MODALITY_JSON="$(normalize_bool_01 REQUIRE_MODALITY_JSON "${REQUIRE_MODALITY_JSON}")"
SKIP_DATA_PREFLIGHT="$(normalize_bool_01 SKIP_DATA_PREFLIGHT "${SKIP_DATA_PREFLIGHT}")"
DRY_RUN="$(normalize_bool_01 DRY_RUN "${DRY_RUN}")"
INCLUDE_STATE="$(normalize_bool_tf INCLUDE_STATE "${INCLUDE_STATE}")"
USE_TENSORBOARD="$(normalize_bool_tf USE_TENSORBOARD "${USE_TENSORBOARD}")"

if [[ -n "${PI_STATE_DIM}" ]]; then
  if ! [[ "${PI_STATE_DIM}" =~ ^[0-9]+$ ]] || (( PI_STATE_DIM < 1 )); then
    echo "PI_STATE_DIM must be empty or a positive integer, got: ${PI_STATE_DIM}" >&2
    exit 2
  fi
fi

if [[ -n "${PI_STATE_DIM}" ]]; then
  PI_EFFECTIVE_STATE_DIM="${PI_STATE_DIM}"
elif [[ "${INCLUDE_STATE}" == "true" ]]; then
  PI_EFFECTIVE_STATE_DIM="${STATE_DIM}"
else
  PI_EFFECTIVE_STATE_DIM="${ACTION_DIM}"
fi

require_file() {
  local path=$1
  local label=$2
  if [[ ! -f "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 2
  fi
}

require_executable() {
  local path=$1
  local label=$2
  if [[ ! -x "${path}" ]]; then
    echo "${label} is not executable: ${path}" >&2
    exit 2
  fi
}

require_executable "${STAR_VLA_PYTHON}" STAR_VLA_PYTHON
require_file "${CONFIG_YAML}" CONFIG_YAML
require_file "${ACCELERATE_CONFIG}" ACCELERATE_CONFIG
require_file "${QWEN35_0P8B}/config.json" QWEN35_0P8B
require_file "${QWEN35_4B}/config.json" QWEN35_4B
require_file "${QWEN35_9B}/config.json" QWEN35_9B

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

preflight_cross_data() {
  if [[ "${SKIP_DATA_PREFLIGHT}" == 1 ]]; then
    echo "SKIP_DATA_PREFLIGHT=1: dataset checks skipped by explicit request."
    return
  fi

  "${STAR_VLA_PYTHON}" - "${CROSS_DATA_ROOT}" "${CROSS_DATA_MIX}" "${REQUIRE_MODALITY_JSON}" "${INCLUDE_STATE}" "${STATE_DIM}" <<'PY'
import json
import sys
from pathlib import Path

from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES, ROBOT_TYPE_CONFIG_MAP

root = Path(sys.argv[1]).expanduser()
mix_name = sys.argv[2]
require_modality = sys.argv[3] == "1"
include_state = sys.argv[4] == "true"
expected_state_dim = int(sys.argv[5])

if not root.is_dir():
    raise SystemExit(f"CROSS_DATA_ROOT is not a directory: {root}")
if mix_name not in DATASET_NAMED_MIXTURES:
    candidates = sorted(k for k in DATASET_NAMED_MIXTURES if k.startswith("h200_three"))
    raise SystemExit(f"Unknown CROSS_DATA_MIX={mix_name!r}. Available H200 mixes: {candidates}")

mixture = DATASET_NAMED_MIXTURES[mix_name]
print(f"CROSS_DATA_ROOT={root}")
print(f"CROSS_DATA_MIX={mix_name}")
print(f"datasets={len(mixture)}")

for dataset_name, weight, robot_type in mixture:
    if robot_type not in ROBOT_TYPE_CONFIG_MAP:
        raise SystemExit(f"Robot type {robot_type!r} for dataset {dataset_name!r} is not registered")

    dataset_dir = root / dataset_name
    info_path = dataset_dir / "meta" / "info.json"
    modality_path = dataset_dir / "meta" / "modality.json"
    data_dir = dataset_dir / "data"
    videos_dir = dataset_dir / "videos"

    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset directory missing: {dataset_dir}")
    if not info_path.is_file():
        raise SystemExit(f"Missing LeRobot metadata: {info_path}")
    if require_modality and not modality_path.is_file():
        raise SystemExit(
            f"Missing required modality metadata: {modality_path}\n"
            "The current StarVLA LeRobot loader requires meta/modality.json. "
            "Do not include this dataset until modality metadata is generated and verified."
        )
    if not data_dir.is_dir():
        raise SystemExit(f"Missing LeRobot parquet directory: {data_dir}")
    if not videos_dir.is_dir():
        print(f"WARNING: videos directory not found for {dataset_name}: {videos_dir}")

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    features = info.get("features", {})
    action_feature = features.get("action")
    state_feature = features.get("observation.state") or features.get("state")
    state_shape = state_feature.get("shape") if isinstance(state_feature, dict) else None
    if include_state and (not isinstance(state_shape, list) or not state_shape or int(state_shape[0]) != expected_state_dim):
        raise SystemExit(
            f"INCLUDE_STATE=true expects state dim {expected_state_dim}, "
            f"but {info_path} has state shape {state_shape}"
        )

    print(
        f"- {dataset_name}: robot={robot_type} weight={weight} "
        f"codebase={info.get('codebase_version')} episodes={info.get('total_episodes')} "
        f"frames={info.get('total_frames')} fps={info.get('fps')} "
        f"action_shape={action_feature.get('shape') if isinstance(action_feature, dict) else None} "
        f"state_shape={state_shape} modality_json={modality_path.is_file()}"
    )
PY
}

check_port_available() {
  local port=$1
  if ! MAIN_PROCESS_PORT="${port}" "${STAR_VLA_PYTHON}" - <<'PY'
import os
import socket
import sys

port = int(os.environ["MAIN_PROCESS_PORT"])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(("0.0.0.0", port))
except OSError as exc:
    print(f"MAIN_PROCESS_PORT={port} is unavailable before launch: {exc}", file=sys.stderr)
    sys.exit(1)
finally:
    sock.close()
print(f"MAIN_PROCESS_PORT={port} is available")
PY
  then
    if [[ -f examples/calvin/train_files/describe_port_users.py ]]; then
      "${STAR_VLA_PYTHON}" examples/calvin/train_files/describe_port_users.py "${port}" >&2 || true
    fi
    echo "Choose a free MAIN_PROCESS_PORT for this cross-dataset run." >&2
    exit 2
  fi
}

mkdir -p "${BATCH_ROOT}"
preflight_cross_data

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
  check_port_available "${main_process_port}"

  local run_id="${ROUTE_LABEL}_${CROSS_DATA_MIX}_${model_tag}_${MAX_TRAIN_STEPS}step"
  local log_dir="${BATCH_ROOT}/${SERVER_DIR}/${model_tag}/${run_id}"
  local launcher_log="${log_dir}/terminal/launcher.log"
  local env_log="${log_dir}/terminal/env.log"
  local tb_log_dir="${log_dir}/tensorboard"

  mkdir -p "${log_dir}"/{terminal,configs,checkpoints,metrics,train,eval}
  cp "${CONFIG_YAML}" "${log_dir}/configs/"
  cp "${ACCELERATE_CONFIG}" "${log_dir}/configs/"
  cp "$0" "${log_dir}/configs/"

  echo "launch crossdata route=QwenPI mix=${CROSS_DATA_MIX} model=${model_tag} gpus=${gpus} port=${main_process_port} hidden=${hidden_dim} log_dir=${log_dir}"

  {
    echo "timestamp=$(date -Is)"
    echo "server=${SERVER_DIR}"
    echo "route=QwenPI"
    echo "route_label=${ROUTE_LABEL}"
    echo "cross_data_root=${CROSS_DATA_ROOT}"
    echo "cross_data_mix=${CROSS_DATA_MIX}"
    echo "model_tag=${model_tag}"
    echo "base_vlm=${base_vlm}"
    echo "config_yaml=${CONFIG_YAML}"
    echo "cuda_visible_devices=${gpus}"
    echo "num_processes=${num_processes}"
    echo "main_process_port=${main_process_port}"
    echo "hidden_dim=${hidden_dim}"
    echo "max_train_steps=${MAX_TRAIN_STEPS}"
    echo "save_interval=${SAVE_INTERVAL}"
    echo "eval_interval=${EVAL_INTERVAL}"
    echo "checkpoint_keep_latest=${CHECKPOINT_KEEP_LATEST}"
    echo "checkpoint_keep_steps=${CHECKPOINT_KEEP_STEPS}"
    echo "logging_frequency=${LOGGING_FREQUENCY}"
    echo "include_state=${INCLUDE_STATE}"
    echo "state_dim=${STATE_DIM}"
    echo "pi_effective_state_dim=${PI_EFFECTIVE_STATE_DIM}"
    echo "per_device_batch_size=${PER_DEVICE_BATCH_SIZE}"
    echo "run_id=${run_id}"
    echo "log_dir=${log_dir}"
    echo "tensorboard_log_dir=${tb_log_dir}"
    echo "launcher_log=${launcher_log}"
  } > "${env_log}"

  local -a cmd=(
    "${STAR_VLA_PYTHON}" -m accelerate.commands.launch
    --config_file "${ACCELERATE_CONFIG}"
    --num_processes "${num_processes}"
    --main_process_port "${main_process_port}"
    starVLA/training/train_starvla.py
    --config_yaml "${CONFIG_YAML}"
    --run_id "${run_id}"
    --run_root_dir "${log_dir}/checkpoints"
    --framework.name QwenPI
    --framework.qwenvl.base_vlm "${base_vlm}"
    --framework.qwenvl.attn_implementation sdpa
    --framework.action_model.action_model_type LayerwiseFM
    --framework.action_model.action_dim "${ACTION_DIM}"
    --framework.action_model.state_dim "${PI_EFFECTIVE_STATE_DIM}"
    --framework.action_model.action_horizon "${ACTION_HORIZON}"
    --framework.action_model.repeated_diffusion_steps "${PI_REPEATED_DIFFUSION_STEPS}"
    --framework.action_model.num_inference_timesteps "${PI_NUM_INFERENCE_TIMESTEPS}"
    --framework.action_model.num_target_vision_tokens "${PI_NUM_TARGET_VISION_TOKENS}"
    --datasets.vla_data.data_root_dir "${CROSS_DATA_ROOT}"
    --datasets.vla_data.data_mix "${CROSS_DATA_MIX}"
    --datasets.vla_data.include_state "${INCLUDE_STATE}"
    --datasets.vla_data.obs_image_size "${OBS_IMAGE_SIZE}"
    --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}"
    --datasets.vla_data.num_workers "${DATALOADER_NUM_WORKERS}"
    --datasets.vla_data.require_existing_parquet true
    --datasets.vla_data.video_backend torchvision_av
    --trainer.max_train_steps "${MAX_TRAIN_STEPS}"
    --trainer.num_warmup_steps "$(( MAX_TRAIN_STEPS < 1000 ? MAX_TRAIN_STEPS : 1000 ))"
    --trainer.save_interval "${SAVE_INTERVAL}"
    --trainer.eval_interval "${EVAL_INTERVAL}"
    --trainer.logging_frequency "${LOGGING_FREQUENCY}"
    --trainer.use_tensorboard "${USE_TENSORBOARD}"
    --trainer.tensorboard_log_dir "${tb_log_dir}"
    --trainer.checkpoint_keep_latest "${CHECKPOINT_KEEP_LATEST}"
    --trainer.checkpoint_keep_steps "${CHECKPOINT_KEEP_STEPS}"
    --trainer.freeze_modules qwen_vl_interface
  )

  if [[ "${DRY_RUN}" == 1 ]]; then
    printf 'DRY_RUN command for %s:\n' "${model_tag}" | tee -a "${launcher_log}"
    printf '%q ' CUDA_VISIBLE_DEVICES="${gpus}" "${cmd[@]}" | tee -a "${launcher_log}"
    printf '\n' | tee -a "${launcher_log}"
    return 0
  fi

  (
    set +e
    echo "===== launcher start $(date -Is) ====="
    cat "${env_log}"
    echo "===== terminal stdout/stderr ====="
    CUDA_VISIBLE_DEVICES="${gpus}" "${cmd[@]}"
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

launch_three_sizes() {
  launch_qwen_size qwen35_0p8b "${QWEN35_0P8B}" 0 1 33100
  launch_qwen_size qwen35_4b "${QWEN35_4B}" 1,2 2 33110
  launch_qwen_size qwen35_9b "${QWEN35_9B}" 3,4,5,6,7 5 33120

  if [[ "${DRY_RUN}" == 1 ]]; then
    echo "DRY_RUN=1: commands were written under ${BATCH_ROOT}/${SERVER_DIR}/<model>/<run_id>/terminal/launcher.log"
    return 0
  fi

  local status=0
  local idx
  for idx in "${!H200_BATCH_PIDS[@]}"; do
    local pid="${H200_BATCH_PIDS[$idx]}"
    local name="${H200_BATCH_NAMES[$idx]}"
    local log_dir="${H200_BATCH_LOG_DIRS[$idx]}"
    if ! wait "${pid}"; then
      echo "FAILED model=${name} pid=${pid} log_dir=${log_dir}" >&2
      tail -n 160 "${log_dir}/terminal/launcher.log" >&2 || true
      status=1
    else
      echo "DONE model=${name} pid=${pid} log_dir=${log_dir}"
    fi
  done
  return "${status}"
}

launch_three_sizes
