#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export WANDB_MODE=${WANDB_MODE:-disabled}
export NO_ALBUMENTATIONS_UPDATE=${NO_ALBUMENTATIONS_UPDATE:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

ROUTE=${ROUTE:-p0_oft}
NUM_PROCESSES=${NUM_PROCESSES:-1}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-${ACCELERATE_MAIN_PROCESS_PORT:-${MASTER_PORT:-29500}}}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-1}
SAVE_INTERVAL=${SAVE_INTERVAL:-${MAX_TRAIN_STEPS}}
EVAL_INTERVAL=${EVAL_INTERVAL:-1000}
RUN_TS=${RUN_TS:-$(date +"%Y%m%d_%H%M%S")}
RUN_ID=${RUN_ID:-route_validation_${ROUTE}_${MAX_TRAIN_STEPS}step}
LOG_DIR=${LOG_DIR:-logs/route_validation/log_${RUN_TS}_${RUN_ID}}
CONFIG_YAML=${CONFIG_YAML:-examples/calvin/train_files/starvla_train_calvin_qwen35_oft_smoke.yaml}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2_route_validation.yaml}
STAR_VLA_PYTHON=${STAR_VLA_PYTHON:-"$(conda info --base)/envs/starVLA_qwen35/bin/python"}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-0}
BASE_VLM=${BASE_VLM:-./playground/Pretrained_models/Qwen3.5-9B}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
OBS_IMAGE_SIZE=${OBS_IMAGE_SIZE:-"[112,112]"}
ACTION_DIM=${ACTION_DIM:-7}
ACTION_HORIZON=${ACTION_HORIZON:-8}
ACTION_QUERY_NUM=${ACTION_QUERY_NUM:-64}
NUM_ACTIONS_CHUNK=${NUM_ACTIONS_CHUNK:-8}
ADAPTER_HIDDEN_DIM=${ADAPTER_HIDDEN_DIM:-1024}
LORA_R=${LORA_R:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
PI_NUM_INFERENCE_TIMESTEPS=${PI_NUM_INFERENCE_TIMESTEPS:-4}
PI_REPEATED_DIFFUSION_STEPS=${PI_REPEATED_DIFFUSION_STEPS:-2}
PI_NUM_TARGET_VISION_TOKENS=${PI_NUM_TARGET_VISION_TOKENS:-32}
CALVIN_DATA_ROOT=${CALVIN_DATA_ROOT:-}
CALVIN_DATA_MIX=${CALVIN_DATA_MIX:-}
CALVIN_DATA_NAME=${CALVIN_DATA_NAME:-}
H200_DEFAULT_CALVIN_DATA_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d
H200_DEFAULT_CALVIN_DATA_NAME=calvin_task_ABC_D
H200_DEFAULT_CALVIN_DATA_MIX=calvin_abc_d_h200
CALVIN_DATA_SOURCE=config_yaml

if [[ -z "${CALVIN_DATA_ROOT}" && -z "${CALVIN_DATA_MIX}" && -z "${CALVIN_DATA_NAME}" ]]; then
  H200_CANDIDATE_ROOT=${H200_CALVIN_DATA_ROOT:-${H200_DEFAULT_CALVIN_DATA_ROOT}}
  H200_CANDIDATE_NAME=${H200_CALVIN_DATA_NAME:-${H200_DEFAULT_CALVIN_DATA_NAME}}
  H200_CANDIDATE_MIX=${H200_CALVIN_DATA_MIX:-${H200_DEFAULT_CALVIN_DATA_MIX}}
  if [[ -d "${H200_CANDIDATE_ROOT%/}/${H200_CANDIDATE_NAME}" ]]; then
    CALVIN_DATA_ROOT="${H200_CANDIDATE_ROOT}"
    CALVIN_DATA_NAME="${H200_CANDIDATE_NAME}"
    CALVIN_DATA_MIX="${H200_CANDIDATE_MIX}"
    CALVIN_DATA_SOURCE=auto_h200
  fi
fi

if ! [[ "${NUM_PROCESSES}" =~ ^[0-9]+$ ]] || (( NUM_PROCESSES < 1 )); then
  echo "NUM_PROCESSES must be a positive integer, got: ${NUM_PROCESSES}" >&2
  exit 2
fi

if ! [[ "${MAIN_PROCESS_PORT}" =~ ^[0-9]+$ ]] || (( MAIN_PROCESS_PORT < 1 || MAIN_PROCESS_PORT > 65535 )); then
  echo "MAIN_PROCESS_PORT must be an integer in [1, 65535], got: ${MAIN_PROCESS_PORT}" >&2
  exit 2
fi

if [[ ! -x "${STAR_VLA_PYTHON}" ]]; then
  echo "STAR_VLA_PYTHON is not executable: ${STAR_VLA_PYTHON}" >&2
  exit 2
fi

if [[ ! -f "${BASE_VLM}/config.json" ]]; then
  echo "BASE_VLM does not contain config.json: ${BASE_VLM}" >&2
  exit 2
fi

if [[ -n "${CALVIN_DATA_ROOT}" || -n "${CALVIN_DATA_MIX}" || -n "${CALVIN_DATA_NAME}" ]]; then
  : "${CALVIN_DATA_ROOT:?Set CALVIN_DATA_ROOT when overriding CALVIN data}"
  : "${CALVIN_DATA_MIX:?Set CALVIN_DATA_MIX when overriding CALVIN data}"
  : "${CALVIN_DATA_NAME:?Set CALVIN_DATA_NAME when overriding CALVIN data}"
  CALVIN_DATASET_DIR="${CALVIN_DATA_ROOT%/}/${CALVIN_DATA_NAME}"
  if [[ ! -d "${CALVIN_DATASET_DIR}" ]]; then
    echo "CALVIN LeRobot dataset directory not found: ${CALVIN_DATASET_DIR}" >&2
    exit 2
  fi
  if [[ ! -f "${CALVIN_DATASET_DIR}/meta/info.json" ]]; then
    echo "Missing ${CALVIN_DATASET_DIR}/meta/info.json. Training requires LeRobot-format CALVIN data, not raw CALVIN." >&2
    exit 2
  fi
  if [[ ! -f "${CALVIN_DATASET_DIR}/meta/modality.json" ]]; then
    echo "Missing ${CALVIN_DATASET_DIR}/meta/modality.json. Copy examples/calvin/train_files/modality.json into the dataset meta directory." >&2
    exit 2
  fi
  if [[ ! -d "${CALVIN_DATASET_DIR}/data" ]]; then
    echo "Missing ${CALVIN_DATASET_DIR}/data. Training requires LeRobot parquet data." >&2
    exit 2
  fi
fi

mkdir -p "${LOG_DIR}"/{train,eval,terminal,mp4,configs,metrics,checkpoints}
cp "${CONFIG_YAML}" "${LOG_DIR}/configs/"
cp "$0" "${LOG_DIR}/configs/"
cp "${ACCELERATE_CONFIG}" "${LOG_DIR}/configs/"

COMMON_ARGS=(
  --config_file "${ACCELERATE_CONFIG}"
  --num_processes "${NUM_PROCESSES}"
  --main_process_port "${MAIN_PROCESS_PORT}"
  starVLA/training/train_starvla.py
  --config_yaml "${CONFIG_YAML}"
  --framework.qwenvl.base_vlm "${BASE_VLM}"
  --framework.qwenvl.attn_implementation "${ATTN_IMPLEMENTATION}"
  --datasets.vla_data.obs_image_size "${OBS_IMAGE_SIZE}"
  --datasets.vla_data.num_workers "${DATALOADER_NUM_WORKERS}"
  --trainer.max_train_steps "${MAX_TRAIN_STEPS}"
  --trainer.save_interval "${SAVE_INTERVAL}"
  --trainer.eval_interval "${EVAL_INTERVAL}"
  --run_root_dir "${LOG_DIR}/checkpoints"
  --run_id "${RUN_ID}"
)

if [[ -n "${CALVIN_DATA_ROOT}" ]]; then
  COMMON_ARGS+=(
    --datasets.vla_data.data_root_dir "${CALVIN_DATA_ROOT}"
    --datasets.vla_data.data_mix "${CALVIN_DATA_MIX}"
  )
fi

case "${ROUTE}" in
  p0_oft)
    ROUTE_ARGS=(
      --framework.name QwenOFT
      --framework.action_model.action_model_type MLP
      --framework.action_model.action_dim "${ACTION_DIM}"
      --framework.action_model.action_horizon "${ACTION_HORIZON}"
      --trainer.freeze_modules qwen_vl_interface
    )
    ;;
  p1_adapter)
    ROUTE_ARGS=(
      --framework.name QwenAdapter
      --framework.action_model.action_model_type VLA_Adapter
      --framework.action_model.action_query_num "${ACTION_QUERY_NUM}"
      --framework.action_model.num_actions_chunk "${NUM_ACTIONS_CHUNK}"
      --framework.action_model.action_dim "${ACTION_DIM}"
      --framework.action_model.hidden_dim "${ADAPTER_HIDDEN_DIM}"
      --trainer.freeze_modules qwen_vl_interface
    )
    ;;
  p2_lora_oft)
    ROUTE_ARGS=(
      --framework.name QwenOFT
      --framework.action_model.action_model_type MLP
      --framework.action_model.action_dim "${ACTION_DIM}"
      --framework.action_model.action_horizon "${ACTION_HORIZON}"
      --trainer.freeze_modules qwen_vl_interface
      --trainer.lora.enabled true
      --trainer.lora.r "${LORA_R}"
      --trainer.lora.alpha "${LORA_ALPHA}"
      --trainer.lora.dropout "${LORA_DROPOUT}"
    )
    ;;
  p3_lora_adapter)
    ROUTE_ARGS=(
      --framework.name QwenAdapter
      --framework.action_model.action_model_type VLA_Adapter
      --framework.action_model.action_query_num "${ACTION_QUERY_NUM}"
      --framework.action_model.num_actions_chunk "${NUM_ACTIONS_CHUNK}"
      --framework.action_model.action_dim "${ACTION_DIM}"
      --framework.action_model.hidden_dim "${ADAPTER_HIDDEN_DIM}"
      --trainer.freeze_modules qwen_vl_interface
      --trainer.lora.enabled true
      --trainer.lora.r "${LORA_R}"
      --trainer.lora.alpha "${LORA_ALPHA}"
      --trainer.lora.dropout "${LORA_DROPOUT}"
    )
    ;;
  p4_pi)
    ROUTE_ARGS=(
      --framework.name QwenPI
      --framework.action_model.action_model_type LayerwiseFM
      --framework.action_model.action_dim "${ACTION_DIM}"
      --framework.action_model.state_dim "${ACTION_DIM}"
      --framework.action_model.action_horizon "${ACTION_HORIZON}"
      --framework.action_model.repeated_diffusion_steps "${PI_REPEATED_DIFFUSION_STEPS}"
      --framework.action_model.num_inference_timesteps "${PI_NUM_INFERENCE_TIMESTEPS}"
      --framework.action_model.num_target_vision_tokens "${PI_NUM_TARGET_VISION_TOKENS}"
      --trainer.freeze_modules qwen_vl_interface
    )
    ;;
  *)
    echo "Unknown ROUTE=${ROUTE}. Use p0_oft, p1_adapter, p2_lora_oft, p3_lora_adapter, or p4_pi." >&2
    exit 2
    ;;
esac

set +e
{
  echo "ROUTE=${ROUTE}"
  echo "RUN_ID=${RUN_ID}"
  echo "LOG_DIR=${LOG_DIR}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "NUM_PROCESSES=${NUM_PROCESSES}"
  echo "MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT}"
  echo "MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS}"
  echo "SAVE_INTERVAL=${SAVE_INTERVAL}"
  echo "ACCELERATE_CONFIG=${ACCELERATE_CONFIG}"
  echo "STAR_VLA_PYTHON=${STAR_VLA_PYTHON}"
  echo "DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS}"
  echo "BASE_VLM=${BASE_VLM}"
  echo "ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION}"
  echo "OBS_IMAGE_SIZE=${OBS_IMAGE_SIZE}"
  echo "ACTION_DIM=${ACTION_DIM}"
  echo "ACTION_HORIZON=${ACTION_HORIZON}"
  echo "ACTION_QUERY_NUM=${ACTION_QUERY_NUM}"
  echo "NUM_ACTIONS_CHUNK=${NUM_ACTIONS_CHUNK}"
  echo "ADAPTER_HIDDEN_DIM=${ADAPTER_HIDDEN_DIM}"
  echo "LORA_R=${LORA_R}"
  echo "LORA_ALPHA=${LORA_ALPHA}"
  echo "LORA_DROPOUT=${LORA_DROPOUT}"
  echo "PI_NUM_INFERENCE_TIMESTEPS=${PI_NUM_INFERENCE_TIMESTEPS}"
  echo "PI_REPEATED_DIFFUSION_STEPS=${PI_REPEATED_DIFFUSION_STEPS}"
  echo "PI_NUM_TARGET_VISION_TOKENS=${PI_NUM_TARGET_VISION_TOKENS}"
  echo "CALVIN_DATA_SOURCE=${CALVIN_DATA_SOURCE}"
  echo "CALVIN_DATA_ROOT=${CALVIN_DATA_ROOT:-<config_yaml>}"
  echo "CALVIN_DATA_MIX=${CALVIN_DATA_MIX:-<config_yaml>}"
  echo "CALVIN_DATA_NAME=${CALVIN_DATA_NAME:-<registry>}"

  if ! "${STAR_VLA_PYTHON}" - <<'PY'
import transformers
print(f"transformers={transformers.__version__}")
from transformers import Qwen3_5ForConditionalGeneration
print("Qwen3.5 import OK")
PY
  then
    echo "Qwen3.5 import failed. Install a Qwen3.5-compatible transformers version in STAR_VLA_PYTHON's conda environment before training." >&2
    exit 2
  fi

  if ! MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT}" "${STAR_VLA_PYTHON}" - <<'PY'
import os
import socket
import sys

port = int(os.environ["MAIN_PROCESS_PORT"])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(("0.0.0.0", port))
except OSError as exc:
    print(
        f"MAIN_PROCESS_PORT={port} is unavailable before accelerate launch: {exc}",
        file=sys.stderr,
    )
    sys.exit(1)
finally:
    sock.close()
print(f"MAIN_PROCESS_PORT={port} is available")
PY
  then
    echo "Choose a free MAIN_PROCESS_PORT for this training route." >&2
    exit 2
  fi

  "${STAR_VLA_PYTHON}" -m accelerate.commands.launch "${COMMON_ARGS[@]}" "${ROUTE_ARGS[@]}"
} 2>&1 | tee "${LOG_DIR}/terminal/train.log"
train_status=${PIPESTATUS[0]}
set -e

if [[ "${train_status}" -ne 0 ]]; then
  echo "Training failed with status ${train_status}. See ${LOG_DIR}/terminal/train.log" >&2
  exit "${train_status}"
fi

"${STAR_VLA_PYTHON}" examples/calvin/train_files/check_training_loss.py \
  --train-log "${LOG_DIR}/terminal/train.log" \
  --output-json "${LOG_DIR}/metrics/loss_check.json" \
  2>&1 | tee "${LOG_DIR}/terminal/loss_check.log"
