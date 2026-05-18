#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
export CONDA_ROOT=${CONDA_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/26220216/miniconda3}
export STARVLA_ENV=${STARVLA_ENV:-starVLA_qwen35}
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
export DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-16}
export CONFIG_YAML=${CONFIG_YAML:-examples/calvin/train_files/starvla_train_calvin_qwen35_oft_h200.yaml}
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

mkdir -p "${LOG_ROOT}/terminal" "${LOG_ROOT}/summary"

select_largest_qwen35() {
  for candidate in \
    ./playground/Pretrained_models/Qwen3.5-9B \
    ./playground/Pretrained_models/Qwen3.5-4B \
    ./playground/Pretrained_models/Qwen3.5-2B \
    ./playground/Pretrained_models/Qwen3.5-0.8B
  do
    if [[ -f "${candidate}/config.json" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

if [[ -z "${BASE_VLM:-}" ]]; then
  if [[ "${ROUTE}" == "p4_qwen4b_pi" || "${ROUTE}" == "p4_pi" ]]; then
    BASE_VLM=./playground/Pretrained_models/Qwen3.5-4B
  else
    BASE_VLM=$(select_largest_qwen35) || {
      echo "No Qwen3.5 weight found under playground/Pretrained_models" >&2
      exit 2
    }
  fi
  export BASE_VLM
fi

if [[ ! -f "${BASE_VLM}/config.json" ]]; then
  echo "BASE_VLM does not contain config.json: ${BASE_VLM}" >&2
  exit 2
fi

if [[ ! -d "${H200_CALVIN_DATA_ROOT%/}/${H200_CALVIN_DATA_NAME}" ]]; then
  echo "CALVIN dataset not found: ${H200_CALVIN_DATA_ROOT%/}/${H200_CALVIN_DATA_NAME}" >&2
  exit 2
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
echo "BASE_VLM=${BASE_VLM}"
echo "OBS_IMAGE_SIZE=${OBS_IMAGE_SIZE}"
echo "LOG_ROOT=${LOG_ROOT}"

run_stage() {
  local stage_name=$1
  local steps=$2
  local save_interval=$3
  local workers=$4
  local run_id="${stage_name}_${ROUTE}_${steps}step"
  local log_dir="${LOG_ROOT}/${PIPELINE_TS}_${run_id}"
  local ckpt_path="${log_dir}/checkpoints/${run_id}/checkpoints/steps_${steps}_pytorch_model.pt"

  echo "===== START stage=${stage_name} route=${ROUTE} steps=${steps} ====="
  CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
  NUM_PROCESSES="${NUM_PROCESSES}" \
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

  echo "===== DONE stage=${stage_name} route=${ROUTE} ckpt=${ckpt_path} ====="
}

run_stage smoke1k 1000 1000 "${SMOKE_WORKERS:-8}"
run_stage fast10k 10000 5000 "${FAST_WORKERS:-${DATALOADER_NUM_WORKERS}}"
run_stage decision30k 30000 10000 "${DECISION_WORKERS:-${DATALOADER_NUM_WORKERS}}"

echo "PIPELINE_DONE route=${ROUTE} log=${PIPELINE_LOG}"
