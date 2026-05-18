#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export WANDB_MODE=${WANDB_MODE:-disabled}
export NO_ALBUMENTATIONS_UPDATE=${NO_ALBUMENTATIONS_UPDATE:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

H200_RUN_TS=${H200_RUN_TS:-$(date +"%Y%m%d_%H%M%S")}
LOG_ROOT=${LOG_ROOT:-logs/h200_route_train/log_${H200_RUN_TS}_qwen35_0p8b_matrix}
STAR_VLA_PYTHON=${STAR_VLA_PYTHON:-"$(conda info --base)/envs/starVLA_qwen35/bin/python"}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-0}

MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-30000}
SAVE_INTERVAL=${SAVE_INTERVAL:-5000}
EVAL_INTERVAL=${EVAL_INTERVAL:-1000000}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2_route_validation.yaml}

ROUTE_LIST=${ROUTE_LIST:-"p0_oft p1_adapter p2_lora_oft p3_lora_adapter"}
GPU_LIST=${GPU_LIST:-"0 1 2 3"}
FAIL_FAST=${FAIL_FAST:-0}

mkdir -p "${LOG_ROOT}/terminal" "${LOG_ROOT}/metrics"
QUEUE_LOG="${LOG_ROOT}/terminal/h200_matrix.log"

log_msg() {
  echo "[$(date '+%F %T')] $*" | tee -a "${QUEUE_LOG}"
}

read -r -a ROUTES <<< "${ROUTE_LIST}"
read -r -a GPUS <<< "${GPU_LIST}"

if (( ${#ROUTES[@]} == 0 )); then
  echo "ROUTE_LIST is empty" >&2
  exit 2
fi

if (( ${#GPUS[@]} < ${#ROUTES[@]} )); then
  echo "GPU_LIST has ${#GPUS[@]} GPUs but ROUTE_LIST has ${#ROUTES[@]} routes" >&2
  exit 2
fi

log_msg "LOG_ROOT=${LOG_ROOT}"
log_msg "STAR_VLA_PYTHON=${STAR_VLA_PYTHON}"
log_msg "MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS}"
log_msg "SAVE_INTERVAL=${SAVE_INTERVAL}"
log_msg "ACCELERATE_CONFIG=${ACCELERATE_CONFIG}"
log_msg "DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS}"
log_msg "ROUTE_LIST=${ROUTE_LIST}"
log_msg "GPU_LIST=${GPU_LIST}"

declare -a PIDS=()
declare -a NAMES=()
declare -a LOG_DIRS=()

for idx in "${!ROUTES[@]}"; do
  route="${ROUTES[$idx]}"
  gpu="${GPUS[$idx]}"
  run_id="h200_${route}_${MAX_TRAIN_STEPS}step"
  log_dir="${LOG_ROOT}/${run_id}"
  runner_log="${log_dir}/terminal/h200_runner.log"
  mkdir -p "${log_dir}/terminal" "${log_dir}/metrics"

  log_msg "launch route=${route} gpu=${gpu} run_id=${run_id}"

  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="${gpu}"
    ROUTE="${route}" \
    MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS}" \
    SAVE_INTERVAL="${SAVE_INTERVAL}" \
    EVAL_INTERVAL="${EVAL_INTERVAL}" \
    RUN_TS="${H200_RUN_TS}" \
    RUN_ID="${run_id}" \
    LOG_DIR="${log_dir}" \
    ACCELERATE_CONFIG="${ACCELERATE_CONFIG}" \
    STAR_VLA_PYTHON="${STAR_VLA_PYTHON}" \
    DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS}" \
    bash examples/calvin/train_files/run_route_validation_train.sh

    ckpt_path="${log_dir}/checkpoints/${run_id}/checkpoints/steps_${MAX_TRAIN_STEPS}_pytorch_model.pt"
    "${STAR_VLA_PYTHON}" examples/calvin/eval_files/check_checkpoint_reload.py \
      --ckpt-path "${ckpt_path}" \
      --expected-action-chunk-size 8 \
      --expected-unnorm-key franka \
      --output-json "${log_dir}/metrics/reload_check_steps_${MAX_TRAIN_STEPS}.json"
  ) > "${runner_log}" 2>&1 &

  PIDS+=("$!")
  NAMES+=("${route}")
  LOG_DIRS+=("${log_dir}")
done

status=0
for idx in "${!PIDS[@]}"; do
  pid="${PIDS[$idx]}"
  route="${NAMES[$idx]}"
  log_dir="${LOG_DIRS[$idx]}"
  if wait "${pid}"; then
    log_msg "finished route=${route} log_dir=${log_dir}"
  else
    route_status=$?
    log_msg "FAILED route=${route} status=${route_status} log_dir=${log_dir}"
    status=1
    if [[ "${FAIL_FAST}" == "1" ]]; then
      log_msg "FAIL_FAST=1, stopping remaining jobs"
      for remaining in "${PIDS[@]}"; do
        kill "${remaining}" 2>/dev/null || true
      done
      exit "${status}"
    fi
  fi
done

if [[ "${status}" -ne 0 ]]; then
  log_msg "one or more H200 route jobs failed"
  exit "${status}"
fi

log_msg "all H200 route jobs finished"
