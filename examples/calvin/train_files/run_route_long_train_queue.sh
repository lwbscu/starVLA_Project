#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export WANDB_MODE=${WANDB_MODE:-disabled}
export NO_ALBUMENTATIONS_UPDATE=${NO_ALBUMENTATIONS_UPDATE:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

LONG_RUN_TS=${LONG_RUN_TS:-$(date +"%Y%m%d_%H%M%S")}
LOG_ROOT=${LOG_ROOT:-logs/route_long_train/log_${LONG_RUN_TS}_qwen35_0p8b_p0_p3}
STAR_VLA_PYTHON=${STAR_VLA_PYTHON:-"$(conda info --base)/envs/starVLA_qwen35/bin/python"}

STEPS_P0=${STEPS_P0:-10000}
STEPS_P1=${STEPS_P1:-10000}
STEPS_P2=${STEPS_P2:-10000}
STEPS_P3=${STEPS_P3:-10000}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
EVAL_INTERVAL=${EVAL_INTERVAL:-1000000}

ACCELERATE_NO_OFFLOAD=${ACCELERATE_NO_OFFLOAD:-starVLA/config/deepseeds/deepspeed_zero2_route_validation.yaml}
ACCELERATE_CPU_OFFLOAD=${ACCELERATE_CPU_OFFLOAD:-starVLA/config/deepseeds/deepspeed_zero2_route_validation_cpu_offload.yaml}

mkdir -p "${LOG_ROOT}/terminal" "${LOG_ROOT}/metrics"

QUEUE_LOG="${LOG_ROOT}/terminal/queue.log"

log_msg() {
  echo "[$(date '+%F %T')] $*" | tee -a "${QUEUE_LOG}"
}

check_reload() {
  local route="$1"
  local run_id="$2"
  local steps="$3"
  local log_dir="$4"
  local ckpt_path="${log_dir}/checkpoints/${run_id}/checkpoints/steps_${steps}_pytorch_model.pt"

  if [[ ! -s "${ckpt_path}" ]]; then
    log_msg "ERROR: checkpoint missing or empty: ${ckpt_path}"
    return 1
  fi

  "${STAR_VLA_PYTHON}" examples/calvin/eval_files/check_checkpoint_reload.py \
    --ckpt-path "${ckpt_path}" \
    --expected-action-chunk-size 8 \
    --expected-unnorm-key franka \
    --output-json "${log_dir}/metrics/reload_check_steps_${steps}.json" \
    2>&1 | tee "${log_dir}/terminal/reload_check_steps_${steps}.log"

  log_msg "reload passed: ${route} ${ckpt_path}"
}

run_one() {
  local route="$1"
  local steps="$2"
  local accelerate_config="$3"
  local run_id="long_${route}_${steps}step"
  local log_dir="${LOG_ROOT}/${run_id}"

  if [[ -f "${log_dir}/metrics/reload_check_steps_${steps}.json" ]]; then
    log_msg "skip completed route: ${run_id}"
    return 0
  fi

  log_msg "start route=${route} steps=${steps} save_interval=${SAVE_INTERVAL} accelerate=${accelerate_config}"

  ROUTE="${route}" \
  MAX_TRAIN_STEPS="${steps}" \
  SAVE_INTERVAL="${SAVE_INTERVAL}" \
  EVAL_INTERVAL="${EVAL_INTERVAL}" \
  RUN_TS="${LONG_RUN_TS}" \
  RUN_ID="${run_id}" \
  LOG_DIR="${log_dir}" \
  ACCELERATE_CONFIG="${accelerate_config}" \
  STAR_VLA_PYTHON="${STAR_VLA_PYTHON}" \
  bash examples/calvin/train_files/run_route_validation_train.sh

  check_reload "${route}" "${run_id}" "${steps}" "${log_dir}"
  log_msg "finished route=${route} steps=${steps}"
}

log_msg "queue root: ${LOG_ROOT}"
log_msg "STAR_VLA_PYTHON=${STAR_VLA_PYTHON}"
log_msg "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
log_msg "routes: p0_oft=${STEPS_P0}, p1_adapter=${STEPS_P1}, p2_lora_oft=${STEPS_P2}, p3_lora_adapter=${STEPS_P3}"

run_one p0_oft "${STEPS_P0}" "${ACCELERATE_NO_OFFLOAD}"
run_one p1_adapter "${STEPS_P1}" "${ACCELERATE_CPU_OFFLOAD}"
run_one p2_lora_oft "${STEPS_P2}" "${ACCELERATE_NO_OFFLOAD}"
run_one p3_lora_adapter "${STEPS_P3}" "${ACCELERATE_CPU_OFFLOAD}"

log_msg "all route long-train jobs finished"
