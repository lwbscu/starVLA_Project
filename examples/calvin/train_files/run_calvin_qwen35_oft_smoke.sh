#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export WANDB_MODE=${WANDB_MODE:-disabled}
export NO_ALBUMENTATIONS_UPDATE=${NO_ALBUMENTATIONS_UPDATE:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

RUN_ID=${RUN_ID:-qwen35_oft_calvin_smoke}
RUN_TS=${RUN_TS:-$(date +"%Y%m%d_%H%M%S")}
LOG_DIR=${LOG_DIR:-logs/log_${RUN_TS}_${RUN_ID}}
CONFIG_YAML=${CONFIG_YAML:-examples/calvin/train_files/starvla_train_calvin_qwen35_oft_smoke.yaml}
STAR_VLA_PYTHON=${STAR_VLA_PYTHON:-"$(conda info --base)/envs/starVLA_qwen35/bin/python"}

mkdir -p "${LOG_DIR}"/{train,eval,terminal,mp4,configs,metrics,checkpoints}
cp "${CONFIG_YAML}" "${LOG_DIR}/configs/"
cp "$0" "${LOG_DIR}/configs/"

{
  echo "RUN_ID=${RUN_ID}"
  echo "LOG_DIR=${LOG_DIR}"
  echo "CONFIG_YAML=${CONFIG_YAML}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "STAR_VLA_PYTHON=${STAR_VLA_PYTHON}"

  "${STAR_VLA_PYTHON}" -m accelerate.commands.launch \
    --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --num_processes 1 \
    starVLA/training/train_starvla.py \
    --config_yaml "${CONFIG_YAML}" \
    --trainer.max_train_steps "${MAX_TRAIN_STEPS:-1}" \
    --trainer.save_interval "${SAVE_INTERVAL:-1}" \
    --trainer.eval_interval "${EVAL_INTERVAL:-1000}" \
    --run_root_dir "${LOG_DIR}/checkpoints" \
    --run_id "${RUN_ID}"
} 2>&1 | tee "${LOG_DIR}/terminal/train.log"
