#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export WANDB_MODE=${WANDB_MODE:-disabled}
export NO_ALBUMENTATIONS_UPDATE=${NO_ALBUMENTATIONS_UPDATE:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

ROUTE=${ROUTE:-p0_oft}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-1}
SAVE_INTERVAL=${SAVE_INTERVAL:-${MAX_TRAIN_STEPS}}
EVAL_INTERVAL=${EVAL_INTERVAL:-1000}
RUN_TS=${RUN_TS:-$(date +"%Y%m%d_%H%M%S")}
RUN_ID=${RUN_ID:-route_validation_${ROUTE}_${MAX_TRAIN_STEPS}step}
LOG_DIR=${LOG_DIR:-logs/route_validation/log_${RUN_TS}_${RUN_ID}}
CONFIG_YAML=${CONFIG_YAML:-examples/calvin/train_files/starvla_train_calvin_qwen35_oft_smoke.yaml}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2_route_validation.yaml}
STAR_VLA_PYTHON=${STAR_VLA_PYTHON:-"$(conda info --base)/envs/starVLA_qwen35/bin/python"}

mkdir -p "${LOG_DIR}"/{train,eval,terminal,mp4,configs,metrics,checkpoints}
cp "${CONFIG_YAML}" "${LOG_DIR}/configs/"
cp "$0" "${LOG_DIR}/configs/"
cp "${ACCELERATE_CONFIG}" "${LOG_DIR}/configs/"

COMMON_ARGS=(
  --config_file "${ACCELERATE_CONFIG}"
  --num_processes 1
  starVLA/training/train_starvla.py
  --config_yaml "${CONFIG_YAML}"
  --framework.qwenvl.base_vlm ./playground/Pretrained_models/Qwen3.5-0.8B
  --framework.qwenvl.attn_implementation sdpa
  --datasets.vla_data.obs_image_size "[112,112]"
  --trainer.max_train_steps "${MAX_TRAIN_STEPS}"
  --trainer.save_interval "${SAVE_INTERVAL}"
  --trainer.eval_interval "${EVAL_INTERVAL}"
  --run_root_dir "${LOG_DIR}/checkpoints"
  --run_id "${RUN_ID}"
)

case "${ROUTE}" in
  p0_oft)
    ROUTE_ARGS=(
      --framework.name QwenOFT
      --framework.action_model.action_model_type MLP
      --framework.action_model.action_dim 7
      --framework.action_model.action_horizon 8
      --trainer.freeze_modules qwen_vl_interface
    )
    ;;
  p1_adapter)
    ROUTE_ARGS=(
      --framework.name QwenAdapter
      --framework.action_model.action_model_type VLA_Adapter
      --framework.action_model.action_query_num 64
      --framework.action_model.num_actions_chunk 8
      --framework.action_model.action_dim 7
      --framework.action_model.hidden_dim 1024
      --trainer.freeze_modules qwen_vl_interface
    )
    ;;
  p2_lora_oft)
    ROUTE_ARGS=(
      --framework.name QwenOFT
      --framework.action_model.action_model_type MLP
      --framework.action_model.action_dim 7
      --framework.action_model.action_horizon 8
      --trainer.freeze_modules qwen_vl_interface
      --trainer.lora.enabled true
      --trainer.lora.r 16
      --trainer.lora.alpha 32
      --trainer.lora.dropout 0.05
    )
    ;;
  p3_lora_adapter)
    ROUTE_ARGS=(
      --framework.name QwenAdapter
      --framework.action_model.action_model_type VLA_Adapter
      --framework.action_model.action_query_num 64
      --framework.action_model.num_actions_chunk 8
      --framework.action_model.action_dim 7
      --framework.action_model.hidden_dim 1024
      --trainer.freeze_modules qwen_vl_interface
      --trainer.lora.enabled true
      --trainer.lora.r 16
      --trainer.lora.alpha 32
      --trainer.lora.dropout 0.05
    )
    ;;
  *)
    echo "Unknown ROUTE=${ROUTE}. Use p0_oft, p1_adapter, p2_lora_oft, or p3_lora_adapter." >&2
    exit 2
    ;;
esac

set +e
{
  echo "ROUTE=${ROUTE}"
  echo "RUN_ID=${RUN_ID}"
  echo "LOG_DIR=${LOG_DIR}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS}"
  echo "SAVE_INTERVAL=${SAVE_INTERVAL}"
  echo "ACCELERATE_CONFIG=${ACCELERATE_CONFIG}"
  echo "STAR_VLA_PYTHON=${STAR_VLA_PYTHON}"

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
