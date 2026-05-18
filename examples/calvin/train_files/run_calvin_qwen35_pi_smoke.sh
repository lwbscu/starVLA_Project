#!/usr/bin/env bash
# CALVIN smoke train: Qwen3.5-VL + QwenPI (LayerwiseFM action head).
# Mirror of run_calvin_qwen35_oft_smoke.sh; framework fields match
# run_route_validation_train.sh ROUTE=p4_qwen4b_pi / QwenPIDefaultConfig.
set -euo pipefail

cd "$(dirname "$0")/../../.."

export WANDB_MODE=${WANDB_MODE:-disabled}
export NO_ALBUMENTATIONS_UPDATE=${NO_ALBUMENTATIONS_UPDATE:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

RUN_ID=${RUN_ID:-qwen35_pi_calvin_smoke}
RUN_TS=${RUN_TS:-$(date +"%Y%m%d_%H%M%S")}
LOG_DIR=${LOG_DIR:-logs/log_${RUN_TS}_${RUN_ID}}
CONFIG_YAML=${CONFIG_YAML:-examples/calvin/train_files/starvla_train_calvin_qwen35_pi_smoke.yaml}
STAR_VLA_PYTHON=${STAR_VLA_PYTHON:-"$(conda info --base)/envs/starVLA_qwen35/bin/python"}
BASE_VLM=${BASE_VLM:-}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-}
PI_NUM_INFERENCE_TIMESTEPS=${PI_NUM_INFERENCE_TIMESTEPS:-}
PI_REPEATED_DIFFUSION_STEPS=${PI_REPEATED_DIFFUSION_STEPS:-}
PI_NUM_TARGET_VISION_TOKENS=${PI_NUM_TARGET_VISION_TOKENS:-}

if [[ ! -x "${STAR_VLA_PYTHON}" ]]; then
  echo "STAR_VLA_PYTHON is not executable: ${STAR_VLA_PYTHON}" >&2
  exit 2
fi

if [[ ! -f "${CONFIG_YAML}" ]]; then
  echo "CONFIG_YAML not found: ${CONFIG_YAML}" >&2
  exit 2
fi

if [[ -n "${BASE_VLM}" && ! -f "${BASE_VLM}/config.json" ]]; then
  echo "BASE_VLM does not contain config.json: ${BASE_VLM}" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"/{train,eval,terminal,mp4,configs,metrics,checkpoints}
cp "${CONFIG_YAML}" "${LOG_DIR}/configs/"
cp "$0" "${LOG_DIR}/configs/"

CLI_ARGS=(
  --config_yaml "${CONFIG_YAML}"
  --trainer.max_train_steps "${MAX_TRAIN_STEPS:-1}"
  --trainer.save_interval "${SAVE_INTERVAL:-1}"
  --trainer.eval_interval "${EVAL_INTERVAL:-1000}"
  --run_root_dir "${LOG_DIR}/checkpoints"
  --run_id "${RUN_ID}"
)

if [[ -n "${BASE_VLM}" ]]; then
  CLI_ARGS+=(--framework.qwenvl.base_vlm "${BASE_VLM}")
fi
if [[ -n "${ATTN_IMPLEMENTATION}" ]]; then
  CLI_ARGS+=(--framework.qwenvl.attn_implementation "${ATTN_IMPLEMENTATION}")
fi
if [[ -n "${PI_NUM_INFERENCE_TIMESTEPS}" ]]; then
  CLI_ARGS+=(--framework.action_model.num_inference_timesteps "${PI_NUM_INFERENCE_TIMESTEPS}")
fi
if [[ -n "${PI_REPEATED_DIFFUSION_STEPS}" ]]; then
  CLI_ARGS+=(--framework.action_model.repeated_diffusion_steps "${PI_REPEATED_DIFFUSION_STEPS}")
fi
if [[ -n "${PI_NUM_TARGET_VISION_TOKENS}" ]]; then
  CLI_ARGS+=(--framework.action_model.num_target_vision_tokens "${PI_NUM_TARGET_VISION_TOKENS}")
fi

{
  echo "RUN_ID=${RUN_ID}"
  echo "LOG_DIR=${LOG_DIR}"
  echo "CONFIG_YAML=${CONFIG_YAML}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "STAR_VLA_PYTHON=${STAR_VLA_PYTHON}"
  echo "BASE_VLM=${BASE_VLM:-<from yaml>}"
  echo "PI_NUM_INFERENCE_TIMESTEPS=${PI_NUM_INFERENCE_TIMESTEPS:-<from yaml>}"
  echo "PI_REPEATED_DIFFUSION_STEPS=${PI_REPEATED_DIFFUSION_STEPS:-<from yaml>}"
  echo "PI_NUM_TARGET_VISION_TOKENS=${PI_NUM_TARGET_VISION_TOKENS:-<from yaml>}"

  "${STAR_VLA_PYTHON}" -m accelerate.commands.launch \
    --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --num_processes "${NUM_PROCESSES:-1}" \
    starVLA/training/train_starvla.py \
    "${CLI_ARGS[@]}"
} 2>&1 | tee "${LOG_DIR}/terminal/train.log"
