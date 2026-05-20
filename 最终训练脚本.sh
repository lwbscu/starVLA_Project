#!/usr/bin/env bash
# Final BC stage: Qwen3.5-4B + PI action head + semantic LoRA, 8 H200 GPUs.
set -Eeuo pipefail
trap 'echo "[final-train] FAILED at line ${LINENO}" >&2' ERR

PROJECT_ROOT=${PROJECT_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project}
CONDA_ROOT=${CONDA_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/26220216/miniconda3}
STAR_VLA_ENV=${STAR_VLA_ENV:-starVLA_qwen35}
STAR_VLA_PYTHON=${STAR_VLA_PYTHON:-${CONDA_ROOT}/envs/${STAR_VLA_ENV}/bin/python}

cd "${PROJECT_ROOT}"
export PATH="${CONDA_ROOT}/bin:${PATH}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${STAR_VLA_ENV}"

PUBLIC_QWEN35_4B=${PUBLIC_QWEN35_4B:-/inspire/qb-ilm2/project/26summer-camp-10/public/Qwen/Qwen3.5-4B}
PUBLIC_CALVIN_ROOT=${PUBLIC_CALVIN_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d}
PUBLIC_CALVIN_NAME=${PUBLIC_CALVIN_NAME:-calvin_task_ABC_D}
PUBLIC_CALVIN_MIX=${PUBLIC_CALVIN_MIX:-calvin_abc_d_h200}
SOURCE_PI_STATE_CKPT_4B=${SOURCE_PI_STATE_CKPT_4B:-${PROJECT_ROOT}/logs/20260519_h200_server1_pi_state_30k_v1/server1_pi_state/qwen35_4b/pi_state_qwen35_4b_30000step/checkpoints/pi_state_qwen35_4b_30000step/checkpoints/steps_30000_pytorch_model.pt}

RUN_ID=${RUN_ID:-pi_state_lora_sem_qwen35_4b_8gpu_b16_20000step}
LOG_DIR=${LOG_DIR:-${PROJECT_ROOT}/logs/final_runs/${RUN_ID}}
ALLOW_EXISTING_LOG_DIR=${ALLOW_EXISTING_LOG_DIR:-false}

if [[ -e "${LOG_DIR}" && "${ALLOW_EXISTING_LOG_DIR}" != "true" ]]; then
  echo "[final-train] ERROR: LOG_DIR already exists: ${LOG_DIR}" >&2
  echo "[final-train] Use a new RUN_ID or set ALLOW_EXISTING_LOG_DIR=true only when resuming intentionally." >&2
  exit 2
fi

test -x "${STAR_VLA_PYTHON}"
test -f "${PUBLIC_QWEN35_4B}/config.json"
test -f "${SOURCE_PI_STATE_CKPT_4B}"
test -d "${PUBLIC_CALVIN_ROOT}/${PUBLIC_CALVIN_NAME}/data"
test -f "${PUBLIC_CALVIN_ROOT}/${PUBLIC_CALVIN_NAME}/meta/info.json"
test -f "${PUBLIC_CALVIN_ROOT}/${PUBLIC_CALVIN_NAME}/meta/modality.json"
test -f "${PROJECT_ROOT}/examples/calvin/train_files/run_route_validation_train.sh"
test -f "${PROJECT_ROOT}/examples/calvin/train_files/starvla_train_calvin_qwen35_pi_h200.yaml"

export STAR_VLA_PYTHON
export WANDB_MODE=${WANDB_MODE:-offline}
export NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

export ROUTE=p4_pi
export NUM_PROCESSES=${NUM_PROCESSES:-8}
export MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29620}
export MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-20000}
export SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
export EVAL_INTERVAL=${EVAL_INTERVAL:-1000000}
export CHECKPOINT_KEEP_LATEST=${CHECKPOINT_KEEP_LATEST:-1}
export CHECKPOINT_KEEP_STEPS=${CHECKPOINT_KEEP_STEPS:-1000,5000,10000,15000,20000}
export LOGGING_FREQUENCY=${LOGGING_FREQUENCY:-10}
export DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-12}
export PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-16}
export USE_TENSORBOARD=true
export TENSORBOARD_LOG_DIR="${LOG_DIR}/tensorboard"

export CONFIG_YAML=examples/calvin/train_files/starvla_train_calvin_qwen35_pi_h200.yaml
export ACCELERATE_CONFIG=starVLA/config/deepseeds/deepspeed_zero2_route_validation.yaml
export BASE_VLM="${PUBLIC_QWEN35_4B}"
export ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
export OBS_IMAGE_SIZE=${OBS_IMAGE_SIZE:-"[224,224]"}
export ACTION_DIM=7
export ACTION_HORIZON=8
export INCLUDE_STATE=true
export STATE_DIM=8
export PI_STATE_DIM=8
export PI_NUM_INFERENCE_TIMESTEPS=${PI_NUM_INFERENCE_TIMESTEPS:-4}
export PI_REPEATED_DIFFUSION_STEPS=${PI_REPEATED_DIFFUSION_STEPS:-2}
export PI_NUM_TARGET_VISION_TOKENS=${PI_NUM_TARGET_VISION_TOKENS:-32}

export PRETRAINED_CHECKPOINT="${SOURCE_PI_STATE_CKPT_4B}"
export RELOAD_MODULES=action_model
export FREEZE_MODULES=qwen_vl_interface,action_model

export LORA_ENABLED=true
export LORA_R=${LORA_R:-32}
export LORA_ALPHA=${LORA_ALPHA:-64}
export LORA_DROPOUT=${LORA_DROPOUT:-0.05}
export LORA_FAIL_IF_NO_TARGET_MODULES=true
export LORA_TARGET_INCLUDE_PREFIXES=model
export LORA_TARGET_EXCLUDE_PREFIXES=model.visual
export LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj

export CALVIN_DATA_ROOT="${PUBLIC_CALVIN_ROOT}"
export CALVIN_DATA_NAME="${PUBLIC_CALVIN_NAME}"
export CALVIN_DATA_MIX="${PUBLIC_CALVIN_MIX}"
export H200_CALVIN_DATA_ROOT="${PUBLIC_CALVIN_ROOT}"
export H200_CALVIN_DATA_NAME="${PUBLIC_CALVIN_NAME}"
export H200_CALVIN_DATA_MIX="${PUBLIC_CALVIN_MIX}"

echo "[final-train] RUN_ID=${RUN_ID}"
echo "[final-train] LOG_DIR=${LOG_DIR}"
echo "[final-train] BASE_VLM=${BASE_VLM}"
echo "[final-train] CALVIN=${CALVIN_DATA_ROOT}/${CALVIN_DATA_NAME}"
echo "[final-train] SOURCE_PI_STATE_CKPT_4B=${SOURCE_PI_STATE_CKPT_4B}"

bash examples/calvin/train_files/run_route_validation_train.sh

FINAL_TRAIN_CKPT="${LOG_DIR}/checkpoints/${RUN_ID}/checkpoints/steps_${MAX_TRAIN_STEPS}_pytorch_model.pt"
test -f "${FINAL_TRAIN_CKPT}"

echo "[final-train] OK"
echo "FINAL_TRAIN_CKPT=${FINAL_TRAIN_CKPT}"
