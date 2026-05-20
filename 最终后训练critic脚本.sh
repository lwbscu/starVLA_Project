#!/usr/bin/env bash
# Final AWAC phase 1: train Q critic from the PI+LoRA BC checkpoint.
set -Eeuo pipefail
trap 'echo "[final-awac-critic] FAILED at line ${LINENO}" >&2' ERR

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
TRAIN_RUN_ID=${TRAIN_RUN_ID:-pi_state_lora_sem_qwen35_4b_8gpu_b16_20000step}

bc_checkpoint=${BC_CHECKPOINT:-${bc_checkpoint:-${PROJECT_ROOT}/logs/final_runs/${TRAIN_RUN_ID}/checkpoints/${TRAIN_RUN_ID}/checkpoints/steps_20000_pytorch_model.pt}}
run_root_dir=${RUN_ROOT_DIR:-${run_root_dir:-${PROJECT_ROOT}/logs/final_awac_pi_lora_sem_onfly}}
run_id=${RUN_ID:-${run_id:-awac_critic_pi_lora_sem_qwen35_4b_bc20k_onfly}}
ALLOW_EXISTING_RUN_DIR=${ALLOW_EXISTING_RUN_DIR:-false}

if [[ -e "${run_root_dir}/${run_id}" && "${ALLOW_EXISTING_RUN_DIR}" != "true" ]]; then
  echo "[final-awac-critic] ERROR: run directory already exists: ${run_root_dir}/${run_id}" >&2
  echo "[final-awac-critic] Use a new RUN_ID or set ALLOW_EXISTING_RUN_DIR=true only when resuming intentionally." >&2
  exit 2
fi

test -x "${STAR_VLA_PYTHON}"
test -f "${PUBLIC_QWEN35_4B}/config.json"
test -f "${bc_checkpoint}"
test -d "${PUBLIC_CALVIN_ROOT}/${PUBLIC_CALVIN_NAME}/data"
test -f "${PUBLIC_CALVIN_ROOT}/${PUBLIC_CALVIN_NAME}/meta/info.json"
test -f "${PUBLIC_CALVIN_ROOT}/${PUBLIC_CALVIN_NAME}/meta/modality.json"
test -f "${PROJECT_ROOT}/examples/calvin/train_files/run_calvin_awac_critic.sh"

export STAR_VLA_PYTHON
export WANDB_MODE=${WANDB_MODE:-offline}
export NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export AWAC_NUM_GPUS=${AWAC_NUM_GPUS:-8}
export num_processes=${num_processes:-8}
export main_process_port=${main_process_port:-29630}

export Framework_name=QwenPI
export base_vlm="${PUBLIC_QWEN35_4B}"
export calvin_data_root="${PUBLIC_CALVIN_ROOT}"
export calvin_dataset_name="${PUBLIC_CALVIN_NAME}"
export data_mix="${PUBLIC_CALVIN_MIX}"
export include_state=true
export state_dim=8
export compute_rewards_on_the_fly=true
export success_column=success
export assume_success_if_missing=false
export step_penalty=-1.0
export success_reward=0.0
export failure_reward=-3000.0
export critic_max_train_steps=${critic_max_train_steps:-50000}
export save_interval=${save_interval:-5000}
export per_device_batch_size=${per_device_batch_size:-16}
export run_root_dir
export run_id
export bc_checkpoint

export LORA_ENABLED=true
export LORA_R=${LORA_R:-32}
export LORA_ALPHA=${LORA_ALPHA:-64}
export LORA_DROPOUT=${LORA_DROPOUT:-0.05}
export LORA_FAIL_IF_NO_TARGET_MODULES=true
export LORA_TARGET_INCLUDE_PREFIXES=model
export LORA_TARGET_EXCLUDE_PREFIXES=model.visual
export LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj

echo "[final-awac-critic] run_dir=${run_root_dir}/${run_id}"
echo "[final-awac-critic] base_vlm=${base_vlm}"
echo "[final-awac-critic] bc_checkpoint=${bc_checkpoint}"
echo "[final-awac-critic] data=${calvin_data_root}/${calvin_dataset_name}"
echo "[final-awac-critic] compute_rewards_on_the_fly=${compute_rewards_on_the_fly} assume_success_if_missing=${assume_success_if_missing}"

bash examples/calvin/train_files/run_calvin_awac_critic.sh

FINAL_CRITIC_CKPT="${run_root_dir}/${run_id}/checkpoints/steps_${critic_max_train_steps}_critic.pt"
test -f "${FINAL_CRITIC_CKPT}"

echo "[final-awac-critic] OK"
echo "FINAL_CRITIC_CKPT=${FINAL_CRITIC_CKPT}"
