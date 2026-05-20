#!/usr/bin/env bash
# Phase 1: train Q critic only (BC actor frozen; TD loss, no V network).
set -euo pipefail

cd "$(dirname "$0")/../../.."

_awac_count_physical_gpus() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '
  else
    echo 0
  fi
}

_awac_count_visible_from_env() {
  local cvd="${CUDA_VISIBLE_DEVICES:-}"
  if [[ -z "${cvd}" ]]; then
    echo 0
    return
  fi
  local count=0
  local part
  IFS=',' read -ra _parts <<< "${cvd}"
  for part in "${_parts[@]}"; do
    part="${part//[[:space:]]/}"
    if [[ -n "${part}" ]]; then
      count=$((count + 1))
    fi
  done
  echo "${count}"
}

_awac_build_gpu_list() {
  local n="$1"
  local out=""
  local i
  for ((i = 0; i < n; i++)); do
    if [[ -n "${out}" ]]; then
      out+=","
    fi
    out+="${i}"
  done
  echo "${out}"
}

# H200 default: use 8 GPUs when the node has them. Many login shells set CUDA_VISIBLE_DEVICES=0
# by mistake; set AWAC_RESPECT_CUDA_VISIBLE_DEVICES=true to keep a narrow Slurm allocation.
_PHYSICAL_GPUS="$(_awac_count_physical_gpus)"
_AWAC_TARGET_GPUS="${AWAC_NUM_GPUS:-8}"
if [[ "${AWAC_RESPECT_CUDA_VISIBLE_DEVICES:-false}" == "true" ]]; then
  :
elif [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if [[ "${_PHYSICAL_GPUS}" -ge "${_AWAC_TARGET_GPUS}" ]]; then
    export CUDA_VISIBLE_DEVICES="$(_awac_build_gpu_list "${_AWAC_TARGET_GPUS}")"
  elif [[ "${_PHYSICAL_GPUS}" -gt 0 ]]; then
    export CUDA_VISIBLE_DEVICES="$(_awac_build_gpu_list "${_PHYSICAL_GPUS}")"
  else
    export CUDA_VISIBLE_DEVICES="$(_awac_build_gpu_list "${_AWAC_TARGET_GPUS}")"
  fi
else
  _VISIBLE_FROM_ENV="$(_awac_count_visible_from_env)"
  if [[ "${_PHYSICAL_GPUS}" -ge "${_AWAC_TARGET_GPUS}" && "${_VISIBLE_FROM_ENV}" -lt "${_AWAC_TARGET_GPUS}" ]]; then
    echo "[awac-critic] NOTE: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} exposes only ${_VISIBLE_FROM_ENV} GPU(s),"
    echo "[awac-critic]       but nvidia-smi reports ${_PHYSICAL_GPUS}. Expanding to 0..$((_AWAC_TARGET_GPUS - 1))."
    export CUDA_VISIBLE_DEVICES="$(_awac_build_gpu_list "${_AWAC_TARGET_GPUS}")"
  fi
fi

_VISIBLE_FROM_ENV="$(_awac_count_visible_from_env)"
export num_processes="${num_processes:-${_VISIBLE_FROM_ENV}}"

export WANDB_MODE=${WANDB_MODE:-online}
export NO_ALBUMENTATIONS_UPDATE=${NO_ALBUMENTATIONS_UPDATE:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

Framework_name=${Framework_name:-QwenPI}
# AWAC critic must use Qwen3.5-4B (BC checkpoint is 4B PI-State); do not default to 9B.
H200_QWEN35_4B_DEFAULT=/inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project/playground/Pretrained_models/Qwen3.5-4B
base_vlm=${base_vlm:-${H200_QWEN35_4B_DEFAULT}}
calvin_data_root=${calvin_data_root:-/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d}
calvin_dataset_name=${calvin_dataset_name:-calvin_task_ABC_D}
rollout_eval_root=${rollout_eval_root:-}
episode_allowlists_path=${episode_allowlists_path:-}
data_mix=${data_mix:-calvin_abc_d_h200}
balance_datasets=${balance_datasets:-}

if [[ "${data_mix}" == "calvin_awac_mixed_h200" ]]; then
  config_yaml=${config_yaml:-./examples/calvin/train_files/starvla_awac_calvin_mixed.yaml}
  balance_datasets=${balance_datasets:-true}
  critic_max_train_steps=${critic_max_train_steps:-10000}
  save_interval=${save_interval:-1000}
  per_device_batch_size=${per_device_batch_size:-32}
else
  config_yaml=${config_yaml:-./examples/calvin/train_files/starvla_awac_calvin.yaml}
  critic_max_train_steps=${critic_max_train_steps:-50000}
  per_device_batch_size=${per_device_batch_size:-}
fi
include_state=${include_state:-true}
state_dim=${state_dim:-8}
compute_rewards_on_the_fly=${compute_rewards_on_the_fly:-false}
success_column=${success_column:-success}
assume_success_if_missing=${assume_success_if_missing:-false}
step_penalty=${step_penalty:--1.0}
success_reward=${success_reward:-0.0}
failure_reward=${failure_reward:--3000.0}
bc_checkpoint=${bc_checkpoint:-./results/Checkpoints/your_bc_run/checkpoints/steps_30000_pytorch_model.pt}
run_root_dir=${run_root_dir:-logs}
run_id=${run_id:-awac_calvin_critic}
save_interval=${save_interval:-5000}

STAR_VLA_PYTHON=${STAR_VLA_PYTHON:-python}
ACCELERATE_LAUNCH=("${STAR_VLA_PYTHON}" -m accelerate.commands.launch)

_TORCH_VISIBLE="$("${STAR_VLA_PYTHON}" - <<'PY'
import os
import torch
# Must match the shell-exported CUDA_VISIBLE_DEVICES for this check.
print(torch.cuda.device_count())
PY
)"

echo "[awac-critic] physical_gpus(nvidia-smi)=${_PHYSICAL_GPUS}"
echo "[awac-critic] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[awac-critic] visible_gpus(env)=${_VISIBLE_FROM_ENV} visible_gpus(torch)=${_TORCH_VISIBLE} num_processes=${num_processes}"

if [[ "${num_processes}" -lt 1 ]]; then
  echo "[awac-critic] ERROR: no GPU detected; check nvidia-smi and CUDA_VISIBLE_DEVICES." >&2
  exit 1
fi
if [[ "${_TORCH_VISIBLE}" -lt "${num_processes}" ]]; then
  echo "[awac-critic] ERROR: torch sees ${_TORCH_VISIBLE} GPU(s) but num_processes=${num_processes}." >&2
  echo "[awac-critic] Fix: export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 (or your Slurm allocation) before launch." >&2
  exit 1
fi
if [[ "${num_processes}" -lt 2 ]]; then
  echo "[awac-critic] WARNING: num_processes=1 -> DeepSpeed world_size=1." >&2
fi

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

# Run once before first AWAC training if parquet has no reward/done columns.
# Prefer writing to a personal AWAC working copy; only use --allow_in_place on
# non-public copied datasets:
# python examples/calvin/scripts/prepare_awac_rewards.py \
#   --dataset_root "${calvin_data_root}/${calvin_dataset_name}" \
#   --output_dataset_root "/path/to/awac_work/${calvin_dataset_name}" \
#   --action_horizon 8 --gamma 0.996
#
# Disk-light alternative for verified all-success demos:
#   compute_rewards_on_the_fly=true assume_success_if_missing=true bash ...
# This does not mutate parquet; reward/done are computed in the dataloader.

# DeepSpeed launch: do NOT pass --multi_gpu (conflicts with DEEPSPEED config -> world_size=1).
"${ACCELERATE_LAUNCH[@]}" \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${num_processes}" \
  starVLA/training/train_awac_critic.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${Framework_name}" \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --framework.action_model.state_dim "${state_dim}" \
  --datasets.awac_data.data_root_dir "${calvin_data_root}" \
  --datasets.awac_data.data_mix "${data_mix}" \
  --datasets.awac_data.include_state "${include_state}" \
  --datasets.awac_data.compute_rewards_on_the_fly "${compute_rewards_on_the_fly}" \
  --datasets.awac_data.success_column "${success_column}" \
  --datasets.awac_data.assume_success_if_missing "${assume_success_if_missing}" \
  --datasets.awac_data.step_penalty "${step_penalty}" \
  --datasets.awac_data.success_reward "${success_reward}" \
  --datasets.awac_data.failure_reward "${failure_reward}" \
  --trainer.pretrained_checkpoint "${bc_checkpoint}" \
  --trainer.critic_max_train_steps "${critic_max_train_steps}" \
  --trainer.save_interval "${save_interval}" \
  --trainer.use_tensorboard true \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  ${per_device_batch_size:+--datasets.awac_data.per_device_batch_size "${per_device_batch_size}"} \
  ${balance_datasets:+--datasets.awac_data.balance_datasets "${balance_datasets}"} \
  ${rollout_eval_root:+--datasets.awac_data.dataset_roots.calvin_task_ABC_D "${calvin_data_root}"} \
  ${rollout_eval_root:+--datasets.awac_data.dataset_roots.rollout_lerobot "${rollout_eval_root}"} \
  ${episode_allowlists_path:+--datasets.awac_data.episode_allowlists_path "${episode_allowlists_path}"}
