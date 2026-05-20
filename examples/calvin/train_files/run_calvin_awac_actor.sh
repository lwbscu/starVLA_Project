#!/usr/bin/env bash
# Phase 2: weighted actor (trainable actor + frozen actor_pi snapshot; frozen Q critic).
set -euo pipefail

cd "$(dirname "$0")/../../.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export WANDB_MODE=${WANDB_MODE:-online}
export NO_ALBUMENTATIONS_UPDATE=${NO_ALBUMENTATIONS_UPDATE:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

base_vlm=${base_vlm:-./playground/Pretrained_models/Qwen3.5-9B}
config_yaml=${config_yaml:-./examples/calvin/train_files/starvla_awac_calvin.yaml}
calvin_data_root=${calvin_data_root:-/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d}
data_mix=${data_mix:-calvin_abc_d_h200}
include_state=${include_state:-true}
state_dim=${state_dim:-8}
compute_rewards_on_the_fly=${compute_rewards_on_the_fly:-false}
success_column=${success_column:-success}
assume_success_if_missing=${assume_success_if_missing:-false}
step_penalty=${step_penalty:--1.0}
success_reward=${success_reward:-0.0}
failure_reward=${failure_reward:--3000.0}
bc_checkpoint=${bc_checkpoint:-./results/Checkpoints/your_bc_run/checkpoints/steps_30000_pytorch_model.pt}
critic_checkpoint=${critic_checkpoint:-./logs/awac_calvin_critic/checkpoints/steps_50000_critic.pt}
run_root_dir=${run_root_dir:-logs}
run_id=${run_id:-awac_calvin_actor}
actor_max_train_steps=${actor_max_train_steps:-30000}
save_interval=${save_interval:-5000}
per_device_batch_size=${per_device_batch_size:-}
num_processes=${num_processes:-8}
STAR_VLA_PYTHON=${STAR_VLA_PYTHON:-python}
ACCELERATE_LAUNCH=("${STAR_VLA_PYTHON}" -m accelerate.commands.launch)

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

"${ACCELERATE_LAUNCH[@]}" \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${num_processes}" \
  starVLA/training/train_awac_actor.py \
  --config_yaml "${config_yaml}" \
  --framework.name QwenPI_AWAC \
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
  --trainer.awac_critic_checkpoint "${critic_checkpoint}" \
  --trainer.actor_max_train_steps "${actor_max_train_steps}" \
  --trainer.save_interval "${save_interval}" \
  --trainer.freeze_modules qwen_vl_interface \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  ${per_device_batch_size:+--datasets.awac_data.per_device_batch_size "${per_device_batch_size}"}
