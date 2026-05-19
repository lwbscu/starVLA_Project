#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

base_vlm=./playground/Pretrained_models/Qwen3.5-9B
config_yaml=./examples/calvin/train_files/starvla_awac_calvin.yaml
calvin_data_root=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d
data_mix=calvin_abc_d_h200
bc_checkpoint=./results/Checkpoints/your_bc_run/checkpoints/steps_30000_pytorch_model.pt
critic_checkpoint=./results/Checkpoints/awac_calvin_critic/checkpoints/steps_50000_critic.pt
run_root_dir=./results/Checkpoints
run_id=awac_calvin_actor

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_awac_actor.py \
  --config_yaml "${config_yaml}" \
  --framework.name QwenPI_AWAC \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --datasets.awac_data.data_root_dir "${calvin_data_root}" \
  --datasets.vla_data.data_root_dir "${calvin_data_root}" \
  --datasets.awac_data.data_mix "${data_mix}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --trainer.pretrained_checkpoint "${bc_checkpoint}" \
  --trainer.awac_critic_checkpoint "${critic_checkpoint}" \
  --trainer.actor_max_train_steps 30000 \
  --trainer.save_interval 5000 \
  --trainer.freeze_modules qwen_vl_interface \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}"
