#!/usr/bin/env bash
# Phase 2: AWAC actor — train action head only; frozen actor_pi for A; frozen Q critic.
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
    echo "[awac-actor] NOTE: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} exposes only ${_VISIBLE_FROM_ENV} GPU(s),"
    echo "[awac-actor]       but nvidia-smi reports ${_PHYSICAL_GPUS}. Expanding to 0..$((_AWAC_TARGET_GPUS - 1))."
    export CUDA_VISIBLE_DEVICES="$(_awac_build_gpu_list "${_AWAC_TARGET_GPUS}")"
  fi
fi

_VISIBLE_FROM_ENV="$(_awac_count_visible_from_env)"
export num_processes="${num_processes:-${_VISIBLE_FROM_ENV}}"

export WANDB_MODE=${WANDB_MODE:-online}
export NO_ALBUMENTATIONS_UPDATE=${NO_ALBUMENTATIONS_UPDATE:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

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
  actor_max_train_steps=${actor_max_train_steps:-30000}
  save_interval=${save_interval:-5000}
  per_device_batch_size=${per_device_batch_size:-16}
else
  config_yaml=${config_yaml:-./examples/calvin/train_files/starvla_awac_calvin.yaml}
  actor_max_train_steps=${actor_max_train_steps:-30000}
  save_interval=${save_interval:-5000}
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
critic_checkpoint=${critic_checkpoint:-}
run_root_dir=${run_root_dir:-logs}
run_id=${run_id:-awac_calvin_actor}
main_process_port=${main_process_port:-${MAIN_PROCESS_PORT:-}}

LORA_R=${LORA_R:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LORA_ENABLED=${LORA_ENABLED:-false}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-}
LORA_TARGET_INCLUDE_PREFIXES=${LORA_TARGET_INCLUDE_PREFIXES:-}
LORA_TARGET_EXCLUDE_PREFIXES=${LORA_TARGET_EXCLUDE_PREFIXES:-}
LORA_FAIL_IF_NO_TARGET_MODULES=${LORA_FAIL_IF_NO_TARGET_MODULES:-true}

STAR_VLA_PYTHON=${STAR_VLA_PYTHON:-python}
ACCELERATE_LAUNCH=("${STAR_VLA_PYTHON}" -m accelerate.commands.launch)
PORT_ARGS=()
if [[ -n "${main_process_port}" ]]; then
  if ! [[ "${main_process_port}" =~ ^[0-9]+$ ]] || (( main_process_port < 1 || main_process_port > 65535 )); then
    echo "[awac-actor] ERROR: main_process_port must be an integer in [1, 65535], got: ${main_process_port}" >&2
    exit 2
  fi
  PORT_ARGS=(--main_process_port "${main_process_port}")
fi

case "${LORA_ENABLED}" in
  true|false) ;;
  *) echo "[awac-actor] ERROR: LORA_ENABLED must be true or false, got: ${LORA_ENABLED}" >&2; exit 2 ;;
esac
case "${LORA_FAIL_IF_NO_TARGET_MODULES}" in
  true|false) ;;
  *) echo "[awac-actor] ERROR: LORA_FAIL_IF_NO_TARGET_MODULES must be true or false, got: ${LORA_FAIL_IF_NO_TARGET_MODULES}" >&2; exit 2 ;;
esac

LORA_ARGS=()
if [[ "${LORA_ENABLED}" == "true" ]]; then
  LORA_ARGS+=(
    --trainer.lora.enabled true
    --trainer.lora.r "${LORA_R}"
    --trainer.lora.alpha "${LORA_ALPHA}"
    --trainer.lora.dropout "${LORA_DROPOUT}"
    --trainer.lora.fail_if_no_target_modules "${LORA_FAIL_IF_NO_TARGET_MODULES}"
  )
  if [[ -n "${LORA_TARGET_MODULES}" ]]; then
    LORA_ARGS+=(--trainer.lora.target_modules "${LORA_TARGET_MODULES}")
  fi
  if [[ -n "${LORA_TARGET_INCLUDE_PREFIXES}" ]]; then
    LORA_ARGS+=(--trainer.lora.target_include_prefixes "${LORA_TARGET_INCLUDE_PREFIXES}")
  fi
  if [[ -n "${LORA_TARGET_EXCLUDE_PREFIXES}" ]]; then
    LORA_ARGS+=(--trainer.lora.target_exclude_prefixes "${LORA_TARGET_EXCLUDE_PREFIXES}")
  fi
fi

if [[ -z "${critic_checkpoint}" || ! -f "${critic_checkpoint}" ]]; then
  echo "[awac-actor] ERROR: critic_checkpoint must point to an existing steps_*_critic.pt file." >&2
  echo "[awac-actor] Set critic_checkpoint=... or CRITIC_RUN_ROOT + auto-resolve in oneclick script." >&2
  exit 1
fi
if [[ ! -f "${bc_checkpoint}" ]]; then
  echo "[awac-actor] ERROR: bc_checkpoint not found: ${bc_checkpoint}" >&2
  exit 1
fi

_TORCH_VISIBLE="$("${STAR_VLA_PYTHON}" - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"

echo "[awac-actor] physical_gpus(nvidia-smi)=${_PHYSICAL_GPUS}"
echo "[awac-actor] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[awac-actor] visible_gpus(env)=${_VISIBLE_FROM_ENV} visible_gpus(torch)=${_TORCH_VISIBLE} num_processes=${num_processes}"
echo "[awac-actor] bc_checkpoint=${bc_checkpoint}"
echo "[awac-actor] critic_checkpoint=${critic_checkpoint}"
echo "[awac-actor] main_process_port=${main_process_port:-<accelerate-default>}"
echo "[awac-actor] LORA_ENABLED=${LORA_ENABLED}"
if [[ "${LORA_ENABLED}" == "true" ]]; then
  echo "[awac-actor] LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-<auto>}"
  echo "[awac-actor] LORA_TARGET_INCLUDE_PREFIXES=${LORA_TARGET_INCLUDE_PREFIXES:-<none>}"
  echo "[awac-actor] LORA_TARGET_EXCLUDE_PREFIXES=${LORA_TARGET_EXCLUDE_PREFIXES:-<none>}"
fi
echo "[awac-actor] freeze_modules=qwen_vl_interface (action head only)"

if [[ "${num_processes}" -lt 1 ]]; then
  echo "[awac-actor] ERROR: no GPU detected." >&2
  exit 1
fi
if [[ "${_TORCH_VISIBLE}" -lt "${num_processes}" ]]; then
  echo "[awac-actor] ERROR: torch sees ${_TORCH_VISIBLE} GPU(s) but num_processes=${num_processes}." >&2
  exit 1
fi

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

"${ACCELERATE_LAUNCH[@]}" \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${num_processes}" \
  "${PORT_ARGS[@]}" \
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
  --trainer.use_tensorboard true \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  "${LORA_ARGS[@]}" \
  ${per_device_batch_size:+--datasets.awac_data.per_device_batch_size "${per_device_batch_size}"} \
  ${balance_datasets:+--datasets.awac_data.balance_datasets "${balance_datasets}"} \
  ${rollout_eval_root:+--datasets.awac_data.dataset_roots.calvin_task_ABC_D "${calvin_data_root}"} \
  ${rollout_eval_root:+--datasets.awac_data.dataset_roots.rollout_lerobot "${rollout_eval_root}"} \
  ${episode_allowlists_path:+--datasets.awac_data.episode_allowlists_path "${episode_allowlists_path}"}
