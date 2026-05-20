#!/usr/bin/env bash
# H200 one-click: merge rollout shards -> match expert subset 1:1 -> train AWAC critic.
# Usage (on H200, from any directory):
#   bash /inspire/.../starVLA_Project/examples/calvin/train_files/h200_awac_critic_mixed_oneclick.sh
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project}"
export CONDA_ROOT="${CONDA_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/26220216/miniconda3}"
export ROLLOUT_SRC="${ROLLOUT_SRC:-${PROJECT_ROOT}/results/rollout/pi_state_4b_awac_parallel_500seq_v1}"
export ROLLOUT_EVAL_ROOT="${ROLLOUT_EVAL_ROOT:-${PROJECT_ROOT}/awac_datasets/rollout_4b_500seq_merged/eval}"
export CALVIN_DATA_ROOT="${CALVIN_DATA_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d}"

cd "${PROJECT_ROOT}"

export PATH="${CONDA_ROOT}/bin:${PATH}"
# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate starVLA_qwen35
export STAR_VLA_PYTHON="${STAR_VLA_PYTHON:-/inspire/qb-ilm2/project/26summer-camp-10/26220216/miniconda3/envs/starVLA_qwen35/bin/python}"

echo "========== [1/4] Merge rollout shards =========="
MERGE_ARGS=()
if [[ "${MERGE_FORCE:-false}" == "true" ]]; then
  MERGE_ARGS+=(--force)
fi
"${STAR_VLA_PYTHON}" examples/calvin/scripts/merge_rollout_lerobot_shards.py \
  --src "${ROLLOUT_SRC}" \
  --dst "${ROLLOUT_EVAL_ROOT}/rollout_lerobot" \
  "${MERGE_ARGS[@]}"

echo "========== [2/4] Build expert allowlist (success-only, same episode count as rollout) =========="
ALLOWLIST_PATH="${ROLLOUT_EVAL_ROOT}/mixed_episode_allowlists.json"
"${STAR_VLA_PYTHON}" examples/calvin/scripts/prepare_awac_mixed_expert_subset.py \
  --expert-root "${CALVIN_DATA_ROOT}" \
  --expert-name calvin_task_ABC_D \
  --rollout-eval-root "${ROLLOUT_EVAL_ROOT}" \
  --output "${ALLOWLIST_PATH}" \
  --seed "${MIXED_SEED:-42}"

echo "========== [3/4] Preflight =========="
export base_vlm="${base_vlm:-${PROJECT_ROOT}/playground/Pretrained_models/Qwen3.5-4B}"
export bc_checkpoint="${bc_checkpoint:-${PROJECT_ROOT}/logs/20260519_h200_server1_pi_state_30k_v1/server1_pi_state/qwen35_4b/pi_state_qwen35_4b_30000step/checkpoints/pi_state_qwen35_4b_30000step/checkpoints/steps_30000_pytorch_model.pt}"
test -f "${base_vlm}/config.json"
test -f "${bc_checkpoint}"
test -f "${CALVIN_DATA_ROOT}/calvin_task_ABC_D/meta/info.json"
test -f "${ROLLOUT_EVAL_ROOT}/rollout_lerobot/meta/info.json"
test -f "${ALLOWLIST_PATH}"

"${STAR_VLA_PYTHON}" - <<'PY' "${base_vlm}"
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
cfg_path = base / "config.json"
if not cfg_path.exists():
    raise SystemExit(f"Missing {cfg_path}")
cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
path_s = str(base).lower()
name_s = json.dumps(cfg).lower()
if "9b" in path_s or "qwen3.5-9" in path_s or "35-9b" in path_s:
    raise SystemExit(f"REFUSE 9B base_vlm for AWAC critic: {base}")
if "4b" not in path_s and "qwen3.5-4" not in path_s:
    print(f"[preflight] WARNING: base_vlm path does not contain '4B': {base}")
hidden = cfg.get("hidden_size") or cfg.get("text_config", {}).get("hidden_size")
print(f"[preflight] base_vlm OK: {base} hidden_size={hidden}")
PY

echo "preflight_ok"

echo "========== [4/4] Launch mixed AWAC critic training =========="
# Use all 8 GPUs on the node unless AWAC_RESPECT_CUDA_VISIBLE_DEVICES=true (narrow Slurm job).
export AWAC_NUM_GPUS="${AWAC_NUM_GPUS:-8}"
unset num_processes
export WANDB_MODE="${WANDB_MODE:-offline}"

export calvin_data_root="${CALVIN_DATA_ROOT}"
export rollout_eval_root="${ROLLOUT_EVAL_ROOT}"
export episode_allowlists_path="${ALLOWLIST_PATH}"
export data_mix=calvin_awac_mixed_h200
export balance_datasets=true
export include_state=true
export state_dim=8
export compute_rewards_on_the_fly=true
export assume_success_if_missing=false
export critic_max_train_steps="${critic_max_train_steps:-10000}"
export save_interval="${save_interval:-1000}"
export per_device_batch_size="${per_device_batch_size:-32}"
export run_root_dir="${run_root_dir:-${PROJECT_ROOT}/logs/20260520_awac_pi_state_mixed}"
export run_id="${run_id:-awac_critic_mixed_8gpu_10k}"
export TB_PORT="${TB_PORT:-6006}"

TB_LOGDIR="${run_root_dir}/${run_id}/tensorboard"
bash examples/calvin/train_files/awac_start_tensorboard.sh "${TB_LOGDIR}"

bash examples/calvin/train_files/run_calvin_awac_critic.sh

echo "========== Done. Checkpoints under: ${run_root_dir}/${run_id}/checkpoints =========="
