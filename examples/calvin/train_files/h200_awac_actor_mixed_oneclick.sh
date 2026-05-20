#!/usr/bin/env bash
# H200 one-click: mixed AWAC actor (same data as critic; action-head-only training).
#
# Prerequisites:
#   - Merged rollout + expert allowlist (from critic oneclick, or set SKIP_DATA_PREP=false)
#   - Trained critic checkpoints under CRITIC_RUN_ROOT
#
# Usage:
#   bash examples/calvin/train_files/h200_awac_actor_mixed_oneclick.sh
#
# Pick a specific critic step:
#   export critic_checkpoint=.../checkpoints/steps_10000_critic.pt
#   bash examples/calvin/train_files/h200_awac_actor_mixed_oneclick.sh
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project}"
export CONDA_ROOT="${CONDA_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/26220216/miniconda3}"
export ROLLOUT_SRC="${ROLLOUT_SRC:-${PROJECT_ROOT}/results/rollout/pi_state_4b_awac_parallel_500seq_v1}"
export ROLLOUT_EVAL_ROOT="${ROLLOUT_EVAL_ROOT:-${PROJECT_ROOT}/awac_datasets/rollout_4b_500seq_merged/eval}"
export CALVIN_DATA_ROOT="${CALVIN_DATA_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d}"
export AWAC_SMOKE_LOG_ROOT="${AWAC_SMOKE_LOG_ROOT:-${PROJECT_ROOT}/logs/awac_smoke_20260519_232545}"
export CRITIC_RUN_ROOT="${CRITIC_RUN_ROOT:-${AWAC_SMOKE_LOG_ROOT}/critic_alone}"
export critic_checkpoint="${critic_checkpoint:-}"
export SKIP_DATA_PREP="${SKIP_DATA_PREP:-true}"

_resolve_critic_checkpoint() {
  # 1) User-provided path or CRITIC_RUN_ROOT default.
  local candidates=()
  if [[ -n "${critic_checkpoint:-}" ]]; then
    candidates+=("${critic_checkpoint}")
  fi
  candidates+=(
    "${CRITIC_RUN_ROOT}/checkpoints/steps_3000_critic.pt"
    "${AWAC_SMOKE_LOG_ROOT}/critic_alone/checkpoints/steps_3000_critic.pt"
    "${AWAC_SMOKE_LOG_ROOT}/critic_realone/checkpoints/steps_3000_critic.pt"
  )
  local c
  for c in "${candidates[@]}"; do
    if [[ -n "${c}" && -f "${c}" ]]; then
      echo "${c}"
      return 0
    fi
  done
  # 2) Search under smoke log root (any sub-run folder name).
  if [[ -d "${AWAC_SMOKE_LOG_ROOT}" ]]; then
    local found
    found="$(
      find "${AWAC_SMOKE_LOG_ROOT}" -maxdepth 4 -type f -name 'steps_3000_critic.pt' 2>/dev/null \
        | sort -V \
        | tail -n 1
    )"
    if [[ -n "${found}" && -f "${found}" ]]; then
      echo "${found}"
      return 0
    fi
    found="$(
      find "${AWAC_SMOKE_LOG_ROOT}" -maxdepth 4 -type f -name 'steps_*_critic.pt' 2>/dev/null \
        | sort -V \
        | tail -n 1
    )"
    if [[ -n "${found}" && -f "${found}" ]]; then
      echo "${found}"
      return 0
    fi
  fi
  return 1
}

cd "${PROJECT_ROOT}"

export PATH="${CONDA_ROOT}/bin:${PATH}"
# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate starVLA_qwen35
export STAR_VLA_PYTHON="${STAR_VLA_PYTHON:-/inspire/qb-ilm2/project/26summer-camp-10/26220216/miniconda3/envs/starVLA_qwen35/bin/python}"

ALLOWLIST_PATH="${ROLLOUT_EVAL_ROOT}/mixed_episode_allowlists.json"

if [[ "${SKIP_DATA_PREP}" != "true" ]]; then
  echo "========== [1/4] Merge rollout shards =========="
  MERGE_ARGS=()
  if [[ "${MERGE_FORCE:-false}" == "true" ]]; then
    MERGE_ARGS+=(--force)
  fi
  "${STAR_VLA_PYTHON}" examples/calvin/scripts/merge_rollout_lerobot_shards.py \
    --src "${ROLLOUT_SRC}" \
    --dst "${ROLLOUT_EVAL_ROOT}/rollout_lerobot" \
    "${MERGE_ARGS[@]}"

  echo "========== [2/4] Build expert allowlist =========="
  "${STAR_VLA_PYTHON}" examples/calvin/scripts/prepare_awac_mixed_expert_subset.py \
    --expert-root "${CALVIN_DATA_ROOT}" \
    --expert-name calvin_task_ABC_D \
    --rollout-eval-root "${ROLLOUT_EVAL_ROOT}" \
    --output "${ALLOWLIST_PATH}" \
    --seed "${MIXED_SEED:-42}"
else
  echo "========== [1-2/4] SKIP_DATA_PREP=true — using existing merged rollout + allowlist =========="
fi

echo "========== [3/4] Preflight =========="
export base_vlm="${base_vlm:-${PROJECT_ROOT}/playground/Pretrained_models/Qwen3.5-4B}"
export bc_checkpoint="${bc_checkpoint:-${PROJECT_ROOT}/logs/20260519_h200_server1_pi_state_30k_v1/server1_pi_state/qwen35_4b/pi_state_qwen35_4b_30000step/checkpoints/pi_state_qwen35_4b_30000step/checkpoints/steps_30000_pytorch_model.pt}"

test -f "${base_vlm}/config.json"
test -f "${bc_checkpoint}"
test -f "${CALVIN_DATA_ROOT}/calvin_task_ABC_D/meta/info.json"
test -f "${ROLLOUT_EVAL_ROOT}/rollout_lerobot/meta/info.json"
test -f "${ALLOWLIST_PATH}"

_REQUESTED_CRITIC="${critic_checkpoint:-}"
if ! critic_checkpoint="$(_resolve_critic_checkpoint)"; then
  echo "ERROR: Could not find critic checkpoint." >&2
  echo "  Requested critic_checkpoint=${_REQUESTED_CRITIC:-<unset>}" >&2
  echo "  CRITIC_RUN_ROOT=${CRITIC_RUN_ROOT}" >&2
  echo "  AWAC_SMOKE_LOG_ROOT=${AWAC_SMOKE_LOG_ROOT}" >&2
  if [[ -d "${AWAC_SMOKE_LOG_ROOT}" ]]; then
    echo "  Subdirs under smoke log root:" >&2
    ls -la "${AWAC_SMOKE_LOG_ROOT}" >&2 || true
    echo "  Any steps_*_critic.pt under smoke log root:" >&2
    find "${AWAC_SMOKE_LOG_ROOT}" -maxdepth 5 -type f -name 'steps_*_critic.pt' 2>/dev/null >&2 || true
  fi
  exit 2
fi
export critic_checkpoint
echo "[preflight] Using critic_checkpoint=${critic_checkpoint}"
test -f "${critic_checkpoint}"

"${STAR_VLA_PYTHON}" - <<'PY' "${base_vlm}" "${critic_checkpoint}"
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
ckpt = Path(sys.argv[2])
cfg = json.loads((base / "config.json").read_text(encoding="utf-8"))
path_s = str(base).lower()
if "9b" in path_s or "qwen3.5-9" in path_s:
    raise SystemExit(f"REFUSE 9B base_vlm for AWAC actor: {base}")
print(f"[preflight] base_vlm OK: {base}")
print(f"[preflight] critic OK: {ckpt} ({ckpt.stat().st_size / 1e6:.1f} MB)")
PY

echo "preflight_ok"

echo "========== [4/4] Launch mixed AWAC actor training =========="
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

export actor_max_train_steps="${actor_max_train_steps:-30000}"
export save_interval="${save_interval:-5000}"
export per_device_batch_size="${per_device_batch_size:-16}"
export run_root_dir="${run_root_dir:-${PROJECT_ROOT}/logs/20260520_awac_pi_state_mixed}"
export run_id="${run_id:-awac_actor_mixed_8gpu_30k}"
export TB_PORT="${TB_PORT:-6007}"

TB_LOGDIR="${run_root_dir}/${run_id}/tensorboard"
bash examples/calvin/train_files/awac_start_tensorboard.sh "${TB_LOGDIR}"

bash examples/calvin/train_files/run_calvin_awac_actor.sh

echo "========== Done. Actor checkpoints under: ${run_root_dir}/${run_id}/checkpoints =========="
echo "========== (steps_*_pytorch_model.pt — load like BC for eval) =========="
