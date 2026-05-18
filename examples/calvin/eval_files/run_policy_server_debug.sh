#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export NO_ALBUMENTATIONS_UPDATE=1

RUN_ID=${RUN_ID:-calvin_policy_server_debug}
RUN_TS=${RUN_TS:-$(date +"%Y%m%d_%H%M%S")}
LOG_DIR=${LOG_DIR:-logs/log_${RUN_TS}_${RUN_ID}}
PORT=${PORT:-5694}
STAR_VLA_PYTHON=${STAR_VLA_PYTHON:-"$(conda info --base)/envs/starVLA_qwen35/bin/python"}
POLICY_PID=""

: "${CKPT_PATH:?Set CKPT_PATH to a StarVLA checkpoint .pt or .safetensors file}"

mkdir -p "${LOG_DIR}"/{train,eval,terminal,mp4,configs,metrics,checkpoints}
cp "$0" "${LOG_DIR}/configs/"
exec > >(tee "${LOG_DIR}/terminal/policy_server.log") 2>&1

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "${POLICY_PID}" ]] && kill -0 "${POLICY_PID}" 2>/dev/null; then
    echo "Stopping policy server child pid=${POLICY_PID}"
    kill "${POLICY_PID}" 2>/dev/null || true
    wait "${POLICY_PID}" 2>/dev/null || true
  fi
  exit "${status}"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "CKPT_PATH=${CKPT_PATH}"
echo "PORT=${PORT}"
echo "STAR_VLA_PYTHON=${STAR_VLA_PYTHON}"
env -u DEBUG CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${STAR_VLA_PYTHON}" \
  deployment/model_server/server_policy.py \
  --ckpt_path "${CKPT_PATH}" \
  --port "${PORT}" \
  --use_bf16 &
POLICY_PID=$!
wait "${POLICY_PID}"
