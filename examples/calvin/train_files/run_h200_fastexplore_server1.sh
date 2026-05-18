#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export LOG_ROOT=${LOG_ROOT:-logs/h200_fastexplore}
export PIPELINE_TS=${PIPELINE_TS:-$(date +"%Y%m%d_%H%M%S")_server1}
export P0_GPUS=${P0_GPUS:-0,1,2,3}
export P0_NUM_PROCESSES=${P0_NUM_PROCESSES:-4}
export P4_GPUS=${P4_GPUS:-4,5,6,7}
export P4_NUM_PROCESSES=${P4_NUM_PROCESSES:-4}
export P4_BASE_VLM=${P4_BASE_VLM:-./playground/Pretrained_models/Qwen3.5-4B}
export P0_EVAL_PORT=${P0_EVAL_PORT:-5694}
export P4_EVAL_PORT=${P4_EVAL_PORT:-5695}
export P0_EVAL_GPU=${P0_EVAL_GPU:-${P0_GPUS%%,*}}
export P4_EVAL_GPU=${P4_EVAL_GPU:-${P4_GPUS%%,*}}

mkdir -p "${LOG_ROOT}/terminal"

(
  ROUTE=p0_oft \
  TRAIN_GPUS="${P0_GPUS}" \
  NUM_PROCESSES="${P0_NUM_PROCESSES}" \
  EVAL_PORT="${P0_EVAL_PORT}" \
  EVAL_GPU="${P0_EVAL_GPU}" \
  bash examples/calvin/train_files/run_h200_fastexplore_route.sh
) > "${LOG_ROOT}/terminal/${PIPELINE_TS}_server1_p0_oft.launch.log" 2>&1 &
pid_p0=$!

(
  ROUTE=p4_qwen4b_pi \
  TRAIN_GPUS="${P4_GPUS}" \
  NUM_PROCESSES="${P4_NUM_PROCESSES}" \
  BASE_VLM="${P4_BASE_VLM}" \
  EVAL_PORT="${P4_EVAL_PORT}" \
  EVAL_GPU="${P4_EVAL_GPU}" \
  bash examples/calvin/train_files/run_h200_fastexplore_route.sh
) > "${LOG_ROOT}/terminal/${PIPELINE_TS}_server1_p4_qwen4b_pi.launch.log" 2>&1 &
pid_p4=$!

echo "server1 p0_oft pid=${pid_p0} gpus=${P0_GPUS} eval_port=${P0_EVAL_PORT} eval_gpu=${P0_EVAL_GPU}"
echo "server1 p4_qwen4b_pi pid=${pid_p4} gpus=${P4_GPUS} eval_port=${P4_EVAL_PORT} eval_gpu=${P4_EVAL_GPU}"

status=0
if ! wait "${pid_p0}"; then
  echo "server1 p0_oft failed" >&2
  status=1
fi
if ! wait "${pid_p4}"; then
  echo "server1 p4_qwen4b_pi failed" >&2
  status=1
fi

exit "${status}"
