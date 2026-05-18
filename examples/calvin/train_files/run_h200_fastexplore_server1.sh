#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export LOG_ROOT=${LOG_ROOT:-logs/h200_fastexplore}
export PIPELINE_TS=${PIPELINE_TS:-$(date +"%Y%m%d_%H%M%S")_server1}
export H200_QWEN35_9B=${H200_QWEN35_9B:-./playground/Pretrained_models/Qwen3.5-9B}
export P0_GPUS=${P0_GPUS:-0,1,2,3}
export P0_NUM_PROCESSES=${P0_NUM_PROCESSES:-4}
export P0_MAIN_PROCESS_PORT=${P0_MAIN_PROCESS_PORT:-29600}
export P4_GPUS=${P4_GPUS:-4,5,6,7}
export P4_NUM_PROCESSES=${P4_NUM_PROCESSES:-4}
export P4_MAIN_PROCESS_PORT=${P4_MAIN_PROCESS_PORT:-29610}
export P4_BASE_VLM=${P4_BASE_VLM:-${H200_QWEN35_9B}}
export P0_EVAL_PORT=${P0_EVAL_PORT:-5794}
export P4_EVAL_PORT=${P4_EVAL_PORT:-5795}
export P0_EVAL_GPU=${P0_EVAL_GPU:-${P0_GPUS%%,*}}
export P4_EVAL_GPU=${P4_EVAL_GPU:-${P4_GPUS%%,*}}

mkdir -p "${LOG_ROOT}/terminal"

p0_launch_log="${LOG_ROOT}/terminal/${PIPELINE_TS}_server1_p0_oft.launch.log"
p4_launch_log="${LOG_ROOT}/terminal/${PIPELINE_TS}_server1_p4_pi.launch.log"

print_failure_context() {
  local route=$1
  local launch_log=$2
  echo "----- ${route} failure context -----" >&2
  echo "launch_log=${launch_log}" >&2
  if [[ -f "${launch_log}" ]]; then
    tail -n 160 "${launch_log}" >&2 || true
  else
    echo "missing launch log: ${launch_log}" >&2
  fi

  local pipeline_log="${LOG_ROOT}/terminal/${PIPELINE_TS}_${route}_pipeline.log"
  echo "current_pipeline_log=${pipeline_log}" >&2
  if [[ -f "${pipeline_log}" ]]; then
    tail -n 120 "${pipeline_log}" >&2 || true
  else
    echo "current pipeline log was not created; failure happened before route pipeline logging started." >&2
  fi

  echo "older pipeline logs for ${route} are not tailed here to avoid mixing stale errors." >&2
  echo "----- end ${route} failure context -----" >&2
}

(
  ROUTE=p0_oft \
  TRAIN_GPUS="${P0_GPUS}" \
  NUM_PROCESSES="${P0_NUM_PROCESSES}" \
  MAIN_PROCESS_PORT="${P0_MAIN_PROCESS_PORT}" \
  EVAL_PORT="${P0_EVAL_PORT}" \
  EVAL_GPU="${P0_EVAL_GPU}" \
  bash examples/calvin/train_files/run_h200_fastexplore_route.sh
) > "${p0_launch_log}" 2>&1 &
pid_p0=$!

(
  ROUTE=p4_pi \
  TRAIN_GPUS="${P4_GPUS}" \
  NUM_PROCESSES="${P4_NUM_PROCESSES}" \
  MAIN_PROCESS_PORT="${P4_MAIN_PROCESS_PORT}" \
  BASE_VLM="${P4_BASE_VLM}" \
  EVAL_PORT="${P4_EVAL_PORT}" \
  EVAL_GPU="${P4_EVAL_GPU}" \
  bash examples/calvin/train_files/run_h200_fastexplore_route.sh
) > "${p4_launch_log}" 2>&1 &
pid_p4=$!

echo "server1 p0_oft pid=${pid_p0} gpus=${P0_GPUS} main_process_port=${P0_MAIN_PROCESS_PORT} eval_port=${P0_EVAL_PORT} eval_gpu=${P0_EVAL_GPU}"
echo "server1 p4_pi pid=${pid_p4} gpus=${P4_GPUS} main_process_port=${P4_MAIN_PROCESS_PORT} eval_port=${P4_EVAL_PORT} eval_gpu=${P4_EVAL_GPU}"

status=0
if ! wait "${pid_p0}"; then
  echo "server1 p0_oft failed" >&2
  print_failure_context "p0_oft" "${p0_launch_log}"
  status=1
fi
if ! wait "${pid_p4}"; then
  echo "server1 p4_pi failed" >&2
  print_failure_context "p4_pi" "${p4_launch_log}"
  status=1
fi

exit "${status}"
