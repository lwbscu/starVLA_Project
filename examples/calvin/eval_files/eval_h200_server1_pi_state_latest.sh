#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export PROJECT_ROOT=${PROJECT_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project}
export EVAL_BATCH_ROOT=${EVAL_BATCH_ROOT:-"${PROJECT_ROOT}/logs/20260519_h200_server1_pi_state_30k_v1"}
export EVAL_OUTPUT_ROOT=${EVAL_OUTPUT_ROOT:-"${PROJECT_ROOT}/logs/calvin_eval_latest_20260519_h200_server1_pi_state_30k_v1"}
export EVAL_PORT=${EVAL_PORT:-6300}
export SEND_STATE_TO_POLICY=${SEND_STATE_TO_POLICY:-1}

if [[ -z "${EXPECTED_JOBS:-}" ]]; then
EXPECTED_JOBS=$(cat <<'EOF'
server1_pi_state/qwen35_0p8b
server1_pi_state/qwen35_4b
server1_pi_state/qwen35_9b
EOF
)
fi
export EXPECTED_JOBS

exec examples/calvin/eval_files/eval_h200_qwen_mix_latest.sh "$@"
