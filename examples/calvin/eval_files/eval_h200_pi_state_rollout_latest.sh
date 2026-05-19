#!/usr/bin/env bash
# Strict CALVIN rollout launcher for PI-State BC checkpoints intended for AWAC follow-up.
#
# This wrapper deliberately keeps the generic eval implementation unchanged, but fixes
# the post-training contract at the boundary:
#   - only PI-State jobs are selected by default;
#   - state is sent to the policy server;
#   - LeRobot rollout data is written;
#   - each selected job must have the expected checkpoint step.
set -euo pipefail

cd "$(dirname "$0")/../../.."

export PROJECT_ROOT=${PROJECT_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project}
export CONDA_ROOT=${CONDA_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/26220216/miniconda3}
export STAR_VLA_PYTHON=${STAR_VLA_PYTHON:-"${CONDA_ROOT}/envs/starVLA_qwen35/bin/python"}

export EVAL_BATCH_ROOT=${EVAL_BATCH_ROOT:-"${PROJECT_ROOT}/logs/20260519_h200_server1_pi_state_30k_v1"}
export EVAL_OUTPUT_ROOT=${EVAL_OUTPUT_ROOT:-"${PROJECT_ROOT}/logs/calvin_eval_rollout_20260519_h200_server1_pi_state_30k_v1"}
export EVAL_EXPECTED_STEP=${EVAL_EXPECTED_STEP:-30000}
export NUM_SEQUENCES=${NUM_SEQUENCES:-5}
export EVAL_GPU=${EVAL_GPU:-0}
export EVAL_PORT=${EVAL_PORT:-6200}
export FAIL_FAST=${FAIL_FAST:-0}
export DEBUG_MP4=${DEBUG_MP4:-1}
export REQUIRE_MP4=${REQUIRE_MP4:-1}
export SEND_STATE_TO_POLICY=1
export WRITE_ROLLOUT_LEROBOT=1
export WRITE_ROLLOUT_VIDEOS=${WRITE_ROLLOUT_VIDEOS:-0}

if [[ -z "${EXPECTED_JOBS:-}" ]]; then
EXPECTED_JOBS=$(cat <<'EOF'
server1_pi_state/qwen35_0p8b
server1_pi_state/qwen35_4b
server1_pi_state/qwen35_9b
EOF
)
fi
export EXPECTED_JOBS

if [[ -z "${H200_CALVIN_EVAL_DATASET_PATH:-}" ]]; then
  cat >&2 <<'EOF'
H200_CALVIN_EVAL_DATASET_PATH is required for strict PI-State rollout.
Set it to a real CALVIN eval dataset/split root that contains .hydra/merged_config.yaml,
for example a task_D_D or task_ABC_D dataset directory on the H200 server.
EOF
  exit 2
fi

if [[ ! -f "${H200_CALVIN_EVAL_DATASET_PATH}/validation/.hydra/merged_config.yaml" \
   && ! -f "${H200_CALVIN_EVAL_DATASET_PATH}/training/.hydra/merged_config.yaml" \
   && ! -f "${H200_CALVIN_EVAL_DATASET_PATH}/.hydra/merged_config.yaml" ]]; then
  echo "Invalid H200_CALVIN_EVAL_DATASET_PATH: ${H200_CALVIN_EVAL_DATASET_PATH}" >&2
  echo "Expected validation/.hydra/merged_config.yaml, training/.hydra/merged_config.yaml, or .hydra/merged_config.yaml." >&2
  exit 2
fi

"${STAR_VLA_PYTHON}" - <<'PY'
from __future__ import annotations

import os
import re
from pathlib import Path

root = Path(os.environ["EVAL_BATCH_ROOT"])
expected_step = int(os.environ["EVAL_EXPECTED_STEP"])
expected_jobs = [line.strip() for line in os.environ["EXPECTED_JOBS"].splitlines() if line.strip()]
pattern = re.compile(r"steps_(\d+)_pytorch_model\.pt$")

errors: list[str] = []
for job in expected_jobs:
    job_root = root / job
    if not job_root.is_dir():
        errors.append(f"{job}: missing job directory {job_root}")
        continue
    candidates: list[tuple[int, Path]] = []
    for ckpt in job_root.rglob("steps_*_pytorch_model.pt"):
        match = pattern.search(ckpt.name)
        if match:
            candidates.append((int(match.group(1)), ckpt))
    if not candidates:
        errors.append(f"{job}: no steps_*_pytorch_model.pt checkpoint")
        continue
    step, ckpt = max(candidates, key=lambda item: item[0])
    if step < expected_step:
        errors.append(f"{job}: latest step {step} < expected {expected_step}; latest={ckpt}")

if errors:
    print("PI-State rollout checkpoint preflight failed:", flush=True)
    for item in errors:
        print(f"  - {item}", flush=True)
    raise SystemExit(2)

print(f"PI-State rollout checkpoint preflight OK: expected_step>={expected_step}, jobs={len(expected_jobs)}")
PY

bash examples/calvin/eval_files/eval_h200_qwen_mix_latest.sh
