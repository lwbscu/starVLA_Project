#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}
export MUJOCO_GL=${MUJOCO_GL:-osmesa}

RUN_ID=${RUN_ID:-calvin_eval_debug}
RUN_TS=${RUN_TS:-$(date +"%Y%m%d_%H%M%S")}
LOG_DIR=${LOG_DIR:-logs/log_${RUN_TS}_${RUN_ID}}
CALVIN_PYTHON=${CALVIN_PYTHON:-"$(conda info --base)/envs/calvin/bin/python"}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-5694}
UNNORM_KEY=${UNNORM_KEY:-franka}
NUM_SEQUENCES=${NUM_SEQUENCES:-1}
CKPT_PATH=${CKPT_PATH:-}
DATASET_PATH=${DATASET_PATH:-/home/lwb/Projects/SII/starVLA_Projects/calvin/dataset/calvin_debug_dataset}
CALVIN_CONFIG_PATH=${CALVIN_CONFIG_PATH:-/home/lwb/Projects/SII/starVLA_Projects/calvin/calvin_models/conf}
EVAL_SEQUENCES_PATH=${EVAL_SEQUENCES_PATH:-examples/calvin/eval_files/eval_sequences.json}

mkdir -p "${LOG_DIR}"/{train,eval,terminal,mp4,configs,metrics,checkpoints}
cp "$0" "${LOG_DIR}/configs/"

{
  echo "LOG_DIR=${LOG_DIR}"
  echo "DATASET_PATH=${DATASET_PATH}"
  echo "NUM_SEQUENCES=${NUM_SEQUENCES}"
  echo "HOST=${HOST}"
  echo "PORT=${PORT}"

  "${CALVIN_PYTHON}" examples/calvin/eval_files/eval_calvin.py \
    --args.pretrained-path "${CKPT_PATH}" \
    --args.unnorm-key "${UNNORM_KEY}" \
    --args.host "${HOST}" \
    --args.port "${PORT}" \
    --args.dataset-path "${DATASET_PATH}" \
    --args.calvin-config-path "${CALVIN_CONFIG_PATH}" \
    --args.eval-sequences-path "${EVAL_SEQUENCES_PATH}" \
    --args.num-sequences "${NUM_SEQUENCES}" \
    --args.eval-log-dir "${LOG_DIR}/mp4" \
    --args.debug
} 2>&1 | tee "${LOG_DIR}/terminal/eval.log"
