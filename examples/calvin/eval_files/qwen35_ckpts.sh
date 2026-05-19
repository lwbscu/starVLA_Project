#!/usr/bin/env bash
# Checkpoint set for examples/calvin/eval_files/run_calvin_qwen35_cot_eval.sh.
#
# To switch weights or action heads, edit the three CKPT_* paths below. You may
# also copy this file and run with CKPT_CONFIG=/path/to/your_ckpts.sh.

CKPT_QWEN35_0P8B="${CKPT_QWEN35_0P8B:-/inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project/logs/20260519_h200_server1_pi_state_30k_v1/server1_pi_state/qwen35_0p8b/pi_state_qwen35_0p8b_30000step/checkpoints/pi_state_qwen35_0p8b_30000step/checkpoints/steps_30000_pytorch_model.pt}"
CKPT_QWEN35_4B="${CKPT_QWEN35_4B:-/inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project/logs/20260519_h200_server1_pi_state_30k_v1/server1_pi_state/qwen35_4b/pi_state_qwen35_4b_30000step/checkpoints/pi_state_qwen35_4b_30000step/checkpoints/steps_30000_pytorch_model.pt}"
CKPT_QWEN35_9B="${CKPT_QWEN35_9B:-/inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project/logs/20260519_h200_server1_pi_state_30k_v1/server1_pi_state/qwen35_9b/pi_state_qwen35_9b_30000step/checkpoints/pi_state_qwen35_9b_30000step/checkpoints/steps_10000_pytorch_model.pt}"

MODEL_IDS=(qwen35_0p8b qwen35_4b qwen35_9b)
MODEL_LABELS=("0.8B" "4B" "9B")
MODEL_CKPTS=("${CKPT_QWEN35_0P8B}" "${CKPT_QWEN35_4B}" "${CKPT_QWEN35_9B}")
