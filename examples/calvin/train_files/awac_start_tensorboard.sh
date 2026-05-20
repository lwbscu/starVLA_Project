#!/usr/bin/env bash
# Start TensorBoard for an AWAC run (call before training so port is ready for SSH forward).
set -euo pipefail

TB_PORT="${TB_PORT:-6006}"
TB_LOGDIR="${1:-}"
if [[ -z "${TB_LOGDIR}" ]]; then
  echo "Usage: TB_PORT=6006 bash awac_start_tensorboard.sh <tensorboard_log_dir>" >&2
  exit 1
fi

mkdir -p "${TB_LOGDIR}"
STAR_VLA_PYTHON="${STAR_VLA_PYTHON:-python}"

if command -v tensorboard >/dev/null 2>&1; then
  TB_CMD=(tensorboard)
else
  TB_CMD=("${STAR_VLA_PYTHON}" -m tensorboard.main)
fi

# Stop previous TensorBoard on the same port (best effort).
pkill -f "tensorboard.*--port[ =]${TB_PORT}" 2>/dev/null || true
sleep 1

nohup "${TB_CMD[@]}" --logdir "${TB_LOGDIR}" --port "${TB_PORT}" --bind_all \
  > "${TB_LOGDIR}/tensorboard_server.log" 2>&1 &
TB_PID=$!

echo "[tensorboard] started pid=${TB_PID}"
echo "[tensorboard] logdir=${TB_LOGDIR}"
echo "[tensorboard] port=${TB_PORT}"
echo "[tensorboard] H200 本机浏览器: http://127.0.0.1:${TB_PORT}"
echo "[tensorboard] 笔记本转发（在本地终端执行）:"
echo "  ssh -N -L ${TB_PORT}:127.0.0.1:${TB_PORT} <你的用户名>@<H200主机>"
echo "  然后打开 http://127.0.0.1:${TB_PORT}"
