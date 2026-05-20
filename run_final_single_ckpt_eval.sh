#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   CKPT_PATH=/absolute/path/to/steps_2000_pytorch_model.pt \
#   H200_CALVIN_EVAL_DATASET_PATH=/absolute/path/to/task_D_D \
#   NUM_SEQUENCES=5 \
#   WRITE_MP4=1 \
#   ./run_final_single_ckpt_eval.sh
#
# Parameters:
#   CKPT_PATH: checkpoint file to evaluate.
#   H200_CALVIN_EVAL_DATASET_PATH: CALVIN eval dataset root, usually task_D_D or task_ABC_D.
#   NUM_SEQUENCES: number of CALVIN long-horizon task chains to evaluate; default is 5.
#   WRITE_MP4: 1 saves rollout mp4 videos; 0 disables mp4 video output. Default is 1.
#   EVAL_GPU: GPU id used by policy server and CALVIN eval. Default is 0.
#   EVAL_PORT: localhost port for the policy server. Default is 6200.
#   BASE_VLM_OVERRIDE: local Qwen base model used when the checkpoint config points to
#     a missing path. Default is the H200 public Qwen3.5-4B path. The script patches
#     only an eval-time copy of config.yaml under the run dir; it never edits the
#     original training checkpoint directory.

PROJECT_ROOT="${PROJECT_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project}"
CONDA_ROOT="${CONDA_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/26220216/miniconda3}"
cd "${PROJECT_ROOT}"

export STAR_VLA_PYTHON="${STAR_VLA_PYTHON:-${CONDA_ROOT}/envs/starVLA_qwen35/bin/python}"
export CALVIN_PYTHON="${CALVIN_PYTHON:-${CONDA_ROOT}/envs/calvin/bin/python}"

export CKPT_PATH="${CKPT_PATH:-${PROJECT_ROOT}/logs/route_validation/log_20260520_061136_hlx_pi_state_lora_sem_qwen35_4b_8gpu_b16_acc2_20000step/checkpoints/hlx_pi_state_lora_sem_qwen35_4b_8gpu_b16_acc2_20000step/checkpoints/steps_1000_pytorch_model.pt}"
export NUM_SEQUENCES="${NUM_SEQUENCES:-5}"
export WRITE_MP4="${WRITE_MP4:-1}"
export EVAL_GPU="${EVAL_GPU:-0}"
export EVAL_PORT="${EVAL_PORT:-6200}"
export UNNORM_KEY="${UNNORM_KEY:-franka}"
export SEND_STATE_TO_POLICY="${SEND_STATE_TO_POLICY:-1}"

export H200_CALVIN_DATA_ROOT="${H200_CALVIN_DATA_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d}"
export H200_CALVIN_DATA_NAME="${H200_CALVIN_DATA_NAME:-calvin_task_ABC_D}"
export EVAL_SEQUENCES_PATH="${EVAL_SEQUENCES_PATH:-examples/calvin/eval_files/eval_sequences.json}"
export H200_PUBLIC_QWEN35_4B="${H200_PUBLIC_QWEN35_4B:-/inspire/qb-ilm2/project/26summer-camp-10/public/Qwen/Qwen3.5-4B}"
export BASE_VLM_OVERRIDE="${BASE_VLM_OVERRIDE:-${H200_PUBLIC_QWEN35_4B}}"

export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export GIT_PYTHON_REFRESH="${GIT_PYTHON_REFRESH:-quiet}"
export CALVIN_ALLOW_OFFLINE_GIT_METADATA="${CALVIN_ALLOW_OFFLINE_GIT_METADATA:-1}"
export CALVIN_FORCE_NO_EGL="${CALVIN_FORCE_NO_EGL:-1}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
CKPT_BASENAME="$(basename "${CKPT_PATH}")"
CKPT_TAG="${CKPT_BASENAME%_pytorch_model.pt}"
CKPT_TAG="${CKPT_TAG%.pt}"
CKPT_TAG="${CKPT_TAG%.safetensors}"
RUN_ID="${RUN_ID:-final_eval_pi_state_lora_sem_${CKPT_TAG}_${NUM_SEQUENCES}seq_mp4${WRITE_MP4}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/final_single_ckpt_eval}"
RUN_DIR="${OUTPUT_ROOT}/log_${RUN_TS}_${RUN_ID}"
SERVER_DIR="${RUN_DIR}/policy_server"
EVAL_DIR="${RUN_DIR}/eval"
MP4_DIR="${EVAL_DIR}/mp4"
mkdir -p "${SERVER_DIR}/terminal" "${EVAL_DIR}/terminal" "${MP4_DIR}" "${RUN_DIR}/configs"

resolve_first_existing_dir() {
  local candidate
  for candidate in "$@"; do
    if [[ -d "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

is_calvin_eval_dataset() {
  local candidate="$1"
  [[ -f "${candidate}/validation/.hydra/merged_config.yaml" ]] \
    || [[ -f "${candidate}/training/.hydra/merged_config.yaml" ]] \
    || [[ -f "${candidate}/.hydra/merged_config.yaml" ]]
}

resolve_eval_dataset() {
  if [[ -n "${H200_CALVIN_EVAL_DATASET_PATH:-}" ]]; then
    echo "${H200_CALVIN_EVAL_DATASET_PATH}"
    return 0
  fi

  local candidate
  for candidate in \
    "${H200_CALVIN_DATA_ROOT%/}/task_D_D" \
    "${H200_CALVIN_DATA_ROOT%/}/task_ABC_D" \
    "${PROJECT_ROOT}/calvin/dataset/task_D_D" \
    "${PROJECT_ROOT}/calvin/dataset/task_ABC_D" \
    "${PROJECT_ROOT}/calvin/dataset/calvin_debug_dataset" \
    "${PROJECT_ROOT}/../calvin/dataset/task_D_D" \
    "${PROJECT_ROOT}/../calvin/dataset/task_ABC_D" \
    "${PROJECT_ROOT}/../calvin/dataset/calvin_debug_dataset" \
    "${H200_CALVIN_DATA_ROOT%/}/${H200_CALVIN_DATA_NAME}"
  do
    if [[ -d "${candidate}" ]] && is_calvin_eval_dataset "${candidate}"; then
      echo "${candidate}"
      return 0
    fi
  done

  echo "No CALVIN eval dataset found. Set H200_CALVIN_EVAL_DATASET_PATH explicitly." >&2
  return 2
}

prepare_eval_checkpoint() {
  CKPT_PATH="${CKPT_PATH}" \
  PROJECT_ROOT="${PROJECT_ROOT}" \
  RUN_DIR="${RUN_DIR}" \
  BASE_VLM_OVERRIDE="${BASE_VLM_OVERRIDE}" \
  H200_PUBLIC_QWEN35_4B="${H200_PUBLIC_QWEN35_4B}" \
  "${STAR_VLA_PYTHON}" - <<'PY'
import os
import shutil
import sys
from pathlib import Path

from omegaconf import OmegaConf


def die(message: str, status: int = 2) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(status)


def config_path_is_valid(value: str, project_root: Path) -> Path | None:
    if not value:
        return None
    raw = Path(value).expanduser()
    candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw, project_root / raw]
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "config.json").is_file():
            return candidate.resolve()
    return None


ckpt = Path(os.environ["CKPT_PATH"]).expanduser().resolve()
project_root = Path(os.environ["PROJECT_ROOT"]).expanduser().resolve()
run_dir = Path(os.environ["RUN_DIR"]).expanduser().resolve()
override = os.environ.get("BASE_VLM_OVERRIDE", "")
public_qwen = os.environ.get("H200_PUBLIC_QWEN35_4B", "")

if not ckpt.is_file():
    die(f"Checkpoint not found: {ckpt}")
if ckpt.suffix not in {".pt", ".safetensors"}:
    die(f"Checkpoint suffix must be .pt or .safetensors: {ckpt}")

source_run_dir = ckpt.parents[1]
source_config = source_run_dir / "config.yaml"
source_stats = source_run_dir / "dataset_statistics.json"
if not source_config.is_file():
    die(f"Missing source config.yaml for checkpoint run dir: {source_config}")
if not source_stats.is_file():
    die(f"Missing source dataset_statistics.json for checkpoint run dir: {source_stats}")

cfg = OmegaConf.load(str(source_config))
try:
    original_base = str(cfg.framework.qwenvl.base_vlm)
except Exception as exc:
    die(f"Checkpoint config has no framework.qwenvl.base_vlm: {source_config} ({exc})")

original_base_valid = config_path_is_valid(original_base, project_root)
if original_base_valid is not None:
    print(f"[eval-base-vlm] checkpoint base_vlm exists: {original_base_valid}", file=sys.stderr)
    print(str(ckpt))
    raise SystemExit(0)

replacement_candidates: list[str] = []
for candidate in (
    override,
    public_qwen,
    str(project_root / "playground" / "Pretrained_models" / "Qwen3.5-4B"),
):
    if candidate and candidate not in replacement_candidates:
        replacement_candidates.append(candidate)

replacement: Path | None = None
for candidate in replacement_candidates:
    resolved = config_path_is_valid(candidate, project_root)
    if resolved is not None:
        replacement = resolved
        break

if replacement is None:
    candidate_lines = "\n  ".join(replacement_candidates) or "<none>"
    die(
        "Checkpoint config points to a missing base_vlm, and no valid replacement was found.\n"
        f"  checkpoint={ckpt}\n"
        f"  config_base_vlm={original_base}\n"
        f"  checked_replacements:\n  {candidate_lines}\n"
        "Set BASE_VLM_OVERRIDE=/absolute/path/to/Qwen3.5-4B with config.json present."
    )

patched_run_dir = run_dir / "patched_checkpoint"
patched_ckpt_dir = patched_run_dir / "checkpoints"
patched_ckpt_dir.mkdir(parents=True, exist_ok=True)

patched_config = patched_run_dir / "config.yaml"
patched_stats = patched_run_dir / "dataset_statistics.json"
patched_ckpt = patched_ckpt_dir / ckpt.name

cfg.framework.qwenvl.base_vlm = str(replacement)
OmegaConf.save(cfg, str(patched_config))
shutil.copy2(source_config, patched_run_dir / "config.original.yaml")
shutil.copy2(source_stats, patched_stats)

if patched_ckpt.exists() or patched_ckpt.is_symlink():
    patched_ckpt.unlink()
try:
    patched_ckpt.symlink_to(ckpt)
    ckpt_link_mode = "symlink"
except OSError:
    shutil.copy2(ckpt, patched_ckpt)
    ckpt_link_mode = "copy"

patch_note = run_dir / "configs" / "base_vlm_patch.txt"
patch_note.write_text(
    "\n".join(
        [
            f"original_ckpt={ckpt}",
            f"eval_ckpt={patched_ckpt}",
            f"source_config={source_config}",
            f"original_base_vlm={original_base}",
            f"replacement_base_vlm={replacement}",
            f"checkpoint_reference_mode={ckpt_link_mode}",
        ]
    )
    + "\n",
    encoding="utf-8",
)

print(
    f"[eval-base-vlm] patched eval config: {original_base} -> {replacement}",
    file=sys.stderr,
)
print(str(patched_ckpt))
PY
}

cleanup_port() {
  "${STAR_VLA_PYTHON}" examples/calvin/train_files/cleanup_policy_server.py \
    --port "${EVAL_PORT}" \
    --timeout 10 \
    >/dev/null 2>&1 || true
}

wait_for_policy_server() {
  local server_pid="$1"
  local deadline=$((SECONDS + 600))
  while (( SECONDS < deadline )); do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "Policy server exited before ready. See ${SERVER_DIR}/terminal/policy_server.log" >&2
      return 1
    fi
    if EVAL_PORT="${EVAL_PORT}" "${STAR_VLA_PYTHON}" - <<'PY' >/dev/null 2>&1
import os
import socket
port = int(os.environ["EVAL_PORT"])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(0.5)
try:
    sock.connect(("127.0.0.1", port))
finally:
    sock.close()
PY
    then
      return 0
    fi
    sleep 2
  done
  echo "Timed out waiting for policy server on port ${EVAL_PORT}." >&2
  return 1
}

if [[ ! -x "${STAR_VLA_PYTHON}" ]]; then
  echo "STAR_VLA_PYTHON not executable: ${STAR_VLA_PYTHON}" >&2
  exit 2
fi
if [[ ! -x "${CALVIN_PYTHON}" ]]; then
  echo "CALVIN_PYTHON not executable: ${CALVIN_PYTHON}" >&2
  exit 2
fi
if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "Checkpoint not found: ${CKPT_PATH}" >&2
  exit 2
fi
if [[ ! -f "${EVAL_SEQUENCES_PATH}" ]]; then
  echo "EVAL_SEQUENCES_PATH not found: ${EVAL_SEQUENCES_PATH}" >&2
  exit 2
fi

CALVIN_ROOT="${CALVIN_ROOT:-$(resolve_first_existing_dir "${PROJECT_ROOT}/calvin" "${PROJECT_ROOT}/../calvin" || true)}"
CALVIN_CONFIG_PATH="${CALVIN_CONFIG_PATH:-$(resolve_first_existing_dir "${PROJECT_ROOT}/calvin/calvin_models/conf" "${PROJECT_ROOT}/../calvin/calvin_models/conf" || true)}"
CALVIN_ASSET_ROOT="${CALVIN_ASSET_ROOT:-$(resolve_first_existing_dir "${PROJECT_ROOT}/calvin/calvin_env/data" "${PROJECT_ROOT}/../calvin/calvin_env/data" || true)}"
DATASET_PATH="$(resolve_eval_dataset)"
EVAL_CKPT_PATH="$(prepare_eval_checkpoint)"

if [[ ! -d "${DATASET_PATH}" ]] || ! is_calvin_eval_dataset "${DATASET_PATH}"; then
  echo "DATASET_PATH is not a valid CALVIN eval dataset: ${DATASET_PATH}" >&2
  exit 2
fi
if [[ -z "${CALVIN_CONFIG_PATH}" || ! -d "${CALVIN_CONFIG_PATH}" ]]; then
  echo "CALVIN_CONFIG_PATH not found. Set CALVIN_CONFIG_PATH manually." >&2
  exit 2
fi
if [[ -z "${CALVIN_ASSET_ROOT}" || ! -d "${CALVIN_ASSET_ROOT}" ]]; then
  echo "CALVIN_ASSET_ROOT not found. Set CALVIN_ASSET_ROOT manually." >&2
  exit 2
fi

export CALVIN_ROOT CALVIN_CONFIG_PATH CALVIN_ASSET_ROOT DATASET_PATH
export PYTHONPATH="${PROJECT_ROOT}:${CALVIN_ROOT:-}:${CALVIN_ROOT:-}/calvin_models:${CALVIN_ROOT:-}/calvin_env:${PYTHONPATH:-}"

{
  echo "run_dir=${RUN_DIR}"
  echo "ckpt_path=${CKPT_PATH}"
  echo "eval_ckpt_path=${EVAL_CKPT_PATH}"
  echo "base_vlm_override=${BASE_VLM_OVERRIDE}"
  echo "num_sequences=${NUM_SEQUENCES}"
  echo "write_mp4=${WRITE_MP4}"
  echo "send_state_to_policy=${SEND_STATE_TO_POLICY}"
  echo "eval_gpu=${EVAL_GPU}"
  echo "eval_port=${EVAL_PORT}"
  echo "dataset_path=${DATASET_PATH}"
  echo "calvin_config_path=${CALVIN_CONFIG_PATH}"
  echo "calvin_asset_root=${CALVIN_ASSET_ROOT}"
  echo "eval_sequences_path=${EVAL_SEQUENCES_PATH}"
  echo "unnorm_key=${UNNORM_KEY}"
} | tee "${RUN_DIR}/env.log"

cleanup_port

CKPT_PATH="${EVAL_CKPT_PATH}" \
PORT="${EVAL_PORT}" \
RUN_ID="policy_${RUN_ID}" \
LOG_DIR="${SERVER_DIR}" \
CUDA_VISIBLE_DEVICES="${EVAL_GPU}" \
STAR_VLA_PYTHON="${STAR_VLA_PYTHON}" \
bash examples/calvin/eval_files/run_policy_server_debug.sh &
SERVER_PID=$!

trap 'cleanup_port; kill "${SERVER_PID}" 2>/dev/null || true' EXIT

wait_for_policy_server "${SERVER_PID}"

eval_args=(
  examples/calvin/eval_files/eval_calvin.py
  --args.pretrained-path "${EVAL_CKPT_PATH}"
  --args.unnorm-key "${UNNORM_KEY}"
  --args.host 127.0.0.1
  --args.port "${EVAL_PORT}"
  --args.dataset-path "${DATASET_PATH}"
  --args.calvin-config-path "${CALVIN_CONFIG_PATH}"
  --args.eval-sequences-path "${EVAL_SEQUENCES_PATH}"
  --args.num-sequences "${NUM_SEQUENCES}"
  --args.eval-log-dir "${MP4_DIR}"
)

if [[ "${SEND_STATE_TO_POLICY}" == "1" ]]; then
  eval_args+=(--args.send-state-to-policy)
fi
if [[ "${WRITE_MP4}" == "1" ]]; then
  eval_args+=(--args.debug)
fi

set +e
CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${CALVIN_PYTHON}" "${eval_args[@]}" 2>&1 | tee "${EVAL_DIR}/terminal/eval.log"
EVAL_STATUS=${PIPESTATUS[0]}
set -e

cleanup_port
kill "${SERVER_PID}" 2>/dev/null || true
wait "${SERVER_PID}" 2>/dev/null || true

if [[ "${EVAL_STATUS}" -ne 0 ]]; then
  echo "Eval failed. See ${EVAL_DIR}/terminal/eval.log" >&2
  exit "${EVAL_STATUS}"
fi

RESULT_JSON="$(find "${MP4_DIR}" -type f -name results.json -print -quit)"
MP4_COUNT="$(find "${MP4_DIR}" -type f -name '*.mp4' | wc -l | tr -d ' ')"

if [[ -z "${RESULT_JSON}" ]]; then
  echo "Eval finished but results.json was not produced under ${MP4_DIR}" >&2
  exit 3
fi
if [[ "${WRITE_MP4}" == "1" && "${MP4_COUNT}" -lt 1 ]]; then
  echo "WRITE_MP4=1 but no mp4 was produced under ${MP4_DIR}" >&2
  exit 3
fi

{
  echo "status=OK"
  echo "run_dir=${RUN_DIR}"
  echo "original_ckpt_path=${CKPT_PATH}"
  echo "eval_ckpt_path=${EVAL_CKPT_PATH}"
  echo "result_json=${RESULT_JSON}"
  echo "mp4_count=${MP4_COUNT}"
  echo "eval_log=${EVAL_DIR}/terminal/eval.log"
  echo "server_log=${SERVER_DIR}/terminal/policy_server.log"
} | tee "${RUN_DIR}/summary.txt"

echo "Done."
