#!/usr/bin/env bash
# Run configured Qwen3.5 CALVIN eval checkpoints with CoT off/on and generate a comparison report.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CALVIN_FORCE_NO_EGL="${CALVIN_FORCE_NO_EGL:-1}"

STAR_VLA_PYTHON="${STAR_VLA_PYTHON:-${star_vla_python:-python}}"
CALVIN_PYTHON="${CALVIN_PYTHON:-${calvin_python:-${STAR_VLA_PYTHON}}}"
REPORT_PYTHON="${REPORT_PYTHON:-${CALVIN_PYTHON}}"

DATASET_PATH="${DATASET_PATH:-/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_d_d}"
CALVIN_CONFIG_PATH="${CALVIN_CONFIG_PATH:-${PROJECT_ROOT}/calvin/calvin_models/conf}"
EVAL_SEQUENCES_PATH="${EVAL_SEQUENCES_PATH:-examples/calvin/eval_files/eval_sequences.json}"
NUM_SEQUENCES="${NUM_SEQUENCES:-1000}"
UNNORM_KEY="${UNNORM_KEY:-franka}"
HOST="${HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-5694}"

SERVER_GPU_ID="${SERVER_GPU_ID:-0}"
EVAL_GPU_ID="${EVAL_GPU_ID:-${SERVER_GPU_ID}}"
USE_BF16="${USE_BF16:-1}"
WRITE_MP4="${WRITE_MP4:-1}"
DEBUG_EVAL="${DEBUG_EVAL:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
ONLY_SUMMARY="${ONLY_SUMMARY:-0}"
SERVER_STARTUP_TIMEOUT="${SERVER_STARTUP_TIMEOUT:-1800}"
SERVER_IDLE_TIMEOUT="${SERVER_IDLE_TIMEOUT:--1}"
MIRROR_WEIGHT_MODE="${MIRROR_WEIGHT_MODE:-symlink}" # symlink, hardlink, or copy

RUN_ID_PREFIX="${RUN_ID_PREFIX:-calvin_qwen35_30000_$(date +%Y%m%d_%H%M%S)}"
MIRROR_ROOT="${MIRROR_ROOT:-logs/${RUN_ID_PREFIX}_ckpt_mirrors}"
MANIFEST_PATH="${MANIFEST_PATH:-logs/${RUN_ID_PREFIX}_manifest.tsv}"
REPORT_PATH="${REPORT_PATH:-logs/${RUN_ID_PREFIX}_report.md}"
COT_PROMPT="${COT_PROMPT:-Your task is {instruction}. To identify the key objects for your task. Locate their bounding boxes in [x1,y1,x2,y2] format.}"

CKPT_CONFIG="${CKPT_CONFIG:-${SCRIPT_DIR}/qwen35_ckpts.sh}"
if [[ ! -f "${CKPT_CONFIG}" ]]; then
  echo "[ERROR] CKPT_CONFIG not found: ${CKPT_CONFIG}" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${CKPT_CONFIG}"
if [[ -z "${MODEL_IDS+x}" || -z "${MODEL_LABELS+x}" || -z "${MODEL_CKPTS+x}" ]]; then
  echo "[ERROR] CKPT_CONFIG must define MODEL_IDS, MODEL_LABELS, and MODEL_CKPTS arrays: ${CKPT_CONFIG}" >&2
  exit 1
fi
if (( ${#MODEL_IDS[@]} != ${#MODEL_LABELS[@]} || ${#MODEL_IDS[@]} != ${#MODEL_CKPTS[@]} )); then
  echo "[ERROR] CKPT_CONFIG arrays must have the same length: ${CKPT_CONFIG}" >&2
  exit 1
fi
COT_MODES=(CoT_off CoT_on)

CURRENT_SERVER_PID=""

info() { echo "[INFO] $*"; }
warn() { echo "[WARN] $*" >&2; }
die() { echo "[ERROR] $*" >&2; exit 1; }

is_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

stop_server() {
  if [[ -n "${CURRENT_SERVER_PID}" ]] && kill -0 "${CURRENT_SERVER_PID}" 2>/dev/null; then
    info "Stopping policy server pid=${CURRENT_SERVER_PID}"
    kill "${CURRENT_SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "${CURRENT_SERVER_PID}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "${CURRENT_SERVER_PID}" 2>/dev/null; then
      warn "Policy server pid=${CURRENT_SERVER_PID} did not exit; sending SIGKILL"
      kill -9 "${CURRENT_SERVER_PID}" 2>/dev/null || true
    fi
    wait "${CURRENT_SERVER_PID}" 2>/dev/null || true
  fi
  CURRENT_SERVER_PID=""
}

trap 'stop_server' EXIT
trap 'stop_server; exit 130' INT TERM

require_file() {
  local label="$1"
  local path="$2"
  [[ -f "${path}" ]] || die "${label} not found: ${path}"
}

require_dir() {
  local label="$1"
  local path="$2"
  [[ -d "${path}" ]] || die "${label} not found: ${path}"
}

checkpoint_run_dir() {
  local ckpt="$1"
  local ckpt_dir
  ckpt_dir="$(cd -- "$(dirname -- "${ckpt}")" && pwd -P)" || return 1
  cd -- "${ckpt_dir}/.." && pwd -P
}

validate_checkpoint_bundle() {
  local label="$1"
  local ckpt="$2"
  require_file "${label} checkpoint" "${ckpt}"
  local run_dir
  run_dir="$(checkpoint_run_dir "${ckpt}")" || die "Cannot resolve run_dir for ${label}: ${ckpt}"
  require_file "${label} config.yaml" "${run_dir}/config.yaml"
  require_file "${label} dataset_statistics.json" "${run_dir}/dataset_statistics.json"
}

validate_calvin_dataset() {
  require_dir "CALVIN dataset root" "${DATASET_PATH}"
  if [[ -f "${DATASET_PATH}/validation/.hydra/merged_config.yaml" ]]; then
    return 0
  fi
  if [[ -f "${DATASET_PATH}/training/.hydra/merged_config.yaml" ]]; then
    warn "Using CALVIN training split config because validation/.hydra/merged_config.yaml was not found."
    return 0
  fi
  if [[ -f "${DATASET_PATH}/.hydra/merged_config.yaml" ]]; then
    warn "Using CALVIN root .hydra config because validation/.hydra/merged_config.yaml was not found."
    return 0
  fi
  die "No CALVIN .hydra/merged_config.yaml found under ${DATASET_PATH}"
}

validate_eval_sequences() {
  require_file "CALVIN eval sequences" "${EVAL_SEQUENCES_PATH}"
  local available
  available="$("${REPORT_PYTHON}" - "${EVAL_SEQUENCES_PATH}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
print(len(json.loads(path.read_text(encoding="utf-8"))))
PY
)"
  if [[ "${available}" != "1000" ]]; then
    warn "eval_sequences.json contains ${available} sequences, not 1000."
  fi
  if (( NUM_SEQUENCES > available )); then
    die "NUM_SEQUENCES=${NUM_SEQUENCES} exceeds available sequences=${available}"
  fi
}

preflight() {
  info "Preflight checks in ${PROJECT_ROOT}"
  validate_calvin_dataset
  require_dir "CALVIN config dir" "${CALVIN_CONFIG_PATH}"
  require_file "CALVIN task oracle config" "${CALVIN_CONFIG_PATH}/callbacks/rollout/tasks/new_playtable_tasks.yaml"
  require_file "CALVIN validation annotations" "${CALVIN_CONFIG_PATH}/annotations/new_playtable_validation.yaml"
  validate_eval_sequences
  for i in "${!MODEL_IDS[@]}"; do
    validate_checkpoint_bundle "${MODEL_IDS[$i]}" "${MODEL_CKPTS[$i]}"
  done
}

patch_cot_config() {
  local config_path="$1"
  "${STAR_VLA_PYTHON}" - "${config_path}" "${COT_PROMPT}" <<'PY'
import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
prompt = sys.argv[2]

try:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(str(path))
    try:
        OmegaConf.update(cfg, "datasets.vla_data.CoT_prompt", prompt, merge=True, force_add=True)
    except TypeError:
        OmegaConf.update(cfg, "datasets.vla_data.CoT_prompt", prompt, merge=True)
    with path.open("w", encoding="utf-8") as handle:
        OmegaConf.save(config=cfg, f=handle)
    raise SystemExit(0)
except SystemExit:
    raise
except Exception:
    pass

text = path.read_text(encoding="utf-8")
quoted = json.dumps(prompt)
lines = text.splitlines()
for index, line in enumerate(lines):
    if re.match(r"^\s*CoT_prompt\s*:", line):
        indent = re.match(r"^(\s*)", line).group(1)
        lines[index] = f"{indent}CoT_prompt: {quoted}"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        raise SystemExit(0)

for index, line in enumerate(lines):
    if re.match(r"^\s{2}vla_data\s*:\s*(#.*)?$", line):
        lines.insert(index + 1, f"    CoT_prompt: {quoted}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        raise SystemExit(0)

with path.open("a", encoding="utf-8") as handle:
    handle.write("\ndatasets:\n  vla_data:\n")
    handle.write(f"    CoT_prompt: {quoted}\n")
PY
}

link_or_copy_weight() {
  local src="$1"
  local dst="$2"
  if [[ -e "${dst}" || -L "${dst}" ]]; then
    return 0
  fi

  case "${MIRROR_WEIGHT_MODE}" in
    symlink)
      ln -s "${src}" "${dst}" 2>/dev/null || cp "${src}" "${dst}"
      ;;
    hardlink)
      ln "${src}" "${dst}" 2>/dev/null || cp "${src}" "${dst}"
      ;;
    copy)
      cp "${src}" "${dst}"
      ;;
    *)
      die "Unsupported MIRROR_WEIGHT_MODE=${MIRROR_WEIGHT_MODE}; use symlink, hardlink, or copy"
      ;;
  esac
}

prepare_cot_on_mirror() {
  local model_id="$1"
  local original_ckpt="$2"
  local src_run_dir
  src_run_dir="$(checkpoint_run_dir "${original_ckpt}")" || die "Cannot resolve run_dir: ${original_ckpt}"

  local dst_run_dir="${MIRROR_ROOT}/${model_id}_CoT_on"
  local dst_ckpt="${dst_run_dir}/checkpoints/$(basename -- "${original_ckpt}")"
  mkdir -p "${dst_run_dir}/checkpoints"
  cp "${src_run_dir}/config.yaml" "${dst_run_dir}/config.yaml"
  cp "${src_run_dir}/dataset_statistics.json" "${dst_run_dir}/dataset_statistics.json"
  link_or_copy_weight "${original_ckpt}" "${dst_ckpt}"
  patch_cot_config "${dst_run_dir}/config.yaml"
  echo "${dst_ckpt}"
}

write_manifest_header() {
  mkdir -p "$(dirname -- "${MANIFEST_PATH}")"
  printf "model_id\tmodel_label\tcot\trun_id\tport\tstatus\tlog_dir\tresult_json\teval_log\tserver_log\tckpt_path\toriginal_ckpt\n" > "${MANIFEST_PATH}"
}

append_manifest() {
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$@" >> "${MANIFEST_PATH}"
}

wait_for_port() {
  local host="$1"
  local port="$2"
  local pid="$3"
  local timeout_s="$4"
  local deadline=$((SECONDS + timeout_s))
  while (( SECONDS < deadline )); do
    if (echo > /dev/tcp/"${host}"/"${port}") >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      return 1
    fi
    sleep 2
  done
  return 1
}

run_summarizer() {
  require_file "manifest" "${MANIFEST_PATH}"
  mkdir -p "$(dirname -- "${REPORT_PATH}")"
  "${REPORT_PYTHON}" examples/calvin/eval_files/summarize_calvin_cot_eval.py \
    --manifest "${MANIFEST_PATH}" \
    --output "${REPORT_PATH}"
}

run_one() {
  local model_id="$1"
  local model_label="$2"
  local cot="$3"
  local original_ckpt="$4"
  local port="$5"

  local eval_ckpt="${original_ckpt}"
  if [[ "${cot}" == "CoT_on" ]]; then
    eval_ckpt="$(prepare_cot_on_mirror "${model_id}" "${original_ckpt}")"
  fi

  local run_id="${RUN_ID_PREFIX}_${model_id}_${cot}"
  local log_dir="logs/${run_id}"
  local terminal_dir="${log_dir}/terminal"
  local mp4_dir="${log_dir}/mp4"
  local result_json="${mp4_dir}/results.json"
  local eval_log="${terminal_dir}/eval.log"
  local server_log="${terminal_dir}/server.log"
  local status="planned"

  mkdir -p "${terminal_dir}" "${mp4_dir}"
  {
    echo "run_id=${run_id}"
    echo "model_id=${model_id}"
    echo "model_label=${model_label}"
    echo "cot=${cot}"
    echo "port=${port}"
    echo "ckpt_path=${eval_ckpt}"
    echo "original_ckpt=${original_ckpt}"
    echo "dataset_path=${DATASET_PATH}"
    echo "calvin_config_path=${CALVIN_CONFIG_PATH}"
    echo "eval_sequences_path=${EVAL_SEQUENCES_PATH}"
    echo "num_sequences=${NUM_SEQUENCES}"
    echo "write_mp4=${WRITE_MP4}"
  } > "${terminal_dir}/run.env"

  if is_truthy "${SKIP_COMPLETED}" && [[ -f "${result_json}" ]]; then
    info "Skip completed ${run_id}: ${result_json}"
    append_manifest "${model_id}" "${model_label}" "${cot}" "${run_id}" "${port}" "skipped_completed" \
      "${log_dir}" "${result_json}" "${eval_log}" "${server_log}" "${eval_ckpt}" "${original_ckpt}"
    return 0
  fi

  if is_truthy "${DRY_RUN}"; then
    info "DRY_RUN ${run_id}: ckpt=${eval_ckpt}, port=${port}, log_dir=${log_dir}"
    append_manifest "${model_id}" "${model_label}" "${cot}" "${run_id}" "${port}" "dry_run" \
      "${log_dir}" "${result_json}" "${eval_log}" "${server_log}" "${eval_ckpt}" "${original_ckpt}"
    return 0
  fi

  info "Starting policy server for ${run_id} on port ${port}"
  local server_cmd=(
    "${STAR_VLA_PYTHON}" deployment/model_server/server_policy.py
    --ckpt_path "${eval_ckpt}"
    --port "${port}"
    --idle_timeout "${SERVER_IDLE_TIMEOUT}"
  )
  if is_truthy "${USE_BF16}"; then
    server_cmd+=(--use_bf16)
  fi

  CUDA_VISIBLE_DEVICES="${SERVER_GPU_ID}" "${server_cmd[@]}" > "${server_log}" 2>&1 &
  CURRENT_SERVER_PID=$!

  if ! wait_for_port "${HOST}" "${port}" "${CURRENT_SERVER_PID}" "${SERVER_STARTUP_TIMEOUT}"; then
    warn "Policy server failed to become ready for ${run_id}; see ${server_log}"
    status="server_start_failed"
    stop_server
    append_manifest "${model_id}" "${model_label}" "${cot}" "${run_id}" "${port}" "${status}" \
      "${log_dir}" "${result_json}" "${eval_log}" "${server_log}" "${eval_ckpt}" "${original_ckpt}"
    is_truthy "${CONTINUE_ON_ERROR}" && return 0
    return 1
  fi

  info "Running CALVIN eval for ${run_id}"
  local eval_cmd=(
    "${CALVIN_PYTHON}" ./examples/calvin/eval_files/eval_calvin_save_videos.py
    --args.pretrained-path "${eval_ckpt}"
    --args.unnorm-key "${UNNORM_KEY}"
    --args.host "${HOST}"
    --args.port "${port}"
    --args.dataset-path "${DATASET_PATH}"
    --args.calvin-config-path "${CALVIN_CONFIG_PATH}"
    --args.eval-sequences-path "${EVAL_SEQUENCES_PATH}"
    --args.num-sequences "${NUM_SEQUENCES}"
    --args.eval-log-dir "${mp4_dir}"
  )
  if is_truthy "${WRITE_MP4}"; then
    eval_cmd+=(--args.save-videos)
  fi
  if is_truthy "${DEBUG_EVAL}"; then
    eval_cmd+=(--args.debug)
  fi

  CUDA_VISIBLE_DEVICES="${EVAL_GPU_ID}" "${eval_cmd[@]}" > "${eval_log}" 2>&1
  local eval_status=$?
  stop_server

  if [[ "${eval_status}" -eq 0 && -f "${result_json}" ]]; then
    status="completed"
  elif [[ "${eval_status}" -eq 0 ]]; then
    status="missing_results"
  else
    status="eval_failed"
  fi

  info "Finished ${run_id}: status=${status}, result=${result_json}"
  append_manifest "${model_id}" "${model_label}" "${cot}" "${run_id}" "${port}" "${status}" \
    "${log_dir}" "${result_json}" "${eval_log}" "${server_log}" "${eval_ckpt}" "${original_ckpt}"

  if [[ "${status}" != "completed" ]] && [[ "${status}" != "skipped_completed" ]]; then
    is_truthy "${CONTINUE_ON_ERROR}" && return 0
    return 1
  fi
}

if is_truthy "${ONLY_SUMMARY}"; then
  run_summarizer
  exit 0
fi

if ! is_truthy "${SKIP_PREFLIGHT}"; then
  preflight
fi

mkdir -p "logs" "${MIRROR_ROOT}"
write_manifest_header

run_index=0
for i in "${!MODEL_IDS[@]}"; do
  for cot in "${COT_MODES[@]}"; do
    port=$((BASE_PORT + run_index))
    run_one "${MODEL_IDS[$i]}" "${MODEL_LABELS[$i]}" "${cot}" "${MODEL_CKPTS[$i]}" "${port}" || exit 1
    run_index=$((run_index + 1))
  done
done

info "Generating report"
run_summarizer
info "Report: ${REPORT_PATH}"
