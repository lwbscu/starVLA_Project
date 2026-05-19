# H200 PIHead-LoRA 微调命令行

目标：继承已经训练好的 `PI-State` action head，只用 LoRA 微调 QwenVL 的指定部分，验证视觉塔、语言模型、二者联合适配的贡献。

每个实验 ID 单独占一台 server；每台 server 内同时跑三种模型尺寸：

| Model | GPUs | Processes | Port Offset |
| --- | --- | --- | --- |
| Qwen3.5-0.8B | `0` | `1` | `+0` |
| Qwen3.5-4B | `1,2` | `2` | `+10` |
| Qwen3.5-9B | `3,4,5,6,7` | `5` | `+20` |

实验矩阵：

| ID | 继承模块 | LoRA 微调模块 | 目的 | 默认 server |
| --- | --- | --- | --- | --- |
| E4 | PI-State action head | QwenVL ViT / visual tower | 看视觉塔适配是否有效 | `server_e4_lora_vit` |
| E5 | PI-State action head | QwenVL LLM / text tower | 看语言/高层语义适配是否有效 | `server_e5_lora_llm` |
| E6 | PI-State action head | QwenVL ViT + LLM | 联合适配 | `server_e6_lora_vit_llm` |

严格约束：

- `RELOAD_MODULES=action_model`：只从 PI-State checkpoint 继承 action head，找不到会报错。
- `FREEZE_MODULES=qwen_vl_interface,action_model`：LoRA 注入后脚本会自动移除顶层 `qwen_vl_interface` 冻结，保留 `action_model` 冻结，所以可训练参数只应是 LoRA。
- `LORA_FAIL_IF_NO_TARGET_MODULES=true`：如果 ViT/LLM 前缀没有匹配到 Linear 模块，直接失败，不允许静默退化。
- 默认使用 `INCLUDE_STATE=true`、`STATE_DIM=8`、`PI_STATE_DIM=8`，保持和 PI-State action head 一致。
- 当前 9B PI-State 源权重使用已确认存在的 `steps_25000_pytorch_model.pt`；做结果表时要标注 9B 继承源为 25k。

## 0. 公共启动块

每台 server 先执行一次：

```bash
cd /inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project
export PATH="/inspire/qb-ilm2/project/26summer-camp-10/26220216/miniconda3/bin:$PATH"
source /inspire/qb-ilm2/project/26summer-camp-10/26220216/miniconda3/etc/profile.d/conda.sh
conda activate starVLA_qwen35

export PROJECT_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project
export CONDA_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/26220216/miniconda3
export STAR_VLA_PYTHON="${CONDA_ROOT}/envs/starVLA_qwen35/bin/python"

export QWEN35_0P8B="${PROJECT_ROOT}/playground/Pretrained_models/Qwen3.5-0.8B"
export QWEN35_4B="${PROJECT_ROOT}/playground/Pretrained_models/Qwen3.5-4B"
export QWEN35_9B="${PROJECT_ROOT}/playground/Pretrained_models/Qwen3.5-9B"

export H200_CALVIN_DATA_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d
export H200_CALVIN_DATA_NAME=calvin_task_ABC_D
export H200_CALVIN_DATA_MIX=calvin_abc_d_h200

# 如果你要严格 CALVIN-1k，请把上面三行指向真实的 1k LeRobot 子集。
test -f "${H200_CALVIN_DATA_ROOT}/${H200_CALVIN_DATA_NAME}/meta/info.json"
test -f "${H200_CALVIN_DATA_ROOT}/${H200_CALVIN_DATA_NAME}/meta/modality.json"
test -d "${H200_CALVIN_DATA_ROOT}/${H200_CALVIN_DATA_NAME}/data"

export PIHEAD_SOURCE_ROOT="${PROJECT_ROOT}/logs/20260519_h200_server1_pi_state_30k_v1/server1_pi_state"
export PIHEAD_CKPT_0P8B="${PIHEAD_SOURCE_ROOT}/qwen35_0p8b/pi_state_qwen35_0p8b_30000step/checkpoints/pi_state_qwen35_0p8b_30000step/checkpoints/steps_30000_pytorch_model.pt"
export PIHEAD_CKPT_4B="${PIHEAD_SOURCE_ROOT}/qwen35_4b/pi_state_qwen35_4b_30000step/checkpoints/pi_state_qwen35_4b_30000step/checkpoints/steps_30000_pytorch_model.pt"
export PIHEAD_CKPT_9B="${PIHEAD_SOURCE_ROOT}/qwen35_9b/pi_state_qwen35_9b_30000step/checkpoints/pi_state_qwen35_9b_30000step/checkpoints/steps_25000_pytorch_model.pt"

test -f "${PIHEAD_CKPT_0P8B}"
test -f "${PIHEAD_CKPT_4B}"
test -f "${PIHEAD_CKPT_9B}"

# 如果之后补齐 9B steps_30000，可手动覆盖 PIHEAD_CKPT_9B 并在结果表中同步修改继承源 step。
# 不建议自动降级，否则 E4/E5/E6 与 PI-State baseline 不可公平对齐。
# find "${PROJECT_ROOT}/logs" -path "*pi_state*qwen35_9b*steps_*_pytorch_model.pt" -print | sort -V

export ACCELERATE_CONFIG=starVLA/config/deepseeds/deepspeed_zero2_route_validation.yaml
export CONFIG_YAML=examples/calvin/train_files/starvla_train_calvin_qwen35_pi_h200.yaml

export WANDB_MODE=disabled
export NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export ROUTE=p4_pi
export INCLUDE_STATE=true
export STATE_DIM=8
export PI_STATE_DIM=8
export ACTION_DIM=7
export ACTION_HORIZON=8
export NUM_ACTIONS_CHUNK=8
export OBS_IMAGE_SIZE="[224,224]"
export PI_NUM_INFERENCE_TIMESTEPS=4
export PI_REPEATED_DIFFUSION_STEPS=2
export PI_NUM_TARGET_VISION_TOKENS=32

export MAX_TRAIN_STEPS=30000
export SAVE_INTERVAL=1000
export EVAL_INTERVAL=1000000
export CHECKPOINT_KEEP_LATEST=1
export CHECKPOINT_KEEP_STEPS=10000,20000,30000
export LOGGING_FREQUENCY=10
export USE_TENSORBOARD=true
export DATALOADER_NUM_WORKERS=12
export MIN_LOG_FREE_GB=250

# LoRA 容量。E4/E5/E6 必须保持一致。
# 若要和旧 LoRA baseline 完全对齐，可改成 LORA_R=64 / LORA_ALPHA=128。
export LORA_ENABLED=true
export LORA_R=32
export LORA_ALPHA=64
export LORA_DROPOUT=0.05
export LORA_FAIL_IF_NO_TARGET_MODULES=true
export LORA_TARGET_EXCLUDE_PREFIXES=

# 冻结 action head，只训练 QwenVL LoRA。
export FREEZE_MODULES=qwen_vl_interface,action_model
export RELOAD_MODULES=action_model

# 激进但相对稳的 batch 默认值；9B LoRA 反传显存更重，先用 per-device 1。
export PER_DEVICE_BATCH_SIZE_0P8B=8
export PER_DEVICE_BATCH_SIZE_4B=2
export PER_DEVICE_BATCH_SIZE_9B=1

qwen_hidden_size() {
  "${STAR_VLA_PYTHON}" - "$1" <<'PY'
import json
import sys

model_dir = sys.argv[1]
with open(f"{model_dir}/config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)
text_cfg = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}
vision_cfg = cfg.get("vision_config") if isinstance(cfg.get("vision_config"), dict) else {}
hidden = cfg.get("hidden_size") or text_cfg.get("hidden_size") or vision_cfg.get("out_hidden_size")
if not isinstance(hidden, int) or hidden <= 0:
    raise SystemExit(f"cannot resolve hidden size from {model_dir}/config.json")
print(hidden)
PY
}

declare -a H200_PIHEAD_LORA_PIDS=()
declare -a H200_PIHEAD_LORA_NAMES=()
declare -a H200_PIHEAD_LORA_LOG_DIRS=()

check_pihead_lora_storage() {
  test -n "${BATCH_ROOT:-}" || { echo "BATCH_ROOT is empty" >&2; return 2; }
  mkdir -p "${BATCH_ROOT}"

  echo "===== storage preflight ====="
  df -h "${PROJECT_ROOT}" "${PROJECT_ROOT}/logs" "${BATCH_ROOT}" 2>/dev/null || true
  df -hi "${PROJECT_ROOT}" "${PROJECT_ROOT}/logs" "${BATCH_ROOT}" 2>/dev/null || true
  quota -s 2>/dev/null || true
  du -sh "${PROJECT_ROOT}/logs" "${BATCH_ROOT}" 2>/dev/null || true

  local available_kb
  available_kb="$(df -Pk "${BATCH_ROOT}" | awk 'NR==2 {print $4}')"
  local required_kb=$((MIN_LOG_FREE_GB * 1024 * 1024))
  if [[ -n "${available_kb}" && "${available_kb}" =~ ^[0-9]+$ && "${available_kb}" -lt "${required_kb}" ]]; then
    echo "free space below MIN_LOG_FREE_GB=${MIN_LOG_FREE_GB}GB: available_kb=${available_kb}" >&2
    return 2
  fi

  local probe="${BATCH_ROOT}/.quota_probe_${EXPERIMENT_ID:-unknown}_$$"
  if ! dd if=/dev/zero of="${probe}" bs=1M count=64 status=none; then
    rm -f "${probe}" 2>/dev/null || true
    echo "quota probe failed in ${BATCH_ROOT}; free quota before launching." >&2
    return 2
  fi
  rm -f "${probe}"
  sync
  echo "===== storage preflight OK ====="
}

launch_pihead_lora_size() {
  local model_tag=$1
  local base_vlm=$2
  local ckpt=$3
  local gpus=$4
  local num_processes=$5
  local main_process_port=$6
  local per_device_batch_size=$7
  local hidden_dim
  hidden_dim="$(qwen_hidden_size "${base_vlm}")"

  test -f "${ckpt}" || { echo "missing PI head checkpoint: ${ckpt}" >&2; return 2; }

  local run_id="${ROUTE_LABEL}_${model_tag}_${MAX_TRAIN_STEPS}step"
  local log_dir="${BATCH_ROOT}/${SERVER_DIR}/${model_tag}/${run_id}"
  local launcher_log="${log_dir}/terminal/launcher.log"
  local env_log="${log_dir}/terminal/env.log"
  mkdir -p "${log_dir}/terminal"

  echo "launch ${EXPERIMENT_ID} model=${model_tag} gpus=${gpus} batch=${per_device_batch_size} ckpt=${ckpt}"

  {
    echo "timestamp=$(date -Is)"
    echo "experiment_id=${EXPERIMENT_ID}"
    echo "server_dir=${SERVER_DIR}"
    echo "route_label=${ROUTE_LABEL}"
    echo "model_tag=${model_tag}"
    echo "base_vlm=${base_vlm}"
    echo "pretrained_checkpoint=${ckpt}"
    echo "reload_modules=${RELOAD_MODULES}"
    echo "freeze_modules=${FREEZE_MODULES}"
    echo "lora_target_modules=${LORA_TARGET_MODULES}"
    echo "lora_target_include_prefixes=${LORA_TARGET_INCLUDE_PREFIXES}"
    echo "lora_target_exclude_prefixes=${LORA_TARGET_EXCLUDE_PREFIXES:-<none>}"
    echo "cuda_visible_devices=${gpus}"
    echo "num_processes=${num_processes}"
    echo "main_process_port=${main_process_port}"
    echo "per_device_batch_size=${per_device_batch_size}"
    echo "hidden_dim=${hidden_dim}"
    echo "log_dir=${log_dir}"
  } > "${env_log}"

  (
    set +e
    echo "===== launcher start $(date -Is) ====="
    cat "${env_log}"
    echo "===== terminal stdout/stderr ====="
    ROUTE="${ROUTE}" \
    BASE_VLM="${base_vlm}" \
    CONFIG_YAML="${CONFIG_YAML}" \
    CUDA_VISIBLE_DEVICES="${gpus}" \
    NUM_PROCESSES="${num_processes}" \
    MAIN_PROCESS_PORT="${main_process_port}" \
    PRETRAINED_CHECKPOINT="${ckpt}" \
    RELOAD_MODULES="${RELOAD_MODULES}" \
    FREEZE_MODULES="${FREEZE_MODULES}" \
    LORA_ENABLED="${LORA_ENABLED}" \
    LORA_R="${LORA_R}" \
    LORA_ALPHA="${LORA_ALPHA}" \
    LORA_DROPOUT="${LORA_DROPOUT}" \
    LORA_TARGET_MODULES="${LORA_TARGET_MODULES}" \
    LORA_TARGET_INCLUDE_PREFIXES="${LORA_TARGET_INCLUDE_PREFIXES}" \
    LORA_TARGET_EXCLUDE_PREFIXES="${LORA_TARGET_EXCLUDE_PREFIXES:-}" \
    LORA_FAIL_IF_NO_TARGET_MODULES="${LORA_FAIL_IF_NO_TARGET_MODULES}" \
    INCLUDE_STATE="${INCLUDE_STATE}" \
    STATE_DIM="${STATE_DIM}" \
    PI_STATE_DIM="${PI_STATE_DIM}" \
    PER_DEVICE_BATCH_SIZE="${per_device_batch_size}" \
    DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS}" \
    MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS}" \
    SAVE_INTERVAL="${SAVE_INTERVAL}" \
    EVAL_INTERVAL="${EVAL_INTERVAL}" \
    CHECKPOINT_KEEP_LATEST="${CHECKPOINT_KEEP_LATEST}" \
    CHECKPOINT_KEEP_STEPS="${CHECKPOINT_KEEP_STEPS}" \
    LOGGING_FREQUENCY="${LOGGING_FREQUENCY}" \
    USE_TENSORBOARD="${USE_TENSORBOARD}" \
    RUN_TS="${BATCH_NAME}_${SERVER_DIR}_${model_tag}" \
    RUN_ID="${run_id}" \
    LOG_DIR="${log_dir}" \
    H200_CALVIN_DATA_ROOT="${H200_CALVIN_DATA_ROOT}" \
    H200_CALVIN_DATA_NAME="${H200_CALVIN_DATA_NAME}" \
    H200_CALVIN_DATA_MIX="${H200_CALVIN_DATA_MIX}" \
    bash examples/calvin/train_files/run_route_validation_train.sh
    status=$?
    echo "===== launcher end $(date -Is) status=${status} ====="
    exit "${status}"
  ) > "${launcher_log}" 2>&1 &

  local pid=$!
  echo "${pid}" > "${log_dir}/pid.txt"
  H200_PIHEAD_LORA_PIDS+=("${pid}")
  H200_PIHEAD_LORA_NAMES+=("${model_tag}")
  H200_PIHEAD_LORA_LOG_DIRS+=("${log_dir}")
}

launch_three_sizes_for_pihead_lora() {
  H200_PIHEAD_LORA_PIDS=()
  H200_PIHEAD_LORA_NAMES=()
  H200_PIHEAD_LORA_LOG_DIRS=()

  check_pihead_lora_storage || return 2

  local missing=0
  test -f "${PIHEAD_CKPT_0P8B}" || { echo "missing PI head checkpoint for qwen35_0p8b: ${PIHEAD_CKPT_0P8B}" >&2; missing=1; }
  test -f "${PIHEAD_CKPT_4B}" || { echo "missing PI head checkpoint for qwen35_4b: ${PIHEAD_CKPT_4B}" >&2; missing=1; }
  test -f "${PIHEAD_CKPT_9B}" || { echo "missing PI head checkpoint for qwen35_9b: ${PIHEAD_CKPT_9B}" >&2; missing=1; }
  if [[ "${missing}" -ne 0 ]]; then
    echo "Abort before launch: all three PI-State source checkpoints are required." >&2
    return 2
  fi

  launch_pihead_lora_size qwen35_0p8b "${QWEN35_0P8B}" "${PIHEAD_CKPT_0P8B}" 0 1 "${PORT_0P8B}" "${PER_DEVICE_BATCH_SIZE_0P8B}"
  launch_pihead_lora_size qwen35_4b "${QWEN35_4B}" "${PIHEAD_CKPT_4B}" 1,2 2 "${PORT_4B}" "${PER_DEVICE_BATCH_SIZE_4B}"
  launch_pihead_lora_size qwen35_9b "${QWEN35_9B}" "${PIHEAD_CKPT_9B}" 3,4,5,6,7 5 "${PORT_9B}" "${PER_DEVICE_BATCH_SIZE_9B}"

  local status=0
  local idx
  for idx in "${!H200_PIHEAD_LORA_PIDS[@]}"; do
    local pid="${H200_PIHEAD_LORA_PIDS[$idx]}"
    local name="${H200_PIHEAD_LORA_NAMES[$idx]}"
    local log_dir="${H200_PIHEAD_LORA_LOG_DIRS[$idx]}"
    if ! wait "${pid}"; then
      echo "FAILED model=${name} pid=${pid} log_dir=${log_dir}" >&2
      tail -n 160 "${log_dir}/terminal/launcher.log" >&2 || true
      status=1
    else
      echo "DONE model=${name} pid=${pid} log_dir=${log_dir}"
    fi
  done
  return "${status}"
}
```

## 1. E4：PIHead-LoRA-ViT

在 E4 server 执行公共启动块后运行：

```bash
export EXPERIMENT_ID=E4
export ROUTE_LABEL=pihead_lora_vit
export SERVER_DIR=server_e4_lora_vit
export BATCH_NAME=20260520_h200_pihead_lora_vit_30k_v1
export BATCH_ROOT="${PROJECT_ROOT}/logs/${BATCH_NAME}"
mkdir -p "${BATCH_ROOT}"

export PORT_0P8B=34100
export PORT_4B=34110
export PORT_9B=34120

export LORA_TARGET_INCLUDE_PREFIXES=model.visual
export LORA_TARGET_EXCLUDE_PREFIXES=
export LORA_TARGET_MODULES=qkv,proj,linear_fc1,linear_fc2

launch_three_sizes_for_pihead_lora
```

## 2. E5：PIHead-LoRA-LLM

在 E5 server 执行公共启动块后运行：

```bash
export EXPERIMENT_ID=E5
export ROUTE_LABEL=pihead_lora_llm
export SERVER_DIR=server_e5_lora_llm
export BATCH_NAME=20260520_h200_pihead_lora_llm_30k_v1
export BATCH_ROOT="${PROJECT_ROOT}/logs/${BATCH_NAME}"
mkdir -p "${BATCH_ROOT}"

export PORT_0P8B=34200
export PORT_4B=34210
export PORT_9B=34220

export LORA_TARGET_INCLUDE_PREFIXES=model
export LORA_TARGET_EXCLUDE_PREFIXES=model.visual
export LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj

launch_three_sizes_for_pihead_lora
```

## 3. E6：PIHead-LoRA-ViT-LLM

在 E6 server 执行公共启动块后运行：

```bash
export EXPERIMENT_ID=E6
export ROUTE_LABEL=pihead_lora_vit_llm
export SERVER_DIR=server_e6_lora_vit_llm
export BATCH_NAME=20260520_h200_pihead_lora_vit_llm_30k_v1
export BATCH_ROOT="${PROJECT_ROOT}/logs/${BATCH_NAME}"
mkdir -p "${BATCH_ROOT}"

export PORT_0P8B=34300
export PORT_4B=34310
export PORT_9B=34320

export LORA_TARGET_INCLUDE_PREFIXES=model
export LORA_TARGET_EXCLUDE_PREFIXES=
export LORA_TARGET_MODULES=qkv,proj,linear_fc1,linear_fc2,q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj

launch_three_sizes_for_pihead_lora
```

## 4. 启动后检查

每个 job 的 `terminal/launcher.log` 必须出现类似：

```text
✅ Resolved LoRA target modules: ...
✅ LoRA enabled on qwen_vl_interface.model
📦 loading checkpoint: ...steps_30000_pytorch_model.pt  # 9B 当前为 steps_25000_pytorch_model.pt
✅ parameters loaded to module 'action_model'
FREEZE_MODULES=qwen_vl_interface,action_model
RELOAD_MODULES=action_model
```

如果出现：

```text
LoRA target prefix filtering matched no Linear modules
```

不要改成跳过，直接停下来，根据日志里列出的 `First Linear module names` 修正 `LORA_TARGET_INCLUDE_PREFIXES` 或 `LORA_TARGET_MODULES`。

## 5. 快速查看

```bash
tail -f logs/20260520_h200_pihead_lora_vit_30k_v1/server_e4_lora_vit/qwen35_0p8b/pihead_lora_vit_qwen35_0p8b_30000step/terminal/launcher.log
tail -f logs/20260520_h200_pihead_lora_llm_30k_v1/server_e5_lora_llm/qwen35_0p8b/pihead_lora_llm_qwen35_0p8b_30000step/terminal/launcher.log
tail -f logs/20260520_h200_pihead_lora_vit_llm_30k_v1/server_e6_lora_vit_llm/qwen35_0p8b/pihead_lora_vit_llm_qwen35_0p8b_30000step/terminal/launcher.log
```

TensorBoard：

```bash
tensorboard --logdir logs/20260520_h200_pihead_lora_vit_30k_v1 --port 6006
tensorboard --logdir logs/20260520_h200_pihead_lora_llm_30k_v1 --port 6007
tensorboard --logdir logs/20260520_h200_pihead_lora_vit_llm_30k_v1 --port 6008
```

## 6. 磁盘配额失败处理

如果日志出现：

```text
OSError: [Errno 122] Disk quota exceeded
```

这说明训练已经能启动，模型、LoRA target 和数据链路不是当前失败点；失败发生在向 `LOG_DIR/checkpoints/<run_id>/config.full.yaml`、TensorBoard、checkpoint 或统计文件写盘时。不要通过关闭配置保存来绕过，应先释放配额或换输出目录。

先停掉当前失败批次中残留的进程：

```bash
ps -ef | grep "20260520_h200_pihead_lora_llm_30k_v1" | grep -v grep
pkill -f "20260520_h200_pihead_lora_llm_30k_v1" || true
```

检查和定位大文件：

```bash
df -h "${PROJECT_ROOT}" "${PROJECT_ROOT}/logs"
df -hi "${PROJECT_ROOT}" "${PROJECT_ROOT}/logs"
quota -s 2>/dev/null || true
du -sh "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/logs/20260520_h200_pihead_lora_llm_30k_v1" 2>/dev/null || true
find "${PROJECT_ROOT}/logs" -type f -name "steps_*_pytorch_model.pt" -printf '%s\t%p\n' 2>/dev/null | sort -nr | head -40 | numfmt --field=1 --to=iec
```

确认失败批次没有可保留结果后再删。不要删除 `PIHEAD_SOURCE_ROOT` 里的 PI-State 源 checkpoint：

```bash
rm -rf "${PROJECT_ROOT}/logs/20260520_h200_pihead_lora_llm_30k_v1"
sync
```

如果 `qb-ilm2` 个人目录配额仍然紧张，可以把新实验输出到 `hdd` 个人目录，代码和数据仍留在原位置：

```bash
export BATCH_NAME=20260520_h200_pihead_lora_llm_30k_v1
export BATCH_ROOT="/inspire/hdd/project/26summer-camp-10/26220216/starVLA_logs/${BATCH_NAME}"
mkdir -p "${BATCH_ROOT}"
```

然后重新执行对应 E4/E5/E6 启动块。公共函数会先做 `64MiB` 写盘探针和 `MIN_LOG_FREE_GB` 检查，失败就不会启动三路训练。
