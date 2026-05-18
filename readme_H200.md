# H200 四路线训练与评测命令

本文档汇总 H200 服务器上四条 Qwen3.5-0.8B 路线的训练、checkpoint 检查、policy server 和 CALVIN eval 命令。默认目标是高标准复现：任何路线失败就保留日志并报错，不用其它路线或错误配置伪装成功。

## 1. 路线定义

| 路线 | `ROUTE` | 模型结构 | 默认 GPU 用法 | 当前定位 |
| --- | --- | --- | --- | --- |
| P0 | `p0_oft` | `Qwen3.5-0.8B + QwenOFT + MLP action head` | 单卡 | 保底主线，已本地跑通 |
| P1 | `p1_adapter` | `Qwen3.5-0.8B + QwenAdapter + VLA_Adapter action head` | 单卡 | AdapterVLA 思路验证 |
| P2 | `p2_lora_oft` | `Qwen3.5-0.8B + LoRA + QwenOFT` | 单卡 | LoRA 工程增强，需确认保存/加载/eval |
| P3 | `p3_lora_adapter` | `Qwen3.5-0.8B + LoRA + QwenAdapter` | 单卡 | P1/P2 稳定后再跑 |

默认训练脚本是“多路线并行，每条路线绑定一张 GPU”。
`GPU_LIST` 的顺序必须和 `ROUTE_LIST` 一一对应，例如 `ROUTE_LIST="p0_oft p1_adapter"` 配 `GPU_LIST="2 5"` 表示 P0 跑 GPU2、P1 跑 GPU5。

每个新终端都先执行第 2 节公共环境，再执行后面的训练、server 或 eval 命令。

## 2. H200 公共环境

先复制执行这一段。它按你当前服务器路径设置项目根目录和 conda 根目录：

```bash
export PROJECT_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project
export CONDA_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/26220216/miniconda3
export STARVLA_ENV=starVLA
export CALVIN_ENV=calvin

test -d "${PROJECT_ROOT}"
test -f "${CONDA_ROOT}/etc/profile.d/conda.sh"

export PATH="${CONDA_ROOT}/bin:${PATH}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"

export WANDB_MODE=disabled
export NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export STAR_VLA_PYTHON="$(python -c 'import sys; print(sys.executable)')"
export CALVIN_CONFIG_PATH="${PROJECT_ROOT}/calvin/calvin_models/conf"
export H200_CALVIN_DATA_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d
export H200_CALVIN_DATA_NAME=calvin_task_ABC_D
export H200_CALVIN_DATA_MIX=calvin_abc_d_h200
export H200_CALVIN_DATASET_PATH="${H200_CALVIN_DATA_ROOT}/${H200_CALVIN_DATA_NAME}"
```

基础检查：

```bash
nvidia-smi
df -h

if command -v git >/dev/null 2>&1; then
  git branch --show-current
  git log -1 --oneline --decorate
  git status --short
else
  echo "git not found in PATH, skip git checks"
fi

"${STAR_VLA_PYTHON}" -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"
"${STAR_VLA_PYTHON}" -c "from transformers import Qwen3_5ForConditionalGeneration; print('Qwen3.5 import OK')"
test -f playground/Pretrained_models/Qwen3.5-0.8B/config.json
test -d "${PROJECT_ROOT}/calvin/calvin_models/conf"
test -d "${H200_CALVIN_DATASET_PATH}"
test -f "${H200_CALVIN_DATASET_PATH}/meta/info.json"
test -f "${H200_CALVIN_DATASET_PATH}/meta/modality.json"
test -d "${H200_CALVIN_DATASET_PATH}/data"
```

训练要求 `H200_CALVIN_DATASET_PATH` 是 LeRobot 格式目录，必须包含 `meta/info.json`、`meta/modality.json` 和 `data/`。如果这里只包含原始 CALVIN 的 `training/validation/`，不要继续训练，先转换成 LeRobot 格式。

脚本语法检查：

```bash
bash -n examples/calvin/train_files/run_route_validation_train.sh
bash -n examples/calvin/train_files/run_route_h200_matrix.sh
bash -n examples/calvin/eval_files/run_policy_server_debug.sh
bash -n examples/calvin/eval_files/eval_calvin_debug.sh
```

LoRA 路线检查：

```bash
"${STAR_VLA_PYTHON}" - <<'PY'
import importlib.util
print("peft_installed", importlib.util.find_spec("peft") is not None)
PY
```

如果 `peft_installed=False` 或项目还没有 LoRA 注入、保存、加载和 policy server 兼容逻辑，P2/P3 只允许做工程适配与 smoke，不允许写成 LoRA 长训成功。

## 3. 一键并行跑四条路线

### 3.1 四路线 100 step smoke

指定 GPU0/1/2/3 分别跑 P0/P1/P2/P3：

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"

H200_RUN_TS=$(date +"%Y%m%d_%H%M%S")

ROUTE_LIST="p0_oft p1_adapter p2_lora_oft p3_lora_adapter" \
GPU_LIST="0 1 2 3" \
H200_CALVIN_DATA_ROOT="${H200_CALVIN_DATA_ROOT}" \
H200_CALVIN_DATA_NAME="${H200_CALVIN_DATA_NAME}" \
H200_CALVIN_DATA_MIX="${H200_CALVIN_DATA_MIX}" \
MAX_TRAIN_STEPS=100 \
SAVE_INTERVAL=100 \
EVAL_INTERVAL=1000000 \
DATALOADER_NUM_WORKERS=0 \
H200_RUN_TS="${H200_RUN_TS}" \
LOG_ROOT="logs/h200_route_train/log_${H200_RUN_TS}_qwen35_0p8b_4route_smoke100" \
bash examples/calvin/train_files/run_route_h200_matrix.sh
```

成功后检查：

```bash
find logs/h200_route_train -path "*/terminal/train.log" -print
find logs/h200_route_train -path "*/metrics/loss_check.json" -print
find logs/h200_route_train -path "*/metrics/reload_check_steps_100.json" -print
find logs/h200_route_train -path "*/checkpoints/*/checkpoints/steps_100_pytorch_model.pt" -print
```

### 3.2 四路线 10k 对照

指定 GPU0/1/2/3：

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"

H200_RUN_TS=$(date +"%Y%m%d_%H%M%S")

ROUTE_LIST="p0_oft p1_adapter p2_lora_oft p3_lora_adapter" \
GPU_LIST="0 1 2 3" \
H200_CALVIN_DATA_ROOT="${H200_CALVIN_DATA_ROOT}" \
H200_CALVIN_DATA_NAME="${H200_CALVIN_DATA_NAME}" \
H200_CALVIN_DATA_MIX="${H200_CALVIN_DATA_MIX}" \
MAX_TRAIN_STEPS=10000 \
SAVE_INTERVAL=5000 \
EVAL_INTERVAL=1000000 \
DATALOADER_NUM_WORKERS=2 \
H200_RUN_TS="${H200_RUN_TS}" \
LOG_ROOT="logs/h200_route_train/log_${H200_RUN_TS}_qwen35_0p8b_4route_10k" \
bash examples/calvin/train_files/run_route_h200_matrix.sh
```

### 3.3 四路线 30k 正式训练

指定 GPU0/1/2/3：

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"

H200_RUN_TS=$(date +"%Y%m%d_%H%M%S")

ROUTE_LIST="p0_oft p1_adapter p2_lora_oft p3_lora_adapter" \
GPU_LIST="0 1 2 3" \
H200_CALVIN_DATA_ROOT="${H200_CALVIN_DATA_ROOT}" \
H200_CALVIN_DATA_NAME="${H200_CALVIN_DATA_NAME}" \
H200_CALVIN_DATA_MIX="${H200_CALVIN_DATA_MIX}" \
MAX_TRAIN_STEPS=30000 \
SAVE_INTERVAL=5000 \
EVAL_INTERVAL=1000000 \
DATALOADER_NUM_WORKERS=4 \
H200_RUN_TS="${H200_RUN_TS}" \
LOG_ROOT="logs/h200_route_train/log_${H200_RUN_TS}_qwen35_0p8b_4route_30k" \
bash examples/calvin/train_files/run_route_h200_matrix.sh
```

### 3.4 后台运行

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"

H200_RUN_TS=$(date +"%Y%m%d_%H%M%S")
LOG_ROOT="logs/h200_route_train/log_${H200_RUN_TS}_qwen35_0p8b_4route_30k"
mkdir -p "${LOG_ROOT}/terminal"

nohup bash -lc "
  ROUTE_LIST='p0_oft p1_adapter p2_lora_oft p3_lora_adapter' \
  GPU_LIST='0 1 2 3' \
  H200_CALVIN_DATA_ROOT='${H200_CALVIN_DATA_ROOT}' \
  H200_CALVIN_DATA_NAME='${H200_CALVIN_DATA_NAME}' \
  H200_CALVIN_DATA_MIX='${H200_CALVIN_DATA_MIX}' \
  MAX_TRAIN_STEPS=30000 \
  SAVE_INTERVAL=5000 \
  EVAL_INTERVAL=1000000 \
  DATALOADER_NUM_WORKERS=4 \
  H200_RUN_TS='${H200_RUN_TS}' \
  LOG_ROOT='${LOG_ROOT}' \
  bash examples/calvin/train_files/run_route_h200_matrix.sh
" > "${LOG_ROOT}/terminal/nohup.log" 2>&1 &

echo "LOG_ROOT=${LOG_ROOT}"
echo "PID=$!"
```

查看进度：

```bash
tail -f "${LOG_ROOT}/terminal/nohup.log"
tail -f "${LOG_ROOT}/terminal/h200_matrix.log"
nvidia-smi
```

## 4. 指定某一张 GPU 跑单条路线

单路线脚本入口是 `examples/calvin/train_files/run_route_validation_train.sh`。
用 `CUDA_VISIBLE_DEVICES=<gpu_id>` 指定某张卡。

### 4.1 P0：QwenOFT

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"

RUN_TS=$(date +"%Y%m%d_%H%M%S")
ROUTE=p0_oft
RUN_ID="h200_${ROUTE}_30000step"
LOG_DIR="logs/h200_route_train/log_${RUN_TS}_${RUN_ID}"

CUDA_VISIBLE_DEVICES=0 \
CALVIN_DATA_ROOT="${H200_CALVIN_DATA_ROOT}" \
CALVIN_DATA_NAME="${H200_CALVIN_DATA_NAME}" \
CALVIN_DATA_MIX="${H200_CALVIN_DATA_MIX}" \
CONFIG_YAML=examples/calvin/train_files/starvla_train_calvin_qwen35_oft_h200.yaml \
ROUTE="${ROUTE}" \
MAX_TRAIN_STEPS=30000 \
SAVE_INTERVAL=5000 \
EVAL_INTERVAL=1000000 \
RUN_TS="${RUN_TS}" \
RUN_ID="${RUN_ID}" \
LOG_DIR="${LOG_DIR}" \
DATALOADER_NUM_WORKERS=4 \
bash examples/calvin/train_files/run_route_validation_train.sh
```

### 4.2 P1：QwenAdapter

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"

RUN_TS=$(date +"%Y%m%d_%H%M%S")
ROUTE=p1_adapter
RUN_ID="h200_${ROUTE}_30000step"
LOG_DIR="logs/h200_route_train/log_${RUN_TS}_${RUN_ID}"

CUDA_VISIBLE_DEVICES=1 \
CALVIN_DATA_ROOT="${H200_CALVIN_DATA_ROOT}" \
CALVIN_DATA_NAME="${H200_CALVIN_DATA_NAME}" \
CALVIN_DATA_MIX="${H200_CALVIN_DATA_MIX}" \
CONFIG_YAML=examples/calvin/train_files/starvla_train_calvin_qwen35_oft_h200.yaml \
ROUTE="${ROUTE}" \
MAX_TRAIN_STEPS=30000 \
SAVE_INTERVAL=5000 \
EVAL_INTERVAL=1000000 \
RUN_TS="${RUN_TS}" \
RUN_ID="${RUN_ID}" \
LOG_DIR="${LOG_DIR}" \
DATALOADER_NUM_WORKERS=4 \
bash examples/calvin/train_files/run_route_validation_train.sh
```

### 4.3 P2：LoRA + QwenOFT

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"

RUN_TS=$(date +"%Y%m%d_%H%M%S")
ROUTE=p2_lora_oft
RUN_ID="h200_${ROUTE}_100step"
LOG_DIR="logs/h200_route_train/log_${RUN_TS}_${RUN_ID}"

CUDA_VISIBLE_DEVICES=2 \
CALVIN_DATA_ROOT="${H200_CALVIN_DATA_ROOT}" \
CALVIN_DATA_NAME="${H200_CALVIN_DATA_NAME}" \
CALVIN_DATA_MIX="${H200_CALVIN_DATA_MIX}" \
CONFIG_YAML=examples/calvin/train_files/starvla_train_calvin_qwen35_oft_h200.yaml \
ROUTE="${ROUTE}" \
MAX_TRAIN_STEPS=100 \
SAVE_INTERVAL=100 \
EVAL_INTERVAL=1000000 \
RUN_TS="${RUN_TS}" \
RUN_ID="${RUN_ID}" \
LOG_DIR="${LOG_DIR}" \
DATALOADER_NUM_WORKERS=0 \
bash examples/calvin/train_files/run_route_validation_train.sh
```

P2 先跑 100 step 严格验证 LoRA 注入、checkpoint 保存、reload 和 policy server。通过后再把 `MAX_TRAIN_STEPS` 改成 `10000` 或 `30000`。

### 4.4 P3：LoRA + QwenAdapter

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"

RUN_TS=$(date +"%Y%m%d_%H%M%S")
ROUTE=p3_lora_adapter
RUN_ID="h200_${ROUTE}_100step"
LOG_DIR="logs/h200_route_train/log_${RUN_TS}_${RUN_ID}"

CUDA_VISIBLE_DEVICES=3 \
CALVIN_DATA_ROOT="${H200_CALVIN_DATA_ROOT}" \
CALVIN_DATA_NAME="${H200_CALVIN_DATA_NAME}" \
CALVIN_DATA_MIX="${H200_CALVIN_DATA_MIX}" \
CONFIG_YAML=examples/calvin/train_files/starvla_train_calvin_qwen35_oft_h200.yaml \
ROUTE="${ROUTE}" \
MAX_TRAIN_STEPS=100 \
SAVE_INTERVAL=100 \
EVAL_INTERVAL=1000000 \
RUN_TS="${RUN_TS}" \
RUN_ID="${RUN_ID}" \
LOG_DIR="${LOG_DIR}" \
DATALOADER_NUM_WORKERS=0 \
bash examples/calvin/train_files/run_route_validation_train.sh
```

P3 必须在 P1 和 P2 都通过 smoke 后再长训。

## 5. 指定任意几张 GPU 跑任意路线组合

只跑 P0/P1，并指定 GPU4/GPU5：

```bash
H200_RUN_TS=$(date +"%Y%m%d_%H%M%S")

ROUTE_LIST="p0_oft p1_adapter" \
GPU_LIST="4 5" \
H200_CALVIN_DATA_ROOT="${H200_CALVIN_DATA_ROOT}" \
H200_CALVIN_DATA_NAME="${H200_CALVIN_DATA_NAME}" \
H200_CALVIN_DATA_MIX="${H200_CALVIN_DATA_MIX}" \
MAX_TRAIN_STEPS=30000 \
SAVE_INTERVAL=5000 \
DATALOADER_NUM_WORKERS=4 \
H200_RUN_TS="${H200_RUN_TS}" \
LOG_ROOT="logs/h200_route_train/log_${H200_RUN_TS}_p0_p1_gpu4_gpu5_30k" \
bash examples/calvin/train_files/run_route_h200_matrix.sh
```

只跑 P1/P3，并指定 GPU2/GPU7：

```bash
H200_RUN_TS=$(date +"%Y%m%d_%H%M%S")

ROUTE_LIST="p1_adapter p3_lora_adapter" \
GPU_LIST="2 7" \
H200_CALVIN_DATA_ROOT="${H200_CALVIN_DATA_ROOT}" \
H200_CALVIN_DATA_NAME="${H200_CALVIN_DATA_NAME}" \
H200_CALVIN_DATA_MIX="${H200_CALVIN_DATA_MIX}" \
MAX_TRAIN_STEPS=10000 \
SAVE_INTERVAL=5000 \
DATALOADER_NUM_WORKERS=2 \
H200_RUN_TS="${H200_RUN_TS}" \
LOG_ROOT="logs/h200_route_train/log_${H200_RUN_TS}_p1_p3_gpu2_gpu7_10k" \
bash examples/calvin/train_files/run_route_h200_matrix.sh
```

只跑 P0，并指定 GPU6：

```bash
H200_RUN_TS=$(date +"%Y%m%d_%H%M%S")

ROUTE_LIST="p0_oft" \
GPU_LIST="6" \
H200_CALVIN_DATA_ROOT="${H200_CALVIN_DATA_ROOT}" \
H200_CALVIN_DATA_NAME="${H200_CALVIN_DATA_NAME}" \
H200_CALVIN_DATA_MIX="${H200_CALVIN_DATA_MIX}" \
MAX_TRAIN_STEPS=50000 \
SAVE_INTERVAL=5000 \
DATALOADER_NUM_WORKERS=4 \
H200_RUN_TS="${H200_RUN_TS}" \
LOG_ROOT="logs/h200_route_train/log_${H200_RUN_TS}_p0_gpu6_50k" \
bash examples/calvin/train_files/run_route_h200_matrix.sh
```

## 6. checkpoint 路径与严格检查

训练完成后 checkpoint 路径格式：

```text
<LOG_DIR>/checkpoints/<RUN_ID>/checkpoints/steps_<MAX_TRAIN_STEPS>_pytorch_model.pt
```

严格 reload 检查：

```bash
CKPT_PATH=<checkpoint_pt_path>
OUT_JSON=<log_dir>/metrics/reload_check_manual.json

"${STAR_VLA_PYTHON}" examples/calvin/eval_files/check_checkpoint_reload.py \
  --ckpt-path "${CKPT_PATH}" \
  --expected-action-chunk-size 8 \
  --expected-unnorm-key franka \
  --output-json "${OUT_JSON}"
```

必须通过：

- `action_chunk_size == 8`
- `available_unnorm_keys` 包含 `franka`
- `passed == true`

## 7. 启动 policy server

给每个待评测 checkpoint 分配一个空闲端口和一张 GPU。示例使用 GPU4、端口 5694：

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"
export STAR_VLA_PYTHON="$(python -c 'import sys; print(sys.executable)')"

export CKPT_PATH=<checkpoint_pt_path>
export PORT=5694
export RUN_ID=h200_eval_server_p0_steps30000
export LOG_DIR=<route_log_dir>/server_steps30000
export CUDA_VISIBLE_DEVICES=4

bash examples/calvin/eval_files/run_policy_server_debug.sh
```

通过标准：

- `terminal/policy_server.log` 中没有 traceback。
- server 监听指定 `PORT`。
- metadata 中 `available_unnorm_keys` 包含 `franka`。
- metadata 中 `action_chunk_size=8`。

## 8. CALVIN debug eval

另开一个终端，使用 `calvin` 环境跑 eval。示例跑 5 条 debug sequence：

```bash
cd "${PROJECT_ROOT}"
conda activate "${CALVIN_ENV}"
export CALVIN_PYTHON="$(python -c 'import sys; print(sys.executable)')"

export CKPT_PATH=<checkpoint_pt_path>
export PORT=5694
export NUM_SEQUENCES=5
export UNNORM_KEY=franka
export RUN_ID=h200_eval_debug5_p0_steps30000
export LOG_DIR=<route_log_dir>/eval_debug5_steps30000
export DATASET_PATH=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d/calvin_task_ABC_D
export CALVIN_CONFIG_PATH="${CALVIN_CONFIG_PATH}"

bash examples/calvin/eval_files/eval_calvin_debug.sh
```

必须产出：

```bash
test -f "${LOG_DIR}/terminal/eval.log"
test -f "${LOG_DIR}/mp4/results.json"
find "${LOG_DIR}/mp4" -name "*.mp4" -print
```

读取结果：

```bash
cat "${LOG_DIR}/mp4/results.json"
rg -n "Average successful sequence length|Success rates|Subtask:" "${LOG_DIR}/terminal/eval.log"
```

## 9. 完整 ABC->D eval

完整原始 CALVIN ABC->D validation 数据准备好后，只替换 `DATASET_PATH` 和 `NUM_SEQUENCES`：

```bash
cd "${PROJECT_ROOT}"
conda activate "${CALVIN_ENV}"
export CALVIN_PYTHON="$(python -c 'import sys; print(sys.executable)')"

export CKPT_PATH=<checkpoint_pt_path>
export PORT=5694
export NUM_SEQUENCES=1000
export UNNORM_KEY=franka
export RUN_ID=h200_eval_abcd_full_<route>_<step>
export LOG_DIR=<route_log_dir>/eval_abcd_full_<step>
export DATASET_PATH=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d/calvin_task_ABC_D
export CALVIN_CONFIG_PATH="${CALVIN_CONFIG_PATH}"

bash examples/calvin/eval_files/eval_calvin_debug.sh
```

如果时间不够，先跑：

```bash
export NUM_SEQUENCES=100
export LOG_DIR=<route_log_dir>/eval_abcd_100_<step>
bash examples/calvin/eval_files/eval_calvin_debug.sh
```

报告中必须明确写实际 `NUM_SEQUENCES`，不能把 100 条写成完整 1000 条。

## 10. 推荐执行顺序

1. P0/P1 先跑 100 step smoke。
2. P0 通过 reload + policy server + debug eval 后，立即扩到 30k。
3. P1 通过后扩到 30k。
4. P2 先做 100 step LoRA smoke，确认 LoRA 参数、保存、加载、eval 全链路。
5. P3 只在 P1/P2 都稳定后运行。
6. 每个候选 checkpoint 必须跑 `reload_check`、policy server、debug eval，并保存 `results.json` 和 mp4。
7. 最终选择不只看 loss，优先看平均任务链长度、Task1~5 成功率、mp4 failure pattern 和日志完整性。

## 11. 常用排查命令

查看训练 loss：

```bash
rg -n "loss|nan|inf|Traceback|CUDA out of memory" <route_log_dir>/terminal/train.log
cat <route_log_dir>/metrics/loss_check.json
```

查看矩阵调度状态：

```bash
cat <LOG_ROOT>/terminal/h200_matrix.log
find <LOG_ROOT> -path "*/terminal/h200_runner.log" -print
```

查看 checkpoint：

```bash
find <LOG_ROOT> -path "*/checkpoints/*/checkpoints/*.pt" -print
```

查看 eval 产物：

```bash
find <route_log_dir> -path "*/mp4/results.json" -print
find <route_log_dir> -path "*/mp4/*.mp4" -print
```

停止残留 server：

```bash
ps -ef | rg "server_policy.py|eval_calvin.py|train_starvla.py"
```

确认无误后再按 PID 精确停止，不要使用粗暴全杀命令。
