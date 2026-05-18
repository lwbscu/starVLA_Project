# H200 四路线训练与验证命令

服务器项目路径：

```bash
/inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project
```

服务器 CALVIN 数据路径：

```bash
/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d/calvin_task_ABC_D
```

## 0. 每个新终端先执行

```bash
export PROJECT_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project
export CONDA_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/26220216/miniconda3
export STARVLA_ENV=starVLA
export CALVIN_ENV=calvin

export PATH="${CONDA_ROOT}/bin:${PATH}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"

export WANDB_MODE=disabled
export NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export STAR_VLA_PYTHON="$(python -c 'import sys; print(sys.executable)')"

export H200_CALVIN_DATA_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d
export H200_CALVIN_DATA_NAME=calvin_task_ABC_D
export H200_CALVIN_DATA_MIX=calvin_abc_d_h200
export H200_CALVIN_DATASET_PATH="${H200_CALVIN_DATA_ROOT}/${H200_CALVIN_DATA_NAME}"
export CALVIN_CONFIG_PATH="${PROJECT_ROOT}/calvin/calvin_models/conf"
```

## 1. GPU 指定方式

单路线指定某一张 GPU：

```bash
CUDA_VISIBLE_DEVICES=0 ROUTE=p0_oft bash examples/calvin/train_files/run_route_validation_train.sh
```

多路线指定多张 GPU：

```bash
ROUTE_LIST="p0_oft p1_adapter" GPU_LIST="2 5" bash examples/calvin/train_files/run_route_h200_matrix.sh
```

规则：`ROUTE_LIST` 和 `GPU_LIST` 按顺序一一对应。上例表示 `p0_oft` 跑 GPU2，`p1_adapter` 跑 GPU5。

可调训练超参数：

```bash
MAX_TRAIN_STEPS=30000
SAVE_INTERVAL=5000
EVAL_INTERVAL=1000000
DATALOADER_NUM_WORKERS=4
```

## 2. 四路线并行训练

### 2.1 四路线 100 step smoke

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"
export STAR_VLA_PYTHON="$(python -c 'import sys; print(sys.executable)')"

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

### 2.2 四路线 10k 对照

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"
export STAR_VLA_PYTHON="$(python -c 'import sys; print(sys.executable)')"

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

### 2.3 四路线 30k 正式训练

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"
export STAR_VLA_PYTHON="$(python -c 'import sys; print(sys.executable)')"

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

### 2.4 后台四路线 30k

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"
export STAR_VLA_PYTHON="$(python -c 'import sys; print(sys.executable)')"

H200_RUN_TS=$(date +"%Y%m%d_%H%M%S")
LOG_ROOT="logs/h200_route_train/log_${H200_RUN_TS}_qwen35_0p8b_4route_30k"
mkdir -p "${LOG_ROOT}/terminal"

nohup bash -lc "
  cd '${PROJECT_ROOT}' && \
  source '${CONDA_ROOT}/etc/profile.d/conda.sh' && \
  conda activate '${STARVLA_ENV}' && \
  export STAR_VLA_PYTHON=\"\$(python -c 'import sys; print(sys.executable)')\" && \
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

查看：

```bash
tail -f "${LOG_ROOT}/terminal/nohup.log"
tail -f "${LOG_ROOT}/terminal/h200_matrix.log"
nvidia-smi
```

## 3. 单路线训练命令

### 3.1 P0：Qwen3.5-0.8B + OFT

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"
export STAR_VLA_PYTHON="$(python -c 'import sys; print(sys.executable)')"

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

### 3.2 P1：Qwen3.5-0.8B + QwenAdapter

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"
export STAR_VLA_PYTHON="$(python -c 'import sys; print(sys.executable)')"

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

### 3.3 P2：Qwen3.5-0.8B + LoRA + OFT

先 smoke：

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"
export STAR_VLA_PYTHON="$(python -c 'import sys; print(sys.executable)')"

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

通过后长训：

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"
export STAR_VLA_PYTHON="$(python -c 'import sys; print(sys.executable)')"

RUN_TS=$(date +"%Y%m%d_%H%M%S")
ROUTE=p2_lora_oft
RUN_ID="h200_${ROUTE}_30000step"
LOG_DIR="logs/h200_route_train/log_${RUN_TS}_${RUN_ID}"

CUDA_VISIBLE_DEVICES=2 \
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

### 3.4 P3：Qwen3.5-0.8B + LoRA + QwenAdapter

先 smoke：

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"
export STAR_VLA_PYTHON="$(python -c 'import sys; print(sys.executable)')"

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

通过后长训：

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"
export STAR_VLA_PYTHON="$(python -c 'import sys; print(sys.executable)')"

RUN_TS=$(date +"%Y%m%d_%H%M%S")
ROUTE=p3_lora_adapter
RUN_ID="h200_${ROUTE}_30000step"
LOG_DIR="logs/h200_route_train/log_${RUN_TS}_${RUN_ID}"

CUDA_VISIBLE_DEVICES=3 \
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

## 4. 指定任意路线和任意 GPU 组合

只跑 P0/P1，用 GPU4/GPU5：

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"
export STAR_VLA_PYTHON="$(python -c 'import sys; print(sys.executable)')"

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

只跑 P0，用 GPU6：

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"
export STAR_VLA_PYTHON="$(python -c 'import sys; print(sys.executable)')"

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

## 5. checkpoint 路径

训练完成后 checkpoint 路径：

```text
<LOG_DIR>/checkpoints/<RUN_ID>/checkpoints/steps_<MAX_TRAIN_STEPS>_pytorch_model.pt
```

矩阵训练路径示例：

```text
logs/h200_route_train/log_<timestamp>_qwen35_0p8b_4route_30k/h200_p0_oft_30000step/checkpoints/h200_p0_oft_30000step/checkpoints/steps_30000_pytorch_model.pt
```

## 6. checkpoint reload 验证

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"
export STAR_VLA_PYTHON="$(python -c 'import sys; print(sys.executable)')"

CKPT_PATH=<checkpoint_pt_path>
OUT_JSON=<route_log_dir>/metrics/reload_check_manual.json

"${STAR_VLA_PYTHON}" examples/calvin/eval_files/check_checkpoint_reload.py \
  --ckpt-path "${CKPT_PATH}" \
  --expected-action-chunk-size 8 \
  --expected-unnorm-key franka \
  --output-json "${OUT_JSON}"
```

## 7. 启动 policy server

用 GPU4 启动 server，端口 5694：

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"
export STAR_VLA_PYTHON="$(python -c 'import sys; print(sys.executable)')"

export CKPT_PATH=<checkpoint_pt_path>
export PORT=5694
export RUN_ID=h200_eval_server_<route>_<step>
export LOG_DIR=<route_log_dir>/server_<step>
export CUDA_VISIBLE_DEVICES=4

bash examples/calvin/eval_files/run_policy_server_debug.sh
```

## 8. CALVIN debug eval

另开新终端，先执行第 0 节公共环境，然后执行：

```bash
cd "${PROJECT_ROOT}"
conda activate "${CALVIN_ENV}"
export CALVIN_PYTHON="$(python -c 'import sys; print(sys.executable)')"

export CKPT_PATH=<checkpoint_pt_path>
export HOST=127.0.0.1
export PORT=5694
export NUM_SEQUENCES=5
export UNNORM_KEY=franka
export RUN_ID=h200_eval_debug5_<route>_<step>
export LOG_DIR=<route_log_dir>/eval_debug5_<step>
export DATASET_PATH=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d/calvin_task_ABC_D
export CALVIN_CONFIG_PATH="${CALVIN_CONFIG_PATH}"

bash examples/calvin/eval_files/eval_calvin_debug.sh
```

检查产物：

```bash
test -f "${LOG_DIR}/terminal/eval.log"
test -f "${LOG_DIR}/mp4/results.json"
find "${LOG_DIR}/mp4" -name "*.mp4" -print
cat "${LOG_DIR}/mp4/results.json"
```

## 9. 完整 ABC->D eval

保持 policy server 不关，另开 eval 终端：

```bash
cd "${PROJECT_ROOT}"
conda activate "${CALVIN_ENV}"
export CALVIN_PYTHON="$(python -c 'import sys; print(sys.executable)')"

export CKPT_PATH=<checkpoint_pt_path>
export HOST=127.0.0.1
export PORT=5694
export NUM_SEQUENCES=1000
export UNNORM_KEY=franka
export RUN_ID=h200_eval_abcd_full_<route>_<step>
export LOG_DIR=<route_log_dir>/eval_abcd_full_<step>
export DATASET_PATH=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d/calvin_task_ABC_D
export CALVIN_CONFIG_PATH="${CALVIN_CONFIG_PATH}"

bash examples/calvin/eval_files/eval_calvin_debug.sh
```

时间不够时先跑 100 条：

```bash
export NUM_SEQUENCES=100
export LOG_DIR=<route_log_dir>/eval_abcd_100_<step>
bash examples/calvin/eval_files/eval_calvin_debug.sh
```

## 10. 常用查看命令

```bash
find logs/h200_route_train -path "*/terminal/train.log" -print
find logs/h200_route_train -path "*/metrics/loss_check.json" -print
find logs/h200_route_train -path "*/metrics/reload_check_steps_*.json" -print
find logs/h200_route_train -path "*/checkpoints/*/checkpoints/*.pt" -print
find logs/h200_route_train -path "*/mp4/results.json" -print
find logs/h200_route_train -path "*/mp4/*.mp4" -print
```

```bash
tail -f <route_log_dir>/terminal/train.log
tail -f <route_log_dir>/terminal/policy_server.log
tail -f <route_log_dir>/terminal/eval.log
```
