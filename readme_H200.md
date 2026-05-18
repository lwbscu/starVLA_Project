# H200 四条路线训练与验证命令

本文只写服务器直接执行命令。四条路线均为单路线命令；每条路线可以通过 `CUDA_VISIBLE_DEVICES` 指定一张或多张 GPU，通过 `NUM_PROCESSES` 指定进程数。`NUM_PROCESSES` 必须等于本次可见 GPU 数量。

## 0. 每个新终端先执行

```bash
export PROJECT_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project
export CONDA_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/26220216/miniconda3
export STARVLA_ENV=starVLA_qwen35
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

## 1. 必须先通过环境检查

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"
export STAR_VLA_PYTHON="$(python -c 'import sys; print(sys.executable)')"

"${STAR_VLA_PYTHON}" -c "import torch, transformers; print('torch=', torch.__version__); print('transformers=', transformers.__version__); print('cuda=', torch.cuda.is_available())"
"${STAR_VLA_PYTHON}" -c "from transformers import Qwen3_5ForConditionalGeneration; print('Qwen3.5 import OK')"
```

如果第二条报错 `cannot import name 'Qwen3_5ForConditionalGeneration'`，在当前 `STARVLA_ENV` 中升级 `transformers`：

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"

pip install -U "transformers==5.3.0"
pip install -e .

python -c "from transformers import Qwen3_5ForConditionalGeneration; print('Qwen3.5 import OK')"
```

如果当前 pip 源没有 `transformers==5.3.0`，改用源码版本：

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"

pip install -U "transformers[serving] @ git+https://github.com/huggingface/transformers.git@main"
pip install -e .

python -c "from transformers import Qwen3_5ForConditionalGeneration; print('Qwen3.5 import OK')"
```

## 2. 必须先通过数据检查

```bash
cd "${PROJECT_ROOT}"

test -d "${H200_CALVIN_DATASET_PATH}"
test -f "${H200_CALVIN_DATASET_PATH}/meta/info.json"
test -f "${H200_CALVIN_DATASET_PATH}/meta/modality.json" || cp examples/calvin/train_files/modality.json "${H200_CALVIN_DATASET_PATH}/meta/modality.json"
test -d "${H200_CALVIN_DATASET_PATH}/data"

du -sh "${H200_CALVIN_DATASET_PATH}"
ls "${H200_CALVIN_DATASET_PATH}/meta"
```

## 3. GPU 参数写法

执行第 0 节公共环境后，下面短命令会自动使用服务器数据集：

```text
/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d/calvin_task_ABC_D
```

训练日志开头应显示：

```text
CALVIN_DATA_SOURCE=auto_h200
CALVIN_DATA_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d
CALVIN_DATA_MIX=calvin_abc_d_h200
CALVIN_DATA_NAME=calvin_task_ABC_D
```

单路线使用 1 张 GPU：

```bash
CUDA_VISIBLE_DEVICES=0 NUM_PROCESSES=1 ROUTE=p0_oft bash examples/calvin/train_files/run_route_validation_train.sh
```

单路线使用 2 张 GPU：

```bash
CUDA_VISIBLE_DEVICES=0,1 NUM_PROCESSES=2 ROUTE=p0_oft bash examples/calvin/train_files/run_route_validation_train.sh
```

单路线使用 4 张 GPU：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PROCESSES=4 ROUTE=p1_adapter bash examples/calvin/train_files/run_route_validation_train.sh
```

常用训练超参数：

```bash
MAX_TRAIN_STEPS=30000
SAVE_INTERVAL=5000
EVAL_INTERVAL=1000000
DATALOADER_NUM_WORKERS=4
```

## 4. P0 路线：Qwen3.5-0.8B + OFT

### 4.1 P0 100 step smoke

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"
export STAR_VLA_PYTHON="$(python -c 'import sys; print(sys.executable)')"

RUN_TS=$(date +"%Y%m%d_%H%M%S")
ROUTE=p0_oft
RUN_ID="h200_${ROUTE}_100step"
LOG_DIR="logs/h200_route_train/log_${RUN_TS}_${RUN_ID}"

CUDA_VISIBLE_DEVICES=0 \
NUM_PROCESSES=1 \
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

### 4.2 P0 正式训练，可改 GPU 数

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"
export STAR_VLA_PYTHON="$(python -c 'import sys; print(sys.executable)')"

RUN_TS=$(date +"%Y%m%d_%H%M%S")
ROUTE=p0_oft
RUN_ID="h200_${ROUTE}_30000step"
LOG_DIR="logs/h200_route_train/log_${RUN_TS}_${RUN_ID}"

CUDA_VISIBLE_DEVICES=0,1 \
NUM_PROCESSES=2 \
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

## 5. P1 路线：Qwen3.5-0.8B + QwenAdapter

### 5.1 P1 100 step smoke

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"
export STAR_VLA_PYTHON="$(python -c 'import sys; print(sys.executable)')"

RUN_TS=$(date +"%Y%m%d_%H%M%S")
ROUTE=p1_adapter
RUN_ID="h200_${ROUTE}_100step"
LOG_DIR="logs/h200_route_train/log_${RUN_TS}_${RUN_ID}"

CUDA_VISIBLE_DEVICES=2 \
NUM_PROCESSES=1 \
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

### 5.2 P1 正式训练，可改 GPU 数

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"
export STAR_VLA_PYTHON="$(python -c 'import sys; print(sys.executable)')"

RUN_TS=$(date +"%Y%m%d_%H%M%S")
ROUTE=p1_adapter
RUN_ID="h200_${ROUTE}_30000step"
LOG_DIR="logs/h200_route_train/log_${RUN_TS}_${RUN_ID}"

CUDA_VISIBLE_DEVICES=2,3 \
NUM_PROCESSES=2 \
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

## 6. P2 路线：Qwen3.5-0.8B + LoRA + OFT

### 6.1 P2 100 step smoke

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"
export STAR_VLA_PYTHON="$(python -c 'import sys; print(sys.executable)')"

RUN_TS=$(date +"%Y%m%d_%H%M%S")
ROUTE=p2_lora_oft
RUN_ID="h200_${ROUTE}_100step"
LOG_DIR="logs/h200_route_train/log_${RUN_TS}_${RUN_ID}"

CUDA_VISIBLE_DEVICES=4 \
NUM_PROCESSES=1 \
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

### 6.2 P2 正式训练，可改 GPU 数

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"
export STAR_VLA_PYTHON="$(python -c 'import sys; print(sys.executable)')"

RUN_TS=$(date +"%Y%m%d_%H%M%S")
ROUTE=p2_lora_oft
RUN_ID="h200_${ROUTE}_30000step"
LOG_DIR="logs/h200_route_train/log_${RUN_TS}_${RUN_ID}"

CUDA_VISIBLE_DEVICES=4,5 \
NUM_PROCESSES=2 \
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

## 7. P3 路线：Qwen3.5-0.8B + LoRA + QwenAdapter

### 7.1 P3 100 step smoke

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"
export STAR_VLA_PYTHON="$(python -c 'import sys; print(sys.executable)')"

RUN_TS=$(date +"%Y%m%d_%H%M%S")
ROUTE=p3_lora_adapter
RUN_ID="h200_${ROUTE}_100step"
LOG_DIR="logs/h200_route_train/log_${RUN_TS}_${RUN_ID}"

CUDA_VISIBLE_DEVICES=6 \
NUM_PROCESSES=1 \
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

### 7.2 P3 正式训练，可改 GPU 数

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"
export STAR_VLA_PYTHON="$(python -c 'import sys; print(sys.executable)')"

RUN_TS=$(date +"%Y%m%d_%H%M%S")
ROUTE=p3_lora_adapter
RUN_ID="h200_${ROUTE}_30000step"
LOG_DIR="logs/h200_route_train/log_${RUN_TS}_${RUN_ID}"

CUDA_VISIBLE_DEVICES=6,7 \
NUM_PROCESSES=2 \
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

## 8. 后台运行单条路线

示例：P0 使用 GPU0、GPU1 后台训练 30k。

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"
export STAR_VLA_PYTHON="$(python -c 'import sys; print(sys.executable)')"

RUN_TS=$(date +"%Y%m%d_%H%M%S")
ROUTE=p0_oft
RUN_ID="h200_${ROUTE}_30000step"
LOG_DIR="logs/h200_route_train/log_${RUN_TS}_${RUN_ID}"
mkdir -p "${LOG_DIR}/terminal"

nohup bash -lc "
  cd '${PROJECT_ROOT}' && \
  source '${CONDA_ROOT}/etc/profile.d/conda.sh' && \
  conda activate '${STARVLA_ENV}' && \
  export STAR_VLA_PYTHON=\"\$(python -c 'import sys; print(sys.executable)')\" && \
  CUDA_VISIBLE_DEVICES='0,1' \
  NUM_PROCESSES=2 \
  CALVIN_DATA_ROOT='${H200_CALVIN_DATA_ROOT}' \
  CALVIN_DATA_NAME='${H200_CALVIN_DATA_NAME}' \
  CALVIN_DATA_MIX='${H200_CALVIN_DATA_MIX}' \
  CONFIG_YAML=examples/calvin/train_files/starvla_train_calvin_qwen35_oft_h200.yaml \
  ROUTE='${ROUTE}' \
  MAX_TRAIN_STEPS=30000 \
  SAVE_INTERVAL=5000 \
  EVAL_INTERVAL=1000000 \
  RUN_TS='${RUN_TS}' \
  RUN_ID='${RUN_ID}' \
  LOG_DIR='${LOG_DIR}' \
  DATALOADER_NUM_WORKERS=4 \
  bash examples/calvin/train_files/run_route_validation_train.sh
" > "${LOG_DIR}/terminal/nohup.log" 2>&1 &

echo "LOG_DIR=${LOG_DIR}"
echo "PID=$!"
```

查看：

```bash
tail -f "${LOG_DIR}/terminal/nohup.log"
tail -f "${LOG_DIR}/terminal/train.log"
nvidia-smi
```

## 9. checkpoint 路径

```text
<LOG_DIR>/checkpoints/<RUN_ID>/checkpoints/steps_<MAX_TRAIN_STEPS>_pytorch_model.pt
```

示例：

```text
logs/h200_route_train/log_<timestamp>_h200_p0_oft_30000step/checkpoints/h200_p0_oft_30000step/checkpoints/steps_30000_pytorch_model.pt
```

## 10. checkpoint reload 验证

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"
export STAR_VLA_PYTHON="$(python -c 'import sys; print(sys.executable)')"

CKPT_PATH=<checkpoint_pt_path>
OUT_JSON=<route_log_dir>/metrics/reload_check_manual.json
mkdir -p "$(dirname "${OUT_JSON}")"

"${STAR_VLA_PYTHON}" examples/calvin/eval_files/check_checkpoint_reload.py \
  --ckpt-path "${CKPT_PATH}" \
  --expected-action-chunk-size 8 \
  --expected-unnorm-key franka \
  --output-json "${OUT_JSON}"
```

## 11. 启动 policy server

用任意空闲 GPU 启动 server。示例用 GPU4、端口 5694：

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

## 12. CALVIN debug eval

另开新终端，先执行第 0 节公共环境，再执行：

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

检查：

```bash
test -f "${LOG_DIR}/terminal/eval.log"
test -f "${LOG_DIR}/mp4/results.json"
find "${LOG_DIR}/mp4" -name "*.mp4" -print
cat "${LOG_DIR}/mp4/results.json"
```

## 13. 完整 ABC->D eval

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

## 14. 常用查看命令

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
