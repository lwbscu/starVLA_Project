# H200 四服务器强训练命令

目标：四台服务器并行训练四条 Qwen3.5 路线；每台服务器默认占满 8 张 H200；每台服务器只跑一条路线。

## 0. 每台服务器先执行

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

export TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3,4,5,6,7}
export NUM_PROCESSES=${NUM_PROCESSES:-8}
export DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-16}
export MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-200000}
export SAVE_INTERVAL=${SAVE_INTERVAL:-10000}
export EVAL_INTERVAL=${EVAL_INTERVAL:-1000000}
export OBS_IMAGE_SIZE=${OBS_IMAGE_SIZE:-"[224,224]"}
export ACTION_HORIZON=${ACTION_HORIZON:-8}
export NUM_ACTIONS_CHUNK=${NUM_ACTIONS_CHUNK:-8}
export ACTION_QUERY_NUM=${ACTION_QUERY_NUM:-128}
export ADAPTER_HIDDEN_DIM=${ADAPTER_HIDDEN_DIM:-2048}
export LORA_R=${LORA_R:-64}
export LORA_ALPHA=${LORA_ALPHA:-128}
export LORA_DROPOUT=${LORA_DROPOUT:-0.05}
```

## 1. 自动选择最强 Qwen3.5 权重

按 9B、4B、2B、0.8B 顺序选择本机已有权重。没有任何 Qwen3.5 权重就直接报错。

```bash
cd "${PROJECT_ROOT}"

unset BASE_VLM
for candidate in \
  ./playground/Pretrained_models/Qwen3.5-9B \
  ./playground/Pretrained_models/Qwen3.5-4B \
  ./playground/Pretrained_models/Qwen3.5-2B \
  ./playground/Pretrained_models/Qwen3.5-0.8B
do
  if test -f "${candidate}/config.json"; then
    export BASE_VLM="${candidate}"
    break
  fi
done

test -n "${BASE_VLM:-}" || { echo "No Qwen3.5 weight found under playground/Pretrained_models"; exit 2; }
echo "BASE_VLM=${BASE_VLM}"
```

强制指定某个权重：

```bash
export BASE_VLM=./playground/Pretrained_models/Qwen3.5-9B
test -f "${BASE_VLM}/config.json"
```

## 2. 必须通过环境和数据检查

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"
export STAR_VLA_PYTHON="$(python -c 'import sys; print(sys.executable)')"

"${STAR_VLA_PYTHON}" -c "import torch, transformers; print('torch=', torch.__version__); print('transformers=', transformers.__version__); print('cuda=', torch.cuda.is_available())"
"${STAR_VLA_PYTHON}" -c "from transformers import Qwen3_5ForConditionalGeneration; print('Qwen3.5 import OK')"

test -f "${BASE_VLM}/config.json"
test -d "${H200_CALVIN_DATASET_PATH}"
test -f "${H200_CALVIN_DATASET_PATH}/meta/info.json"
test -f "${H200_CALVIN_DATASET_PATH}/meta/modality.json" || cp examples/calvin/train_files/modality.json "${H200_CALVIN_DATASET_PATH}/meta/modality.json"
test -d "${H200_CALVIN_DATASET_PATH}/data"

nvidia-smi
du -sh "${H200_CALVIN_DATASET_PATH}"
ls "${H200_CALVIN_DATASET_PATH}/meta"
```

## 3. 四台服务器分工

| 服务器 | 路线 | 默认 GPU | 目标 |
| --- | --- | --- | --- |
| Server-1 | `p0_oft` | 8xH200 | 强保底：Qwen3.5 + OFT |
| Server-2 | `p1_adapter` | 8xH200 | 泛化主攻：Qwen3.5 + Adapter |
| Server-3 | `p2_lora_oft` | 8xH200 | LoRA 对照：Qwen3.5 + LoRA + OFT |
| Server-4 | `p3_lora_adapter` | 8xH200 | 最强尝试：Qwen3.5 + LoRA + Adapter |

指定 GPU 写法：

```bash
export TRAIN_GPUS=0,1,2,3,4,5,6,7
export NUM_PROCESSES=8
```

只用 4 张 GPU：

```bash
export TRAIN_GPUS=0,1,2,3
export NUM_PROCESSES=4
```

训练日志开头必须显示：

```text
CALVIN_DATA_SOURCE=auto_h200
CALVIN_DATA_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d
CALVIN_DATA_MIX=calvin_abc_d_h200
CALVIN_DATA_NAME=calvin_task_ABC_D
BASE_VLM=<Qwen3.5 path>
OBS_IMAGE_SIZE=[224,224]
NUM_PROCESSES=8
```

## 4. Server-1 跑 P0：Qwen3.5 + OFT

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"
export STAR_VLA_PYTHON="$(python -c 'import sys; print(sys.executable)')"

RUN_TS=$(date +"%Y%m%d_%H%M%S")
ROUTE=p0_oft
RUN_ID="strong_${ROUTE}_${MAX_TRAIN_STEPS}step"
LOG_DIR="logs/h200_strong_train/log_${RUN_TS}_${RUN_ID}"
mkdir -p "${LOG_DIR}/terminal"

nohup bash -lc "
  cd '${PROJECT_ROOT}' && \
  source '${CONDA_ROOT}/etc/profile.d/conda.sh' && \
  conda activate '${STARVLA_ENV}' && \
  export STAR_VLA_PYTHON=\"\$(python -c 'import sys; print(sys.executable)')\" && \
  CUDA_VISIBLE_DEVICES='${TRAIN_GPUS}' \
  NUM_PROCESSES='${NUM_PROCESSES}' \
  BASE_VLM='${BASE_VLM}' \
  OBS_IMAGE_SIZE='${OBS_IMAGE_SIZE}' \
  ACTION_HORIZON='${ACTION_HORIZON}' \
  CALVIN_DATA_ROOT='${H200_CALVIN_DATA_ROOT}' \
  CALVIN_DATA_NAME='${H200_CALVIN_DATA_NAME}' \
  CALVIN_DATA_MIX='${H200_CALVIN_DATA_MIX}' \
  CONFIG_YAML=examples/calvin/train_files/starvla_train_calvin_qwen35_oft_h200.yaml \
  ROUTE='${ROUTE}' \
  MAX_TRAIN_STEPS='${MAX_TRAIN_STEPS}' \
  SAVE_INTERVAL='${SAVE_INTERVAL}' \
  EVAL_INTERVAL='${EVAL_INTERVAL}' \
  RUN_TS='${RUN_TS}' \
  RUN_ID='${RUN_ID}' \
  LOG_DIR='${LOG_DIR}' \
  DATALOADER_NUM_WORKERS='${DATALOADER_NUM_WORKERS}' \
  bash examples/calvin/train_files/run_route_validation_train.sh
" > "${LOG_DIR}/terminal/nohup.log" 2>&1 &

echo "ROUTE=${ROUTE}"
echo "LOG_DIR=${LOG_DIR}"
echo "PID=$!"
```

## 5. Server-2 跑 P1：Qwen3.5 + QwenAdapter

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"
export STAR_VLA_PYTHON="$(python -c 'import sys; print(sys.executable)')"

RUN_TS=$(date +"%Y%m%d_%H%M%S")
ROUTE=p1_adapter
RUN_ID="strong_${ROUTE}_${MAX_TRAIN_STEPS}step"
LOG_DIR="logs/h200_strong_train/log_${RUN_TS}_${RUN_ID}"
mkdir -p "${LOG_DIR}/terminal"

nohup bash -lc "
  cd '${PROJECT_ROOT}' && \
  source '${CONDA_ROOT}/etc/profile.d/conda.sh' && \
  conda activate '${STARVLA_ENV}' && \
  export STAR_VLA_PYTHON=\"\$(python -c 'import sys; print(sys.executable)')\" && \
  CUDA_VISIBLE_DEVICES='${TRAIN_GPUS}' \
  NUM_PROCESSES='${NUM_PROCESSES}' \
  BASE_VLM='${BASE_VLM}' \
  OBS_IMAGE_SIZE='${OBS_IMAGE_SIZE}' \
  ACTION_QUERY_NUM='${ACTION_QUERY_NUM}' \
  NUM_ACTIONS_CHUNK='${NUM_ACTIONS_CHUNK}' \
  ADAPTER_HIDDEN_DIM='${ADAPTER_HIDDEN_DIM}' \
  CALVIN_DATA_ROOT='${H200_CALVIN_DATA_ROOT}' \
  CALVIN_DATA_NAME='${H200_CALVIN_DATA_NAME}' \
  CALVIN_DATA_MIX='${H200_CALVIN_DATA_MIX}' \
  CONFIG_YAML=examples/calvin/train_files/starvla_train_calvin_qwen35_oft_h200.yaml \
  ROUTE='${ROUTE}' \
  MAX_TRAIN_STEPS='${MAX_TRAIN_STEPS}' \
  SAVE_INTERVAL='${SAVE_INTERVAL}' \
  EVAL_INTERVAL='${EVAL_INTERVAL}' \
  RUN_TS='${RUN_TS}' \
  RUN_ID='${RUN_ID}' \
  LOG_DIR='${LOG_DIR}' \
  DATALOADER_NUM_WORKERS='${DATALOADER_NUM_WORKERS}' \
  bash examples/calvin/train_files/run_route_validation_train.sh
" > "${LOG_DIR}/terminal/nohup.log" 2>&1 &

echo "ROUTE=${ROUTE}"
echo "LOG_DIR=${LOG_DIR}"
echo "PID=$!"
```

## 6. Server-3 跑 P2：Qwen3.5 + LoRA + OFT

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"
export STAR_VLA_PYTHON="$(python -c 'import sys; print(sys.executable)')"

RUN_TS=$(date +"%Y%m%d_%H%M%S")
ROUTE=p2_lora_oft
RUN_ID="strong_${ROUTE}_${MAX_TRAIN_STEPS}step"
LOG_DIR="logs/h200_strong_train/log_${RUN_TS}_${RUN_ID}"
mkdir -p "${LOG_DIR}/terminal"

nohup bash -lc "
  cd '${PROJECT_ROOT}' && \
  source '${CONDA_ROOT}/etc/profile.d/conda.sh' && \
  conda activate '${STARVLA_ENV}' && \
  export STAR_VLA_PYTHON=\"\$(python -c 'import sys; print(sys.executable)')\" && \
  CUDA_VISIBLE_DEVICES='${TRAIN_GPUS}' \
  NUM_PROCESSES='${NUM_PROCESSES}' \
  BASE_VLM='${BASE_VLM}' \
  OBS_IMAGE_SIZE='${OBS_IMAGE_SIZE}' \
  ACTION_HORIZON='${ACTION_HORIZON}' \
  LORA_R='${LORA_R}' \
  LORA_ALPHA='${LORA_ALPHA}' \
  LORA_DROPOUT='${LORA_DROPOUT}' \
  CALVIN_DATA_ROOT='${H200_CALVIN_DATA_ROOT}' \
  CALVIN_DATA_NAME='${H200_CALVIN_DATA_NAME}' \
  CALVIN_DATA_MIX='${H200_CALVIN_DATA_MIX}' \
  CONFIG_YAML=examples/calvin/train_files/starvla_train_calvin_qwen35_oft_h200.yaml \
  ROUTE='${ROUTE}' \
  MAX_TRAIN_STEPS='${MAX_TRAIN_STEPS}' \
  SAVE_INTERVAL='${SAVE_INTERVAL}' \
  EVAL_INTERVAL='${EVAL_INTERVAL}' \
  RUN_TS='${RUN_TS}' \
  RUN_ID='${RUN_ID}' \
  LOG_DIR='${LOG_DIR}' \
  DATALOADER_NUM_WORKERS='${DATALOADER_NUM_WORKERS}' \
  bash examples/calvin/train_files/run_route_validation_train.sh
" > "${LOG_DIR}/terminal/nohup.log" 2>&1 &

echo "ROUTE=${ROUTE}"
echo "LOG_DIR=${LOG_DIR}"
echo "PID=$!"
```

## 7. Server-4 跑 P3：Qwen3.5 + LoRA + QwenAdapter

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"
export STAR_VLA_PYTHON="$(python -c 'import sys; print(sys.executable)')"

RUN_TS=$(date +"%Y%m%d_%H%M%S")
ROUTE=p3_lora_adapter
RUN_ID="strong_${ROUTE}_${MAX_TRAIN_STEPS}step"
LOG_DIR="logs/h200_strong_train/log_${RUN_TS}_${RUN_ID}"
mkdir -p "${LOG_DIR}/terminal"

nohup bash -lc "
  cd '${PROJECT_ROOT}' && \
  source '${CONDA_ROOT}/etc/profile.d/conda.sh' && \
  conda activate '${STARVLA_ENV}' && \
  export STAR_VLA_PYTHON=\"\$(python -c 'import sys; print(sys.executable)')\" && \
  CUDA_VISIBLE_DEVICES='${TRAIN_GPUS}' \
  NUM_PROCESSES='${NUM_PROCESSES}' \
  BASE_VLM='${BASE_VLM}' \
  OBS_IMAGE_SIZE='${OBS_IMAGE_SIZE}' \
  ACTION_QUERY_NUM='${ACTION_QUERY_NUM}' \
  NUM_ACTIONS_CHUNK='${NUM_ACTIONS_CHUNK}' \
  ADAPTER_HIDDEN_DIM='${ADAPTER_HIDDEN_DIM}' \
  LORA_R='${LORA_R}' \
  LORA_ALPHA='${LORA_ALPHA}' \
  LORA_DROPOUT='${LORA_DROPOUT}' \
  CALVIN_DATA_ROOT='${H200_CALVIN_DATA_ROOT}' \
  CALVIN_DATA_NAME='${H200_CALVIN_DATA_NAME}' \
  CALVIN_DATA_MIX='${H200_CALVIN_DATA_MIX}' \
  CONFIG_YAML=examples/calvin/train_files/starvla_train_calvin_qwen35_oft_h200.yaml \
  ROUTE='${ROUTE}' \
  MAX_TRAIN_STEPS='${MAX_TRAIN_STEPS}' \
  SAVE_INTERVAL='${SAVE_INTERVAL}' \
  EVAL_INTERVAL='${EVAL_INTERVAL}' \
  RUN_TS='${RUN_TS}' \
  RUN_ID='${RUN_ID}' \
  LOG_DIR='${LOG_DIR}' \
  DATALOADER_NUM_WORKERS='${DATALOADER_NUM_WORKERS}' \
  bash examples/calvin/train_files/run_route_validation_train.sh
" > "${LOG_DIR}/terminal/nohup.log" 2>&1 &

echo "ROUTE=${ROUTE}"
echo "LOG_DIR=${LOG_DIR}"
echo "PID=$!"
```

## 8. 训练监控

```bash
nvidia-smi
tail -f "${LOG_DIR}/terminal/nohup.log"
tail -f "${LOG_DIR}/terminal/train.log"
```

查所有强训练日志：

```bash
find logs/h200_strong_train -path "*/terminal/train.log" -print
find logs/h200_strong_train -path "*/metrics/loss_check.json" -print
find logs/h200_strong_train -path "*/checkpoints/*/checkpoints/*.pt" -print
```

## 9. checkpoint 路径

```text
<LOG_DIR>/checkpoints/<RUN_ID>/checkpoints/steps_<MAX_TRAIN_STEPS>_pytorch_model.pt
```

手动设置：

```bash
export CKPT_PATH=<checkpoint_pt_path>
export ROUTE_LOG_DIR=<route_log_dir>
```

## 10. checkpoint reload 验证

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"
export STAR_VLA_PYTHON="$(python -c 'import sys; print(sys.executable)')"

mkdir -p "${ROUTE_LOG_DIR}/metrics"

"${STAR_VLA_PYTHON}" examples/calvin/eval_files/check_checkpoint_reload.py \
  --ckpt-path "${CKPT_PATH}" \
  --expected-action-chunk-size 8 \
  --expected-unnorm-key franka \
  --output-json "${ROUTE_LOG_DIR}/metrics/reload_check_manual.json"
```

## 11. 启动 policy server

用任意空闲 GPU 启动 server。示例用 GPU0、端口 5694：

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"
export STAR_VLA_PYTHON="$(python -c 'import sys; print(sys.executable)')"

export CKPT_PATH=<checkpoint_pt_path>
export ROUTE_LOG_DIR=<route_log_dir>
export PORT=5694
export RUN_ID=h200_eval_server
export LOG_DIR="${ROUTE_LOG_DIR}/server"
export CUDA_VISIBLE_DEVICES=0

bash examples/calvin/eval_files/run_policy_server_debug.sh
```

## 12. CALVIN debug eval

另开新终端：

```bash
export PROJECT_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project
export CONDA_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/26220216/miniconda3
export PATH="${CONDA_ROOT}/bin:${PATH}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

cd "${PROJECT_ROOT}"
conda activate calvin
export CALVIN_PYTHON="$(python -c 'import sys; print(sys.executable)')"

export CKPT_PATH=<checkpoint_pt_path>
export ROUTE_LOG_DIR=<route_log_dir>
export HOST=127.0.0.1
export PORT=5694
export NUM_SEQUENCES=5
export UNNORM_KEY=franka
export RUN_ID=h200_eval_debug5
export LOG_DIR="${ROUTE_LOG_DIR}/eval_debug5"
export DATASET_PATH=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d/calvin_task_ABC_D
export CALVIN_CONFIG_PATH="${PROJECT_ROOT}/calvin/calvin_models/conf"

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
conda activate calvin
export CALVIN_PYTHON="$(python -c 'import sys; print(sys.executable)')"

export CKPT_PATH=<checkpoint_pt_path>
export ROUTE_LOG_DIR=<route_log_dir>
export HOST=127.0.0.1
export PORT=5694
export NUM_SEQUENCES=1000
export UNNORM_KEY=franka
export RUN_ID=h200_eval_abcd_full
export LOG_DIR="${ROUTE_LOG_DIR}/eval_abcd_full"
export DATASET_PATH=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d/calvin_task_ABC_D
export CALVIN_CONFIG_PATH="${PROJECT_ROOT}/calvin/calvin_models/conf"

bash examples/calvin/eval_files/eval_calvin_debug.sh
```

时间不够时先跑 100 条：

```bash
export NUM_SEQUENCES=100
export LOG_DIR="${ROUTE_LOG_DIR}/eval_abcd_100"
bash examples/calvin/eval_files/eval_calvin_debug.sh
```

## 14. 最终选权重

每条路线至少比较这些 checkpoint：

```text
steps_50000_pytorch_model.pt
steps_100000_pytorch_model.pt
steps_150000_pytorch_model.pt
steps_200000_pytorch_model.pt
```

优先级：

```text
1. 完整 ABC->D 平均链长最高
2. Task 1~5 成功率更均衡
3. debug mp4 中动作尺度正常、夹爪方向正确
4. 若分数接近，优先选 P3，再 P1，再 P2，再 P0
```
