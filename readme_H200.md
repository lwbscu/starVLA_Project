# H200 两天快速探索命令

目标：四台服务器并行探索四条 Qwen3.5 路线。每台服务器默认占满 8 张 H200，但训练按阶段推进：`1k smoke -> 10k 快评 -> 30k 决策 -> 赢家 60k/100k 加训`。不要一开始直接 200k 长训。

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

## 1. 选择 Qwen3.5 权重

优先使用本机已有最大 Qwen3.5 权重：

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

强制指定：

```bash
export BASE_VLM=./playground/Pretrained_models/Qwen3.5-4B
test -f "${BASE_VLM}/config.json"
```

## 2. 必须通过检查

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
```

## 3. 四台服务器分工

| 服务器 | 路线 | 目的 |
| --- | --- | --- |
| Server-1 | `p0_oft` | 最稳保底 |
| Server-2 | `p1_adapter` | Adapter 泛化主攻 |
| Server-3 | `p2_lora_oft` | LoRA + OFT 对照 |
| Server-4 | `p3_lora_adapter` | LoRA + Adapter 最强尝试 |

在四台服务器分别设置：

```bash
# Server-1
export ROUTE=p0_oft
```

```bash
# Server-2
export ROUTE=p1_adapter
```

```bash
# Server-3
export ROUTE=p2_lora_oft
```

```bash
# Server-4
export ROUTE=p3_lora_adapter
```

GPU 可指定：

```bash
export TRAIN_GPUS=0,1,2,3,4,5,6,7
export NUM_PROCESSES=8
```

## 4. 通用启动命令

每台服务器设置好 `ROUTE`、`MAX_TRAIN_STEPS`、`SAVE_INTERVAL`、`STAGE_NAME` 后，执行同一段启动命令。

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"
export STAR_VLA_PYTHON="$(python -c 'import sys; print(sys.executable)')"

RUN_TS=$(date +"%Y%m%d_%H%M%S")
RUN_ID="${STAGE_NAME}_${ROUTE}_${MAX_TRAIN_STEPS}step"
LOG_DIR="logs/h200_fast_explore/log_${RUN_TS}_${RUN_ID}"
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
echo "STAGE_NAME=${STAGE_NAME}"
echo "LOG_DIR=${LOG_DIR}"
echo "PID=$!"
```

日志开头必须显示：

```text
CALVIN_DATA_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d
CALVIN_DATA_MIX=calvin_abc_d_h200
CALVIN_DATA_NAME=calvin_task_ABC_D
BASE_VLM=<Qwen3.5 path>
NUM_PROCESSES=8
```

## 5. 阶段 A：1k smoke

四台服务器同时跑。目标是确认四条路线都能保存 checkpoint，不在环境、数据、forward、loss 上失败。

```bash
export STAGE_NAME=smoke1k
export MAX_TRAIN_STEPS=1000
export SAVE_INTERVAL=1000
export DATALOADER_NUM_WORKERS=8
```

然后执行第 4 节通用启动命令。

成功标准：

```bash
tail -f "${LOG_DIR}/terminal/train.log"
test -f "${LOG_DIR}/metrics/loss_check.json"
find "${LOG_DIR}/checkpoints" -name "steps_1000_pytorch_model.pt" -print
```

## 6. 阶段 B：10k 快评

四条路线 smoke 成功后立刻跑 10k。目标是尽快得到可评测 checkpoint。

```bash
export STAGE_NAME=fast10k
export MAX_TRAIN_STEPS=10000
export SAVE_INTERVAL=5000
export DATALOADER_NUM_WORKERS=16
```

然后执行第 4 节通用启动命令。

10k 完成后立刻做：

```text
checkpoint reload
debug eval 5 sequences
ABC->D eval 100 sequences
mp4 failure check
```

## 7. 阶段 C：30k 决策

10k 结果没有明显崩的路线继续跑 30k。目标是两天内第一轮路线排序。

```bash
export STAGE_NAME=decision30k
export MAX_TRAIN_STEPS=30000
export SAVE_INTERVAL=10000
export DATALOADER_NUM_WORKERS=16
```

然后执行第 4 节通用启动命令。

30k 后必须筛选：

```text
保留：动作尺度正常、gripper 方向正常、debug eval 有完成迹象、ABC->D 100 条不全崩
淘汰：加载失败、动作爆炸、完全静止、10k/30k 都无改善、eval 链路不稳定
```

## 8. 阶段 D：赢家加训

把四台服务器转给 1-2 条最好路线。推荐优先级：

```text
P3 > P1 > P2 > P0
```

如果 P1/P3 不稳定，P0 继续作为保底。

60k 加训：

```bash
export STAGE_NAME=winner60k
export MAX_TRAIN_STEPS=60000
export SAVE_INTERVAL=10000
export DATALOADER_NUM_WORKERS=16
```

100k 加训：

```bash
export STAGE_NAME=winner100k
export MAX_TRAIN_STEPS=100000
export SAVE_INTERVAL=10000
export DATALOADER_NUM_WORKERS=16
```

然后执行第 4 节通用启动命令。

## 9. 监控命令

```bash
nvidia-smi
tail -f "${LOG_DIR}/terminal/nohup.log"
tail -f "${LOG_DIR}/terminal/train.log"
```

查所有探索结果：

```bash
find logs/h200_fast_explore -path "*/terminal/train.log" -print
find logs/h200_fast_explore -path "*/metrics/loss_check.json" -print
find logs/h200_fast_explore -path "*/checkpoints/*/checkpoints/*.pt" -print
```

## 10. 设置 checkpoint 路径

```bash
export CKPT_PATH=<checkpoint_pt_path>
export ROUTE_LOG_DIR=<route_log_dir>
```

checkpoint 示例：

```text
<LOG_DIR>/checkpoints/<RUN_ID>/checkpoints/steps_10000_pytorch_model.pt
<LOG_DIR>/checkpoints/<RUN_ID>/checkpoints/steps_30000_pytorch_model.pt
```

## 11. checkpoint reload

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

## 12. 启动 policy server

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"
export STAR_VLA_PYTHON="$(python -c 'import sys; print(sys.executable)')"

export CKPT_PATH=<checkpoint_pt_path>
export ROUTE_LOG_DIR=<route_log_dir>
export PORT=5694
export RUN_ID=eval_server
export LOG_DIR="${ROUTE_LOG_DIR}/server"
export CUDA_VISIBLE_DEVICES=0

bash examples/calvin/eval_files/run_policy_server_debug.sh
```

## 13. debug eval 5 条

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
export RUN_ID=debug5
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

## 14. ABC->D 快评 100 条

保持 policy server 不关，eval 终端执行：

```bash
export CKPT_PATH=<checkpoint_pt_path>
export ROUTE_LOG_DIR=<route_log_dir>
export NUM_SEQUENCES=100
export RUN_ID=abcd100
export LOG_DIR="${ROUTE_LOG_DIR}/eval_abcd100"
bash examples/calvin/eval_files/eval_calvin_debug.sh
```

## 15. ABC->D 完整评测

只给 30k 后最好的 1-2 条路线跑完整 1000 条。

```bash
export CKPT_PATH=<checkpoint_pt_path>
export ROUTE_LOG_DIR=<route_log_dir>
export NUM_SEQUENCES=1000
export RUN_ID=abcd1000
export LOG_DIR="${ROUTE_LOG_DIR}/eval_abcd1000"
bash examples/calvin/eval_files/eval_calvin_debug.sh
```

## 16. 两天时间节点

```text
T+0h：四台服务器完成环境、数据、权重检查。
T+0.5h：四条路线同时启动 1k smoke。
T+1h：能保存 checkpoint 的路线进入 10k。
T+3h：10k checkpoint 做 reload、debug5、ABCD100、mp4 检查。
T+6h：保留未崩路线进入 30k。
T+10h：30k 做第一轮路线排序。
T+10h-24h：最好的 1-2 条路线加训 60k/100k。
T+24h-36h：对最好 checkpoint 跑完整 ABCD1000，整理失败视频。
T+36h-48h：补训、补 eval、写报告和 PPT。
```

## 17. 最终取舍

```text
先保证 P0 有可提交结果。
若 P1/P3 的 ABCD100 或 debug mp4 明显更好，优先转向 Adapter。
若 LoRA 路线训练或加载不稳定，立即停掉，不拖主线。
不要等 200k 才评测；10k 和 30k 必须评。
```
