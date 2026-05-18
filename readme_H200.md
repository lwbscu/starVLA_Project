# H200 五路线一小时快速探索

目标：四台服务器并行跑五条路线，所有日志统一写入：

```text
logs/h200_fastexplore/
```

五条路线：

```text
P0: p0_oft              Qwen3.5 + OFT
P1: p1_adapter          Qwen3.5 + Adapter
P2: p2_lora_oft         Qwen3.5 + LoRA + OFT
P3: p3_lora_adapter     Qwen3.5 + LoRA + Adapter
P4: p4_qwen4b_pi        Qwen3.5-4B + PI / Flow-Matching
```

节奏：

```text
1k smoke -> 10k 快评 -> 30k 决策
```

## 0. 每台服务器先执行

```bash
export PROJECT_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project
export CONDA_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/26220216/miniconda3
export STARVLA_ENV=starVLA_qwen35

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

export LOG_ROOT=logs/h200_fastexplore
export OBS_IMAGE_SIZE='[224,224]'
export ACTION_HORIZON=8
export DATALOADER_NUM_WORKERS=16
```

## 1. 检查环境和数据

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"

python -c "from transformers import Qwen3_5ForConditionalGeneration; print('Qwen3.5 import OK')"
test -d "${H200_CALVIN_DATASET_PATH}"
test -f "${H200_CALVIN_DATASET_PATH}/meta/info.json"
test -f "${H200_CALVIN_DATASET_PATH}/meta/modality.json" || cp examples/calvin/train_files/modality.json "${H200_CALVIN_DATASET_PATH}/meta/modality.json"
test -d "${H200_CALVIN_DATASET_PATH}/data"
nvidia-smi
```

P4 需要 4B 权重：

```bash
test -f playground/Pretrained_models/Qwen3.5-4B/config.json
```

## 2. 四台服务器启动命令

### Server-1：同时跑 P0 和 P4

P0 默认用 GPU0-3，P4 默认用 GPU4-7。

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"

export P0_GPUS=0,1,2,3
export P0_NUM_PROCESSES=4
export P4_GPUS=4,5,6,7
export P4_NUM_PROCESSES=4
export P4_BASE_VLM=./playground/Pretrained_models/Qwen3.5-4B

bash examples/calvin/train_files/run_h200_fastexplore_server1.sh
```

### Server-2：跑 P1

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"

export TRAIN_GPUS=0,1,2,3,4,5,6,7
export NUM_PROCESSES=8

bash examples/calvin/train_files/run_h200_fastexplore_server2.sh
```

### Server-3：跑 P2

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"

export TRAIN_GPUS=0,1,2,3,4,5,6,7
export NUM_PROCESSES=8

bash examples/calvin/train_files/run_h200_fastexplore_server3.sh
```

### Server-4：跑 P3

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"

export TRAIN_GPUS=0,1,2,3,4,5,6,7
export NUM_PROCESSES=8

bash examples/calvin/train_files/run_h200_fastexplore_server4.sh
```

## 3. 脚本会自动做什么

每条路线自动执行：

```text
smoke1k:      train 1000 steps, save steps_1000 checkpoint, reload check
fast10k:      train 10000 steps, save steps_5000/steps_10000 checkpoint, reload check
decision30k:  train 30000 steps, save steps_10000/steps_20000/steps_30000 checkpoint, reload check
```

每个阶段失败就直接停止该路线，不会伪装成功。

## 4. 监控

```bash
nvidia-smi
find logs/h200_fastexplore -path "*/terminal/train.log" -print
find logs/h200_fastexplore -path "*/metrics/loss_check.json" -print
find logs/h200_fastexplore -path "*/metrics/reload_check_steps_*.json" -print
find logs/h200_fastexplore -path "*/checkpoints/*/checkpoints/*.pt" -print
```

看某条路线：

```bash
tail -f logs/h200_fastexplore/terminal/*p0_oft_pipeline.log
tail -f logs/h200_fastexplore/terminal/*p4_qwen4b_pi_pipeline.log
```

## 5. 一键比较五条路线

在任意一台能看到共享 `logs/h200_fastexplore/` 的服务器执行：

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"

bash examples/calvin/train_files/compare_h200_fastexplore.sh
```

输出：

```text
logs/h200_fastexplore/summary/compare_routes.md
logs/h200_fastexplore/summary/compare_routes.csv
```

查看：

```bash
cat logs/h200_fastexplore/summary/compare_routes.md
```

## 6. 比较标准

优先级：

```text
1. 有 eval 结果时，优先看 avg_seq_len。
2. 没有 eval 结果时，先看是否完成 decision30k。
3. 同阶段比较 loss_check 是否 passed。
4. 同阶段比较 checkpoint reload 是否 passed。
5. loss 接近时优先 P3 / P1，再 P4，再 P2，再 P0。
```

一小时后先比较：

```text
是否完成 1k、10k、30k
loss_check 是否 passed
reload_check 是否 passed
latest_checkpoint 是否存在
last_loss 是否异常
```

如果某条路线没跑完 30k，也不要等它；先用已完成阶段比较。

## 7. 给最好路线补 eval

设置 checkpoint：

```bash
export CKPT_PATH=<checkpoint_pt_path>
export ROUTE_LOG_DIR=<route_log_dir>
```

启动 policy server：

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"
export STAR_VLA_PYTHON="$(python -c 'import sys; print(sys.executable)')"

export PORT=5694
export RUN_ID=eval_server
export LOG_DIR="${ROUTE_LOG_DIR}/server"
export CUDA_VISIBLE_DEVICES=0

bash examples/calvin/eval_files/run_policy_server_debug.sh
```

另开终端跑 ABCD100 快评：

```bash
export PROJECT_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project
export CONDA_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/26220216/miniconda3
export PATH="${CONDA_ROOT}/bin:${PATH}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

cd "${PROJECT_ROOT}"
conda activate calvin

export CKPT_PATH=<checkpoint_pt_path>
export ROUTE_LOG_DIR=<route_log_dir>
export HOST=127.0.0.1
export PORT=5694
export NUM_SEQUENCES=100
export UNNORM_KEY=franka
export RUN_ID=abcd100
export LOG_DIR="${ROUTE_LOG_DIR}/eval_abcd100"
export DATASET_PATH=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d/calvin_task_ABC_D
export CALVIN_CONFIG_PATH="${PROJECT_ROOT}/calvin/calvin_models/conf"

bash examples/calvin/eval_files/eval_calvin_debug.sh
```

eval 完成后重新比较：

```bash
bash examples/calvin/train_files/compare_h200_fastexplore.sh
cat logs/h200_fastexplore/summary/compare_routes.md
```
