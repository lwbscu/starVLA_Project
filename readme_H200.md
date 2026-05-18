# H200 五路线快速探索与自动评测命令

目标：用完整 `calvin_task_ABC_D` 训练，用 `task_D_D` 测试。每轮 checkpoint 后自动跑 D 环境可视化评测。所有日志、checkpoint、eval 结果和 mp4 统一写入：

```text
logs/h200_fastexplore/
```

当前服务器数据结论：

```text
训练数据：calvin_task_ABC_D
训练范围：全部 17870 episodes / 1071743 frames，不从训练集切 eval 子集
格式：LeRobot/HF v2.1
规模：17870 episodes, 1071743 frames, 17870 parquet, 35740 mp4

快速评测环境：task_D_D
格式：原始 CALVIN npz
可用配置：task_D_D/training/.hydra/merged_config.yaml
限制：没有 validation/，所以这是 D 环境 smoke/可视化评估，不是官方 validation 指标。
```

五条路线：

```text
P0: p0_oft              Qwen3.5 + OFT
P1: p1_adapter          Qwen3.5 + Adapter
P2: p2_lora_oft         Qwen3.5 + LoRA + OFT
P3: p3_lora_adapter     Qwen3.5 + LoRA + Adapter
P4: p4_pi               Qwen3.5-9B + PI / Flow-Matching
```

本地单独 smoke / 短训 PI（不跑五路线流水线）见：[examples/calvin/train_files/README_qwen35_pi_train.md](examples/calvin/train_files/README_qwen35_pi_train.md)。

默认正式节奏：

```text
1k smoke -> D 环境 eval + mp4
10k 快评 -> D 环境 eval + mp4
30k 决策 -> D 环境 eval + mp4
```

## 1. 每台服务器先执行

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
export H200_CALVIN_EVAL_DATASET_PATH="${H200_CALVIN_DATA_ROOT}/task_D_D"
export H200_QWEN35_9B="${PROJECT_ROOT}/playground/Pretrained_models/Qwen3.5-9B"
export BASE_VLM="${H200_QWEN35_9B}"

export CALVIN_CONFIG_PATH="${PROJECT_ROOT}/calvin/calvin_models/conf"
export EVAL_SEQUENCES_PATH=examples/calvin/eval_files/eval_sequences.json
export CALVIN_PYTHON="${CONDA_ROOT}/envs/calvin/bin/python"
export GIT_PYTHON_REFRESH=quiet
export CALVIN_ALLOW_OFFLINE_GIT_METADATA=1
export CALVIN_FORCE_NO_EGL=1

export LOG_ROOT=logs/h200_fastexplore
export OBS_IMAGE_SIZE='[224,224]'
export ACTION_HORIZON=8
export DATALOADER_NUM_WORKERS=16

export EVAL_ENABLED=1
export EVAL_UNNORM_KEY=franka
export SMOKE_EVAL_SEQUENCES=1
export FAST_EVAL_SEQUENCES=3
export DECISION_EVAL_SEQUENCES=5
```

## 2. 环境和数据检查

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"

python -c "from transformers import Qwen3_5ForConditionalGeneration; print('Qwen3.5 import OK')"

test -f "${H200_CALVIN_DATASET_PATH}/meta/info.json"
test -f "${H200_CALVIN_DATASET_PATH}/meta/modality.json"
test -d "${H200_CALVIN_DATASET_PATH}/data"
test -d "${H200_CALVIN_DATASET_PATH}/videos"

test -f "${H200_CALVIN_EVAL_DATASET_PATH}/training/.hydra/merged_config.yaml"
test -d "${CALVIN_CONFIG_PATH}"
test -f "${EVAL_SEQUENCES_PATH}"
test -x "${CALVIN_PYTHON}"
"${CALVIN_PYTHON}" -c "import cv2; print('cv2 import OK', cv2.__version__)"
"${CALVIN_PYTHON}" -c "import os, git; print('GitPython import OK', os.environ.get('GIT_PYTHON_REFRESH'), os.environ.get('CALVIN_ALLOW_OFFLINE_GIT_METADATA'))"

nvidia-smi
```

所有路线都强制使用 9B 权重：

```bash
test -f "${H200_QWEN35_9B}/config.json"
```

如果 `cv2` 报 `ImportError: libGL.so.1`，先修 `calvin` 环境再跑训练。评测必须产出 mp4，所以这里不能跳过：

```bash
cd "${PROJECT_ROOT}"
export PATH="${CONDA_ROOT}/bin:${PATH}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
export CALVIN_PYTHON="${CONDA_ROOT}/envs/calvin/bin/python"

"${CALVIN_PYTHON}" -c "import sys; print(sys.executable); print(sys.version)"

"${CALVIN_PYTHON}" -m pip uninstall -y \
  opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless
"${CALVIN_PYTHON}" -m pip install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn \
  --timeout 120 --retries 20 --root-user-action=ignore \
  "opencv-python-headless==4.11.0.86"

"${CALVIN_PYTHON}" -c "import sys, cv2; print(sys.executable); print('cv2 import OK', cv2.__version__, cv2.__file__)"
conda activate "${STARVLA_ENV}"
```

如果安装日志里出现 `cp313` 这类 Python 3.13 wheel，但 `${CALVIN_PYTHON}` 是 Python 3.8，说明裸 `pip/python` 修的是另一个环境；必须重跑上面的绝对路径命令。

如果清华源慢，替换为阿里源：

```bash
"${CALVIN_PYTHON}" -m pip install \
  -i https://mirrors.aliyun.com/pypi/simple/ \
  --trusted-host mirrors.aliyun.com \
  --timeout 120 --retries 20 --root-user-action=ignore \
  "opencv-python-headless==4.11.0.86"
```

如果评测日志卡在 `Loading EGL plugin` 后报 `failed to EGL with glad`，说明服务器 EGL 渲染栈不可用。当前脚本默认走 PyBullet DIRECT/no-EGL 路径：

```bash
export CALVIN_FORCE_NO_EGL=1
```

这不是跳过评测；脚本仍要求生成 `results.json` 和至少一个 mp4，否则会报错退出。

## 3. 先测试已有 P0 30k 权重

你之前的 P0 训练已经完成：

```text
logs/h200_route_train/log_20260518_151413_h200_p0_oft_30000step
```

先找 checkpoint：

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"

find logs/h200_route_train/log_20260518_151413_h200_p0_oft_30000step \
  -type f \( -name "steps_30000_pytorch_model.pt" -o -name "model.safetensors" -o -name "*.pt" \) | sort
```

设置 checkpoint。优先使用 `steps_30000_pytorch_model.pt`：

```bash
export CKPT_PATH=$(find logs/h200_route_train/log_20260518_151413_h200_p0_oft_30000step \
  -type f -name "steps_30000_pytorch_model.pt" | sort | tail -n 1)

test -f "${CKPT_PATH}"
echo "${CKPT_PATH}"
```

先做 checkpoint reload：

```bash
python examples/calvin/eval_files/check_checkpoint_reload.py \
  --ckpt-path "${CKPT_PATH}" \
  --expected-action-chunk-size 8 \
  --expected-unnorm-key franka \
  --output-json logs/h200_route_train/p0_30k_reload_check.json
```

启动 policy server：

```bash
export PORT=5694
export CUDA_VISIBLE_DEVICES=0
export RUN_ID=p0_30k_server
export LOG_DIR=logs/h200_route_train/p0_30k_server

bash examples/calvin/eval_files/run_policy_server_debug.sh
```

另开终端跑 D 环境 3 条序列评测并输出 mp4：

```bash
export PROJECT_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project
export CONDA_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/26220216/miniconda3
export PATH="${CONDA_ROOT}/bin:${PATH}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

cd "${PROJECT_ROOT}"
conda activate calvin

export CKPT_PATH=$(find logs/h200_route_train/log_20260518_151413_h200_p0_oft_30000step \
  -type f -name "steps_30000_pytorch_model.pt" | sort | tail -n 1)
export HOST=127.0.0.1
export PORT=5694
export GIT_PYTHON_REFRESH=quiet
export CALVIN_ALLOW_OFFLINE_GIT_METADATA=1
export CALVIN_FORCE_NO_EGL=1
export NUM_SEQUENCES=3
export UNNORM_KEY=franka
export RUN_ID=p0_30k_d_env_eval3
export LOG_DIR=logs/h200_route_train/p0_30k_d_env_eval3
export DATASET_PATH=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d/task_D_D
export CALVIN_CONFIG_PATH="${PROJECT_ROOT}/calvin/calvin_models/conf"
export EVAL_SEQUENCES_PATH=examples/calvin/eval_files/eval_sequences.json

bash examples/calvin/eval_files/eval_calvin_debug.sh
```

检查输出：

```bash
find logs/h200_route_train/p0_30k_d_env_eval3 -type f \( -name "results.json" -o -name "*.mp4" \) | sort
```

## 4. 多路线快速脚本测试

正式跑之前，先用极短步数完整验证链路：训练、保存 checkpoint、reload、启动 server、D 环境 eval、输出 mp4。

注意两个端口不是一回事：

```text
MAIN_PROCESS_PORT: accelerate / deepspeed 分布式训练 rendezvous 端口。
EVAL_PORT: policy server 推理评测端口。
```

Server-1 同时跑 P0 和 P4，必须给两条训练路线不同的 `MAIN_PROCESS_PORT`。否则会出现 `EADDRINUSE: address already in use`。

当前 P4 使用 `p4_pi` 名称；所有路线都强制 9B：

```bash
export H200_QWEN35_9B="${PROJECT_ROOT}/playground/Pretrained_models/Qwen3.5-9B"
export BASE_VLM="${H200_QWEN35_9B}"
```

如果上一次 eval 失败后端口仍被旧 policy server 或旧 accelerate 进程占用，脚本会直接报错并打印占用 PID/命令。重新测试时优先换一组干净端口：

```bash
export P0_MAIN_PROCESS_PORT=30600
export P4_MAIN_PROCESS_PORT=30610
export P0_EVAL_PORT=6094
export P4_EVAL_PORT=6095
```

需要定位旧进程时，用训练环境 Python 查看端口占用：

```bash
"${STAR_VLA_PYTHON}" examples/calvin/train_files/describe_port_users.py 29600 29610 5894 5895 6094 6095
```

### Server-1 快速测试 P0 + P4

```bash
cd "${PROJECT_ROOT}"
export CONDA_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/26220216/miniconda3
export PATH="${CONDA_ROOT}/bin:${PATH}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${STARVLA_ENV}"

export LOG_ROOT=logs/h200_fastexplore_test
export SMOKE_STEPS=20
export FAST_STEPS=50
export DECISION_STEPS=100
export SMOKE_SAVE_INTERVAL=20
export FAST_SAVE_INTERVAL=50
export DECISION_SAVE_INTERVAL=100
export SMOKE_EVAL_SEQUENCES=1
export FAST_EVAL_SEQUENCES=1
export DECISION_EVAL_SEQUENCES=1

export P0_GPUS=0,1,2,3
export P0_NUM_PROCESSES=4
export P0_MAIN_PROCESS_PORT=30600
export P4_GPUS=4,5,6,7
export P4_NUM_PROCESSES=4
export P4_MAIN_PROCESS_PORT=30610
export P4_BASE_VLM="${H200_QWEN35_9B}"
export P0_EVAL_PORT=6094
export P4_EVAL_PORT=6095
export P0_EVAL_GPU=0
export P4_EVAL_GPU=4

bash examples/calvin/train_files/run_h200_fastexplore_server1.sh
```

### Server-2 快速测试 P1

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"

export LOG_ROOT=logs/h200_fastexplore_test
export SMOKE_STEPS=20
export FAST_STEPS=50
export DECISION_STEPS=100
export SMOKE_SAVE_INTERVAL=20
export FAST_SAVE_INTERVAL=50
export DECISION_SAVE_INTERVAL=100
export SMOKE_EVAL_SEQUENCES=1
export FAST_EVAL_SEQUENCES=1
export DECISION_EVAL_SEQUENCES=1

export TRAIN_GPUS=0,1,2,3,4,5,6,7
export NUM_PROCESSES=8
export MAIN_PROCESS_PORT=29600
export EVAL_PORT=5694
export EVAL_GPU=0

bash examples/calvin/train_files/run_h200_fastexplore_server2.sh
```

### Server-3 快速测试 P2

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"

export LOG_ROOT=logs/h200_fastexplore_test
export SMOKE_STEPS=20
export FAST_STEPS=50
export DECISION_STEPS=100
export SMOKE_SAVE_INTERVAL=20
export FAST_SAVE_INTERVAL=50
export DECISION_SAVE_INTERVAL=100
export SMOKE_EVAL_SEQUENCES=1
export FAST_EVAL_SEQUENCES=1
export DECISION_EVAL_SEQUENCES=1

export TRAIN_GPUS=0,1,2,3,4,5,6,7
export NUM_PROCESSES=8
export MAIN_PROCESS_PORT=29600
export EVAL_PORT=5694
export EVAL_GPU=0

bash examples/calvin/train_files/run_h200_fastexplore_server3.sh
```

### Server-4 快速测试 P3

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"

export LOG_ROOT=logs/h200_fastexplore_test
export SMOKE_STEPS=20
export FAST_STEPS=50
export DECISION_STEPS=100
export SMOKE_SAVE_INTERVAL=20
export FAST_SAVE_INTERVAL=50
export DECISION_SAVE_INTERVAL=100
export SMOKE_EVAL_SEQUENCES=1
export FAST_EVAL_SEQUENCES=1
export DECISION_EVAL_SEQUENCES=1

export TRAIN_GPUS=0,1,2,3,4,5,6,7
export NUM_PROCESSES=8
export MAIN_PROCESS_PORT=29600
export EVAL_PORT=5694
export EVAL_GPU=0

bash examples/calvin/train_files/run_h200_fastexplore_server4.sh
```

### 快速测试结果比较

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"

LOG_ROOT=logs/h200_fastexplore_test \
bash examples/calvin/train_files/compare_h200_fastexplore.sh

cat logs/h200_fastexplore_test/summary/compare_routes.md
find logs/h200_fastexplore_test -type f -name "*.mp4" | sort | head -20
```

快速测试必须满足：

```text
1. 每条路线至少保存 checkpoint。
2. reload_check passed。
3. 每条路线至少生成一个 results.json。
4. 每条路线至少生成一个 mp4。
```

## 5. 正式一小时五路线探索

确认第 4 节没问题后，重新开干净终端，执行第 1 节公共环境变量，然后不要覆盖 `SMOKE_STEPS/FAST_STEPS/DECISION_STEPS`，使用默认：

```text
SMOKE_STEPS=1000
FAST_STEPS=10000
DECISION_STEPS=30000
```

### Server-1 正式跑 P0 + P4

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"

export LOG_ROOT=logs/h200_fastexplore
export P0_GPUS=0,1,2,3
export P0_NUM_PROCESSES=4
export P0_MAIN_PROCESS_PORT=29600
export P4_GPUS=4,5,6,7
export P4_NUM_PROCESSES=4
export P4_MAIN_PROCESS_PORT=29610
export P4_BASE_VLM="${H200_QWEN35_9B}"
export P0_EVAL_PORT=5794
export P4_EVAL_PORT=5795
export P0_EVAL_GPU=0
export P4_EVAL_GPU=4

bash examples/calvin/train_files/run_h200_fastexplore_server1.sh
```

### Server-2 正式跑 P1

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"

export LOG_ROOT=logs/h200_fastexplore
export TRAIN_GPUS=0,1,2,3,4,5,6,7
export NUM_PROCESSES=8
export MAIN_PROCESS_PORT=29600
export EVAL_PORT=5694
export EVAL_GPU=0

bash examples/calvin/train_files/run_h200_fastexplore_server2.sh
```

### Server-3 正式跑 P2

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"

export LOG_ROOT=logs/h200_fastexplore
export TRAIN_GPUS=0,1,2,3,4,5,6,7
export NUM_PROCESSES=8
export MAIN_PROCESS_PORT=29600
export EVAL_PORT=5694
export EVAL_GPU=0

bash examples/calvin/train_files/run_h200_fastexplore_server3.sh
```

### Server-4 正式跑 P3

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"

export LOG_ROOT=logs/h200_fastexplore
export TRAIN_GPUS=0,1,2,3,4,5,6,7
export NUM_PROCESSES=8
export MAIN_PROCESS_PORT=29600
export EVAL_PORT=5694
export EVAL_GPU=0

bash examples/calvin/train_files/run_h200_fastexplore_server4.sh
```

## 6. 监控

```bash
nvidia-smi
find logs/h200_fastexplore -path "*/terminal/train.log" -print
find logs/h200_fastexplore -path "*/metrics/loss_check.json" -print
find logs/h200_fastexplore -path "*/metrics/reload_check_steps_*.json" -print
find logs/h200_fastexplore -path "*/checkpoints/*/checkpoints/*.pt" -print
find logs/h200_fastexplore \( -path "*/eval_*/*.json" -o -path "*/eval_*/*.mp4" \) -print
```

看单条路线：

```bash
tail -f logs/h200_fastexplore/terminal/*p0_oft_pipeline.log
tail -f logs/h200_fastexplore/terminal/*p1_adapter_pipeline.log
tail -f logs/h200_fastexplore/terminal/*p2_lora_oft_pipeline.log
tail -f logs/h200_fastexplore/terminal/*p3_lora_adapter_pipeline.log
tail -f logs/h200_fastexplore/terminal/*p4_pi_pipeline.log
```

## 7. 一键比较

```bash
cd "${PROJECT_ROOT}"
conda activate "${STARVLA_ENV}"

bash examples/calvin/train_files/compare_h200_fastexplore.sh
cat logs/h200_fastexplore/summary/compare_routes.md
```

重点看：

```text
eval_avg_seq_len
mp4_count
First MP4
loss_passed
reload_passed
latest_checkpoint
```

打开视频：

```bash
find logs/h200_fastexplore -type f -name "*.mp4" | sort | head -30
```

## 8. 当前判断标准

一小时内优先选：

```text
1. 30k 阶段完成。
2. loss_check passed。
3. reload_check passed。
4. D 环境 eval 能产生 results.json 和 mp4。
5. mp4 中动作不是完全静止，不爆炸，夹爪方向基本合理。
6. eval_avg_seq_len 更高。
```

如果没有正式 validation，不要把 D_D training smoke eval 写成最终 CALVIN validation 指标。
