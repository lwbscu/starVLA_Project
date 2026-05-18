# StarVLA CALVIN 本地 Debug 复现步骤

## 1. 克隆仓库

```bash
cd /home/lwb/Projects/SII/starVLA_Projects
git clone https://github.com/lwbscu/starVLA_Project.git
cd /home/lwb/Projects/SII/starVLA_Projects/starVLA_Project
```

## 2. 安装 StarVLA 官方环境

```bash
conda create -n starVLA python=3.10 -y
conda activate starVLA

pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install -e .
```

```bash
conda run -n starVLA python -c "import torch, transformers, starVLA; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"
```

## 3. 安装 Qwen3.5 环境

```bash
cd /home/lwb/Projects/SII/starVLA_Projects/starVLA_Project

conda create -n starVLA_qwen35 python=3.10 -y
conda activate starVLA_qwen35

pip install -r requirements.txt
pip install -U "transformers==5.3.0"
pip install -e .
```

```bash
conda run -n starVLA_qwen35 python -c "from transformers import Qwen3_5ForConditionalGeneration; print('Qwen3.5 import OK')"
conda run -n starVLA_qwen35 python -c "import torch, transformers, starVLA; print(torch.__version__, transformers.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

## 4. 下载模型权重

```bash
cd /home/lwb/Projects/SII/starVLA_Projects/starVLA_Project
mkdir -p playground/Pretrained_models
```

```bash
huggingface-cli download Qwen/Qwen3.5-0.8B \
  --local-dir playground/Pretrained_models/Qwen3.5-0.8B \
  --local-dir-use-symlinks False
```

```bash
huggingface-cli download StarVLA/Qwen2.5-VL-3B-Instruct-Action \
  --local-dir playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action \
  --local-dir-use-symlinks False
```

```bash
test -f playground/Pretrained_models/Qwen3.5-0.8B/config.json && echo "Qwen3.5 OK"
test -f playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action/config.json && echo "Qwen2.5 Action OK"
du -sh playground/Pretrained_models/Qwen3.5-0.8B
du -sh playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action
```

## 5. 下载 CALVIN LeRobot 训练数据

```bash
cd /home/lwb/Projects/SII/starVLA_Projects/starVLA_Project
mkdir -p playground/Datasets/calvin
git lfs install

git clone https://huggingface.co/datasets/CollisionCode/calvin_abc_d_lerobot_v2.1 \
  playground/Datasets/calvin/calvin_abc_d_lerobot_v2.1
```

```bash
test -f playground/Datasets/calvin/calvin_abc_d_lerobot_v2.1/meta/modality.json \
  || cp examples/calvin/train_files/modality.json \
        playground/Datasets/calvin/calvin_abc_d_lerobot_v2.1/meta/modality.json

ls playground/Datasets/calvin/calvin_abc_d_lerobot_v2.1/meta
du -sh playground/Datasets/calvin/calvin_abc_d_lerobot_v2.1
```

本地 smoke 使用已存在的 parquet 轨迹；配置中已启用 `require_existing_parquet: true`，不会把缺失 parquet 当作成功。

## 6. 安装 CALVIN Debug 评测环境

```bash
cd /home/lwb/Projects/SII/starVLA_Projects

git clone --recurse-submodules https://github.com/mees/calvin.git
cd calvin

conda create -n calvin python=3.8 -y
conda activate calvin

pip install "setuptools<58" wheel
conda install -c conda-forge cmake multicore-tsne -y
sh install.sh
pip install tyro websockets msgpack jinja2
```

本地只下载 debug validation：

```bash
cd /home/lwb/Projects/SII/starVLA_Projects/calvin/dataset
sh download_data.sh debug
```

```bash
conda run -n calvin python -c "import calvin_agent; print('calvin_agent import OK')"
conda run -n calvin python -c "import tyro, websockets.sync.client, msgpack, jinja2, imageio; print('calvin eval deps OK')"
find /home/lwb/Projects/SII/starVLA_Projects/calvin/dataset -maxdepth 3 -type d -name validation
```

## 7. 检查数据注册与样本

```bash
cd /home/lwb/Projects/SII/starVLA_Projects/starVLA_Project

conda run -n starVLA_qwen35 python -c "from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES; print(DATASET_NAMED_MIXTURES['calvin_abc_d'])"
```

```bash
NO_ALBUMENTATIONS_UPDATE=1 conda run -n starVLA_qwen35 python -c "from omegaconf import OmegaConf; from starVLA.dataloader.lerobot_datasets import get_vla_dataset; cfg=OmegaConf.create({'data_root_dir':'playground/Datasets/calvin','data_mix':'calvin_abc_d','delete_pause_frame':False,'video_backend':'torchvision_av','load_all_data_for_training':False,'require_existing_parquet':True,'num_trajectories':1}); dataset=get_vla_dataset(cfg); sample=dataset[0]; print('dataset_len', len(dataset)); print('sample_keys', sorted(sample.keys())); print('action_shape', sample['action'].shape); print('lang', sample['lang'])"
```

成功标准：

```text
DATASET_NAMED_MIXTURES['calvin_abc_d'] = [('calvin_abc_d_lerobot_v2.1', 1.0, 'libero_franka')]
sample_keys 包含 action、image、lang、robot_tag
action_shape = (8, 7)
```

## 8. 日志目录

所有脚本默认输出到：

```text
logs/log_YYYYMMDD_HHMMSS_<run_id>/
  train/
  eval/
  terminal/
  mp4/
  configs/
  metrics/
  checkpoints/
```

终端完整日志：

```text
logs/log_YYYYMMDD_HHMMSS_<run_id>/terminal/*.log
```

## 9. Qwen2.5 官方工程 Smoke

Qwen2.5 只用于工程链路验证，不作为考核正式路线。

```bash
cd /home/lwb/Projects/SII/starVLA_Projects/starVLA_Project

CUDA_VISIBLE_DEVICES=0 \
STAR_VLA_PYTHON="$(conda info --base)/envs/starVLA/bin/python" \
bash examples/calvin/train_files/run_calvin_qwen25_smoke.sh
```

本地 8GB GPU 如果 OOM，保留 `terminal/train.log`，不继续 Qwen2.5 eval。

## 10. Qwen3.5-OFT 本地训练 Smoke

1 step：

```bash
cd /home/lwb/Projects/SII/starVLA_Projects/starVLA_Project

CUDA_VISIBLE_DEVICES=0 \
STAR_VLA_PYTHON="$(conda info --base)/envs/starVLA_qwen35/bin/python" \
bash examples/calvin/train_files/run_calvin_qwen35_oft_smoke.sh
```

100 step：

```bash
cd /home/lwb/Projects/SII/starVLA_Projects/starVLA_Project

CUDA_VISIBLE_DEVICES=0 \
MAX_TRAIN_STEPS=100 \
SAVE_INTERVAL=100 \
STAR_VLA_PYTHON="$(conda info --base)/envs/starVLA_qwen35/bin/python" \
bash examples/calvin/train_files/run_calvin_qwen35_oft_smoke.sh
```

checkpoint：

```bash
find logs -path "*qwen35_oft_calvin_smoke/checkpoints/*/checkpoints/*_pytorch_model.pt" | sort
```

## 11. 启动 Qwen3.5 Policy Server

```bash
cd /home/lwb/Projects/SII/starVLA_Projects/starVLA_Project

export CKPT_PATH=logs/log_YYYYMMDD_HHMMSS_qwen35_oft_calvin_smoke/checkpoints/qwen35_oft_calvin_smoke/checkpoints/steps_100_pytorch_model.pt
export PORT=5694
export STAR_VLA_PYTHON="$(conda info --base)/envs/starVLA_qwen35/bin/python"

bash examples/calvin/eval_files/run_policy_server_debug.sh
```

成功标准：

```text
server running
action_chunk_size=8
available_unnorm_keys=['franka']
```

## 12. CALVIN Debug Eval 与 MP4

另开终端执行：

```bash
cd /home/lwb/Projects/SII/starVLA_Projects/starVLA_Project

export CKPT_PATH=logs/log_YYYYMMDD_HHMMSS_qwen35_oft_calvin_smoke/checkpoints/qwen35_oft_calvin_smoke/checkpoints/steps_100_pytorch_model.pt
export PORT=5694
export NUM_SEQUENCES=1
export UNNORM_KEY=franka

bash examples/calvin/eval_files/eval_calvin_debug.sh
```

跑 5 条 debug sequence：

```bash
cd /home/lwb/Projects/SII/starVLA_Projects/starVLA_Project

export CKPT_PATH=logs/log_YYYYMMDD_HHMMSS_qwen35_oft_calvin_smoke/checkpoints/qwen35_oft_calvin_smoke/checkpoints/steps_100_pytorch_model.pt
export PORT=5694
export NUM_SEQUENCES=5
export UNNORM_KEY=franka
export RUN_ID=qwen35_oft_calvin_eval_debug5

bash examples/calvin/eval_files/eval_calvin_debug.sh
```

查看结果：

```bash
find logs -path "*/mp4/*.mp4" | sort
find logs -path "*/mp4/results.json" | sort
```

## 13. Qwen3.5 四路线严谨验证

当前正式候选只使用 `Qwen3.5-0.8B`，不使用 QwenFast、PI、GR00T、Qwen2.5 作为考核路线。

四条路线：

```text
P0: Qwen3.5-0.8B + QwenOFT
P1: Qwen3.5-0.8B + QwenAdapter
P2: Qwen3.5-0.8B + LoRA + QwenOFT
P3: Qwen3.5-0.8B + LoRA + QwenAdapter
```

本地 RTX 4060 8GB 已验证：

```text
P0: 100 step、loss check、checkpoint reload、policy server、debug eval、mp4 通过
P1: 100 step、loss check、checkpoint reload、policy server、debug eval、mp4 通过；本地需 CPU optimizer offload
P2: 100 step、loss check、checkpoint reload、policy server、debug eval、mp4 通过
P3: 100 step、loss check、checkpoint reload、policy server、debug eval、mp4 通过；本地需 CPU optimizer offload
```

100 step 只用于工程闭环验证，不代表 CALVIN 指标。

单路线 smoke 命令：

```bash
cd /home/lwb/Projects/SII/starVLA_Projects/starVLA_Project

ROUTE=p0_oft \
MAX_TRAIN_STEPS=100 \
SAVE_INTERVAL=100 \
CUDA_VISIBLE_DEVICES=0 \
bash examples/calvin/train_files/run_route_validation_train.sh
```

可选路线：

```bash
ROUTE=p0_oft
ROUTE=p1_adapter
ROUTE=p2_lora_oft
ROUTE=p3_lora_adapter
```

本地 8GB 跑 P1/P3 时使用 CPU optimizer offload：

```bash
ROUTE=p1_adapter \
MAX_TRAIN_STEPS=100 \
SAVE_INTERVAL=100 \
CUDA_VISIBLE_DEVICES=0 \
ACCELERATE_CONFIG=starVLA/config/deepseeds/deepspeed_zero2_route_validation_cpu_offload.yaml \
bash examples/calvin/train_files/run_route_validation_train.sh
```

成功标准：

```text
metrics/loss_check.json 中 passed=true
metrics/reload_check.json 或 reload_check_steps_*.json 中 passed=true
checkpoints/<run_id>/checkpoints/steps_<N>_pytorch_model.pt 存在
policy server metadata 中 action_chunk_size=8
available_unnorm_keys 包含 franka
eval 生成 results.json 和 mp4
```

## 14. H200 服务器环境

服务器建议目录仍保持一致：

```bash
cd /home/lwb/Projects/SII/starVLA_Projects
git clone https://github.com/lwbscu/starVLA_Project.git
cd /home/lwb/Projects/SII/starVLA_Projects/starVLA_Project
git checkout starVLA_dev
```

安装环境：

```bash
conda create -n starVLA_qwen35 python=3.10 -y
conda activate starVLA_qwen35

pip install -r requirements.txt
pip install -U "transformers==5.3.0"
pip install flash-attn --no-build-isolation
pip install -e .
```

验证：

```bash
conda run -n starVLA_qwen35 python -c "from transformers import Qwen3_5ForConditionalGeneration; print('Qwen3.5 import OK')"
conda run -n starVLA_qwen35 python -c "import torch, peft, starVLA; print(torch.__version__, peft.__version__, torch.cuda.device_count())"
```

准备权重：

```bash
mkdir -p playground/Pretrained_models

git lfs install
git clone https://huggingface.co/Qwen/Qwen3.5-0.8B \
  playground/Pretrained_models/Qwen3.5-0.8B
```

如果 `huggingface-cli download` 在服务器网络环境稳定，也可以使用：

```bash
huggingface-cli download Qwen/Qwen3.5-0.8B \
  --local-dir playground/Pretrained_models/Qwen3.5-0.8B \
  --local-dir-use-symlinks False
```

准备 LeRobot 训练数据：

```bash
mkdir -p playground/Datasets/calvin
git lfs install
git clone https://huggingface.co/datasets/CollisionCode/calvin_abc_d_lerobot_v2.1 \
  playground/Datasets/calvin/calvin_abc_d_lerobot_v2.1

test -f playground/Datasets/calvin/calvin_abc_d_lerobot_v2.1/meta/modality.json \
  || cp examples/calvin/train_files/modality.json \
        playground/Datasets/calvin/calvin_abc_d_lerobot_v2.1/meta/modality.json
```

检查：

```bash
test -f playground/Pretrained_models/Qwen3.5-0.8B/config.json && echo "Qwen3.5 OK"
test -f playground/Datasets/calvin/calvin_abc_d_lerobot_v2.1/meta/modality.json && echo "CALVIN LeRobot OK"

NO_ALBUMENTATIONS_UPDATE=1 conda run -n starVLA_qwen35 python -c "from omegaconf import OmegaConf; from starVLA.dataloader.lerobot_datasets import get_vla_dataset; cfg=OmegaConf.create({'data_root_dir':'playground/Datasets/calvin','data_mix':'calvin_abc_d','delete_pause_frame':False,'video_backend':'torchvision_av','load_all_data_for_training':False,'require_existing_parquet':True,'num_trajectories':1}); dataset=get_vla_dataset(cfg); sample=dataset[0]; print(len(dataset), sample['action'].shape, sample['robot_tag'])"
```

## 15. H200 四路线 Smoke

先做 100 step 并行 smoke，确认服务器环境、数据、checkpoint 和 reload 全部正常。

默认使用 GPU 0/1/2/3 分别跑 P0/P1/P2/P3：

```bash
cd /home/lwb/Projects/SII/starVLA_Projects/starVLA_Project

MAX_TRAIN_STEPS=100 \
SAVE_INTERVAL=100 \
GPU_LIST="0 1 2 3" \
ROUTE_LIST="p0_oft p1_adapter p2_lora_oft p3_lora_adapter" \
bash examples/calvin/train_files/run_route_h200_matrix.sh
```

输出目录：

```text
logs/h200_route_train/log_YYYYMMDD_HHMMSS_qwen35_0p8b_matrix/
  h200_p0_oft_100step/
  h200_p1_adapter_100step/
  h200_p2_lora_oft_100step/
  h200_p3_lora_adapter_100step/
```

检查：

```bash
find logs/h200_route_train -path "*/metrics/loss_check.json" | sort
find logs/h200_route_train -path "*/metrics/reload_check_steps_100.json" | sort
find logs/h200_route_train -path "*/checkpoints/*/checkpoints/steps_100_pytorch_model.pt" | sort
```

## 16. H200 四路线长训

100 step smoke 全部通过后，启动 30k 长训：

```bash
cd /home/lwb/Projects/SII/starVLA_Projects/starVLA_Project

MAX_TRAIN_STEPS=30000 \
SAVE_INTERVAL=5000 \
GPU_LIST="0 1 2 3" \
ROUTE_LIST="p0_oft p1_adapter p2_lora_oft p3_lora_adapter" \
bash examples/calvin/train_files/run_route_h200_matrix.sh
```

如果只跑某一条路线：

```bash
MAX_TRAIN_STEPS=30000 \
SAVE_INTERVAL=5000 \
GPU_LIST="0" \
ROUTE_LIST="p0_oft" \
bash examples/calvin/train_files/run_route_h200_matrix.sh
```

如果要用 8 张 H200 同时跑两组随机对照，使用不同 `LOG_ROOT` 和不同 GPU：

```bash
LOG_ROOT=logs/h200_route_train/log_runA \
MAX_TRAIN_STEPS=30000 \
SAVE_INTERVAL=5000 \
GPU_LIST="0 1 2 3" \
ROUTE_LIST="p0_oft p1_adapter p2_lora_oft p3_lora_adapter" \
bash examples/calvin/train_files/run_route_h200_matrix.sh
```

```bash
LOG_ROOT=logs/h200_route_train/log_runB \
MAX_TRAIN_STEPS=30000 \
SAVE_INTERVAL=5000 \
GPU_LIST="4 5 6 7" \
ROUTE_LIST="p0_oft p1_adapter p2_lora_oft p3_lora_adapter" \
bash examples/calvin/train_files/run_route_h200_matrix.sh
```

长训通过标准：

```text
训练进程 exit code = 0
metrics/loss_check.json 中 passed=true
metrics/reload_check_steps_30000.json 中 passed=true
steps_5000/10000/15000/20000/25000/30000 checkpoint 存在
terminal/train.log 中无 NaN、Inf、OOM、checkpoint load error
```

## 17. H200 训练后 Eval

对每条路线选取 `steps_10000`、`steps_20000`、`steps_30000` 做 debug eval，再挑候选跑完整 CALVIN ABC→D eval。

启动 policy server：

```bash
cd /home/lwb/Projects/SII/starVLA_Projects/starVLA_Project

export CKPT_PATH=logs/h200_route_train/log_YYYYMMDD_HHMMSS_qwen35_0p8b_matrix/h200_p0_oft_30000step/checkpoints/h200_p0_oft_30000step/checkpoints/steps_30000_pytorch_model.pt
export PORT=5694
export CUDA_VISIBLE_DEVICES=0
export STAR_VLA_PYTHON="$(conda info --base)/envs/starVLA_qwen35/bin/python"

bash examples/calvin/eval_files/run_policy_server_debug.sh
```

另开终端跑 debug eval：

```bash
cd /home/lwb/Projects/SII/starVLA_Projects/starVLA_Project

export CKPT_PATH=logs/h200_route_train/log_YYYYMMDD_HHMMSS_qwen35_0p8b_matrix/h200_p0_oft_30000step/checkpoints/h200_p0_oft_30000step/checkpoints/steps_30000_pytorch_model.pt
export PORT=5694
export NUM_SEQUENCES=5
export UNNORM_KEY=franka
export RUN_ID=h200_p0_oft_30000_eval_debug5

bash examples/calvin/eval_files/eval_calvin_debug.sh
```

评估产物：

```text
terminal/eval.log
mp4/results.json
mp4/*.mp4
```

权重选择规则：

```text
优先看 debug/full eval 的平均任务链长度
再看 Task1 成功率
再看 failure pattern 是否容易针对性修复
不只按训练 loss 选择 checkpoint
不提交没有 results.json 和 mp4 的 checkpoint
```

输出文件：

```text
logs/log_YYYYMMDD_HHMMSS_qwen35_oft_calvin_eval_debug*/terminal/eval.log
logs/log_YYYYMMDD_HHMMSS_qwen35_oft_calvin_eval_debug*/mp4/*.mp4
logs/log_YYYYMMDD_HHMMSS_qwen35_oft_calvin_eval_debug*/mp4/results.json
```

## 13. 关闭 Policy Server

在 server 终端按：

```text
Ctrl-C
```

确认没有残留：

```bash
pgrep -af "server_policy.py|run_policy_server_debug.sh" || true
```

## 14. 本机已验证产物

Qwen3.5 100 step checkpoint：

```text
logs/log_20260518_052007_qwen35_oft_calvin_smoke/checkpoints/qwen35_oft_calvin_smoke/checkpoints/steps_100_pytorch_model.pt
```

Qwen3.5 5 条 CALVIN debug eval：

```text
logs/log_20260518_052655_qwen35_oft_calvin_eval_debug5/terminal/eval.log
logs/log_20260518_052655_qwen35_oft_calvin_eval_debug5/mp4/results.json
logs/log_20260518_052655_qwen35_oft_calvin_eval_debug5/mp4/0-0-rotate_blue_block_right-fail.mp4
logs/log_20260518_052655_qwen35_oft_calvin_eval_debug5/mp4/1-0-turn_off_led-fail.mp4
logs/log_20260518_052655_qwen35_oft_calvin_eval_debug5/mp4/2-0-lift_pink_block_slider-fail.mp4
logs/log_20260518_052655_qwen35_oft_calvin_eval_debug5/mp4/3-0-rotate_blue_block_right-fail.mp4
logs/log_20260518_052655_qwen35_oft_calvin_eval_debug5/mp4/4-0-open_drawer-fail.mp4
```

本地 100 step 仅用于链路复现，指标不代表最终 H200 训练结果。
