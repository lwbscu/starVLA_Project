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
