# Qwen3.5 + QwenPI（LayerwiseFM）CALVIN 训练说明

本文档对应 **StarVLA-π** 路线：`framework.name=QwenPI` + `LayerwiseFM` 动作头（flow matching + 层间 cross-DiT）。  
与 **QwenOFT** 的 smoke 脚本对称，便于在 CALVIN LeRobot 数据上单独调试 PI。

> 数据集准备、评测环境等通用说明见 [examples/calvin/README.md](../README.md)。  
> H200 五路线流水线（含 P4 与 OFT 并行）见仓库根目录 [readme_H200.md](../../../readme_H200.md)。

---

## 1. 相关文件

| 文件 | 用途 |
|------|------|
| `run_calvin_qwen35_pi_smoke.sh` | 本地 smoke / 短训入口（对称 `run_calvin_qwen35_oft_smoke.sh`） |
| `starvla_train_calvin_qwen35_pi_smoke.yaml` | 本地配置：默认 `Qwen3.5-0.8B`，`calvin_abc_d`，`obs_image_size: [112,112]` |
| `starvla_train_calvin_qwen35_pi_h200.yaml` | H200 配置：默认 `Qwen3.5-4B`，`calvin_abc_d_h200`，`obs_image_size: [224,224]` |
| `starVLA/model/framework/VLM4A/QwenPI.py` | Framework：VLM 多层 hidden → action head |
| `starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py` | Flow-matching 动作头实现 |

**对比 OFT：**

| | OFT | PI（本文档） |
|---|-----|----------------|
| 脚本 | `run_calvin_qwen35_oft_smoke.sh` | `run_calvin_qwen35_pi_smoke.sh` |
| YAML | `starvla_train_calvin_qwen35_oft_smoke.yaml` | `starvla_train_calvin_qwen35_pi_smoke.yaml` |
| Framework | `QwenOFT` | `QwenPI` |
| Action 头 | `MLP` | `LayerwiseFM` |
| 默认 VLM | `Qwen3.5-0.8B` | `Qwen3.5-0.8B`（smoke）；H200 建议 `4B` |
| 显存 | 较低 | **明显更高** |

---

## 2. 环境与依赖

### Conda 环境

与 Qwen3.5-OFT 相同，使用 **`starVLA_qwen35`**（见 [README_zh.md](../../../README_zh.md) 安装章节）。

```bash
conda activate starVLA_qwen35
python -c "from transformers import Qwen3_5ForConditionalGeneration; print('Qwen3.5 import OK')"
```

### 预训练权重

Smoke 默认（yaml 内）：

```text
./playground/Pretrained_models/Qwen3.5-0.8B/config.json
```

H200 / 正式 PI 路线推荐：

```text
./playground/Pretrained_models/Qwen3.5-4B/config.json
```

检查：

```bash
test -f playground/Pretrained_models/Qwen3.5-0.8B/config.json
# 或
test -f playground/Pretrained_models/Qwen3.5-4B/config.json
```

> PI **不需要** `*-Instruct-Action` 专用 checkpoint（无 action special token）；普通 Qwen3.5-VL Instruct 即可。

### 训练数据（LeRobot）

与 OFT 相同：LeRobot 格式 CALVIN，且目录下需有：

```text
<meta/info.json>
<meta/modality.json>    # 可从 examples/calvin/train_files/modality.json 复制
<data/>                 # parquet
```

Smoke yaml 默认：

```text
data_root_dir: playground/Datasets/calvin
data_mix: calvin_abc_d
```

按机器修改 yaml 或训练时用 CLI 覆盖 `datasets.vla_data.*`。

---

## 3. 模型在代码里如何接上

训练入口统一为 `starVLA/training/train_starvla.py`，由 yaml 中 `framework.name: QwenPI` 构建：

1. **VLM**：`get_vlm_model` 加载 `framework.qwenvl.base_vlm`
2. **多层特征**：前向时 `output_hidden_states=True`，取最后 `num_vl_layers` 层
3. **动作头**：`LayerwiseFlowmatchingActionHead`（`action_model_type: LayerwiseFM`）
4. **冻结**：默认 `trainer.freeze_modules: qwen_vl_interface`（只训 action head）

DiT 的 `num_layers` / `cross_attention_dim` 等在运行时由 `populate_layerwise_dit_cfg` 按 VLM 结构自动写入，**无需在 yaml 里手写层数**。

---

## 4. 本地 Smoke 训练

在仓库根目录执行。

### 4.1 1 step（验证能跑通）

```bash
cd /path/to/starVLA_Project

export WANDB_MODE=disabled
CUDA_VISIBLE_DEVICES=0 \
STAR_VLA_PYTHON="$(conda info --base)/envs/starVLA_qwen35/bin/python" \
bash examples/calvin/train_files/run_calvin_qwen35_pi_smoke.sh
```

成功时日志目录类似：

```text
logs/log_YYYYMMDD_HHMMSS_qwen35_pi_calvin_smoke/
  terminal/train.log
  checkpoints/qwen35_pi_calvin_smoke/checkpoints/steps_1_pytorch_model.pt
```

### 4.2 100 step 短训

```bash
CUDA_VISIBLE_DEVICES=0 \
MAX_TRAIN_STEPS=100 \
SAVE_INTERVAL=100 \
STAR_VLA_PYTHON="$(conda info --base)/envs/starVLA_qwen35/bin/python" \
bash examples/calvin/train_files/run_calvin_qwen35_pi_smoke.sh
```

查找 checkpoint：

```bash
find logs -path "*qwen35_pi_calvin_smoke/checkpoints/*/checkpoints/*_pytorch_model.pt" | sort
```

### 4.3 常用环境变量覆盖

| 变量 | 说明 |
|------|------|
| `BASE_VLM` | 覆盖 yaml 中的 VLM 路径（OOM 时可试 `Qwen3.5-4B` 或多卡） |
| `CONFIG_YAML` | 换用 `starvla_train_calvin_qwen35_pi_h200.yaml` 等 |
| `NUM_PROCESSES` | `accelerate` 进程数（多卡） |
| `MAX_TRAIN_STEPS` / `SAVE_INTERVAL` | 训练步数与保存间隔 |
| `PI_NUM_INFERENCE_TIMESTEPS` | 推理去噪步数（默认 4） |
| `PI_REPEATED_DIFFUSION_STEPS` | 训练侧 repeat（yaml 默认 2；`QwenPI.forward` 内另有固定逻辑） |
| `PI_NUM_TARGET_VISION_TOKENS` | `future_tokens` 数量（默认 32） |
| `LOG_DIR` / `RUN_ID` | 日志与 run 名称 |

示例（4B + 自定义 yaml）：

```bash
BASE_VLM=./playground/Pretrained_models/Qwen3.5-4B \
CONFIG_YAML=examples/calvin/train_files/starvla_train_calvin_qwen35_pi_h200.yaml \
CUDA_VISIBLE_DEVICES=0,1 \
NUM_PROCESSES=2 \
MAX_TRAIN_STEPS=1000 \
SAVE_INTERVAL=500 \
bash examples/calvin/train_files/run_calvin_qwen35_pi_smoke.sh
```

---

## 5. H200 流水线（P4）

不经过本 smoke 脚本时，可用仓库已有 H200 入口（训练 + D 环境 eval + mp4）：

```bash
# 见 readme_H200.md §4 / §5
export ROUTE=p4_qwen4b_pi   # 或 p4_pi
export P4_BASE_VLM=./playground/Pretrained_models/Qwen3.5-4B
bash examples/calvin/train_files/run_h200_fastexplore_server1.sh
```

`run_h200_fastexplore_route.sh` 在 `ROUTE=p4_*` 时会默认加载 `starvla_train_calvin_qwen35_pi_h200.yaml`，并通过 `run_route_validation_train.sh` 注入 `QwenPI` / `LayerwiseFM` 参数。

监控：

```bash
tail -f logs/h200_fastexplore/terminal/*p4_qwen4b_pi_pipeline.log
```

---

## 6. 训练后：Policy Server 与 Debug Eval

与 OFT 相同，换成本次 PI 的 checkpoint 路径即可。

**终端 1 — Policy Server（`starVLA_qwen35`）：**

```bash
export CKPT_PATH=logs/log_YYYYMMDD_HHMMSS_qwen35_pi_calvin_smoke/checkpoints/qwen35_pi_calvin_smoke/checkpoints/steps_100_pytorch_model.pt
export PORT=5694
export STAR_VLA_PYTHON="$(conda info --base)/envs/starVLA_qwen35/bin/python"

bash examples/calvin/eval_files/run_policy_server_debug.sh
```

**终端 2 — CALVIN eval（`calvin` 环境）：**

```bash
export CKPT_PATH=<同上>
export PORT=5694
export NUM_SEQUENCES=1
export UNNORM_KEY=franka

bash examples/calvin/eval_files/eval_calvin_debug.sh
```

Server 启动成功应看到：`action_chunk_size=8`、`available_unnorm_keys=['franka']`。

---

## 7. 常见问题

### CUDA OOM

- 减小 `datasets.vla_data.per_device_batch_size`（yaml 已为 1）
- 使用更小 VLM：`BASE_VLM=./playground/Pretrained_models/Qwen3.5-0.8B`
- 多卡：`NUM_PROCESSES=2` + `CUDA_VISIBLE_DEVICES=0,1`
- 减小 `obs_image_size`（如 `[112,112]`）
- PI 比 OFT 重很多，**8GB 单卡可能不够**；参考 H200 使用 4B + 多卡

### `STAR_VLA_PYTHON is not executable`

指定 conda 环境 Python：

```bash
export STAR_VLA_PYTHON="$(conda info --base)/envs/starVLA_qwen35/bin/python"
```

### 与 Physical Intelligence openpi 的关系

本仓库 **StarVLA-π** 是「Qwen VLM + 外挂 LayerwiseFM」，**不是** openpi 的 PaliGemma + Gemma-300M 双 expert，**不能**直接加载 openpi 的 `pi0_base` 权重。算法上同为 flow matching 族，实现与 checkpoint 均独立。

### 更强变体 `QwenPI_v3`

压缩 DiT + π₀.5 式离散 state 进 prompt，见 `starVLA/model/framework/VLM4A/QwenPI_v3.py` 与 HF：`StarVLA/Qwen3VL-PI_v3-Bridge-RT_1`。需将 yaml / CLI 中 `framework.name` 改为 `QwenPI_v3`（本 smoke 脚本默认 `QwenPI`）。

---

## 8. 与 OFT smoke 命令对照

```bash
# OFT
bash examples/calvin/train_files/run_calvin_qwen35_oft_smoke.sh

# PI（本文档）
bash examples/calvin/train_files/run_calvin_qwen35_pi_smoke.sh
```

其余数据路径、eval、policy server 流程一致，仅 checkpoint 目录名由 `qwen35_oft_calvin_smoke` 变为 `qwen35_pi_calvin_smoke`。
