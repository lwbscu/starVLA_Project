# Calvin AWAC 后训练指南

本文档说明 StarVLA 在 Calvin 上的 **离线 AWAC（Advantage Weighted Actor-Critic）** 后训练流程：原理 → 数据预处理 → Critic 训练 → Actor 训练。

**前置条件**：已完成 BC 训练（`train_starvla.py`），得到 `QwenPI` checkpoint。

---

## 1. 原理简述

### 1.1 整体流程

两阶段、顺序执行（不交替）：

1. **Critic 阶段**：用离线轨迹训练 Q 网络（TD loss）；**BC actor 全程冻结**，仅作 VLM/visual 初始化来源。  
2. **Actor 阶段**：**Q critic 全程冻结**；维护 **可训练 actor** + **冻结 actor_pi 快照**（BC 加载后 deepcopy，仅用于 `predict_action`）；\(A=Q(s,a_\pi)-Q(s,a_{\text{data}})\) 后对可训练 actor 做加权 flow-matching。

Actor 主干仍为 `QwenPI`（VLM + Layerwise flow-matching action head）；Critic 单独一套网络，视觉塔从 BC actor **复制初始化**。

### 1.2 Critic：Q 网络

- **输入**：当前观测 `s`（多相机图像 + 语言 + 可选 proprio）、动作块 `a`（长度 `H` 的连续动作）。  
- **结构**：复制 actor 的 **visual 塔**（头部相机 view0、腕部相机 view1 各一路）→ `(B,2,D)`；语言 **tokenize + embed** 后 mean pool → `(B,1,D)`；state/action 线性投影 → `(B,1,D)` / `(B,H,D)`；可学习 query → `(B,E,D)`；拼接后过 6 层 Transformer，**query 位置** 输出 Q。  
  Token 顺序：`[vision(2) | text(1) | action(H) | state(1) | query(E)]`。  
- **TD 目标**（chunk 时间自举）：

  \[
  y = R_{chunk}(t) + (1 - \text{done}) \cdot \gamma^{H} \cdot \min_{e} Q_{\text{target}}(s', a')
  \]

  - `R_chunk(t)`：数据预处理写入 parquet 的 `reward` 列（见第 2 节）。  
  - `γ`：默认 `0.996`；`H`：`action_horizon`（Calvin 默认 8）。  
  - `a'`：数据集行为动作（下一 chunk）。  
  - Target 网络：Polyak 软更新（`polyak_tau=0.005`）。

### 1.3 Actor：Q 对比优势加权（仅 Q，无 V）

Actor 阶段在训练时在线计算（**critic 与 actor_pi 均冻结**）：

\[
a_\pi = \pi_{\text{frozen}}(s),\quad
A = \min_e Q(s, a_\pi) - \min_e Q(s, a_{\text{data}}),\quad
w = \mathrm{clip}\Big(\exp\big(\frac{A}{\lambda}\big),\; w_{\max}\Big)
\]

- \(\pi_{\text{frozen}}\)：`actor_pi` 快照（BC 权重 deepcopy，Actor 阶段不更新），仅用于算 \(a_\pi\)。  
- \(\pi_{\text{train}}\)：可训练 actor，用于加权 flow-matching loss 并保存 checkpoint。  
- \(a_{\text{data}}\)：离线数据里的行为动作 chunk。  
- 默认 `λ=0.5`，`w_max=20`。

---

## 2. 数据处理

### 2.1 原始数据要求

- **格式**：LeRobot（与 BC 相同），`meta/modality.json` 已配置。  
- **原始 parquet**：通常只有轨迹末帧的 **`success`**（成功/失败）；可无 `reward` / `done`。  
- **训练前**必须运行预处理脚本，写入 `step_reward`、`reward`、`done`。

### 2.2 预处理在做什么

| 步骤 | 函数 | 说明 |
|------|------|------|
| 逐步奖励 | `compute_step_rewards` | 非末帧：`-1`（鼓励尽快完成）；末帧成功：`0`；末帧失败：`-3000` |
| Chunk 折扣回报 | `compute_discounted_chunk_return` | \(R_{chunk}(t)=\sum_{k=0}^{\min(H-1,T-1-t)}\gamma^k r_{t+k}\) → 写入 **`reward`** |
| 终止标志 | `compute_done_flags` | 每个 episode **最后 H 帧** `done=True`（与 Critic bootstrap 对齐） |

实现代码：`starVLA/dataloader/awac_reward_preprocessing.py`。

### 2.3 训练时 Dataloader 做什么

`AWACTransitionDataset` **不再**重复算 reward/done，只：

- 组 transition：`(s, a, s', a', r, done)`  
  - `s` / `s'`：时刻 `t` / `t+H` 的图像、语言、state  
  - `a` / `a'`：长度 `H` 的动作块  
  - `r`：直接读 `reward.iloc[t]`（已是 \(R_{chunk}(t)\)）  
  - `done`：读 `done.iloc[t]`（无列时才用启发式）  
- 过滤：仅保留 `t + 2H ≤ traj_len` 的步（保证 `s'`、`a'` 真实存在）。

### 2.4 预处理脚本示例

```bash
# 在仓库根目录执行
python examples/calvin/scripts/prepare_awac_rewards.py \
  --dataset_root /path/to/calvin_task_ABC_D \
  --action_horizon 8 \
  --gamma 0.996
```

**仅扫描、不写盘**：

```bash
python examples/calvin/scripts/prepare_awac_rewards.py \
  --dataset_root /path/to/calvin_task_ABC_D \
  --action_horizon 8 \
  --gamma 0.996 \
  --dry_run
```

### 2.5 预处理参数怎么改

| 参数 | 默认 | 何时修改 |
|------|------|----------|
| `--dataset_root` | （必填） | LeRobot 数据集根目录（含 `data/**/*.parquet`） |
| `--action_horizon` | 8 | 与 yaml 中 `framework.action_model.action_horizon` **一致** |
| `--gamma` | 0.996 | 与 yaml 中 `awac.gamma` **一致** |
| `--step_penalty` | -1 | 调稀疏/稠密奖励尺度 |
| `--success_reward` | 0 | 成功末帧奖励 |
| `--failure_reward` | -3000 | 失败末帧惩罚 |
| `--success_column` | success | parquet 里成功标志列名 |
| `--assume_success_if_missing` | 关 | 无 success 列且全是成功 demo 时可打开 |

**注意**：改 `H` 或 `γ` 后需重新跑预处理；训练 yaml 里 `reward_is_chunk_return: true`（默认）表示 dataloader **不再对 H 步求和**。

---

## 3. Critic 训练

### 3.1 脚本示例

```bash
# 仓库根目录
bash examples/calvin/train_files/run_calvin_awac_critic.sh
```

或直接：

```bash
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_awac_critic.py \
  --config_yaml examples/calvin/train_files/starvla_awac_calvin.yaml \
  --trainer.pretrained_checkpoint /path/to/bc/steps_30000_pytorch_model.pt \
  --datasets.awac_data.data_root_dir /path/to/calvin_abc_d \
  --datasets.awac_data.data_mix calvin_abc_d_h200 \
  --run_id awac_calvin_critic
```

### 3.2 输出与监控

| 输出 | 路径 |
|------|------|
| Critic checkpoint | `logs/<run_id>/checkpoints/steps_*_critic.pt` |
| TensorBoard | `logs/<run_id>/tensorboard/` |
| W&B | `logs/<run_id>/wandb/` |

```bash
tensorboard --logdir logs/awac_calvin_critic/tensorboard --port 6006
```

标量：`critic_loss`、`q_mean`、`target_mean`、`learning_rate`。

### 3.3 Shell 里常改的参数（`run_calvin_awac_critic.sh`）

| 变量 | 含义 |
|------|------|
| `CUDA_VISIBLE_DEVICES` | GPU |
| `base_vlm` | Qwen VLM 路径（与 BC 一致） |
| `calvin_data_root` | 数据集父目录 |
| `data_mix` | `data_registry` 里注册的 mix 名 |
| `bc_checkpoint` | **BC actor** 权重（用于加载 VLM、复制 visual 到 Critic） |
| `run_root_dir` / `run_id` | 日志与 checkpoint 目录 |
| `--num_processes` | GPU 数量 |

### 3.4 YAML / CLI 里常改的参数（`starvla_awac_calvin.yaml`）

**数据 `datasets.awac_data`**

| 键 | 默认 | 说明 |
|----|------|------|
| `data_root_dir` | — | 数据根目录 |
| `data_mix` | calvin_abc_d_h200 | mixture 名 |
| `per_device_batch_size` | 4 | 显存不足则减小 |
| `include_state` | true | Critic 是否用 proprio |
| `reward_is_chunk_return` | true | 预处理后保持 true |

**AWAC `awac`**

| 键 | 默认 | 说明 |
|----|------|------|
| `gamma` | 0.996 | 与预处理一致 |
| `num_q_heads` | 2 | Q 头数 E |
| `hidden_dim` | 512 | Critic token 维度 |
| `transformer_layers` | 6 | Transformer 层数 |
| `awac_lambda` | 0.5 | Actor 权重温度 λ |
| `awac_weight_max` | 20 | Actor 权重上限 |
| `polyak_tau` | 0.005 | Target 软更新 |
| `freeze_visual` | true | 是否冻结复制的 visual |

**训练 `trainer`（Critic 专用）**

| 键 | 默认 | 说明 |
|----|------|------|
| `pretrained_checkpoint` | null | BC checkpoint（建议 shell 传入） |
| `critic_max_train_steps` | 50000 | 训练步数 |
| `learning_rate.critic` | 3e-4 | Critic 学习率 |
| `num_warmup_steps` | 1000 | 预热 |
| `save_interval` | 5000 | 存盘间隔 |
| `logging_frequency` | 50 | 日志间隔 |
| `use_tensorboard` | true | 关闭则 `false` |
| `max_grad_norm` | 1.0 | 梯度裁剪 |

**框架 `framework.action_model`**

| 键 | 默认 | 说明 |
|----|------|------|
| `action_horizon` | 8 | **H**，与预处理一致 |
| `action_dim` / `state_dim` | 7 | Calvin 关节+gripper |

CLI 覆盖示例：

```bash
--trainer.critic_max_train_steps 100000 \
--awac.hidden_dim 768 \
--datasets.awac_data.per_device_batch_size 2 \
--trainer.use_tensorboard true
```

---

## 4. Actor 训练

### 4.1 脚本示例

```bash
bash examples/calvin/train_files/run_calvin_awac_actor.sh
```

或直接：

```bash
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_awac_actor.py \
  --config_yaml examples/calvin/train_files/starvla_awac_calvin.yaml \
  --framework.name QwenPI_AWAC \
  --trainer.pretrained_checkpoint /path/to/bc/steps_30000_pytorch_model.pt \
  --trainer.awac_critic_checkpoint /path/to/steps_50000_critic.pt \
  --datasets.awac_data.data_root_dir /path/to/calvin_abc_d \
  --run_id awac_calvin_actor
```

### 4.2 输出

| 输出 | 路径 |
|------|------|
| Actor checkpoint | `logs/<run_id>/checkpoints/steps_*_pytorch_model.pt` |

格式与 BC 相同，可直接用于 Calvin 评测脚本。

### 4.3 Shell 里常改的参数（`run_calvin_awac_actor.sh`）

| 变量 | 含义 |
|------|------|
| `bc_checkpoint` | 初始化 **可训练 actor**（与 BC 相同权重） |
| `critic_checkpoint` | Phase 1 的 `*_critic.pt`（**必填**） |
| （自动） | 加载 BC 后对 actor 做 `deepcopy` 得到 **冻结 actor_pi**，仅用于算 \(a_\pi\)，不写入 checkpoint |
| `run_id` | 建议与 critic 阶段区分，如 `awac_calvin_actor` |
| `--trainer.freeze_modules` | 默认 `qwen_vl_interface`，只训 action head |

其余 `calvin_data_root`、`data_mix`、`base_vlm` 与 Critic 阶段一致。

### 4.4 YAML / CLI 里常改的参数

**Actor 专用 `trainer`**

| 键 | 默认 | 说明 |
|----|------|------|
| `awac_critic_checkpoint` | null | Critic 权重（shell 传入） |
| `actor_max_train_steps` | 30000 | Actor 训练步数 |
| `learning_rate.base` | 1e-4 | 与 BC 同量级 |
| `learning_rate.action_model` | 1e-4 | action head 学习率 |
| `freeze_modules` | qwen_vl_interface | 解冻 VLM：`''` 或改列表 |

**AWAC 权重（仅 Actor 阶段）**

| 键 | 默认 | 说明 |
|----|------|------|
| `awac_lambda` | 0.5 | 越小权重分布越尖 |
| `awac_weight_max` | 20.0 | 防止单样本权重爆炸 |

**框架**

| 键 | 说明 |
|----|------|
| `framework.name` | 必须为 **`QwenPI_AWAC`** |

CLI 示例：

```bash
--trainer.actor_max_train_steps 50000 \
--awac.awac_lambda 0.3 \
--awac.awac_weight_max 10.0 \
--trainer.freeze_modules ""
```

---

## 5. 推荐执行顺序

```text
BC 训练 (train_starvla.py)
    ↓
prepare_awac_rewards.py   # 写 reward / done
    ↓
train_awac_critic.py      # 得 *_critic.pt
    ↓
train_awac_actor.py       # 得 *_pytorch_model.pt
    ↓
Calvin eval
```

## 6. 相关文件索引

| 文件 | 作用 |
|------|------|
| `examples/calvin/scripts/prepare_awac_rewards.py` | 数据预处理入口 |
| `starVLA/dataloader/awac_reward_preprocessing.py` | 奖励/done 计算逻辑 |
| `starVLA/dataloader/awac_transition_dataset.py` | 训练 dataloader |
| `starVLA/training/train_awac_critic.py` | Critic 训练 |
| `starVLA/training/train_awac_actor.py` | Actor 训练 |
| `examples/calvin/train_files/starvla_awac_calvin.yaml` | 统一配置 |
| `examples/calvin/train_files/run_calvin_awac_critic.sh` | Critic 启动脚本 |
| `examples/calvin/train_files/run_calvin_awac_actor.sh` | Actor 启动脚本 |
