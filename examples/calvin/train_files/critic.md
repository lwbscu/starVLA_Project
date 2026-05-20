# Calvin AWAC 后训练指南

本文档说明 StarVLA 在 Calvin 上的 **离线 AWAC（Advantage Weighted Actor-Critic）** 后训练流程：原理 → 数据预处理 → Critic 训练 → Actor 训练。

**前置条件**：已完成 BC 训练（`train_starvla.py`），得到 `QwenPI` checkpoint。

**跑实验 / 排错**：请先读 **[§2 核心不可改逻辑](#2-核心不可改逻辑debug-必读)**，再改 yaml、shell 或训练代码。

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

  - `R_chunk(t)`：数据预处理写入 parquet 的 `reward` 列（见第 3 节）。  
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

## 2. 核心不可改逻辑（Debug 必读）

本节写给**负责跑通 / 排错的同学**：下面约定是整条 AWAC 链路的契约。改路径、改步数、改学习率可以；**除非你愿意连同预处理、Critic、Actor 三处一起重推公式并重训，否则不要动这些逻辑。**

### 2.1 三处必须对齐的超参：`H`、`γ`、reward 语义

| 位置 | 字段 | 必须一致 |
|------|------|----------|
| `prepare_awac_rewards.py` | `--action_horizon`、`--gamma` | 基准 |
| `starvla_awac_calvin.yaml` | `framework.action_model.action_horizon`、`awac.gamma` | 与预处理相同 |
| Critic `td_loss` | `gamma ** action_horizon` | 自举项是 **γ^H**，不是 γ |

**不可改语义**：

- parquet 里 **`reward` 列 = 已算好的 chunk 折扣回报** \(R_{chunk}(t)\)（见 `awac_reward_preprocessing.compute_discounted_chunk_return`）。
- 训练时 `datasets.awac_data.reward_is_chunk_return: true`（默认）→ dataloader **只读** `reward.iloc[t]`，**禁止**再对 H 步求和（见 `read_transition_reward`）。
- 若把 `reward_is_chunk_return` 改成 `false` 而 parquet 仍是预处理后的 `reward`，会**重复折扣**，Q 目标错误。

**`done` 语义**（`compute_done_flags`）：

- 每个 episode **最后 H 帧** `done=True`，与「chunk 末端是否终止」对齐。
- Critic bootstrap：`(1 - done) * γ^H * Q_target(s', a')`；`done=1` 时不 bootstrap。

### 2.2 Transition 构造（`awac_transition_dataset.py`）

**不可改索引关系**（`AWACTransitionDataset.__getitem__`）：

| 符号 | 时刻 / 内容 |
|------|-------------|
| `s` | 帧 `t`：图像、语言、state |
| `a` | `[t, t+H)` 动作块 |
| `r` | `reward.iloc[t]`（已是 \(R_{chunk}(t)\)） |
| `s'` | 帧 `min(t+H, T-1)`；常规步为 `t+H`，末段 clamp 到最后一帧 |
| `a'` | 帧 `min(t+H, T-1)` 上的动作块（**离线数据**，不是 policy rollout） |
| `done` | `done.iloc[t]` |

**不可改过滤条件**：

```text
base_index + H - 1 < traj_len
```

保证 chunk 奖励/终止标记使用的窗口 `[t, t+H-1]` 完全落在 episode 内（含最后 H 帧的有效 `done`/`r`）。`s'`/`a'` 在 `t+H >= T` 时 clamp 到 `T-1` 做 bootstrap。日志里 `[AWAC] ... valid transitions` 比例骤降时，先查 `H` 是否与数据长度、预处理一致。

**相机顺序**（与 Critic 一致）：`modality_keys["video"]` 第 0 路 = 头部静态相机，第 1 路 = 腕部相机 → Critic `HEAD_CAMERA_INDEX=0`、`WRIST_CAMERA_INDEX=1`。

### 2.3 Critic 网络结构（`awac_q_critic.py`）

**不可改 token 拼接顺序**（`_build_tokens`）：

```text
[vision(2) | text(1) | action(H) | state(1) | query(E)]
```

- **Vision**：从 BC 的 Qwen 视觉塔 **deepcopy** 两路编码（Qwen3.5-VL 路径为 `qwen_vl_interface.model.model.visual`；Qwen2.5-VL 为 `model.visual`），**不是**整段 VLM forward 再 pool。找不到 visual 子模块时会 **直接报错**，不会静默退回 full-VLM。
- **Text**：`tokenize → embed_tokens → mean pool → (B,1,D)`，**不走** visual 塔。
- **Q 读出位置**：Transformer 输出序列**最后 E 个 token**（query 槽位）经 `q_head` → 每头一个 Q。
- **Twin Q**：`num_q_heads=E`；TD 与 Actor 优势一律用 **`min` over E**（`min_q` / `q_next.min`），不要改成 mean/max，除非整套算法重写。

**不可改 TD 目标**（`td_loss`）：

```text
y = r + (1 - done) * (γ^H) * min_E Q_target(s', a')
loss = MSE(Q_pred, y)   # 每个 head 同一 scalar target
```

- `a'` **必须**是 batch 的 `next_action`（行为策略），**不能**改成 `actor.predict_action(s')`。
- **没有 V 网络**；不要恢复 expectile / `value_net` / `V(s)` 优势。
- Target 网络：每步 **Polyak** `polyak_tau` 软更新（`soft_update_target`），不是周期性硬拷贝。

**Critic 阶段冻结契约**（`train_awac_critic.py`）：

- BC **actor 全程 `requires_grad=False`**，只用于加载 VLM、复制 visual 权重。
- Optimizer **只含 critic** 可训练参数；actor 不得进 optimizer。

### 2.4 Actor 阶段（`train_awac_actor.py` + `QwenPI_awac.py`）

**不可改对象分工**：

| 模块 | 是否训练 | 用途 |
|------|----------|------|
| `actor`（`QwenPI_AWAC`） | 是（受 `freeze_modules` 约束） | 加权 flow-matching；**保存的 checkpoint** |
| `actor_pi` | **否**（BC 加载后 `deepcopy` 一次，永不更新） | 仅 `predict_action` → \(a_\pi\) |
| `critic` | **否** | 仅 `min Q(s,·)` 算优势 |

**不可改优势与权重**（`compute_awac_weights`）：

```text
A = min_E Q(s, a_pi) - min_E Q(s, a_data)
w = clip(exp(A / λ), w_max)
```

- `a_pi`：**只能**来自 **`actor_pi`**，不能来自正在训练的 `actor`。
- `a_data`：batch 离线 `action`（行为 chunk）。
- **禁止**用 \(Q - V(s)\) 或单独 V 网络。

**不可改 Actor loss**（`QwenPI_awac.forward`）：

- 对 **数据里的 `a_data`** 做 flow-matching，得到 **逐样本** `loss_per_sample`。
- 再 `(loss_per_sample * w).mean()`；**不是**对 `a_pi` 做 BC loss。

**不可改入口名**：

- Actor 训练 **`framework.name` 必须是 `QwenPI_AWAC`**（shell / CLI 覆盖）；Critic 阶段用 `QwenPI`。

**Optimizer 契约**：

- `build_param_lr_groups` 只含 **可训练 actor** 参数；**不得**包含 `critic` 或 `actor_pi`（代码里会 `RuntimeError`）。

### 2.5 Checkpoint 与数据注册

| 阶段 | 文件名 | 内容 |
|------|--------|------|
| Critic | `steps_*_critic.pt` | `critic`、`critic_target`、`steps`、`awac`（**无** `value_net`） |
| Actor | `steps_*_pytorch_model.pt` | 与 BC 相同，仅 **可训练 actor** 权重 |

- 旧 ckpt 若含 `value_net` 键，Actor 加载会 **warning 并忽略**；应用新 Critic 重训。
- `datasets.awac_data.data_mix`（如 `calvin_abc_d_h200` / `calvin_hlx_h200`）只是在 **`data_config.py` 注册表**里查子目录名，不是路径本身；实际目录 = `{data_root_dir}/{d_name}`。H200 原始数据 `calvin_abc_d_h200 → calvin_task_ABC_D`，增强数据 `calvin_hlx_h200 → hlx/calvin_task_ABC_D`。

### 2.6 可以安全改的配置（不破坏契约时）

- `run_root_dir`、`run_id`、`bc_checkpoint`、`critic_checkpoint` 路径
- `critic_max_train_steps` / `actor_max_train_steps`、`save_interval`、batch size、学习率、`freeze_modules`
- `awac_lambda`、`awac_weight_max`（只影响 Actor 权重形状）
- 预处理里的 `step_penalty` / `failure_reward` 等 — **改后必须重跑 `prepare_awac_rewards.py`**

### 2.7 常见 Debug 对照表

| 现象 | 优先检查 |
|------|----------|
| `reward` 全 0 / warning | 未跑 `prepare_awac_rewards.py` 或列名不对 |
| `critic_loss` NaN / `target_mean` 爆炸 | `reward_is_chunk_return` 与 parquet 语义不一致；或 `γ`/`H` 与预处理不一致 |
| `valid transitions` 极少 | `H` 过大；或轨迹太短 |
| Actor 报 optimizer 含 frozen 参数 | critic / actor_pi 误入 optimizer（不应改训练脚本绕过，应查注册逻辑） |
| 优势恒为 0、`awac_weight_mean≈1` | critic 未加载或 Q 未区分 \(a_\pi\) / \(a_{data}\)；或 `actor_pi` 被误训练 |
| 加载 critic 报 shape 错 | Critic 与 BC 的 `action_dim`/`state_dim`/`H`/`hidden_dim` 与训练时不一致 |
| 数据找不到 | `data_root_dir` + `data_mix` 子目录名不匹配（应用 `calvin_abc_d_h200` → `calvin_task_ABC_D`） |

### 2.8 逻辑所在文件（改代码前先定位）

| 契约 | 文件 | 函数 / 类 |
|------|------|-----------|
| 逐步奖励 + chunk return + done | `starVLA/dataloader/awac_reward_preprocessing.py` | `compute_step_rewards`、`compute_discounted_chunk_return`、`compute_done_flags` |
| transition 索引 | `starVLA/dataloader/awac_transition_dataset.py` | `AWACTransitionDataset` |
| Q 结构 + TD | `starVLA/model/modules/critic/awac_q_critic.py` | `AWACQCritic._build_tokens`、`td_loss` |
| Critic 训练循环 | `starVLA/training/train_awac_critic.py` | `AWACCriticTrainer` |
| 优势 + actor_pi | `starVLA/training/train_awac_actor.py` | `build_frozen_actor_pi_snapshot`、`compute_awac_weights` |
| 加权 flow loss | `starVLA/model/framework/VLM4A/QwenPI_awac.py` | `Qwen_PI_AWAC.forward` |
| 数据 mix 注册 | `examples/calvin/train_files/data_registry/data_config.py` | `DATASET_NAMED_MIXTURES` |

---

## 3. 数据处理

### 3.1 原始数据要求

- **格式**：LeRobot（与 BC 相同），`meta/modality.json` 已配置。  
- **原始 parquet**：通常只有轨迹末帧的 **`success`**（成功/失败）；可无 `reward` / `done`。  
- **训练前**必须运行预处理脚本，写入 `step_reward`、`reward`、`done`。

### 3.2 预处理在做什么

| 步骤 | 函数 | 说明 |
|------|------|------|
| 逐步奖励 | `compute_step_rewards` | 非末帧：`-1`（鼓励尽快完成）；末帧成功：`0`；末帧失败：`-3000` |
| Chunk 折扣回报 | `compute_discounted_chunk_return` | \(R_{chunk}(t)=\sum_{k=0}^{\min(H-1,T-1-t)}\gamma^k r_{t+k}\) → 写入 **`reward`** |
| 终止标志 | `compute_done_flags` | 每个 episode **最后 H 帧** `done=True`（与 Critic bootstrap 对齐） |

实现代码：`starVLA/dataloader/awac_reward_preprocessing.py`。

### 3.3 训练时 Dataloader 做什么

`AWACTransitionDataset` **不再**重复算 reward/done，只：

- 组 transition：`(s, a, s', a', r, done)`  
  - `s` / `s'`：时刻 `t` / `t+H` 的图像、语言、state  
  - `a` / `a'`：长度 `H` 的动作块  
  - `r`：直接读 `reward.iloc[t]`（已是 \(R_{chunk}(t)\)）  
  - `done`：读 `done.iloc[t]`（无列时才用启发式）  
- 过滤：仅保留 `t + 2H ≤ traj_len` 的步（保证 `s'`、`a'` 真实存在）。

### 3.4 预处理脚本示例

```bash
# 在仓库根目录执行
python examples/calvin/scripts/prepare_awac_rewards.py \
  --dataset_root /path/to/calvin_task_ABC_D \
  --output_dataset_root /path/to/calvin_task_ABC_D_awac_work \
  --action_horizon 8 \
  --gamma 0.996
```

若确认输入目录已经是个人工作副本、允许被写入，可不传 `--output_dataset_root`，但必须显式加 `--allow_in_place`。不要对官方/比赛数据目录原地写 `success/reward/done`。

**仅扫描、不写盘**：

```bash
python examples/calvin/scripts/prepare_awac_rewards.py \
  --dataset_root /path/to/calvin_task_ABC_D \
  --action_horizon 8 \
  --gamma 0.996 \
  --dry_run
```

### 3.5 预处理参数怎么改

| 参数 | 默认 | 何时修改 |
|------|------|----------|
| `--dataset_root` | （必填） | LeRobot 数据集根目录（含 `data/**/*.parquet`） |
| `--output_dataset_root` | 空 | 推荐填写个人 AWAC 工作副本目录，避免修改源数据 |
| `--allow_in_place` | 关 | 仅当 `--dataset_root` 已是个人工作副本时开启 |
| `--action_horizon` | 8 | 与 yaml 中 `framework.action_model.action_horizon` **一致** |
| `--gamma` | 0.996 | 与 yaml 中 `awac.gamma` **一致** |
| `--step_penalty` | -1 | 调稀疏/稠密奖励尺度 |
| `--success_reward` | 0 | 成功末帧奖励 |
| `--failure_reward` | -3000 | 失败末帧惩罚 |
| `--success_column` | success | parquet 里成功标志列名 |
| `--assume_success_if_missing` | 关 | 无 success 列且全是成功 demo 时可打开 |

**注意**：改 `H` 或 `γ` 后需重新跑预处理；训练 yaml 里 `reward_is_chunk_return: true`（默认）表示 dataloader **不再对 H 步求和**。

### 3.6 磁盘不足时的只读数据模式

如果完整复制 public 数据集会超出个人目录配额，可以不写 parquet 工作副本，改为训练时即时计算 AWAC reward/done：

```bash
export calvin_data_root=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d
export data_mix=calvin_abc_d_h200
export compute_rewards_on_the_fly=true
export assume_success_if_missing=true
```

语义仍与第 2 节保持一致：

- `reward` 在 dataloader 内按 \(R_{chunk}(t)\) 计算。
- `done` 在 dataloader 内按最后 H 帧为 True 计算。
- 如果 parquet 缺 `success`，只有在确认全是成功 expert demo 时才允许 `assume_success_if_missing=true`。
- 该模式不修改 public 数据，也不创建完整 parquet 副本。

---

## 4. Critic 训练

### 4.1 脚本示例

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

### 4.2 输出与监控

| 输出 | 路径 |
|------|------|
| Critic checkpoint | `logs/<run_id>/checkpoints/steps_*_critic.pt` |
| TensorBoard | `logs/<run_id>/tensorboard/` |
| W&B | `logs/<run_id>/wandb/` |

```bash
tensorboard --logdir logs/awac_calvin_critic/tensorboard --port 6006
```

标量：`critic_loss`、`q_mean`、`target_mean`、`learning_rate`。

### 4.3 Shell 里常改的参数（`run_calvin_awac_critic.sh`）

| 变量 | 含义 |
|------|------|
| `CUDA_VISIBLE_DEVICES` | GPU |
| `base_vlm` | Qwen VLM 路径（与 BC 一致） |
| `calvin_data_root` | 数据集父目录 |
| `calvin_dataset_name` | 数据集子目录名，仅用于预处理注释/检查，如 `calvin_task_ABC_D` 或 `hlx` |
| `data_mix` | `data_registry` 里注册的 mix 名 |
| `include_state` / `state_dim` | H200 LeRobot PI-State 使用 `true` / `8` |
| `bc_checkpoint` | **BC actor** 权重（用于加载 VLM、复制 visual 到 Critic） |
| `run_root_dir` / `run_id` | 日志与 checkpoint 目录 |
| `num_processes` | GPU 数量 |

### 4.4 YAML / CLI 里常改的参数（`starvla_awac_calvin.yaml`）

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
| `action_dim` / `state_dim` | 7 / 7 | YAML 默认；H200 LeRobot PI-State 必须用脚本环境变量或 CLI 覆盖 `state_dim=8` |

CLI 覆盖示例：

```bash
--trainer.critic_max_train_steps 100000 \
--awac.hidden_dim 768 \
--datasets.awac_data.per_device_batch_size 2 \
--trainer.use_tensorboard true
```

---

## 5. Actor 训练

### 5.1 脚本示例

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

### 5.2 输出

| 输出 | 路径 |
|------|------|
| Actor checkpoint | `logs/<run_id>/checkpoints/steps_*_pytorch_model.pt` |

格式与 BC 相同，可直接用于 Calvin 评测脚本。

### 5.3 Shell 里常改的参数（`run_calvin_awac_actor.sh`）

| 变量 | 含义 |
|------|------|
| `bc_checkpoint` | 初始化 **可训练 actor**（与 BC 相同权重） |
| `critic_checkpoint` | Phase 1 的 `*_critic.pt`（**必填**） |
| （自动） | 加载 BC 后对 actor 做 `deepcopy` 得到 **冻结 actor_pi**，仅用于算 \(a_\pi\)，不写入 checkpoint |
| `run_id` | 建议与 critic 阶段区分，如 `awac_calvin_actor` |
| `--trainer.freeze_modules` | 默认 `qwen_vl_interface`，只训 action head |

其余 `calvin_data_root`、`data_mix`、`include_state`、`state_dim`、`base_vlm` 与 Critic 阶段一致。

### 5.4 YAML / CLI 里常改的参数

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

## 6. 推荐执行顺序

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

## 7. 相关文件索引

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
