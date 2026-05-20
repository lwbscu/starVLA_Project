# AWAC 后训练配置过程与经验

本文档用于交接当前 Calvin AWAC 后训练搭建过程。重点不是罗列所有历史方案，而是说明**当前为什么这样跑**、**哪些地方不能动**、**接手后先看什么**。

**H200 上实际跑混训时，只认两个 oneclick（与 `01_H200_AWAC后训练运行命令.md` 一致）：**

- `examples/calvin/train_files/h200_awac_critic_mixed_oneclick.sh`
- `examples/calvin/train_files/h200_awac_actor_mixed_oneclick.sh`

下文 §3–§6 中「单源 public + on-the-fly」是早期磁盘受限方案的背景说明；**当前混训主线**是 expert + rollout **1:1**、`data_mix=calvin_awac_mixed_h200`。

## 1. 背景与目标

当前任务是把已经训练好的 PI-State BC baseline 接入 AWAC 后训练。BC 根目录：

```text
/inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project/logs/20260519_h200_server1_pi_state_30k_v1
```

主力路线固定为：

- 模型：`Qwen3.5-4B`
- BC checkpoint：`server1_pi_state/qwen35_4b/.../steps_30000_pytorch_model.pt`
- 数据：public `calvin_abc_d/calvin_task_ABC_D`
- 数据 mix：`calvin_abc_d_h200`
- 输入：双视角图像 + 语言 + 8 维 state
- 动作：7 维 action，`action_horizon=8`
- AWAC：先 critic，后 actor，不交替训练

必须严格对齐 `examples/calvin/train_files/critic.md` 的核心契约：

- `H=8`
- `gamma=0.996`
- `reward` 语义是 chunk return
- `done` 语义是每个 episode 最后 H 帧为 True
- critic 的 `s'` 是 `t+H`，`a'` 是离线数据 `[t+H, t+2H)`，不是 policy rollout
- actor 阶段用冻结 `actor_pi` 计算 `a_pi`，critic 冻结

## 2. 数据事实

`calvin_task_ABC_D` 已按 LeRobot 格式组织：

- `meta/info.json`
- `meta/modality.json`
- `data/**/*.parquet`

readiness 抽样已确认：

- state 维度：8
- action 维度：7
- 视觉：双视角 `image`、`wrist_image`
- 语言/任务字段：`task_index`
- sampled parquet 已有真实 `success`
- sampled parquet 可以没有 `step_reward/reward/done`

因此当前不需要“全成功假设”，也不需要补写 parquet。训练时固定：

```bash
export compute_rewards_on_the_fly=true
export assume_success_if_missing=false
```

这表示：

- `success` 从 parquet 真实读取。
- `reward` 在 dataloader 内按 `R_chunk(t)` 即时计算。
- `done` 在 dataloader 内按最后 H 帧即时计算。
- public parquet 不写 `success/reward/done`。

## 3. 关键设计选择

### 不复制完整 AWAC 工作副本

完整 public 数据集太大，个人目录放不下。最初考虑过复制一份并写入 `step_reward/reward/done`，但这条路线不适合当前磁盘条件。

最终选择是 dataloader 即时计算 reward/done：

- 节省空间。
- 不改 public parquet。
- 保持 `critic.md` 的 reward/done/H/gamma 契约。
- 代价是 dataloader 每次取 transition 时多做少量计算。

### 不使用全成功兜底

虽然训练数据是 expert demo，但当前 sampled parquet 已有真实 `success` 列，所以不应使用：

```bash
export assume_success_if_missing=true
```

当前固定为：

```bash
export assume_success_if_missing=false
```

这样如果后续换数据而缺 `success`，训练会硬报错，不会悄悄伪造成功标签。

### 优先 4B

已有 rollout 观察显示：

- 0.8B 成功较少
- 9B 当前只有 25k checkpoint，且 rollout 表现较差
- 4B 成功更多，作为 AWAC 后训练主力更合理

所以当前交接只要求接手同学把 4B critic/actor 跑通。

## 4. 已踩坑与注意事项

### conda 没激活会导致 `accelerate: command not found`

H200 shell 默认环境很裸。必须先：

```bash
export CONDA_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/26220216/miniconda3
export PATH="${CONDA_ROOT}/bin:$PATH"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate starVLA_qwen35
```

验证：

```bash
python -c "import sys; print(sys.executable)"
python -m accelerate.commands.launch --help >/dev/null && echo "accelerate_module=ok"
```

### H200 可能没有 `git`

现场遇到过：

```text
git: command not found
```

所以不要假设服务器上能 `git pull`。如果需要同步代码，要先确认环境里有 git，或者通过平台文件同步/手动补丁处理。

### rollout parquet 写入依赖 `pyarrow`

`calvin` env 里可能缺 `pyarrow`，导致 rollout writer 的 `pandas.to_parquet` 失败。解决方式是让 parquet 写入使用 `starVLA_qwen35` 环境：

```bash
export ROLLOUT_PARQUET_PYTHON="${STAR_VLA_PYTHON}"
```

当前 AWAC 主线不依赖 rollout 数据作为训练源，但后续做 rollout 诊断时要记住这个点。

### public 目录 cache 写入隐患

训练启动时出现过：

```text
Cached steps config changed; rebuilding .../public/.../meta/steps_data_index.pkl
```

这不是修改 parquet 内容，也不是写 `success/reward/done` 标签；它是 LeRobot dataloader 的 step index cache。但如果后续要求 public 目录完全零写入，需要另起任务把这个 cache 路径重定向到个人目录，或预先准备个人 cache。

当前不要为了这个中断正在跑的 critic。

## 5. 当前训练状态

4B AWAC critic 已成功启动并进入训练。关键启动日志：

```text
[AWAC] calvin_task_ABC_D: 803693/1071743 valid transitions (H=8)
[AWAC] calvin_task_ABC_D: computing reward/done on the fly (H=8, gamma=0.996, success_column=success, assume_success_if_missing=False)
AWAC critic phase: actor frozen; training Q network only.
```

初始训练状态健康：

```text
54/50000
critic_loss=0.742
q_mean=-10
target_mean=-10.8
```

这说明：

- 数据链路已经进到训练。
- reward/done 即时计算已生效。
- loss/q/target 是有限数，没有一开始就 nan/inf。

## 6. 接手后的检查清单

### 先确认 critic 是否仍在跑

```bash
ps -ef | grep -E 'train_awac_critic.py|accelerate.commands.launch' | grep -v grep || true
```

### 再确认 checkpoint

```bash
export PROJECT_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project
export CRITIC_RUN="${PROJECT_ROOT}/logs/20260520_awac_pi_state_onfly/awac_critic_pi_state_qwen35_4b_bc30k_onfly"

find "${CRITIC_RUN}/checkpoints" -maxdepth 1 -type f -name 'steps_*_critic.pt' -print | sort || true
```

第一目标是看到：

```text
steps_5000_critic.pt
```

混训 critic（oneclick）目标是：

```text
steps_1000_critic.pt ... steps_10000_critic.pt
```

单源备选才可能以 `steps_50000_critic.pt` 为终点（`calvin_abc_d_h200`，非混训）。

### 如果 loss 出问题

不要先改网络结构。按顺序查：

1. readiness 是否仍 PASS。
2. `compute_rewards_on_the_fly=true` 是否传入。
3. `assume_success_if_missing=false` 是否传入。
4. 日志是否显示 `H=8, gamma=0.996`。
5. `critic_loss/q_mean/target_mean` 是否从某一步开始发散。

不要改：

- `H`
- `gamma`
- chunk return 语义
- last-H done 语义
- `a'` 的离线数据索引关系
- twin Q 的 `min` 逻辑

### Critic 完成后再 actor

Actor 必须等 critic checkpoint 完成后再开。不要 critic/actor 同时乱开。

actor 输入：

- 同一个 4B BC checkpoint
- 最新 critic checkpoint
- 同一份 public CALVIN 数据
- 同样 `compute_rewards_on_the_fly=true`
- 同样 `assume_success_if_missing=false`

actor 输出 `steps_*_pytorch_model.pt` 后，再做 CALVIN eval / rollout 成功率评估。

## 7. 当前口径总结

交接给下一位同学时，直接说：

```text
H200 混训后训练：h200_awac_critic_mixed_oneclick.sh -> h200_awac_actor_mixed_oneclick.sh。
BC：Qwen3.5-4B PI-State steps_30000；数据：calvin_awac_mixed_h200（expert + rollout 1:1）。
reward/done：compute_rewards_on_the_fly=true；assume_success_if_missing=false。
Critic：10k 步，save 每 1k；Actor：30k 步，save 每 5k，仅训 action head。
不写 public parquet。9B 只有 25k，不作混训主线。
```
