# 当前 CALVIN 项目：预训练 -> rollout -> critic -> 评估

一句话先捋顺：

```text
预训练/BC：看老师数据学动作
rollout：让当前模型进 CALVIN 环境自己跑，留下失败/成功轨迹
critic：用这些轨迹训练一个“动作好坏裁判”
评估：不是看 loss，而是看 CALVIN 环境里连续完成几个任务，以及 mp4 里到底动得对不对
```

## 0. 当前真实状态

当前本地已经跑通的是链路，不是效果达标：

- OFT 100 step：训练、policy server、3 条 rollout、mp4、AWAC reward 预处理、critic 1 step smoke 已通。
- PI CPU offload：计划 800 step，训练进度跑满但最终保存阶段被 SIGKILL，所以只认已经落盘的 `steps_600_pytorch_model.pt`；用它跑了 3 条 rollout 和 120 step critic。
- 这两次 3 条 debug eval 的 `avg_seq_len` 都是 `0.0`，说明策略还没学好。这个结果不能包装成有效策略，只能说工程闭环和后训练数据链路可用。

代表性产物：

```text
logs/log_20260519_1548_qwen35_oft_rollout_debug_100step/
  checkpoints/qwen35_oft_rollout_debug_100step/checkpoints/steps_100_pytorch_model.pt
  mp4_3seq/results.json
  rollout_lerobot_debug_3seq/
  awac_critic_smoke_1step/checkpoints/steps_1_critic.pt

logs/log_20260519_1640_qwen35_pi_rollout_debug_800step_cpuoffload/
  checkpoints/qwen35_pi_rollout_debug_800step_cpuoffload/checkpoints/steps_600_pytorch_model.pt
  mp4_3seq/results.json
  rollout_lerobot_debug_3seq/
  awac_critic_pi_120step_state7_trimmed/checkpoints/steps_120_critic.pt
```

## 1. 预训练/BC 是什么

这里的“预训练”更准确说是行为克隆 BC/SFT：给模型看 CALVIN LeRobot 数据里的两路图像、语言指令和老师动作，让它学会输出未来 `8` 步、每步 `7` 维的机器人动作。

动作大概是：

```text
[x, y, z, roll, pitch, yaw, gripper]
```

当前主入口：

| 用途 | 文件/位置 |
| --- | --- |
| 本地 Qwen3.5 OFT smoke 训练命令 | `README_zh.md` 第 10 节；脚本 `examples/calvin/train_files/run_calvin_qwen35_oft_smoke.sh` |
| 四路线统一训练入口 | `examples/calvin/train_files/run_route_validation_train.sh` |
| H200 四台机器批量命令 | `H200_批量训练命令行.md` |
| OFT 配置 | `examples/calvin/train_files/starvla_train_calvin_qwen35_oft_smoke.yaml`、`starvla_train_calvin_qwen35_oft_h200.yaml` |
| PI 配置 | `examples/calvin/train_files/starvla_train_calvin_qwen35_pi_h200.yaml` |
| 真正训练循环 | `starVLA/training/train_starvla.py` |
| OFT 模型 | `starVLA/model/framework/VLM4A/QwenOFT.py` |

本地 100 step OFT 示例：

```bash
cd /home/lwb/Projects/SII/starVLA_Projects/starVLA_Project

CUDA_VISIBLE_DEVICES=0 \
MAX_TRAIN_STEPS=100 \
SAVE_INTERVAL=100 \
STAR_VLA_PYTHON="$(conda info --base)/envs/starVLA_qwen35/bin/python" \
bash examples/calvin/train_files/run_calvin_qwen35_oft_smoke.sh
```

四路线统一入口示例：

```bash
cd /home/lwb/Projects/SII/starVLA_Projects/starVLA_Project

ROUTE=p0_oft \
MAX_TRAIN_STEPS=100 \
SAVE_INTERVAL=100 \
CUDA_VISIBLE_DEVICES=0 \
bash examples/calvin/train_files/run_route_validation_train.sh
```

可选路线：

```text
p0_oft           QwenOFT
p1_adapter       QwenAdapter
p2_lora_oft      LoRA + QwenOFT
p3_lora_adapter  LoRA + QwenAdapter
p4_pi            QwenPI
```

H200 正式跑法看 `H200_批量训练命令行.md`：那里把 0.8B、4B、9B 三个模型规格、四台 server、端口、checkpoint 保留策略都列好了。

## 2. Rollout 是什么

rollout 就是“把训练出的 checkpoint 放进真实 CALVIN 仿真里跑”。它不是训练 loss，而是闭环执行：

```text
checkpoint -> policy server -> websocket client -> CALVIN env.step()
          -> task oracle 判断成功/失败 -> results.json / mp4 / rollout dataset
```

普通 eval 只保存指标和 mp4；打开 rollout writer 后，还会把每一步写成 LeRobot 风格数据，后面给 critic/AWAC 用。

核心文件：

| 用途 | 文件/位置 |
| --- | --- |
| 启动 policy server | `examples/calvin/eval_files/run_policy_server_debug.sh` |
| CALVIN eval client | `examples/calvin/eval_files/eval_calvin_debug.sh` |
| rollout 主逻辑 | `examples/calvin/eval_files/eval_calvin.py` 的 `rollout()` |
| LeRobot rollout 写盘 | `examples/calvin/eval_files/rollout_lerobot_writer.py` |
| server 加载 checkpoint | `deployment/model_server/server_policy.py`、`deployment/model_server/policy_wrapper.py` |

先开 policy server：

```bash
cd /home/lwb/Projects/SII/starVLA_Projects/starVLA_Project

export CKPT_PATH=logs/log_20260519_1640_qwen35_pi_rollout_debug_800step_cpuoffload/checkpoints/qwen35_pi_rollout_debug_800step_cpuoffload/checkpoints/steps_600_pytorch_model.pt
export PORT=5694
export CUDA_VISIBLE_DEVICES=0
export STAR_VLA_PYTHON="$(conda info --base)/envs/starVLA_qwen35/bin/python"

bash examples/calvin/eval_files/run_policy_server_debug.sh
```

另开终端跑普通 debug eval：

```bash
cd /home/lwb/Projects/SII/starVLA_Projects/starVLA_Project

export CKPT_PATH=logs/log_20260519_1640_qwen35_pi_rollout_debug_800step_cpuoffload/checkpoints/qwen35_pi_rollout_debug_800step_cpuoffload/checkpoints/steps_600_pytorch_model.pt
export PORT=5694
export NUM_SEQUENCES=3
export UNNORM_KEY=franka
export RUN_ID=qwen35_pi_debug3

bash examples/calvin/eval_files/eval_calvin_debug.sh
```

如果要产出后训练用 rollout LeRobot 数据，当前更直接的是绕过 shell，直接调 `eval_calvin.py`：

```bash
cd /home/lwb/Projects/SII/starVLA_Projects/starVLA_Project

export LOG_DIR=logs/log_$(date +"%Y%m%d_%H%M%S")_rollout_collect_debug3
export CKPT_PATH=logs/log_20260519_1640_qwen35_pi_rollout_debug_800step_cpuoffload/checkpoints/qwen35_pi_rollout_debug_800step_cpuoffload/checkpoints/steps_600_pytorch_model.pt
export CALVIN_PYTHON="$(conda info --base)/envs/calvin/bin/python"

"${CALVIN_PYTHON}" examples/calvin/eval_files/eval_calvin.py \
  --args.pretrained-path "${CKPT_PATH}" \
  --args.unnorm-key franka \
  --args.host 127.0.0.1 \
  --args.port 5694 \
  --args.dataset-path /home/lwb/Projects/SII/starVLA_Projects/calvin/dataset/calvin_debug_dataset \
  --args.calvin-config-path /home/lwb/Projects/SII/starVLA_Projects/calvin/calvin_models/conf \
  --args.eval-sequences-path examples/calvin/eval_files/eval_sequences.json \
  --args.num-sequences 3 \
  --args.eval-log-dir "${LOG_DIR}/mp4_3seq" \
  --args.debug \
  --args.rollout-lerobot-dir "${LOG_DIR}/rollout_lerobot_debug_3seq" \
  --args.rollout-lerobot-write-videos
```

rollout 数据里会有：

```text
meta/info.json
meta/modality.json
meta/tasks.jsonl
meta/episodes.jsonl
data/chunk-000/episode_000000.parquet
videos/chunk-000/image/*.mp4
videos/chunk-000/wrist_image/*.mp4
rollout_analysis/rollout_steps.jsonl
```

关键列包括 `state`、`actions`、`done`、`env_done`、`episode_success`、`robot_obs`、`robot_joint_pos`、`distance_to_target`。

## 3. Critic 是什么

critic 是后训练里的“裁判”，不是直接控制机器人的 policy。

它学的是：

```text
给定当前图像/语言/state 和一段动作 action chunk，
这段动作之后更可能成功还是失败？
```

当前实现是 AWAC 风格：

- `Q(s,a)`：估计这段动作有多好。
- `V(s)`：估计当前状态本身的价值。
- 后续 actor fine-tune 时，用 `exp((Q - V) / lambda)` 给好动作更大权重。

核心文件：

| 用途 | 文件/位置 |
| --- | --- |
| AWAC 配置 | `examples/calvin/train_files/starvla_awac_calvin.yaml` |
| reward/done 预处理 | `examples/calvin/scripts/prepare_awac_rewards.py` |
| critic 启动模板 | `examples/calvin/train_files/run_calvin_awac_critic.sh` |
| actor 后训练模板 | `examples/calvin/train_files/run_calvin_awac_actor.sh` |
| critic 训练循环 | `starVLA/training/train_awac_critic.py` |
| actor 后训练循环 | `starVLA/training/train_awac_actor.py` |
| transition dataloader | `starVLA/dataloader/awac_transition_dataset.py` |
| Q critic | `starVLA/model/modules/critic/awac_q_critic.py` |
| V network | `starVLA/model/modules/critic/awac_v_network.py` |

先给 rollout 数据补 reward/done：

```bash
cd /home/lwb/Projects/SII/starVLA_Projects/starVLA_Project

export ROLLOUT_DIR=logs/log_20260519_1640_qwen35_pi_rollout_debug_800step_cpuoffload/rollout_lerobot_debug_3seq

"$(conda info --base)/envs/starVLA_qwen35/bin/python" \
  examples/calvin/scripts/prepare_awac_rewards.py \
  --dataset_root "${ROLLOUT_DIR}" \
  --action_horizon 8 \
  --gamma 0.996 \
  --success_column episode_success
```

为什么要建 `awac_debug_root/calvin_task_ABC_D`：

当前 registry 里 `calvin_abc_d_h200` 对应的数据集名是 `calvin_task_ABC_D`。rollout 目录本身叫 `rollout_lerobot_debug_3seq`，所以本地调试时用 symlink 适配 registry，不改全局注册表：

```bash
export RUN_DIR=logs/log_20260519_1640_qwen35_pi_rollout_debug_800step_cpuoffload
mkdir -p "${RUN_DIR}/awac_debug_root"
ln -sfn ../rollout_lerobot_debug_3seq "${RUN_DIR}/awac_debug_root/calvin_task_ABC_D"
```

本地 critic smoke/调试命令：

```bash
cd /home/lwb/Projects/SII/starVLA_Projects/starVLA_Project

export RUN_DIR=logs/log_20260519_1640_qwen35_pi_rollout_debug_800step_cpuoffload
export CKPT_PATH="${RUN_DIR}/checkpoints/qwen35_pi_rollout_debug_800step_cpuoffload/checkpoints/steps_600_pytorch_model.pt"
export STAR_VLA_PYTHON="$(conda info --base)/envs/starVLA_qwen35/bin/python"
export WANDB_MODE=offline

CUDA_VISIBLE_DEVICES=0 "${STAR_VLA_PYTHON}" -m accelerate.commands.launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2_route_validation_cpu_offload.yaml \
  --num_processes 1 \
  starVLA/training/train_awac_critic.py \
  --config_yaml examples/calvin/train_files/starvla_awac_calvin.yaml \
  --framework.name QwenPI \
  --framework.qwenvl.base_vlm ./playground/Pretrained_models/Qwen3.5-0.8B \
  --framework.action_model.state_dim 7 \
  --datasets.awac_data.data_root_dir "${RUN_DIR}/awac_debug_root" \
  --datasets.vla_data.data_root_dir "${RUN_DIR}/awac_debug_root" \
  --datasets.awac_data.data_mix calvin_abc_d_h200 \
  --datasets.vla_data.data_mix calvin_abc_d_h200 \
  --trainer.pretrained_checkpoint "${CKPT_PATH}" \
  --trainer.critic_max_train_steps 120 \
  --trainer.save_interval 120 \
  --trainer.use_tensorboard true \
  --trainer.tensorboard_log_dir "${RUN_DIR}/tensorboard_critic" \
  --run_root_dir "${RUN_DIR}" \
  --run_id awac_critic_pi_120step_state7_trimmed
```

注意：这里的 `state_dim=7` 是当前 PI/AWAC 调试链路的设置；rollout writer 写的是 CALVIN 8D state，当前 dataloader 会在数据维度大于配置时裁剪尾部并报警。正式做 proprio 版本时，必须统一训练、rollout、critic 的 state 维度，不能混着用。

## 4. 此时评估怎么搞

评估要分三层，不要混：

1. 训练内 eval：`train_starvla.py` 里只是对 batch 算动作误差，不等于 CALVIN 成功率。
2. CALVIN debug eval：少量 sequence，主要看 server、归一化、mp4、动作有没有明显错。
3. 完整 CALVIN ABC->D eval：候选 checkpoint 才跑，核心看 `avg_seq_len` 和 1~5 连续任务成功率。

最小评估命令就是上面的两终端：

```text
终端 A：bash examples/calvin/eval_files/run_policy_server_debug.sh
终端 B：bash examples/calvin/eval_files/eval_calvin_debug.sh
```

输出看这里：

```text
logs/log_*/terminal/eval.log
logs/log_*/mp4/results.json
logs/log_*/mp4/*.mp4
```

`results.json` 的关键字段：

```text
avg_seq_len     平均每条 5-step chain 连续成功几个子任务
chain_sr.1      第 1 个任务成功率
chain_sr.2      连续 2 个任务成功率
...
chain_sr.5      连续 5 个任务成功率
task_info       每个具体任务的成功/总数
```

H200 训练后评估命令集中在 `README_zh.md` 第 17 节。当前建议：

```text
每条路线先评 steps_10000 / steps_20000 / steps_30000 的 debug eval
再挑候选跑完整 1000 sequence eval
不提交没有 results.json 和 mp4 的 checkpoint
不只按训练 loss 选 checkpoint
```

## 5. 查文件速查表

| 你想查什么 | 去哪里 |
| --- | --- |
| 本地训练命令 | `README_zh.md` 第 10、13 节 |
| H200 批量训练命令 | `H200_批量训练命令行.md` |
| H200 训练后 eval | `README_zh.md` 第 17 节 |
| 训练入口 | `starVLA/training/train_starvla.py` |
| policy server | `deployment/model_server/server_policy.py` |
| server 反归一化 | `deployment/model_server/policy_wrapper.py`、`deployment/model_server/policy_norm_processor.py` |
| CALVIN eval/rollout | `examples/calvin/eval_files/eval_calvin.py` |
| rollout 写 LeRobot 数据 | `examples/calvin/eval_files/rollout_lerobot_writer.py` |
| AWAC reward 预处理 | `examples/calvin/scripts/prepare_awac_rewards.py` |
| AWAC critic | `starVLA/training/train_awac_critic.py` |
| AWAC actor | `starVLA/training/train_awac_actor.py` |

## 6. 高标准检查清单

跑通不代表成功，当前每一步都要硬检查：

```text
训练：
- terminal/train.log 里没有 OOM / NaN / Inf / traceback
- checkpoints/steps_N_pytorch_model.pt 存在且可 reload
- config.yaml / dataset_statistics.json 跟 checkpoint 同目录结构可被 from_pretrained 找到

server：
- policy_server.log 里 server running
- action_chunk_size=8
- available_unnorm_keys 包含 franka

eval：
- terminal/eval.log 有 Average successful sequence length
- mp4/results.json 存在
- mp4/*.mp4 存在，人工看动作

rollout 数据：
- meta/info.json total_frames > 0
- data/**/*.parquet 存在
- rollout_analysis/rollout_steps.jsonl 存在
- 如果给 critic，用 prepare_awac_rewards.py 写入 reward/done

critic：
- critic_loss/value_loss 有 TensorBoard 曲线
- checkpoints/steps_N_critic.pt 可 torch.load
- 如果 state 维度不一致，必须明确是裁剪、改配置，还是重采数据，不能静默混用
```

## 7. 以 `log_20260519_1640_qwen35_pi_rollout_debug_800step_cpuoffload` 为例看目录

样例目录：

```text
logs/log_20260519_1640_qwen35_pi_rollout_debug_800step_cpuoffload
```

这个目录不是单一用途目录，而是一次 PI 调试 run 的“总工作台”。里面同时放了：

```text
1. PI 行为克隆训练产物
2. policy server 启动日志
3. CALVIN debug eval 结果和 mp4
4. rollout 转成 LeRobot 后的数据集
5. AWAC critic 多次尝试和最终成功 checkpoint
6. TensorBoard event
```

### 7.1 根目录下各文件夹是什么

| 路径 | 作用 | 当前怎么看 |
| --- | --- | --- |
| `checkpoints/qwen35_pi_rollout_debug_800step_cpuoffload/` | PI 行为克隆训练的正式 run dir。policy server 加载 checkpoint 时主要靠这里。 | 有 `steps_600_pytorch_model.pt`、`config.yaml`、`dataset_statistics.json`。 |
| `configs/` | 训练启动时复制的脚本和配置快照，方便复盘当时怎么启动。 | 里面有 `run_route_validation_train.sh`、DeepSpeed CPU offload 配置、基础 YAML。 |
| `terminal/` | 这次调试的主要终端日志集合。 | `train.log`、`eval_pi_600_3seq.log`、多个 critic 尝试日志都在这里。 |
| `policy_server_pi_600/` | 用 `steps_600` checkpoint 启动 policy server 时生成的 server run 目录。 | 真正有用的是 `terminal/policy_server.log` 和 `configs/run_policy_server_debug.sh`。 |
| `mp4_3seq/` | 3 条 CALVIN debug eval 的普通评估输出。 | 有 `results.json` 和 3 个失败 mp4。 |
| `rollout_lerobot_debug_3seq/` | 把上面 3 条 eval 过程逐步写成 LeRobot 风格数据集。 | critic/AWAC 后训练主要吃这里的数据。 |
| `awac_debug_root/` | 给 AWAC dataloader 用的 registry 适配根目录。 | `calvin_task_ABC_D -> ../rollout_lerobot_debug_3seq` 是 symlink。 |
| `awac_critic_pi_120step_state7_trimmed/` | 最终跑通的 AWAC critic 训练目录。 | 有 `checkpoints/steps_120_critic.pt`。 |
| `awac_critic_pi_120step*` 其他目录 | 前面几次 critic 调试尝试。 | 多数没有 checkpoint，只保留失败/中间痕迹。 |
| `tensorboard/` | 原训练过程中直接写出的 TensorBoard event。 | 有两个 event 文件，可能包含不完整/重复尝试。 |
| `tensorboard_complete/` | 后处理导出的“清洁版”训练曲线。 | 推荐看这个训练曲线。 |
| `tensorboard_critic/` | critic 训练曲线。 | 推荐看这个 critic 曲线。 |
| `train/`、`eval/`、`metrics/`、`mp4/` | 标准脚本预创建的占位目录。 | 当前基本为空，真正 eval 在 `mp4_3seq/`。 |

### 7.2 训练产物：`checkpoints/qwen35_pi_rollout_debug_800step_cpuoffload/`

关键文件：

```text
checkpoints/qwen35_pi_rollout_debug_800step_cpuoffload/
  config.yaml
  config.full.yaml
  dataset_statistics.json
  summary.jsonl
  checkpoints/steps_600_pytorch_model.pt
```

含义：

| 文件 | 作用 |
| --- | --- |
| `checkpoints/steps_600_pytorch_model.pt` | 这次能拿来 eval/rollout 的 PI 模型权重。约 2.8G。 |
| `config.yaml` | policy server 恢复模型时用的关键配置快照。这个比 `configs/` 里的基础 YAML 更可信。 |
| `config.full.yaml` | 完整合并后的训练配置，包含 CLI override。复盘真实训练参数看它。 |
| `dataset_statistics.json` | 训练时数据归一化统计。server 反归一化 action 必须用它，缺了 eval 会不可靠或失败。 |
| `summary.jsonl` | 每次保存 checkpoint 时写一行 step。当前有 `200/400/600`。 |

当前真实训练信息：

```text
framework.name = QwenPI
base_vlm = ./playground/Pretrained_models/Qwen3.5-0.8B
action_model_type = LayerwiseFM
action_horizon = 8
state_dim = 7
include_state = false
max_train_steps = 800
save_interval = 200
```

注意一个坑：

```text
configs/starvla_train_calvin_qwen35_oft_smoke.yaml
```

这个文件名里有 `oft_smoke`，但本次真实模型是 `QwenPI`。原因是启动脚本先复制了基础 YAML，随后命令行 `ROUTE=p4_pi` 和 CLI override 把模型改成了 PI。所以判断真实训练参数时，不要只看 `configs/` 里的文件名，要看：

```text
checkpoints/qwen35_pi_rollout_debug_800step_cpuoffload/config.yaml
checkpoints/qwen35_pi_rollout_debug_800step_cpuoffload/config.full.yaml
terminal/train.log
```

当前训练不是完整成功结束：

```text
train.log 里保存了 steps_200 / steps_400 / steps_600
最后在收尾/保存阶段收到 SIGKILL
所以只认 steps_600_pytorch_model.pt，不认 800 step final model
```

### 7.3 终端日志：`terminal/`

关键文件：

| 文件 | 作用 |
| --- | --- |
| `train.log` | PI 训练日志。看 step、loss、checkpoint、OOM/SIGKILL。 |
| `eval_pi_600_3seq.log` | 用 `steps_600` 做 3 条 CALVIN debug eval 的日志。 |
| `critic_pi_120step.log` | 最早 critic 尝试日志，未形成最终可用结果。 |
| `critic_pi_120step_state7.log` | 调 state_dim 后的中间尝试。 |
| `critic_pi_120step_state7_offline.log` | 使用 wandb offline 的中间尝试。 |
| `critic_pi_120step_state7_trimmed.log` | 最终跑通 120 step critic 的日志。 |

训练日志要看：

```bash
rg -a -n "ROUTE=|RUN_ID=|MAX_TRAIN_STEPS=|Checkpoint saved|Traceback|SIGKILL" \
  logs/log_20260519_1640_qwen35_pi_rollout_debug_800step_cpuoffload/terminal/train.log
```

当前结论：

```text
ROUTE=p4_pi
MAX_TRAIN_STEPS=800
SAVE_INTERVAL=200
保存成功：steps_200、steps_400、steps_600
最后：Signal 9 (SIGKILL)
```

eval 日志要看：

```bash
rg -a -n "policy_setup|Evaluating sequence|Subtask:|Average successful sequence length|Best model" \
  logs/log_20260519_1640_qwen35_pi_rollout_debug_800step_cpuoffload/terminal/eval_pi_600_3seq.log
```

当前结论：

```text
server 加载的是 steps_600_pytorch_model.pt
action_chunk_size=8
available_unnorm_keys=['franka']
3 条 sequence 的第一个 subtask 都 fail
Average successful sequence length = 0.0
```

critic 最终日志要看：

```bash
rg -a -n "valid transitions|state dimension mismatch|critic_loss|Saved AWAC critic" \
  logs/log_20260519_1640_qwen35_pi_rollout_debug_800step_cpuoffload/terminal/critic_pi_120step_state7_trimmed.log
```

当前结论：

```text
AWAC transition: 1035/1080 valid transitions
critic 跑满 120/120
保存了 steps_120_critic.pt
有 state 8 -> state_dim 7 的裁剪 warning
```

这个 warning 不能忽略。它表示 rollout 写出的 state 是 8 维，但当前 PI/AWAC 配置用的是 7 维。调试时明确裁剪了尾部，正式实验要统一 state 定义。

### 7.4 policy server：`policy_server_pi_600/`

关键文件：

```text
policy_server_pi_600/
  configs/run_policy_server_debug.sh
  terminal/policy_server.log
```

作用：

```text
记录当时怎么把 steps_600 checkpoint 起成 websocket policy server。
```

有用信息在 `policy_server.log`：

```text
ckpt_path = .../steps_600_pytorch_model.pt
action_chunk_size = 8
available_unnorm_keys = ['franka']
default_unnorm_key = franka
```

`policy_server_pi_600/checkpoints`、`eval`、`metrics`、`mp4`、`train` 当前是空占位目录，不是核心产物。

### 7.5 普通评估输出：`mp4_3seq/`

目录：

```text
mp4_3seq/
  results.json
  0-0-rotate_blue_block_right-fail.mp4
  1-0-turn_off_led-fail.mp4
  2-0-lift_pink_block_slider-fail.mp4
```

作用：

```text
这是 CALVIN debug eval 的直接结果。
results.json 给机器读，mp4 给人看。
```

当前 `results.json`：

```text
avg_seq_len = 0.0
chain_sr.1 到 chain_sr.5 全是 0.0
失败任务：
  rotate_blue_block_right
  turn_off_led
  lift_pink_block_slider
```

判断模型有没有真实进步，不能只看训练 loss，要看这个目录里的：

```text
results.json
mp4 视频动作
```

### 7.6 rollout 数据集：`rollout_lerobot_debug_3seq/`

结构：

```text
rollout_lerobot_debug_3seq/
  meta/
  data/chunk-000/
  videos/chunk-000/image/
  videos/chunk-000/wrist_image/
  rollout_analysis/
```

这是把 3 条 eval 轨迹写成 LeRobot 风格数据集，后面给 AWAC critic 用。

#### `meta/info.json`

数据集总说明。当前关键值：

```text
total_episodes = 3
total_frames = 1080
total_tasks = 3
total_videos = 6
fps = 10
image shape = [200, 200, 3]
wrist_image shape = [84, 84, 3]
state shape = [8]
actions shape = [7]
robot_obs shape = [15]
robot_joint_pos shape = [7]
```

#### `meta/modality.json`

告诉 StarVLA dataloader：

```text
哪一列是 state
哪一列是 action
哪一列是 image / wrist_image
哪一列是语言任务 task_index
```

没有它，LeRobot 数据很容易“看起来有 parquet，但 dataloader 不知道怎么读”。

#### `meta/tasks.jsonl`

语言任务字典。当前 3 个 task：

```text
0: take the blue block and rotate it to the right
1: press the button to turn off the led light
2: lift the pink block from the sliding cabinet
```

#### `meta/episodes.jsonl`

每个 episode 的摘要。当前 3 个 episode 都是：

```text
length = 360
success = false
subtask_index = 0
```

意思是每条 5-step chain 都卡在第一个子任务，所以只写了第一个 subtask 的 rollout。

#### `meta/episodes_stats.jsonl`

每个 episode 的统计量：

```text
image / wrist_image 的 min/max/mean/std
state 的 min/max/mean/std
actions 的 min/max/mean/std
robot_obs / robot_joint_pos 的 min/max/mean/std
done / episode_success / distance_to_target 等统计
```

用于 sanity check。例如看 action 尺度、gripper 分布、state 是否全 0、distance 是否 NaN。

#### `meta/stats_gr00t.json`

给 GR00T/LeRobot 风格 loader 使用的统计文件。它偏数据归一化/统计用途，不是评估指标。

#### `meta/steps_data_index.pkl`

dataloader 快速定位 step 的索引缓存。人一般不读它，但训练读数据时会用到。

#### `data/chunk-000/episode_*.parquet`

真正的逐步 rollout 数据。当前有 3 个：

```text
episode_000000.parquet
episode_000001.parquet
episode_000002.parquet
```

每个 360 行。当前列包括：

```text
image
wrist_image
state
actions
robot_obs
robot_joint_pos
done
env_done
episode_success
distance_to_target
distance_to_robot_target
timestamp
frame_index
episode_index
index
task_index
step_reward
reward
```

其中：

```text
state: 8 维
actions: 7 维
robot_obs: 15 维
robot_joint_pos: 7 维
done: 最后 8 帧为 True
step_reward: 普通步 -1，失败终点 -3000
reward: 按 action_horizon=8 折扣后的 chunk return
```

`step_reward/reward/done` 不是 rollout writer 原生就有的，是后面跑过：

```bash
examples/calvin/scripts/prepare_awac_rewards.py \
  --success_column episode_success
```

之后写进去的。

#### `videos/chunk-000/image/*.mp4`

LeRobot 数据集里 `image` 视角的视频，3 个 episode 各一个。对应 static camera。

#### `videos/chunk-000/wrist_image/*.mp4`

LeRobot 数据集里 `wrist_image` 视角的视频，3 个 episode 各一个。对应 gripper/wrist camera。

#### `rollout_analysis/rollout_steps.jsonl`

逐 step 诊断日志，一行一个环境 step。比 parquet 更适合分析失败原因。

每行包含：

```text
episode_index
sequence_index
subtask_index
subtask
language
frame_index
done
env_done
step_success
distance_to_target
distance_to_target_source
distance_to_robot_target
distance_to_robot_target_source
current_info
episode_success
```

`current_info` 很大，里面有：

```text
robot_info.tcp_pos
robot_info.tcp_orn
robot_info.gripper_opening_width
robot_info.arm_joint_states
scene_info.movable_objects
scene_info.doors/buttons/switches/lights
contacts
```

这个文件可以用来回答“为什么失败”：比如目标距离有没有变小，机械臂是否接近物体，开关/抽屉状态有没有变化。

### 7.7 AWAC 适配目录：`awac_debug_root/`

当前内容：

```text
awac_debug_root/calvin_task_ABC_D -> ../rollout_lerobot_debug_3seq
```

这是一个 symlink，不是复制数据。

原因：

```text
AWAC 配置使用 data_mix=calvin_abc_d_h200
registry 里 calvin_abc_d_h200 对应的数据集名是 calvin_task_ABC_D
但真实 rollout 目录名是 rollout_lerobot_debug_3seq
所以用 symlink 把名字对齐，让 dataloader 不用改 registry 也能读
```

### 7.8 critic 目录

这几个目录含义不同：

| 路径 | 当前状态 | 解释 |
| --- | --- | --- |
| `awac_critic_pi_120step/` | 基本空，无 checkpoint | 早期尝试，未成功落盘。 |
| `awac_critic_pi_120step_state7/` | 基本空，无 checkpoint | 调 state_dim 后的尝试，未最终成功。 |
| `awac_critic_pi_120step_state7_offline/` | 有 wandb offline 目录，无 checkpoint | 中间尝试留下的 W&B 记录。 |
| `awac_critic_pi_120step_state7_trimmed/` | 有 checkpoint | 当前真正有用的 critic 结果。 |

最终有用文件：

```text
awac_critic_pi_120step_state7_trimmed/checkpoints/steps_120_critic.pt
```

这个 checkpoint 里保存的是 critic/value 网络，不是可直接 rollout 的 policy。它用于后续 AWAC actor fine-tune。

### 7.9 TensorBoard 目录

当前有三类：

```text
tensorboard/
tensorboard_complete/
tensorboard_critic/
```

建议看：

```bash
tensorboard --logdir logs/log_20260519_1640_qwen35_pi_rollout_debug_800step_cpuoffload/tensorboard_complete
```

看训练曲线：

```text
action_dit_loss
learning_rate
timing/data
timing/model
eval/avg_seq_len
rollout/total_frames
```

看 critic 曲线：

```bash
tensorboard --logdir logs/log_20260519_1640_qwen35_pi_rollout_debug_800step_cpuoffload/tensorboard_critic
```

重点看：

```text
critic_loss
value_loss
q_mean
target_mean
v_mean
total_loss
learning_rate
```

`tensorboard_complete/` 是更干净的训练曲线；`tensorboard/` 是训练过程中和后处理过程中都写过的目录，可能有多个 event 文件。

### 7.10 这个 run 的结论

这次 run 能证明：

```text
PI 训练至少能保存到 steps_600
steps_600 能被 policy server 加载
server 能返回 action_chunk_size=8 的动作
CALVIN eval 能跑并生成 results.json / mp4
rollout 能写成 LeRobot parquet + meta + videos
reward/done 能后处理写入 parquet
AWAC critic 能读 rollout 数据并训练 120 step
```

这次 run 不能证明：

```text
PI 策略已经有效
800 step 最终 checkpoint 有效
critic 已经可用于提升策略
state/proprio 链路已经完全统一
```

原因：

```text
训练最后 SIGKILL，只能用 steps_600
3 条 debug eval 的 avg_seq_len=0.0
rollout 三个 episode 都失败
critic 只是 120 step 调试，并且存在 state 8 -> 7 的裁剪
```

所以这个目录最适合当作：

```text
工程闭环样例
rollout 数据格式样例
AWAC critic 调试样例
TensorBoard/MP4 可视化样例
```

但不能当作：

```text
最终可汇报的 CALVIN 高分模型
```
