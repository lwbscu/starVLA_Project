# H200 AWAC 后训练 PI-State 检查与 Rollout 命令

目标：严格按 `examples/calvin/train_files/critic.md` 的 §2 契约检查 PI-State BC 结果、生成可后训练的 rollout LeRobot 数据，并进入 AWAC critic / actor。

当前已检查的 BC 根目录：

```text
/inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project/logs/20260519_h200_server1_pi_state_30k_v1
```

本文件只写命令。训练、真实 CALVIN eval、AWAC critic/actor 都必须在 H200 训练服务器执行。

## 1. 当前阻塞结论

我在当前 CPU 环境只读运行了 readiness 检查，结果是 `FAIL`，阻塞项如下：

- `qwen35_0p8b`、`qwen35_4b`：PI-State、`include_state=true`、`state_dim=8`、`steps_30000` 都通过。
- `qwen35_9b`：最新完整 checkpoint 只有 `steps_25000`；训练在 step 26000 存盘时因磁盘写入失败中断，没有 `steps_30000`。
- 原始 `calvin_task_ABC_D` parquet 抽样缺 `success` 列。
- 原始 `calvin_task_ABC_D` parquet 抽样缺 `step_reward/reward/done`，尚不能直接跑 AWAC critic。

严格结论：现在不能直接开 AWAC critic/actor。先补成功标签并跑 `prepare_awac_rewards.py`，或用严格 PI-State rollout 生成带 `success` 的 LeRobot 数据再预处理。

代码层核对结论：

- dataloader 在 `include_state=true` 时会把 parquet 的 `state` 拼进 sample。
- `QwenPI.forward/predict_action` 会读取 `example["state"]`。
- `LayerwiseFM` 在 `state_dim=8` 且 state 非空时会通过 `state_encoder`，并把 state token 拼到 `[state | future_tokens | action_features]` 里。
- 所以 0.8B/4B 这两个已完成 checkpoint 的 state 消费链路是成立的；9B 阻塞是 checkpoint 不完整，不是 state 链路错误。

为了避免后续误用，H200 PI yaml 已改为默认 `include_state: true`、`state_dim: 8`；PI-state launcher 默认 `SAVE_INTERVAL=10000`，只保留 10000/20000/30000 这类大间隔 checkpoint。

## 2. Readiness 检查

```bash
cd /inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project

export STAR_VLA_PYTHON=/inspire/qb-ilm2/project/26summer-camp-10/26220216/miniconda3/envs/starVLA_qwen35/bin/python
export OUT="logs/awac_readiness_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

"${STAR_VLA_PYTHON}" examples/calvin/scripts/inspect_awac_readiness.py \
  --batch-root "${PROJECT_ROOT}/logs/20260519_h200_server1_pi_state_30k_v1" \
  --data-root /inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d \
  --dataset-name calvin_task_ABC_D \
  --data-mix calvin_abc_d_h200 \
  --expected-step 30000 \
  --json-out "$OUT/awac_readiness.json" \
  > "$OUT/awac_readiness.md"

echo "inspect_status=$?"
sed -n '1,260p' "$OUT/awac_readiness.md"
```

只检查“是否能先跑 reward 预处理”，不把缺 `step_reward/reward/done` 当硬失败：

```bash
"${STAR_VLA_PYTHON}" examples/calvin/scripts/inspect_awac_readiness.py \
  --batch-root "${PROJECT_ROOT}/logs/20260519_h200_server1_pi_state_30k_v1" \
  --data-root /inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d \
  --dataset-name calvin_task_ABC_D \
  --data-mix calvin_abc_d_h200 \
  --expected-step 30000 \
  --no-require-preprocessed
```

## 3. 严格 PI-State Rollout

这个 wrapper 会强制：

- `SEND_STATE_TO_POLICY=1`
- `WRITE_ROLLOUT_LEROBOT=1`
- `EXPECTED_JOBS` 默认只包含 `server1_pi_state`
- 每个 job 的最新 checkpoint 必须 `>= EVAL_EXPECTED_STEP`
- 必须显式提供真实 CALVIN eval dataset，不走 debug fallback

当前 `qwen35_9b` 没有 `steps_30000`，所以要么先补训 9B，要么只 rollout 0.8B / 4B：

```bash
cd /inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project

export H200_CALVIN_EVAL_DATASET_PATH=/path/to/real/calvin/task_D_D_or_task_ABC_D
export EVAL_BATCH_ROOT="${PROJECT_ROOT}/logs/20260519_h200_server1_pi_state_30k_v1"
export EVAL_OUTPUT_ROOT="${PROJECT_ROOT}/logs/calvin_eval_rollout_20260519_h200_server1_pi_state_30k_v1"
export EVAL_GPU=0
export EVAL_PORT=6200
export NUM_SEQUENCES=5

export EXPECTED_JOBS=$'server1_pi_state/qwen35_0p8b\nserver1_pi_state/qwen35_4b'

bash examples/calvin/eval_files/eval_h200_pi_state_rollout_latest.sh
```

如果 9B 后续补齐了 `steps_30000`，删掉 `EXPECTED_JOBS` 覆盖即可检查并 rollout 三个模型。

## 3.1 严格重训 PI Baseline

如果你要“完全干净”的 PI baseline，建议直接重训到一个新 batch。这个命令会在启动前检查：

- LeRobot 根目录存在；
- `meta/info.json` 的 state dim 是 8；
- 抽样 parquet 有 `state/actions`，维度分别是 8/7；
- 抽样 episode 长度满足 AWAC 的 `t + 2H`，其中 `H=8`。

三模型全量重训：

```bash
cd /inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project

export PROJECT_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project
export CONDA_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/26220216/miniconda3
export STAR_VLA_PYTHON="${CONDA_ROOT}/envs/starVLA_qwen35/bin/python"

export H200_CALVIN_DATA_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d
export H200_CALVIN_DATA_NAME=calvin_task_ABC_D
export H200_CALVIN_DATA_MIX=calvin_abc_d_h200

export BATCH_NAME=20260520_h200_server1_pi_state_strict_30k_v1
export H200_PI_STATE_MODELS="qwen35_0p8b qwen35_4b qwen35_9b"
export MAX_TRAIN_STEPS=30000
export SAVE_INTERVAL=10000
export CHECKPOINT_KEEP_LATEST=1
export CHECKPOINT_KEEP_STEPS=10000,20000,30000
export INCLUDE_STATE=true
export STATE_DIM=8
export PI_STATE_DIM=8
export ACTION_DIM=7
export ACTION_HORIZON=8

bash examples/calvin/train_files/run_h200_server1_pi_state_three_sizes.sh
```

如果你只想补 9B，一个新目录单独跑 9B：

```bash
export BATCH_NAME=20260520_h200_server1_pi_state_strict_9b_30k_v1
export H200_PI_STATE_MODELS="qwen35_9b"
export MAX_TRAIN_STEPS=30000
export SAVE_INTERVAL=10000
export CHECKPOINT_KEEP_STEPS=10000,20000,30000

bash examples/calvin/train_files/run_h200_server1_pi_state_three_sizes.sh
```

不建议把旧 9B 的 `steps_25000` 冒充 30k baseline。若要从 25k 初始化再训 5k，也应单独命名为 `from25k_to30k`，因为当前训练脚本保存步数会从新 run 的 0 开始计数，不等价于原生 `steps_30000` 命名。

## 4. Rollout 数据预处理

新生成的 rollout parquet 会包含标准 `success` 列。对每个 `rollout_lerobot` 跑 AWAC reward 预处理：

```bash
find "${EVAL_OUTPUT_ROOT}" -path "*/rollout_lerobot/meta/info.json" -print \
  | while read -r info; do
      dataset_root="$(dirname "$(dirname "$info")")"
      echo "prepare_awac_rewards: ${dataset_root}"
      "${STAR_VLA_PYTHON}" examples/calvin/scripts/prepare_awac_rewards.py \
        --dataset_root "${dataset_root}" \
        --action_horizon 8 \
        --gamma 0.996
    done
```

预处理后重新检查 rollout：

```bash
"${STAR_VLA_PYTHON}" examples/calvin/scripts/inspect_awac_readiness.py \
  --batch-root "${PROJECT_ROOT}/logs/20260519_h200_server1_pi_state_30k_v1" \
  --data-root /inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d \
  --dataset-name calvin_task_ABC_D \
  --data-mix calvin_abc_d_h200 \
  --expected-step 30000 \
  --rollout-root "${EVAL_OUTPUT_ROOT}" \
  --require-rollout \
  --no-require-preprocessed
```

## 5. 用 Rollout 跑 AWAC

`calvin_rollout_h200` 已注册为：

```text
data_root_dir + rollout_lerobot
```

以 4B rollout 为例，`calvin_data_root` 应该指到包含 `rollout_lerobot/` 的 `eval` 目录：

```bash
export bc_checkpoint="${PROJECT_ROOT}/logs/20260519_h200_server1_pi_state_30k_v1/server1_pi_state/qwen35_4b/pi_state_qwen35_4b_30000step/checkpoints/pi_state_qwen35_4b_30000step/checkpoints/steps_30000_pytorch_model.pt"
export calvin_data_root="${PROJECT_ROOT}/logs/calvin_eval_rollout_20260519_h200_server1_pi_state_30k_v1/server1_pi_state/qwen35_4b/steps_30000/eval"
export data_mix=calvin_rollout_h200
export include_state=true
export state_dim=8
export base_vlm="${PROJECT_ROOT}/playground/Pretrained_models/Qwen3.5-4B"
export run_id=awac_pi_state_qwen35_4b_rollout_critic

bash examples/calvin/train_files/run_calvin_awac_critic.sh
```

Actor 阶段：

```bash
export critic_checkpoint="${PROJECT_ROOT}/logs/awac_pi_state_qwen35_4b_rollout_critic/checkpoints/steps_50000_critic.pt"
export run_id=awac_pi_state_qwen35_4b_rollout_actor

bash examples/calvin/train_files/run_calvin_awac_actor.sh
```

## 6. 原始 CALVIN 数据路线

原始 `calvin_task_ABC_D` 目前缺 `success`。不要直接用 `--assume_success_if_missing`，除非你能确认这些 episode 全部是成功 demo，并且在结果表里标注这个假设。严格路线是先补 `success` 标签，再跑：

```bash
"${STAR_VLA_PYTHON}" examples/calvin/scripts/prepare_awac_rewards.py \
  --dataset_root /inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d/calvin_task_ABC_D \
  --action_horizon 8 \
  --gamma 0.996
```

未补 `success` 前，不开原始数据 AWAC critic。
