# H200 AWAC 后训练准备命令行

本文档用于把已训练好的 PI-State baseline 接到 `examples/calvin/train_files/critic.md` 的 AWAC 后训练流程。核心原则：不伪造成功，不绕过 `H=8 / gamma=0.996 / reward=chunk return / done=最后 H 帧` 的契约。

当前接受的 BC baseline：

| 模型 | checkpoint 口径 |
|---|---|
| Qwen3.5-0.8B | `steps_30000` |
| Qwen3.5-4B | `steps_30000` |
| Qwen3.5-9B | `steps_25000`，必须在实验表中显式记录为 25k |

## 0. 关键判断

`success` 分两种：

- **评估成功率**：由 CALVIN eval 的 task oracle 判断，写入 `results.json`。这不依赖训练 parquet 是否有 `success` 列。
- **AWAC reward 预处理标签**：`prepare_awac_rewards.py` 需要 parquet 末帧 `success` 来写 `step_reward/reward/done`。训练数据没有 `success` 时，不能直接训 critic。

`rollout` 也分清楚：

- AWAC critic/actor 的主训练数据是离线 LeRobot 训练集，不是把所有训练集都 rollout 一遍。
- rollout 是当前 policy 在 CALVIN 环境里的诊断/评估数据，用来确认 PI-State 推理接口、state 传入、动作闭环、成功率和失败轨迹。
- 论文/汇报的正式成功率应跑官方 eval sequences；排错可先小量 `NUM_SEQUENCES=5/20`。

## 1. 初始化环境

```bash
cd /inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project

export PATH="/inspire/qb-ilm2/project/26summer-camp-10/26220216/miniconda3/bin:$PATH"
source /inspire/qb-ilm2/project/26summer-camp-10/26220216/miniconda3/etc/profile.d/conda.sh
conda activate starVLA_qwen35

export PROJECT_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project
export CONDA_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/26220216/miniconda3
export STAR_VLA_PYTHON="${CONDA_ROOT}/envs/starVLA_qwen35/bin/python"
export CALVIN_PYTHON="${CONDA_ROOT}/envs/calvin/bin/python"

export H200_CALVIN_DATA_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d
export H200_CALVIN_DATA_NAME=calvin_task_ABC_D
export H200_CALVIN_DATA_MIX=calvin_abc_d_h200
export AWAC_DATASET_ROOT="${H200_CALVIN_DATA_ROOT}/${H200_CALVIN_DATA_NAME}"

export PI_STATE_BATCH_ROOT="${PROJECT_ROOT}/logs/20260519_h200_server1_pi_state_30k_v1"
export PI_STATE_CKPT_0P8B="${PI_STATE_BATCH_ROOT}/server1_pi_state/qwen35_0p8b/pi_state_qwen35_0p8b_30000step/checkpoints/pi_state_qwen35_0p8b_30000step/checkpoints/steps_30000_pytorch_model.pt"
export PI_STATE_CKPT_4B="${PI_STATE_BATCH_ROOT}/server1_pi_state/qwen35_4b/pi_state_qwen35_4b_30000step/checkpoints/pi_state_qwen35_4b_30000step/checkpoints/steps_30000_pytorch_model.pt"
export PI_STATE_CKPT_9B="${PI_STATE_BATCH_ROOT}/server1_pi_state/qwen35_9b/pi_state_qwen35_9b_30000step/checkpoints/pi_state_qwen35_9b_30000step/checkpoints/steps_25000_pytorch_model.pt"

test -f "${PI_STATE_CKPT_0P8B}"
test -f "${PI_STATE_CKPT_4B}"
test -f "${PI_STATE_CKPT_9B}"
```

## 2. 严格 readiness 检查

这里明确只允许 9B 用 25k，0.8B/4B 仍按 30k 检查。

```bash
export OUT="logs/awac_readiness_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

"${STAR_VLA_PYTHON}" examples/calvin/scripts/inspect_awac_readiness.py \
  --batch-root "${PI_STATE_BATCH_ROOT}" \
  --data-root "${H200_CALVIN_DATA_ROOT}" \
  --dataset-name "${H200_CALVIN_DATA_NAME}" \
  --data-mix "${H200_CALVIN_DATA_MIX}" \
  --expected-step 30000 \
  --job-expected-step server1_pi_state/qwen35_9b=25000 \
  --json-out "$OUT/awac_readiness.json" \
  > "$OUT/awac_readiness.md"

echo "inspect_status=$?"
sed -n '1,260p' "$OUT/awac_readiness.md"
```

此时如果只剩 `success labels` 和 `AWAC reward preprocessing` 失败，进入下一节。

## 3. 补齐 success 标签

高标准做法有两种，只能选一种：

1. 如果有带 `success` 的源数据集，使用 `--source-dataset-root` 按 episode 相对路径复制。
2. 如果确认 `calvin_task_ABC_D` 是成功 expert demonstration 数据，显式使用 `--all-success-demo`，脚本会写成“仅末帧 success=True，其余帧 False”。

先 dry-run：

```bash
export SUCCESS_OUT="logs/awac_success_stamp_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SUCCESS_OUT"

"${STAR_VLA_PYTHON}" examples/calvin/scripts/stamp_awac_success_labels.py \
  --dataset-root "${AWAC_DATASET_ROOT}" \
  --all-success-demo \
  --manifest-jsonl "$SUCCESS_OUT/success_stamp_manifest.jsonl" \
  > "$SUCCESS_OUT/success_stamp_dry_run.log"

cat "$SUCCESS_OUT/success_stamp_dry_run.log"
```

确认 `failures=0` 后才写入：

```bash
"${STAR_VLA_PYTHON}" examples/calvin/scripts/stamp_awac_success_labels.py \
  --dataset-root "${AWAC_DATASET_ROOT}" \
  --all-success-demo \
  --manifest-jsonl "$SUCCESS_OUT/success_stamp_manifest_write.jsonl" \
  --write \
  > "$SUCCESS_OUT/success_stamp_write.log"

cat "$SUCCESS_OUT/success_stamp_write.log"
```

如果不能确认全是成功 demo，不要执行 `--all-success-demo --write`；必须先拿到源成功标签。

## 4. 写入 AWAC reward/done

先 dry-run：

```bash
"${STAR_VLA_PYTHON}" examples/calvin/scripts/prepare_awac_rewards.py \
  --dataset_root "${AWAC_DATASET_ROOT}" \
  --action_horizon 8 \
  --gamma 0.996 \
  --dry_run
```

dry-run 通过后正式写入：

```bash
"${STAR_VLA_PYTHON}" examples/calvin/scripts/prepare_awac_rewards.py \
  --dataset_root "${AWAC_DATASET_ROOT}" \
  --action_horizon 8 \
  --gamma 0.996
```

## 5. 跑 PI-State rollout 诊断

rollout 不是重放训练集；它是在 CALVIN 环境中评估当前 policy。这里先跑小量诊断，确认 state 和 rollout 写盘链路。

```bash
conda activate starVLA_qwen35

export EVAL_BATCH_ROOT="${PI_STATE_BATCH_ROOT}"
export EVAL_OUTPUT_ROOT="${PROJECT_ROOT}/logs/calvin_eval_rollout_20260519_h200_server1_pi_state_30k_v1"
export EXPECTED_JOBS=$'server1_pi_state/qwen35_0p8b\nserver1_pi_state/qwen35_4b\nserver1_pi_state/qwen35_9b'

export EVAL_GPU=0
export EVAL_PORT=6200
export NUM_SEQUENCES=20
export FAIL_FAST=0
export REQUIRE_MP4=1
export SEND_STATE_TO_POLICY=1
export WRITE_ROLLOUT_LEROBOT=1
export WRITE_ROLLOUT_VIDEOS=1
export UNNORM_KEY=franka

export H200_CALVIN_EVAL_DATASET_PATH="${PROJECT_ROOT}/calvin/dataset/calvin_debug_dataset"
export EVAL_SEQUENCES_PATH=examples/calvin/eval_files/eval_sequences.json

bash examples/calvin/eval_files/eval_h200_qwen_mix_latest.sh
```

诊断通过后，可把 `NUM_SEQUENCES` 提高到正式评估规模。

## 6. 再跑最终 readiness

```bash
export OUT="logs/awac_readiness_final_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

"${STAR_VLA_PYTHON}" examples/calvin/scripts/inspect_awac_readiness.py \
  --batch-root "${PI_STATE_BATCH_ROOT}" \
  --data-root "${H200_CALVIN_DATA_ROOT}" \
  --dataset-name "${H200_CALVIN_DATA_NAME}" \
  --data-mix "${H200_CALVIN_DATA_MIX}" \
  --expected-step 30000 \
  --job-expected-step server1_pi_state/qwen35_9b=25000 \
  --rollout-root "${PROJECT_ROOT}/logs/calvin_eval_rollout_20260519_h200_server1_pi_state_30k_v1" \
  --require-rollout \
  --json-out "$OUT/awac_readiness.json" \
  > "$OUT/awac_readiness.md"

echo "inspect_status=$?"
sed -n '1,320p' "$OUT/awac_readiness.md"
```

只有 `overall: PASS` 后，才进入 `run_calvin_awac_critic.sh` / `run_calvin_awac_actor.sh`。

## 7. 后训练入口示例

先从单模型开始，不要三模型一起开，避免排错时互相污染。

### 0.8B critic

```bash
export CUDA_VISIBLE_DEVICES=0
export num_processes=1
export base_vlm="${PROJECT_ROOT}/playground/Pretrained_models/Qwen3.5-0.8B"
export bc_checkpoint="${PI_STATE_CKPT_0P8B}"
export calvin_data_root="${H200_CALVIN_DATA_ROOT}"
export data_mix="${H200_CALVIN_DATA_MIX}"
export include_state=true
export state_dim=8
export run_root_dir="${PROJECT_ROOT}/logs/20260520_awac_pi_state"
export run_id="awac_critic_pi_state_qwen35_0p8b_bc30k"

bash examples/calvin/train_files/run_calvin_awac_critic.sh
```

### 9B critic，明确 25k

```bash
export CUDA_VISIBLE_DEVICES=3,4,5,6,7
export num_processes=5
export base_vlm="${PROJECT_ROOT}/playground/Pretrained_models/Qwen3.5-9B"
export bc_checkpoint="${PI_STATE_CKPT_9B}"
export calvin_data_root="${H200_CALVIN_DATA_ROOT}"
export data_mix="${H200_CALVIN_DATA_MIX}"
export include_state=true
export state_dim=8
export run_root_dir="${PROJECT_ROOT}/logs/20260520_awac_pi_state"
export run_id="awac_critic_pi_state_qwen35_9b_bc25k"

bash examples/calvin/train_files/run_calvin_awac_critic.sh
```

