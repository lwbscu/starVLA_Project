# H200 AWAC 后训练准备命令行

本文档用于 **readiness、rollout 诊断、环境变量** 准备。混训 **Critic / Actor 训练** 请以两个 oneclick 为准（与 `README_后训练/01_H200_AWAC后训练运行命令.md` 一致）：

```bash
bash examples/calvin/train_files/h200_awac_critic_mixed_oneclick.sh
bash examples/calvin/train_files/h200_awac_actor_mixed_oneclick.sh
```

算法契约见 `examples/calvin/train_files/critic.md` §2。核心原则：不伪造成功，不绕过 `H=8 / gamma=0.996 / reward=chunk return / done=最后 H 帧` 的契约。

**数据安全原则**：不修改 `/inspire/.../public` 下的官方/比赛数据。`success`、`step_reward`、`reward`、`done` 只写入个人目录下的 AWAC 工作副本。

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
export ROLLOUT_PARQUET_PYTHON="${STAR_VLA_PYTHON}"

export H200_CALVIN_DATA_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d
export H200_CALVIN_DATA_NAME=calvin_task_ABC_D
export H200_CALVIN_DATA_MIX=calvin_abc_d_h200
export AWAC_SOURCE_DATA_ROOT="${H200_CALVIN_DATA_ROOT}"
export AWAC_SOURCE_DATASET_ROOT="${AWAC_SOURCE_DATA_ROOT}/${H200_CALVIN_DATA_NAME}"
export AWAC_WORK_DATA_ROOT="${PROJECT_ROOT}/awac_datasets/calvin_abc_d_awac_h8_g0996"
export AWAC_WORK_DATASET_ROOT="${AWAC_WORK_DATA_ROOT}/${H200_CALVIN_DATA_NAME}"

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
  --data-root "${AWAC_SOURCE_DATA_ROOT}" \
  --dataset-name "${H200_CALVIN_DATA_NAME}" \
  --data-mix "${H200_CALVIN_DATA_MIX}" \
  --expected-step 30000 \
  --job-expected-step server1_pi_state/qwen35_9b=25000 \
  --json-out "$OUT/awac_readiness.json" \
  > "$OUT/awac_readiness.md"

echo "inspect_status=$?"
sed -n '1,260p' "$OUT/awac_readiness.md"
```

此时如果只剩 `success labels` 和 `AWAC reward preprocessing` 失败，进入下一节。这里检查的是官方源数据，只读，不写。

## 3. 补齐 success 标签

高标准做法有两种，只能选一种：

1. 如果有带 `success` 的源数据集，使用 `--source-dataset-root` 按 episode 相对路径复制。
2. 如果确认 `calvin_task_ABC_D` 是成功 expert demonstration 数据，显式使用 `--all-success-demo`，脚本会写成“仅末帧 success=True，其余帧 False”。

当前已确认训练 demo 都是成功轨迹，因此可以使用 `--all-success-demo`，但仍然只写到 `${AWAC_WORK_DATASET_ROOT}` 工作副本。

先 dry-run：

```bash
export SUCCESS_OUT="logs/awac_success_stamp_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SUCCESS_OUT"

"${STAR_VLA_PYTHON}" examples/calvin/scripts/stamp_awac_success_labels.py \
  --dataset-root "${AWAC_SOURCE_DATASET_ROOT}" \
  --output-dataset-root "${AWAC_WORK_DATASET_ROOT}" \
  --all-success-demo \
  --manifest-jsonl "$SUCCESS_OUT/success_stamp_manifest.jsonl" \
  > "$SUCCESS_OUT/success_stamp_dry_run.log"

cat "$SUCCESS_OUT/success_stamp_dry_run.log"
```

确认 `failures=0` 后才写入：

```bash
"${STAR_VLA_PYTHON}" examples/calvin/scripts/stamp_awac_success_labels.py \
  --dataset-root "${AWAC_SOURCE_DATASET_ROOT}" \
  --output-dataset-root "${AWAC_WORK_DATASET_ROOT}" \
  --all-success-demo \
  --manifest-jsonl "$SUCCESS_OUT/success_stamp_manifest_write.jsonl" \
  --write \
  > "$SUCCESS_OUT/success_stamp_write.log"

cat "$SUCCESS_OUT/success_stamp_write.log"
```

如果不能确认全是成功 demo，不要执行 `--all-success-demo --write`；必须先拿到源成功标签。执行后，官方源数据 `${AWAC_SOURCE_DATASET_ROOT}` 不变，后续只使用 `${AWAC_WORK_DATASET_ROOT}`。

### 3.1 磁盘不足：不复制数据，训练时即时计算 reward/done

如果个人目录放不下完整 parquet 工作副本，可以不执行第 3 节和第 4 节的写盘预处理，改用 dataloader 的显式即时计算模式：

- 源数据仍从 public 目录只读加载。
- 不向 parquet 写 `success`、`step_reward`、`reward`、`done`。
- 训练时设置 `compute_rewards_on_the_fly=true`，按同一套 `H=8 / gamma=0.996 / reward=chunk return / done=最后 H 帧` 逻辑在内存里计算。
- 如果源数据没有 `success` 列，只有在确认全是成功 expert demo 时，才设置 `assume_success_if_missing=true`。

只读 readiness 检查：

```bash
export OUT="logs/awac_readiness_onfly_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

"${STAR_VLA_PYTHON}" examples/calvin/scripts/inspect_awac_readiness.py \
  --batch-root "${PI_STATE_BATCH_ROOT}" \
  --data-root "${AWAC_SOURCE_DATA_ROOT}" \
  --dataset-name "${H200_CALVIN_DATA_NAME}" \
  --data-mix "${H200_CALVIN_DATA_MIX}" \
  --expected-step 30000 \
  --job-expected-step server1_pi_state/qwen35_9b=25000 \
  --allow-on-the-fly-rewards \
  --assume-success-if-missing \
  --json-out "$OUT/awac_readiness.json" \
  > "$OUT/awac_readiness.md"

echo "inspect_status=$?"
sed -n '1,300p' "$OUT/awac_readiness.md"
```

后续训练时用 public 源数据根目录，并显式打开即时计算：

```bash
export calvin_data_root="${AWAC_SOURCE_DATA_ROOT}"
export data_mix="${H200_CALVIN_DATA_MIX}"
export compute_rewards_on_the_fly=true
export assume_success_if_missing=true
```

这条路线不修改官方数据，也不创建完整 parquet 副本；代价是每次 dataloader 读取 transition 时会多做少量 reward/done 计算。

## 4. 写入 AWAC reward/done

先 dry-run：

```bash
"${STAR_VLA_PYTHON}" examples/calvin/scripts/prepare_awac_rewards.py \
  --dataset_root "${AWAC_WORK_DATASET_ROOT}" \
  --action_horizon 8 \
  --gamma 0.996 \
  --dry_run
```

dry-run 通过后正式写入：

```bash
"${STAR_VLA_PYTHON}" examples/calvin/scripts/prepare_awac_rewards.py \
  --dataset_root "${AWAC_WORK_DATASET_ROOT}" \
  --action_horizon 8 \
  --gamma 0.996 \
  --allow_in_place
```

这里的 in-place 只发生在个人 AWAC 工作副本 `${AWAC_WORK_DATASET_ROOT}`，不会改 public 官方数据。

## 5. 跑 PI-State rollout 诊断

rollout 不是重放训练集；它是在 CALVIN 环境中评估当前 policy。这里先跑小量诊断，确认 state 和 rollout 写盘链路。

### 5.1 并发吃满 8 卡生成大量 rollout

模型服务当前不是 tensor parallel；单个 server 传 `CUDA_VISIBLE_DEVICES=3,4,5,6,7` 不会自动把 9B 切到 5 张卡。要真正吃满 8 张 H200，使用 8 个单卡 rollout worker。

当前 rollout 观察显示 Qwen3.5-4B 成功轨迹明显更多，因此默认把 8 张卡全部给 4B，优先生成可用于后训练的成功/失败混合轨迹。0.8B/9B 的失败轨迹保留作诊断，不作为主力 rollout 生产源。

每个 worker 使用不同 `EVAL_SEQUENCE_INDEX_OFFSET`，同一模型内部不重复采样。默认生成 500 条 4B eval sequence 的 rollout；如需更大，调 `ROLLOUT_NUM_SEQUENCES_PER_MODEL`。

```bash
conda activate starVLA_qwen35

export PROJECT_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project
export CONDA_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/26220216/miniconda3
export STAR_VLA_PYTHON="${CONDA_ROOT}/envs/starVLA_qwen35/bin/python"
export CALVIN_PYTHON="${CONDA_ROOT}/envs/calvin/bin/python"
export ROLLOUT_PARQUET_PYTHON="${STAR_VLA_PYTHON}"

export EVAL_BATCH_ROOT="${PROJECT_ROOT}/logs/20260519_h200_server1_pi_state_30k_v1"
export ROLLOUT_OUTPUT_ROOT="${PROJECT_ROOT}/results/rollout"
export ROLLOUT_RUN_NAME=pi_state_4b_awac_parallel_500seq_v1
export ROLLOUT_NUM_SEQUENCES_PER_MODEL=500

export H200_CALVIN_EVAL_DATASET_PATH="${PROJECT_ROOT}/calvin/dataset/calvin_debug_dataset"
export EVAL_SEQUENCES_PATH=examples/calvin/eval_files/eval_sequences.json

# 大量 rollout 默认不写 mp4，避免视频 IO 拖慢。需要诊断视频时小量单独跑。
export DEBUG_MP4=0
export REQUIRE_MP4=0
export WRITE_ROLLOUT_VIDEOS=0

bash examples/calvin/eval_files/eval_h200_pi_state_rollout_parallel_8gpu.sh
```

输出根目录：

```bash
${PROJECT_ROOT}/results/rollout/pi_state_4b_awac_parallel_500seq_v1
```

每个 worker 的 LeRobot rollout 在：

```bash
${PROJECT_ROOT}/results/rollout/pi_state_4b_awac_parallel_500seq_v1/qwen35_4b/shard_*/server1_pi_state/qwen35_4b/steps_30000/eval/rollout_lerobot
```

汇总文件：

```bash
cat "${PROJECT_ROOT}/results/rollout/pi_state_4b_awac_parallel_500seq_v1/manifest.tsv"
cat "${PROJECT_ROOT}/results/rollout/pi_state_4b_awac_parallel_500seq_v1/summary.tsv"
```

核查 rollout 是否完整：

```bash
export RUN_ROOT="${PROJECT_ROOT}/results/rollout/pi_state_4b_awac_parallel_500seq_v1"

cat "${RUN_ROOT}/manifest.tsv"
cat "${RUN_ROOT}/summary.tsv"

find "${RUN_ROOT}" -path "*/rollout_lerobot/meta/info.json" -print | sort
find "${RUN_ROOT}" -path "*/rollout_lerobot/data/*.parquet" -print | wc -l

"${STAR_VLA_PYTHON}" - "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
infos = sorted(root.glob("**/rollout_lerobot/meta/info.json"))
if not infos:
    raise SystemExit(f"no rollout_lerobot/meta/info.json under {root}")
total_episodes = 0
total_frames = 0
bad = []
for info_path in infos:
    info = json.loads(info_path.read_text())
    ds_root = info_path.parents[1]
    parquets = sorted(ds_root.glob("data/**/*.parquet"))
    total_episodes += int(info.get("total_episodes", 0))
    total_frames += int(info.get("total_frames", 0))
    if len(parquets) != int(info.get("total_episodes", 0)):
        bad.append(f"{ds_root}: parquet_count={len(parquets)} total_episodes={info.get('total_episodes')}")
    for pq in parquets[:3]:
        df = pd.read_parquet(pq)
        missing = {"state", "actions", "success", "done", "episode_success"} - set(df.columns)
        if missing:
            bad.append(f"{pq}: missing={sorted(missing)}")
        if df["state"].iloc[0].shape[0] != 8:
            bad.append(f"{pq}: state dim != 8")
        if df["actions"].iloc[0].shape[0] != 7:
            bad.append(f"{pq}: action dim != 7")
if bad:
    raise SystemExit("\n".join(bad))
print(f"rollout_check=ok datasets={len(infos)} total_episodes={total_episodes} total_frames={total_frames}")
PY
```

如果后续还想复查三模型分布，可覆盖 `ROLLOUT_WORKER_LAYOUT`：

```bash
export ROLLOUT_WORKER_LAYOUT=$'qwen35_0p8b server1_pi_state/qwen35_0p8b 0 6200\nqwen35_4b server1_pi_state/qwen35_4b 1 6210\nqwen35_4b server1_pi_state/qwen35_4b 2 6211\nqwen35_9b server1_pi_state/qwen35_9b 3 6220\nqwen35_9b server1_pi_state/qwen35_9b 4 6221\nqwen35_9b server1_pi_state/qwen35_9b 5 6222\nqwen35_9b server1_pi_state/qwen35_9b 6 6223\nqwen35_9b server1_pi_state/qwen35_9b 7 6224'
```

### 5.2 单进程小量诊断

如果需要 mp4 诊断，先用下面的小量串行命令，确认一条链路可视化正常。

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
export ROLLOUT_PARQUET_PYTHON="${STAR_VLA_PYTHON}"
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
  --data-root "${AWAC_WORK_DATA_ROOT}" \
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

只有 `overall: PASS` 后，才进入 AWAC 训练。混训入口为 **§ 上方 oneclick**，不是本节手写 `run_calvin_awac_*` 单源示例。

## 7. 后训练入口示例（备选：单源 / 多模型排查）

> **混训主线不要用本节**；见文档开头两个 `h200_awac_*_mixed_oneclick.sh`。

先从单模型开始，不要三模型一起开，避免排错时互相污染。

### 0.8B critic

```bash
export CUDA_VISIBLE_DEVICES=0
export num_processes=1
export base_vlm="${PROJECT_ROOT}/playground/Pretrained_models/Qwen3.5-0.8B"
export bc_checkpoint="${PI_STATE_CKPT_0P8B}"
export calvin_data_root="${AWAC_WORK_DATA_ROOT}"
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
export calvin_data_root="${AWAC_WORK_DATA_ROOT}"
export data_mix="${H200_CALVIN_DATA_MIX}"
export include_state=true
export state_dim=8
export run_root_dir="${PROJECT_ROOT}/logs/20260520_awac_pi_state"
export run_id="awac_critic_pi_state_qwen35_9b_bc25k"

bash examples/calvin/train_files/run_calvin_awac_critic.sh
```
