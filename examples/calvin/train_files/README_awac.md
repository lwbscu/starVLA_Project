# Calvin AWAC Post-Training

Offline AWAC post-training for StarVLA `QwenPI` policies on Calvin LeRobot data.

**H200 mixed training (expert + rollout 1:1)** — canonical entry points:

```bash
bash examples/calvin/train_files/h200_awac_critic_mixed_oneclick.sh
bash examples/calvin/train_files/h200_awac_actor_mixed_oneclick.sh
```

See `README_后训练/01_H200_AWAC后训练运行命令.md` (Chinese) for defaults aligned with those scripts. Algorithm contract: `critic.md` §2.

## Prerequisites

1. **BC checkpoint** from `train_starvla.py` (Phase 0).
2. **LeRobot Calvin dataset** with terminal `success` (and optional `done`) in each episode parquet.
3. Copy `modality.json` (or `modality_awac.json`) into `<dataset>/meta/modality.json`.
4. Provide AWAC rewards either by **offline preprocessing** or by the explicit
   **read-only on-the-fly mode** below.

### Step 0A — Reward / done preprocessing

Raw Calvin LeRobot data usually only has **episode-end `success`**. Run:

```bash
python examples/calvin/scripts/prepare_awac_rewards.py \
  --dataset_root /path/to/calvin_task_ABC_D \
  --output_dataset_root /path/to/calvin_task_ABC_D_awac_work \
  --action_horizon 8 \
  --gamma 0.996
```

Use `--output_dataset_root` for public or competition datasets. In-place writes
are intentionally refused unless `--allow_in_place` is passed explicitly.

This writes into each episode parquet:

| Column | Meaning |
|--------|---------|
| `step_reward` | Per-frame reward before discounting (non-terminal: `-1`, last frame: `0` or `-3000`) |
| `reward` | Discounted chunk return \(R_{chunk}(t)=\sum_{k=0}^{\min(H-1,T-1-t)} \gamma^k r_{t+k}\) |
| `done` | `True` on the **last H frames** of the episode (aligned with critic bootstrap) |

Defaults: `step_penalty=-1`, `success_reward=0`, `failure_reward=-3000`, `H=8`, `γ=0.996`.

Implementation: [`starVLA/dataloader/awac_reward_preprocessing.py`](../../../starVLA/dataloader/awac_reward_preprocessing.py).

The AWAC dataloader reads `reward` at index `t` as **precomputed** \(R_{chunk}(t)\) (no second sum over the chunk). Set `datasets.awac_data.reward_is_chunk_return: false` only if `reward` stores per-step values instead.

### Step 0B — Read-only on-the-fly rewards

If the Calvin dataset is too large to copy, keep the public dataset read-only and
make the dataloader compute `reward` and `done` at load time:

```bash
export calvin_data_root=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d
export data_mix=calvin_abc_d_h200
export compute_rewards_on_the_fly=true
export assume_success_if_missing=false
```

H200 mixed oneclick uses `assume_success_if_missing=false` (real `success` in parquet). Use `assume_success_if_missing=true` only for verified expert demos where every
episode is known to be successful. Otherwise, keep it false; missing `success`
will raise an error instead of silently fabricating labels.

This mode uses the same defaults as preprocessing (`H=8`, `gamma=0.996`,
`step_penalty=-1`, `success_reward=0`, `failure_reward=-3000`) and does not
modify parquet files.

## Training (two phases)

### Phase 1 — Critic (Q only, TD loss)

```bash
bash examples/calvin/train_files/run_calvin_awac_critic.sh
```

Edit the script:

- `bc_checkpoint`: BC actor weights (copies visual encoder into critic)
- `calvin_data_root`, `data_mix`

Output: `logs/<run_id>/checkpoints/steps_*_critic.pt`

TensorBoard (critic phase, rank 0):

```bash
tensorboard --logdir logs/<run_id>/tensorboard --port 6006
```

Scalars: `critic_loss`, `q_mean`, `target_mean`, `learning_rate`.

### Phase 2 — Actor (weighted flow matching)

```bash
bash examples/calvin/train_files/run_calvin_awac_actor.sh
```

Edit:

- `bc_checkpoint`: same BC checkpoint (actor init)
- `critic_checkpoint`: Phase 1 `*_critic.pt`

Output: `logs/<run_id>/checkpoints/steps_*_pytorch_model.pt` (compatible with Calvin eval).

## Key hyperparameters (`starvla_awac_calvin.yaml`)

| Key | Default | Meaning |
|-----|---------|---------|
| `awac.gamma` | 0.996 | Discount |
| `awac.num_q_heads` | 2 | Twin Q heads (`E`) |
| `awac.hidden_dim` | 512 | Critic token dim `D` |
| `awac.awac_lambda` | 0.5 | AWAC temperature |
| `awac.awac_weight_max` | 20.0 | Weight clip |

TD bootstrap: `target = r + (1-done) * gamma^H * min_E Q_target(s', a')`.

Actor weights: `exp((min_E Q(s, a_pi) - min_E Q(s, a_data)) / lambda)`, where `a_pi` from a **frozen actor_pi snapshot** (BC copy) and training updates a separate **trainable actor**. No `V` network.

Checkpoints default to `logs/<run_id>/checkpoints/` (`run_root_dir` in yaml or shell).

## Smoke tests

```bash
# Critic — 10 steps, batch 1
accelerate launch starVLA/training/train_awac_critic.py \
  --config_yaml examples/calvin/train_files/starvla_awac_calvin.yaml \
  --trainer.critic_max_train_steps 10 \
  --datasets.awac_data.per_device_batch_size 1

# Actor — 10 steps (must use QwenPI_AWAC)
accelerate launch starVLA/training/train_awac_actor.py \
  --config_yaml examples/calvin/train_files/starvla_awac_calvin.yaml \
  --framework.name QwenPI_AWAC \
  --trainer.actor_max_train_steps 10 \
  --trainer.awac_critic_checkpoint <critic.pt> \
  --trainer.pretrained_checkpoint <bc.pt> \
  --datasets.awac_data.per_device_batch_size 1
```

Check logs for:

- `[AWAC] ... valid transitions` ratio
- `q_mean`, `target_mean`, `critic_loss` finite
- `awac_weight_mean` > 0 in actor phase

## File map

| File | Role |
|------|------|
| `starVLA/dataloader/awac_transition_dataset.py` | `(s,a,r,s',a',done)` loader |
| `starVLA/model/modules/critic/awac_q_critic.py` | Q network |
| `starVLA/model/framework/VLM4A/QwenPI_awac.py` | Weighted actor |
| `starVLA/training/train_awac_critic.py` | Phase 1 |
| `starVLA/training/train_awac_actor.py` | Phase 2 |
