# Calvin AWAC Post-Training

Offline AWAC post-training for StarVLA `QwenPI` policies on Calvin LeRobot data.

## Prerequisites

1. **BC checkpoint** from `train_starvla.py` (Phase 0).
2. **LeRobot Calvin dataset** with terminal `success` (and optional `done`) in each episode parquet.
3. Copy `modality.json` (or `modality_awac.json`) into `<dataset>/meta/modality.json`.
4. **Run reward preprocessing** (below) before AWAC training.

### Step 0 — Reward / done preprocessing (required)

Raw Calvin LeRobot data usually only has **episode-end `success`**. Run:

```bash
python examples/calvin/scripts/prepare_awac_rewards.py \
  --dataset_root /path/to/calvin_task_ABC_D \
  --action_horizon 8 \
  --gamma 0.996
```

This writes into each episode parquet:

| Column | Meaning |
|--------|---------|
| `step_reward` | Per-frame reward before discounting (non-terminal: `-1`, last frame: `0` or `-3000`) |
| `reward` | Discounted chunk return \(R_{chunk}(t)=\sum_{k=0}^{\min(H-1,T-1-t)} \gamma^k r_{t+k}\) |
| `done` | `True` on the **last H frames** of the episode (aligned with critic bootstrap) |

Defaults: `step_penalty=-1`, `success_reward=0`, `failure_reward=-3000`, `H=8`, `γ=0.996`.

Implementation: [`starVLA/dataloader/awac_reward_preprocessing.py`](../../../starVLA/dataloader/awac_reward_preprocessing.py).

The AWAC dataloader reads `reward` at index `t` as **precomputed** \(R_{chunk}(t)\) (no second sum over the chunk). Set `datasets.awac_data.reward_is_chunk_return: false` only if `reward` stores per-step values instead.

## Training (two phases)

### Phase 1 — Critic (Q + expectile V)

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

Scalars: `critic_loss`, `value_loss`, `total_loss`, `q_mean`, `target_mean`, `v_mean`, `learning_rate`.

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
| `awac.expectile_tau` | 0.7 | Value expectile |

TD bootstrap: `target = r + (1-done) * gamma^H * min_E Q_target(s', a')`.

Actor weights: `exp((min_E Q(s,a) - V(s)) / lambda)`, clipped at `awac_weight_max`.

## Smoke tests

```bash
# Critic — 10 steps, batch 1
accelerate launch starVLA/training/train_awac_critic.py \
  --config_yaml examples/calvin/train_files/starvla_awac_calvin.yaml \
  --trainer.critic_max_train_steps 10 \
  --datasets.awac_data.per_device_batch_size 1

# Actor — 10 steps
accelerate launch starVLA/training/train_awac_actor.py \
  --config_yaml examples/calvin/train_files/starvla_awac_calvin.yaml \
  --trainer.actor_max_train_steps 10 \
  --trainer.awac_critic_checkpoint <critic.pt> \
  --trainer.pretrained_checkpoint <bc.pt>
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
| `starVLA/model/modules/critic/awac_v_network.py` | Expectile V |
| `starVLA/model/framework/VLM4A/QwenPI_awac.py` | Weighted actor |
| `starVLA/training/train_awac_critic.py` | Phase 1 |
| `starVLA/training/train_awac_actor.py` | Phase 2 |
