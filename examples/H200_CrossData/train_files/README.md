# H200 Cross-Dataset PI Pretraining

This launcher targets the shared dataset scan result under:

```bash
/inspire/qb-ilm2/project/26summer-camp-10/public/three/dataset
```

Default mix:

- `h200_three_libero_only`: `libero_object`, `libero_goal`, `libero_spatial`, `libero_10`
- All four are LeRobot v2.1, `franka`, and include `meta/modality.json`.

Experimental mixes are registered but intentionally fail the launcher preflight until verified:

- `h200_three_oxe_experimental`: `bridge`, `fractal`
- `h200_three_all_experimental`: LIBERO + Bridge + Fractal

The current scan found `bridge` and `fractal` as LeRobot v2.0 roots without `meta/modality.json`; the StarVLA LeRobot loader requires that file, so the launcher refuses to train on them by default.

## Launch

```bash
cd /inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project
export PATH="/inspire/qb-ilm2/project/26summer-camp-10/26220216/miniconda3/bin:$PATH"
source /inspire/qb-ilm2/project/26summer-camp-10/26220216/miniconda3/etc/profile.d/conda.sh
conda activate starVLA_qwen35

git pull

bash examples/H200_CrossData/train_files/run_h200_cross_dataset_pi_pretrain.sh
```

The launcher starts three QwenPI jobs:

| Model | GPUs | Processes | Port |
| --- | --- | --- | --- |
| Qwen3.5-0.8B | `0` | `1` | `33100` |
| Qwen3.5-4B | `1,2` | `2` | `33110` |
| Qwen3.5-9B | `3,4,5,6,7` | `5` | `33120` |

## Common Overrides

```bash
export CROSS_DATA_MIX=h200_three_libero_only
export MAX_TRAIN_STEPS=30000
export SAVE_INTERVAL=1000
export CHECKPOINT_KEEP_LATEST=1
export CHECKPOINT_KEEP_STEPS=10000,20000,30000
export BATCH_NAME=20260519_h200_three_libero_pi_30k_v1

bash examples/H200_CrossData/train_files/run_h200_cross_dataset_pi_pretrain.sh
```

Dry-run command generation:

```bash
export DRY_RUN=1
bash examples/H200_CrossData/train_files/run_h200_cross_dataset_pi_pretrain.sh
```
