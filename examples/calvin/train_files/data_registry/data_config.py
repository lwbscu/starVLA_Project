"""CALVIN dataset registry entries for StarVLA smoke and training runs."""

DATASET_NAMED_MIXTURES = {
    "calvin_abc_d": [
        ("calvin_abc_d_lerobot_v2.1", 1.0, "libero_franka"),
    ],
    "calvin_abc_d_h200": [
        ("calvin_task_ABC_D", 1.0, "libero_franka"),
    ],
    "calvin_hlx_h200": [
        ("hlx/calvin_task_ABC_D", 1.0, "libero_franka"),
    ],
    "calvin_rollout_h200": [
        ("rollout_lerobot", 1.0, "libero_franka"),
    ],
    # Expert demos + policy rollout (1:1 per-sample when balance_datasets=true in awac_data).
    "calvin_awac_mixed_h200": [
        ("calvin_task_ABC_D", 1.0, "libero_franka"),
        ("rollout_lerobot", 1.0, "libero_franka"),
    ],
}
