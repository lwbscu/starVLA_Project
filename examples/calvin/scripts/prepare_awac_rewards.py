#!/usr/bin/env python3
# Copyright 2025 starVLA community. All rights reserved.
"""
Prepare ``reward`` and ``done`` columns for Calvin AWAC training.

Raw LeRobot Calvin parquets typically only have terminal ``success`` (and optional
``done``). This script writes:

  - ``step_reward``: per-frame reward before chunk discounting
  - ``reward``: discounted chunk return R_chunk(t) aligned with action horizon H
  - ``done``: True on the last H frames of each episode

Run **before** ``train_awac_critic.py`` / ``train_awac_actor.py``.

Example::

  python examples/calvin/scripts/prepare_awac_rewards.py \\
    --dataset_root /path/to/calvin_task_ABC_D \\
    --action_horizon 8 \\
    --gamma 0.996
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from starVLA.dataloader.awac_reward_preprocessing import AWACRewardConfig, preprocess_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare AWAC reward/done columns for Calvin LeRobot data.")
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="LeRobot dataset directory (contains data/**/*.parquet).",
    )
    parser.add_argument("--action_horizon", type=int, default=8, help="Action chunk length H.")
    parser.add_argument("--gamma", type=float, default=0.996, help="Discount factor for chunk return.")
    parser.add_argument("--step_penalty", type=float, default=-1.0, help="Per-step penalty for non-terminal frames.")
    parser.add_argument("--success_reward", type=float, default=0.0, help="Terminal reward when episode succeeds.")
    parser.add_argument("--failure_reward", type=float, default=-3000.0, help="Terminal reward when episode fails.")
    parser.add_argument("--success_column", type=str, default="success", help="Source success column in parquet.")
    parser.add_argument(
        "--assume_success_if_missing",
        action="store_true",
        help="Treat episodes as successful when success column is absent (demo data).",
    )
    parser.add_argument("--dry_run", action="store_true", help="Scan only; do not write parquet files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AWACRewardConfig(
        action_horizon=args.action_horizon,
        gamma=args.gamma,
        step_penalty=args.step_penalty,
        success_reward=args.success_reward,
        failure_reward=args.failure_reward,
        success_column=args.success_column,
        assume_success_if_missing=args.assume_success_if_missing,
    )

    stats = preprocess_dataset(
        dataset_root=args.dataset_root,
        cfg=cfg,
        dry_run=args.dry_run,
    )
    mode = "dry-run" if args.dry_run else "written"
    print(
        f"[prepare_awac_rewards] {mode}: "
        f"episodes={stats['episodes']}, frames={stats['frames']}, "
        f"successes={stats['successes']}, H={cfg.action_horizon}, gamma={cfg.gamma}"
    )


if __name__ == "__main__":
    main()
