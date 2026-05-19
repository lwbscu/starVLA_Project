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
Use ``--output_dataset_root`` when the source dataset is public or should remain
byte-for-byte unchanged.

Example::

  python examples/calvin/scripts/prepare_awac_rewards.py \\
    --dataset_root /path/to/calvin_task_ABC_D \\
    --action_horizon 8 \\
    --gamma 0.996
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_AWAC_REWARD_PREPROCESSING_PATH = _REPO_ROOT / "starVLA/dataloader/awac_reward_preprocessing.py"


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
    parser.add_argument(
        "--output_dataset_root",
        type=str,
        default=None,
        help=(
            "Optional LeRobot dataset root for processed output. Sidecar files are copied from --dataset_root "
            "and processed parquets are written to this root instead of mutating the source dataset."
        ),
    )
    parser.add_argument(
        "--allow_in_place",
        action="store_true",
        help="Allow writing into --dataset_root when --output_dataset_root is omitted or identical.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Scan only; do not write parquet files.")
    return parser.parse_args()


def same_path(left: Path, right: Path) -> bool:
    return left.resolve(strict=False) == right.resolve(strict=False)


def output_path_for_parquet(input_root: Path, output_root: Path, parquet_path: Path) -> Path:
    try:
        rel_to_data = parquet_path.relative_to(input_root / "data")
        return output_root / "data" / rel_to_data
    except ValueError:
        return output_root / parquet_path.relative_to(input_root)


def copy_sidecar_files(input_root: Path, output_root: Path) -> int:
    copied = 0
    for source_path in input_root.rglob("*"):
        if source_path.is_dir():
            continue
        if source_path.suffix == ".parquet":
            continue
        rel_path = source_path.relative_to(input_root)
        dest_path = output_root / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)
        copied += 1
    return copied


def load_awac_reward_preprocessing() -> Any:
    spec = importlib.util.spec_from_file_location("awac_reward_preprocessing", _AWAC_REWARD_PREPROCESSING_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load AWAC reward preprocessing module: {_AWAC_REWARD_PREPROCESSING_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preprocess_dataset_to_output(dataset_root: Path, output_root: Path, cfg: Any, dry_run: bool, reward_module: Any) -> dict:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("prepare_awac_rewards.py requires pandas with parquet support.") from exc

    paths = reward_module.find_episode_parquet_files(dataset_root)
    if not paths:
        raise FileNotFoundError(f"No parquet files found under {dataset_root}")

    stats = {"episodes": 0, "frames": 0, "successes": 0, "sidecars_copied": 0}
    if not dry_run:
        stats["sidecars_copied"] = copy_sidecar_files(dataset_root, output_root)

    for parquet_path in paths:
        df = pd.read_parquet(parquet_path)
        episode_success = reward_module.infer_episode_success(
            df,
            success_column=cfg.success_column,
            assume_success_if_missing=cfg.assume_success_if_missing,
        )
        processed = reward_module.process_episode(df, cfg)
        stats["episodes"] += 1
        stats["frames"] += len(processed)
        stats["successes"] += int(episode_success)
        if not dry_run:
            out_path = output_path_for_parquet(dataset_root, output_root, parquet_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = out_path.with_name(f".{out_path.name}.tmp")
            processed.to_parquet(tmp_path, index=False)
            tmp_path.replace(out_path)
    return stats


def main() -> None:
    args = parse_args()
    reward_module = load_awac_reward_preprocessing()
    cfg = reward_module.AWACRewardConfig(
        action_horizon=args.action_horizon,
        gamma=args.gamma,
        step_penalty=args.step_penalty,
        success_reward=args.success_reward,
        failure_reward=args.failure_reward,
        success_column=args.success_column,
        assume_success_if_missing=args.assume_success_if_missing,
    )

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_dataset_root) if args.output_dataset_root else dataset_root
    in_place = same_path(dataset_root, output_root)
    if not args.dry_run and in_place and not args.allow_in_place:
        raise SystemExit(
            "Refusing in-place parquet writes. Pass --output_dataset_root for an AWAC working copy, "
            "or explicitly pass --allow_in_place when you are sure the dataset is not public/official."
        )

    if in_place:
        stats = reward_module.preprocess_dataset(
            dataset_root=dataset_root,
            cfg=cfg,
            dry_run=args.dry_run,
        )
        stats["sidecars_copied"] = 0
    else:
        stats = preprocess_dataset_to_output(
            dataset_root=dataset_root,
            output_root=output_root,
            cfg=cfg,
            dry_run=args.dry_run,
            reward_module=reward_module,
        )
    mode = "dry-run" if args.dry_run else "written"
    print(
        f"[prepare_awac_rewards] {mode}: "
        f"episodes={stats['episodes']}, frames={stats['frames']}, "
        f"successes={stats['successes']}, H={cfg.action_horizon}, gamma={cfg.gamma}, "
        f"output_root={output_root}, sidecars_copied={stats['sidecars_copied']}"
    )


if __name__ == "__main__":
    main()
