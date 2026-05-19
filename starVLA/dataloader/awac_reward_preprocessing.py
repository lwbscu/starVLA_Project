# Copyright 2025 starVLA community. All rights reserved.
"""
Offline reward / done preprocessing for AWAC (before training).

Pipeline per episode parquet:
  1. compute_step_rewards  — per-frame step rewards from terminal success/failure
  2. compute_discounted_chunk_return — write discounted chunk return to ``reward``
  3. process_episode — mark ``done`` on the last *H* frames (action horizon)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class AWACRewardConfig:
    """Hyper-parameters for AWAC reward preprocessing."""

    action_horizon: int = 8
    gamma: float = 0.996
    step_penalty: float = -1.0
    success_reward: float = 0.0
    failure_reward: float = -3000.0
    success_column: str = "success"
    done_column: str = "done"
    reward_column: str = "reward"
    step_reward_column: str = "step_reward"
    assume_success_if_missing: bool = False


def _to_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.number)):
        return float(value) > 0.5
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value).reshape(-1)
        return bool(arr[-1]) if arr.size else False
    return bool(value)


def infer_episode_success(
    df: pd.DataFrame,
    success_column: str = "success",
    assume_success_if_missing: bool = False,
) -> bool:
    """
    Read episode-level success from the terminal row.

    Raw Calvin/LeRobot parquets often only annotate success on the last frame.
    """
    if success_column not in df.columns:
        if assume_success_if_missing:
            return True
        raise KeyError(
            f"Column `{success_column}` not found in episode parquet. "
            "Add success labels or set assume_success_if_missing=True."
        )
    return _to_bool(df[success_column].iloc[-1])


def compute_step_rewards(
    traj_len: int,
    episode_success: bool,
    step_penalty: float = -1.0,
    success_reward: float = 0.0,
    failure_reward: float = -3000.0,
) -> np.ndarray:
    """
  Assign per-step rewards before chunk discounting.

  - Non-terminal frames: ``step_penalty`` (default -1)
  - Last frame: ``success_reward`` (0) or ``failure_reward`` (-3000)
    """
    if traj_len <= 0:
        return np.zeros(0, dtype=np.float64)

    step_rewards = np.full(traj_len, step_penalty, dtype=np.float64)
    terminal_reward = success_reward if episode_success else failure_reward
    step_rewards[-1] = terminal_reward
    return step_rewards


def compute_discounted_chunk_return(
    step_rewards: np.ndarray,
    action_horizon: int,
    gamma: float,
) -> np.ndarray:
    """
  For each timestep t:

  .. math::

      R_{chunk}(t) = \\sum_{k=0}^{\\min(H-1,\\, T-1-t)} \\gamma^k \\cdot r_{t+k}
    """
    step_rewards = np.asarray(step_rewards, dtype=np.float64)
    traj_len = len(step_rewards)
    if traj_len == 0:
        return step_rewards

    horizon = int(action_horizon)
    chunk_returns = np.zeros(traj_len, dtype=np.float64)
    for t in range(traj_len):
        max_offset = min(horizon - 1, traj_len - 1 - t)
        total = 0.0
        for k in range(max_offset + 1):
            total += (gamma**k) * step_rewards[t + k]
        chunk_returns[t] = total
    return chunk_returns


def compute_done_flags(traj_len: int, action_horizon: int) -> np.ndarray:
    """
  Mark ``done=True`` on the last *H* frames (inclusive) of the episode.

  Aligns with critic bootstrap truncation over the action chunk horizon.
    """
    if traj_len <= 0:
        return np.zeros(0, dtype=bool)

    horizon = int(action_horizon)
    done = np.zeros(traj_len, dtype=bool)
    start = max(0, traj_len - horizon)
    done[start:] = True
    return done


def process_episode(
    df: pd.DataFrame,
    cfg: AWACRewardConfig,
) -> pd.DataFrame:
    """
  Run the full preprocessing pipeline on one episode DataFrame (in-place copy).

  Writes:
    - ``cfg.step_reward_column``: per-step rewards before discounting
    - ``cfg.reward_column``: discounted chunk return R_chunk(t)
    - ``cfg.done_column``: terminal-region done flags
    """
    out = df.copy()
    traj_len = len(out)
    episode_success = infer_episode_success(
        out,
        success_column=cfg.success_column,
        assume_success_if_missing=cfg.assume_success_if_missing,
    )

    step_rewards = compute_step_rewards(
        traj_len=traj_len,
        episode_success=episode_success,
        step_penalty=cfg.step_penalty,
        success_reward=cfg.success_reward,
        failure_reward=cfg.failure_reward,
    )
    chunk_returns = compute_discounted_chunk_return(
        step_rewards,
        action_horizon=cfg.action_horizon,
        gamma=cfg.gamma,
    )
    done_flags = compute_done_flags(traj_len, cfg.action_horizon)

    out[cfg.step_reward_column] = step_rewards.astype(np.float32)
    out[cfg.reward_column] = chunk_returns.astype(np.float32)
    out[cfg.done_column] = done_flags.astype(np.bool_)
    return out


def find_episode_parquet_files(dataset_root: Union[str, Path]) -> list[Path]:
    """Discover LeRobot v2-style episode parquet files under ``data/``."""
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    paths = sorted(root.glob("data/**/*.parquet"))
    if not paths:
        paths = sorted(root.glob("**/*.parquet"))
    return [p for p in paths if "meta" not in p.parts]


def preprocess_dataset(
    dataset_root: Union[str, Path],
    cfg: AWACRewardConfig,
    parquet_paths: Optional[Iterable[Union[str, Path]]] = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """
  Preprocess all episode parquets under a LeRobot dataset root.

  Returns:
      Summary dict with keys ``episodes``, ``frames``, ``successes``.
    """
    root = Path(dataset_root)
    paths = [Path(p) for p in parquet_paths] if parquet_paths is not None else find_episode_parquet_files(root)
    if not paths:
        raise FileNotFoundError(f"No parquet files found under {root}")

    stats = {"episodes": 0, "frames": 0, "successes": 0}
    for parquet_path in paths:
        df = pd.read_parquet(parquet_path)
        episode_success = infer_episode_success(
            df,
            success_column=cfg.success_column,
            assume_success_if_missing=cfg.assume_success_if_missing,
        )
        processed = process_episode(df, cfg)
        stats["episodes"] += 1
        stats["frames"] += len(processed)
        stats["successes"] += int(episode_success)
        if not dry_run:
            processed.to_parquet(parquet_path, index=False)

    return stats
