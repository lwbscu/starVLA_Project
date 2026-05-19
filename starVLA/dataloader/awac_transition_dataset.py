# Copyright 2025 starVLA community. All rights reserved.
"""AWAC offline RL transition dataset built on top of LeRobot loaders."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from starVLA.dataloader.awac_collate import awac_collate_fn
from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES
from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset


def _scalar_reward(value: Any) -> float:
    if isinstance(value, (int, float, np.number)):
        return float(value)
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value).reshape(-1)
        return float(arr[0]) if arr.size else 0.0
    return float(value)


def read_chunk_reward(traj_df, start_index: int, horizon: int, reward_column: str = "reward") -> float:
    """Sum per-step rewards over [start_index, start_index + horizon) (legacy)."""
    if reward_column not in traj_df.columns:
        return 0.0
    total = 0.0
    traj_len = len(traj_df)
    for offset in range(horizon):
        idx = start_index + offset
        if idx >= traj_len:
            break
        value = traj_df[reward_column].iloc[idx]
        total += _scalar_reward(value)
    return total


def read_transition_reward(
    traj_df,
    start_index: int,
    horizon: int,
    reward_column: str = "reward",
    reward_is_chunk_return: bool = True,
) -> float:
    """
    Read transition reward at ``start_index``.

    After ``prepare_awac_rewards.py``, ``reward`` stores discounted chunk return
    R_chunk(t) and should **not** be summed again over the horizon.
    """
    if reward_column not in traj_df.columns:
        return 0.0
    if reward_is_chunk_return:
        return _scalar_reward(traj_df[reward_column].iloc[start_index])
    return read_chunk_reward(traj_df, start_index, horizon, reward_column)


def read_transition_done(
    traj_df,
    base_index: int,
    next_index: int,
    traj_len: int,
    done_column: str = "done",
) -> float:
    """Read ``done`` from parquet (last-H-frame flags) or fall back to index heuristic."""
    if done_column in traj_df.columns:
        value = traj_df[done_column].iloc[base_index]
        if isinstance(value, (bool, np.bool_)):
            return float(value)
        return float(_scalar_reward(value) > 0.5)
    return float(next_index >= traj_len - 1)


def _pack_observation(data: dict, dataset: LeRobotSingleDataset, include_state: bool) -> dict:
    step_images = []
    for video_key in dataset.modality_keys["video"]:
        image = data[video_key][0]
        image = Image.fromarray(image).resize((224, 224))
        step_images.append(image)

    language = data[dataset.modality_keys["language"][0]][0]
    action_parts = [data[action_key] for action_key in dataset.modality_keys["action"]]
    action = np.concatenate(action_parts, axis=1).astype(np.float32)

    sample = {"action": action, "image": step_images, "lang": language}
    if include_state and "state" in dataset.modality_keys:
        state_parts = [data[state_key] for state_key in dataset.modality_keys["state"]]
        sample["state"] = np.concatenate(state_parts, axis=1).astype(np.float32)
    return sample


class AWACTransitionDataset(Dataset):
    """Build (s, a, r, s', a', done) transitions from a LeRobotSingleDataset."""

    def __init__(
        self,
        base_dataset: LeRobotSingleDataset,
        action_horizon: int,
        include_state: bool = True,
        reward_column: str = "reward",
        done_column: str = "done",
        reward_is_chunk_return: bool = True,
    ):
        self.base_dataset = base_dataset
        self.action_horizon = int(action_horizon)
        self.include_state = include_state
        self.reward_column = reward_column
        self.done_column = done_column
        self.reward_is_chunk_return = reward_is_chunk_return
        self._reward_warning_issued = False

        self.valid_steps: list[tuple[int, int]] = []
        for trajectory_id, base_index in base_dataset.all_steps:
            traj_idx = base_dataset.get_trajectory_index(trajectory_id)
            traj_len = int(base_dataset.trajectory_lengths[traj_idx])
            if base_index + 2 * self.action_horizon <= traj_len:
                self.valid_steps.append((trajectory_id, base_index))

        if dist.is_initialized() and dist.get_rank() == 0:
            print(
                f"[AWAC] {base_dataset.dataset_name}: "
                f"{len(self.valid_steps)}/{len(base_dataset.all_steps)} valid transitions "
                f"(H={self.action_horizon})"
            )

    def __len__(self) -> int:
        return len(self.valid_steps)

    def __getitem__(self, index: int) -> dict[str, Any]:
        trajectory_id, base_index = self.valid_steps[index]
        horizon = self.action_horizon
        next_index = base_index + horizon

        raw_current = self.base_dataset.get_step_data(trajectory_id, base_index)
        raw_next = self.base_dataset.get_step_data(trajectory_id, next_index)
        current = self.base_dataset.transforms(raw_current)
        nxt = self.base_dataset.transforms(raw_next)

        sample = _pack_observation(current, self.base_dataset, self.include_state)
        next_sample = _pack_observation(nxt, self.base_dataset, self.include_state)

        traj_df = self.base_dataset.get_trajectory_data(trajectory_id)
        if self.reward_column not in traj_df.columns and not self._reward_warning_issued:
            warnings.warn(
                f"Column `{self.reward_column}` not found in trajectory parquet; "
                "using zero rewards. Add per-step rewards for AWAC.",
                stacklevel=2,
            )
            self._reward_warning_issued = True

        traj_idx = self.base_dataset.get_trajectory_index(trajectory_id)
        traj_len = int(self.base_dataset.trajectory_lengths[traj_idx])
        reward = read_transition_reward(
            traj_df,
            base_index,
            horizon,
            reward_column=self.reward_column,
            reward_is_chunk_return=self.reward_is_chunk_return,
        )
        done = read_transition_done(
            traj_df,
            base_index,
            next_index,
            traj_len,
            done_column=self.done_column,
        )

        transition = {
            "image": sample["image"],
            "lang": sample["lang"],
            "action": sample["action"],
            "next_image": next_sample["image"],
            "next_lang": next_sample["lang"],
            "next_action": next_sample["action"],
            "reward": reward,
            "done": done,
        }
        if self.include_state and "state" in sample:
            transition["state"] = sample["state"]
            transition["next_state"] = next_sample["state"]
        return transition


class AWACTransitionMixtureDataset(Dataset):
    """Weighted mixture of AWACTransitionDataset instances."""

    def __init__(self, transition_datasets: list[AWACTransitionDataset], weights: list[float], seed: int = 42):
        if not transition_datasets:
            raise ValueError("No AWAC transition datasets provided.")
        self.datasets = transition_datasets
        weights_arr = np.asarray(weights, dtype=np.float64)
        lengths = np.asarray([len(ds) for ds in transition_datasets], dtype=np.float64)
        weights_arr = weights_arr * lengths
        if weights_arr.sum() <= 0:
            weights_arr = np.ones(len(transition_datasets), dtype=np.float64)
        self.weights = weights_arr / weights_arr.sum()
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return int(sum(len(ds) for ds in self.datasets))

    def __getitem__(self, index: int) -> dict[str, Any]:
        dataset_idx = int(self._rng.choice(len(self.datasets), p=self.weights))
        local_index = int(self._rng.integers(0, len(self.datasets[dataset_idx])))
        return self.datasets[dataset_idx][local_index]


def get_awac_dataset(data_cfg, action_horizon: int) -> Dataset:
    data_root_dir = Path(data_cfg.data_root_dir)
    data_mix = data_cfg.data_mix
    delete_pause_frame = data_cfg.get("delete_pause_frame", False)
    include_state = data_cfg.get("include_state", True) not in ["False", False]
    reward_column = data_cfg.get("reward_column", "reward")
    done_column = data_cfg.get("done_column", "done")
    reward_is_chunk_return = data_cfg.get("reward_is_chunk_return", True) not in ["False", False]

    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    transition_sets: list[AWACTransitionDataset] = []
    weights: list[float] = []
    for d_name, d_weight, robot_type in mixture_spec:
        base = make_LeRobotSingleDataset(
            data_root_dir,
            d_name,
            robot_type,
            delete_pause_frame=delete_pause_frame,
            data_cfg=data_cfg,
        )
        transition_sets.append(
            AWACTransitionDataset(
                base_dataset=base,
                action_horizon=action_horizon,
                include_state=include_state,
                reward_column=reward_column,
                done_column=done_column,
                reward_is_chunk_return=reward_is_chunk_return,
            )
        )
        weights.append(float(d_weight))

    if len(transition_sets) == 1:
        return transition_sets[0]
    return AWACTransitionMixtureDataset(transition_sets, weights, seed=data_cfg.get("seed", 42))


def build_awac_dataloader(cfg):
    data_cfg = cfg.datasets.awac_data
    action_horizon = int(cfg.framework.action_model.action_horizon)
    dataset = get_awac_dataset(data_cfg, action_horizon=action_horizon)
    return DataLoader(
        dataset,
        batch_size=data_cfg.per_device_batch_size,
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        collate_fn=awac_collate_fn,
        pin_memory=True,
    )
