# Copyright 2025 starVLA community. All rights reserved.
"""AWAC offline RL transition dataset built on top of LeRobot loaders."""

from __future__ import annotations

import json
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

_STATE_DIM_WARNINGS_ISSUED: set[tuple[str, int, int]] = set()


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


def read_episode_success(
    traj_df,
    success_column: str = "success",
    assume_success_if_missing: bool = False,
) -> bool:
    if success_column not in traj_df.columns:
        if assume_success_if_missing:
            return True
        raise KeyError(
            f"Column `{success_column}` not found in trajectory parquet. "
            "Set datasets.awac_data.assume_success_if_missing=true only for verified all-success demos, "
            "or provide explicit success labels."
        )
    if len(traj_df) == 0:
        return False
    value = traj_df[success_column].iloc[-1]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return _scalar_reward(value) > 0.5


def compute_awac_step_reward(
    frame_index: int,
    traj_len: int,
    episode_success: bool,
    step_penalty: float = -1.0,
    success_reward: float = 0.0,
    failure_reward: float = -3000.0,
) -> float:
    if traj_len <= 0:
        return 0.0
    if frame_index >= traj_len - 1:
        return float(success_reward if episode_success else failure_reward)
    return float(step_penalty)


def compute_awac_chunk_return(
    start_index: int,
    traj_len: int,
    horizon: int,
    gamma: float,
    episode_success: bool,
    step_penalty: float = -1.0,
    success_reward: float = 0.0,
    failure_reward: float = -3000.0,
) -> float:
    if traj_len <= 0:
        return 0.0
    total = 0.0
    max_offset = min(int(horizon) - 1, traj_len - 1 - int(start_index))
    for offset in range(max_offset + 1):
        total += (float(gamma) ** offset) * compute_awac_step_reward(
            frame_index=start_index + offset,
            traj_len=traj_len,
            episode_success=episode_success,
            step_penalty=step_penalty,
            success_reward=success_reward,
            failure_reward=failure_reward,
        )
    return float(total)


def compute_awac_done(base_index: int, traj_len: int, horizon: int) -> float:
    return float(int(base_index) >= max(0, int(traj_len) - int(horizon)))


def _align_state_dim(state: np.ndarray, target_dim: int | None, dataset_name: str) -> np.ndarray:
    if target_dim is None or target_dim <= 0:
        return state
    current_dim = int(state.shape[-1])
    if current_dim == target_dim:
        return state
    if current_dim < target_dim:
        raise ValueError(
            f"AWAC state dimension mismatch for {dataset_name}: "
            f"data has {current_dim} dims but framework.action_model.state_dim={target_dim}."
        )
    warning_key = (dataset_name, current_dim, target_dim)
    if warning_key not in _STATE_DIM_WARNINGS_ISSUED:
        warnings.warn(
            f"AWAC state dimension mismatch for {dataset_name}: "
            f"data has {current_dim} dims, trimming trailing dims to configured state_dim={target_dim}.",
            stacklevel=2,
        )
        _STATE_DIM_WARNINGS_ISSUED.add(warning_key)
    return state[..., :target_dim]


def _pack_observation(
    data: dict,
    dataset: LeRobotSingleDataset,
    include_state: bool,
    state_dim: int | None = None,
) -> dict:
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
        state = np.concatenate(state_parts, axis=1).astype(np.float32)
        sample["state"] = _align_state_dim(state, state_dim, dataset.dataset_name)
    return sample


class AWACTransitionDataset(Dataset):
    """Build (s, a, r, s', a', done) transitions from a LeRobotSingleDataset."""

    def __init__(
        self,
        base_dataset: LeRobotSingleDataset,
        action_horizon: int,
        include_state: bool = True,
        state_dim: int | None = None,
        reward_column: str = "reward",
        done_column: str = "done",
        reward_is_chunk_return: bool = True,
        compute_rewards_on_the_fly: bool = False,
        success_column: str = "success",
        assume_success_if_missing: bool = False,
        gamma: float = 0.996,
        step_penalty: float = -1.0,
        success_reward: float = 0.0,
        failure_reward: float = -3000.0,
        episode_allowlist: set[int] | None = None,
    ):
        self.base_dataset = base_dataset
        self.episode_allowlist = episode_allowlist
        self.action_horizon = int(action_horizon)
        self.include_state = include_state
        self.state_dim = int(state_dim) if state_dim is not None else None
        self.reward_column = reward_column
        self.done_column = done_column
        self.reward_is_chunk_return = reward_is_chunk_return
        self.compute_rewards_on_the_fly = compute_rewards_on_the_fly
        self.success_column = success_column
        self.assume_success_if_missing = assume_success_if_missing
        self.gamma = float(gamma)
        self.step_penalty = float(step_penalty)
        self.success_reward = float(success_reward)
        self.failure_reward = float(failure_reward)
        self._reward_warning_issued = False

        self.valid_steps: list[tuple[int, int]] = []
        for trajectory_id, base_index in base_dataset.all_steps:
            if self.episode_allowlist is not None and int(trajectory_id) not in self.episode_allowlist:
                continue
            traj_idx = base_dataset.get_trajectory_index(trajectory_id)
            traj_len = int(base_dataset.trajectory_lengths[traj_idx])
            # Chunk reward/done use frames [t, t+H-1]; require the full H-step window in-episode.
            # (Legacy filter t+2H<=T kept s' at t+H and a' at t+2H; last-H done/r were excluded.)
            if base_index + self.action_horizon - 1 < traj_len:
                self.valid_steps.append((trajectory_id, base_index))

        if not dist.is_initialized() or dist.get_rank() == 0:
            allow_msg = (
                f", episode_allowlist={len(self.episode_allowlist)}"
                if self.episode_allowlist is not None
                else ""
            )
            print(
                f"[AWAC] {base_dataset.dataset_name}: "
                f"{len(self.valid_steps)}/{len(base_dataset.all_steps)} valid transitions "
                f"(H={self.action_horizon}{allow_msg})"
            )
            if self.compute_rewards_on_the_fly:
                print(
                    f"[AWAC] {base_dataset.dataset_name}: computing reward/done on the fly "
                    f"(H={self.action_horizon}, gamma={self.gamma}, "
                    f"success_column={self.success_column}, "
                    f"assume_success_if_missing={self.assume_success_if_missing})"
                )

    def __len__(self) -> int:
        return len(self.valid_steps)

    def __getitem__(self, index: int) -> dict[str, Any]:
        trajectory_id, base_index = self.valid_steps[index]
        horizon = self.action_horizon
        traj_idx = self.base_dataset.get_trajectory_index(trajectory_id)
        traj_len = int(self.base_dataset.trajectory_lengths[traj_idx])
        # Bootstrap state s' at t+H when in range; else last frame (near episode end).
        next_index = min(base_index + horizon, traj_len - 1)

        raw_current = self.base_dataset.get_step_data(trajectory_id, base_index)
        raw_next = self.base_dataset.get_step_data(trajectory_id, next_index)
        current = self.base_dataset.transforms(raw_current)
        nxt = self.base_dataset.transforms(raw_next)

        sample = _pack_observation(current, self.base_dataset, self.include_state, self.state_dim)
        next_sample = _pack_observation(nxt, self.base_dataset, self.include_state, self.state_dim)

        traj_df = self.base_dataset.get_trajectory_data(trajectory_id)
        if (
            self.reward_column not in traj_df.columns
            and not self.compute_rewards_on_the_fly
            and not self._reward_warning_issued
        ):
            warnings.warn(
                f"Column `{self.reward_column}` not found in trajectory parquet; "
                "using zero rewards. Add per-step rewards for AWAC.",
                stacklevel=2,
            )
            self._reward_warning_issued = True

        if self.compute_rewards_on_the_fly and self.reward_column not in traj_df.columns:
            episode_success = read_episode_success(
                traj_df,
                success_column=self.success_column,
                assume_success_if_missing=self.assume_success_if_missing,
            )
            reward = compute_awac_chunk_return(
                start_index=base_index,
                traj_len=traj_len,
                horizon=horizon,
                gamma=self.gamma,
                episode_success=episode_success,
                step_penalty=self.step_penalty,
                success_reward=self.success_reward,
                failure_reward=self.failure_reward,
            )
        else:
            reward = read_transition_reward(
                traj_df,
                base_index,
                horizon,
                reward_column=self.reward_column,
                reward_is_chunk_return=self.reward_is_chunk_return,
            )

        if self.compute_rewards_on_the_fly and self.done_column not in traj_df.columns:
            done = compute_awac_done(base_index, traj_len, horizon)
        else:
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


def _load_episode_allowlist(data_cfg, dataset_name: str) -> set[int] | None:
    allowlist_path = data_cfg.get("episode_allowlists_path", None)
    if not allowlist_path:
        return None
    payload = json.loads(Path(allowlist_path).read_text(encoding="utf-8"))
    episode_ids = payload.get(dataset_name, None)
    if episode_ids is None:
        return None
    return {int(ep_id) for ep_id in episode_ids}


def _resolve_awac_dataset_root(data_cfg, dataset_name: str) -> Path:
    """Per-dataset LeRobot parent dir (subdir name is ``dataset_name``)."""
    roots = data_cfg.get("dataset_roots", None)
    if roots is not None:
        try:
            from omegaconf import OmegaConf

            roots = OmegaConf.to_container(roots, resolve=True)
        except Exception:
            roots = dict(roots) if hasattr(roots, "items") else roots
        if isinstance(roots, dict) and dataset_name in roots:
            return Path(roots[dataset_name])
    return Path(data_cfg.data_root_dir)


class AWACTransitionMixtureDataset(Dataset):
    """Weighted mixture of AWACTransitionDataset instances."""

    def __init__(
        self,
        transition_datasets: list[AWACTransitionDataset],
        weights: list[float],
        seed: int = 42,
        balance_by_dataset: bool = False,
    ):
        if not transition_datasets:
            raise ValueError("No AWAC transition datasets provided.")
        self.datasets = transition_datasets
        weights_arr = np.asarray(weights, dtype=np.float64)
        if balance_by_dataset:
            if weights_arr.sum() <= 0:
                weights_arr = np.ones(len(transition_datasets), dtype=np.float64)
            self.weights = weights_arr / weights_arr.sum()
        else:
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


def get_awac_dataset(
    data_cfg,
    action_horizon: int,
    state_dim: int | None = None,
    gamma: float | None = None,
) -> Dataset:
    data_mix = data_cfg.data_mix
    balance_by_dataset = data_cfg.get("balance_datasets", False) not in ["False", False]
    delete_pause_frame = data_cfg.get("delete_pause_frame", False)
    include_state = data_cfg.get("include_state", True) not in ["False", False]
    reward_column = data_cfg.get("reward_column", "reward")
    done_column = data_cfg.get("done_column", "done")
    reward_is_chunk_return = data_cfg.get("reward_is_chunk_return", True) not in ["False", False]
    compute_rewards_on_the_fly = data_cfg.get("compute_rewards_on_the_fly", False) not in ["False", False]
    success_column = data_cfg.get("success_column", "success")
    assume_success_if_missing = data_cfg.get("assume_success_if_missing", False) not in ["False", False]
    gamma = float(gamma if gamma is not None else data_cfg.get("gamma", 0.996))
    step_penalty = float(data_cfg.get("step_penalty", -1.0))
    success_reward = float(data_cfg.get("success_reward", 0.0))
    failure_reward = float(data_cfg.get("failure_reward", -3000.0))

    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    transition_sets: list[AWACTransitionDataset] = []
    weights: list[float] = []
    for d_name, d_weight, robot_type in mixture_spec:
        dataset_root = _resolve_awac_dataset_root(data_cfg, d_name)
        base = make_LeRobotSingleDataset(
            dataset_root,
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
                state_dim=state_dim,
                reward_column=reward_column,
                done_column=done_column,
                reward_is_chunk_return=reward_is_chunk_return,
                compute_rewards_on_the_fly=compute_rewards_on_the_fly,
                success_column=success_column,
                assume_success_if_missing=assume_success_if_missing,
                gamma=gamma,
                step_penalty=step_penalty,
                success_reward=success_reward,
                failure_reward=failure_reward,
                episode_allowlist=_load_episode_allowlist(data_cfg, d_name),
            )
        )
        weights.append(float(d_weight))

    if len(transition_sets) == 1:
        return transition_sets[0]
    if not dist.is_initialized() or dist.get_rank() == 0:
        for ds, w in zip(transition_sets, weights):
            print(
                f"[AWAC] mixture: {ds.base_dataset.dataset_name} "
                f"len={len(ds)} nominal_weight={w} root={ds.base_dataset.dataset_path}"
            )
        print(f"[AWAC] mixture sampling: balance_by_dataset={balance_by_dataset}")
    return AWACTransitionMixtureDataset(
        transition_sets,
        weights,
        seed=data_cfg.get("seed", 42),
        balance_by_dataset=balance_by_dataset,
    )


def build_awac_dataloader(cfg):
    data_cfg = cfg.datasets.awac_data
    action_horizon = int(cfg.framework.action_model.action_horizon)
    state_dim = int(cfg.framework.action_model.state_dim) if cfg.framework.action_model.get("state_dim", None) else None
    dataset = get_awac_dataset(
        data_cfg,
        action_horizon=action_horizon,
        state_dim=state_dim,
        gamma=float(cfg.awac.gamma),
    )
    return DataLoader(
        dataset,
        batch_size=data_cfg.per_device_batch_size,
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        collate_fn=awac_collate_fn,
        pin_memory=True,
    )
