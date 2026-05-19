# Copyright 2025 starVLA community. All rights reserved.
"""Collate utilities for AWAC transition batches."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def awac_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack numeric fields; keep PIL images and language as lists for VLM encoding."""
    out: dict[str, Any] = {
        "image": [sample["image"] for sample in batch],
        "next_image": [sample["next_image"] for sample in batch],
        "lang": [sample["lang"] for sample in batch],
        "next_lang": [sample.get("next_lang", sample["lang"]) for sample in batch],
        "action": torch.tensor(np.stack([sample["action"] for sample in batch]), dtype=torch.float32),
        "next_action": torch.tensor(np.stack([sample["next_action"] for sample in batch]), dtype=torch.float32),
        "reward": torch.tensor([sample["reward"] for sample in batch], dtype=torch.float32),
        "done": torch.tensor([sample["done"] for sample in batch], dtype=torch.float32),
    }
    if "state" in batch[0]:
        states = []
        for sample in batch:
            state = np.asarray(sample["state"], dtype=np.float32)
            if state.ndim == 2:
                state = state[-1:]
            if state.ndim == 1:
                state = state[None, :]
            states.append(state)
        out["state"] = torch.tensor(np.stack(states), dtype=torch.float32)
    if "next_state" in batch[0]:
        next_states = []
        for sample in batch:
            state = np.asarray(sample["next_state"], dtype=np.float32)
            if state.ndim == 2:
                state = state[-1:]
            if state.ndim == 1:
                state = state[None, :]
            next_states.append(state)
        out["next_state"] = torch.tensor(np.stack(next_states), dtype=torch.float32)
    return out
