# Copyright 2025 starVLA community. All rights reserved.
"""Expectile value network for AWAC."""

from __future__ import annotations

from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    weight = torch.where(diff > 0, tau, 1.0 - tau)
    return (weight * (diff**2)).mean()


class AWACValueNetwork(nn.Module):
    """V(s) from pooled visual + state features."""

    def __init__(
        self,
        critic: nn.Module,
        hidden_dim: int = 256,
    ):
        super().__init__()
        # Share the critic feature encoder without registering the full critic
        # as a child module. Registering it here duplicates parameters in the
        # optimizer and breaks DeepSpeed's single-model prepare path.
        object.__setattr__(self, "critic", critic)
        in_dim = critic.hidden_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_state(
        self,
        batch_images: List[List[Any]],
        instructions: List[str],
        state: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = len(batch_images)
        device = next(self.parameters()).device
        visual_tokens = self.critic._encode_view(batch_images, instructions)
        visual_pool = visual_tokens.mean(dim=1)

        if state is None:
            state = torch.zeros(batch_size, 1, self.critic.state_dim, device=device)
        if state.ndim == 2:
            state = state.unsqueeze(1)
        state_flat = state[:, -1, :].to(device=device, dtype=self.critic.state_proj.weight.dtype)
        state_feat = self.critic.state_proj(state_flat)
        return torch.cat([visual_pool, state_feat], dim=-1)

    def forward(
        self,
        batch_images: List[List[Any]],
        instructions: List[str],
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features = self.encode_state(batch_images, instructions, state)
        return self.mlp(features)

    def expectile_loss(
        self,
        batch: dict[str, Any],
        q_target: torch.Tensor,
        tau: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        v_pred = self.forward(batch["image"], batch["lang"], state=batch.get("state")).squeeze(-1)
        diff = q_target.detach() - v_pred
        loss = expectile_loss(diff, tau)
        return loss, {"value_loss": float(loss.detach().cpu()), "v_mean": float(v_pred.detach().mean().cpu())}
