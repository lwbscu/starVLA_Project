# Copyright 2025 starVLA community. All rights reserved.
"""AWAC Q-critic with actor-copied visual encoder and token transformer."""

from __future__ import annotations

import copy
from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_update_target(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


class AWACQCritic(nn.Module):
    """
    Q(s, a) with E heads.

    Token layout: [state(1) | action(H) | visual(V) | query(E)].
    """

    def __init__(
        self,
        qwen_vl_interface: nn.Module,
        action_dim: int = 7,
        state_dim: int = 7,
        action_horizon: int = 8,
        hidden_dim: int = 512,
        num_q_heads: int = 2,
        num_layers: int = 6,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        freeze_visual: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.qwen_vl_interface = qwen_vl_interface
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.action_horizon = action_horizon
        self.hidden_dim = hidden_dim
        self.num_q_heads = num_q_heads

        actor_visual = qwen_vl_interface.model.visual
        self.visual = copy.deepcopy(actor_visual)
        self.visual.load_state_dict(actor_visual.state_dict())

        if freeze_visual:
            for param in self.visual.parameters():
                param.requires_grad = False

        visual_out_dim = self._infer_visual_dim()
        self.visual_proj = nn.Linear(visual_out_dim, hidden_dim)
        self.shared_proj = nn.Linear(action_dim, hidden_dim)
        self.query_tokens = nn.Parameter(torch.randn(num_q_heads, hidden_dim) * 0.02)

        max_tokens = 1 + action_horizon + 16 + num_q_heads
        self.pos_embed = nn.Parameter(torch.randn(max_tokens, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.q_head = nn.Linear(hidden_dim, 1)

    def _infer_visual_dim(self) -> int:
        for tensor in self.visual.parameters():
            if tensor.ndim == 2:
                return tensor.shape[1]
        return self.hidden_dim

    def _encode_view(
        self,
        batch_images: List[List[Any]],
        instructions: List[str],
    ) -> torch.Tensor:
        """Encode each camera view -> (B, hidden_dim)."""
        num_views = len(batch_images[0])
        view_embeds = []
        device = next(self.parameters()).device
        for view_idx in range(num_views):
            view_batch = [[sample[view_idx]] for sample in batch_images]
            inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=view_batch,
                instructions=instructions,
            )
            pixel_values = inputs["pixel_values"].to(device)
            grid_thw = inputs.get("image_grid_thw")
            if grid_thw is not None:
                grid_thw = grid_thw.to(device)
            with torch.set_grad_enabled(any(p.requires_grad for p in self.visual.parameters())):
                if grid_thw is not None:
                    visual_out = self.visual(pixel_values, grid_thw=grid_thw)
                else:
                    visual_out = self.visual(pixel_values)
            if isinstance(visual_out, tuple):
                visual_out = visual_out[0]
            if visual_out.ndim == 3:
                pooled = visual_out.mean(dim=1)
            elif visual_out.ndim == 2:
                pooled = visual_out
            else:
                pooled = visual_out.reshape(visual_out.shape[0], -1, visual_out.shape[-1]).mean(dim=1)
            view_embeds.append(self.visual_proj(pooled))
        return torch.stack(view_embeds, dim=1)

    def _build_tokens(
        self,
        batch_images: List[List[Any]],
        instructions: List[str],
        state: Optional[torch.Tensor],
        action: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = action.shape[0]
        device = action.device

        if state is None:
            state = torch.zeros(batch_size, 1, self.action_dim, device=device, dtype=action.dtype)
        if state.ndim == 2:
            state = state.unsqueeze(1)
        state_flat = state[:, -1, :]
        state_token = self.shared_proj(state_flat).unsqueeze(1)

        bsz, horizon, adim = action.shape
        action_flat = action.reshape(bsz * horizon, adim)
        action_tokens = self.shared_proj(action_flat).reshape(bsz, horizon, self.hidden_dim)

        visual_tokens = self._encode_view(batch_images, instructions)
        query_tokens = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        tokens = torch.cat([state_token, action_tokens, visual_tokens, query_tokens], dim=1)
        seq_len = tokens.shape[1]
        tokens = tokens + self.pos_embed[:seq_len].unsqueeze(0)
        return tokens

    def forward(
        self,
        batch_images: List[List[Any]],
        instructions: List[str],
        action: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tokens = self._build_tokens(batch_images, instructions, state, action)
        encoded = self.transformer(tokens)
        query_out = encoded[:, -self.num_q_heads :, :]
        return self.q_head(query_out)

    def min_q(
        self,
        batch_images: List[List[Any]],
        instructions: List[str],
        action: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q_values = self.forward(batch_images, instructions, action, state=state)
        return q_values.min(dim=1).values

    def td_loss(
        self,
        batch: dict[str, Any],
        target_critic: "AWACQCritic",
        gamma: float,
        action_horizon: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        q_pred = self.forward(
            batch["image"],
            batch["lang"],
            batch["action"],
            state=batch.get("state"),
        )
        with torch.no_grad():
            q_next = target_critic(
                batch["next_image"],
                batch["next_lang"],
                batch["next_action"],
                state=batch.get("next_state"),
            )
            q_next_min = q_next.min(dim=1).values.squeeze(-1)
            bootstrap = (1.0 - batch["done"]) * (gamma ** action_horizon)
            target = batch["reward"] + bootstrap * q_next_min

        target_heads = target.unsqueeze(1).expand(-1, self.num_q_heads)
        loss = F.mse_loss(q_pred.squeeze(-1), target_heads)
        metrics = {
            "critic_loss": float(loss.detach().cpu()),
            "q_mean": float(q_pred.detach().mean().cpu()),
            "target_mean": float(target.detach().mean().cpu()),
        }
        return loss, metrics
