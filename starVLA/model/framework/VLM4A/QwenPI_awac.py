# Copyright 2025 starVLA community. All rights reserved.
"""QwenPI with AWAC-weighted per-sample flow-matching loss."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("QwenPI_AWAC")
class Qwen_PI_AWAC(Qwen_PI):
    """QwenPI actor with optional AWAC importance weights on flow-matching loss."""

    def _flow_matching_loss_per_sample(
        self,
        vl_embs_list: list[torch.Tensor],
        actions_target: torch.Tensor,
        state: Optional[torch.Tensor],
    ) -> torch.Tensor:
        action_model = self.action_model
        device = actions_target.device
        noise = torch.randn(actions_target.shape, device=device, dtype=actions_target.dtype)
        t = action_model.sample_time(actions_target.shape[0], device=device, dtype=actions_target.dtype)
        t = t[:, None, None]

        noisy_trajectory = (1 - t) * noise + t * actions_target
        velocity = actions_target - noise
        t_discretized = (t[:, 0, 0] * action_model.num_timestep_buckets).long()
        action_features = action_model.action_encoder(noisy_trajectory, t_discretized)

        state_features = action_model.state_encoder(state) if state is not None else None
        if action_model.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = action_model.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        future_tokens = action_model.future_tokens.weight.unsqueeze(0).expand(actions_target.shape[0], -1, -1)
        sa_embs = (
            torch.cat((state_features, future_tokens, action_features), dim=1)
            if state_features is not None
            else torch.cat((future_tokens, action_features), dim=1)
        )
        temb = action_model.model.timestep_encoder(t_discretized)

        model_output = sa_embs
        for layer_idx, layer in enumerate(action_model.model.transformer_blocks):
            model_output = layer(
                hidden_states=model_output,
                encoder_hidden_states=vl_embs_list[layer_idx],
                temb=temb,
            )
        pred = action_model.action_decoder(model_output)
        pred_actions = pred[:, -actions_target.shape[1] :]
        return ((pred_actions - velocity) ** 2).mean(dim=(1, 2))

    def forward(
        self,
        examples: List[dict] = None,
        awac_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if examples and "state" in examples[0] else None

        vl_embs_list = self._encode_vl_hidden_states(batch_images, instructions)
        base_hidden = vl_embs_list[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            actions_tensor = torch.tensor(
                np.array(actions), device=base_hidden.device, dtype=base_hidden.dtype
            )
            actions_target = actions_tensor[:, -self.action_horizon :, :]

            state_tensor = None
            if state is not None:
                state_tensor = torch.tensor(np.array(state), device=base_hidden.device, dtype=base_hidden.dtype)

            loss_per_sample = self._flow_matching_loss_per_sample(vl_embs_list, actions_target, state_tensor)
            if awac_weights is not None:
                weights = awac_weights.to(loss_per_sample.device, dtype=loss_per_sample.dtype)
                if weights.ndim > 1:
                    weights = weights.squeeze(-1)
                action_loss = (loss_per_sample * weights).mean()
            else:
                action_loss = loss_per_sample.mean()

        return {"action_loss": action_loss, "loss_per_sample": loss_per_sample}
