# Copyright 2025 starVLA community. All rights reserved.
"""AWAC Q-critic: copied visual tower (2 cameras) + text embed + state/action + query."""

from __future__ import annotations

import copy
from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Calvin default: index 0 = static/head, index 1 = wrist/gripper.
NUM_CAMERA_VIEWS = 2
HEAD_CAMERA_INDEX = 0
WRIST_CAMERA_INDEX = 1


def soft_update_target(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


def resolve_qwen_visual_module(qwen_vl_interface: nn.Module) -> tuple[nn.Module | None, str]:
    """
    Locate the standalone vision tower on the wrapped Qwen-VL backbone.

    - Qwen2.5-VL: ``qwen_vl_interface.model.visual``
    - Qwen3.5-VL: ``qwen_vl_interface.model.model.visual`` (see project README freeze_modules)
    """
    backbone = getattr(qwen_vl_interface, "model", None)
    if backbone is None:
        return None, "missing:qwen_vl_interface.model"

    inner = getattr(backbone, "model", None)
    candidates: list[tuple[str, nn.Module | None]] = []
    if inner is not None:
        candidates.append(("model.model.visual", getattr(inner, "visual", None)))
    candidates.append(("model.visual", getattr(backbone, "visual", None)))

    for path, module in candidates:
        if module is not None:
            return module, path
    return None, "not_found"


class AWACQCritic(nn.Module):
    """
    Q(s, a) with E heads.

    Token layout (concatenated into 6-layer Transformer, Q read from query slots):
        [vision(2) | text(1) | action(H) | state(1) | query(E)]

    - Vision: copied Qwen ``visual`` tower per camera -> (B, 2, D); view0=head, view1=wrist.
    - Text: tokenize instruction -> token embeddings -> mean pool -> (B, 1, D).
    - Action / state: separate linear projections -> (B, H, D) and (B, 1, D).
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
        max_vision_tokens: int = 256,  # kept for checkpoint/config compat; unused
        max_text_tokens: int = 128,
    ):
        super().__init__()
        del max_vision_tokens, max_text_tokens  # unused in this layout

        object.__setattr__(self, "qwen_vl_interface", qwen_vl_interface)
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.action_horizon = action_horizon
        self.hidden_dim = hidden_dim
        self.num_q_heads = num_q_heads
        self.num_camera_views = NUM_CAMERA_VIEWS

        if freeze_visual:
            qwen_vl_interface.eval()
            for param in qwen_vl_interface.parameters():
                param.requires_grad = False

        actor_visual, self.visual_source_path = resolve_qwen_visual_module(qwen_vl_interface)
        self.visual = None
        if actor_visual is not None:
            self.visual = copy.deepcopy(actor_visual)
            self.visual.load_state_dict(actor_visual.state_dict())
            if freeze_visual:
                for param in self.visual.parameters():
                    param.requires_grad = False
        elif freeze_visual:
            raise RuntimeError(
                "AWACQCritic could not find a standalone Qwen visual tower. "
                "Tried model.model.visual (Qwen3.5-VL) and model.visual (Qwen2.5-VL). "
                "Refusing full-VLM fallback for critic training."
            )

        visual_out_dim = self._infer_visual_out_dim()
        text_out_dim = self._infer_text_hidden_dim()
        self.visual_proj = nn.Linear(visual_out_dim, hidden_dim)
        self.text_proj = nn.Linear(text_out_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.query_tokens = nn.Parameter(torch.randn(num_q_heads, hidden_dim) * 0.02)

        max_tokens = NUM_CAMERA_VIEWS + 1 + action_horizon + 1 + num_q_heads
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

    def _infer_visual_out_dim(self) -> int:
        backbone = getattr(self.qwen_vl_interface, "model", None)
        if backbone is not None:
            cfg = getattr(backbone, "config", None)
            if cfg is not None:
                vision_cfg = getattr(cfg, "vision_config", None)
                if vision_cfg is not None:
                    for key in ("out_hidden_size", "hidden_size", "embed_dim"):
                        value = getattr(vision_cfg, key, None)
                        if value is not None:
                            return int(value)
        if self.visual is not None:
            max_dim = 0
            for tensor in self.visual.parameters():
                if tensor.ndim == 2:
                    max_dim = max(max_dim, int(tensor.shape[0]), int(tensor.shape[1]))
            if max_dim > 0:
                return max_dim
        return self._infer_text_hidden_dim()

    def _infer_text_hidden_dim(self) -> int:
        model_cfg = getattr(getattr(self.qwen_vl_interface, "model", None), "config", None)
        if model_cfg is None:
            return self.hidden_dim
        text_cfg = getattr(model_cfg, "text_config", model_cfg)
        return int(getattr(model_cfg, "hidden_size", getattr(text_cfg, "hidden_size", self.hidden_dim)))

    def _move_vlm_interface(self, device: torch.device) -> None:
        try:
            current_device = next(self.qwen_vl_interface.parameters()).device
        except StopIteration:
            return
        if current_device != device:
            self.qwen_vl_interface.to(device)
        if self.visual is not None:
            self.visual.to(device)

    def _get_text_embedding_layer(self) -> nn.Module:
        backbone = self.qwen_vl_interface.model
        if hasattr(backbone, "get_input_embeddings"):
            return backbone.get_input_embeddings()
        inner = getattr(backbone, "model", None)
        if inner is not None and hasattr(inner, "embed_tokens"):
            return inner.embed_tokens
        raise RuntimeError("Cannot locate text embedding layer on qwen_vl_interface.model")

    def _unwrap_visual_tensor(self, visual_out: Any) -> torch.Tensor:
        """Qwen2.5 visual may return Tensor; Qwen3.5 visual returns ModelOutput with pooling."""
        if isinstance(visual_out, tuple):
            visual_out = visual_out[0]
        if torch.is_tensor(visual_out):
            return visual_out
        pooler = getattr(visual_out, "pooler_output", None)
        if pooler is not None:
            return pooler
        last_hidden = getattr(visual_out, "last_hidden_state", None)
        if last_hidden is not None:
            return last_hidden
        hidden_states = getattr(visual_out, "hidden_states", None)
        if hidden_states is not None:
            return hidden_states[-1]
        raise TypeError(f"Unsupported visual tower output type: {type(visual_out)}")

    def _pool_visual_features(
        self,
        visual_out: Any,
        grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pool visual tower output to (B, visual_dim). Qwen3.5 returns flattened patch tokens."""
        features = self._unwrap_visual_tensor(visual_out)
        if features.ndim == 3:
            return features.mean(dim=1)
        if features.ndim != 2:
            return features.reshape(features.shape[0], -1, features.shape[-1]).mean(dim=1)

        if grid_thw is None:
            return features

        grid = grid_thw.to(device=features.device)
        if grid.ndim == 1:
            grid = grid.unsqueeze(0)
        batch_size = int(grid.shape[0])

        # Already per-sample (B, D), e.g. pooler_output.
        if features.shape[0] == batch_size:
            return features

        pooled: list[torch.Tensor] = []
        start = 0
        for i in range(batch_size):
            if grid.shape[1] >= 3:
                t, h, w = grid[i, 0], grid[i, 1], grid[i, 2]
                num_tokens = int((t * h * w).item())
            else:
                num_tokens = int(grid[i].prod().item())
            end = start + num_tokens
            chunk = features[start:end]
            if chunk.numel() == 0:
                pooled.append(
                    torch.zeros(features.shape[-1], device=features.device, dtype=features.dtype)
                )
            else:
                pooled.append(chunk.mean(dim=0))
            start = end

        if start == features.shape[0] and len(pooled) == batch_size:
            return torch.stack(pooled, dim=0)

        # Fallback: uniform token split when grid metadata does not match length.
        tokens_per = max(features.shape[0] // max(batch_size, 1), 1)
        pooled = []
        for i in range(batch_size):
            chunk = features[i * tokens_per : (i + 1) * tokens_per]
            if chunk.numel():
                pooled.append(chunk.mean(dim=0))
            else:
                pooled.append(
                    torch.zeros(features.shape[-1], device=features.device, dtype=features.dtype)
                )
        return torch.stack(pooled, dim=0)

    def _run_visual_tower(
        self,
        pixel_values: torch.Tensor,
        grid_thw: Optional[torch.Tensor],
    ) -> torch.Tensor:
        with torch.set_grad_enabled(self.visual is not None and any(p.requires_grad for p in self.visual.parameters())):
            if grid_thw is not None:
                return self.visual(pixel_values, grid_thw=grid_thw)
            return self.visual(pixel_values)

    def _encode_single_view_visual(
        self,
        batch_images: List[List[Any]],
        view_index: int,
        instructions: List[str],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """One camera view -> (B, visual_out_dim)."""
        view_batch: List[List[Any]] = []
        for sample in batch_images:
            if len(sample) > view_index:
                view_batch.append([sample[view_index]])
            elif sample:
                view_batch.append([sample[0]])
            else:
                raise ValueError("AWAC critic received an empty image list for a sample.")

        inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=view_batch,
            instructions=instructions,
        )
        inputs = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }

        if self.visual is not None:
            visual_out = self._run_visual_tower(
                inputs["pixel_values"],
                inputs.get("image_grid_thw"),
            )
            return self._pool_visual_features(
                visual_out,
                inputs.get("image_grid_thw"),
            ).to(dtype=dtype)

        # Fallback when backbone has no standalone visual module (e.g. some Qwen3.5 layouts).
        with torch.set_grad_enabled(any(p.requires_grad for p in self.qwen_vl_interface.parameters())):
            outputs = self.qwen_vl_interface(
                **inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden = outputs.hidden_states[-1]
        return hidden.mean(dim=1).to(dtype=dtype)

    def _encode_vision_tokens(
        self,
        batch_images: List[List[Any]],
        instructions: List[str],
    ) -> torch.Tensor:
        """Head + wrist cameras -> (B, 2, hidden_dim)."""
        device = next(self.parameters()).device
        dtype = self.visual_proj.weight.dtype
        self._move_vlm_interface(device)

        view_features = []
        for view_idx in (HEAD_CAMERA_INDEX, WRIST_CAMERA_INDEX):
            pooled = self._encode_single_view_visual(
                batch_images,
                view_index=view_idx,
                instructions=instructions,
                device=device,
                dtype=dtype,
            )
            view_features.append(pooled)
        stacked = torch.stack(view_features, dim=1)
        return self.visual_proj(stacked)

    def _encode_text_token(
        self,
        instructions: List[str],
    ) -> torch.Tensor:
        """Tokenize language only -> (B, 1, hidden_dim)."""
        device = next(self.parameters()).device
        dtype = self.text_proj.weight.dtype
        self._move_vlm_interface(device)

        processor = self.qwen_vl_interface.processor
        messages = [
            [{"role": "user", "content": [{"type": "text", "text": instruction}]}]
            for instruction in instructions
        ]

        if hasattr(processor, "apply_chat_template"):
            tokenized = processor.apply_chat_template(
                messages,
                tokenize=True,
                padding=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized.get("attention_mask")
        else:
            texts = [processor.apply_chat_template([m], tokenize=False) for m in messages]
            tokenized = processor(text=texts, padding=True, return_tensors="pt")
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized.get("attention_mask")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)
        else:
            attention_mask = attention_mask.to(device)

        embed_layer = self._get_text_embedding_layer()
        token_embeds = embed_layer(input_ids).to(dtype=dtype)
        mask = attention_mask.unsqueeze(-1).to(dtype=dtype)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = (token_embeds * mask).sum(dim=1) / denom
        return self.text_proj(pooled).unsqueeze(1)

    def _encode_view(
        self,
        batch_images: List[List[Any]],
        instructions: List[str],
    ) -> torch.Tensor:
        """Backward-compatible (B, 2, hidden_dim): [head, wrist] rows."""
        return self._encode_vision_tokens(batch_images, instructions)

    def _build_tokens(
        self,
        batch_images: List[List[Any]],
        instructions: List[str],
        state: Optional[torch.Tensor],
        action: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = action.shape[0]
        device = action.device

        vision_tokens = self._encode_vision_tokens(batch_images, instructions)
        text_token = self._encode_text_token(instructions)

        if state is None:
            state = torch.zeros(batch_size, 1, self.state_dim, device=device, dtype=action.dtype)
        if state.ndim == 2:
            state = state.unsqueeze(1)
        proj_dtype = self.action_proj.weight.dtype
        state_flat = state[:, -1, :].to(device=device, dtype=proj_dtype)
        state_token = self.state_proj(state_flat).unsqueeze(1)

        action = action.to(device=device, dtype=proj_dtype)
        bsz, horizon, adim = action.shape
        action_flat = action.reshape(bsz * horizon, adim)
        action_tokens = self.action_proj(action_flat).reshape(bsz, horizon, self.hidden_dim)

        query_tokens = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        tokens = torch.cat(
            [vision_tokens, text_token, action_tokens, state_token, query_tokens],
            dim=1,
        )
        seq_len = tokens.shape[1]
        if seq_len > self.pos_embed.shape[0]:
            raise ValueError(
                f"Critic sequence length {seq_len} exceeds pos_embed size {self.pos_embed.shape[0]}."
            )
        return tokens + self.pos_embed[:seq_len].unsqueeze(0)

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
            done = batch["done"].to(device=q_next_min.device, dtype=q_next_min.dtype)
            reward = batch["reward"].to(device=q_next_min.device, dtype=q_next_min.dtype)
            bootstrap = (1.0 - done) * (gamma ** action_horizon)
            target = reward + bootstrap * q_next_min

        target_heads = target.to(dtype=q_pred.dtype).unsqueeze(1).expand(-1, self.num_q_heads)
        loss = F.mse_loss(q_pred.squeeze(-1), target_heads)
        metrics = {
            "critic_loss": float(loss.detach().cpu()),
            "q_mean": float(q_pred.detach().mean().cpu()),
            "target_mean": float(target.detach().mean().cpu()),
        }
        return loss, metrics
