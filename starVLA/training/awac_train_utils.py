# Copyright 2025 starVLA community. All rights reserved.
"""Shared helpers for AWAC critic / actor training."""

from __future__ import annotations

import torch.nn as nn


def freeze_module(module: nn.Module, *, eval_mode: bool = True) -> None:
    """Disable gradients (and optionally set eval mode) for every parameter."""
    if eval_mode:
        module.eval()
    for param in module.parameters():
        param.requires_grad = False


def assert_module_frozen(module: nn.Module, name: str) -> None:
    """Raise if any parameter remains trainable."""
    trainable = [p_name for p_name, p in module.named_parameters() if p.requires_grad]
    if trainable:
        raise RuntimeError(
            f"{name} must be fully frozen but found trainable parameters: "
            f"{trainable[:8]}{'...' if len(trainable) > 8 else ''}"
        )
