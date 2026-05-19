"""Minimal PEFT/LoRA integration helpers for StarVLA frameworks."""

from __future__ import annotations

from typing import Any

from torch import nn


DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _cfg_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if hasattr(node, "get"):
        return node.get(key, default)
    return getattr(node, key, default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return list(DEFAULT_LORA_TARGET_MODULES)
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


def _as_optional_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _matches_prefix(name: str, prefixes: list[str]) -> bool:
    if not prefixes:
        return True
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


def _matches_suffix(name: str, suffixes: list[str]) -> bool:
    return any(name == suffix or name.endswith("." + suffix) for suffix in suffixes)


def _resolve_lora_target_modules(backbone: nn.Module, lora_cfg: Any) -> list[str]:
    """Resolve optional prefix-filtered LoRA targets for PEFT.

    ``target_modules`` keeps PEFT's usual suffix semantics.  The optional
    ``target_include_prefixes`` / ``target_exclude_prefixes`` filters are for
    experiments that must isolate QwenVL vision-tower LoRA from text-LLM LoRA.
    When no prefix filter is provided, return suffixes unchanged so existing
    training routes preserve PEFT's default matching behaviour.
    """

    suffixes = _as_list(_cfg_get(lora_cfg, "target_modules", None))
    include_prefixes = _as_optional_list(_cfg_get(lora_cfg, "target_include_prefixes", None))
    exclude_prefixes = _as_optional_list(_cfg_get(lora_cfg, "target_exclude_prefixes", None))

    if not include_prefixes and not exclude_prefixes:
        return suffixes

    matched = []
    linear_names = []
    for name, module in backbone.named_modules():
        if not name or not isinstance(module, nn.Linear):
            continue
        linear_names.append(name)
        if not _matches_prefix(name, include_prefixes):
            continue
        if exclude_prefixes and _matches_prefix(name, exclude_prefixes):
            continue
        if not _matches_suffix(name, suffixes):
            continue
        matched.append(name)

    if matched:
        print(
            "✅ Resolved LoRA target modules: "
            f"{len(matched)} modules | include={include_prefixes or '<all>'} "
            f"exclude={exclude_prefixes or '<none>'} suffixes={suffixes}"
        )
        print("   first targets: " + ", ".join(matched[:12]))
        return matched

    fail_if_empty = _as_bool(_cfg_get(lora_cfg, "fail_if_no_target_modules", True))
    if fail_if_empty:
        sample = ", ".join(linear_names[:40])
        raise RuntimeError(
            "LoRA target prefix filtering matched no Linear modules. "
            f"include={include_prefixes or '<all>'}, "
            f"exclude={exclude_prefixes or '<none>'}, suffixes={suffixes}. "
            f"First Linear module names: {sample}"
        )

    print(
        "⚠️ LoRA prefix filters matched no modules; falling back to suffix-only PEFT matching. "
        f"include={include_prefixes or '<all>'}, exclude={exclude_prefixes or '<none>'}, suffixes={suffixes}"
    )
    return suffixes


def _remove_freeze_pattern(cfg: Any, pattern_to_remove: str) -> None:
    trainer_cfg = _cfg_get(cfg, "trainer", None)
    freeze_modules = _cfg_get(trainer_cfg, "freeze_modules", "")
    if not isinstance(freeze_modules, str) or not freeze_modules:
        return

    patterns = [item.strip() for item in freeze_modules.split(",") if item.strip()]
    kept = [item for item in patterns if item != pattern_to_remove]
    if kept == patterns:
        return
    trainer_cfg.freeze_modules = ",".join(kept)


def apply_lora_if_enabled(model: Any, cfg: Any, *, sanitize_freeze_modules: bool = False) -> Any:
    """Attach PEFT LoRA adapters to the VLM backbone when ``trainer.lora.enabled`` is true.

    LoRA is applied before optimizer construction so the optimizer sees LoRA
    parameters.  When ``sanitize_freeze_modules`` is true, a top-level
    ``qwen_vl_interface`` freeze pattern is removed because PEFT already freezes
    the base backbone and that broad freeze would also freeze LoRA weights.
    """

    trainer_cfg = _cfg_get(cfg, "trainer", None)
    lora_cfg = _cfg_get(trainer_cfg, "lora", None)
    if not _as_bool(_cfg_get(lora_cfg, "enabled", False)):
        return model

    if not hasattr(model, "qwen_vl_interface") or not hasattr(model.qwen_vl_interface, "model"):
        raise RuntimeError("LoRA requires model.qwen_vl_interface.model, but this framework does not expose it.")

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError(
            "trainer.lora.enabled=true but `peft` is not installed. "
            "Install peft before running LoRA routes."
        ) from exc

    if sanitize_freeze_modules:
        _remove_freeze_pattern(cfg, "qwen_vl_interface")

    backbone = model.qwen_vl_interface.model
    target_modules = _resolve_lora_target_modules(backbone, lora_cfg)

    peft_config = LoraConfig(
        r=int(_cfg_get(lora_cfg, "r", 16)),
        lora_alpha=int(_cfg_get(lora_cfg, "alpha", 32)),
        lora_dropout=float(_cfg_get(lora_cfg, "dropout", 0.05)),
        bias=str(_cfg_get(lora_cfg, "bias", "none")),
        target_modules=target_modules,
    )

    model.qwen_vl_interface.model = get_peft_model(backbone, peft_config)
    print("✅ LoRA enabled on qwen_vl_interface.model")
    if hasattr(model.qwen_vl_interface.model, "print_trainable_parameters"):
        model.qwen_vl_interface.model.print_trainable_parameters()
    return model
