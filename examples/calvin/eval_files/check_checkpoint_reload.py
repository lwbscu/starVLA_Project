#!/usr/bin/env python
"""Strictly reload a StarVLA checkpoint and report route metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import read_mode_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", required=True, type=Path)
    parser.add_argument("--expected-action-chunk-size", type=int, default=8)
    parser.add_argument("--expected-unnorm-key", type=str, default="franka")
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    if not args.ckpt_path.is_file():
        raise FileNotFoundError(args.ckpt_path)

    model_cfg, norm_stats = read_mode_config(str(args.ckpt_path))
    framework = baseframework.from_pretrained(str(args.ckpt_path))
    action_model_cfg = model_cfg["framework"]["action_model"]
    if "action_horizon" in action_model_cfg:
        action_chunk_size = int(action_model_cfg["action_horizon"])
    elif "future_action_window_size" in action_model_cfg:
        action_chunk_size = int(action_model_cfg["future_action_window_size"]) + 1
    elif "num_actions_chunk" in action_model_cfg:
        action_chunk_size = int(action_model_cfg["num_actions_chunk"])
    else:
        raise ValueError("No action chunk size field found in checkpoint config")

    available_unnorm_keys = list(norm_stats.keys())
    trainable_params = int(sum(p.numel() for p in framework.parameters() if p.requires_grad))
    total_params = int(sum(p.numel() for p in framework.parameters()))
    result = {
        "ckpt_path": str(args.ckpt_path),
        "framework": model_cfg["framework"]["name"],
        "action_chunk_size": action_chunk_size,
        "expected_action_chunk_size": args.expected_action_chunk_size,
        "available_unnorm_keys": available_unnorm_keys,
        "expected_unnorm_key": args.expected_unnorm_key,
        "total_params": total_params,
        "trainable_params_after_reload": trainable_params,
        "passed": False,
        "reason": "",
    }

    if action_chunk_size != args.expected_action_chunk_size:
        result["reason"] = "action_chunk_size mismatch"
    elif args.expected_unnorm_key not in available_unnorm_keys:
        result["reason"] = "expected unnorm key missing"
    else:
        result["passed"] = True
        result["reason"] = "checkpoint strict reload and metadata checks passed"

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
