#!/usr/bin/env python
"""Check StarVLA training loss logs for finite and non-exploding losses."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import median


LOSS_PATTERNS = [
    re.compile(r"action_dit_loss['\"]?:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", re.S),
    re.compile(r"action_loss['\"]?:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", re.S),
]


def parse_losses(log_path: Path) -> list[float]:
    text = log_path.read_text(errors="replace")
    losses: list[float] = []
    for pattern in LOSS_PATTERNS:
        matches = [float(match.group(1)) for match in pattern.finditer(text)]
        if matches:
            losses.extend(matches)
            break
    return losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-log", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--max-median-ratio", type=float, default=1.2)
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()

    losses = parse_losses(args.train_log)
    finite = all(math.isfinite(value) for value in losses)
    result = {
        "train_log": str(args.train_log),
        "num_losses": len(losses),
        "finite": finite,
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "first_window_median": None,
        "last_window_median": None,
        "median_ratio": None,
        "max_median_ratio": args.max_median_ratio,
        "passed": False,
        "reason": "",
    }

    if not losses:
        result["reason"] = "no loss values found"
    elif not finite:
        result["reason"] = "loss contains NaN or Inf"
    elif len(losses) >= args.window * 2:
        first_med = median(losses[: args.window])
        last_med = median(losses[-args.window :])
        ratio = (last_med / first_med) if first_med != 0 else float("inf")
        result.update(
            {
                "first_window_median": first_med,
                "last_window_median": last_med,
                "median_ratio": ratio,
            }
        )
        if ratio <= args.max_median_ratio:
            result["passed"] = True
            result["reason"] = "finite and median ratio within threshold"
        else:
            result["reason"] = "loss median ratio exceeded threshold"
    else:
        result["passed"] = True
        result["reason"] = "finite losses; not enough values for median-ratio check"

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
