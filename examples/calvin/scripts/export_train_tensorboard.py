#!/usr/bin/env python
"""Export completed CALVIN debug runs into TensorBoard event files."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

from torch.utils.tensorboard import SummaryWriter


METRIC_BLOCK_RE = re.compile(r"Step\s+(\d+),\s+Loss:.*?(\{.*?\}\))", re.DOTALL)
METRIC_PAIR_RE = re.compile(
    r"'([^']+)':\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def finite_float(value) -> float | None:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(scalar):
        return None
    return scalar


def clean_tag(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_./-]+", "_", text).strip("_") or "unnamed"


def iter_train_metrics(train_log: Path) -> Iterable[tuple[int, dict[str, float]]]:
    text = train_log.read_text(encoding="utf-8", errors="ignore")
    text = ANSI_RE.sub("", text)
    for match in METRIC_BLOCK_RE.finditer(text):
        step = int(match.group(1))
        metrics: dict[str, float] = {}
        for key, raw_value in METRIC_PAIR_RE.findall(match.group(2)):
            scalar = finite_float(raw_value)
            if scalar is not None:
                metrics[key] = scalar
        if metrics:
            yield step, metrics


def export_train_log(writer: SummaryWriter, train_log: Path) -> int:
    count = 0
    for step, metrics in iter_train_metrics(train_log):
        for key, value in metrics.items():
            writer.add_scalar(key, value, step)
        count += 1
    return count


def export_eval_results(writer: SummaryWriter, results_json: Path, step: int) -> int:
    if not results_json.exists():
        return 0
    payload = json.loads(results_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in {results_json}")

    count = 0
    for _, result in payload.items():
        if not isinstance(result, dict):
            continue
        avg_seq_len = finite_float(result.get("avg_seq_len"))
        if avg_seq_len is not None:
            writer.add_scalar("eval/avg_seq_len", avg_seq_len, step)
            count += 1

        chain_sr = result.get("chain_sr", {})
        if isinstance(chain_sr, dict):
            for horizon, value in chain_sr.items():
                scalar = finite_float(value)
                if scalar is not None:
                    writer.add_scalar(f"eval/chain_sr/{clean_tag(str(horizon))}", scalar, step)
                    count += 1

        task_info = result.get("task_info", {})
        if isinstance(task_info, dict):
            for task_name, task_result in task_info.items():
                if not isinstance(task_result, dict):
                    continue
                total = finite_float(task_result.get("total"))
                success = finite_float(task_result.get("success"))
                if total and success is not None:
                    writer.add_scalar(f"eval/task_success/{clean_tag(task_name)}", success / total, step)
                    count += 1
    return count


def export_rollout_info(writer: SummaryWriter, info_json: Path, step: int) -> int:
    if not info_json.exists():
        return 0
    payload = json.loads(info_json.read_text(encoding="utf-8"))
    count = 0
    for key in ("total_episodes", "total_frames", "total_tasks", "total_videos", "fps"):
        scalar = finite_float(payload.get(key))
        if scalar is not None:
            writer.add_scalar(f"rollout/{key}", scalar, step)
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train-log", type=Path, default=None)
    parser.add_argument("--eval-results-json", type=Path, default=None)
    parser.add_argument("--rollout-info-json", type=Path, default=None)
    args = parser.parse_args()

    log_dir = args.log_dir
    train_log = args.train_log or log_dir / "terminal" / "train.log"
    output_dir = args.output_dir or log_dir / "tensorboard"
    eval_results_json = args.eval_results_json or log_dir / "mp4_3seq" / "results.json"
    rollout_info_json = (
        args.rollout_info_json
        or log_dir / "rollout_lerobot_debug_3seq" / "meta" / "info.json"
    )

    if not train_log.exists():
        raise FileNotFoundError(f"Training log not found: {train_log}")

    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_dir))
    try:
        train_count = export_train_log(writer, train_log)
        last_step = 0
        for step, _ in iter_train_metrics(train_log):
            last_step = step
        eval_count = export_eval_results(writer, eval_results_json, last_step)
        rollout_count = export_rollout_info(writer, rollout_info_json, last_step)
        writer.flush()
    finally:
        writer.close()

    total = train_count + eval_count + rollout_count
    if total == 0:
        raise RuntimeError(f"No scalar metrics were exported from {log_dir}")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "train_steps_exported": train_count,
                "eval_scalars_exported": eval_count,
                "rollout_scalars_exported": rollout_count,
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
