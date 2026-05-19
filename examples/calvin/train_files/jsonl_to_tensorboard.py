#!/usr/bin/env python
"""Convert StarVLA JSONL/metric logs into TensorBoard event files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def add_jsonl_scalars(writer: SummaryWriter, jsonl_path: Path, tag_prefix: str = "") -> int:
    count = 0
    with jsonl_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = record.get("step", record.get("steps"))
            if step is None:
                continue
            for key, value in record.items():
                if key in {"step", "steps"}:
                    continue
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    writer.add_scalar(f"{tag_prefix}{key}", value, int(step))
                    count += 1
    return count


def add_loss_log_scalars(writer: SummaryWriter, log_path: Path) -> int:
    import re

    text = log_path.read_text(errors="replace")
    step_pat = re.compile(r">> Step (\d+), Loss:")
    block_pat = re.compile(r">> Step (\d+), Loss:\s*\{(.*?)\}\)", re.S)
    kv_pat = re.compile(
        r"['\"]([^'\"]+)['\"]:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    )

    count = 0
    for step_s, body in block_pat.findall(text):
        step = int(step_s)
        for key, value_s in kv_pat.findall(body):
            try:
                value = float(value_s)
            except ValueError:
                continue
            writer.add_scalar(key, value, step)
            count += 1

    if count == 0:
        for match in step_pat.finditer(text):
            step = int(match.group(1))
            writer.add_scalar("step_seen", 1, step)
            count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--name", default="starVLA_logs")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.output_dir / args.name))

    total = 0
    summary_files = list(args.log_root.rglob("summary.jsonl"))
    for path in summary_files:
        rel = path.relative_to(args.log_root).as_posix()
        prefix = rel.replace("/summary.jsonl", "") + "/"
        total += add_jsonl_scalars(writer, path, tag_prefix=prefix)

    train_logs = list(args.log_root.rglob("train.log"))
    for path in train_logs:
        total += add_loss_log_scalars(writer, path)

    writer.flush()
    writer.close()
    print(
        json.dumps(
            {
                "log_root": str(args.log_root),
                "output_dir": str(args.output_dir / args.name),
                "summary_files": len(summary_files),
                "train_logs": len(train_logs),
                "scalar_points": total,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
