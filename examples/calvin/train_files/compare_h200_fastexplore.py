#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


def read_text(path: Path) -> str:
    return path.read_text(errors="replace") if path.is_file() else ""


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(errors="replace"))
    except Exception:
        return None


def parse_key(text: str, key: str) -> str:
    match = re.search(rf"^{re.escape(key)}=(.*)$", text, re.M)
    return match.group(1).strip() if match else ""


def parse_stage_step(log_dir: Path, train_text: str) -> tuple[str, int]:
    run_id = parse_key(train_text, "RUN_ID")
    if not run_id:
        run_id = log_dir.name
    step_match = re.search(r"_(\d+)step", run_id)
    stage = run_id.split("_")[0] if "_" in run_id else ""
    return stage, int(step_match.group(1)) if step_match else 0


def best_eval(log_dir: Path) -> tuple[float | None, str]:
    best_score: float | None = None
    best_path = ""
    for result_path in log_dir.rglob("results.json"):
        data = read_json(result_path)
        if not isinstance(data, dict):
            continue
        for value in data.values():
            if isinstance(value, dict) and "avg_seq_len" in value:
                try:
                    score = float(value["avg_seq_len"])
                except Exception:
                    continue
                if best_score is None or score > best_score:
                    best_score = score
                    best_path = str(result_path)
    return best_score, best_path


def collect(log_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for train_log in sorted(log_root.rglob("terminal/train.log")):
        log_dir = train_log.parents[1]
        train_text = read_text(train_log)
        route = parse_key(train_text, "ROUTE") or "unknown"
        stage, steps = parse_stage_step(log_dir, train_text)
        loss_json = read_json(log_dir / "metrics/loss_check.json") or {}
        reload_files = sorted((log_dir / "metrics").glob("reload_check_steps_*.json"))
        reload_json = read_json(reload_files[-1]) if reload_files else None
        ckpts = sorted(log_dir.rglob("steps_*_pytorch_model.pt"))
        eval_score, eval_path = best_eval(log_dir)
        loss_passed = loss_json.get("passed")
        reload_passed = reload_json.get("passed") if reload_json else None
        last_loss = loss_json.get("last_loss")
        first_loss = loss_json.get("first_loss")
        median_ratio = loss_json.get("median_ratio")

        stage_score = steps
        if loss_passed is True:
            stage_score += 100
        if reload_passed is True:
            stage_score += 100
        if eval_score is not None:
            stage_score += 1000 + eval_score * 100
        if not ckpts:
            stage_score -= 10000
        if loss_passed is False or reload_passed is False:
            stage_score -= 1000
        if isinstance(last_loss, (int, float)) and math.isfinite(float(last_loss)):
            stage_score -= min(float(last_loss), 1000.0) / 1000.0

        rows.append(
            {
                "route": route,
                "stage": stage,
                "steps": steps,
                "base_vlm": parse_key(train_text, "BASE_VLM"),
                "gpus": parse_key(train_text, "CUDA_VISIBLE_DEVICES"),
                "num_processes": parse_key(train_text, "NUM_PROCESSES"),
                "obs_image_size": parse_key(train_text, "OBS_IMAGE_SIZE"),
                "loss_passed": loss_passed,
                "first_loss": first_loss,
                "last_loss": last_loss,
                "median_ratio": median_ratio,
                "reload_passed": reload_passed,
                "eval_avg_seq_len": eval_score,
                "eval_results": eval_path,
                "num_checkpoints": len(ckpts),
                "latest_checkpoint": str(ckpts[-1]) if ckpts else "",
                "log_dir": str(log_dir),
                "score": stage_score,
            }
        )
    rows.sort(key=lambda item: (item["route"], item["steps"], item["score"]))
    return rows


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "compare_routes.csv"
    md_path = output_dir / "compare_routes.md"
    fields = [
        "route",
        "stage",
        "steps",
        "loss_passed",
        "last_loss",
        "median_ratio",
        "reload_passed",
        "eval_avg_seq_len",
        "num_checkpoints",
        "base_vlm",
        "gpus",
        "num_processes",
        "latest_checkpoint",
        "eval_results",
        "log_dir",
        "score",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    best_by_route: dict[str, dict[str, Any]] = {}
    for row in rows:
        current = best_by_route.get(row["route"])
        if current is None or row["score"] > current["score"]:
            best_by_route[row["route"]] = row
    ranking = sorted(best_by_route.values(), key=lambda item: item["score"], reverse=True)

    lines = [
        "# H200 Fast Explore Compare",
        "",
        "排序规则：优先使用 eval `avg_seq_len`；如果还没有 eval，则按 checkpoint 阶段、loss 检查、reload 检查排序。",
        "",
        "| Rank | Route | Stage | Steps | Loss OK | Reload OK | Eval Avg Len | Last Loss | Checkpoints | Log Dir |",
        "| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for idx, row in enumerate(ranking, start=1):
        eval_value = row["eval_avg_seq_len"]
        lines.append(
            f"| {idx} | {row['route']} | {row['stage']} | {row['steps']} | "
            f"{row['loss_passed']} | {row['reload_passed']} | "
            f"{'' if eval_value is None else eval_value} | "
            f"{'' if row['last_loss'] is None else row['last_loss']} | "
            f"{row['num_checkpoints']} | `{row['log_dir']}` |"
        )
    lines.extend(["", f"CSV: `{csv_path}`", ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(md_path)
    print(csv_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, default=Path("logs/h200_fastexplore"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/h200_fastexplore/summary"))
    args = parser.parse_args()
    rows = collect(args.log_root)
    if not rows:
        raise SystemExit(f"No train.log found under {args.log_root}")
    write_outputs(rows, args.output_dir)


if __name__ == "__main__":
    main()
