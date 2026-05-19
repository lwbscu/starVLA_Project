#!/usr/bin/env python3
"""Inspect shared dataset trees for StarVLA pretraining/eval readiness.

The script is intentionally read-only. It detects:

* LeRobot-style training roots: meta/info.json + data/
* CALVIN eval roots: .hydra/merged_config.yaml under direct/training/validation
* raw dataset hints: npz/parquet/mp4/json/jsonl/zarr/hdf5 counts

It prints a Markdown report and can optionally save a JSON summary.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


COUNT_EXTS = {
    ".parquet",
    ".mp4",
    ".npz",
    ".npy",
    ".json",
    ".jsonl",
    ".hdf5",
    ".h5",
    ".zarr",
    ".jpg",
    ".jpeg",
    ".png",
    ".txt",
    ".yaml",
    ".yml",
}


def human_size(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "unknown"
    value = float(num_bytes)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if value < 1024 or unit == "TiB":
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024
    return f"{value:.1f}TiB"


def safe_read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def count_files(root: Path, max_files: int | None = None) -> tuple[Counter[str], int, int]:
    counts: Counter[str] = Counter()
    total_files = 0
    total_bytes = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        total_files += 1
        if max_files is not None and total_files > max_files:
            break
        ext = path.suffix.lower()
        if ext in COUNT_EXTS:
            counts[ext] += 1
        try:
            total_bytes += path.stat().st_size
        except OSError:
            pass
    return counts, total_files, total_bytes


def direct_children(root: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        children = sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name))
    except OSError:
        return rows
    for child in children[:limit]:
        entry: dict[str, Any] = {"name": child.name, "type": "dir" if child.is_dir() else "file"}
        if child.is_file():
            try:
                entry["size"] = child.stat().st_size
            except OSError:
                entry["size"] = None
        rows.append(entry)
    return rows


def discover_lerobot_roots(root: Path) -> list[Path]:
    roots: set[Path] = set()
    for info in root.rglob("meta/info.json"):
        dataset_root = info.parents[1]
        if (dataset_root / "data").is_dir():
            roots.add(dataset_root)
    return sorted(roots)


def discover_calvin_eval_roots(root: Path) -> list[dict[str, str]]:
    rows: dict[Path, dict[str, str]] = {}
    for cfg in root.rglob(".hydra/merged_config.yaml"):
        split_dir = cfg.parents[1]
        if split_dir.name in {"training", "validation"}:
            eval_root = split_dir.parent
            split = split_dir.name
        else:
            eval_root = split_dir
            split = "direct"
        rows.setdefault(eval_root, {"path": str(eval_root), "splits": ""})
        splits = set(filter(None, rows[eval_root]["splits"].split(",")))
        splits.add(split)
        rows[eval_root]["splits"] = ",".join(sorted(splits))
    return [rows[p] for p in sorted(rows)]


def sample_parquet_columns(path: Path) -> dict[str, Any]:
    try:
        import pandas as pd  # type: ignore

        df = pd.read_parquet(path)
        return {
            "reader": "pandas",
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        }
    except Exception as pandas_exc:
        try:
            import pyarrow.parquet as pq  # type: ignore

            schema = pq.read_schema(path)
            return {
                "reader": "pyarrow",
                "columns": schema.names,
                "schema": str(schema),
            }
        except Exception as pyarrow_exc:
            return {
                "reader": "unavailable",
                "error": f"pandas={pandas_exc!r}; pyarrow={pyarrow_exc!r}",
            }


def inspect_lerobot_root(dataset_root: Path, sample_parquet: bool) -> dict[str, Any]:
    meta = dataset_root / "meta"
    info = safe_read_json(meta / "info.json")
    modality = safe_read_json(meta / "modality.json")
    embodiment = safe_read_json(meta / "embodiment.json")
    data_dir = dataset_root / "data"
    videos_dir = dataset_root / "videos"
    parquet_files = sorted(data_dir.rglob("*.parquet")) if data_dir.is_dir() else []
    video_files = sorted(videos_dir.rglob("*.mp4")) if videos_dir.is_dir() else []

    row: dict[str, Any] = {
        "path": str(dataset_root),
        "name": dataset_root.name,
        "has_info": (meta / "info.json").is_file(),
        "has_modality": (meta / "modality.json").is_file(),
        "has_embodiment": (meta / "embodiment.json").is_file(),
        "parquet_count": len(parquet_files),
        "mp4_count": len(video_files),
        "meta_files": sorted(p.name for p in meta.glob("*")) if meta.is_dir() else [],
    }
    if isinstance(info, dict):
        row["info"] = {
            key: info.get(key)
            for key in [
                "codebase_version",
                "robot_type",
                "total_episodes",
                "total_frames",
                "total_tasks",
                "total_videos",
                "fps",
                "splits",
            ]
            if key in info
        }
        features = info.get("features")
        if isinstance(features, dict):
            row["feature_keys"] = sorted(features.keys())
            row["features"] = features
    if isinstance(modality, dict):
        row["modality"] = modality
    if isinstance(embodiment, dict):
        row["embodiment"] = embodiment
    if sample_parquet and parquet_files:
        row["sample_parquet"] = str(parquet_files[0])
        row["sample_parquet_schema"] = sample_parquet_columns(parquet_files[0])
    return row


def scan_root(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    if not root.is_dir():
        return {"root": str(root), "exists": False}

    ext_counts, total_files, total_bytes = count_files(root, args.max_count_files)
    lerobot_roots = discover_lerobot_roots(root)
    calvin_roots = discover_calvin_eval_roots(root)

    top_ext_dirs: dict[str, Counter[str]] = defaultdict(Counter)
    for child in root.iterdir():
        if not child.is_dir():
            continue
        counts, _total, _bytes = count_files(child, args.max_count_files_per_child)
        top_ext_dirs[child.name].update(counts)

    return {
        "root": str(root),
        "exists": True,
        "direct_children": direct_children(root, args.children_limit),
        "total_files_seen": total_files,
        "total_bytes_seen": total_bytes,
        "extension_counts": dict(sorted(ext_counts.items())),
        "top_level_extension_counts": {
            key: dict(sorted(counter.items()))
            for key, counter in sorted(top_ext_dirs.items())
            if counter
        },
        "lerobot_roots": [inspect_lerobot_root(path, args.sample_parquet) for path in lerobot_roots],
        "calvin_eval_roots": calvin_roots,
    }


def print_markdown(report: dict[str, Any]) -> None:
    for root_report in report["roots"]:
        print(f"# Dataset Layout: `{root_report['root']}`")
        print()
        if not root_report.get("exists"):
            print("missing: true")
            print()
            continue

        print("## Summary")
        print()
        print(f"- files_seen: {root_report['total_files_seen']}")
        print(f"- bytes_seen: {human_size(root_report['total_bytes_seen'])}")
        print(f"- extension_counts: `{json.dumps(root_report['extension_counts'], ensure_ascii=False)}`")
        print(f"- lerobot_roots: {len(root_report['lerobot_roots'])}")
        print(f"- calvin_eval_roots: {len(root_report['calvin_eval_roots'])}")
        print()

        print("## Direct Children")
        print()
        for child in root_report["direct_children"]:
            size = f" {human_size(child.get('size'))}" if child["type"] == "file" else ""
            print(f"- {child['type']}: `{child['name']}`{size}")
        print()

        if root_report["calvin_eval_roots"]:
            print("## CALVIN Eval Roots")
            print()
            for item in root_report["calvin_eval_roots"]:
                print(f"- `{item['path']}` splits={item['splits']}")
            print()

        if root_report["lerobot_roots"]:
            print("## LeRobot Roots")
            print()
            for item in root_report["lerobot_roots"]:
                print(f"### `{item['path']}`")
                print()
                print(f"- parquet_count: {item['parquet_count']}")
                print(f"- mp4_count: {item['mp4_count']}")
                print(f"- meta_files: `{', '.join(item['meta_files'])}`")
                if item.get("info"):
                    print(f"- info: `{json.dumps(item['info'], ensure_ascii=False)}`")
                if item.get("feature_keys"):
                    print(f"- feature_keys: `{', '.join(item['feature_keys'])}`")
                if item.get("modality"):
                    print(f"- modality_keys: `{', '.join(sorted(item['modality'].keys()))}`")
                if item.get("sample_parquet_schema"):
                    schema = item["sample_parquet_schema"]
                    print(f"- sample_parquet: `{item.get('sample_parquet')}`")
                    print(f"- sample_columns: `{', '.join(schema.get('columns', []))}`")
                print()

        if root_report["top_level_extension_counts"]:
            print("## Top-Level Extension Counts")
            print()
            for name, counts in root_report["top_level_extension_counts"].items():
                print(f"- `{name}`: `{json.dumps(counts, ensure_ascii=False)}`")
            print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path, help="Dataset roots to inspect.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path for machine-readable JSON report.")
    parser.add_argument("--sample-parquet", action="store_true", help="Read one parquet per LeRobot root to show columns.")
    parser.add_argument("--children-limit", type=int, default=80)
    parser.add_argument("--max-count-files", type=int, default=None, help="Cap recursive file counting per root.")
    parser.add_argument("--max-count-files-per-child", type=int, default=20000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = {"roots": [scan_root(root, args) for root in args.roots]}
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print_markdown(report)


if __name__ == "__main__":
    main()
