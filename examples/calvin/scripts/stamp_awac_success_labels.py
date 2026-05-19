#!/usr/bin/env python3
"""
Stamp terminal success labels into LeRobot episode parquets for AWAC preprocessing.

This script exists because AWAC reward preprocessing must not silently assume
success when a dataset lacks labels. Use exactly one explicit source:

  1. ``--source-dataset-root``: copy terminal success from a matching dataset.
  2. ``--all-success-demo``: assert every episode is a successful expert demo.

By default the script is a dry run. Pass ``--write`` to modify parquet files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strictly add AWAC success labels to LeRobot parquet episodes.")
    parser.add_argument("--dataset-root", required=True, help="Target LeRobot dataset root containing data/**/*.parquet.")
    parser.add_argument(
        "--source-dataset-root",
        default=None,
        help="Optional source LeRobot dataset root. Terminal success is copied by relative data path.",
    )
    parser.add_argument(
        "--all-success-demo",
        action="store_true",
        help="Explicitly assert every target episode is a successful demonstration.",
    )
    parser.add_argument("--success-column", default="success", help="Target success column to write.")
    parser.add_argument("--source-success-column", default="success", help="Source success column to read.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing success column.")
    parser.add_argument("--allow-length-mismatch", action="store_true", help="Allow source/target episode length mismatch.")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of episodes to process.")
    parser.add_argument("--manifest-jsonl", default=None, help="Optional path for per-episode JSONL audit manifest.")
    parser.add_argument("--write", action="store_true", help="Actually write parquet files. Without this, dry-run only.")
    return parser.parse_args()


def find_parquets(root: Path) -> list[Path]:
    paths = sorted(root.glob("data/**/*.parquet"))
    if not paths:
        paths = sorted(p for p in root.glob("**/*.parquet") if "meta" not in p.parts)
    return paths


def to_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.number)):
        return float(value) > 0.5
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value).reshape(-1)
        return bool(arr[-1]) if arr.size else False
    return bool(value)


def relative_data_path(dataset_root: Path, parquet: Path) -> Path:
    data_root = dataset_root / "data"
    try:
        return parquet.relative_to(data_root)
    except ValueError:
        return parquet.relative_to(dataset_root)


def source_index(source_root: Path) -> dict[Path, Path]:
    return {relative_data_path(source_root, path): path for path in find_parquets(source_root)}


def write_manifest_row(manifest_path: Path | None, row: dict[str, Any]) -> None:
    if manifest_path is None:
        return
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def atomic_write_parquet(df: Any, parquet_path: Path) -> None:
    tmp_path = parquet_path.with_name(f".{parquet_path.name}.tmp")
    df.to_parquet(tmp_path, index=False)
    tmp_path.replace(parquet_path)


def main() -> int:
    args = parse_args()
    if bool(args.source_dataset_root) == bool(args.all_success_demo):
        print("Exactly one of --source-dataset-root or --all-success-demo is required.", file=sys.stderr)
        return 2

    try:
        import pandas as pd  # noqa: F401
    except ImportError as exc:
        print(f"pandas with parquet support is required: {exc}", file=sys.stderr)
        return 2

    target_root = Path(args.dataset_root)
    target_paths = find_parquets(target_root)
    if args.limit and args.limit > 0:
        target_paths = target_paths[: args.limit]
    if not target_paths:
        print(f"No target parquet files found under {target_root}", file=sys.stderr)
        return 2

    src_by_rel: dict[Path, Path] = {}
    source_root: Path | None = None
    if args.source_dataset_root:
        source_root = Path(args.source_dataset_root)
        src_by_rel = source_index(source_root)
        if not src_by_rel:
            print(f"No source parquet files found under {source_root}", file=sys.stderr)
            return 2

    manifest_path = Path(args.manifest_jsonl) if args.manifest_jsonl else None
    if manifest_path is not None and manifest_path.exists() and args.write:
        manifest_path.unlink()

    stats = {
        "episodes": 0,
        "would_write": 0,
        "written": 0,
        "already_ok": 0,
        "success_true": 0,
        "success_false": 0,
        "failures": 0,
    }

    for target_path in target_paths:
        stats["episodes"] += 1
        rel_path = relative_data_path(target_root, target_path)
        row: dict[str, Any] = {
            "target": str(target_path),
            "relative_path": str(rel_path),
            "mode": "all_success_demo" if args.all_success_demo else "source_dataset",
            "write": bool(args.write),
        }
        try:
            target_df = pd.read_parquet(target_path)
            if target_df.empty:
                raise ValueError("target episode parquet is empty")

            if args.all_success_demo:
                terminal_success = True
                source_path = None
            else:
                source_path = src_by_rel.get(rel_path)
                if source_path is None:
                    raise FileNotFoundError(f"source parquet missing for relative path {rel_path}")
                source_df = pd.read_parquet(source_path)
                if source_df.empty:
                    raise ValueError(f"source episode parquet is empty: {source_path}")
                if not args.allow_length_mismatch and len(source_df) != len(target_df):
                    raise ValueError(
                        f"source/target length mismatch for {rel_path}: source={len(source_df)}, target={len(target_df)}"
                    )
                if args.source_success_column not in source_df.columns:
                    raise KeyError(f"source success column `{args.source_success_column}` missing in {source_path}")
                terminal_success = to_bool(source_df[args.source_success_column].iloc[-1])

            row["source"] = str(source_path) if source_path else None
            row["frames"] = int(len(target_df))
            row["terminal_success"] = bool(terminal_success)
            stats["success_true" if terminal_success else "success_false"] += 1

            if args.success_column in target_df.columns:
                current_terminal = to_bool(target_df[args.success_column].iloc[-1])
                if current_terminal == terminal_success and not args.overwrite:
                    stats["already_ok"] += 1
                    row["status"] = "already_ok"
                    write_manifest_row(manifest_path, row)
                    continue
                if not args.overwrite:
                    raise ValueError(
                        f"target already has `{args.success_column}` terminal={current_terminal}; "
                        "pass --overwrite to replace it"
                    )

            success_values = np.zeros(len(target_df), dtype=bool)
            success_values[-1] = bool(terminal_success)
            target_df[args.success_column] = success_values
            stats["would_write"] += 1
            if args.write:
                atomic_write_parquet(target_df, target_path)
                stats["written"] += 1
                row["status"] = "written"
            else:
                row["status"] = "dry_run_would_write"
        except Exception as exc:  # noqa: BLE001
            stats["failures"] += 1
            row["status"] = "failure"
            row["error"] = str(exc)
        write_manifest_row(manifest_path, row)

    mode = "write" if args.write else "dry-run"
    print(
        f"[stamp_awac_success_labels] {mode}: "
        f"episodes={stats['episodes']} would_write={stats['would_write']} written={stats['written']} "
        f"already_ok={stats['already_ok']} success_true={stats['success_true']} "
        f"success_false={stats['success_false']} failures={stats['failures']}"
    )
    if manifest_path:
        print(f"[stamp_awac_success_labels] manifest={manifest_path}")
    return 1 if stats["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
