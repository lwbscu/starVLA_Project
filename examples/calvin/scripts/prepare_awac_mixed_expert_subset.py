#!/usr/bin/env python3
"""Build expert episode allowlist matched to rollout size (success-only, random)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _rollout_episode_count(rollout_eval_root: Path) -> int:
    info_path = rollout_eval_root / "rollout_lerobot" / "meta" / "info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        return int(info["total_episodes"])
    episodes_path = rollout_eval_root / "rollout_lerobot" / "meta" / "episodes.jsonl"
    return len(_load_jsonl(episodes_path))


def _expert_success_episodes(expert_root: Path, expert_name: str) -> list[int]:
    dataset_path = expert_root / expert_name
    episodes_jsonl = dataset_path / "meta" / "episodes.jsonl"
    if episodes_jsonl.exists():
        success_ids = []
        for row in _load_jsonl(episodes_jsonl):
            ep_idx = int(row["episode_index"])
            if bool(row.get("success", True)):
                success_ids.append(ep_idx)
        return success_ids

    episodes_parquet = sorted((dataset_path / "meta").glob("episodes/*/*.parquet"))
    if episodes_parquet:
        success_ids = []
        for ep_file in episodes_parquet:
            df = pd.read_parquet(ep_file)
            for _, episode in df.iterrows():
                ep_idx = int(episode["episode_index"])
                if "success" in df.columns:
                    if bool(episode["success"]):
                        success_ids.append(ep_idx)
                else:
                    success_ids.append(ep_idx)
        return success_ids

    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Cannot list expert episodes under {dataset_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    data_pattern = info["data_path"]
    chunks_size = int(info.get("chunks_size", 1000))
    total = int(info["total_episodes"])
    success_ids = []
    for ep_idx in range(total):
        chunk = ep_idx // chunks_size
        parquet_path = dataset_path / data_pattern.format(episode_chunk=chunk, episode_index=ep_idx)
        if not parquet_path.exists():
            continue
        df = pd.read_parquet(parquet_path)
        if "success" not in df.columns:
            success_ids.append(ep_idx)
            continue
        last_success = bool(df["success"].iloc[-1])
        if isinstance(df["success"].iloc[-1], (list, tuple)):
            last_success = bool(df["success"].iloc[-1][0])
        if last_success:
            success_ids.append(ep_idx)
    return success_ids


def build_allowlist(
    expert_root: Path,
    expert_name: str,
    rollout_eval_root: Path,
    *,
    seed: int = 42,
    success_only: bool = True,
) -> dict:
    target_n = _rollout_episode_count(rollout_eval_root)
    expert_ids = _expert_success_episodes(expert_root, expert_name) if success_only else list(range(target_n))
    if not expert_ids:
        raise RuntimeError(f"No expert episodes found under {expert_root / expert_name}")

    rng = random.Random(seed)
    if len(expert_ids) >= target_n:
        selected = sorted(rng.sample(expert_ids, target_n))
    else:
        print(
            f"[allowlist] warning: expert success episodes ({len(expert_ids)}) < rollout ({target_n}); "
            "using all expert success episodes"
        )
        selected = sorted(expert_ids)

    return {
        "seed": seed,
        "success_only": success_only,
        "rollout_episode_count": target_n,
        "expert_episode_count_selected": len(selected),
        "calvin_task_ABC_D": selected,
        "rollout_lerobot": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-root", type=Path, required=True)
    parser.add_argument("--expert-name", type=str, default="calvin_task_ABC_D")
    parser.add_argument("--rollout-eval-root", type=Path, required=True, help="Parent of rollout_lerobot/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-success-only", action="store_true")
    args = parser.parse_args()

    payload = build_allowlist(
        args.expert_root,
        args.expert_name,
        args.rollout_eval_root,
        seed=args.seed,
        success_only=not args.no_success_only,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in payload if k not in ("calvin_task_ABC_D",)}, indent=2))
    print(f"Wrote allowlist: {args.output}")


if __name__ == "__main__":
    main()
