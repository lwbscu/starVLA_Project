#!/usr/bin/env python3
"""Merge multiple CALVIN rollout_lerobot shards into one LeRobot v2-style dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd


def _discover_shard_roots(src: Path) -> list[Path]:
    infos = sorted(src.glob("**/rollout_lerobot/meta/info.json"))
    if not infos:
        raise FileNotFoundError(f"No rollout_lerobot/meta/info.json under {src}")
    return [p.parents[1] for p in infos]


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def merge_rollout_shards(
    src: Path,
    dst: Path,
    *,
    force: bool = False,
    chunks_size: int = 1000,
) -> dict:
    shard_roots = _discover_shard_roots(src)
    if dst.exists():
        if not force:
            info_path = dst / "meta" / "info.json"
            if info_path.exists():
                info = json.loads(info_path.read_text(encoding="utf-8"))
                return {
                    "status": "skipped",
                    "dst": str(dst),
                    "total_episodes": int(info.get("total_episodes", 0)),
                    "shards": len(shard_roots),
                }
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    meta_dst = dst / "meta"
    meta_dst.mkdir(parents=True, exist_ok=True)
    (dst / "data").mkdir(parents=True, exist_ok=True)
    (dst / "rollout_analysis").mkdir(parents=True, exist_ok=True)

    template = shard_roots[0]
    for rel in ("meta/modality.json", "meta/info.json"):
        src_path = template / rel
        if src_path.exists():
            shutil.copy2(src_path, dst / rel)

    for name in ("episodes.jsonl", "episodes_stats.jsonl", "tasks.jsonl"):
        out = meta_dst / name
        if out.exists():
            out.unlink()

    global_index = 0
    global_frames = 0
    task_to_index: dict[str, int] = {}
    merged_episodes = 0
    total_to_merge = sum(len(_load_jsonl(s / "meta" / "episodes.jsonl")) for s in shard_roots)
    merge_t0 = time.time()
    print(
        f"[merge] start: shards={len(shard_roots)} episodes≈{total_to_merge} dst={dst}",
        flush=True,
    )

    for shard_i, shard_root in enumerate(shard_roots):
        shard_data = shard_root / "data"
        episodes = _load_jsonl(shard_root / "meta" / "episodes.jsonl")
        episode_stats = _load_jsonl(shard_root / "meta" / "episodes_stats.jsonl")
        stats_by_ep = {int(row["episode_index"]): row for row in episode_stats}
        analysis_by_ep: dict[int, list[dict]] = defaultdict(list)
        analysis_src = shard_root / "rollout_analysis" / "rollout_steps.jsonl"
        if analysis_src.exists():
            with analysis_src.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    analysis_by_ep[int(row["episode_index"])].append(row)
        print(
            f"[merge] shard {shard_i + 1}/{len(shard_roots)}: {shard_root.name} "
            f"episodes={len(episodes)}",
            flush=True,
        )

        for ep in episodes:
            old_idx = int(ep["episode_index"])
            old_chunk = old_idx // chunks_size
            src_parquet = shard_data / f"chunk-{old_chunk:03d}" / f"episode_{old_idx:06d}.parquet"
            if not src_parquet.exists():
                candidates = list(shard_data.glob(f"**/episode_{old_idx:06d}.parquet"))
                if not candidates:
                    print(f"[merge] skip missing episode parquet: {src_parquet}")
                    continue
                src_parquet = candidates[0]

            new_idx = global_index
            new_chunk = new_idx // chunks_size
            dst_chunk_dir = dst / "data" / f"chunk-{new_chunk:03d}"
            dst_chunk_dir.mkdir(parents=True, exist_ok=True)
            dst_parquet = dst_chunk_dir / f"episode_{new_idx:06d}.parquet"

            df = pd.read_parquet(src_parquet)
            if "episode_index" in df.columns:
                df["episode_index"] = new_idx
            if "index" in df.columns:
                base = int(global_frames)
                df["index"] = base + df["frame_index"].astype(int)
            df.to_parquet(dst_parquet, index=False)

            language = ep.get("tasks", [""])[0] if isinstance(ep.get("tasks"), list) else str(ep.get("tasks", ""))
            if language not in task_to_index:
                task_to_index[language] = len(task_to_index)
                _append_jsonl(meta_dst / "tasks.jsonl", {"task_index": task_to_index[language], "task": language})

            new_ep = dict(ep)
            new_ep["episode_index"] = new_idx
            _append_jsonl(meta_dst / "episodes.jsonl", new_ep)

            if old_idx in stats_by_ep:
                stat_row = dict(stats_by_ep[old_idx])
                stat_row["episode_index"] = new_idx
                _append_jsonl(meta_dst / "episodes_stats.jsonl", stat_row)

            for row in analysis_by_ep.get(old_idx, []):
                row = dict(row)
                row["episode_index"] = new_idx
                _append_jsonl(dst / "rollout_analysis" / "rollout_steps.jsonl", row)

            global_index += 1
            merged_episodes += 1
            global_frames += int(ep.get("length", len(df)))
            if merged_episodes == 1 or merged_episodes % 50 == 0 or merged_episodes == total_to_merge:
                elapsed = time.time() - merge_t0
                rate = merged_episodes / max(elapsed, 1e-6)
                print(
                    f"[merge] progress {merged_episodes}/{total_to_merge} "
                    f"({100.0 * merged_episodes / max(total_to_merge, 1):.1f}%) "
                    f"elapsed={elapsed:.0f}s ~{rate:.1f} ep/s",
                    flush=True,
                )

    total_chunks = (max(global_index - 1, 0) // chunks_size) + (1 if global_index else 0)
    info_path = dst / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    info.update(
        {
            "total_episodes": global_index,
            "total_frames": int(global_frames),
            "total_tasks": len(task_to_index),
            "total_chunks": total_chunks,
            "chunks_size": chunks_size,
            "splits": {"train": f"0:{global_index}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        }
    )
    info_path.write_text(json.dumps(info, indent=4), encoding="utf-8")
    print(
        f"[merge] done: episodes={global_index} frames={global_frames} "
        f"elapsed={time.time() - merge_t0:.0f}s",
        flush=True,
    )

    return {
        "status": "merged",
        "dst": str(dst),
        "shards": len(shard_roots),
        "total_episodes": global_index,
        "total_frames": int(global_frames),
        "merged_episodes": merged_episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge CALVIN rollout_lerobot shards.")
    parser.add_argument("--src", type=Path, required=True, help="Root containing shard_*/.../rollout_lerobot")
    parser.add_argument("--dst", type=Path, required=True, help="Output rollout_lerobot directory")
    parser.add_argument("--force", action="store_true", help="Rebuild even if dst already exists")
    parser.add_argument("--chunks-size", type=int, default=1000)
    args = parser.parse_args()
    result = merge_rollout_shards(args.src, args.dst, force=args.force, chunks_size=args.chunks_size)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
