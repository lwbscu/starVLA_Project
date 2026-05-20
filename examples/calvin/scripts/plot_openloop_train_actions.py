#!/usr/bin/env python3
"""Plot open-loop action predictions against training-set actions.

This is a read-only diagnostic for CALVIN LeRobot parquets. It loads a policy
checkpoint, samples observation frames from training episodes, predicts an
action chunk, and plots predicted vs. ground-truth action trajectories.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


ACTION_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Policy checkpoint: steps_*_pytorch_model.pt")
    parser.add_argument("--dataset-root", required=True, type=Path, help="LeRobot dataset root, e.g. calvin_task_ABC_D")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for PNG/NPZ/manifest outputs")
    parser.add_argument("--num-episodes", type=int, default=3, help="Number of episode parquets to sample")
    parser.add_argument("--frames-per-episode", type=int, default=4, help="Observation frames sampled per episode")
    parser.add_argument("--action-horizon", type=int, default=8, help="Predicted / target chunk length")
    parser.add_argument("--unnorm-key", default="franka", help="Unnormalization key passed to PolicyServerWrapper")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Inference device")
    parser.add_argument("--use-bf16", action="store_true", help="Load policy in bfloat16")
    parser.add_argument("--max-episodes-scan", type=int, default=256, help="Max parquets to scan when selecting episodes")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def original_keys(modality: dict[str, Any], group: str) -> list[str]:
    keys = []
    for key, cfg in modality.get(group, {}).items():
        if isinstance(cfg, dict):
            keys.append(str(cfg.get("original_key", key)))
    return keys


def load_task_map(dataset_root: Path) -> dict[int, str]:
    tasks_path = dataset_root / "meta" / "tasks.jsonl"
    if not tasks_path.is_file():
        raise FileNotFoundError(f"missing task language map: {tasks_path}")
    mapping: dict[int, str] = {}
    with tasks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "task_index" not in item:
                continue
            text = item.get("task") or item.get("language") or item.get("annotation")
            if text is None:
                continue
            mapping[int(item["task_index"])] = str(text)
    if not mapping:
        raise ValueError(f"no usable task_index -> language rows in {tasks_path}")
    return mapping


def decode_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if "bytes" in value:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if "path" in value:
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).convert("RGB")
    arr = np.asarray(value)
    if arr.ndim == 3:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0 + 0.5).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    raise TypeError(f"cannot decode image value type={type(value)} shape={getattr(arr, 'shape', None)}")


def as_vector(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == object:
        arr = np.asarray(value.tolist() if hasattr(value, "tolist") else list(value), dtype=np.float32)
    return arr.astype(np.float32).reshape(-1)


def action_chunk(df: pd.DataFrame, action_key: str, start: int, horizon: int) -> np.ndarray:
    rows = [as_vector(value) for value in df[action_key].iloc[start : start + horizon]]
    if len(rows) != horizon:
        raise ValueError(f"need {horizon} action rows, got {len(rows)}")
    chunk = np.stack(rows, axis=0).astype(np.float32)
    if chunk.ndim != 2 or chunk.shape[1] != 7:
        raise ValueError(f"expected action chunk shape (H,7), got {chunk.shape}")
    return chunk


def resolve_language(row: pd.Series, task_map: dict[int, str]) -> str:
    for key in ("language", "lang", "annotation", "task"):
        if key in row and isinstance(row[key], str) and row[key].strip():
            return row[key]
    if "task_index" not in row:
        raise KeyError("row has no language column and no task_index")
    task_index = int(row["task_index"])
    if task_index not in task_map:
        raise KeyError(f"task_index={task_index} missing from meta/tasks.jsonl")
    return task_map[task_index]


def select_parquets(dataset_root: Path, max_scan: int, count: int) -> list[Path]:
    parquets = sorted((dataset_root / "data").glob("**/*.parquet"))[:max_scan]
    if not parquets:
        raise FileNotFoundError(f"no parquet files under {dataset_root / 'data'}")
    usable = []
    for path in parquets:
        try:
            if len(pd.read_parquet(path, columns=["timestamp"])) > 0:
                usable.append(path)
        except Exception:
            continue
        if len(usable) >= count:
            break
    if len(usable) < count:
        raise RuntimeError(f"only found {len(usable)} usable parquets, requested {count}")
    return usable


def plot_sample(
    out_png: Path,
    pred: np.ndarray,
    gt: np.ndarray,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    horizon = gt.shape[0]
    x = np.arange(horizon)
    fig, axes = plt.subplots(7, 1, figsize=(10, 14), sharex=True)
    for dim, ax in enumerate(axes):
        label = ACTION_NAMES[dim] if dim < len(ACTION_NAMES) else f"dim{dim}"
        ax.plot(x, gt[:, dim], label="gt", linewidth=2)
        ax.plot(x, pred[:horizon, dim], label="pred", linewidth=2, linestyle="--")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
        if dim == 0:
            ax.legend(loc="upper right")
    axes[-1].set_xlabel("chunk step")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {args.dataset_root}")
    if args.action_horizon < 1:
        raise ValueError("--action-horizon must be positive")

    info = read_json(args.dataset_root / "meta" / "info.json")
    modality = read_json(args.dataset_root / "meta" / "modality.json")
    video_keys = original_keys(modality, "video")
    state_keys = original_keys(modality, "state")
    action_keys = original_keys(modality, "action")
    if len(video_keys) < 2:
        raise ValueError(f"expected two video keys in modality.json, got {video_keys}")
    if len(state_keys) != 1 or len(action_keys) != 1:
        raise ValueError(f"expected single state/action original key, got state={state_keys}, action={action_keys}")
    state_key = state_keys[0]
    action_key = action_keys[0]
    image_key, wrist_key = video_keys[:2]
    if int(info.get("features", {}).get(state_key, {}).get("shape", [0])[0]) != 8:
        raise ValueError(f"expected state dim 8 in meta/info.json for {state_key}")
    if int(info.get("features", {}).get(action_key, {}).get("shape", [0])[0]) != 7:
        raise ValueError(f"expected action dim 7 in meta/info.json for {action_key}")

    task_map = load_task_map(args.dataset_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from deployment.model_server.policy_wrapper import PolicyServerWrapper

    policy = PolicyServerWrapper(
        ckpt_path=str(args.checkpoint),
        device=args.device,
        use_bf16=args.use_bf16,
        unnorm_key=args.unnorm_key,
    )

    parquets = select_parquets(args.dataset_root, args.max_episodes_scan, args.num_episodes)
    manifest = []
    sample_id = 0
    for episode_idx, parquet in enumerate(parquets):
        df = pd.read_parquet(parquet)
        max_start = len(df) - args.action_horizon
        if max_start < 0:
            continue
        starts = np.linspace(0, max_start, num=args.frames_per_episode, dtype=int)
        for start in starts:
            row = df.iloc[int(start)]
            gt = action_chunk(df, action_key, int(start), args.action_horizon)
            image = decode_image(row[image_key])
            wrist = decode_image(row[wrist_key])
            state = as_vector(row[state_key])
            if state.shape[0] != 8:
                raise ValueError(f"expected state dim 8, got {state.shape} in {parquet}:{start}")
            lang = resolve_language(row, task_map)
            output = policy.predict_action(
                examples=[{"image": [image, wrist], "lang": lang, "state": state[None, :]}],
                unnorm_key=args.unnorm_key,
            )
            pred = np.asarray(output["actions"])[0].astype(np.float32)
            if pred.ndim != 2 or pred.shape[1] != 7:
                raise ValueError(f"expected prediction shape (H,7), got {pred.shape}")
            pred = pred[: args.action_horizon]
            mae = float(np.mean(np.abs(pred - gt)))
            mse = float(np.mean((pred - gt) ** 2))

            stem = f"sample_{sample_id:03d}_episode_{episode_idx:03d}_frame_{int(start):06d}"
            out_png = args.output_dir / f"{stem}.png"
            out_npz = args.output_dir / f"{stem}.npz"
            plot_sample(out_png, pred, gt, title=f"{stem} | mae={mae:.4f} | {lang}")
            np.savez_compressed(out_npz, pred=pred, gt=gt, state=state, language=lang, parquet=str(parquet), frame=int(start))
            manifest.append(
                {
                    "sample_id": sample_id,
                    "episode_parquet": str(parquet),
                    "frame": int(start),
                    "language": lang,
                    "mae": mae,
                    "mse": mse,
                    "png": str(out_png),
                    "npz": str(out_npz),
                }
            )
            sample_id += 1

    if not manifest:
        raise RuntimeError("no open-loop samples were generated")
    manifest_path = args.output_dir / "manifest.json"
    summary_path = args.output_dir / "summary.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "num_samples": len(manifest),
        "mean_mae": float(np.mean([row["mae"] for row in manifest])),
        "mean_mse": float(np.mean([row["mse"] for row in manifest])),
        "manifest": str(manifest_path),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"openloop_plot=ok output_dir={args.output_dir}")
    print(f"summary={summary_path}")
    print(f"manifest={manifest_path}")
    print(f"mean_mae={summary['mean_mae']:.6f} mean_mse={summary['mean_mse']:.6f} samples={len(manifest)}")


if __name__ == "__main__":
    main()
