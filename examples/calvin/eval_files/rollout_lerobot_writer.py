"""Write CALVIN evaluation rollouts as a LeRobot-style dataset.

This module is intentionally independent from policy inference. It only records
observations/actions produced by ``eval_calvin.py`` when explicitly enabled.
The core parquet columns follow the existing CALVIN LeRobot dataset:
``image``, ``wrist_image``, ``state``, ``actions``, ``timestamp``,
``frame_index``, ``episode_index``, ``index`` and ``task_index``.

Rollout diagnostics such as ``done``, ``success``, ``episode_success``, target
distances and joint positions are stored as extra columns plus a sidecar JSONL
file. The training dataloader ignores those extra columns, while analysis code
can consume them directly.
"""

from __future__ import annotations

import json
import math
import numbers
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image


STATE_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"]
ACTION_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def calvin_robot_obs_to_state(robot_obs: Any) -> np.ndarray:
    """Convert CALVIN ``robot_obs`` to the existing 8D CALVIN LeRobot state.

    CALVIN eval exposes ``robot_obs`` as
    ``[ee_pos(3), ee_ori(3), gripper(2), joint_pos(7)]``. The existing training
    dataset's ``state`` modality is
    ``[x, y, z, roll, pitch, yaw, pad, gripper]``. We keep that schema and store
    the full 15D robot observation in an extra ``robot_obs`` column.
    """
    arr = np.asarray(robot_obs, dtype=np.float32).reshape(-1)
    if arr.shape[0] < 7:
        raise ValueError(f"CALVIN robot_obs must contain at least 7 values, got shape={arr.shape}")
    state = np.zeros((8,), dtype=np.float32)
    state[: min(6, arr.shape[0])] = arr[: min(6, arr.shape[0])]
    state[6] = 0.0
    state[7] = float(arr[6])
    return state


def calvin_robot_obs_to_joint_pos(robot_obs: Any) -> np.ndarray:
    arr = np.asarray(robot_obs, dtype=np.float32).reshape(-1)
    if arr.shape[0] >= 15:
        return arr[-7:].astype(np.float32)
    return np.full((7,), np.nan, dtype=np.float32)


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, numbers.Number) or value is None or isinstance(value, (str, bool)):
        return value
    return repr(value)


def _vector3(value: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.shape[0] < 3 or not np.isfinite(arr[:3]).all():
        return None
    return arr[:3]


def _flatten_numeric_vectors(data: Any, prefix: str = "") -> list[tuple[str, np.ndarray]]:
    vectors: list[tuple[str, np.ndarray]] = []
    if isinstance(data, dict):
        for key, value in data.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            vectors.extend(_flatten_numeric_vectors(value, child_prefix))
    elif isinstance(data, (list, tuple, np.ndarray)):
        vec = _vector3(data)
        if vec is not None:
            vectors.append((prefix, vec))
    return vectors


def _first_matching_vector(
    vectors: list[tuple[str, np.ndarray]],
    needles: tuple[str, ...],
) -> tuple[str, np.ndarray] | None:
    lowered = [(key.lower(), key, value) for key, value in vectors]
    for key_lower, key, value in lowered:
        if all(needle in key_lower for needle in needles):
            return key, value
    return None


def _target_needles_for_subtask(subtask: str) -> list[tuple[str, ...]]:
    text = subtask.lower()
    priorities: list[tuple[str, ...]] = []
    for color in ("red", "blue", "pink"):
        if color in text:
            priorities.append((color, "block"))
            priorities.append((color,))
    if "drawer" in text:
        priorities.extend([("drawer", "handle"), ("drawer",)])
    if "slider" in text or "door" in text:
        priorities.extend([("slider",), ("door",)])
    if "button" in text:
        priorities.append(("button",))
    if "switch" in text:
        priorities.append(("switch",))
    if "led" in text:
        priorities.append(("led",))
    if "light" in text or "lamp" in text:
        priorities.extend([("lightbulb",), ("light",), ("lamp",)])
    priorities.extend([("target",), ("goal",)])
    return priorities


def estimate_rollout_distances(
    obs: dict,
    current_info: dict,
    subtask: str,
) -> dict[str, Any]:
    """Estimate useful distance signals from CALVIN obs/info.

    There is no single official "target distance" field across all CALVIN
    tasks. This records a transparent best effort:
    - ``distance_to_target``: TCP to a task-related scene vector inferred from
      the subtask name or explicit target/goal keys.
    - ``distance_to_robot_target``: TCP to controller target pose if present.

    Missing distances are stored as ``NaN`` with source ``"unavailable"``.
    """
    robot_obs = np.asarray(obs.get("robot_obs", []), dtype=np.float32).reshape(-1)
    tcp_pos = robot_obs[:3] if robot_obs.shape[0] >= 3 else None
    vectors = _flatten_numeric_vectors(current_info)

    robot_target = (
        _first_matching_vector(vectors, ("robot", "target"))
        or _first_matching_vector(vectors, ("tcp", "target"))
        or _first_matching_vector(vectors, ("eef", "target"))
    )

    task_target = None
    for needles in _target_needles_for_subtask(subtask):
        task_target = _first_matching_vector(vectors, needles)
        if task_target is not None:
            break

    def _distance(target: tuple[str, np.ndarray] | None) -> tuple[float, str]:
        if tcp_pos is None or target is None:
            return math.nan, "unavailable"
        source, vec = target
        return float(np.linalg.norm(tcp_pos - vec)), source

    target_distance, target_source = _distance(task_target)
    robot_distance, robot_source = _distance(robot_target)
    return {
        "distance_to_target": target_distance,
        "distance_to_target_source": target_source,
        "distance_to_robot_target": robot_distance,
        "distance_to_robot_target_source": robot_source,
    }


def _numeric_stats(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values)
    if arr.ndim == 1:
        arr = arr[:, None]
    arr = arr.astype(np.float64)

    finite = np.isfinite(arr)

    def _reduce(func) -> list[float]:
        cols = []
        for col_idx in range(arr.shape[1]):
            col = arr[:, col_idx]
            finite_col = col[finite[:, col_idx]]
            cols.append(float(func(finite_col)) if finite_col.size else math.nan)
        return cols

    return {
        "min": _reduce(np.min),
        "max": _reduce(np.max),
        "mean": _reduce(np.mean),
        "std": _reduce(np.std),
        "count": [int(arr.shape[0])],
    }


def _image_stats(frames: list[np.ndarray]) -> dict[str, Any]:
    if not frames:
        return {"min": [0.0], "max": [0.0], "mean": [0.0], "std": [0.0], "count": [0]}
    arr = np.stack([np.asarray(frame, dtype=np.float32) / 255.0 for frame in frames], axis=0)
    axes = tuple(range(arr.ndim - 1))
    return {
        "min": arr.min(axis=axes).reshape(-1, 1, 1).tolist(),
        "max": arr.max(axis=axes).reshape(-1, 1, 1).tolist(),
        "mean": arr.mean(axis=axes).reshape(-1, 1, 1).tolist(),
        "std": arr.std(axis=axes).reshape(-1, 1, 1).tolist(),
        "count": [int(arr.shape[0])],
    }


def _encode_png_image(frame: np.ndarray) -> dict[str, Any]:
    buffer = BytesIO()
    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(buffer, format="PNG")
    return {"bytes": buffer.getvalue(), "path": None}


def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_as_jsonable(obj), ensure_ascii=False) + "\n")


@dataclass
class RolloutEpisodeBuffer:
    episode_index: int
    sequence_index: int
    subtask_index: int
    subtask: str
    language: str
    initial_state: Any
    start_info: dict
    rows: list[dict[str, Any]] = field(default_factory=list)
    analysis_rows: list[dict[str, Any]] = field(default_factory=list)
    image_frames: list[np.ndarray] = field(default_factory=list)
    wrist_frames: list[np.ndarray] = field(default_factory=list)


class RolloutLeRobotWriter:
    """Append-only writer for CALVIN rollout episodes."""

    def __init__(
        self,
        root_dir: str | Path,
        fps: int = 10,
        chunks_size: int = 1000,
        write_videos: bool = False,
        require_target_distance: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.fps = int(fps)
        self.chunks_size = int(chunks_size)
        self.write_videos = bool(write_videos)
        self.require_target_distance = bool(require_target_distance)
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        if self.chunks_size <= 0:
            raise ValueError(f"chunks_size must be positive, got {chunks_size}")

        self.meta_dir = self.root_dir / "meta"
        self.data_dir = self.root_dir / "data"
        self.video_dir = self.root_dir / "videos"
        self.analysis_dir = self.root_dir / "rollout_analysis"
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        if self.write_videos:
            self.video_dir.mkdir(parents=True, exist_ok=True)

        self.info_path = self.meta_dir / "info.json"
        self.modality_path = self.meta_dir / "modality.json"
        self.tasks_path = self.meta_dir / "tasks.jsonl"
        self.episodes_path = self.meta_dir / "episodes.jsonl"
        self.episode_stats_path = self.meta_dir / "episodes_stats.jsonl"
        self.rollout_stats_path = self.analysis_dir / "rollout_steps.jsonl"

        self.episode_index = self._count_jsonl_lines(self.episodes_path)
        self.total_frames = self._read_total_frames()
        self.task_to_index = self._read_tasks()
        self._write_modality()
        self._write_info()

    @staticmethod
    def _count_jsonl_lines(path: Path) -> int:
        if not path.is_file():
            return 0
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    def _read_total_frames(self) -> int:
        if not self.info_path.is_file():
            return 0
        try:
            with self.info_path.open("r", encoding="utf-8") as f:
                return int(json.load(f).get("total_frames", 0))
        except (json.JSONDecodeError, ValueError):
            return 0

    def _read_tasks(self) -> dict[str, int]:
        tasks: dict[str, int] = {}
        if not self.tasks_path.is_file():
            return tasks
        with self.tasks_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                tasks[str(item["task"])] = int(item["task_index"])
        return tasks

    def _task_index(self, task: str) -> int:
        if task not in self.task_to_index:
            self.task_to_index[task] = len(self.task_to_index)
            _append_jsonl(self.tasks_path, {"task_index": self.task_to_index[task], "task": task})
        return self.task_to_index[task]

    def _write_modality(self) -> None:
        if self.modality_path.exists():
            return
        modality = {
            "state": {
                name: {"start": idx, "end": idx + 1, "original_key": "state"}
                for idx, name in enumerate(STATE_NAMES)
            },
            "action": {
                name: {"start": idx, "end": idx + 1, "original_key": "actions"}
                for idx, name in enumerate(ACTION_NAMES)
            },
            "video": {
                "primary_image": {"original_key": "image"},
                "image_0": {"original_key": "image"},
                "wrist_image": {"original_key": "wrist_image"},
            },
            "annotation": {
                "human.action.task_description": {"original_key": "task_index"},
            },
        }
        self.modality_path.write_text(json.dumps(modality, indent=4), encoding="utf-8")

    def _features(self) -> dict[str, Any]:
        image_dtype = "video" if self.write_videos else "image"
        image_info = {
            "dtype": image_dtype,
            "shape": [200, 200, 3],
            "names": ["height", "width", "channel"],
        }
        wrist_info = {
            "dtype": image_dtype,
            "shape": [84, 84, 3],
            "names": ["height", "width", "channel"],
        }
        if self.write_videos:
            image_info["info"] = {
                "video.height": 200,
                "video.width": 200,
                "video.codec": "libx264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": self.fps,
                "video.channels": 3,
                "has_audio": False,
            }
            wrist_info["info"] = {
                "video.height": 84,
                "video.width": 84,
                "video.codec": "libx264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": self.fps,
                "video.channels": 3,
                "has_audio": False,
            }
        return {
            "image": image_info,
            "wrist_image": wrist_info,
            "state": {"dtype": "float32", "shape": [8], "names": STATE_NAMES},
            "actions": {"dtype": "float32", "shape": [7], "names": ACTION_NAMES},
            "robot_obs": {"dtype": "float32", "shape": [15], "names": None},
            "robot_joint_pos": {"dtype": "float32", "shape": [7], "names": [f"joint_{i}" for i in range(7)]},
            "done": {"dtype": "bool", "shape": [1], "names": None},
            "env_done": {"dtype": "bool", "shape": [1], "names": None},
            "success": {"dtype": "bool", "shape": [1], "names": None},
            "episode_success": {"dtype": "bool", "shape": [1], "names": None},
            "distance_to_target": {"dtype": "float32", "shape": [1], "names": None},
            "distance_to_robot_target": {"dtype": "float32", "shape": [1], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }

    def _write_info(self) -> None:
        total_episodes = self.episode_index
        total_chunks = (max(total_episodes - 1, 0) // self.chunks_size) + (1 if total_episodes else 0)
        info = {
            "codebase_version": "v2.1",
            "robot_type": "panda",
            "total_episodes": total_episodes,
            "total_frames": int(self.total_frames),
            "total_tasks": len(self.task_to_index),
            "total_videos": total_episodes * 2 if self.write_videos else 0,
            "total_chunks": total_chunks,
            "chunks_size": self.chunks_size,
            "fps": self.fps,
            "splits": {"train": f"0:{total_episodes}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": self._features(),
        }
        self.info_path.write_text(json.dumps(info, indent=4), encoding="utf-8")

    def start_episode(
        self,
        sequence_index: int,
        subtask_index: int,
        subtask: str,
        language: str,
        initial_state: Any,
        start_info: dict,
    ) -> RolloutEpisodeBuffer:
        return RolloutEpisodeBuffer(
            episode_index=self.episode_index,
            sequence_index=sequence_index,
            subtask_index=subtask_index,
            subtask=subtask,
            language=language,
            initial_state=_as_jsonable(initial_state),
            start_info=_as_jsonable(start_info),
        )

    def add_step(
        self,
        episode: RolloutEpisodeBuffer,
        obs: dict,
        action: np.ndarray,
        env_done: bool,
        done: bool,
        success: bool,
        current_info: dict,
        distance_info: dict[str, Any],
    ) -> None:
        robot_obs = np.asarray(obs.get("robot_obs", []), dtype=np.float32).reshape(-1)
        if robot_obs.shape[0] < 15:
            padded_robot_obs = np.full((15,), np.nan, dtype=np.float32)
            padded_robot_obs[: robot_obs.shape[0]] = robot_obs
            robot_obs = padded_robot_obs
        else:
            robot_obs = robot_obs[:15]

        if self.require_target_distance and not math.isfinite(float(distance_info["distance_to_target"])):
            raise RuntimeError(
                "Target distance was required but unavailable. "
                f"sequence={episode.sequence_index} subtask={episode.subtask!r} step={len(episode.rows)} "
                f"distance_source={distance_info.get('distance_to_target_source')}"
            )

        frame_index = len(episode.rows)
        task_index = self._task_index(episode.language)
        image = np.asarray(obs["rgb_obs"]["rgb_static"], dtype=np.uint8)
        wrist_image = np.asarray(obs["rgb_obs"]["rgb_gripper"], dtype=np.uint8)
        episode.image_frames.append(image)
        episode.wrist_frames.append(wrist_image)

        row = {
            "image": _encode_png_image(image),
            "wrist_image": _encode_png_image(wrist_image),
            "state": calvin_robot_obs_to_state(robot_obs).tolist(),
            "actions": np.asarray(action, dtype=np.float32).reshape(7).tolist(),
            "robot_obs": robot_obs.astype(np.float32).tolist(),
            "robot_joint_pos": calvin_robot_obs_to_joint_pos(robot_obs).tolist(),
            "done": bool(done),
            "env_done": bool(env_done),
            "success": bool(success),
            "episode_success": bool(success),
            "distance_to_target": np.float32(distance_info["distance_to_target"]),
            "distance_to_robot_target": np.float32(distance_info["distance_to_robot_target"]),
            "timestamp": np.float32(frame_index / self.fps),
            "frame_index": np.int64(frame_index),
            "episode_index": np.int64(episode.episode_index),
            "index": np.int64(self.total_frames + frame_index),
            "task_index": np.int64(task_index),
        }
        episode.rows.append(row)
        episode.analysis_rows.append(
            {
                "episode_index": episode.episode_index,
                "sequence_index": episode.sequence_index,
                "subtask_index": episode.subtask_index,
                "subtask": episode.subtask,
                "language": episode.language,
                "frame_index": frame_index,
                "done": bool(done),
                "env_done": bool(env_done),
                "step_success": bool(success),
                "distance_to_target": distance_info["distance_to_target"],
                "distance_to_target_source": distance_info["distance_to_target_source"],
                "distance_to_robot_target": distance_info["distance_to_robot_target"],
                "distance_to_robot_target_source": distance_info["distance_to_robot_target_source"],
                "current_info": _as_jsonable(current_info),
            }
        )

    def finish_episode(self, episode: RolloutEpisodeBuffer, success: bool) -> Path:
        if not episode.rows:
            raise ValueError(
                f"Cannot write empty rollout episode: sequence={episode.sequence_index}, subtask={episode.subtask!r}"
            )
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("RolloutLeRobotWriter requires pandas with parquet support in the eval environment.") from exc

        terminal_frame_index = len(episode.rows) - 1
        for row in episode.rows:
            row["episode_success"] = bool(success)
            if row["frame_index"] == terminal_frame_index:
                row["success"] = bool(success)
                row["done"] = True
            else:
                row["success"] = bool(row.get("success", False))
        for row in episode.analysis_rows:
            row["episode_success"] = bool(success)
            if row["frame_index"] == terminal_frame_index:
                row["success"] = bool(success)
                row["done"] = True
            else:
                row["success"] = bool(row.get("step_success", False))

        episode_chunk = episode.episode_index // self.chunks_size
        chunk_dir = self.data_dir / f"chunk-{episode_chunk:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = chunk_dir / f"episode_{episode.episode_index:06d}.parquet"
        pd.DataFrame(episode.rows).to_parquet(parquet_path, index=False)

        if self.write_videos:
            self._write_episode_video(episode, "image", episode.image_frames)
            self._write_episode_video(episode, "wrist_image", episode.wrist_frames)

        _append_jsonl(
            self.episodes_path,
            {
                "episode_index": episode.episode_index,
                "tasks": [episode.language],
                "length": len(episode.rows),
                "success": bool(success),
                "sequence_index": episode.sequence_index,
                "subtask_index": episode.subtask_index,
                "subtask": episode.subtask,
            },
        )
        _append_jsonl(
            self.episode_stats_path,
            {"episode_index": episode.episode_index, "stats": self._episode_stats(episode)},
        )
        for analysis_row in episode.analysis_rows:
            _append_jsonl(self.rollout_stats_path, analysis_row)

        self.episode_index += 1
        self.total_frames += len(episode.rows)
        self._write_info()
        return parquet_path

    def _write_episode_video(self, episode: RolloutEpisodeBuffer, video_key: str, frames: list[np.ndarray]) -> Path:
        episode_chunk = episode.episode_index // self.chunks_size
        out_dir = self.video_dir / f"chunk-{episode_chunk:03d}" / video_key
        out_dir.mkdir(parents=True, exist_ok=True)
        video_path = out_dir / f"episode_{episode.episode_index:06d}.mp4"
        with imageio.get_writer(video_path, fps=self.fps, codec="libx264", macro_block_size=None) as writer:
            for frame in frames:
                writer.append_data(np.asarray(frame, dtype=np.uint8))
        return video_path

    def _episode_stats(self, episode: RolloutEpisodeBuffer) -> dict[str, Any]:
        rows = episode.rows
        return {
            "image": _image_stats(episode.image_frames),
            "wrist_image": _image_stats(episode.wrist_frames),
            "state": _numeric_stats(np.stack([row["state"] for row in rows])),
            "actions": _numeric_stats(np.stack([row["actions"] for row in rows])),
            "robot_obs": _numeric_stats(np.stack([row["robot_obs"] for row in rows])),
            "robot_joint_pos": _numeric_stats(np.stack([row["robot_joint_pos"] for row in rows])),
            "done": _numeric_stats(np.asarray([row["done"] for row in rows], dtype=np.int64)),
            "env_done": _numeric_stats(np.asarray([row["env_done"] for row in rows], dtype=np.int64)),
            "success": _numeric_stats(np.asarray([row["success"] for row in rows], dtype=np.int64)),
            "episode_success": _numeric_stats(np.asarray([row["episode_success"] for row in rows], dtype=np.int64)),
            "distance_to_target": _numeric_stats(np.asarray([row["distance_to_target"] for row in rows], dtype=np.float32)),
            "distance_to_robot_target": _numeric_stats(
                np.asarray([row["distance_to_robot_target"] for row in rows], dtype=np.float32)
            ),
            "timestamp": _numeric_stats(np.asarray([row["timestamp"] for row in rows], dtype=np.float32)),
            "frame_index": _numeric_stats(np.asarray([row["frame_index"] for row in rows], dtype=np.int64)),
            "episode_index": _numeric_stats(np.asarray([row["episode_index"] for row in rows], dtype=np.int64)),
            "index": _numeric_stats(np.asarray([row["index"] for row in rows], dtype=np.int64)),
            "task_index": _numeric_stats(np.asarray([row["task_index"] for row in rows], dtype=np.int64)),
        }
