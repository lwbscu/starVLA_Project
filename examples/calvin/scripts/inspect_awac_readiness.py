#!/usr/bin/env python3
"""
Inspect whether a CALVIN PI-State BC run and LeRobot dataset are ready for AWAC.

This script is intentionally read-only. It checks the invariants documented in
``examples/calvin/train_files/critic.md`` without modifying checkpoints or
dataset parquet files.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_CONFIG_PATH = REPO_ROOT / "examples/calvin/train_files/data_registry/data_config.py"

DEFAULT_BATCH_ROOT = (
    "/inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project/"
    "logs/20260519_h200_server1_pi_state_30k_v1"
)
DEFAULT_DATA_ROOT = "/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d"
DEFAULT_DATASET_NAME = "calvin_task_ABC_D"
DEFAULT_DATA_MIX = "calvin_abc_d_h200"
DEFAULT_JOBS = (
    "server1_pi_state/qwen35_0p8b",
    "server1_pi_state/qwen35_4b",
    "server1_pi_state/qwen35_9b",
)

CHECKPOINT_RE = re.compile(r"steps_(\d+)_pytorch_model\.pt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only AWAC readiness inspector for H200 CALVIN PI-State runs."
    )
    parser.add_argument("--batch-root", default=DEFAULT_BATCH_ROOT, help="BC training batch root to inspect.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="LeRobot data root directory.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME, help="Dataset subdirectory name.")
    parser.add_argument("--data-mix", default=DEFAULT_DATA_MIX, help="Registered data mixture name.")
    parser.add_argument(
        "--job",
        action="append",
        default=None,
        help="Relative model job path under batch root. Can be repeated.",
    )
    parser.add_argument("--expected-step", type=int, default=30000, help="Expected BC checkpoint step.")
    parser.add_argument("--action-horizon", type=int, default=8, help="AWAC action chunk horizon H.")
    parser.add_argument("--gamma", type=float, default=0.996, help="AWAC reward discount gamma.")
    parser.add_argument("--state-dim", type=int, default=8, help="Required PI-State dimension.")
    parser.add_argument("--action-dim", type=int, default=7, help="Required action dimension.")
    parser.add_argument("--sample-parquets", type=int, default=64, help="Number of dataset parquets to sample.")
    parser.add_argument(
        "--require-preprocessed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require step_reward/reward/done columns for immediate critic training.",
    )
    parser.add_argument(
        "--require-rollout",
        action="store_true",
        help="Treat missing rollout diagnostics as a failure instead of a warning.",
    )
    parser.add_argument(
        "--rollout-root",
        default=None,
        help="Optional rollout/eval root to inspect for results.json and rollout artifacts.",
    )
    parser.add_argument("--json-out", default=None, help="Optional path for a JSON report.")
    return parser.parse_args()


def load_dataset_mixtures() -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("h200_calvin_data_config", DATA_CONFIG_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import data config from {DATA_CONFIG_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mixtures = getattr(module, "DATASET_NAMED_MIXTURES", None)
    if not isinstance(mixtures, dict):
        raise RuntimeError(f"DATASET_NAMED_MIXTURES missing in {DATA_CONFIG_PATH}")
    return mixtures


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


class Report:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(self, section: str, status: str, name: str, detail: str, **data: Any) -> None:
        self.checks.append(
            {
                "section": section,
                "status": status,
                "name": name,
                "detail": detail,
                **{k: jsonable(v) for k, v in data.items()},
            }
        )

    @property
    def has_failures(self) -> bool:
        return any(check["status"] == "FAIL" for check in self.checks)

    def status_counts(self) -> dict[str, int]:
        return dict(Counter(check["status"] for check in self.checks))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def parse_env_log(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.is_file():
        return env
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip().lower()] = value.strip()
    return env


def boolish(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    return None


def find_run_dir(job_root: Path) -> Path | None:
    if not job_root.is_dir():
        return None
    env_logs = sorted(job_root.glob("*/terminal/env.log"))
    if env_logs:
        return env_logs[-1].parents[1]
    checkpoints = sorted(job_root.glob("*/checkpoints"))
    if checkpoints:
        return checkpoints[-1].parent
    dirs = sorted([p for p in job_root.iterdir() if p.is_dir()])
    return dirs[-1] if dirs else None


def find_latest_checkpoint(run_dir: Path) -> tuple[Path | None, int | None, list[tuple[int, str]]]:
    found: list[tuple[int, str]] = []
    for path in run_dir.glob("**/steps_*_pytorch_model.pt"):
        match = CHECKPOINT_RE.search(path.name)
        if not match:
            continue
        found.append((int(match.group(1)), str(path)))
    found.sort()
    if not found:
        return None, None, []
    step, path = found[-1]
    return Path(path), step, found


def feature_shape(info: dict[str, Any], feature_key: str) -> list[int] | None:
    feature = info.get("features", {}).get(feature_key)
    shape = feature.get("shape") if isinstance(feature, dict) else None
    if isinstance(shape, list) and all(isinstance(x, (int, float)) for x in shape):
        return [int(x) for x in shape]
    return None


def modality_original_keys(modality: dict[str, Any], group: str) -> list[str]:
    group_value = modality.get(group)
    if isinstance(group_value, dict):
        candidates = group_value.values()
    elif isinstance(group_value, list):
        candidates = group_value
    else:
        return []

    keys: list[str] = []
    for item in candidates:
        if isinstance(item, dict):
            original_key = item.get("original_key") or item.get("key")
            if isinstance(original_key, str):
                keys.append(original_key)
        elif isinstance(item, str):
            keys.append(item)
    return list(dict.fromkeys(keys))


def scalar_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.number)):
        return float(value) > 0.5
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value).reshape(-1)
        return bool(arr[-1]) if arr.size else False
    return bool(value)


def numeric_scalar(value: Any) -> float:
    if isinstance(value, (int, float, np.number)):
        return float(value)
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value).reshape(-1)
        return float(arr[0]) if arr.size else 0.0
    return float(value)


def sample_dimension(df: Any, column: str) -> int | None:
    if column not in df.columns or df.empty:
        return None
    value = df[column].iloc[0]
    arr = np.asarray(value)
    if arr.ndim == 0:
        return 1
    return int(arr.reshape(-1).shape[0])


def check_registry(report: Report, data_mix: str, dataset_name: str) -> None:
    section = "registry"
    try:
        mixtures = load_dataset_mixtures()
    except Exception as exc:  # noqa: BLE001 - report should explain import failures.
        report.add(section, "FAIL", "load data registry", str(exc))
        return

    spec = mixtures.get(data_mix)
    if spec is None:
        report.add(section, "FAIL", "data mix exists", f"{data_mix!r} not found in registry")
        return
    first_names = [str(item[0]) for item in spec if isinstance(item, (tuple, list)) and item]
    if dataset_name in first_names:
        report.add(section, "PASS", "data mix mapping", f"{data_mix} -> {dataset_name}", spec=spec)
    else:
        report.add(
            section,
            "FAIL",
            "data mix mapping",
            f"{data_mix} does not point to {dataset_name}; registered names={first_names}",
            spec=spec,
        )


def check_bc_runs(
    report: Report,
    batch_root: Path,
    jobs: list[str],
    expected_step: int,
    state_dim: int,
    action_dim: int,
    action_horizon: int,
) -> None:
    section = "bc_run"
    if not batch_root.is_dir():
        report.add(section, "FAIL", "batch root exists", f"missing batch root: {batch_root}")
        return
    report.add(section, "PASS", "batch root exists", str(batch_root))

    for job in jobs:
        job_root = batch_root / job
        run_dir = find_run_dir(job_root)
        label = job.replace("/", "__")
        if run_dir is None:
            report.add(section, "FAIL", f"{label} run dir", f"cannot find run directory under {job_root}")
            continue
        report.add(section, "PASS", f"{label} run dir", str(run_dir))

        env = parse_env_log(run_dir / "terminal/env.log")
        if not env:
            report.add(section, "WARN", f"{label} env log", f"missing terminal/env.log under {run_dir}")
        else:
            include_state = boolish(env.get("include_state"))
            env_state_dim = env.get("state_dim")
            env_pi_state_dim = env.get("pi_state_dim")
            env_action_dim = env.get("action_dim")
            env_action_horizon = env.get("action_horizon")
            route = env.get("route")
            route_label = env.get("route_label")
            if route == "p4_pi" and route_label == "pi_state":
                report.add(section, "PASS", f"{label} route", "route=p4_pi route_label=pi_state")
            else:
                report.add(section, "FAIL", f"{label} route", f"expected p4_pi/pi_state, got {route}/{route_label}")
            if include_state is True:
                report.add(section, "PASS", f"{label} include_state", "include_state=true")
            else:
                report.add(section, "FAIL", f"{label} include_state", f"expected true, got {env.get('include_state')!r}")
            if env_state_dim == str(state_dim) and env_pi_state_dim == str(state_dim):
                report.add(section, "PASS", f"{label} state dim", f"state_dim={state_dim}, pi_state_dim={state_dim}")
            else:
                report.add(
                    section,
                    "FAIL",
                    f"{label} state dim",
                    f"expected state_dim/pi_state_dim={state_dim}, got {env_state_dim}/{env_pi_state_dim}",
                )
            if env_action_dim is None and env_action_horizon is None:
                report.add(section, "WARN", f"{label} action contract", "env log lacks action_dim/action_horizon")
            elif env_action_dim == str(action_dim) and env_action_horizon == str(action_horizon):
                report.add(
                    section,
                    "PASS",
                    f"{label} action contract",
                    f"action_dim={action_dim}, action_horizon={action_horizon}",
                )
            else:
                report.add(
                    section,
                    "FAIL",
                    f"{label} action contract",
                    f"expected action_dim/action_horizon={action_dim}/{action_horizon}, got {env_action_dim}/{env_action_horizon}",
                )

        ckpt, latest_step, all_ckpts = find_latest_checkpoint(run_dir)
        if ckpt is None or latest_step is None:
            report.add(section, "FAIL", f"{label} checkpoint", f"no steps_*_pytorch_model.pt under {run_dir}")
        elif latest_step >= expected_step:
            report.add(
                section,
                "PASS",
                f"{label} checkpoint",
                f"latest step {latest_step} >= expected {expected_step}",
                checkpoint=ckpt,
                checkpoint_steps=[step for step, _ in all_ckpts],
            )
        else:
            report.add(
                section,
                "FAIL",
                f"{label} checkpoint",
                f"latest step {latest_step} < expected {expected_step}",
                checkpoint=ckpt,
                checkpoint_steps=[step for step, _ in all_ckpts],
            )

        if (run_dir / "tensorboard").exists() and list((run_dir / "tensorboard").glob("**/events.out.tfevents*")):
            report.add(section, "PASS", f"{label} tensorboard", "TensorBoard events found")
        else:
            report.add(section, "WARN", f"{label} tensorboard", "no TensorBoard event file found")


def check_dataset(
    report: Report,
    dataset_root: Path,
    state_dim: int,
    action_dim: int,
    horizon: int,
    gamma: float,
    sample_limit: int,
    require_preprocessed: bool,
) -> None:
    section = "dataset"
    try:
        import pandas as pd  # noqa: F401
    except ImportError as exc:
        report.add(section, "FAIL", "pandas import", f"pandas is required to inspect parquet files: {exc}")
        return

    info_path = dataset_root / "meta/info.json"
    modality_path = dataset_root / "meta/modality.json"
    data_dir = dataset_root / "data"

    if not info_path.is_file() or not modality_path.is_file() or not data_dir.is_dir():
        report.add(
            section,
            "FAIL",
            "LeRobot root",
            f"expected meta/info.json, meta/modality.json, and data/ under {dataset_root}",
        )
        return
    report.add(section, "PASS", "LeRobot root", str(dataset_root))

    try:
        info = read_json(info_path)
        modality = read_json(modality_path)
    except Exception as exc:  # noqa: BLE001
        report.add(section, "FAIL", "metadata parse", str(exc))
        return

    state_keys = modality_original_keys(modality, "state") or ["observation.state", "state"]
    action_keys = modality_original_keys(modality, "action") or ["action", "actions"]
    video_keys = modality_original_keys(modality, "video")
    language_keys = modality_original_keys(modality, "annotation") or modality_original_keys(modality, "language")

    report.add(section, "INFO", "modality state keys", ", ".join(state_keys) or "<none>")
    report.add(section, "INFO", "modality action keys", ", ".join(action_keys) or "<none>")
    if len(video_keys) >= 2:
        report.add(section, "PASS", "two-view modality", f"video keys={video_keys[:2]}")
    else:
        report.add(section, "FAIL", "two-view modality", f"expected >=2 video keys, got {video_keys}")
    if language_keys:
        report.add(section, "PASS", "language modality", f"language/annotation keys={language_keys}")
    else:
        report.add(section, "FAIL", "language modality", "no annotation/language modality key found")

    state_shapes = {key: feature_shape(info, key) for key in state_keys if feature_shape(info, key) is not None}
    action_shapes = {key: feature_shape(info, key) for key in action_keys if feature_shape(info, key) is not None}
    if any(shape and int(shape[0]) == state_dim for shape in state_shapes.values()):
        report.add(section, "PASS", "state metadata dim", f"state dim {state_dim}", shapes=state_shapes)
    else:
        report.add(section, "WARN", "state metadata dim", f"could not confirm state dim {state_dim}", shapes=state_shapes)
    if any(shape and int(shape[0]) == action_dim for shape in action_shapes.values()):
        report.add(section, "PASS", "action metadata dim", f"action dim {action_dim}", shapes=action_shapes)
    else:
        report.add(section, "WARN", "action metadata dim", f"could not confirm action dim {action_dim}", shapes=action_shapes)

    parquets = sorted(data_dir.glob("**/*.parquet"))
    if not parquets:
        report.add(section, "FAIL", "episode parquet", f"no parquet files under {data_dir}")
        return
    sampled = parquets[: max(1, sample_limit)]
    report.add(section, "PASS", "episode parquet", f"found {len(parquets)} parquet files; sampled {len(sampled)}")

    column_counts: Counter[str] = Counter()
    missing_state = 0
    missing_action = 0
    state_dim_bad: list[str] = []
    action_dim_bad: list[str] = []
    done_bad: list[str] = []
    reward_bad: list[str] = []
    total_frames = 0
    valid_transitions = 0
    successes = 0
    success_files = 0

    for parquet in sampled:
        try:
            df = pd.read_parquet(parquet)
        except Exception as exc:  # noqa: BLE001
            report.add(section, "FAIL", "read parquet", f"{parquet}: {exc}")
            continue

        total_frames += len(df)
        valid_transitions += max(0, len(df) - 2 * horizon + 1)
        column_counts.update(df.columns)

        present_state = [key for key in state_keys if key in df.columns]
        present_action = [key for key in action_keys if key in df.columns]
        if not present_state:
            missing_state += 1
        else:
            dim = sample_dimension(df, present_state[0])
            if dim != state_dim:
                state_dim_bad.append(f"{parquet}:{present_state[0]} dim={dim}")
        if not present_action:
            missing_action += 1
        else:
            dim = sample_dimension(df, present_action[0])
            if dim != action_dim:
                action_dim_bad.append(f"{parquet}:{present_action[0]} dim={dim}")

        if "success" in df.columns:
            success_files += 1
            if len(df):
                successes += int(scalar_bool(df["success"].iloc[-1]))

        if {"step_reward", "reward", "done"}.issubset(df.columns):
            done = df["done"].map(scalar_bool).to_numpy(dtype=bool)
            expected = np.zeros(len(df), dtype=bool)
            expected[max(0, len(df) - horizon) :] = True
            if not np.array_equal(done, expected):
                done_bad.append(str(parquet))

            try:
                rewards = np.asarray([numeric_scalar(x) for x in df["reward"].to_list()], dtype=float)
                step_rewards = np.asarray([numeric_scalar(x) for x in df["step_reward"].to_list()], dtype=float)
                if not np.isfinite(rewards).all() or not np.isfinite(step_rewards).all():
                    reward_bad.append(f"{parquet}: non-finite reward")
                if len(step_rewards):
                    max_offset = min(horizon - 1, len(step_rewards) - 1)
                    expected_first = sum((gamma**k) * step_rewards[k] for k in range(max_offset + 1))
                    if not math.isclose(float(rewards[0]), float(expected_first), rel_tol=1e-4, abs_tol=1e-4):
                        reward_bad.append(
                            f"{parquet}: reward[0]={rewards[0]:.6g} != recomputed chunk {expected_first:.6g}"
                        )
            except Exception as exc:  # noqa: BLE001
                reward_bad.append(f"{parquet}: {exc}")

    if missing_state == 0 and not state_dim_bad:
        report.add(section, "PASS", "state columns", f"all sampled parquets expose state dim {state_dim}")
    else:
        report.add(
            section,
            "FAIL",
            "state columns",
            f"missing_state_files={missing_state}, dim_mismatch={state_dim_bad[:5]}",
        )
    if missing_action == 0 and not action_dim_bad:
        report.add(section, "PASS", "action columns", f"all sampled parquets expose action dim {action_dim}")
    else:
        report.add(
            section,
            "FAIL",
            "action columns",
            f"missing_action_files={missing_action}, dim_mismatch={action_dim_bad[:5]}",
        )

    if success_files == len(sampled):
        report.add(
            section,
            "PASS",
            "success labels",
            f"success column exists in all sampled parquets; sampled_successes={successes}/{success_files}",
        )
    else:
        report.add(
            section,
            "FAIL",
            "success labels",
            f"success column missing in {len(sampled) - success_files}/{len(sampled)} sampled parquets",
        )

    preprocessed_files = min(column_counts.get("step_reward", 0), column_counts.get("reward", 0), column_counts.get("done", 0))
    if preprocessed_files == len(sampled) and not done_bad and not reward_bad:
        report.add(
            section,
            "PASS",
            "AWAC reward preprocessing",
            f"step_reward/reward/done valid for sampled parquets; H={horizon}, gamma={gamma}",
        )
    else:
        status = "FAIL" if require_preprocessed else "WARN"
        report.add(
            section,
            status,
            "AWAC reward preprocessing",
            (
                "missing or invalid step_reward/reward/done; "
                f"preprocessed_sample_count={preprocessed_files}/{len(sampled)}, "
                f"done_bad={done_bad[:3]}, reward_bad={reward_bad[:3]}"
            ),
        )

    if valid_transitions > 0:
        report.add(
            section,
            "PASS",
            "valid AWAC transitions",
            f"sampled_valid_transitions={valid_transitions}, sampled_frames={total_frames}, H={horizon}",
        )
    else:
        report.add(section, "FAIL", "valid AWAC transitions", f"no sampled t+2H transitions, H={horizon}")


def check_rollout_dataset_columns(
    report: Report,
    rollout_dataset: Path,
    state_dim: int,
    action_dim: int,
    horizon: int,
    gamma: float,
    sample_limit: int,
    hard_fail: bool,
) -> None:
    section = "rollout"
    status_bad = "FAIL" if hard_fail else "WARN"
    try:
        import pandas as pd  # noqa: F401
    except ImportError as exc:
        report.add(section, status_bad, "rollout parquet import", f"pandas is required: {exc}")
        return

    info_path = rollout_dataset / "meta/info.json"
    modality_path = rollout_dataset / "meta/modality.json"
    if not info_path.is_file() or not modality_path.is_file():
        report.add(
            section,
            status_bad,
            f"{rollout_dataset} metadata",
            "missing meta/info.json or meta/modality.json",
        )
        return

    parquets = sorted((rollout_dataset / "data").glob("**/*.parquet"))
    if not parquets:
        report.add(section, status_bad, f"{rollout_dataset} parquet", "no parquet files under data/")
        return

    sampled = parquets[: max(1, sample_limit)]
    missing_required: list[str] = []
    state_dim_bad: list[str] = []
    action_dim_bad: list[str] = []
    reward_bad: list[str] = []
    success_terminal = 0
    valid_transitions = 0
    required_columns = {"state", "actions", "success", "done", "episode_success"}

    for parquet in sampled:
        try:
            df = pd.read_parquet(parquet)
        except Exception as exc:  # noqa: BLE001
            report.add(section, status_bad, "read rollout parquet", f"{parquet}: {exc}")
            continue

        missing = sorted(required_columns.difference(df.columns))
        if missing:
            missing_required.append(f"{parquet}: missing {missing}")
            continue

        valid_transitions += max(0, len(df) - 2 * horizon + 1)
        state_actual_dim = sample_dimension(df, "state")
        action_actual_dim = sample_dimension(df, "actions")
        if state_actual_dim != state_dim:
            state_dim_bad.append(f"{parquet}: state dim={state_actual_dim}")
        if action_actual_dim != action_dim:
            action_dim_bad.append(f"{parquet}: actions dim={action_actual_dim}")
        if len(df) and scalar_bool(df["success"].iloc[-1]):
            success_terminal += 1

        if {"step_reward", "reward"}.issubset(df.columns):
            try:
                rewards = np.asarray([numeric_scalar(x) for x in df["reward"].to_list()], dtype=float)
                step_rewards = np.asarray([numeric_scalar(x) for x in df["step_reward"].to_list()], dtype=float)
                done = df["done"].map(scalar_bool).to_numpy(dtype=bool)
                expected_done = np.zeros(len(df), dtype=bool)
                expected_done[max(0, len(df) - horizon) :] = True
                if not np.array_equal(done, expected_done):
                    reward_bad.append(f"{parquet}: done is not last-H AWAC done")
                if len(step_rewards):
                    max_offset = min(horizon - 1, len(step_rewards) - 1)
                    expected_first = sum((gamma**k) * step_rewards[k] for k in range(max_offset + 1))
                    if not math.isclose(float(rewards[0]), float(expected_first), rel_tol=1e-4, abs_tol=1e-4):
                        reward_bad.append(
                            f"{parquet}: reward[0]={rewards[0]:.6g} != recomputed chunk {expected_first:.6g}"
                        )
            except Exception as exc:  # noqa: BLE001
                reward_bad.append(f"{parquet}: {exc}")

    if missing_required or state_dim_bad or action_dim_bad or reward_bad:
        report.add(
            section,
            status_bad,
            f"{rollout_dataset} parquet schema",
            (
                f"missing_required={missing_required[:3]}, "
                f"state_dim_bad={state_dim_bad[:3]}, "
                f"action_dim_bad={action_dim_bad[:3]}, "
                f"reward_bad={reward_bad[:3]}"
            ),
        )
    else:
        report.add(
            section,
            "PASS",
            f"{rollout_dataset} parquet schema",
            (
                f"sampled={len(sampled)}, terminal_successes={success_terminal}/{len(sampled)}, "
                f"valid_transitions={valid_transitions}, H={horizon}"
            ),
        )


def check_rollout(
    report: Report,
    rollout_root: Path | None,
    require_rollout: bool,
    state_dim: int,
    action_dim: int,
    horizon: int,
    gamma: float,
    sample_limit: int,
) -> None:
    section = "rollout"
    if rollout_root is None:
        status = "FAIL" if require_rollout else "WARN"
        report.add(section, status, "rollout root provided", "no --rollout-root provided")
        return
    if not rollout_root.is_dir():
        status = "FAIL" if require_rollout else "WARN"
        report.add(section, status, "rollout root exists", f"missing rollout root: {rollout_root}")
        return
    report.add(section, "PASS", "rollout root exists", str(rollout_root))

    results = sorted(rollout_root.glob("**/results.json"))
    if results:
        report.add(section, "PASS", "results.json", f"found {len(results)} results.json files")
    else:
        report.add(section, "FAIL" if require_rollout else "WARN", "results.json", "no results.json found")

    rollout_infos = sorted(rollout_root.glob("**/rollout_lerobot/meta/info.json"))
    rollout_parquets = sorted(rollout_root.glob("**/rollout_lerobot/data/**/*.parquet"))
    rollout_steps = sorted(rollout_root.glob("**/rollout_analysis/rollout_steps.jsonl"))
    videos = sorted(rollout_root.glob("**/*.mp4"))

    if rollout_infos and rollout_parquets:
        report.add(
            section,
            "PASS",
            "rollout LeRobot",
            f"info_files={len(rollout_infos)}, parquets={len(rollout_parquets)}",
        )
    else:
        report.add(
            section,
            "FAIL" if require_rollout else "WARN",
            "rollout LeRobot",
            f"info_files={len(rollout_infos)}, parquets={len(rollout_parquets)}",
        )
    if rollout_steps:
        report.add(section, "PASS", "rollout step trace", f"found {len(rollout_steps)} rollout_steps.jsonl files")
    else:
        report.add(section, "WARN", "rollout step trace", "no rollout_analysis/rollout_steps.jsonl found")
    if videos:
        report.add(section, "PASS", "rollout mp4", f"found {len(videos)} mp4 files")
    else:
        report.add(section, "WARN", "rollout mp4", "no mp4 files found")

    rollout_datasets = sorted({path.parents[1] for path in rollout_infos})
    for rollout_dataset in rollout_datasets[: max(1, sample_limit)]:
        check_rollout_dataset_columns(
            report,
            rollout_dataset=rollout_dataset,
            state_dim=state_dim,
            action_dim=action_dim,
            horizon=horizon,
            gamma=gamma,
            sample_limit=min(sample_limit, 16),
            hard_fail=require_rollout,
        )


def render_markdown(args: argparse.Namespace, report: Report) -> str:
    lines = [
        "# AWAC Readiness Report",
        "",
        "## Inputs",
        "",
        f"- batch_root: `{args.batch_root}`",
        f"- dataset_root: `{Path(args.data_root) / args.dataset_name}`",
        f"- data_mix: `{args.data_mix}`",
        f"- expected_step: `{args.expected_step}`",
        f"- H/gamma: `{args.action_horizon}` / `{args.gamma}`",
        f"- require_preprocessed: `{args.require_preprocessed}`",
        f"- require_rollout: `{args.require_rollout}`",
        "",
        "## Summary",
        "",
        f"- status_counts: `{report.status_counts()}`",
        f"- overall: `{'FAIL' if report.has_failures else 'PASS'}`",
        "",
        "## Checks",
        "",
    ]

    current_section = None
    for check in report.checks:
        if check["section"] != current_section:
            current_section = check["section"]
            lines.extend([f"### {current_section}", ""])
        lines.append(f"- **{check['status']}** `{check['name']}`: {check['detail']}")
    lines.append("")
    if report.has_failures:
        lines.extend(
            [
                "## Blocking Interpretation",
                "",
                "- 有 `FAIL` 就不要直接开 AWAC critic/actor；先按失败项补齐。",
                "- `success` 缺失时，先补标签再跑 `prepare_awac_rewards.py`，不要用默认全成功绕过。",
                "- `step_reward/reward/done` 缺失时，说明还没完成 AWAC 预处理，critic 训练会读到错误/零奖励。",
                "- rollout parquet 若缺 `success`，不能直接用默认 `prepare_awac_rewards.py`；需要补标准成功列或显式传 `--success_column`。",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    report = Report()

    batch_root = Path(args.batch_root)
    dataset_root = Path(args.data_root) / args.dataset_name
    jobs = args.job if args.job else list(DEFAULT_JOBS)
    rollout_root = Path(args.rollout_root) if args.rollout_root else None

    check_registry(report, args.data_mix, args.dataset_name)
    check_bc_runs(
        report,
        batch_root,
        jobs,
        args.expected_step,
        args.state_dim,
        args.action_dim,
        args.action_horizon,
    )
    check_dataset(
        report,
        dataset_root,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        horizon=args.action_horizon,
        gamma=args.gamma,
        sample_limit=args.sample_parquets,
        require_preprocessed=args.require_preprocessed,
    )
    check_rollout(
        report,
        rollout_root,
        args.require_rollout,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        horizon=args.action_horizon,
        gamma=args.gamma,
        sample_limit=args.sample_parquets,
    )

    markdown = render_markdown(args, report)
    print(markdown)

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "inputs": {
                "batch_root": args.batch_root,
                "dataset_root": str(dataset_root),
                "data_mix": args.data_mix,
                "expected_step": args.expected_step,
                "action_horizon": args.action_horizon,
                "gamma": args.gamma,
                "state_dim": args.state_dim,
                "action_dim": args.action_dim,
                "jobs": jobs,
                "rollout_root": str(rollout_root) if rollout_root else None,
                "require_preprocessed": args.require_preprocessed,
                "require_rollout": args.require_rollout,
            },
            "status_counts": report.status_counts(),
            "overall": "FAIL" if report.has_failures else "PASS",
            "checks": report.checks,
        }
        json_path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"WROTE {json_path}")

    return 1 if report.has_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
