"""
Calvin Multi-Step Evaluation Script

Based on RoboFlamingo's evaluation protocol:
https://github.com/RoboFlamingo/RoboFlamingo/blob/main/robot_flamingo/eval/eval_utils.py

Evaluates a policy server on Calvin's long-horizon multi-task benchmark.
Measures success rate on chains of 1-5 consecutive tasks.

Usage:
    python examples/calvin/eval_calvin.py \
        --args.host 0.0.0.0 \
        --args.port 8000 \
        --args.dataset_path /path/to/calvin/task_D_D \
        --args.num_sequences 1000
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import hydra
import imageio.v2 as imageio
import numpy as np
import tyro

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
os.environ.setdefault("CALVIN_ALLOW_OFFLINE_GIT_METADATA", "1")


def _patch_gitpython_for_offline_metadata() -> None:
    """Let offline CALVIN eval skip git commit metadata while preserving real eval failures."""
    if os.environ.get("CALVIN_ALLOW_OFFLINE_GIT_METADATA") != "1":
        return

    try:
        import git
    except ImportError:
        return

    original_repo = git.Repo

    class _OfflineRepo:
        class _Index:
            @staticmethod
            def diff(*_args, **_kwargs):
                return []

        class _Head:
            class _Object:
                hexsha = "git-unavailable-offline"

            object = _Object()

        index = _Index()
        head = _Head()

    def _repo_or_offline(*args, **kwargs):
        try:
            return original_repo(*args, **kwargs)
        except Exception as exc:
            logging.getLogger(__name__).warning("Git metadata unavailable during offline CALVIN eval: %s", exc)
            return _OfflineRepo()

    git.Repo = _repo_or_offline


_patch_gitpython_for_offline_metadata()

# # Add Calvin to path
# CALVIN_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "calvin"
# sys.path.insert(0, str(CALVIN_ROOT))
from calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    get_env_state_for_initial_condition,
    get_log_dir,
    print_and_save,
)
from omegaconf import OmegaConf
from termcolor import colored
from tqdm import tqdm

from deployment.model_server.tools import image_tools
from examples.LIBERO.eval_files.model2libero_interface import ModelClient

# from calvin_env.envs.play_table_env import get_env

# Set OpenGL platform for headless rendering
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["MUJOCO_GL"] = "osmesa"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EP_LEN = 360  # Max steps per task


def _safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")[:120]


def _write_debug_mp4(img_queue, eval_log_dir: str, sequence_i: int, subtask_i: int, subtask: str, status: str) -> str:
    os.makedirs(eval_log_dir, exist_ok=True)
    video_name = f"{sequence_i}-{subtask_i}-{_safe_filename(subtask)}-{status}.mp4"
    video_path = os.path.join(eval_log_dir, video_name)
    with imageio.get_writer(video_path, fps=30, codec="libx264", macro_block_size=None) as writer:
        for frame in img_queue:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
    return video_path


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _is_calvin_asset_root(candidate: Path) -> bool:
    required_any = [
        candidate / "plane" / "plane.urdf",
        candidate / "franka_panda" / "panda_longer_finger.urdf",
        candidate / "calvin_table_D" / "urdf" / "calvin_table_D.urdf",
    ]
    return any(path.is_file() for path in required_any)


def _calvin_asset_root_candidates(data_path: Path | None) -> list[Path]:
    candidates: list[Path] = []

    if os.environ.get("CALVIN_ASSET_ROOT"):
        candidates.append(Path(os.environ["CALVIN_ASSET_ROOT"]))

    if data_path is not None and data_path.is_absolute():
        candidates.append(data_path)

    try:
        import calvin_env
    except ImportError:
        calvin_pkg = None
    else:
        calvin_pkg = Path(calvin_env.__file__).resolve()

    if calvin_pkg is not None:
        candidates.extend(
            [
                calvin_pkg.parents[1] / "data",
                calvin_pkg.parents[2] / "data",
            ]
        )
        if data_path is not None and not data_path.is_absolute():
            candidates.extend(
                [
                    calvin_pkg.parents[1] / data_path,
                    calvin_pkg.parents[2] / data_path,
                ]
            )

    project_root = Path(os.environ.get("PROJECT_ROOT", Path.cwd())).resolve()
    candidates.extend(
        [
            project_root / "calvin" / "calvin_env" / "data",
            project_root / "calvin" / "calvin_env" / "calvin_env" / "data",
        ]
    )
    if data_path is not None and not data_path.is_absolute():
        candidates.append(Path.cwd() / data_path)

    unique_candidates: list[Path] = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.expanduser()
        key = str(resolved)
        if key not in seen:
            unique_candidates.append(resolved)
            seen.add(key)
    return unique_candidates


def _select_resolved(cfg, cfg_path: str):
    value = OmegaConf.select(cfg, cfg_path)
    if value is None:
        return None
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _patch_calvin_scene_data_path(cfg) -> Path:
    """Resolve CALVIN scene assets to an absolute path before PyBullet loads URDFs."""
    data_path = OmegaConf.select(cfg, "env.scene_cfg.data_path") or OmegaConf.select(cfg, "scene.data_path")
    data_path = Path(str(data_path)) if data_path is not None else None
    candidates = _calvin_asset_root_candidates(data_path)
    asset_root = next((candidate for candidate in candidates if _is_calvin_asset_root(candidate)), None)
    if asset_root is None:
        raise FileNotFoundError(
            "Could not resolve CALVIN asset root. Expected one of the candidates to contain "
            "plane/plane.urdf, franka_panda/panda_longer_finger.urdf, or "
            "calvin_table_D/urdf/calvin_table_D.urdf. "
            f"data_path={data_path}; candidates={[str(candidate) for candidate in candidates]}"
        )

    env_scene_cfg = _select_resolved(cfg, "env.scene_cfg")
    if isinstance(env_scene_cfg, dict):
        env_scene_cfg["data_path"] = str(asset_root)
        OmegaConf.update(cfg, "env.scene_cfg", env_scene_cfg, merge=False)

    scene_cfg = _select_resolved(cfg, "scene")
    if isinstance(scene_cfg, dict) and "data_path" in scene_cfg:
        scene_cfg["data_path"] = str(asset_root)
        OmegaConf.update(cfg, "scene", scene_cfg, merge=False)

    if OmegaConf.select(cfg, "data_path") is not None:
        OmegaConf.update(cfg, "data_path", str(asset_root), merge=True)
    logger.info("Resolved CALVIN scene data_path to %s", asset_root)
    return asset_root


def _patch_calvin_env_references(cfg, asset_root: Path) -> None:
    """Resolve Hydra references and CALVIN URDF paths that direct env construction needs."""
    robot_cfg = _select_resolved(cfg, "env.robot_cfg")
    if robot_cfg is None:
        robot_cfg = {}
    if not isinstance(robot_cfg, dict):
        raise TypeError(f"Expected env.robot_cfg to resolve to a dict, got {type(robot_cfg).__name__}: {robot_cfg}")

    scene_cfg = _select_resolved(cfg, "env.scene_cfg")
    if scene_cfg is None:
        scene_cfg = {}
    if not isinstance(scene_cfg, dict):
        raise TypeError(f"Expected env.scene_cfg to resolve to a dict, got {type(scene_cfg).__name__}: {scene_cfg}")

    scene_mappings = {
        "base_position": "scene.robot_base_position",
        "base_orientation": "scene.robot_base_orientation",
        "initial_joint_positions": "scene.robot_initial_joint_positions",
    }
    for target_key, source_path in scene_mappings.items():
        source_value = _select_resolved(cfg, source_path)
        if source_value is not None and target_key not in robot_cfg:
            robot_cfg[target_key] = copy.deepcopy(source_value)

    scene_mappings = {
        "robot_base_position": "scene.robot_base_position",
        "robot_base_orientation": "scene.robot_base_orientation",
        "robot_initial_joint_positions": "scene.robot_initial_joint_positions",
    }
    for target_key, source_path in scene_mappings.items():
        source_value = _select_resolved(cfg, source_path)
        if source_value is not None and target_key not in scene_cfg:
            scene_cfg[target_key] = copy.deepcopy(source_value)

    scene_cfg["data_path"] = str(asset_root)

    robot_filename = robot_cfg.get("filename")
    if not robot_filename:
        raise KeyError(f"env.robot_cfg did not resolve a robot filename. robot_cfg={robot_cfg}")

    robot_path = Path(str(robot_filename))
    if not robot_path.is_absolute():
        robot_path = asset_root / robot_path
    if not robot_path.is_file():
        raise FileNotFoundError(
            "CALVIN robot URDF not found after resolving asset root: "
            f"{robot_path}. asset_root={asset_root}; original_filename={robot_filename}"
        )
    robot_cfg["filename"] = str(robot_path)
    logger.info("Resolved CALVIN robot URDF to %s", robot_path)

    OmegaConf.update(cfg, "env.robot_cfg", robot_cfg, merge=False)
    OmegaConf.update(cfg, "env.scene_cfg", scene_cfg, merge=False)


def _instantiate_calvin_env_direct(cfg, instantiate_kwargs: dict):
    """Instantiate PlayTableSimEnv directly so PyBullet errors are not hidden by Hydra wrapping."""
    target_path = OmegaConf.select(cfg, "env._target_")
    if not target_path:
        raise ValueError("CALVIN env config is missing env._target_")

    env_config = OmegaConf.to_container(cfg.env, resolve=True)
    env_kwargs = {key: env_config[key] for key in env_config.keys() if key not in {"_target_", "_recursive_"}}
    env_kwargs.update(instantiate_kwargs)
    target_cls = hydra.utils.get_class(str(target_path))
    try:
        return target_cls(**env_kwargs)
    except Exception:
        logger.exception(
            "Direct CALVIN env construction failed. target=%s use_egl=%s scene_data_path=%s robot_urdf=%s",
            target_path,
            env_kwargs.get("use_egl"),
            OmegaConf.select(cfg, "env.scene_cfg.data_path"),
            OmegaConf.select(cfg, "env.robot_cfg.filename"),
        )
        raise


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    pretrained_path: str = ""
    unnorm_key: str = "franka"

    #################################################################################################################
    # Calvin environment-specific parameters
    #################################################################################################################
    dataset_path: str = "/home/lwb/Projects/SII/starVLA_Projects/calvin/dataset/calvin_debug_dataset"
    calvin_config_path: str = "/home/lwb/Projects/SII/starVLA_Projects/calvin/calvin_models/conf"
    eval_sequences_path: str = "examples/calvin/eval_files/eval_sequences.json"
    num_sequences: int = 1000  # Number of evaluation sequences
    num_workers: int = 1  # For future multi-process support
    seed: int = 0
    create_plan_tsne: bool = False

    #################################################################################################################
    # Evaluation settings
    #################################################################################################################
    debug: bool = False  # Save debug videos
    eval_log_dir: str = "tmp/calvin/eval_logs"  # Path to save evaluation logs and videos
    reset: bool = False  # If True, reset robot state between tasks (easier)
    diverse_inst: bool = False  # Use diverse instructions (zero-shot generalization)


class CalvinPolicyClient:
    """Wrapper around websocket client with Calvin-specific preprocessing."""

    def __init__(
        self,
        host: str,
        port: int,
        resize_size: int = 224,
        replan_steps: int = 5,
        pretrained_path: str = "",
        unnorm_key: str = "",
    ):
        self.client = ModelClient(
            host=host,
            port=port,
            unnorm_key=(unnorm_key or None),
            policy_setup="calvin",
            action_ensemble=False,
        )
        self.resize_size = resize_size
        self.replan_steps = replan_steps
        self.step_count = 0

    def reset(self):
        """Reset action plan buffer."""
        self.step_count = 0

    def step(self, obs: dict, lang_annotation: str) -> np.ndarray:
        """
        Query policy for action given observation and language instruction.

        Args:
            obs: Calvin observation dict with keys:
                - rgb_obs: dict with 'rgb_static' (200x200x3) and 'rgb_gripper' (84x84x3)
                - robot_obs: (15,) proprioceptive state [ee_pos(3), ee_ori(3), gripper(2), joint_pos(7)]
            lang_annotation: Natural language task description
            get_action: If True, query model for new action chunk

        Returns:
            action: (7,) array [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        # Preprocess images
        rgb_static = obs["rgb_obs"]["rgb_static"]  # (200, 200, 3) uint8
        rgb_gripper = obs["rgb_obs"]["rgb_gripper"]  # (84, 84, 3) uint8

        # Resize and pad images
        image = image_tools.convert_to_uint8(image_tools.resize_with_pad(rgb_static, self.resize_size, self.resize_size))
        wrist_image = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(rgb_gripper, self.resize_size, self.resize_size)
        )

        # Prepare input for policy server (aligned with eval_libero)
        example = {
            "image": [image, wrist_image],
            "lang": lang_annotation,
        }

        # Query model
        model_output = self.client.step(example=example, step=self.step_count)
        raw_action = model_output["raw_action"]
        world_vector = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
        rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
        open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)

        action = np.concatenate([world_vector, rotation_delta, open_gripper], axis=0).astype(np.float32)
        self.step_count += 1
        return action


def _resolve_calvin_env_folder(dataset_path: str) -> Path:
    """Resolve a CALVIN split folder that contains .hydra/merged_config.yaml.

    Official CALVIN eval usually passes a dataset root with ``validation/``.
    The H200 shared ABC->D bundle currently exposes D environment config under
    ``task_D_D/training/`` only, so we also accept that split for smoke eval.
    """
    root = Path(dataset_path)
    candidates = [
        root / "validation",
        root / "training",
        root,
    ]
    for candidate in candidates:
        if (candidate / ".hydra" / "merged_config.yaml").is_file():
            if candidate.name != "validation":
                logger.warning(
                    "Using CALVIN split folder %s for eval. This is suitable for smoke/visual checks, "
                    "but not an official validation split.",
                    candidate,
                )
            return candidate
    raise FileNotFoundError(
        "Could not find CALVIN env config. Expected one of: "
        f"{root / 'validation' / '.hydra' / 'merged_config.yaml'}, "
        f"{root / 'training' / '.hydra' / 'merged_config.yaml'}, "
        f"{root / '.hydra' / 'merged_config.yaml'}"
    )


def make_env(dataset_path: str):
    """Initialize Calvin environment without tactile sensor (to avoid OpenGL issues)."""
    val_folder = _resolve_calvin_env_folder(dataset_path)

    # Load config and disable tactile sensor to avoid pyrender/OpenGL conflicts
    config_path = val_folder / ".hydra" / "merged_config.yaml"
    cfg = OmegaConf.load(config_path)
    force_no_egl = _env_flag("CALVIN_FORCE_NO_EGL", "1")
    asset_root = _patch_calvin_scene_data_path(cfg)
    _patch_calvin_env_references(cfg, asset_root)

    # Remove tactile sensor from camera list if it exists
    if hasattr(cfg.env, "cameras") and "tactile" in cfg.env.cameras:
        # Create a new camera dict without tactile
        new_cameras = OmegaConf.create({k: v for k, v in cfg.env.cameras.items() if k != "tactile"})
        cfg.env.cameras = new_cameras

    if force_no_egl:
        updated_paths = []
        for cfg_path in ("env.use_egl", "env.env_cfg.use_egl", "env.renderer.use_egl", "env.simulator.use_egl"):
            if OmegaConf.select(cfg, cfg_path) is not None:
                OmegaConf.update(cfg, cfg_path, False, merge=True)
                updated_paths.append(cfg_path)
        logger.warning(
            "CALVIN_FORCE_NO_EGL=1: forcing CALVIN env use_egl=False for offline/headless eval. "
            "updated_config_paths=%s",
            updated_paths or ["instantiate_kwarg:use_egl"],
        )

    instantiate_kwargs = {
        "show_gui": False,
        "use_vr": False,
        "use_scene_info": True,
    }
    if force_no_egl:
        instantiate_kwargs["use_egl"] = False

    env = _instantiate_calvin_env_direct(cfg, instantiate_kwargs)

    return env


def load_lang_task(dataset_path: str) -> dict:
    """Load language annotations and task oracle for Calvin validation set."""
    conf_dir = Path(dataset_path)
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    return val_annotations, task_oracle


def evaluate_policy_ddp(
    policy,
    env,
    epoch,
    calvin_conf_path,
    eval_sequences_path,
    num_sequences,
    eval_log_dir=None,
    debug=False,
    create_plan_tsne=False,
    reset=False,
    diverse_inst=False,
):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """
    conf_dir = Path(calvin_conf_path)
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)

    # val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    if diverse_inst:
        with open("/mnt/bn/robotics/lxh/robot-flamingo/lang_annotation_cache.json", "r") as f:
            val_annotations = json.load(f)
    else:
        val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)
    with open(eval_sequences_path, "r") as f:
        eval_sequences = json.load(f)
    if num_sequences is not None and int(num_sequences) > 0:
        eval_sequences = eval_sequences[: int(num_sequences)]
    selected_eval_sequences = list(eval_sequences)
    # device_num = int(torch.distributed.get_world_size())
    # device_id = torch.distributed.get_rank()
    # assert num_sequences % device_num == 0
    # interval_len = int(num_sequences // device_num)
    # eval_sequences = eval_sequences[device_id*interval_len:min((device_id+1)*interval_len, num_sequences)]
    results = []
    plans = defaultdict(list)
    local_sequence_i = 0
    base_sequence_i = 0  # device_id * interval_len

    if not debug:
        eval_sequences = tqdm(selected_eval_sequences, position=0, leave=True)
    else:
        eval_sequences = selected_eval_sequences

    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(
            env,
            policy,
            task_oracle,
            initial_state,
            eval_sequence,
            val_annotations,
            plans,
            debug,
            eval_log_dir,
            base_sequence_i + local_sequence_i,
            reset=reset,
            diverse_inst=diverse_inst,
        )
        results.append(result)
        if not debug:
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
            )
        local_sequence_i += 1

    # if create_plan_tsne:
    #     create_tsne(plans, eval_log_dir, epoch)

    print_and_save(results, selected_eval_sequences, eval_log_dir, epoch)

    return results


def evaluate_sequence(
    env,
    policy,
    task_checker,
    initial_state,
    eval_sequence,
    val_annotations,
    plans,
    debug,
    eval_log_dir="",
    sequence_i=-1,
    reset=False,
    diverse_inst=False,
):
    """
    Evaluates a sequence of language instructions.
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask_i, subtask in enumerate(eval_sequence):
        if reset:
            success = rollout(
                env,
                policy,
                task_checker,
                subtask,
                val_annotations,
                plans,
                debug,
                eval_log_dir,
                subtask_i,
                sequence_i,
                robot_obs=robot_obs,
                scene_obs=scene_obs,
                diverse_inst=diverse_inst,
            )
        else:
            success = rollout(
                env,
                policy,
                task_checker,
                subtask,
                val_annotations,
                plans,
                debug,
                eval_log_dir,
                subtask_i,
                sequence_i,
                diverse_inst=diverse_inst,
            )
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def rollout(
    env,
    policy,
    task_oracle,
    subtask,
    val_annotations,
    plans,
    debug,
    eval_log_dir="",
    subtask_i=-1,
    sequence_i=-1,
    robot_obs=None,
    scene_obs=None,
    diverse_inst=False,
):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    if robot_obs is not None and scene_obs is not None:
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    obs = env.get_obs()
    # get lang annotation for subtask
    if diverse_inst:
        lang_annotation = val_annotations[sequence_i][subtask_i]
    else:
        lang_annotation = val_annotations[subtask][0]
    lang_annotation = lang_annotation.split("\n")[0]
    if "\u2019" in lang_annotation:
        lang_annotation.replace("\u2019", "'")
    policy.reset()
    start_info = env.get_info()

    if debug:
        img_queue = []

    for step in range(EP_LEN):

        action = policy.step(obs, lang_annotation)

        # Ensure action is writable (Calvin env modifies it in-place)
        if not action.flags.writeable:
            action = np.array(action, copy=True)
        action[-1] = 1 if action[-1] > 0 else -1

        obs, _, _, current_info = env.step(action)
        if debug:
            img_copy = copy.deepcopy(obs["rgb_obs"]["rgb_static"])
            img_queue.append(img_copy)
        if step == 0:
            # for tsne plot, only if available
            collect_plan(policy, plans, subtask)

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if debug:
                print(colored("success", "green"), end=" ")
                _write_debug_mp4(img_queue, eval_log_dir, sequence_i, subtask_i, subtask, "succ")
            return True
    if debug:
        print(colored("fail", "red"), end=" ")
        _write_debug_mp4(img_queue, eval_log_dir, sequence_i, subtask_i, subtask, "fail")
    return False


def main(args: Args):
    # args = tyro.cli(Args)

    policy = CalvinPolicyClient(
        args.host,
        args.port,
        args.resize_size,
        args.replan_steps,
        pretrained_path=args.pretrained_path,
        unnorm_key=args.unnorm_key,
    )
    env = make_env(args.dataset_path)

    evaluate_policy_ddp(
        policy,
        env,
        0,
        args.calvin_config_path,
        args.eval_sequences_path,
        args.num_sequences,
        args.eval_log_dir,
        args.debug,
        args.create_plan_tsne,
        args.reset,
        args.diverse_inst,
    )


if __name__ == "__main__":
    tyro.cli(main)
