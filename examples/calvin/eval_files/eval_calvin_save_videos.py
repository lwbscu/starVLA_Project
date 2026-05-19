#!/usr/bin/env python
"""CALVIN eval entrypoint that can save mp4 files without enabling debug mode.

This file intentionally keeps examples/calvin/eval_files/eval_calvin.py
unchanged. It reuses that module and only overrides the rollout path needed for
the --args.save-videos option.
"""

from __future__ import annotations

import dataclasses
import inspect

from examples.calvin.eval_files import eval_calvin as base


@dataclasses.dataclass
class Args(base.Args):
    save_videos: bool = False


def evaluate_policy_ddp(
    policy,
    env,
    epoch,
    calvin_conf_path,
    eval_sequences_path,
    num_sequences,
    eval_log_dir=None,
    debug=False,
    save_videos=False,
    create_plan_tsne=False,
    reset=False,
    diverse_inst=False,
    rollout_writer=None,
):
    conf_dir = base.Path(calvin_conf_path)
    task_cfg = base.OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = base.hydra.utils.instantiate(task_cfg)

    if diverse_inst:
        with open("/mnt/bn/robotics/lxh/robot-flamingo/lang_annotation_cache.json", "r") as f:
            val_annotations = base.json.load(f)
    else:
        val_annotations = base.OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = base.get_log_dir(eval_log_dir)
    with open(eval_sequences_path, "r") as f:
        eval_sequences = base.json.load(f)
    if num_sequences is not None and int(num_sequences) > 0:
        eval_sequences = eval_sequences[: int(num_sequences)]
    selected_eval_sequences = list(eval_sequences)

    results = []
    plans = base.defaultdict(list)
    local_sequence_i = 0
    base_sequence_i = 0

    if not debug:
        eval_sequences = base.tqdm(selected_eval_sequences, position=0, leave=True)
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
            save_videos,
            eval_log_dir,
            base_sequence_i + local_sequence_i,
            reset=reset,
            diverse_inst=diverse_inst,
            rollout_writer=rollout_writer,
        )
        results.append(result)
        if not debug:
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(base.count_success(results))]) + "|"
            )
        local_sequence_i += 1

    base.print_and_save(results, selected_eval_sequences, eval_log_dir, epoch)
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
    save_videos=False,
    eval_log_dir="",
    sequence_i=-1,
    reset=False,
    diverse_inst=False,
    rollout_writer=None,
):
    robot_obs, scene_obs = base.get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if debug:
        base.time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask_i, subtask in enumerate(eval_sequence):
        kwargs = {}
        if reset:
            kwargs = {"robot_obs": robot_obs, "scene_obs": scene_obs}
        success = rollout(
            env,
            policy,
            task_checker,
            subtask,
            val_annotations,
            plans,
            debug,
            save_videos,
            eval_log_dir,
            subtask_i,
            sequence_i,
            diverse_inst=diverse_inst,
            rollout_writer=rollout_writer,
            **kwargs,
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
    save_videos=False,
    eval_log_dir="",
    subtask_i=-1,
    sequence_i=-1,
    robot_obs=None,
    scene_obs=None,
    diverse_inst=False,
    rollout_writer=None,
):
    if debug:
        print(f"{subtask} ", end="")
        base.time.sleep(0.5)
    if robot_obs is not None and scene_obs is not None:
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    obs = env.get_obs()

    if diverse_inst:
        lang_annotation = val_annotations[sequence_i][subtask_i]
    else:
        lang_annotation = val_annotations[subtask][0]
    lang_annotation = lang_annotation.split("\n")[0]
    if "\u2019" in lang_annotation:
        lang_annotation.replace("\u2019", "'")
    policy.reset()
    start_info = env.get_info()
    rollout_episode = None
    if rollout_writer is not None:
        rollout_episode = rollout_writer.start_episode(
            sequence_index=sequence_i,
            subtask_index=subtask_i,
            subtask=subtask,
            language=lang_annotation,
            initial_state={
                "robot_obs": obs.get("robot_obs", robot_obs),
                "scene_obs": obs.get("scene_obs", scene_obs),
            },
            start_info=start_info,
        )

    if debug or save_videos:
        img_queue = []

    for step in range(base.EP_LEN):
        obs_before_action = obs
        action = policy.step(obs, lang_annotation)

        if not action.flags.writeable:
            action = base.np.array(action, copy=True)
        action[-1] = 1 if action[-1] > 0 else -1

        obs, _, env_done, current_info = env.step(action)
        if debug or save_videos:
            img_queue.append(base.copy.deepcopy(obs["rgb_obs"]["rgb_static"]))
        if step == 0:
            base.collect_plan(policy, plans, subtask)

        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        step_success = len(current_task_info) > 0
        done = bool(env_done) or step_success or step == base.EP_LEN - 1

        if rollout_episode is not None:
            distance_info = base.estimate_rollout_distances(
                obs=obs_before_action,
                current_info=current_info,
                subtask=subtask,
            )
            rollout_writer.add_step(
                rollout_episode,
                obs=obs_before_action,
                action=action,
                env_done=bool(env_done),
                done=done,
                success=step_success,
                current_info=current_info,
                distance_info=distance_info,
            )

        if step_success:
            if rollout_episode is not None:
                rollout_writer.finish_episode(rollout_episode, success=True)
            if debug:
                print(base.colored("success", "green"), end=" ")
            if debug or save_videos:
                base._write_debug_mp4(img_queue, eval_log_dir, sequence_i, subtask_i, subtask, "succ")
            return True

    if rollout_episode is not None:
        rollout_writer.finish_episode(rollout_episode, success=False)
    if debug:
        print(base.colored("fail", "red"), end=" ")
    if debug or save_videos:
        base._write_debug_mp4(img_queue, eval_log_dir, sequence_i, subtask_i, subtask, "fail")
    return False


def main(args: Args):
    client_kwargs = {}
    client_params = inspect.signature(base.CalvinPolicyClient).parameters
    if "pretrained_path" in client_params:
        client_kwargs["pretrained_path"] = args.pretrained_path
    if "unnorm_key" in client_params:
        client_kwargs["unnorm_key"] = args.unnorm_key
    if "send_state_to_policy" in client_params:
        client_kwargs["send_state_to_policy"] = getattr(args, "send_state_to_policy", False)

    policy = base.CalvinPolicyClient(
        args.host,
        args.port,
        args.resize_size,
        args.replan_steps,
        **client_kwargs,
    )
    env = base.make_env(args.dataset_path)
    rollout_writer = None
    rollout_lerobot_dir = getattr(args, "rollout_lerobot_dir", "")
    if rollout_lerobot_dir:
        rollout_writer = base.RolloutLeRobotWriter(
            rollout_lerobot_dir,
            fps=getattr(args, "rollout_lerobot_fps", 10),
            write_videos=getattr(args, "rollout_lerobot_write_videos", False),
            require_target_distance=getattr(args, "rollout_lerobot_require_target_distance", False),
        )

    evaluate_policy_ddp(
        policy,
        env,
        0,
        args.calvin_config_path,
        args.eval_sequences_path,
        args.num_sequences,
        args.eval_log_dir,
        args.debug,
        args.save_videos,
        args.create_plan_tsne,
        args.reset,
        args.diverse_inst,
        rollout_writer=rollout_writer,
    )


if __name__ == "__main__":
    base.tyro.cli(main)
