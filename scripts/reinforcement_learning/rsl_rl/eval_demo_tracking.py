# Copyright (c) 2024-2025, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--num_rollouts", type=int, default=50, help="Number of rollout pairs to save.")
parser.add_argument(
    "--max_rollouts_before_saving", type=int, default=50, help="Maximum rollout pairs before saving."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
from datetime import datetime
from typing import Any, Mapping, cast

import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from uwlab_rl.rsl_rl.transformer_ppo import PPOWithLongContext
from uwlab_rl.rsl_rl.long_context_ac import LongContextActorCritic

import importlib
runner_mod = importlib.import_module("rsl_rl.runners.on_policy_runner")
runner_mod.LongContextActorCritic = LongContextActorCritic # type: ignore
runner_mod.PPOWithLongContext = PPOWithLongContext # type: ignore

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedEnv,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from uwlab_tasks.utils.hydra import hydra_task_config
from uwlab_tasks.manager_based.manipulation.from_demo.mdp import utils as from_demo_utils
from metalearning.rollout_pair_storage import RolloutPairStorage, demo_to_episode, rollout_to_episode
from metalearning.rollout_storage import RolloutStorage

# PLACEHOLDER: Extension template (do not remove this comment)


def _flatten_debug_obs(debug_obs: Any) -> dict[str, torch.Tensor]:
    if isinstance(debug_obs, Mapping):
        return {f"debug/{key}": value.detach().clone() for key, value in debug_obs.items()}
    if isinstance(debug_obs, torch.Tensor):
        return {"debug": debug_obs.detach().clone()}
    return {}


def _update_demo_snapshot(
    demo_context: Any,
    env_ids: torch.Tensor,
    demo_obs_snapshot: dict[str, torch.Tensor] | None,
    demo_lengths_snapshot: torch.Tensor,
) -> None:
    if env_ids.numel() == 0:
        return
    env_ids_cpu = env_ids.detach().cpu()
    demo_lengths_snapshot[env_ids_cpu] = demo_context.demo_obs_lengths[env_ids].detach().cpu()
    if demo_obs_snapshot is None:
        return
    demo_obs_dict = demo_context.demo_obs_dict
    if demo_obs_dict is None:
        return
    for key, value in demo_obs_dict.items():
        demo_obs_snapshot[key][env_ids_cpu] = value[env_ids].detach().cpu()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    action_discretization_spec = None
    context = getattr(env.unwrapped, "context", None)
    if context is not None:
        action_discretization_spec = getattr(context, "action_discretization_spec", None)
        if action_discretization_spec is not None:
            print("Loaded action discretization spec from demo episodes.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rollout_dir = os.path.join(log_dir, "rollouts", "demo_tracking", timestamp)
    rollout_pair_storage = RolloutPairStorage(args_cli.max_rollouts_before_saving, rollout_dir)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play", timestamp),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    agent_cfg_dict = agent_cfg.to_dict()
    if action_discretization_spec is not None:
        policy_cfg = agent_cfg_dict.get("policy", {})
        policy_cfg["action_discretization_spec"] = action_discretization_spec
        agent_cfg_dict["policy"] = policy_cfg
    if "bc_warmstart_cfg" in agent_cfg_dict.get("algorithm", {}):
        agent_cfg_dict["algorithm"].pop("bc_warmstart_cfg")
    if agent_cfg_dict["algorithm"]["class_name"] == "PPOWithLongContext":
        agent_cfg_dict["algorithm"]["num_learning_iterations"] = agent_cfg.max_iterations
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    # export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    # export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    previous_obs = env.get_observations()
    obs_buf = env.unwrapped.obs_buf
    policy_obs = obs_buf["policy"]
    debug_obs = obs_buf.get("debug") if isinstance(obs_buf, Mapping) else None
    obs_shape = from_demo_utils.extract_obs_shape(policy_obs, debug_obs)
    action_space = getattr(env.unwrapped, "single_action_space", env.action_space)
    action_shape = from_demo_utils.extract_action_shape(action_space)
    max_steps = getattr(env.unwrapped, "max_episode_length", args_cli.video_length)
    rollout_storage = RolloutStorage(
        num_envs=env.unwrapped.num_envs,
        max_steps=max_steps,
        obs_shape=obs_shape,
        action_shape=action_shape,
        device=env.unwrapped.device,
    )
    manager_env = cast(ManagerBasedEnv, env.unwrapped)
    demo_context = getattr(manager_env, "context", None)
    if demo_context is None:
        raise RuntimeError("FromDemoEval requires a demo context on the environment.")
    demo_indices = demo_context.episode_indices.clone()
    demo_obs_snapshot = (
        {key: value.detach().cpu().clone() for key, value in demo_context.demo_obs_dict.items()}
        if demo_context.demo_obs_dict is not None
        else None
    )
    demo_lengths_snapshot = demo_context.demo_obs_lengths.detach().cpu().clone()
    timestep = 0
    rollouts_collected = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(previous_obs)
            # env stepping
            obs, rewards, dones, _ = env.step(actions)
            policy_obs = previous_obs["policy"]
            if isinstance(previous_obs, Mapping):
                debug_obs = previous_obs.get("debug")
            else:
                debug_obs = None
            if debug_obs is None and isinstance(obs_buf, Mapping):
                debug_obs = obs_buf.get("debug")
            debug_flat = _flatten_debug_obs(debug_obs)
            if isinstance(policy_obs, Mapping):
                rollout_obs = {**policy_obs, **debug_flat} if debug_flat else policy_obs
            elif debug_flat:
                rollout_obs = {"policy": policy_obs, **debug_flat}
            else:
                rollout_obs = policy_obs
            rollout_storage.add_step(rollout_obs, actions, rewards, dones)
            # reset recurrent states for episodes that have terminated
            done_mask = dones.to(torch.bool)
            policy_nn.reset(done_mask)
            if torch.any(done_mask):
                done_env_ids = torch.nonzero(done_mask, as_tuple=True)[0]
                rollouts = rollout_storage.get_rollouts(done_env_ids)
                pairs: list[dict[str, Any]] = []
                done_env_list = done_env_ids.detach().cpu().tolist()
                for rollout_idx, env_id in enumerate(done_env_list):
                    rollout_episode = rollout_to_episode(rollouts, rollout_idx, env_id)
                    demo_index = int(demo_indices[env_id].item())
                    demo_episode = demo_to_episode(demo_context, demo_index, env_id)
                    length = int(demo_lengths_snapshot[env_id].item())
                    # Slice per-env context obs sequences for storage.
                    if demo_obs_snapshot is None:
                        raise RuntimeError("Expected demo observation dict for tracking.")
                    context_obs_dict = {
                        key: value[env_id, :length].detach().cpu().clone()
                        for key, value in demo_obs_snapshot.items()
                    }
                    demo_episode["obs"] = context_obs_dict
                    demo_episode["length"] = length
                    pairs.append({"context": demo_episode, "rollout": rollout_episode})
                rollout_pair_storage.add_pairs(pairs)
                rollouts_collected += len(pairs)
                rollout_storage.wipe_envs(done_env_ids)
                _update_demo_snapshot(demo_context, done_env_ids, demo_obs_snapshot, demo_lengths_snapshot)
                demo_indices[done_env_ids] = demo_context.episode_indices[done_env_ids]
            previous_obs = obs
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        if rollouts_collected >= args_cli.num_rollouts:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    rollout_pair_storage.force_save()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
