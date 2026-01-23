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

parser.add_argument("--num_demos", type=int, default=100, help="Number of demos to collect.")
parser.add_argument("--max_demos_before_saving", type=int, default=100, help="Maximum number of demos before saving the dataset.")
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
from typing import cast
import torch
from tqdm import tqdm

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

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
from metalearning.data.environment_noise import EnvironmentNoise
from metalearning.episode_storage import EpisodeStorage
from metalearning.logger import WandbEpisodeLogger, WandbNoiseLogger
from metalearning.rollout_storage import RolloutStorage
from metalearning.state_storage import StateStorage

from datetime import datetime

# PLACEHOLDER: Extension template (do not remove this comment)


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
    
    # for logging, video, etc.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "collect_demos", timestamp),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    environment_noise = None
    noise_cfg = None
    if isinstance(agent_cfg, dict):
        noise_cfg = agent_cfg.get("noise")
    else:
        noise_cfg = getattr(agent_cfg, "noise", None)
    if noise_cfg is not None:
        if hasattr(noise_cfg, "to_dict"):
            noise_cfg = noise_cfg.to_dict()
        environment_noise = EnvironmentNoise(noise_cfg, num_envs=env.unwrapped.num_envs, device=env.unwrapped.device)
        env.unwrapped.environment_noise = environment_noise

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    logger_choice = args_cli.logger or getattr(agent_cfg, "logger", None)
    noise_logger = WandbNoiseLogger(
        enable=logger_choice == "wandb",
        agent_cfg=agent_cfg,
        log_dir=log_dir,
        task_name=args_cli.task,
        log_interval=100,
        project_name=args_cli.log_project_name,
    )
    episode_logger = WandbEpisodeLogger(
        enable=logger_choice == "wandb",
        agent_cfg=agent_cfg,
        log_dir=log_dir,
        task_name=args_cli.task,
        log_interval=100,
        project_name=args_cli.log_project_name,
        run=noise_logger.run,
        wandb_module=noise_logger.wandb,
    )

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
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
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    previous_obs = env.get_observations()
    # previous_obs, _ = env.reset() I think env creation implicitly resets it?
    obs_buf = env.unwrapped.obs_buf
    demo_obs = obs_buf["demo"]
    debug_obs = obs_buf.get("debug") if isinstance(obs_buf, dict) else None
    obs_shape = from_demo_utils.extract_obs_shape(demo_obs, debug_obs)
    max_steps = getattr(env.unwrapped, "max_episode_length", args_cli.video_length)
    action_space = getattr(env.unwrapped, "single_action_space", env.action_space)
    action_shape = from_demo_utils.extract_action_shape(action_space)
    rollout_storage = RolloutStorage(
        num_envs=env.unwrapped.num_envs,
        max_steps=max_steps,
        obs_shape=obs_shape,
        action_shape=action_shape,
        device=env.unwrapped.device,
    )
    episode_storage = EpisodeStorage(
        max_num_episodes=args_cli.max_demos_before_saving,
        save_dir=os.path.join(log_dir, "episodes", timestamp),
    )
    progress_bar = tqdm(total=args_cli.num_demos, desc="Demos collected", unit="demos")
    manager_env = cast(ManagerBasedEnv, env.unwrapped)
    state_storage = StateStorage(manager_env)
    state_storage.capture(torch.arange(manager_env.num_envs, device=manager_env.device))
    success_term_name = from_demo_utils.find_success_term_name(manager_env)
    last_episode_count = 0
    timestep = 0
    global_step = 0
    all_demos_collected = False
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(previous_obs)
            if environment_noise is not None:
                actions = environment_noise.step_action(actions)
            # env stepping
            obs, rewards, dones, extras = env.step(actions)
            demo_obs = previous_obs["demo"]
            debug_obs = previous_obs.get("debug") if isinstance(previous_obs, dict) else None
            if debug_obs is None:
                rollout_obs = demo_obs
            elif isinstance(demo_obs, dict):
                rollout_obs = {**demo_obs, "debug": debug_obs}
            else:
                rollout_obs = {"demo": demo_obs, "debug": debug_obs}
            rollout_storage.add_step(rollout_obs, actions, rewards, dones)
            # reset recurrent states for episodes that have terminated
            done_mask = dones.to(torch.bool)
            policy_nn.reset(done_mask)
            if torch.any(done_mask):
                done_env_ids = torch.nonzero(done_mask, as_tuple=True)[0]
                rollouts = rollout_storage.get_rollouts(done_env_ids)
                states, physics = state_storage.fetch(done_env_ids)
                rollouts["states"] = states
                rollouts["physics"] = physics
                episode_storage.add_episode(rollouts, env_ids=done_env_ids)
                episode_returns, episode_success = from_demo_utils.collect_episode_metrics(
                    rollouts, done_env_ids, manager_env, success_term_name
                )
                episode_logger.log(episode_returns, episode_success, global_step)
                new_episode_count = episode_storage.total_episodes
                if new_episode_count > last_episode_count:
                    remaining = args_cli.num_demos - progress_bar.n
                    progress_bar.update(min(new_episode_count - last_episode_count, max(0, remaining)))
                    last_episode_count = new_episode_count
                rollout_storage.wipe_envs(done_env_ids)
                state_storage.capture(done_env_ids)
                if episode_storage.saved_episodes >= args_cli.num_demos:
                    all_demos_collected = True
                elif episode_storage.total_episodes >= args_cli.num_demos:
                    episode_storage.force_save()
                    all_demos_collected = True
            previous_obs = obs
        if environment_noise is not None:
            noise_logger.log(environment_noise, global_step)
        global_step += 1

        if all_demos_collected:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    progress_bar.close()
    env.close()
    episode_logger.flush(global_step)
    noise_logger.finish()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
