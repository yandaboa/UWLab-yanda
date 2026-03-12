# Copyright (c) 2024-2025, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test corrective action labels in-env for two steps."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Test corrective action labels with two env steps.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent config entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--rand_action_scale", type=float, default=5.0, help="Scale for first-step random arm action.")
parser.add_argument("--input_clip", type=float, nargs=2, default=None, help="Optional scaled input clip [min max].")
parser.add_argument("--action_clip", type=float, nargs=2, default=(-25.0, 25.0), help="Corrective arm action clip [min max].")
parser.add_argument("--rot_clip", type=float, default=None, help="Optional axis-angle clip before scaling back.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from typing import Any, cast

import gymnasium as gym
import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedEnv, ManagerBasedRLEnvCfg, multi_agent_to_single_agent

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from metalearning.corrective_labels import CorrectiveLabeler, CorrectiveLabelerConfig, validate_quat_convention
from uwlab_tasks.manager_based.manipulation.reset_states.mdp import utils as reset_utils
from uwlab_tasks.utils.hydra import hydra_task_config


def _gather_by_timestep(seq: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Gather [N, T, D] by per-env indices t -> [N, D]."""
    if seq.ndim != 3:
        raise ValueError(f"Expected [N,T,D], got {tuple(seq.shape)}.")
    if t.ndim != 1 or t.shape[0] != seq.shape[0]:
        raise ValueError(f"Expected timestep shape [N], got {tuple(t.shape)}.")
    env_ids = torch.arange(seq.shape[0], device=seq.device)
    return seq[env_ids, t, :]


def _axis_angle_to_quat(axis_angle: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    angle = torch.linalg.norm(axis_angle, dim=-1).clamp_min(eps)
    axis = axis_angle / angle.unsqueeze(-1)
    quat = math_utils.quat_from_angle_axis(angle, axis)
    return math_utils.normalize(quat)


def _quat_angle_error(q_cur: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
    q_rel = math_utils.quat_mul(math_utils.quat_inv(q_cur), q_target)
    q_rel = math_utils.normalize(q_rel)
    w = q_rel[:, 0].abs().clamp(max=1.0)
    return 2.0 * torch.acos(w)


def _extract_debug_end_effector_pose(obs: Any) -> torch.Tensor:
    """Extract [N,6] end_effector_pose from debug observation group."""
    assert isinstance(obs, dict), "Expected dict observations with debug group."
    debug_obs = obs.get("debug")
    assert isinstance(debug_obs, dict), "Expected observations['debug'] to be a dict."
    ee_pose = debug_obs.get("end_effector_pose")
    assert isinstance(ee_pose, torch.Tensor), "Expected observations['debug']['end_effector_pose'] tensor."
    assert ee_pose.ndim == 2 and ee_pose.shape[-1] >= 6, f"Expected end_effector_pose shape [N,>=6], got {tuple(ee_pose.shape)}."
    return ee_pose[:, :6]


def _step_env_obs(env: Any, action: torch.Tensor) -> Any:
    """Step env and return only observation for gymnasium vec/single env signatures."""
    out = env.step(action)
    if not isinstance(out, tuple) or len(out) < 1:
        raise RuntimeError("Unexpected env.step return format.")
    return out[0]


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, _agent_cfg: Any):
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(cast(DirectMARLEnv, env))

    manager_env = cast(ManagerBasedEnv, env.unwrapped)
    context = getattr(manager_env, "context", None)
    if context is None:
        raise RuntimeError("Expected env.context to be loaded from config context.")
    if context.demo_obs_dict is None:
        raise RuntimeError("Expected context.demo_obs_dict to be available.")
    if "end_effector_pose" not in context.demo_obs_dict:
        raise RuntimeError("Expected 'end_effector_pose' in context.demo_obs_dict.")

    # Force reset so context and sampled episodes are refreshed.
    env.reset()

    demo_lengths = context.demo_obs_lengths.to(dtype=torch.long).clamp(min=2)
    t = cast(Any, manager_env).episode_length_buf.to(dtype=torch.long)
    t = torch.minimum(t, demo_lengths - 2)
    t1 = t + 1

    s_t = _gather_by_timestep(context.demo_obs_dict["end_effector_pose"], t)
    a_t = _gather_by_timestep(context.demo_actions, t)
    s_t1 = _gather_by_timestep(context.demo_obs_dict["end_effector_pose"], t1)

    robot = manager_env.scene["robot"]
    metadata = reset_utils.read_metadata_from_usd_directory(robot.cfg.spawn.usd_path)
    q_grip_meta = torch.tensor(metadata["gripper_offset"]["quat"], device=manager_env.device, dtype=torch.float32)
    q_aro_meta = torch.tensor(metadata["offset"]["quat"], device=manager_env.device, dtype=torch.float32)
    q_grip_wxyz, _ = validate_quat_convention(q_grip_meta, metadata_order="auto")
    q_aro_wxyz, _ = validate_quat_convention(q_aro_meta, metadata_order="auto")

    labeler = CorrectiveLabeler(
        cfg=CorrectiveLabelerConfig(
            input_clip=tuple(args_cli.input_clip) if args_cli.input_clip is not None else None,
            action_clip=tuple(args_cli.action_clip) if args_cli.action_clip is not None else None,
            rot_clip=args_cli.rot_clip,
            invalid_handling="assert",
        ),
        q_grip_wxyz=q_grip_wxyz,
        q_action_root_offset_wxyz=q_aro_wxyz,
        device=manager_env.device,
    )

    # Step 1: small random action from the current state.
    random_action = torch.zeros((manager_env.num_envs, 7), device=manager_env.device, dtype=torch.float32)
    random_action[:, :6] = args_cli.rand_action_scale * torch.randn(
        (manager_env.num_envs, 6), device=manager_env.device, dtype=torch.float32
    )
    random_action[:, 6:7] = a_t[:, 6:7]
    obs_after_random = _step_env_obs(env, random_action)
    s_g = _extract_debug_end_effector_pose(obs_after_random)

    # Infer instantaneous controller target from (s_t, a_t), then correct from s_g to that target.
    p_target, q_target = labeler.compute_target_pose(s_t, a_t[:, :6])
    a_g_arm = labeler.compute_corrective_action(s_g, p_target, q_target)
    a_g = torch.cat([a_g_arm, a_t[:, 6:7]], dim=-1)

    # Error before correction (from s_g to target).
    p_g = s_g[:, :3]
    q_obs_g = _axis_angle_to_quat(s_g[:, 3:6])
    q_ctrl_g = math_utils.normalize(
        math_utils.quat_mul(q_obs_g, math_utils.quat_inv(q_grip_wxyz.unsqueeze(0).expand(manager_env.num_envs, -1)))
    )
    pre_pos_err = torch.linalg.norm(p_target - p_g, dim=-1)
    pre_rot_err = _quat_angle_error(q_ctrl_g, q_target)

    # Step 2: corrective action.
    obs_after_correction = _step_env_obs(env, a_g)
    s_corr = _extract_debug_end_effector_pose(obs_after_correction)
    p_corr = s_corr[:, :3]
    q_obs_corr = _axis_angle_to_quat(s_corr[:, 3:6])
    q_ctrl_corr = math_utils.normalize(
        math_utils.quat_mul(q_obs_corr, math_utils.quat_inv(q_grip_wxyz.unsqueeze(0).expand(manager_env.num_envs, -1)))
    )
    post_pos_err = torch.linalg.norm(p_target - p_corr, dim=-1)
    post_rot_err = _quat_angle_error(q_ctrl_corr, q_target)

    # Sanity-check errors to dataset s_{t+1} in observation space (pre and post correction).
    st1_pre_pos_err = torch.linalg.norm(s_t1[:, :3] - s_g[:, :3], dim=-1)
    st1_pre_rot_err = torch.linalg.norm(s_t1[:, 3:6] - s_g[:, 3:6], dim=-1)
    st1_pos_err = torch.linalg.norm(s_t1[:, :3] - s_corr[:, :3], dim=-1)
    st1_rot_err = torch.linalg.norm(s_t1[:, 3:6] - s_corr[:, 3:6], dim=-1)
    # Strict sanity check: every env should get closer to s_t+1 in both position and orientation.
    improved_pos_mask = st1_pos_err < st1_pre_pos_err
    improved_rot_mask = st1_rot_err < st1_pre_rot_err
    improved_all_mask = improved_pos_mask & improved_rot_mask
    if not torch.all(improved_all_mask):
        bad_ids = torch.nonzero(~improved_all_mask, as_tuple=True)[0].detach().cpu().tolist()
        raise RuntimeError(
            f"Not all envs improved toward s_t+1. "
            f"failed_env_ids={bad_ids[:32]} "
            f"(showing up to 32 of {len(bad_ids)})."
        )

    print("[INFO] Corrective-action two-step test complete.")
    print(f"[INFO] Num envs: {manager_env.num_envs}")
    print(f"[INFO] Avg pre  pos err to inferred target: {pre_pos_err.mean().item():.6f} m")
    print(f"[INFO] Avg post pos err to inferred target: {post_pos_err.mean().item():.6f} m")
    print(f"[INFO] Avg pre  rot err to inferred target: {pre_rot_err.mean().item():.6f} rad")
    print(f"[INFO] Avg post rot err to inferred target: {post_rot_err.mean().item():.6f} rad")
    print(f"[INFO] Avg pre  err to s_t+1 in obs-pos (s_g): {st1_pre_pos_err.mean().item():.6f} m")
    print(f"[INFO] Avg pre  err to s_t+1 in obs-aa  (s_g): {st1_pre_rot_err.mean().item():.6f} rad")
    print(f"[INFO] Avg post err to s_t+1 in obs-pos: {st1_pos_err.mean().item():.6f} m")
    print(f"[INFO] Avg post err to s_t+1 in obs-aa:  {st1_rot_err.mean().item():.6f} rad")
    print(f"[INFO] Improvement (obs-pos to s_t+1): {(st1_pre_pos_err.mean() - st1_pos_err.mean()).item():.6f} m")
    print(f"[INFO] Improvement (obs-aa to s_t+1):  {(st1_pre_rot_err.mean() - st1_rot_err.mean()).item():.6f} rad")
    print("[INFO] Per-env s_t+1 improvement check passed for all environments.")
    print(f"[INFO] Improvement (pos inferred target): {(pre_pos_err.mean() - post_pos_err.mean()).item():.6f} m")
    print(f"[INFO] Improvement (rot inferred target): {(pre_rot_err.mean() - post_rot_err.mean()).item():.6f} rad")

    env.close()


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
    simulation_app.close()

