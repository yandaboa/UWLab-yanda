from __future__ import annotations

from typing import Any, Callable, cast
import inspect

import torch

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv
from isaaclab.managers import ObservationTermCfg

from uwlab_tasks.manager_based.manipulation.reset_states import mdp as reset_states_mdp

__all__ = [
    "resample_episode",
    "get_demo_rewards",
    "get_demo_obs",
    "get_demo_actions",
    "get_demo_lengths",
    "get_last_demo_obs",
    "get_last_demo_actions",
    "get_last_demo_rewards",
    "tracking_joint_angle_reward",
    "tracking_end_effector_reward"
]


def _huber_saturating(x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Huber loss on nonnegative x. Quadratic near 0, linear past delta."""
    return torch.where(x <= delta, 0.5 * x * x, delta * (x - 0.5 * delta))

def tracking_joint_angle_reward(
    env: "ManagerBasedRLEnv",
    joint_tol: float = 0.1,    # radians
    huber_delta: float = 1.0,  # in normalized units
) -> torch.Tensor:
    context_term = _get_tracking_context(env)
    demo_obs = context_term.demo_obs
    demo_obs_dict = context_term.demo_obs_dict
    if isinstance(demo_obs, dict):
        if "demo" in demo_obs:
            demo_obs = demo_obs["demo"]
        else:
            raise ValueError("demo obs not found in dict keys of context_term.demo_obs")

    num_envs, T, _ = demo_obs.shape
    assert num_envs == env.num_envs

    demo_lengths = context_term.demo_obs_lengths  # [num_envs]
    end_idx = (demo_lengths - 1).clamp(min=0)
    t = torch.minimum(env.episode_length_buf.long(), end_idx)
    env_ids = torch.arange(num_envs, device=demo_obs.device)

    demo_joint_angles = demo_obs_dict["joint_pos"][env_ids, t]
    cur_joint_angles = reset_states_mdp.joint_pos(env)  # type: ignore[attr-defined]

    angle_diff = torch.square(demo_joint_angles - cur_joint_angles).mean(dim=-1)

    return -1 * angle_diff


def tracking_end_effector_reward(
    env: "ManagerBasedRLEnv",
    pos_tol: float = 0.1,     # meters
    huber_delta: float = 1.0,  # in normalized units
) -> torch.Tensor:
    context_term = _get_tracking_context(env)
    demo_obs = context_term.demo_obs
    demo_obs_dict = context_term.demo_obs_dict
    if isinstance(demo_obs, dict):
        if "demo" in demo_obs:
            demo_obs = demo_obs["demo"]
        else:
            raise ValueError("demo obs not found in dict keys of context_term.demo_obs")

    num_envs, T, _ = demo_obs.shape
    assert num_envs == env.num_envs

    demo_lengths = context_term.demo_obs_lengths  # [num_envs]
    end_idx = (demo_lengths - 1).clamp(min=0)
    t = torch.minimum(env.episode_length_buf.long(), end_idx)
    env_ids = torch.arange(num_envs, device=demo_obs.device)

    demo_ee_pos = demo_obs_dict["end_effector_pose"][env_ids, t]

    curr_ee_pose = env.unwrapped.observation_manager._obs_buffer['ee_pose'] # [num_envs, 6] # type: ignore[attr-defined]

    pos_err = torch.square(demo_ee_pos - curr_ee_pose).mean(dim=-1)  # meters
    # pos_norm = pos_err / pos_tol

    return -1 * pos_err
    # return _huber_saturating(pos_norm, delta=huber_delta)

def _get_tracking_context(env: ManagerBasedEnv) -> Any:
    context = getattr(env, "context", None)
    if context is None:
        raise AttributeError("Env has no demo context. Expected env.context to be initialized.")
    return context


def resample_episode(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None) -> None:
    context_term = _get_tracking_context(env)
    context_term.reset(env_ids)


def get_demo_obs(env: ManagerBasedEnv) -> torch.Tensor:
    context_term = _get_tracking_context(env)
    if isinstance(context_term.demo_obs, dict):
        if "demo" in context_term.demo_obs:
            return context_term.demo_obs["demo"]
        raise ValueError("demo obs not found in dict keys of context_term.demo_obs")
    return context_term.demo_obs


def get_demo_actions(env: ManagerBasedEnv) -> torch.Tensor:
    context_term = _get_tracking_context(env)
    return context_term.demo_actions


def get_demo_rewards(env: ManagerBasedEnv) -> torch.Tensor:
    context_term = _get_tracking_context(env)
    return context_term.demo_rewards.unsqueeze(-1)


def get_demo_lengths(env: ManagerBasedEnv) -> torch.Tensor:
    context_term = _get_tracking_context(env)
    return context_term.demo_obs_lengths.unsqueeze(-1)


def get_last_demo_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    demo_obs = get_demo_obs(env)
    t = (env.episode_length_buf.long() + 1).clamp(max=get_demo_lengths(env).squeeze(-1) - 1)
    return demo_obs[:, t, :]


def get_last_demo_actions(env: ManagerBasedRLEnv) -> torch.Tensor:
    demo_actions = get_demo_actions(env)
    t = (env.episode_length_buf.long() + 1).clamp(max=get_demo_lengths(env).squeeze(-1) - 1)
    return demo_actions[:, t, :]


def get_last_demo_rewards(env: ManagerBasedRLEnv) -> torch.Tensor:
    demo_rewards = get_demo_rewards(env)
    t = (env.episode_length_buf.long() + 1).clamp(max=get_demo_lengths(env).squeeze(-1) - 1)
    return demo_rewards[:, t, :]
