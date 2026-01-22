from __future__ import annotations

from typing import Any, Callable, cast
import inspect

import torch

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv
from isaaclab.managers import ObservationTermCfg

from uwlab_tasks.manager_based.manipulation.reset_states import mdp as reset_states_mdp

__all__ = [
    "tracking_context_reward",
    "resample_episode",
    "get_demo_rewards",
    "get_demo_obs",
    "get_demo_actions",
    "get_demo_lengths",
    "get_last_demo_obs",
    "get_last_demo_actions",
    "get_last_demo_rewards",
]


def tracking_context_reward(
    env: ManagerBasedRLEnv,
    angle_weight: float = 1.0,
    position_weight: float = 1.0,
    joint_tol: float = 0.1,  # radians (~5.7 deg)
    pos_tol: float = 0.01,  # meters (1 cm)
) -> torch.Tensor:
    context_term = _get_tracking_context(env)
    demo_obs = context_term.demo_obs
    if isinstance(demo_obs, dict):
        if "demo" in demo_obs:
            demo_obs = demo_obs["demo"]
        else:
            raise ValueError("demo obs not found in dict keys of context_term.demo_obs")
    num_envs, _, _ = demo_obs.shape
    assert num_envs == env.num_envs
    assert len(demo_obs.shape) == 3

    demo_lengths = context_term.demo_obs_lengths  # [num_envs]

    # per-env timestep index
    end_idx = (demo_lengths - 1).clamp(min=0)  # [num_envs]
    t = torch.minimum(env.episode_length_buf.long(), end_idx)  # [num_envs]
    env_ids = torch.arange(num_envs, device=demo_obs.device)

    # get the current demo row per environment: [num_envs, D]
    demo_t = demo_obs[env_ids, t]

    # ---- joint angle tracking (radians) ----
    demo_joint_angles = demo_t[:, 1:8]  # [num_envs, 7] (adjust slice if needed)
    cur_joint_angles = reset_states_mdp.joint_pos(env)  # type: ignore[attr-defined]
    cur_joint_angles = cur_joint_angles[:, : demo_joint_angles.shape[1]]

    angle_diff = torch.abs(demo_joint_angles - cur_joint_angles)  # radians
    angle_reward = -angle_weight * (angle_diff / joint_tol).mean(dim=-1)

    # ---- end-effector position tracking (meters) ----
    demo_ee_pos = demo_t[:, 15:18]  # [num_envs, 3]

    ee_pose_term, ee_pose_params = _get_end_effector_pose_term(env)
    curr_ee_pose = ee_pose_term(env, **ee_pose_params)  # likely [num_envs, 6]
    curr_ee_pos = curr_ee_pose[:, :3]  # keep position only

    pos_err = torch.linalg.norm(demo_ee_pos - curr_ee_pos, dim=-1)  # meters
    position_reward = -position_weight * (pos_err / pos_tol)

    return angle_reward + position_reward


def _get_tracking_context(env: ManagerBasedEnv) -> Any:
    context = getattr(env, "context", None)
    if context is None:
        raise AttributeError("Env has no demo context. Expected env.context to be initialized.")
    return context


def _get_end_effector_pose_term(
    env: ManagerBasedEnv,
) -> tuple[Callable[..., torch.Tensor], dict[str, Any]]:
    extensions = getattr(env, "extensions", None)
    cache_key = "demo_end_effector_pose_term"
    if isinstance(extensions, dict) and cache_key in extensions:
        return extensions[cache_key]
    term_cfg = ObservationTermCfg(
        func=cast(Any, reset_states_mdp.target_asset_pose_in_root_asset_frame_with_metadata),
        params={
            "target_asset_cfg": reset_states_mdp.SceneEntityCfg("robot", body_names="robotiq_base_link"),
            "root_asset_cfg": reset_states_mdp.SceneEntityCfg("robot"),
            "target_asset_offset_metadata_key": "gripper_offset",
            "root_asset_offset_metadata_key": "offset",
            "rotation_repr": "axis_angle",
        },
    )
    if inspect.isclass(term_cfg.func):
        term_cfg.func = term_cfg.func(cfg=term_cfg, env=env)
    payload = (cast(Callable[..., torch.Tensor], term_cfg.func), term_cfg.params)
    if isinstance(extensions, dict):
        extensions[cache_key] = payload
    return payload


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


def get_last_demo_obs(env: ManagerBasedEnv) -> torch.Tensor:
    demo_obs = get_demo_obs(env)
    return demo_obs[:, -1, :]


def get_last_demo_actions(env: ManagerBasedEnv) -> torch.Tensor:
    demo_actions = get_demo_actions(env)
    return demo_actions[:, -1, :]


def get_last_demo_rewards(env: ManagerBasedEnv) -> torch.Tensor:
    demo_rewards = get_demo_rewards(env)
    return demo_rewards[:, -1, :]
