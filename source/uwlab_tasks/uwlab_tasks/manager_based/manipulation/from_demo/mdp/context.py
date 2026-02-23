from __future__ import annotations

from typing import Any, Callable, Sequence, cast
import inspect

import torch

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv
from isaaclab.managers import ObservationTermCfg
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from uwlab_tasks.manager_based.manipulation.reset_states import mdp as reset_states_mdp

__all__ = [
    "resample_episode",
    "get_demo_rewards",
    "get_demo_obs",
    "get_demo_actions",
    "get_demo_lengths",
    "get_supervision_demo_obs",
    "get_supervision_demo_actions",
    "get_last_demo_rewards",
    "pose_quat_tracking_reward",
    "tracking_end_effector_reward",
    "tracking_end_effector_orientation_reward",
    "tracking_end_effector_position_error",
    "tracking_end_effector_orientation_error",
    "tracking_action_reward",
    "demo_success_reward",
    "demo_dense_success_reward",
]


def _huber_saturating(x: torch.Tensor, delta: float = 1.0, tolerance: float = 0.01) -> torch.Tensor:
    """Huber loss on nonnegative x. Quadratic near 0, linear past delta."""
    x = x / tolerance
    return torch.where(x <= delta, 0.5 * x * x, delta * (x - 0.5 * delta))


"""
Rewards for tracking the demo trajectory
"""
def _quat_angle_from_rel(q_rel_wxyz: torch.Tensor) -> torch.Tensor:
    """q_rel: (...,4) wxyz -> angle in radians (...,)"""
    q_rel = math_utils.normalize(q_rel_wxyz)
    w = torch.clamp(torch.abs(q_rel[..., 0]), 0.0, 1.0)  # abs handles q ~ -q
    return 2.0 * torch.acos(w)

def pose_quat_tracking_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    demo_quat_key: str = "demo_link_quats",   # name you store this under in demo_obs
    root_relative: bool = True,
    k: float = 2.0,
) -> torch.Tensor:
    """
    Compares reference link quats (from demo obs) to current link quats (from sim),
    using exp(-k * sum_j angle(q_ref ⊗ q_cur^{-1})^2).

    Returns: [num_envs]
    """
    # ---- get reference from context ----
    context_term = _get_tracking_context(env)  # your TrackingEpisodeContext
    demo_obs_dict = context_term.demo_obs_dict
    demo_q_flat = demo_obs_dict[demo_quat_key]  # [N, T, 4*B]

    demo_lengths = context_term.demo_obs_lengths  # [N]
    N, T, D = demo_q_flat.shape
    assert N == env.num_envs

    end_idx = (demo_lengths - 1).clamp(min=0)
    t = torch.minimum(env.episode_length_buf.long(), end_idx)  # [N]
    env_ids = torch.arange(N, device=demo_q_flat.device)

    q_ref_flat = demo_q_flat[env_ids, t]  # [N, 4*B]

    # ---- current quats from sim ----
    asset: Articulation = env.scene[asset_cfg.name]
    q_cur = asset.data.body_link_quat_w  # [N, B, 4] wxyz

    if root_relative:
        q_root = asset.data.root_link_quat_w  # [N, 4]
        q_cur = math_utils.quat_mul(math_utils.quat_conjugate(q_root)[:, None, :].expand(N, q_cur.shape[1], 4), q_cur)

    B = q_cur.shape[1]
    q_ref = q_ref_flat.view(N, B, 4)

    # normalize
    q_ref = math_utils.normalize(q_ref)
    q_cur = math_utils.normalize(q_cur)

    # relative rotation: q_ref ⊗ q_cur^{-1}
    q_rel = math_utils.quat_mul(q_ref, math_utils.quat_conjugate(q_cur))  # [N, B, 4]

    # angle per link
    ang = _quat_angle_from_rel(q_rel)  # [N, B]

    # DeepMimic-style
    return torch.exp(-k * torch.sum(ang * ang, dim=-1))  # [N]


def tracking_end_effector_reward(
    env: "ManagerBasedRLEnv",
    k: float = 40.0,  # the -40 in the paper
    tolerance: float = 0.1,
) -> torch.Tensor:
    sq_dist = (tracking_end_effector_position_error(env) / tolerance) ** 2
    return torch.exp(-k * sq_dist)


def tracking_end_effector_orientation_reward(
    env: "ManagerBasedRLEnv",
    k: float = 2.0,
    tolerance: float = 1.0,
) -> torch.Tensor:
    """Orientation tracking reward.

    Uses the geodesic angle between unit quaternions:
        theta = 2 * acos(|q_ref · q_cur|)

    Reward shaping:
        exp(-k * (theta / tolerance)^2)

    Notes:
    - Assumes pose[3:6] is a rotation vector (axis * angle), not Euler angles.
    - Clamps dot product away from 1.0 to avoid unstable acos gradients.
    """
    sq_dist = (tracking_end_effector_orientation_error(env) / tolerance) ** 2
    return torch.exp(-k * sq_dist)


def tracking_end_effector_position_error(
    env: "ManagerBasedRLEnv",
) -> torch.Tensor:
    demo_ee_pose, curr_ee_pose = _get_tracking_ee_pose(env)
    demo_ee_pos = demo_ee_pose[:, :3]
    curr_ee_pos = curr_ee_pose[:, :3]
    # ||p_hat - p||^2  (squared L2 norm), NOT mean-square
    return torch.sum((demo_ee_pos - curr_ee_pos), dim=-1)


def tracking_end_effector_orientation_error(
    env: "ManagerBasedRLEnv",
) -> torch.Tensor:
    demo_ee_pose, curr_ee_pose = _get_tracking_ee_pose(env)
    demo_rotvec = demo_ee_pose[:, 3:6]
    curr_rotvec = curr_ee_pose[:, 3:6]
    q_ref = _rotvec_to_quat(demo_rotvec)
    q_cur = _rotvec_to_quat(curr_rotvec)
    quat_dot = torch.sum(q_ref * q_cur, dim=-1).abs()
    quat_dot = quat_dot.clamp(min=0.0, max=1.0 - 1e-6)
    angle = 2.0 * torch.acos(quat_dot)
    return angle

def tracking_action_reward(env: "ManagerBasedRLEnv"):
    context_term = _get_tracking_context(env)
    demo_actions = context_term.demo_actions
    if isinstance(demo_actions, dict):
        if "demo" in demo_actions:
            demo_actions = demo_actions["demo"]
        else:
            raise ValueError("demo actions not found in dict keys of context_term.demo_actions")
    
    last_action = env.action_manager.prev_action
    demo_lengths = context_term.demo_obs_lengths  # [num_envs]
    end_idx = (demo_lengths - 1).clamp(min=0)
    # this function is called after the timestep is incremented, so to compare a_t with demo_a_t, we have to subtract one    
    t = torch.minimum(env.episode_length_buf.long() - 1, end_idx)
    env_ids = torch.arange(env.num_envs, device=demo_actions.device)

    demo_action = demo_actions[env_ids, t, :]
    action_diff = torch.square(demo_action - last_action).mean(dim=-1)
    return 5 - action_diff


def _get_tracking_ee_pose(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    context_term = _get_tracking_context(env)
    demo_lengths = context_term.demo_obs_lengths  # [N]
    end_idx = (demo_lengths - 1).clamp(min=0)
    t = torch.minimum(env.episode_length_buf.long(), end_idx)  # [N]
    demo_obs_dict = context_term.demo_obs_dict
    demo_ee = demo_obs_dict["end_effector_pose"]  # [N, T, D]
    env_ids = torch.arange(env.num_envs, device=demo_ee.device)
    demo_ee_pose = demo_ee[env_ids, t, :]
    obs_manager = cast(Any, env.unwrapped).observation_manager
    curr_ee_pose = obs_manager._obs_buffer["ee_pose"]
    return demo_ee_pose, curr_ee_pose


def _rotvec_to_quat(rotvec: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
    eps = 1e-8 if rotvec.dtype in (torch.float32, torch.float64) else 1e-4
    axis = rotvec / torch.clamp(angle, min=eps)
    default_axis = rotvec.new_tensor([1.0, 0.0, 0.0]).expand_as(axis)
    axis = torch.where(angle > eps, axis, default_axis)
    q = math_utils.quat_from_angle_axis(angle.squeeze(-1), axis)
    return math_utils.normalize(q)


def demo_success_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Success reward masked by whether the demo succeeded."""
    context_term = _get_tracking_context(env)
    demo_success = getattr(context_term, "demo_success", None)
    assert demo_success is not None, "Demo context is missing demo_success."
    reward = reset_states_mdp.success_reward(env)
    return torch.where(demo_success, reward, torch.zeros_like(reward))


def demo_dense_success_reward(env: "ManagerBasedRLEnv", std: float = 1.0) -> torch.Tensor:
    """Dense success reward masked by whether the demo succeeded."""
    context_term = _get_tracking_context(env)
    demo_success = getattr(context_term, "demo_success", None)
    assert demo_success is not None, "Demo context is missing demo_success."
    reward = reset_states_mdp.dense_success_reward(env, std)
    return torch.where(demo_success, reward, torch.zeros_like(reward))

def _get_tracking_context(env: ManagerBasedEnv) -> Any:
    context = getattr(env, "context", None)
    if context is None:
        raise AttributeError("Env has no demo context. Expected env.context to be initialized.")
    return context


def resample_episode(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None) -> None:
    context_term = _get_tracking_context(env)
    context_term.reset(env_ids)



def _concat_demo_obs_keys(
    demo_obs_dict: dict[str, torch.Tensor], observation_keys: Sequence[str]
) -> torch.Tensor:
    missing = [key for key in observation_keys if key not in demo_obs_dict]
    if missing:
        raise KeyError(f"Requested demo obs keys not found: {missing}")
    sequences = [demo_obs_dict[key].flatten(start_dim=2) for key in observation_keys]
    return torch.cat(sequences, dim=-1)



def get_demo_obs(
    env: ManagerBasedEnv,
    observation_keys: Sequence[str] | None = None,
) -> torch.Tensor:
    context_term = _get_tracking_context(env)
    if observation_keys:
        demo_obs_dict = context_term.demo_obs_dict
        if demo_obs_dict is None and isinstance(context_term.demo_obs, dict):
            demo_obs_dict = context_term.demo_obs
        if demo_obs_dict is None:
            raise ValueError("Demo obs dict is required when requesting observation keys.")
        return _concat_demo_obs_keys(demo_obs_dict, observation_keys)
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


def get_supervision_demo_obs(
    env: ManagerBasedRLEnv,
    observation_keys: Sequence[str] | None = None,
) -> torch.Tensor:
    demo_obs = get_demo_obs(env, observation_keys)
    # we supervise to next obs - a 'goal' obs
    t = (env.episode_length_buf.long() + 1).clamp(max=get_demo_lengths(env).squeeze(-1) - 1)
    env_ids = torch.arange(env.num_envs, device=demo_obs.device)
    return demo_obs[env_ids, t, :]


def get_supervision_demo_actions(env: ManagerBasedRLEnv) -> torch.Tensor:
    demo_actions = get_demo_actions(env)
    # we supervise to next action - a 'goal' action
    t = (env.episode_length_buf.long()).clamp(max=get_demo_lengths(env).squeeze(-1) - 1)
    env_ids = torch.arange(env.num_envs, device=demo_actions.device)
    return demo_actions[env_ids, t, :]


def get_last_demo_rewards(env: ManagerBasedRLEnv) -> torch.Tensor:
    demo_rewards = get_demo_rewards(env)
    t = (env.episode_length_buf.long() + 1).clamp(max=get_demo_lengths(env).squeeze(-1) - 1)
    env_ids = torch.arange(env.num_envs, device=demo_rewards.device)
    return demo_rewards[env_ids, t, :]
