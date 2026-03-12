from __future__ import annotations

from typing import Any, Mapping, cast

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

__all__ = [
    "apply_physics_for_envs",
    "collect_episode_metrics",
    "collect_physics_for_envs",
    "extract_action_shape",
    "extract_obs_shape",
    "find_success_term_name",
    "grab_envs_from_state_dict",
    "update_state_dict",
]


def grab_envs_from_state_dict(state_dict: dict[str, Any], env_ids: torch.Tensor) -> dict[str, Any]:
    if isinstance(env_ids, torch.Tensor) and env_ids.device != torch.device("cpu"):
        env_ids = env_ids.to(torch.device("cpu"))
    new_state_dict: dict[str, Any] = {}
    for key, value in state_dict.items():
        if isinstance(value, dict):
            new_state_dict[key] = grab_envs_from_state_dict(value, env_ids)
        elif isinstance(value, torch.Tensor):
            new_state_dict[key] = value[env_ids].clone()
            if new_state_dict[key].ndim == 1:
                new_state_dict[key] = new_state_dict[key].unsqueeze(0)
    return new_state_dict


def update_state_dict(state_dict: dict[str, Any], new_state_dict: dict[str, Any], env_ids: torch.Tensor) -> None:
    if isinstance(env_ids, torch.Tensor) and env_ids.device != torch.device("cpu"):
        env_ids = env_ids.to(torch.device("cpu"))
    for key, value in state_dict.items():
        if isinstance(value, dict):
            update_state_dict(state_dict[key], new_state_dict[key], env_ids)
        elif isinstance(value, torch.Tensor):
            state_dict[key][env_ids] = new_state_dict[key][env_ids].clone()


def _get_physics_randomized_assets(env: ManagerBasedEnv) -> list[tuple[str, RigidObject | Articulation, str]]:
    assets: list[tuple[str, RigidObject | Articulation, str]] = []
    seen: set[str] = set()
    term_names = env.event_manager.active_terms.get("reset", [])
    term_names.extend(env.event_manager.active_terms.get("startup", []))
    for term_name in term_names:
        term_cfg = getattr(env.event_manager.cfg, term_name)
        asset_cfg: SceneEntityCfg | None = term_cfg.params.get("asset_cfg")
        if asset_cfg is None:
            continue
        name = asset_cfg.name
        if name in seen:
            continue
        asset = env.scene[name]
        kind = "articulation" if isinstance(asset, Articulation) else "rigid_object"
        assets.append((name, asset, kind))
        seen.add(name)
    return assets


def _collect_asset_physics(asset: RigidObject | Articulation) -> dict[str, Any]:
    buf: dict[str, Any] = {}
    buf["mass"] = asset.root_physx_view.get_masses().clone()
    buf["materials"] = asset.root_physx_view.get_material_properties().clone()

    if isinstance(asset, Articulation):
        joint: dict[str, torch.Tensor] = {}
        joint["friction"] = asset.root_physx_view.get_dof_friction_coefficients().to(asset.device).clone()
        # friction_props = asset.root_physx_view.get_dof_friction_properties()
        # # static friction is in slot 0 for Isaac Sim >= 5
        # friction = friction_props[..., 0] if friction_props.ndim == 3 else friction_props
        # joint["friction"] = friction.to(asset.device).clone()
        joint["armature"] = asset.root_physx_view.get_dof_armatures().to(asset.device).clone()
        buf["joint"] = joint
        actuator: dict[str, torch.Tensor] = {}
        actuator["stiffness"] = asset.root_physx_view.get_dof_stiffnesses().to(asset.device).clone()
        actuator["damping"] = asset.root_physx_view.get_dof_dampings().to(asset.device).clone()
        buf["actuator"] = actuator
    return buf


def collect_physics_for_envs(env: ManagerBasedEnv, env_ids: torch.Tensor) -> dict[str, dict[str, dict[str, Any]]]:
    physics: dict[str, dict[str, dict[str, Any]]] = {"articulation": {}, "rigid_object": {}}
    assets = _get_physics_randomized_assets(env)
    if not assets:
        return physics
    for name, asset, kind in assets:
        physics[kind][name] = _collect_asset_physics(asset)
    return physics


def find_success_term_name(env: ManagerBasedEnv) -> str | None:
    if not hasattr(env, "termination_manager"):
        return None
    for term_name in env.termination_manager.active_terms:
        if "success" in term_name.lower():
            return term_name
    return None


def collect_episode_metrics(
    rollouts: Mapping[str, Any],
    done_env_ids: torch.Tensor,
    env: ManagerBasedEnv,
    success_term_name: str | None,
) -> tuple[list[float], list[float] | None]:
    rewards = cast(torch.Tensor, rollouts["rewards"])
    episode_returns = rewards.sum(dim=1).detach().cpu().tolist()
    episode_success = None
    if success_term_name is not None and hasattr(env, "termination_manager"):
        success_values = env.termination_manager.get_term(success_term_name)
        episode_success = success_values[done_env_ids].detach().float().cpu().tolist()
    return episode_returns, episode_success


def extract_obs_shape(
    demo_obs: torch.Tensor | Mapping[str, torch.Tensor],
    debug_obs: torch.Tensor | Mapping[str, torch.Tensor] | None,
) -> tuple[int, ...] | dict[str, tuple[int, ...]]:
    debug_shapes: dict[str, tuple[int, ...]] = {}
    if isinstance(debug_obs, Mapping):
        debug_shapes = {f"debug/{key}": value.shape[1:] for key, value in debug_obs.items()}
    elif isinstance(debug_obs, torch.Tensor):
        debug_shapes = {"debug": debug_obs.shape[1:]}
    if isinstance(demo_obs, Mapping):
        obs_shape = {key: value.shape[1:] for key, value in demo_obs.items()}
        obs_shape.update(debug_shapes)
        return obs_shape
    if not debug_shapes:
        return demo_obs.shape[1:]
    return {"demo": demo_obs.shape[1:], **debug_shapes}


def extract_action_shape(action_space: Any, num_envs: int | None = None) -> tuple[int, ...]:
    action_shape = getattr(action_space, "shape", None)
    if action_shape is None:
        discrete_n = getattr(action_space, "n", None)
        if discrete_n is not None:
            return (int(discrete_n),)
        raise ValueError("Action space has no shape or discrete size.")
    action_shape = tuple(int(dim) for dim in action_shape)
    if num_envs is not None and action_shape and action_shape[0] == num_envs:
        return action_shape[1:]
    return action_shape


def _apply_asset_physics(asset: RigidObject | Articulation, env_ids: torch.Tensor, buf: dict[str, Any]) -> None:
    physx_device = "cpu"
    isaaclab_device = asset.device
    env_ids_cpu = env_ids.cpu() if isinstance(env_ids, torch.Tensor) else env_ids
    env_ids_gpu = env_ids.to(torch.device("cuda")) if isinstance(env_ids, torch.Tensor) else env_ids
    # env_ids_index = env_ids_cpu if isinstance(env_ids_cpu, torch.Tensor) else env_ids_cpu
    if "mass" in buf:
        masses = buf["mass"].to(physx_device)
        asset.root_physx_view.set_masses(masses, env_ids_cpu)
    if "materials" in buf:
        materials = buf["materials"].to(physx_device)
        asset.root_physx_view.set_material_properties(materials, env_ids_cpu)
    if isinstance(asset, Articulation):
        joint = buf.get("joint")
        if isinstance(joint, dict):
            if "friction" in joint:
                asset.write_joint_friction_coefficient_to_sim(
                    joint["friction"].to(isaaclab_device)[env_ids_gpu], joint_ids=slice(None), env_ids=env_ids_gpu
                )
            if "armature" in joint:
                asset.write_joint_armature_to_sim(
                    joint["armature"].to(isaaclab_device)[env_ids_gpu], joint_ids=slice(None), env_ids=env_ids_gpu
                )
        actuator = buf.get("actuator")
        if isinstance(actuator, dict):
            if "stiffness" in actuator:
                asset.write_joint_stiffness_to_sim(
                    actuator["stiffness"].to(isaaclab_device)[env_ids_gpu], joint_ids=slice(None), env_ids=env_ids_gpu
                )
            if "damping" in actuator:
                asset.write_joint_damping_to_sim(
                    actuator["damping"].to(isaaclab_device)[env_ids_gpu], joint_ids=slice(None), env_ids=env_ids_gpu
                )


def apply_physics_for_envs(env: ManagerBasedEnv, env_ids: torch.Tensor, physics: dict[str, Any]) -> None:
    for name, buf in physics.get("articulation", {}).items():
        _apply_asset_physics(env.scene[name], env_ids, buf)
    for name, buf in physics.get("rigid_object", {}).items():
        _apply_asset_physics(env.scene[name], env_ids, buf)
