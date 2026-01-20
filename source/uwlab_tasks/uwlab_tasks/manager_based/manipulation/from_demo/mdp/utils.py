from __future__ import annotations

from typing import Any

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

__all__ = [
    "apply_physics_for_envs",
    "collect_physics_for_envs",
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
    term_names = env.event_manager.active_terms.get("new_history_physics", [])
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


def _apply_asset_physics(asset: RigidObject | Articulation, env_ids: torch.Tensor, buf: dict[str, Any]) -> None:
    physx_device = "cpu"
    isaaclab_device = asset.device
    env_ids_cpu = env_ids.cpu() if isinstance(env_ids, torch.Tensor) else env_ids
    env_ids_index = env_ids_cpu.tolist() if isinstance(env_ids_cpu, torch.Tensor) else env_ids_cpu
    if "mass" in buf:
        masses = buf["mass"].to(physx_device)
        asset.root_physx_view.set_masses(masses, env_ids_index)
    if "materials" in buf:
        materials = buf["materials"].to(physx_device)
        asset.root_physx_view.set_material_properties(materials, env_ids_index)
    if isinstance(asset, Articulation):
        joint = buf.get("joint")
        if isinstance(joint, dict):
            if "friction" in joint:
                asset.write_joint_friction_coefficient_to_sim(
                    joint["friction"].to(isaaclab_device), joint_ids=slice(None), env_ids=env_ids_index
                )
            if "armature" in joint:
                asset.write_joint_armature_to_sim(
                    joint["armature"].to(isaaclab_device), joint_ids=slice(None), env_ids=env_ids_index
                )
        actuator = buf.get("actuator")
        if isinstance(actuator, dict):
            if "stiffness" in actuator:
                asset.write_joint_stiffness_to_sim(
                    actuator["stiffness"].to(isaaclab_device), joint_ids=slice(None), env_ids=env_ids_index
                )
            if "damping" in actuator:
                asset.write_joint_damping_to_sim(
                    actuator["damping"].to(isaaclab_device), joint_ids=slice(None), env_ids=env_ids_index
                )


def apply_physics_for_envs(env: ManagerBasedEnv, env_ids: torch.Tensor, physics: dict[str, Any]) -> None:
    for name, buf in physics.get("articulation", {}).items():
        _apply_asset_physics(env.scene[name], env_ids, buf)
    for name, buf in physics.get("rigid_object", {}).items():
        _apply_asset_physics(env.scene[name], env_ids, buf)
