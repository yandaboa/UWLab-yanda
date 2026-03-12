from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch

from .env import ACTION_DISCRETIZATION_SPEC_FILENAME


class DiscreteActionWrapper(gym.Wrapper):
    """Environment wrapper that discretizes actions before stepping."""

    def __init__(self, env: gym.Env, action_discretization_cfg: Any, spec_save_dir: str | Path):
        super().__init__(env)
        self._action_discretization_cfg = action_discretization_cfg
        self._spec_save_dir = Path(spec_save_dir)
        self._configure_action_discretization()

    def _configure_action_discretization(self) -> None:
        if self._action_discretization_cfg is None:
            raise ValueError("DiscreteActionWrapper requires env_cfg.action_discretization to be set.")

        num_bins = _get_cfg_value(self._action_discretization_cfg, "num_bins")
        if num_bins is None:
            raise ValueError("DiscreteActionWrapper requires action_discretization.num_bins to be set.")
        self._num_bins = int(num_bins)
        if self._num_bins < 2:
            raise ValueError("DiscreteActionWrapper requires num_bins >= 2.")

        action_space = getattr(self.env, "single_action_space", None) or self.action_space
        action_shape = getattr(action_space, "shape", None)
        if action_shape is None:
            raise ValueError("DiscreteActionWrapper requires a continuous action space with shape.")
        action_shape = tuple(int(dim) for dim in action_shape)
        num_envs = getattr(getattr(self.env, "unwrapped", None), "num_envs", None)
        per_env_shape = _strip_env_dim_shape(action_shape, num_envs)

        min_actions = _get_cfg_value(self._action_discretization_cfg, "min_actions")
        max_actions = _get_cfg_value(self._action_discretization_cfg, "max_actions")
        if min_actions is None or max_actions is None:
            min_actions = getattr(action_space, "low", None)
            max_actions = getattr(action_space, "high", None)
        if min_actions is None or max_actions is None:
            raise ValueError("DiscreteActionWrapper requires action min/max bounds to be set.")

        self._action_shape = per_env_shape
        self._action_min = _to_action_tensor(min_actions, per_env_shape, num_envs)
        self._action_max = _to_action_tensor(max_actions, per_env_shape, num_envs)
        if not torch.isfinite(self._action_min).all() or not torch.isfinite(self._action_max).all():
            raise ValueError("DiscreteActionWrapper action bounds must be finite.")
        if torch.any(self._action_max < self._action_min):
            raise ValueError("DiscreteActionWrapper action max must be >= action min.")

    def step(self, actions):  # type: ignore[override]
        discretized = self._discretize_actions(actions)
        return self.env.step(discretized)

    def close(self):
        self._save_action_spec()
        return super().close()

    def _discretize_actions(self, actions: Any) -> torch.Tensor:
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, device=_resolve_env_device(self.env), dtype=torch.float32)
        min_actions = self._action_min.to(actions.device, dtype=actions.dtype)
        max_actions = self._action_max.to(actions.device, dtype=actions.dtype)
        if actions.ndim == len(self._action_shape) + 1 and actions.shape[1:] == self._action_shape:
            min_actions = min_actions.unsqueeze(0)
            max_actions = max_actions.unsqueeze(0)
        step = (max_actions - min_actions) / (self._num_bins - 1)
        safe_step = torch.where(step == 0, torch.ones_like(step), step)
        indices = torch.round((actions - min_actions) / safe_step)
        indices = torch.clamp(indices, 0, self._num_bins - 1)
        discretized = min_actions + indices * step
        if torch.any(step == 0):
            discretized = torch.where(step == 0, min_actions, discretized)
        return discretized

    def _save_action_spec(self) -> None:
        self._spec_save_dir.mkdir(parents=True, exist_ok=True)
        min_actions = self._action_min
        max_actions = self._action_max
        bin_values = torch.linspace(
            0, 1, self._num_bins, device=min_actions.device, dtype=min_actions.dtype
        )
        bin_values = min_actions[..., None] + (max_actions - min_actions)[..., None] * bin_values
        spec = {
            "action_shape": list(self._action_shape),
            "num_bins": int(self._num_bins),
            "min_actions": min_actions.tolist(),
            "max_actions": max_actions.tolist(),
            "bin_values": bin_values.tolist(),
        }
        spec_path = self._spec_save_dir / ACTION_DISCRETIZATION_SPEC_FILENAME
        spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True))
        self._write_human_spec(spec, self._spec_save_dir / "action_discretization_spec.txt")

    def _write_human_spec(self, spec: dict[str, Any], spec_path: Path) -> None:
        lines = [
            "Action Discretization Spec",
            f"action_shape: {spec['action_shape']}",
            f"num_bins: {spec['num_bins']}",
            f"min_actions: {spec['min_actions']}",
            f"max_actions: {spec['max_actions']}",
            f"bin_values: {spec['bin_values']}",
        ]
        spec_path.write_text("\n".join(lines) + "\n")


def _get_cfg_value(cfg: Any, name: str) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(name)
    return getattr(cfg, name, None)


def _to_action_tensor(values: Any, action_shape: tuple[int, ...], num_envs: int | None) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 0:
        return tensor.expand(action_shape).clone()
    if tuple(tensor.shape) != action_shape:
        tensor = _strip_env_dim_tensor(tensor, num_envs)
    if tuple(tensor.shape) != action_shape:
        raise ValueError(
            f"Action bounds shape {tuple(tensor.shape)} does not match action shape {action_shape}."
        )
    return tensor


def _strip_env_dim_shape(shape: tuple[int, ...], num_envs: int | None) -> tuple[int, ...]:
    if num_envs is None or not shape:
        return shape
    if shape[0] != num_envs:
        return shape
    return shape[1:]


def _strip_env_dim_tensor(tensor: torch.Tensor, num_envs: int | None) -> torch.Tensor:
    if num_envs is None or tensor.ndim == 0:
        return tensor
    if tensor.shape[0] != num_envs:
        return tensor
    return tensor[0]


def _resolve_env_device(env: gym.Env) -> torch.device | None:
    device = getattr(env, "device", None)
    if device is not None:
        return torch.device(device)
    unwrapped = getattr(env, "unwrapped", None)
    device = getattr(unwrapped, "device", None)
    return torch.device(device) if device is not None else None
