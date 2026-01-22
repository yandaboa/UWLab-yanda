from typing import Any

import torch
from isaaclab.envs import ManagerBasedEnv

from uwlab_tasks.manager_based.manipulation.from_demo.mdp import utils as demo_mdp_utils


class StateStorage:
    """Cache per-environment initial state and physics for demo collection."""

    def __init__(self, env: ManagerBasedEnv) -> None:
        self._env = env
        self._state_buffer: dict[str, Any] = {}
        self._physics_buffer: dict[str, Any] = {}

    def capture(self, env_ids: torch.Tensor) -> None:
        """Capture initial state and physics for the specified envs."""
        if env_ids.numel() == 0:
            return
        current_state = _to_cpu(self._env.scene.get_state(is_relative=True))
        if self._state_buffer:
            demo_mdp_utils.update_state_dict(self._state_buffer, current_state, env_ids)
        else:
            self._state_buffer = current_state
        current_physics = _to_cpu(demo_mdp_utils.collect_physics_for_envs(self._env, env_ids))
        if self._physics_buffer:
            demo_mdp_utils.update_state_dict(self._physics_buffer, current_physics, env_ids)
        else:
            self._physics_buffer = current_physics

    def fetch(self, env_ids: torch.Tensor) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return cached initial state and physics for the specified envs."""
        states = demo_mdp_utils.grab_envs_from_state_dict(self._state_buffer, env_ids)
        physics = demo_mdp_utils.grab_envs_from_state_dict(self._physics_buffer, env_ids)
        return states, physics


def _to_cpu(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        return value.cpu()
    return value
