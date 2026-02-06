from typing import Any

import torch
from isaaclab.envs import ManagerBasedEnv

from uwlab_tasks.manager_based.manipulation.from_demo.mdp import utils as demo_mdp_utils


class StateStorage:
    """Cache per-environment initial state and physics for demo collection."""

    def __init__(self, env: ManagerBasedEnv, use_multi_reset_indices: bool = True) -> None:
        self._env = env
        self._use_multi_reset_indices = use_multi_reset_indices
        self._state_buffer: dict[str, Any] = {}
        self._physics_buffer: dict[str, Any] = {}

    def capture(self, env_ids: torch.Tensor) -> None:
        """Capture initial state and physics for the specified envs."""
        if env_ids.numel() == 0:
            return
        multi_reset_manager = getattr(self._env, "multi_reset_manager", None)
        if self._use_multi_reset_indices and multi_reset_manager is not None:
            task_ids, state_indices = multi_reset_manager.get_cached_state_indices(env_ids)
            env_ids_cpu = env_ids.cpu() if env_ids.device.type != "cpu" else env_ids
            task_ids_cpu = task_ids.cpu()
            state_indices_cpu = state_indices.cpu()
            if not self._state_buffer:
                self._state_buffer = {
                    "multi_reset_task_id": torch.zeros((self._env.num_envs,), dtype=task_ids_cpu.dtype),
                    "multi_reset_state_index": torch.zeros((self._env.num_envs,), dtype=state_indices_cpu.dtype),
                }
            self._state_buffer["multi_reset_task_id"][env_ids_cpu] = task_ids_cpu
            self._state_buffer["multi_reset_state_index"][env_ids_cpu] = state_indices_cpu
        else:
            current_state = _to_cpu(self._env.scene.get_state(is_relative=True))
            if self._state_buffer:
                demo_mdp_utils.update_state_dict(self._state_buffer, current_state, env_ids)
            else:
                def find_tensor(d: dict[str, Any]) -> torch.Tensor | None:
                    for value in d.values():
                        if isinstance(value, torch.Tensor):
                            return value
                        if isinstance(value, dict):
                            return find_tensor(value)
                    return None
                found = find_tensor(current_state)
                assert found is not None and found.shape[0] == self._env.num_envs
                self._state_buffer = current_state
        current_physics = _to_cpu(demo_mdp_utils.collect_physics_for_envs(self._env, env_ids))
        if self._physics_buffer:
            demo_mdp_utils.update_state_dict(self._physics_buffer, current_physics, env_ids)
        else:
            self._physics_buffer = current_physics

    def fetch(self, env_ids: torch.Tensor) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return cached initial state and physics for the specified envs."""
        if self._use_multi_reset_indices and "multi_reset_state_index" in self._state_buffer:
            env_ids_cpu = env_ids.cpu() if env_ids.device.type != "cpu" else env_ids
            states = {
                "multi_reset_task_id": self._state_buffer["multi_reset_task_id"][env_ids_cpu].clone(),
                "multi_reset_state_index": self._state_buffer["multi_reset_state_index"][env_ids_cpu].clone(),
            }
        else:
            states = demo_mdp_utils.grab_envs_from_state_dict(self._state_buffer, env_ids)
        physics = demo_mdp_utils.grab_envs_from_state_dict(self._physics_buffer, env_ids)
        return states, physics


def _to_cpu(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        return value.cpu()
    return value
