from typing import Any

import torch
from isaaclab.envs import ManagerBasedEnv

from uwlab_tasks.manager_based.manipulation.from_demo.mdp import utils as demo_mdp_utils


class StateStorage:
    """Cache per-environment initial state, physics, and optional raw state sequences."""

    def __init__(
        self,
        env: ManagerBasedEnv,
        use_multi_reset_indices: bool = True,
        save_raw_states: bool = True,
        state_capture_interval: int = 1,
    ) -> None:
        self._env = env
        self._use_multi_reset_indices = use_multi_reset_indices
        self._save_raw_states = save_raw_states
        self._state_capture_interval = max(1, state_capture_interval)
        self._state_buffer: dict[str, Any] = {}
        self._physics_buffer: dict[str, Any] = {}
        self._raw_state_buffer: dict[int, list[dict[str, Any]]] = {i: [] for i in range(self._env.num_envs)}
        self._raw_step_buffer: dict[int, int] = {i: 0 for i in range(self._env.num_envs)}

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
        if self._save_raw_states:
            for env_id in env_ids.detach().cpu().tolist():
                self._raw_state_buffer[env_id] = []
                self._raw_step_buffer[env_id] = 0

    def record_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Record raw states for specified envs based on the capture interval."""
        if not self._save_raw_states:
            return
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        if env_ids.numel() == 0:
            return
        env_id_list = env_ids.detach().cpu().tolist()
        episode_length_buf = getattr(self._env, "episode_length_buf", None)
        episode_length_cpu = None
        if episode_length_buf is not None:
            episode_length_cpu = episode_length_buf.detach().cpu()
        should_capture = any(
            (self._raw_step_buffer[env_id] % self._state_capture_interval == 0) for env_id in env_id_list
        )
        if should_capture:
            current_state = _to_cpu(self._env.scene.get_state(is_relative=True))
            for env_id in env_id_list:
                if self._raw_step_buffer[env_id] % self._state_capture_interval == 0:
                    timestep = self._raw_step_buffer[env_id]
                    if episode_length_cpu is not None:
                        timestep = int(episode_length_cpu[env_id].item())
                    self._raw_state_buffer[env_id].append({
                        "timestep": timestep,
                        "state": _extract_env_state(current_state, env_id),
                    })
        for env_id in env_id_list:
            self._raw_step_buffer[env_id] += 1

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

    def fetch_raw_states(self, env_ids: torch.Tensor) -> list[list[dict[str, Any]]]:
        """Return raw state sequences for the specified envs."""
        if not self._save_raw_states:
            return []
        return [self._raw_state_buffer[int(env_id)] for env_id in env_ids.detach().cpu().tolist()]


def _to_cpu(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        return value.cpu()
    return value


def _extract_env_state(full_state: dict[str, Any], env_idx: int) -> dict[str, Any]:
    """Extract a single environment's state from a full scene state dict."""
    env_state: dict[str, Any] = {}
    for asset_type, assets_dict in full_state.items():
        env_state[asset_type] = {}
        for asset_name, asset_data in assets_dict.items():
            env_state[asset_type][asset_name] = {}
            for key, value in asset_data.items():
                if isinstance(value, torch.Tensor):
                    env_state[asset_type][asset_name][key] = value[env_idx].clone()
                else:
                    env_state[asset_type][asset_name][key] = value
    return env_state
