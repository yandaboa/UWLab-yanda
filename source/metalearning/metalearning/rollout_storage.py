"""Rollout storage for per-environment buffers."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Tuple

import torch


class RolloutStorage:
    """Stores per-environment rollouts with fixed horizon."""

    def __init__(
        self,
        num_envs: int,
        max_steps: int,
        obs_shape: Tuple[int, ...] | Mapping[str, Tuple[int, ...]],
        action_shape: Tuple[int, ...],
        device: torch.device | str | None = None,
        obs_dtype: torch.dtype = torch.float32,
        action_dtype: torch.dtype = torch.float32,
        reward_dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize buffers for fixed-length rollouts."""
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        self.obs_buffers = self._allocate_obs_buffers(obs_shape, obs_dtype)
        self.action_buffer = torch.zeros(
            (num_envs, max_steps, *action_shape), dtype=action_dtype, device=self.device
        )
        self.reward_buffer = torch.zeros((num_envs, max_steps), dtype=reward_dtype, device=self.device)
        self.done_buffer = torch.zeros((num_envs, max_steps), dtype=torch.bool, device=self.device)
        self.env_steps = torch.zeros((num_envs,), dtype=torch.long, device=self.device)

    def add_step(
        self,
        obs: torch.Tensor | Mapping[str, torch.Tensor],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        env_ids: Iterable[int] | torch.Tensor | None = None,
    ) -> None:
        """Append one rollout step for the selected environments."""
        env_ids_tensor = self._env_ids_to_tensor(env_ids)
        step_idx = self.env_steps[env_ids_tensor]
        if torch.any(step_idx >= self.max_steps):
            raise RuntimeError("RolloutStorage is full for one or more envs; wipe before adding.")

        if isinstance(obs, Mapping):
            for key, value in obs.items():
                if key not in self.obs_buffers:
                    raise KeyError(f"Observation key '{key}' not in buffers.")
                # Advanced indexing to write per-env per-step obs.
                self.obs_buffers[key][env_ids_tensor, step_idx] = value
        else:
            if "_obs" not in self.obs_buffers:
                raise KeyError("Observation buffer not initialized for tensor obs.")
            # Advanced indexing to write per-env per-step obs.
            self.obs_buffers["_obs"][env_ids_tensor, step_idx] = obs

        # Advanced indexing to write per-env per-step data.
        self.action_buffer[env_ids_tensor, step_idx] = actions
        self.reward_buffer[env_ids_tensor, step_idx] = rewards
        self.done_buffer[env_ids_tensor, step_idx] = dones.to(dtype=torch.bool)

        self.env_steps[env_ids_tensor] = step_idx + 1

    def get_rollouts(
        self, env_ids: Iterable[int] | torch.Tensor
    ) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        """Return rollout buffers and lengths for selected envs."""
        env_ids_tensor = self._env_ids_to_tensor(env_ids)
        lengths = self.env_steps[env_ids_tensor]
        max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0

        obs_out: Dict[str, torch.Tensor] = {}
        for key, buffer in self.obs_buffers.items():
            obs_out[key] = buffer[env_ids_tensor, :max_len]

        return {
            "obs": obs_out if "_obs" not in obs_out else obs_out["_obs"],
            "actions": self.action_buffer[env_ids_tensor, :max_len],
            "rewards": self.reward_buffer[env_ids_tensor, :max_len],
            "dones": self.done_buffer[env_ids_tensor, :max_len],
            "lengths": lengths,
        }

    def wipe_envs(self, env_ids: Iterable[int] | torch.Tensor) -> None:
        """Clear buffers and reset steps for selected envs."""
        env_ids_tensor = self._env_ids_to_tensor(env_ids)
        for buffer in self.obs_buffers.values():
            # Advanced indexing to wipe per-env buffers.
            buffer[env_ids_tensor] = 0
        self.action_buffer[env_ids_tensor] = 0
        self.reward_buffer[env_ids_tensor] = 0
        self.done_buffer[env_ids_tensor] = False
        self.env_steps[env_ids_tensor] = 0

    def _allocate_obs_buffers(
        self, obs_shape: Tuple[int, ...] | Mapping[str, Tuple[int, ...]], obs_dtype: torch.dtype
    ) -> Dict[str, torch.Tensor]:
        if isinstance(obs_shape, Mapping):
            return {
                key: torch.zeros(
                    (self.num_envs, self.max_steps, *shape), dtype=obs_dtype, device=self.device
                )
                for key, shape in obs_shape.items()
            }
        return {
            "_obs": torch.zeros(
                (self.num_envs, self.max_steps, *obs_shape), dtype=obs_dtype, device=self.device
            )
        }

    def _env_ids_to_tensor(self, env_ids: Iterable[int] | torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(list(env_ids), device=self.device, dtype=torch.long)
