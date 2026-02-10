from __future__ import annotations

from collections.abc import Generator

import torch
from tensordict import TensorDict

from rsl_rl.storage.rollout_storage import RolloutStorage


class DaggerRolloutStorage(RolloutStorage):
    """Rollout storage that aggregates distillation data on CPU."""

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
        *,
        max_buffer_steps: int,
        output_device: str,
        storage_device: str = "cpu",
    ) -> None:
        if training_type != "distillation":
            raise ValueError("DaggerRolloutStorage only supports distillation training.")
        self.output_device = output_device
        self.storage_device = storage_device
        self.rollout_steps = num_transitions_per_env
        self.max_buffer_steps = max(max_buffer_steps, num_transitions_per_env)
        super().__init__(training_type, num_envs, self.max_buffer_steps, obs, actions_shape, device=storage_device)
        self._write_index = 0
        self._num_filled = 0

    def add_transitions(self, transition: RolloutStorage.Transition) -> None:
        obs = transition.observations
        actions = transition.actions
        rewards = transition.rewards
        dones = transition.dones
        privileged_actions = transition.privileged_actions
        assert obs is not None
        assert actions is not None
        assert rewards is not None
        assert dones is not None
        assert privileged_actions is not None
        if self._num_filled < self.max_buffer_steps:
            idx = self._write_index
            self._write_index += 1
        else:
            # Advanced indexing: random replacement to avoid FIFO bias.
            idx = int(torch.randint(self.max_buffer_steps, (1,), device="cpu").item())
        self.observations[idx].copy_(obs.to(self.storage_device))
        self.actions[idx].copy_(actions.to(self.storage_device))
        self.rewards[idx].copy_(rewards.view(-1, 1).to(self.storage_device))
        self.dones[idx].copy_(
            dones.view(-1, 1).to(self.storage_device, dtype=self.dones.dtype)
        )
        self.privileged_actions[idx].copy_(privileged_actions.to(self.storage_device))
        self._write_index = self._write_index % self.max_buffer_steps
        self._num_filled = min(self._num_filled + 1, self.max_buffer_steps)

    def clear(self) -> None:
        return

    def generator(self) -> Generator:
        if self.training_type != "distillation":
            raise ValueError("This function is only available for distillation training.")
        if self._num_filled == 0:
            raise RuntimeError("No transitions available for DAgger sampling.")
        sample_idx = torch.randint(self._num_filled, (self.rollout_steps,), device="cpu")
        # Advanced indexing: sample a batch of rollout steps at once.
        obs_batch = self.observations[sample_idx].to(self.output_device)
        actions_batch = self.actions[sample_idx].to(self.output_device)
        privileged_actions_batch = self.privileged_actions[sample_idx].to(self.output_device)
        dones_batch = self.dones[sample_idx].to(self.output_device)
        for step in range(self.rollout_steps):
            yield (
                obs_batch[step],
                actions_batch[step],
                privileged_actions_batch[step],
                dones_batch[step],
            )
