from __future__ import annotations

from collections.abc import Generator
from typing import Any

import torch

from rsl_rl.storage.rollout_storage import RolloutStorage
from rsl_rl.utils import split_and_pad_trajectories


class DiscreteActionRolloutStorage(RolloutStorage):
    class Transition(RolloutStorage.Transition):
        def __init__(self) -> None:
            super().__init__()
            self.action_logits: torch.Tensor | None = None

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: Any,
        actions_shape: tuple[int] | list[int],
        device: str = "cpu",
        action_logits_shape: tuple[int] | list[int] | None = None,
    ) -> None:
        super().__init__(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            device,
        )
        self.action_logits = None
        if action_logits_shape is not None:
            self.action_logits = torch.zeros(
                num_transitions_per_env,
                num_envs,
                *action_logits_shape,
                device=self.device,
            )

    def add_transitions(self, transition: RolloutStorage.Transition) -> None:
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        if self.training_type == "distillation":
            self.privileged_actions[self.step].copy_(transition.privileged_actions)

        if self.training_type == "rl":
            self.values[self.step].copy_(transition.values)
            self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
            if transition.action_mean is not None:
                self.mu[self.step].copy_(transition.action_mean)
            if transition.action_sigma is not None:
                self.sigma[self.step].copy_(transition.action_sigma)
            if self.action_logits is not None:
                if getattr(transition, "action_logits", None) is None:
                    raise ValueError("action_logits must be set for discrete action storage.")
                self.action_logits[self.step].copy_(transition.action_logits)

        self._save_hidden_states(transition.hidden_states)
        self.step += 1

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator:
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)

        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)
        old_action_logits = self.action_logits.flatten(0, 1) if self.action_logits is not None else None

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                batch_idx = indices[start:stop]

                obs_batch = observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                old_action_logits_batch = old_action_logits[batch_idx] if old_action_logits is not None else None

                hidden_state_a_batch = None
                hidden_state_c_batch = None
                masks_batch = None

                if old_action_logits_batch is None:
                    yield (
                        obs_batch,
                        actions_batch,
                        target_values_batch,
                        advantages_batch,
                        returns_batch,
                        old_actions_log_prob_batch,
                        old_mu_batch,
                        old_sigma_batch,
                        (hidden_state_a_batch, hidden_state_c_batch),
                        masks_batch,
                    )
                else:
                    yield (
                        obs_batch,
                        actions_batch,
                        target_values_batch,
                        advantages_batch,
                        returns_batch,
                        old_actions_log_prob_batch,
                        old_mu_batch,
                        old_sigma_batch,
                        old_action_logits_batch,
                        (hidden_state_a_batch, hidden_state_c_batch),
                        masks_batch,
                    )