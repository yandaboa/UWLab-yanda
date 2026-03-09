from __future__ import annotations

from typing import Any, Mapping

import torch

from uwlab_rl.rsl_rl.context_sequence_policy import ContextSequencePolicy
from uwlab_rl.rsl_rl.supervised_context_utils import concat_context_and_rollout, flatten_obs


class SupervisedEvalHelper:
    """Helper for supervised context policy evaluation."""

    def __init__(
        self,
        model: ContextSequencePolicy,
        demo_context: Any,
        device: torch.device,
        obs_keys: list[str],
    ) -> None:
        self.model = model
        self.demo_context = demo_context
        self.device = device
        if demo_context.demo_obs_dict is None:
            raise RuntimeError("Supervised eval requires demo_obs_dict.")
        if not obs_keys:
            raise RuntimeError("Supervised eval requires non-empty obs_keys.")
        self.obs_keys = obs_keys
        self.demo_obs = self._concat_demo_obs(demo_context.demo_obs_dict, self.obs_keys).to(self.device)
        self.demo_actions = self._resolve_demo_actions(demo_context.demo_actions).to(self.device)
        self.demo_rewards = demo_context.demo_rewards.unsqueeze(-1).to(self.device)
        self.demo_lengths = demo_context.demo_obs_lengths.unsqueeze(-1).to(self.device)
        self.include_current_trajectory = bool(model.cfg.input.include_current_trajectory)

    def refresh_envs(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        demo_obs_dict = self.demo_context.demo_obs_dict
        if demo_obs_dict is None:
            raise RuntimeError("Supervised eval requires demo_obs_dict.")
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self.demo_obs = self.demo_obs.to(self.device)
        self.demo_obs[env_ids] = self._concat_demo_obs(
            {k: v[env_ids] for k, v in demo_obs_dict.items()}, self.obs_keys
        ).to(self.device)
        demo_actions = self._resolve_demo_actions(self.demo_context.demo_actions).to(self.device)
        self.demo_actions[env_ids] = demo_actions[env_ids]
        self.demo_rewards[env_ids] = self.demo_context.demo_rewards[env_ids].unsqueeze(-1).to(self.device)
        self.demo_lengths[env_ids] = self.demo_context.demo_obs_lengths[env_ids].unsqueeze(-1).to(self.device)

    def act(self, previous_obs: Mapping[str, torch.Tensor], rollout_storage: Any) -> torch.Tensor:
        policy_obs = previous_obs["policy"]
        current_obs = flatten_obs(policy_obs, self.obs_keys, exclude_keys="debug").to(self.device)
        if self.include_current_trajectory:
            demo_obs, demo_actions, demo_rewards, demo_lengths = self._build_context_with_rollout_prefix(rollout_storage)
        else:
            demo_obs = self.demo_obs
            demo_actions = self.demo_actions
            demo_rewards = self.demo_rewards
            demo_lengths = self.demo_lengths
        return self.model.act(
            demo_obs,
            demo_actions,
            demo_rewards,
            demo_lengths,
            current_obs,
        )

    @staticmethod
    def _concat_demo_obs(demo_obs_dict: dict[str, torch.Tensor], obs_keys: list[str]) -> torch.Tensor:
        sequences = [demo_obs_dict[key].flatten(start_dim=2) for key in obs_keys]
        return torch.cat(sequences, dim=-1)

    @staticmethod
    def _resolve_demo_actions(demo_actions: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(demo_actions, dict):
            if "demo" in demo_actions:
                return demo_actions["demo"]
            raise RuntimeError("Supervised eval requires demo actions under 'demo' key.")
        return demo_actions

    def _build_rollout_obs(self, rollout_storage: Any, max_len: int) -> torch.Tensor:
        sequences = [
            rollout_storage.obs_buffers[key][:, :max_len].flatten(start_dim=2) for key in self.obs_keys
        ]
        return torch.cat(sequences, dim=-1)

    def _build_context_with_rollout_prefix(
        self,
        rollout_storage: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rollout_lengths = rollout_storage.env_steps.to(self.device, dtype=torch.long)
        max_rollout_len = int(rollout_lengths.max().item()) if rollout_lengths.numel() > 0 else 0
        if max_rollout_len == 0:
            return self.demo_obs, self.demo_actions, self.demo_rewards, self.demo_lengths

        rollout_obs = self._build_rollout_obs(rollout_storage, max_rollout_len).to(self.device)
        rollout_actions = rollout_storage.action_buffer[:, :max_rollout_len].to(self.device)
        rollout_rewards = rollout_storage.reward_buffer[:, :max_rollout_len].unsqueeze(-1).to(self.device)
        base_lengths = self.demo_lengths.squeeze(-1).to(self.device, dtype=torch.long)
        demo_obs, demo_lengths = concat_context_and_rollout(
            self.demo_obs,
            base_lengths,
            rollout_obs,
            rollout_lengths,
        )
        demo_actions, _ = concat_context_and_rollout(
            self.demo_actions,
            base_lengths,
            rollout_actions,
            rollout_lengths,
        )
        demo_rewards, _ = concat_context_and_rollout(
            self.demo_rewards,
            base_lengths,
            rollout_rewards,
            rollout_lengths,
        )
        return demo_obs, demo_actions, demo_rewards, demo_lengths.unsqueeze(-1)


class SupervisedOpenLoopEvalHelper(SupervisedEvalHelper):
    """Open-loop helper that feeds context observations as current_obs."""

    def __init__(
        self,
        model: ContextSequencePolicy,
        demo_context: Any,
        device: torch.device,
        obs_keys: list[str],
    ) -> None:
        super().__init__(model=model, demo_context=demo_context, device=device, obs_keys=obs_keys)
        assert not self.include_current_trajectory, (
            "SupervisedOpenLoopEvalHelper does not support include_current_trajectory=True yet."
        )
        self.context_step_idx = torch.zeros(self.demo_obs.shape[0], device=self.device, dtype=torch.long)

    def refresh_envs(self, env_ids: torch.Tensor) -> None:
        super().refresh_envs(env_ids)
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self.context_step_idx[env_ids] = 0

    def act(self, previous_obs: Mapping[str, torch.Tensor], rollout_storage: Any) -> torch.Tensor:
        current_obs = self._current_obs_from_context()
        actions = self.model.act(
            self.demo_obs,
            self.demo_actions,
            self.demo_rewards,
            self.demo_lengths,
            current_obs,
        )
        self._advance_context_steps()
        return actions

    def _current_obs_from_context(self) -> torch.Tensor:
        lengths = self.demo_lengths.squeeze(-1).to(dtype=torch.long)
        max_idx = (lengths - 1).clamp(min=0)
        step_idx = torch.minimum(self.context_step_idx, max_idx)
        env_ids = torch.arange(self.demo_obs.shape[0], device=self.device)
        return self.demo_obs[env_ids, step_idx]

    def _advance_context_steps(self) -> None:
        lengths = self.demo_lengths.squeeze(-1).to(dtype=torch.long)
        max_idx = (lengths - 1).clamp(min=0)
        self.context_step_idx = torch.minimum(self.context_step_idx + 1, max_idx)
