from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules.actor_critic import ActorCritic as BaseActorCritic


class ActorCritic(BaseActorCritic):
    """Actor-critic with optional actor-only checkpoint loading."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.load_actor_only = False

    def enable_actor_only_loading(self, enabled: bool = True) -> None:
        """Enable loading only actor-related parameters from a checkpoint."""
        self.load_actor_only = bool(enabled)

    def _filter_actor_state_dict(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            key: value
            for key, value in state_dict.items()
            if not key.startswith("critic.")
            and not key.startswith("critic_obs_normalizer.")
        }

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load parameters, optionally skipping critic weights."""
        if self.load_actor_only:
            filtered_state = self._filter_actor_state_dict(state_dict)
            nn.Module.load_state_dict(self, filtered_state, strict=False)
            return False
        return super().load_state_dict(state_dict, strict=strict)
