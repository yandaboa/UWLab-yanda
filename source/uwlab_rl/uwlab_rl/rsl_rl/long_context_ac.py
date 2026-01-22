from __future__ import annotations

from typing import Any, Optional

import torch
from tensordict import TensorDict

from rsl_rl.modules.actor_critic import ActorCritic
from .transformer_encoder import EpisodeEncoder
from rsl_rl.networks.normalization import EmpiricalNormalization
from rsl_rl.networks.mlp import MLP

class LongContextActorCritic(ActorCritic):
    """Actor-critic that threads additional context into observations."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        context_keys: Optional[list[str]] = None,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        embedding_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        residual_dropout: float = 0.1,
        max_num_episodes: int = 1,
        context_length_override: int | None = None,
        optimizer: Optional[Any] = None,
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            num_actions=num_actions,
            actor_obs_normalization=actor_obs_normalization,
            critic_obs_normalization=critic_obs_normalization,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            state_dependent_std=state_dependent_std,
            **kwargs,
        )
        """
        Note:
            The context encoder is not used in the critic network.
            The critic network only uses the base critic observations, which should INCLUDE priviledged information that sums up necessary context
            For tracking a trajectory, this means including the state we are tracking
        """

        self.actor_hidden_dims = actor_hidden_dims
        self.activation = activation
        self.num_actions = num_actions

        self.transformer_args = {
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "embedding_dropout": embedding_dropout,
            "attention_dropout": attention_dropout,
            "residual_dropout": residual_dropout,
            "max_num_episodes": max_num_episodes,
            "context_length_override": context_length_override,
        }
        self.context_encoder = None
        self.context_keys = context_keys or "context"
        self.transformer_optimizer_cfg = optimizer

        if self.context_keys and isinstance(self.context_keys, str) and self.context_keys in obs:
            self._ensure_context_encoder(obs)
    
    def _ensure_context_encoder(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        context_obs = obs[self.context_keys]
        demo_obs = context_obs["context_obs"]
        demo_actions = context_obs["context_actions"]
        demo_rewards = context_obs["context_rewards"]
        demo_lengths = context_obs["context_lengths"]
        max_length = demo_lengths.max()
        demo_obs = demo_obs.reshape(demo_obs.shape[0], max_length, -1)
        demo_actions = demo_actions.reshape(demo_actions.shape[0], max_length, -1)
        demo_rewards = demo_rewards.reshape(demo_rewards.shape[0], max_length, -1)
        num_obs = demo_obs.shape[-1]
        num_actions = demo_actions.shape[-1]
        assert num_actions == self.num_actions, "Number of actions must match, or you've shifted action spaces??"
        num_rewards = demo_rewards.shape[-1]
        if self.context_encoder is None:
            num_actor_obs = 0
            for obs_group in self.obs_groups["policy"]:
                assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
                num_actor_obs += obs[obs_group].shape[-1]

            self.context_encoder = EpisodeEncoder(
                state_dim=num_obs,
                action_dim=num_actions,
                reward_dim=num_rewards,
                **self.transformer_args,
            )
            policy_input_size = num_actor_obs + self.context_encoder.hidden_dim
            if self.state_dependent_std:
                self.actor = MLP(policy_input_size, [2, num_actions], self.actor_hidden_dims, self.activation)
            else:
                self.actor = MLP(policy_input_size, num_actions, self.actor_hidden_dims, self.activation)
            if self.actor_obs_normalization:
                self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs + self.context_encoder.hidden_dim)
            else:
                self.actor_obs_normalizer = torch.nn.Identity()
        
        return demo_obs, demo_actions, demo_rewards, demo_lengths

    def get_context_obs(self, obs: TensorDict) -> Optional[torch.Tensor]:
        """Return the context observation tensor to append to base inputs."""
        if not self.context_keys:
            return None
        demo_obs, demo_actions, demo_rewards, demo_lengths = self._ensure_context_encoder(obs)
        assert self.context_encoder is not None, "Context encoder not initialized"
        hidden_states = self.context_encoder(demo_obs, demo_actions, demo_rewards)
        assert hidden_states.shape[0] == demo_obs.shape[0], "Number of environments must match"
        assert hidden_states.shape[1] == demo_lengths.max(), "Number of timesteps must match"
        assert len(hidden_states.shape) == 3, "Hidden states must be a 3D tensor"
        return hidden_states[torch.arange(demo_obs.shape[0]), demo_lengths.squeeze(-1) - 1, :]

    def _merge_obs_with_context(
        self,
        obs_tensor: torch.Tensor,
        context_tensor: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Merge base observations with context for actor/critic."""
        if context_tensor is None:
            return obs_tensor
        return torch.cat([obs_tensor, context_tensor], dim=-1)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        """Return policy observations augmented with context."""
        base_obs = super().get_actor_obs(obs)
        context_obs = self.get_context_obs(obs)
        return self._merge_obs_with_context(base_obs, context_obs)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        """Return critic observations augmented with context."""
        base_obs = super().get_critic_obs(obs)
        return base_obs

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        """Sample actions from the policy with context-aware inputs."""
        obs_tensor = self.get_actor_obs(obs)
        obs_tensor = self.actor_obs_normalizer(obs_tensor)
        self._update_distribution(obs_tensor)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        """Return deterministic actions with context-aware inputs."""
        obs_tensor = self.get_actor_obs(obs)
        obs_tensor = self.actor_obs_normalizer(obs_tensor)
        if self.state_dependent_std:
            return self.actor(obs_tensor)[..., 0, :]
        return self.actor(obs_tensor)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        """Evaluate the value function with context-aware inputs."""
        obs_tensor = self.get_critic_obs(obs)
        obs_tensor = self.critic_obs_normalizer(obs_tensor)
        return self.critic(obs_tensor)
