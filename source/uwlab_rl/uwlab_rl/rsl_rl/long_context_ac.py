from __future__ import annotations

from typing import Any, Optional, cast
import os

import torch
import torch.nn as nn
from torch.distributions import Normal
from tensordict import TensorDict

from rsl_rl.modules.actor_critic import ActorCritic, GSDENoiseDistribution
from uwlab_rl.rsl_rl.distributions import IndependentCategoricalDistribution
from .context_token_builder import ContextTokenBuilder
from .transformers import (
    EpisodeEncoder,
    MergedTokenTransformerActor,
    StateOnlyTransformerActor,
    TransformerActor,
    StateActionTransformerActor,
)
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
        action_distribution: str = "normal",
        action_discretization_spec: dict | None = None,
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
        cross_attention_merge: bool = False,
        obs_token_count: int = 4,
        context_token_layout: str | None = None,
        include_actions_in_context: bool = True,
        include_rewards_in_context: bool = True,
        share_current_and_context_obs_projection: bool = False,
        encoding_projection_hidden_dim: int | None = None,
        optimizer: Optional[Any] = None,
        log_attention_entropy: bool = False,
        attention_entropy_interval: int = 0,
        model_finetune_ckpt: str | None = None,
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
        self.action_distribution = action_distribution
        self.action_discretization_spec = action_discretization_spec
        self.action_bins: tuple[int, ...] | None = None
        self.action_bin_values: list[torch.Tensor] | None = None
        self._last_action_logits = torch.empty(0)
        if self.action_distribution == "categorical":
            if self.state_dependent_std:
                raise ValueError("state_dependent_std is not supported with categorical actions.")
            self.action_bins, self.action_bin_values = self._resolve_action_discretization(num_actions)
            self.distribution = IndependentCategoricalDistribution(
                list(self.action_bins),
                self.action_bin_values,
            )

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
        self.cross_attention_merge = cross_attention_merge
        self.obs_token_count = obs_token_count
        self.context_token_layout = context_token_layout
        self.include_actions_in_context = bool(include_actions_in_context)
        self.include_rewards_in_context = bool(include_rewards_in_context)
        self.share_obs_projection = bool(share_current_and_context_obs_projection)
        self.encoding_projection_hidden_dim = encoding_projection_hidden_dim
        self.context_length_override = context_length_override
        self._cached_padding_mask: Optional[torch.Tensor] = None
        self._cached_token_indices: Optional[torch.Tensor] = None
        self.context_token_proj: Optional[nn.Module] = None
        self.state_token_proj: Optional[nn.Module] = None
        self.action_token_proj: Optional[nn.Module] = None
        self.current_obs_proj: Optional[nn.Module] = None
        self.context_token_builder: Optional[ContextTokenBuilder] = None
        self.token_embed_dim: Optional[int] = None
        self.current_obs_dim: Optional[int] = None
        self.log_attention_entropy = bool(log_attention_entropy)
        self.attention_entropy_interval = int(attention_entropy_interval or 0)
        self.load_actor_only = False
        self.model_finetune_ckpt = model_finetune_ckpt

        if self.context_keys and isinstance(self.context_keys, str) and self.context_keys in obs:
            self._initialize_transformer(obs)
        if self.model_finetune_ckpt:
            if not self._use_transformer_actor():
                raise ValueError("model_finetune_ckpt requires a transformer actor layout.")
            if self.context_token_builder is None:
                raise ValueError("model_finetune_ckpt requires context observations to initialize.")
            self._load_finetune_checkpoint(self.model_finetune_ckpt)

    """
    This function can be called with obs[self.context_keys] as a TensorDict or as a concatenated tensor
    We want to handle both cases and return the demo_obs, demo_actions, demo_rewards, demo_lengths
    demo_obs is a tensor of shape (num_envs, max_length, num_obs)
    demo_actions is a tensor of shape (num_envs, max_length, num_actions)
    demo_rewards is a tensor of shape (num_envs, max_length, num_rewards)
    demo_lengths is a tensor of shape (num_envs)

    IMPORTANT: Obs will be altered in-place (if necessary) so that PPO will recognize the full context_obs (since it starts as a nested TensorDict, which is not compatible with PPO)
    this means it will be a concatenated tensor of shape (num_envs, max_length, num_obs + num_actions + num_rewards)
    we also need to add a length field to the obs tensor to indicate the length of the context
    """

    def _process_context_obs(self, obs: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        context_obs = obs[self.context_keys]
        if isinstance(context_obs, TensorDict):
            demo_obs = context_obs["context_obs"]
            demo_actions = context_obs["context_actions"]
            demo_rewards = context_obs["context_rewards"]
            demo_lengths = context_obs["context_lengths"]
            # max_length = demo_lengths.max()
            max_length = 100
            demo_obs = demo_obs.reshape(demo_obs.shape[0], max_length, -1)
            demo_actions = demo_actions.reshape(demo_actions.shape[0], max_length, -1)
            demo_rewards = demo_rewards.reshape(demo_rewards.shape[0], max_length, -1)
            obs[self.context_keys] = torch.cat([demo_obs, demo_actions, demo_rewards], dim=-1)
            obs["context_lengths"] = demo_lengths
        else:
            start_idx = 0
            demo_obs = context_obs[..., start_idx:start_idx + self.num_obs]
            start_idx += self.num_obs
            demo_actions = context_obs[..., start_idx:start_idx + self.num_actions]
            start_idx += self.num_actions
            demo_rewards = context_obs[..., start_idx:start_idx + self.num_rewards]
            demo_lengths = obs["context_lengths"]

        demo_lengths = demo_lengths.to(dtype=torch.int32)
        return demo_obs, demo_actions, demo_rewards, demo_lengths

    def _resolve_context_lengths(self, demo_lengths: torch.Tensor) -> tuple[torch.Tensor, int]:
        if self.context_token_builder is None:
            lengths = demo_lengths.to(dtype=torch.long).squeeze(-1)
            max_context_length = int(lengths.max().item())
            if self.context_length_override is not None:
                max_context_length = min(max_context_length, self.context_length_override)
                lengths = torch.clamp(lengths, max=max_context_length)
            return lengths, max_context_length
        return self.context_token_builder.resolve_context_lengths(demo_lengths)

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

    def _load_finetune_checkpoint(self, ckpt_path: str) -> None:
        if not ckpt_path:
            return
        assert isinstance(ckpt_path, str), "model_finetune_ckpt must be a string path."
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Finetune checkpoint not found: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            cfg = checkpoint.get("cfg")
            if isinstance(cfg, dict):
                model_cfg = cfg.get("model", {})
                ckpt_layout = model_cfg.get("context_token_layout")
                if ckpt_layout is not None:
                    assert (
                        ckpt_layout == self.context_token_layout
                    ), "Finetune checkpoint layout must match context_token_layout."
                ckpt_action_dist = model_cfg.get("action_distribution")
                if ckpt_action_dist is not None:
                    assert (
                        ckpt_action_dist == self.action_distribution
                    ), "Finetune checkpoint action_distribution must match."
        else:
            state_dict = checkpoint
        assert isinstance(state_dict, dict), "Finetune checkpoint must contain a state_dict."
        self.enable_actor_only_loading(True)
        self.load_state_dict(state_dict, strict=False)
        self.enable_actor_only_loading(False)
    
    def _initialize_transformer(
        self,
        obs: TensorDict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        demo_obs, demo_actions, demo_rewards, demo_lengths = self._process_context_obs(obs)

        self.num_obs = demo_obs.shape[-1]
        num_actions = demo_actions.shape[-1]
        assert num_actions == self.num_actions, "Number of actions must match, or you've shifted action spaces??"
        self.num_rewards = demo_rewards.shape[-1]
        _, max_context_length = self._resolve_context_lengths(demo_lengths)

        if self.context_encoder is None:
            num_actor_obs = 0
            for obs_group in self.obs_groups["policy"]:
                assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
                num_actor_obs += obs[obs_group].shape[-1]
            self.current_obs_dim = num_actor_obs
            if self._use_transformer_actor():
                if self.state_dependent_std:
                    raise ValueError("state_dependent_std is not supported with transformer-only actors.")
                token_embed_dim = int(self.transformer_args["embedding_dim"])
                self.token_embed_dim = token_embed_dim
                layout = self._resolve_context_layout()
                actor_class = self._resolve_transformer_actor_class(layout)
                if layout in {"state_action", "state_only"}:
                    self.state_token_proj = self._make_projection(self.num_obs, token_embed_dim)
                    if layout == "state_action":
                        self.action_token_proj = self._make_projection(self.num_actions, token_embed_dim)
                    if not self.share_obs_projection:
                        self.current_obs_proj = self._make_projection(num_actor_obs, token_embed_dim)
                else:
                    token_input_dim = self.num_obs
                    if self.include_actions_in_context:
                        token_input_dim += self.num_actions
                    if self.include_rewards_in_context:
                        token_input_dim += self.num_rewards
                    self.context_token_proj = self._make_projection(token_input_dim, token_embed_dim)
                    if not self.share_obs_projection:
                        self.current_obs_proj = self._make_projection(num_actor_obs, token_embed_dim)
                if self.share_obs_projection and self.num_obs != num_actor_obs:
                    raise ValueError(
                        "share_current_and_context_obs_projection=True requires matching current_obs "
                        f"and context_obs dims, got current_obs={num_actor_obs}, context_obs={self.num_obs}."
                    )
                max_token_length = max_context_length + 1
                if layout == "state_action":
                    max_token_length = (2 * max_context_length) + 1
                categorical_actions = self.action_distribution == "categorical"
                action_bins = list(self.action_bins or [])
                self.actor = actor_class(
                    input_dim=token_embed_dim,
                    num_actions=num_actions,
                    actor_hidden_dims=self.actor_hidden_dims,
                    action_bins=action_bins if categorical_actions else None,
                    categorical_actions=categorical_actions,
                    embedding_dim=token_embed_dim,
                    hidden_dim=self.transformer_args["hidden_dim"],
                    num_layers=self.transformer_args["num_layers"],
                    num_heads=self.transformer_args["num_heads"],
                    max_len=max_token_length,
                    attention_dropout=self.transformer_args["attention_dropout"],
                    residual_dropout=self.transformer_args["residual_dropout"],
                    embedding_dropout=self.transformer_args["embedding_dropout"],
                    causal=True,
                )
                self.context_encoder = self.actor.encoder
                self.context_token_builder = ContextTokenBuilder(
                    layout=layout,
                    context_length_override=self.context_length_override,
                    include_actions=self.include_actions_in_context,
                    include_rewards=self.include_rewards_in_context,
                    share_obs_projection=self.share_obs_projection,
                )
                if self.actor_obs_normalization:
                    self.actor_obs_normalizer = EmpiricalNormalization(token_embed_dim)
                else:
                    self.actor_obs_normalizer = torch.nn.Identity()
            else:
                self.context_encoder = EpisodeEncoder(
                    state_dim=self.num_obs,
                    action_dim=self.num_actions,
                    reward_dim=self.num_rewards,
                    **self.transformer_args,
                )
                if self.cross_attention_merge:
                    if self.obs_token_count < 1:
                        raise ValueError("obs_token_count must be at least 1.")
                    self.context_encoder.obs_query_proj = nn.Linear(
                        num_actor_obs,
                        self.context_encoder.hidden_dim * self.obs_token_count,
                    )
                    self.context_encoder.obs_cross_attn = nn.MultiheadAttention(
                        self.context_encoder.hidden_dim,
                        self.transformer_args["num_heads"],
                        batch_first=True,
                    )
                    policy_input_size = self.context_encoder.hidden_dim
                else:
                    policy_input_size = num_actor_obs + self.context_encoder.hidden_dim
                if self.state_dependent_std:
                    self.actor = MLP(policy_input_size, [2, num_actions], self.actor_hidden_dims, self.activation)
                else:
                    output_dim = num_actions
                    if self.action_distribution == "categorical":
                        output_dim = int(sum(self.action_bins or []))
                    self.actor = MLP(policy_input_size, output_dim, self.actor_hidden_dims, self.activation)
                if self.actor_obs_normalization:
                    self.actor_obs_normalizer = EmpiricalNormalization(policy_input_size)
                else:
                    self.actor_obs_normalizer = torch.nn.Identity()

            # set up buffers to cache output of transformer hidden states
            # self.hidden_states_cache = torch.zeros(num_envs, max_length, self.context_encoder.hidden_dim)

        return demo_obs, demo_actions, demo_rewards, demo_lengths

    def _resolve_context_layout(self) -> str:
        layout = self.context_token_layout
        if layout is None:
            raise ValueError("context_token_layout must be set for transformer-only actors.")
        return layout

    def _make_projection(self, input_dim: int, output_dim: int) -> nn.Module:
        hidden_dim = self.encoding_projection_hidden_dim
        if hidden_dim is None:
            return nn.Linear(input_dim, output_dim)
        if hidden_dim <= 0:
            raise ValueError("encoding_projection_hidden_dim must be > 0 when provided.")
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def _resolve_transformer_actor_class(self, layout: str) -> type[TransformerActor]:
        match layout:
            case "state_action":
                return StateActionTransformerActor
            case "merged":
                return MergedTokenTransformerActor
            case "state_only":
                return StateOnlyTransformerActor
            case _:
                raise ValueError(f"Unknown context_token_layout: {layout}")

    def _use_transformer_actor(self) -> bool:
        return self.context_token_layout is not None

    def get_context_obs(self, obs: TensorDict) -> Optional[torch.Tensor]:
        """Return the context observation tensor to append to base inputs."""
        if self._use_transformer_actor():
            return None
        if not self.context_keys:
            return None
        demo_obs, demo_actions, demo_rewards, demo_lengths = self._initialize_transformer(obs)
        assert self.context_encoder is not None, "Context encoder not initialized"
        hidden_states = self.context_encoder(demo_obs, demo_actions, demo_rewards)
        assert hidden_states.shape[0] == demo_obs.shape[0], "Number of environments must match"
        assert hidden_states.shape[1] == demo_lengths.max(), "Number of timesteps must match"
        assert len(hidden_states.shape) == 3, "Hidden states must be a 3D tensor"
        hidden_states = hidden_states[torch.arange(demo_obs.shape[0]), demo_lengths.squeeze(-1).to(dtype=torch.int32) - 1, :]
        assert len(hidden_states.shape) == 2 and hidden_states.shape[1] == self.context_encoder.hidden_dim, "Hidden states must be a 2D tensor of shape (num_envs, hidden_dim)"

        # TODO: include a caching to make stepping through environment faster?
        # need a flag for whether we're updating or running inference
        return hidden_states

    def _merge_obs_with_context(
        self,
        obs_tensor: torch.Tensor,
        context_tensor: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Merge base observations with context for actor/critic."""
        if context_tensor is None or not self.cross_attention_merge:
            if context_tensor is None:
                return obs_tensor
            return torch.cat([obs_tensor, context_tensor], dim=-1)
        assert self.context_encoder is not None, "Context encoder not initialized"
        if not hasattr(self.context_encoder, "obs_query_proj") or not hasattr(self.context_encoder, "obs_cross_attn"):
            raise RuntimeError("Cross-attention modules not initialized on context encoder.")
        obs_query_proj = cast(nn.Linear, self.context_encoder.obs_query_proj)
        obs_cross_attn = cast(nn.MultiheadAttention, self.context_encoder.obs_cross_attn)
        query_tokens = obs_query_proj(obs_tensor).reshape(
            obs_tensor.shape[0],
            self.obs_token_count,
            self.context_encoder.hidden_dim,
        )
        context_tokens = context_tensor.unsqueeze(1)
        attn_out, _ = obs_cross_attn(
            query=query_tokens,
            key=context_tokens,
            value=context_tokens,
            need_weights=False,
        )
        return attn_out.mean(dim=1)

    def get_attention_entropy_per_head(self) -> Optional[torch.Tensor]:
        """Return cached attention entropy per layer/head."""
        if self._use_transformer_actor():
            if not isinstance(self.actor, TransformerActor):
                return None
            return self.actor.encoder.get_last_attention_entropy_per_head()
        if self.context_encoder is None or not isinstance(self.context_encoder, EpisodeEncoder):
            return None
        return self.context_encoder.get_last_attention_entropy_per_head()

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        """Return policy observations augmented with context."""
        base_obs = super().get_actor_obs(obs)
        if not self._use_transformer_actor():
            context_obs = self.get_context_obs(obs)
            return self._merge_obs_with_context(base_obs, context_obs)
        tokens, padding_mask, token_indices = self._build_transformer_tokens(obs, base_obs)
        self._cached_padding_mask = padding_mask
        self._cached_token_indices = token_indices
        return tokens

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        """Return critic observations augmented with context."""
        base_obs = super().get_critic_obs(obs)
        return base_obs

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        """Sample actions from the policy with context-aware inputs."""
        obs_tensor = self.get_actor_obs(obs)
        if self._use_transformer_actor():
            obs_tensor = self._normalize_transformer_tokens(obs_tensor)
        else:
            obs_tensor = self.actor_obs_normalizer(obs_tensor)
        self._update_distribution(obs_tensor)
        actions = self.distribution.sample()
        if self.action_distribution == "categorical":
            return self._categorical_indices_to_values(actions)
        return actions

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        """Return deterministic actions with context-aware inputs."""
        obs_tensor = self.get_actor_obs(obs)
        if self._use_transformer_actor():
            obs_tensor = self._normalize_transformer_tokens(obs_tensor)
            self._update_distribution(obs_tensor)
            if self.action_distribution == "categorical":
                return self._categorical_indices_to_values(self.distribution.mode)
            return self.distribution.mean
        obs_tensor = self.actor_obs_normalizer(obs_tensor)
        if self.action_distribution == "categorical":
            self._update_distribution(obs_tensor)
            return self._categorical_indices_to_values(self.distribution.mode)
        if self.state_dependent_std:
            return self.actor(obs_tensor)[..., 0, :]
        return self.actor(obs_tensor)

    def act_from_context(
        self,
        demo_obs: torch.Tensor,
        demo_actions: torch.Tensor,
        demo_rewards: torch.Tensor,
        demo_lengths: torch.Tensor,
        current_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Return action mean for context-conditioned BC."""
        if self._use_transformer_actor():
            tokens, padding_mask, token_indices = self._build_transformer_tokens_from_context(
                demo_obs,
                demo_actions,
                demo_rewards,
                demo_lengths,
                current_obs,
            )
            obs_tensor = self._normalize_transformer_tokens(tokens)
            logits = self.actor(
                obs_tensor,
                padding_mask=padding_mask,
                token_indices=token_indices,
            )
            if self.action_distribution == "categorical":
                self._update_categorical_distribution(logits)
                return self._categorical_indices_to_values(self.distribution.mode)
            else:
                return logits
        if self.context_encoder is None:
            raise RuntimeError("Context encoder is not initialized.")
        context_hidden = self.context_encoder(demo_obs, demo_actions, demo_rewards)
        # Advanced indexing to pick the last valid context token per sample.
        context_hidden = context_hidden[
            torch.arange(demo_obs.shape[0], device=demo_obs.device),
            demo_lengths.squeeze(-1).to(dtype=torch.long) - 1,
            :,
        ]
        obs_tensor = self._merge_obs_with_context(current_obs, context_hidden)
        obs_tensor = self.actor_obs_normalizer(obs_tensor)
        if self.action_distribution == "categorical":
            logits = self.actor(obs_tensor)
            self._update_categorical_distribution(logits)
            return self._categorical_indices_to_values(self.distribution.mode)
        if self.state_dependent_std:
            return self.actor(obs_tensor)[..., 0, :]
        return self.actor(obs_tensor)

    def forward(
        self,
        demo_obs: torch.Tensor,
        demo_actions: torch.Tensor,
        demo_rewards: torch.Tensor,
        demo_lengths: torch.Tensor,
        current_obs: torch.Tensor,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """Forward pass for context-conditioned BC/DDP."""
        if self._use_transformer_actor():
            tokens, padding_mask, token_indices = self._build_transformer_tokens_from_context(
                demo_obs,
                demo_actions,
                demo_rewards,
                demo_lengths,
                current_obs,
            )
            obs_tensor = self._normalize_transformer_tokens(tokens)
            logits_or_mean = self.actor(
                obs_tensor,
                padding_mask=padding_mask,
                token_indices=token_indices,
            )
            if self.action_distribution == "categorical":
                self._last_action_logits = logits_or_mean
                self._update_categorical_distribution(logits_or_mean)
                if return_logits:
                    return logits_or_mean
                return self._categorical_indices_to_values(self.distribution.mode)
            return logits_or_mean
        if self.context_encoder is None:
            raise RuntimeError("Context encoder is not initialized.")
        context_hidden = self.context_encoder(demo_obs, demo_actions, demo_rewards)
        # Advanced indexing to pick the last valid context token per sample.
        context_hidden = context_hidden[
            torch.arange(demo_obs.shape[0], device=demo_obs.device),
            demo_lengths.squeeze(-1).to(dtype=torch.long) - 1,
            :,
        ]
        obs_tensor = self._merge_obs_with_context(current_obs, context_hidden)
        obs_tensor = self.actor_obs_normalizer(obs_tensor)
        if self.action_distribution == "categorical":
            logits = self.actor(obs_tensor)
            self._last_action_logits = logits
            self._update_categorical_distribution(logits)
            if return_logits:
                return logits
            return self._categorical_indices_to_values(self.distribution.mode)
        if self.state_dependent_std:
            return self.actor(obs_tensor)[..., 0, :]
        return self.actor(obs_tensor)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        """Evaluate the value function with context-aware inputs."""
        obs_tensor = self.get_critic_obs(obs)
        obs_tensor = self.critic_obs_normalizer(obs_tensor)
        return self.critic(obs_tensor)

    def _build_transformer_tokens(
        self,
        obs: TensorDict,
        current_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        demo_obs, demo_actions, demo_rewards, demo_lengths = self._initialize_transformer(obs)
        return self._build_transformer_tokens_from_context(
            demo_obs,
            demo_actions,
            demo_rewards,
            demo_lengths,
            current_obs,
        )

    def _build_transformer_tokens_from_context(
        self,
        demo_obs: torch.Tensor,
        demo_actions: torch.Tensor,
        demo_rewards: torch.Tensor,
        demo_lengths: torch.Tensor,
        current_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build transformer tokens from explicit context tensors."""
        if self.context_token_builder is None:
            raise RuntimeError("Context token builder is not initialized.")
        current_obs_proj = self._resolve_current_obs_proj_for_tokens()
        output = self.context_token_builder.build_tokens_from_context(
            demo_obs=demo_obs,
            demo_actions=demo_actions,
            demo_rewards=demo_rewards,
            demo_lengths=demo_lengths,
            current_obs=current_obs,
            context_token_proj=self.context_token_proj,
            state_token_proj=self.state_token_proj,
            action_token_proj=self.action_token_proj,
            current_obs_proj=current_obs_proj,
            current_obs_dim=self.current_obs_dim,
        )
        if output.tokens.isnan().any():
            raise ValueError("NaN detected in tokens.")
        return output.tokens, output.padding_mask, output.token_indices

    def _build_state_action_transformer_tokens_from_context(
        self,
        demo_obs: torch.Tensor,
        demo_actions: torch.Tensor,
        demo_lengths: torch.Tensor,
        current_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build state/action interleaved tokens (s1, a1, s2, a2, ...)."""
        if self.context_token_builder is None:
            raise RuntimeError("Context token builder is not initialized.")
        if self.context_token_builder.layout != "state_action":
            raise RuntimeError("State/action token builder not configured.")
        current_obs_proj = self._resolve_current_obs_proj_for_tokens()
        output = self.context_token_builder.build_tokens_from_context(
            demo_obs=demo_obs,
            demo_actions=demo_actions,
            demo_rewards=torch.empty_like(demo_actions[..., :1]),
            demo_lengths=demo_lengths,
            current_obs=current_obs,
            context_token_proj=self.context_token_proj,
            state_token_proj=self.state_token_proj,
            action_token_proj=self.action_token_proj,
            current_obs_proj=current_obs_proj,
            current_obs_dim=self.current_obs_dim,
        )
        return output.tokens, output.padding_mask, output.token_indices

    def _resolve_current_obs_proj_for_tokens(self) -> nn.Module:
        if not self.share_obs_projection:
            if self.current_obs_proj is None:
                raise RuntimeError("current_obs_proj is not initialized.")
            return self.current_obs_proj
        layout = self._resolve_context_layout()
        if layout in {"state_action", "state_only"}:
            if self.state_token_proj is None:
                raise RuntimeError("state_token_proj is not initialized.")
            return self.state_token_proj
        if self.context_token_proj is None:
            raise RuntimeError("context_token_proj is not initialized.")
        return self.context_token_proj

    def _normalize_transformer_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self.actor_obs_normalization:
            return tokens
        flat_tokens = tokens.reshape(-1, tokens.shape[-1])
        flat_tokens = self.actor_obs_normalizer(flat_tokens)
        return flat_tokens.reshape(tokens.shape)

    def get_last_hidden_features(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        if isinstance(self.actor, TransformerActor):
            return self.actor.get_last_hidden_features(
                obs_tensor,
                padding_mask=self._cached_padding_mask,
                token_indices=self._cached_token_indices,
                use_cached=True,
            )
        return self.actor[:-1](obs_tensor)

    def _update_distribution(self, obs_tensor: torch.Tensor) -> None:
        if self._use_transformer_actor() and self.state_dependent_std:
            raise ValueError("state_dependent_std is not supported with transformer-only actors.")
        if self.action_distribution == "categorical":
            if self._use_transformer_actor():
                logits = self.actor(
                    obs_tensor,
                    padding_mask=self._cached_padding_mask,
                    token_indices=self._cached_token_indices,
                )
            else:
                logits = self.actor(obs_tensor)
            self._last_action_logits = logits
            self._update_categorical_distribution(logits)
            return
        if self.state_dependent_std:
            return super()._update_distribution(obs_tensor)  # type: ignore[arg-type]
        if self._use_transformer_actor():
            mean = self.actor(
                obs_tensor,
                padding_mask=self._cached_padding_mask,
                token_indices=self._cached_token_indices,
            )
        else:
            mean = self.actor(obs_tensor)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        elif self.noise_std_type == "gsde":
            features = self.get_last_hidden_features(obs_tensor)
            distribution = cast(GSDENoiseDistribution, self.distribution)
            distribution.proba_distribution(mean, self.log_std, features)
            return
        else:
            raise ValueError(
                f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar', 'log', or 'gsde'"
            )
        self.distribution = Normal(mean, std)
        self._last_action_logits = torch.empty(0)

    @property
    def action_logits(self) -> torch.Tensor:
        if self._last_action_logits.numel() == 0:
            raise RuntimeError("Action logits are not available for the current policy.")
        return self._last_action_logits

    def _categorical_indices_to_values(self, indices: torch.Tensor) -> torch.Tensor:
        indices = indices.to(dtype=torch.long)
        if self.action_bin_values is None:
            return indices
        values = []
        for idx, bin_values in enumerate(self.action_bin_values):
            values.append(bin_values.to(indices.device)[indices[..., idx]])
        return torch.stack(values, dim=-1)

    def _update_categorical_distribution(self, logits: torch.Tensor) -> None:
        distribution = cast(IndependentCategoricalDistribution, self.distribution)
        distribution.proba_distribution(logits)

    def _resolve_action_discretization(
        self,
        num_actions: int,
    ) -> tuple[tuple[int, ...], list[torch.Tensor] | None]:
        spec = self.action_discretization_spec
        if spec is None:
            raise ValueError("Categorical actions require action_discretization_spec.")

        bin_values = spec.get("bin_values") or spec.get("bin_centers") or spec.get("bins")
        if bin_values is not None:
            if not isinstance(bin_values, list) or not bin_values:
                raise ValueError("bin_values must be a non-empty list.")
            if bin_values and not isinstance(bin_values[0], list):
                bin_values = [bin_values for _ in range(num_actions)]
            if len(bin_values) != num_actions:
                raise ValueError("bin_values length must match num_actions.")
            action_bins = tuple(len(values) for values in bin_values)
            values = [torch.as_tensor(values, dtype=torch.float32) for values in bin_values]
            return action_bins, values

        num_bins = spec.get("num_bins")
        min_actions = spec.get("min_actions")
        max_actions = spec.get("max_actions")
        if num_bins is None or min_actions is None or max_actions is None:
            raise ValueError("action_discretization_spec must include num_bins, min_actions, and max_actions.")

        bins = self._expand_action_param(num_bins, num_actions, param_name="num_bins", cast=int)
        mins = self._expand_action_param(min_actions, num_actions, param_name="min_actions", cast=float)
        maxs = self._expand_action_param(max_actions, num_actions, param_name="max_actions", cast=float)
        values = [torch.linspace(mins[i], maxs[i], bins[i], dtype=torch.float32) for i in range(num_actions)]
        return tuple(bins), values

    @staticmethod
    def _expand_action_param(
        value: Any,
        num_actions: int,
        param_name: str,
        cast: type,
    ) -> list[Any]:
        if isinstance(value, (list, tuple)):
            if len(value) != num_actions:
                raise ValueError(f"{param_name} length must match num_actions.")
            return [cast(item) for item in value]
        return [cast(value) for _ in range(num_actions)]

    def update_normalization(self, obs: TensorDict) -> None:
        if self._use_transformer_actor():
            if self.actor_obs_normalization:
                current_obs = super().get_actor_obs(obs)
                tokens, padding_mask, _ = self._build_transformer_tokens(obs, current_obs)
                # Update normalization with only valid tokens (advanced indexing).
                actor_obs_normalizer = cast(EmpiricalNormalization, self.actor_obs_normalizer)
                actor_obs_normalizer.update(tokens[~padding_mask])
            if self.critic_obs_normalization:
                critic_obs = self.get_critic_obs(obs)
                critic_obs_normalizer = cast(EmpiricalNormalization, self.critic_obs_normalizer)
                critic_obs_normalizer.update(critic_obs)
            return
        super().update_normalization(obs)
