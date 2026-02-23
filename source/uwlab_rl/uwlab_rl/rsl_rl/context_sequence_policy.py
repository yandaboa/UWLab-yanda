from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from uwlab_rl.rsl_rl.context_token_builder import ContextTokenBuilder
from uwlab_rl.rsl_rl.supervised_context_cfg import SupervisedContextTrainerCfg
from uwlab_rl.rsl_rl.discrete_action_utils import actions_to_indices, indices_to_actions
from uwlab_rl.rsl_rl.transformers import ARDiscreteTransformerActor, TransformerActor


class ContextSequencePolicy(nn.Module):
    """Context-conditioned transformer policy for supervised models."""

    def __init__(
        self,
        cfg: SupervisedContextTrainerCfg,
        obs_dim: int,
        action_dim: int,
        reward_dim: int,
        action_bins: list[int] | None = None,
        action_bin_values: list[torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        token_dim = cfg.model.embedding_dim
        self.cfg = cfg
        self.layout = cfg.model.context_token_layout
        self.include_actions = cfg.model.include_actions_in_context
        self.include_rewards = cfg.model.include_rewards_in_context
        self.share_obs_projection = bool(cfg.model.share_current_and_context_obs_projection)
        self.projection_hidden_dim = cfg.model.encoding_projection_hidden_dim
        self.context_token_proj: nn.Module | None = None
        self.state_token_proj: nn.Module | None = None
        self.action_token_proj: nn.Module | None = None
        self.current_obs_proj: nn.Module | None = None
        if self.layout in {"state_action", "state_only"}:
            self.state_token_proj = self._make_projection(obs_dim, token_dim)
            if self.layout == "state_action":
                self.action_token_proj = self._make_projection(action_dim, token_dim)
            if not self.share_obs_projection:
                self.current_obs_proj = self._make_projection(obs_dim, token_dim)
        else:
            token_input_dim = obs_dim
            if self.include_actions:
                token_input_dim += action_dim
            if self.include_rewards:
                token_input_dim += reward_dim
            self.context_token_proj = self._make_projection(token_input_dim, token_dim)
            if not self.share_obs_projection:
                self.current_obs_proj = self._make_projection(obs_dim, token_dim)
        self.action_bins = action_bins
        self.action_bin_values = action_bin_values
        if cfg.model.action_distribution == "categorical":
            assert action_bins is not None and len(action_bins) > 0, (
                "Categorical actions require action_bins."
            )
            assert len(set(action_bins)) == 1, (
                "Autoregressive discrete actor requires a uniform bin count per action dimension."
            )
            self.actor = ARDiscreteTransformerActor(
                input_dim=token_dim,
                num_actions=cfg.model.num_actions,
                num_bins=int(action_bins[0]),
                embedding_dim=cfg.model.embedding_dim,
                hidden_dim=cfg.model.hidden_dim,
                num_layers=cfg.model.num_layers,
                num_heads=cfg.model.num_heads,
                max_len=101 + cfg.model.num_actions,
                attention_dropout=cfg.model.attention_dropout,
                residual_dropout=cfg.model.residual_dropout,
                embedding_dropout=cfg.model.embedding_dropout,
                causal=True,
            )
        else:
            self.actor = TransformerActor(
                input_dim=token_dim,
                num_actions=cfg.model.num_actions,
                actor_hidden_dims=[cfg.model.hidden_dim],
                action_bins=action_bins,
                categorical_actions=False,
                embedding_dim=cfg.model.embedding_dim,
                hidden_dim=cfg.model.hidden_dim,
                num_layers=cfg.model.num_layers,
                num_heads=cfg.model.num_heads,
                max_len=101,
                attention_dropout=cfg.model.attention_dropout,
                residual_dropout=cfg.model.residual_dropout,
                embedding_dropout=cfg.model.embedding_dropout,
                causal=True,
            )
        self.token_builder = ContextTokenBuilder(
            layout=cfg.model.context_token_layout,
            context_length_override=cfg.data.max_context_length,
            include_actions=cfg.model.include_actions_in_context,
            include_rewards=cfg.model.include_rewards_in_context,
            share_obs_projection=cfg.model.share_current_and_context_obs_projection,
        )

    def _make_projection(self, input_dim: int, output_dim: int) -> nn.Module:
        hidden_dim = self.projection_hidden_dim
        if hidden_dim is None:
            return nn.Linear(input_dim, output_dim)
        if hidden_dim <= 0:
            raise ValueError("encoding_projection_hidden_dim must be > 0 when provided.")
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def _resolve_current_obs_proj(self) -> nn.Module:
        if not self.share_obs_projection:
            if self.current_obs_proj is None:
                raise RuntimeError("current_obs_proj is not initialized.")
            return self.current_obs_proj
        if self.layout in {"state_action", "state_only"}:
            if self.state_token_proj is None:
                raise RuntimeError("state_token_proj is not initialized.")
            return self.state_token_proj
        if self.context_token_proj is None:
            raise RuntimeError("context_token_proj is not initialized.")
        return self.context_token_proj

    def build_tokens(
        self,
        demo_obs: torch.Tensor,
        demo_actions: torch.Tensor,
        demo_rewards: torch.Tensor,
        demo_lengths: torch.Tensor,
        current_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.token_builder.build_tokens_from_context(
            demo_obs=demo_obs,
            demo_actions=demo_actions,
            demo_rewards=demo_rewards,
            demo_lengths=demo_lengths,
            current_obs=current_obs,
            context_token_proj=self.context_token_proj,
            state_token_proj=self.state_token_proj,
            action_token_proj=self.action_token_proj,
            current_obs_proj=self._resolve_current_obs_proj(),
            current_obs_dim=current_obs.shape[-1],
        )
        return output.tokens, output.padding_mask, output.token_indices

    def act(
        self,
        demo_obs: torch.Tensor,
        demo_actions: torch.Tensor,
        demo_rewards: torch.Tensor,
        demo_lengths: torch.Tensor,
        current_obs: torch.Tensor,
    ) -> torch.Tensor:
        tokens, padding_mask, token_indices = self.build_tokens(
            demo_obs, demo_actions, demo_rewards, demo_lengths, current_obs
        )
        if self.cfg.model.action_distribution == "categorical":
            assert isinstance(self.actor, ARDiscreteTransformerActor)
            assert self.action_bins is not None, "action_bins must be provided for categorical actions."
            action_indices = self.actor.act(tokens, padding_mask=padding_mask, token_indices=token_indices)
            return indices_to_actions(action_indices, self.action_bins, self.action_bin_values)
        assert isinstance(self.actor, TransformerActor)
        return self.actor(tokens, padding_mask=padding_mask, token_indices=token_indices)

    def compute_supervised_loss(
        self,
        demo_obs: torch.Tensor,
        demo_actions: torch.Tensor,
        demo_rewards: torch.Tensor,
        demo_lengths: torch.Tensor,
        current_obs: torch.Tensor,
        target_action: torch.Tensor,
    ) -> torch.Tensor:
        tokens, padding_mask, token_indices = self.build_tokens(
            demo_obs, demo_actions, demo_rewards, demo_lengths, current_obs
        )
        if self.cfg.model.action_distribution == "categorical":
            assert self.action_bins is not None, "action_bins must be provided for categorical actions."
            assert isinstance(self.actor, ARDiscreteTransformerActor)
            target_idx = actions_to_indices(target_action, self.action_bins, self.action_bin_values)
            return self.actor.cross_entropy_loss(
                x=tokens,
                target_action_indices=target_idx,
                padding_mask=padding_mask,
                token_indices=token_indices,
            )
        assert isinstance(self.actor, TransformerActor)
        preds = self.actor(tokens, padding_mask=padding_mask, token_indices=token_indices)
        return F.mse_loss(preds, target_action)

    def get_state_dict_payload(self) -> dict[str, Any]:
        return {
            "state_dict": self.state_dict(),
            "cfg": asdict(self.cfg),
        }

    @staticmethod
    def from_checkpoint(
        checkpoint: dict[str, Any],
        device: torch.device,
    ) -> tuple["ContextSequencePolicy", dict[str, Any]]:
        cfg_dict = checkpoint.get("cfg")
        if cfg_dict is None:
            raise ValueError("Checkpoint missing cfg.")
        cfg = SupervisedContextTrainerCfg()
        cfg.from_dict(cfg_dict)
        meta = checkpoint.get("meta", {})
        obs_dim = int(meta["obs_dim"])
        action_dim = int(meta["action_dim"])
        reward_dim = int(meta["reward_dim"])
        action_bins = meta.get("action_bins")
        action_bin_values = meta.get("action_bin_values")
        if action_bin_values is not None:
            action_bin_values = [torch.as_tensor(values, dtype=torch.float32) for values in action_bin_values]
        model = ContextSequencePolicy(
            cfg,
            obs_dim,
            action_dim,
            reward_dim,
            action_bins=action_bins,
            action_bin_values=action_bin_values,
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        return model, meta
