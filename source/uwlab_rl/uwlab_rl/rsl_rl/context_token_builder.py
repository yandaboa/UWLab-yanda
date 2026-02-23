from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class TokenBuildOutput:
    """Outputs from context token construction."""

    tokens: torch.Tensor
    padding_mask: torch.Tensor
    token_indices: torch.Tensor
    target_mask: Optional[torch.Tensor]
    max_context_length: int


class ContextTokenBuilder:
    """Build transformer tokens for context-conditioned models."""

    def __init__(
        self,
        layout: str,
        context_length_override: Optional[int] = None,
        include_actions: bool = True,
        include_rewards: bool = True,
        share_obs_projection: bool = False,
    ) -> None:
        self.layout = layout
        self.context_length_override = context_length_override
        self.include_actions = include_actions
        self.include_rewards = include_rewards
        self.share_obs_projection = bool(share_obs_projection)

    def resolve_context_lengths(self, demo_lengths: torch.Tensor) -> tuple[torch.Tensor, int]:
        lengths = demo_lengths.to(dtype=torch.long).squeeze(-1)
        max_context_length = int(lengths.max().item())
        if self.context_length_override is not None:
            max_context_length = min(max_context_length, self.context_length_override)
            lengths = torch.clamp(lengths, max=max_context_length)
        return lengths, max_context_length

    def _pad_current_obs_for_shared_projection(
        self,
        current_obs: torch.Tensor,
        target_dim: int,
    ) -> torch.Tensor:
        current_dim = current_obs.shape[-1]
        if current_dim == target_dim:
            return current_obs
        if current_dim > target_dim:
            raise ValueError(
                "current_obs dim exceeds context token input dim when share_obs_projection is enabled "
                f"(current_obs={current_dim}, context_token={target_dim})."
            )
        pad_dim = target_dim - current_dim
        padding = torch.zeros(
            (*current_obs.shape[:-1], pad_dim),
            device=current_obs.device,
            dtype=current_obs.dtype,
        )
        return torch.cat([current_obs, padding], dim=-1)

    def build_tokens_from_context(
        self,
        demo_obs: torch.Tensor,
        demo_actions: torch.Tensor,
        demo_rewards: torch.Tensor,
        demo_lengths: torch.Tensor,
        current_obs: torch.Tensor,
        *,
        context_token_proj: Optional[nn.Module],
        state_token_proj: Optional[nn.Module],
        action_token_proj: Optional[nn.Module],
        current_obs_proj: Optional[nn.Module],
        current_obs_dim: Optional[int],
    ) -> TokenBuildOutput:
        if current_obs_dim is None or current_obs.shape[-1] != current_obs_dim:
            raise ValueError("Current observation dim must match policy obs dim.")
        if self.share_obs_projection:
            if self.layout in {"state_action", "state_only"}:
                if current_obs.shape[-1] != demo_obs.shape[-1]:
                    raise ValueError(
                        "share_obs_projection=True requires matching current_obs/context_obs dims, "
                        f"got current_obs={current_obs.shape[-1]} and context_obs={demo_obs.shape[-1]}."
                    )
            else:
                token_input_dim = demo_obs.shape[-1]
                if self.include_actions:
                    token_input_dim += demo_actions.shape[-1]
                if self.include_rewards:
                    token_input_dim += demo_rewards.shape[-1]
                current_obs = self._pad_current_obs_for_shared_projection(current_obs, token_input_dim)
        lengths, max_context_length = self.resolve_context_lengths(demo_lengths)
        if self.layout == "state_action":
            return self._build_state_action_tokens(
                demo_obs=demo_obs,
                demo_actions=demo_actions,
                current_obs=current_obs,
                lengths=lengths,
                max_context_length=max_context_length,
                state_token_proj=state_token_proj,
                action_token_proj=action_token_proj,
                current_obs_proj=current_obs_proj,
            )
        if self.layout == "state_only":
            return self._build_state_only_tokens(
                demo_obs=demo_obs,
                current_obs=current_obs,
                lengths=lengths,
                max_context_length=max_context_length,
                state_token_proj=state_token_proj,
                current_obs_proj=current_obs_proj,
            )
        return self._build_merged_tokens(
            demo_obs=demo_obs,
            demo_actions=demo_actions,
            demo_rewards=demo_rewards,
            current_obs=current_obs,
            lengths=lengths,
            max_context_length=max_context_length,
            context_token_proj=context_token_proj,
            current_obs_proj=current_obs_proj,
        )

    def build_sequence_action_targets(
        self,
        demo_actions: torch.Tensor,
        lengths: torch.Tensor,
        max_context_length: int,
        token_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if demo_actions.ndim != 3:
            raise ValueError("Expected demo_actions to have shape (B, T, A).")
        device = demo_actions.device
        batch_size, _, action_dim = demo_actions.shape
        target_actions = torch.zeros((batch_size, token_length, action_dim), device=device, dtype=demo_actions.dtype)
        target_mask = torch.zeros((batch_size, token_length), device=device, dtype=torch.bool)
        time_idx = torch.arange(max_context_length, device=device)
        valid_time = time_idx.unsqueeze(0) < lengths.unsqueeze(1)
        if self.layout == "state_action":
            token_positions = (2 * time_idx + 1).unsqueeze(0).expand(batch_size, -1)
            # Advanced indexing to place per-step action targets at action token positions.
            batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(token_positions)
            target_actions[batch_idx[valid_time], token_positions[valid_time]] = demo_actions[:, :max_context_length, :][
                valid_time
            ]
            target_mask[batch_idx[valid_time], token_positions[valid_time]] = True
            return target_actions, target_mask
        token_positions = time_idx.unsqueeze(0).expand(batch_size, -1)
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(token_positions)
        target_actions[batch_idx[valid_time], token_positions[valid_time]] = demo_actions[:, :max_context_length, :][
            valid_time
        ]
        target_mask[batch_idx[valid_time], token_positions[valid_time]] = True
        return target_actions, target_mask

    def _build_merged_tokens(
        self,
        demo_obs: torch.Tensor,
        demo_actions: torch.Tensor,
        demo_rewards: torch.Tensor,
        current_obs: torch.Tensor,
        lengths: torch.Tensor,
        max_context_length: int,
        *,
        context_token_proj: Optional[nn.Module],
        current_obs_proj: Optional[nn.Module],
    ) -> TokenBuildOutput:
        if context_token_proj is None:
            raise RuntimeError("context_token_proj must be set for merged tokens.")
        parts = [demo_obs]
        if self.include_actions:
            parts.append(demo_actions)
        if self.include_rewards:
            parts.append(demo_rewards)
        context_tokens = torch.cat(parts, dim=-1)
        context_tokens = context_token_proj(context_tokens)
        if current_obs_proj is None:
            raise RuntimeError("current_obs_proj must be set for transformer tokens.")
        context_tokens = context_tokens[:, :max_context_length, :]
        num_envs = context_tokens.shape[0]
        token_dim = context_tokens.shape[-1]
        max_token_length = max_context_length + 1
        tokens = torch.zeros(
            (num_envs, max_token_length, token_dim),
            device=context_tokens.device,
            dtype=context_tokens.dtype,
        )
        tokens[:, :max_context_length, :] = context_tokens
        current_token = current_obs_proj(current_obs)
        batch_indices = torch.arange(num_envs, device=context_tokens.device)
        # Insert current step token at the per-env index (advanced indexing).
        tokens[batch_indices, lengths, :] = current_token
        positions = torch.arange(max_token_length, device=context_tokens.device).unsqueeze(0)
        padding_mask = positions > lengths.unsqueeze(1)
        return TokenBuildOutput(
            tokens=tokens,
            padding_mask=padding_mask,
            token_indices=lengths,
            target_mask=None,
            max_context_length=max_context_length,
        )

    def _build_state_only_tokens(
        self,
        demo_obs: torch.Tensor,
        current_obs: torch.Tensor,
        lengths: torch.Tensor,
        max_context_length: int,
        *,
        state_token_proj: Optional[nn.Module],
        current_obs_proj: Optional[nn.Module],
    ) -> TokenBuildOutput:
        if state_token_proj is None:
            raise RuntimeError("state_token_proj must be set for state-only tokens.")
        if current_obs_proj is None:
            raise RuntimeError("current_obs_proj must be set for transformer tokens.")
        context_tokens = state_token_proj(demo_obs)
        context_tokens = context_tokens[:, :max_context_length, :]
        num_envs = context_tokens.shape[0]
        token_dim = context_tokens.shape[-1]
        max_token_length = max_context_length + 1
        tokens = torch.zeros(
            (num_envs, max_token_length, token_dim),
            device=context_tokens.device,
            dtype=context_tokens.dtype,
        )
        tokens[:, :max_context_length, :] = context_tokens
        current_token = current_obs_proj(current_obs)
        batch_indices = torch.arange(num_envs, device=context_tokens.device)
        # Insert current step token at the per-env index (advanced indexing).
        tokens[batch_indices, lengths, :] = current_token
        positions = torch.arange(max_token_length, device=context_tokens.device).unsqueeze(0)
        padding_mask = positions > lengths.unsqueeze(1)
        return TokenBuildOutput(
            tokens=tokens,
            padding_mask=padding_mask,
            token_indices=lengths,
            target_mask=None,
            max_context_length=max_context_length,
        )

    def _build_state_action_tokens(
        self,
        demo_obs: torch.Tensor,
        demo_actions: torch.Tensor,
        current_obs: torch.Tensor,
        lengths: torch.Tensor,
        max_context_length: int,
        *,
        state_token_proj: Optional[nn.Module],
        action_token_proj: Optional[nn.Module],
        current_obs_proj: Optional[nn.Module],
    ) -> TokenBuildOutput:
        if state_token_proj is None or action_token_proj is None or current_obs_proj is None:
            raise RuntimeError("State/action transformer projections are not initialized.")
        state_tokens = state_token_proj(demo_obs)
        action_tokens = action_token_proj(demo_actions)
        state_tokens = state_tokens[:, :max_context_length, :]
        action_tokens = action_tokens[:, :max_context_length, :]
        num_envs = state_tokens.shape[0]
        token_dim = state_tokens.shape[-1]
        max_token_length = (2 * max_context_length) + 1
        tokens = torch.zeros(
            (num_envs, max_token_length, token_dim),
            device=state_tokens.device,
            dtype=state_tokens.dtype,
        )
        tokens[:, 0 : 2 * max_context_length : 2, :] = state_tokens
        tokens[:, 1 : 2 * max_context_length : 2, :] = action_tokens
        current_token = current_obs_proj(current_obs)
        batch_indices = torch.arange(num_envs, device=state_tokens.device)
        token_indices = 2 * lengths
        # Insert current step token at the per-env index (advanced indexing).
        tokens[batch_indices, token_indices, :] = current_token
        positions = torch.arange(max_token_length, device=state_tokens.device).unsqueeze(0)
        padding_mask = positions > token_indices.unsqueeze(1)
        return TokenBuildOutput(
            tokens=tokens,
            padding_mask=padding_mask,
            token_indices=token_indices,
            target_mask=None,
            max_context_length=max_context_length,
        )
