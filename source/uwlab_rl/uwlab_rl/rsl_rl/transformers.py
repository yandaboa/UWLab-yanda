from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn
from rsl_rl.networks.mlp import MLP

__all__ = [
    "ARDiscreteTransformerActor",
    "EpisodeEncoder",
    "MergedTokenTransformerActor",
    "PositionalEncoding",
    "StateOnlyTransformerActor",
    "StateActionTransformerActor",
    "TransformerActor",
    "TransformerBlock",
    "TransformerEncoder",
]


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for continuous inputs."""

    def __init__(self, hidden_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
        pe = torch.zeros(1, max_len, hidden_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_emb", pe)
        self.pos_emb: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to embeddings."""
        return x + self.pos_emb[:, : x.size(1)]


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and MLP."""

    def __init__(
        self,
        seq_len: int,
        hidden_dim: int,
        num_heads: int,
        attention_dropout: float,
        residual_dropout: float,
        causal: bool,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(residual_dropout)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, attention_dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(residual_dropout),
        )
        self.causal = causal
        if causal:
            causal_mask = ~torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
            self.register_buffer("causal_mask", causal_mask)
            self.causal_mask: torch.Tensor

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply attention and feedforward updates."""
        attn_mask = torch.jit.annotate(Optional[torch.Tensor], None)
        if self.causal:
            assert self.causal_mask is not None
            attn_mask = self.causal_mask[: x.shape[1], : x.shape[1]]
        norm_x = self.norm1(x)
        attention_out, attn_weights = self.attention(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            attn_mask=attn_mask,
            key_padding_mask=padding_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        x = x + self.drop(attention_out)
        x = x + self.mlp(self.norm2(x))
        if return_attention:
            return x, attn_weights
        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder for continuous input sequences."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: Optional[int] = None,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        max_len: int = 1024,
        attention_dropout: float = 0.1,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.1,
        causal: bool = True,
        attention_entropy_sample_ratio: float = 0.1,
    ) -> None:
        super().__init__()
        if embedding_dim is None:
            embedding_dim = hidden_dim
        if not (0.0 < attention_entropy_sample_ratio <= 1.0):
            raise ValueError("attention_entropy_sample_ratio must be in (0, 1].")
        self.input_proj = nn.Linear(input_dim, embedding_dim)
        self.pos_emb = PositionalEncoding(hidden_dim=embedding_dim, max_len=max_len)
        self.embed_proj = torch.jit.annotate(Optional[nn.Linear], None)
        if embedding_dim != hidden_dim:
            self.embed_proj = nn.Linear(embedding_dim, hidden_dim)

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.attention_entropy_sample_ratio = attention_entropy_sample_ratio
        self.emb_drop = nn.Dropout(embedding_dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    seq_len=max_len,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                    causal=causal,
                )
                for _ in range(num_layers)
            ]
        )
        self.register_buffer("_last_attention_entropy_per_head", torch.empty(0), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the final hidden states for the input sequence."""
        x = self.input_proj(x)
        x = self.pos_emb(x)
        x = self.emb_drop(x)
        if self.embed_proj is not None:
            x = self.embed_proj(x)
        entropies = []
        for block in self.blocks:
            x, attn_weights = block(x, padding_mask=padding_mask, return_attention=True)
            with torch.no_grad():
                entropies.append(
                    self._attention_entropy_per_head(
                        attn_weights,
                        padding_mask,
                        sample_ratio=self.attention_entropy_sample_ratio,
                    )
                )
        if entropies:
            self._last_attention_entropy_per_head = torch.stack(entropies, dim=0)
        return x

    def get_last_attention_entropy_per_head(self) -> torch.Tensor:
        """Return cached attention entropy per layer/head."""
        return self._last_attention_entropy_per_head

    @staticmethod
    def _attention_entropy_per_head(
        attn_weights: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        sample_ratio: float = 1.0,
    ) -> torch.Tensor:
        eps = 1.0e-8
        if sample_ratio < 1.0:
            batch_size = attn_weights.shape[0]
            num_samples = max(1, int(batch_size * sample_ratio))
            sample_idx = torch.randperm(batch_size, device=attn_weights.device)[:num_samples]
            # Subsample batch elements for entropy estimation (advanced indexing).
            attn_weights = attn_weights.index_select(dim=0, index=sample_idx)
            if padding_mask is not None:
                padding_mask = padding_mask.index_select(dim=0, index=sample_idx)
        weights = attn_weights.clamp_min(eps)
        entropy = -(weights * weights.log()).sum(dim=-1)
        if padding_mask is not None:
            valid = ~padding_mask
            entropy = entropy * valid.unsqueeze(1)
            denom = valid.sum().clamp_min(1).to(entropy.dtype).unsqueeze(0) * entropy.shape[1]
            return entropy.sum(dim=(0, 2)) / denom
        return entropy.mean(dim=(0, 2))


class EpisodeEncoder(TransformerEncoder):
    """Transformer encoder for episodes of (state, action, reward)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        reward_dim: int = 1,
        embedding_dim: Optional[int] = None,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        max_len: int = 1024,
        attention_dropout: float = 0.1,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.1,
        causal: bool = True,
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__(
            input_dim=state_dim + action_dim + reward_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_len=max_len,
            attention_dropout=attention_dropout,
            residual_dropout=residual_dropout,
            embedding_dropout=embedding_dropout,
            causal=causal,
        )
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode an episode sequence with (state, action, reward) tokens."""
        if rewards.ndim == states.ndim - 1:
            rewards = rewards.unsqueeze(-1)

        if rewards.shape[-1] != self.reward_dim:
            raise ValueError(f"Expected reward_dim={self.reward_dim}, got {rewards.shape[-1]}.")
        if states.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state_dim={self.state_dim}, got {states.shape[-1]}.")
        if actions.shape[-1] != self.action_dim:
            raise ValueError(f"Expected action_dim={self.action_dim}, got {actions.shape[-1]}.")

        if not (states.shape[:-1] == actions.shape[:-1] == rewards.shape[:-1]):
            raise ValueError("States, actions, and rewards must align on all non-feature dims.")

        x = torch.cat([states, actions, rewards], dim=-1)
        batch_shape = x.shape[:-2]
        seq_len = x.shape[-2]

        if batch_shape:
            # Flatten batch dims for parallel encoding across episodes.
            x = x.reshape(-1, seq_len, x.shape[-1])
            if padding_mask is not None:
                padding_mask = padding_mask.reshape(-1, seq_len)

        x = super().forward(x, padding_mask=padding_mask)

        if batch_shape:
            x = x.reshape(*batch_shape, seq_len, x.shape[-1])
        return x



class TransformerActor(nn.Module):
    """Transformer actor that maps token sequences to action means."""

    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        actor_hidden_dims: tuple[int] | list[int],
        action_bins: Optional[list[int]] = None,
        categorical_actions: bool = False,
        embedding_dim: Optional[int] = None,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        max_len: int = 180,
        attention_dropout: float = 0.1,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.1,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.categorical_actions = categorical_actions
        self.action_bins = tuple(action_bins) if action_bins is not None else None
        output_dim = num_actions
        if self.categorical_actions:
            if self.action_bins is None:
                raise ValueError("action_bins must be provided for categorical actions.")
            output_dim = int(sum(self.action_bins))
        self.encoder = TransformerEncoder(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_len=max_len,
            attention_dropout=attention_dropout,
            residual_dropout=residual_dropout,
            embedding_dropout=embedding_dropout,
            causal=causal,
        )
        self.action_head = MLP(hidden_dim, output_dim, actor_hidden_dims, "gelu")
        self.register_buffer("_last_hidden", torch.empty(0), persistent=False)
        self.register_buffer("_last_features", torch.empty(0), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        token_indices: Optional[torch.Tensor] = None,
        return_all_tokens: bool = False,
    ) -> torch.Tensor:
        """Return action means for selected tokens."""
        hidden = self.encoder(x, padding_mask=padding_mask)
        if return_all_tokens:
            return self.action_head(hidden)
        hidden = self._select_tokens(hidden, token_indices)
        self._last_hidden = hidden
        self._last_features = self.action_head[:-1](hidden)
        output = self.action_head[-1](self._last_features)
        if self._last_hidden.isnan().any() or output.isnan().any():
            raise ValueError("NaN detected in last hidden or output.")
        output = output.clamp_min(-50.0)
        output = output.clamp_max(50.0)
        return output

    def get_last_hidden_features(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        token_indices: Optional[torch.Tensor] = None,
        use_cached: bool = False,
    ) -> torch.Tensor:
        """Return action head features before the final linear layer."""
        if use_cached and self._last_features.numel() > 0:
            features = self._last_features
        else:
            hidden = self.encoder(x, padding_mask=padding_mask)
            hidden = self._select_tokens(hidden, token_indices)
            self._last_hidden = hidden
            features = self.action_head[:-1](hidden)
            self._last_features = features
        return features

    def split_action_logits(self, logits: torch.Tensor) -> list[torch.Tensor]:
        """Split flat logits into per-action logits."""
        if not self.categorical_actions or self.action_bins is None:
            raise RuntimeError("split_action_logits is only available for categorical actions.")
        return list(torch.split(logits, list(self.action_bins), dim=-1))

    @staticmethod
    def _select_tokens(
        hidden: torch.Tensor,
        token_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if token_indices is None:
            return hidden[:, -1, :]
        token_indices = token_indices.to(dtype=torch.long)
        batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
        # Select per-batch token positions (advanced indexing).
        return hidden[batch_indices, token_indices, :]


class MergedTokenTransformerActor(TransformerActor):
    """Transformer actor for merged (state, action, reward) tokens."""


class StateOnlyTransformerActor(TransformerActor):
    """Transformer actor for state-only token sequences."""


class StateActionTransformerActor(TransformerActor):
    """Transformer actor that expects state/action token sequences."""


class ARDiscreteTransformerActor(nn.Module):
    """Autoregressive discrete actor over action-token bins."""

    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        num_bins: int,
        embedding_dim: Optional[int] = None,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        max_len: int = 180,
        attention_dropout: float = 0.1,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.1,
        causal: bool = True,
    ) -> None:
        super().__init__()
        assert num_bins > 1, "num_bins must be greater than 1 for discrete autoregressive actions."
        self.num_actions = int(num_actions)
        self.num_bins = int(num_bins)
        self.encoder = TransformerEncoder(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_len=max_len,
            attention_dropout=attention_dropout,
            residual_dropout=residual_dropout,
            embedding_dropout=embedding_dropout,
            causal=causal,
        )
        self.action_token_embedding = nn.Embedding(self.num_bins, input_dim)
        self.action_head = nn.Linear(hidden_dim, self.num_bins)

    @staticmethod
    def _gather_positions(hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        batch_idx = torch.arange(hidden.shape[0], device=hidden.device).unsqueeze(1).expand_as(positions)
        return hidden[batch_idx, positions.to(dtype=torch.long), :]

    def _build_autoregressive_tokens(
        self,
        x: torch.Tensor,
        token_indices: torch.Tensor,
        action_prefix_tokens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Insert action-prefix tokens directly after each sample's current_obs token."""
        batch_size, base_len, token_dim = x.shape
        prefix_len = 0 if action_prefix_tokens is None else int(action_prefix_tokens.shape[1])
        full_len = base_len + prefix_len
        full_tokens = torch.zeros((batch_size, full_len, token_dim), device=x.device, dtype=x.dtype)
        valid_lens = token_indices.to(dtype=torch.long) + 1
        base_positions = torch.arange(base_len, device=x.device).unsqueeze(0)
        valid_base_mask = base_positions < valid_lens.unsqueeze(1)
        full_tokens[:, :base_len, :] = x * valid_base_mask.unsqueeze(-1).to(dtype=x.dtype)
        if prefix_len > 0:
            assert action_prefix_tokens is not None
            action_prefix_tokens = action_prefix_tokens.to(device=x.device, dtype=x.dtype)
            prefix_offsets = torch.arange(prefix_len, device=x.device, dtype=torch.long).unsqueeze(0)
            prefix_positions = valid_lens.unsqueeze(1) + prefix_offsets
            batch_idx = torch.arange(batch_size, device=x.device).unsqueeze(1).expand_as(prefix_positions)
            full_tokens[batch_idx, prefix_positions, :] = action_prefix_tokens
        full_positions = torch.arange(full_len, device=x.device).unsqueeze(0)
        full_padding_mask = full_positions >= (valid_lens + prefix_len).unsqueeze(1)
        return full_tokens, full_padding_mask

    def teacher_forcing_logits(
        self,
        x: torch.Tensor,
        target_action_indices: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        token_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return per-dimension action-bin logits using teacher forcing."""
        assert token_indices is not None, "token_indices are required for autoregressive discrete training."
        assert target_action_indices.ndim == 2, "Expected target_action_indices shape (B, num_actions)."
        assert target_action_indices.shape[1] == self.num_actions, (
            f"Expected num_actions={self.num_actions}, got {target_action_indices.shape[1]}."
        )
        token_indices = token_indices.to(dtype=torch.long)
        # For AR teacher forcing we condition on ground-truth prefix act_1..act_{n-1}
        # and predict act_1..act_n from positions [current_obs, act_1, ..., act_{n-1}].
        prefix_indices = target_action_indices[:, : self.num_actions - 1].to(dtype=torch.long)
        prefix_tokens = self.action_token_embedding(prefix_indices)
        full_tokens, full_padding_mask = self._build_autoregressive_tokens(
            x=x,
            token_indices=token_indices,
            action_prefix_tokens=prefix_tokens,
        )
        hidden = self.encoder(full_tokens, padding_mask=full_padding_mask)
        offset = torch.arange(self.num_actions, device=x.device, dtype=torch.long).unsqueeze(0)
        prediction_positions = token_indices.unsqueeze(1) + offset
        prediction_hidden = self._gather_positions(hidden, prediction_positions)
        return self.action_head(prediction_hidden)

    def cross_entropy_loss(
        self,
        x: torch.Tensor,
        target_action_indices: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        token_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute averaged autoregressive cross-entropy over action dimensions."""
        logits = self.teacher_forcing_logits(
            x=x,
            target_action_indices=target_action_indices,
            padding_mask=padding_mask,
            token_indices=token_indices,
        )
        return nn.functional.cross_entropy(
            logits.reshape(-1, self.num_bins),
            target_action_indices.to(dtype=torch.long).reshape(-1),
            reduction="mean",
        )

    def act(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        token_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Autoregressively decode action-bin indices."""
        assert token_indices is not None, "token_indices are required for autoregressive discrete inference."
        token_indices = token_indices.to(dtype=torch.long)
        generated: list[torch.Tensor] = []
        for action_idx in range(self.num_actions):
            if generated:
                action_prefix = torch.stack(generated, dim=1)
                action_prefix_tokens = self.action_token_embedding(action_prefix)
            else:
                action_prefix_tokens = None
            model_tokens, model_padding_mask = self._build_autoregressive_tokens(
                x=x,
                token_indices=token_indices,
                action_prefix_tokens=action_prefix_tokens,
            )
            hidden = self.encoder(model_tokens, padding_mask=model_padding_mask)
            prediction_pos = (token_indices + action_idx).unsqueeze(1)
            prediction_hidden = self._gather_positions(hidden, prediction_pos).squeeze(1)
            logits = self.action_head(prediction_hidden)
            generated.append(torch.argmax(logits, dim=-1))
        return torch.stack(generated, dim=-1)
