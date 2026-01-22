from __future__ import annotations

from typing import Optional, Any
import math

import torch
import torch.nn as nn


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
    ) -> torch.Tensor:
        """Apply attention and feedforward updates."""
        attn_mask = None
        if self.causal:
            attn_mask = self.causal_mask[: x.shape[1], : x.shape[1]]
        norm_x = self.norm1(x)
        attention_out, _ = self.attention(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            attn_mask=attn_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        x = x + self.drop(attention_out)
        x = x + self.mlp(self.norm2(x))
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
    ) -> None:
        super().__init__()
        if embedding_dim is None:
            embedding_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim, embedding_dim)
        self.pos_emb = PositionalEncoding(hidden_dim=embedding_dim, max_len=max_len)
        self.embed_proj = None
        if embedding_dim != hidden_dim:
            self.embed_proj = nn.Linear(embedding_dim, hidden_dim)
        
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
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
        for block in self.blocks:
            x = block(x, padding_mask=padding_mask)
        return x


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
