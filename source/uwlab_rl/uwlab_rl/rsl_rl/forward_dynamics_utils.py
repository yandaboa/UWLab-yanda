from __future__ import annotations

import glob
from pathlib import Path
from typing import Any, Iterable, Mapping, cast

import torch
import torch.nn as nn
from torch.utils.data import Dataset


def flatten_obs(obs: Any, obs_keys: list[str] | None, exclude_key_substring: str = "debug") -> torch.Tensor:
    if isinstance(obs, Mapping):
        keys = obs_keys or list(obs.keys())
        flat_terms = []
        for key in keys:
            if exclude_key_substring in key:
                continue
            value = obs[key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Observation key '{key}' is not a tensor.")
            flat_terms.append(value.reshape(value.shape[0], -1))
        if not flat_terms:
            raise ValueError("No observation keys selected after filtering.")
        return torch.cat(flat_terms, dim=-1)
    if isinstance(obs, torch.Tensor):
        return obs.reshape(obs.shape[0], -1)
    raise TypeError(f"Unsupported obs type: {type(obs)}")


def expand_episode_paths(episode_paths: Iterable[str]) -> list[str]:
    expanded: list[str] = []
    for path in episode_paths:
        matches = sorted(glob.glob(path))
        if matches:
            expanded.extend(matches)
        else:
            expanded.append(path)
    # preserve order but de-duplicate
    deduped: list[str] = []
    seen: set[str] = set()
    for path in expanded:
        resolved = str(Path(path))
        if resolved in seen:
            continue
        deduped.append(resolved)
        seen.add(resolved)
    return deduped


class InverseDynamicsTransitionDataset(Dataset[dict[str, torch.Tensor]]):
    """Samples (state_t, action_t, state_tp1) transitions from collected episodes."""

    def __init__(self, episode_paths: Iterable[str], obs_keys: list[str] | None) -> None:
        self.obs_keys = obs_keys
        self.episodes: list[dict[str, Any]] = []
        self.transition_index: list[tuple[int, int]] = []

        resolved_paths = expand_episode_paths(episode_paths)
        if not resolved_paths:
            raise ValueError("No episode paths provided for inverse dynamics dataset.")
        for path in resolved_paths:
            data = torch.load(path, map_location="cpu")
            episodes = data.get("episodes", [])
            self.episodes.extend(episodes)

        for ep_idx, episode in enumerate(self.episodes):
            actions = episode["actions"]
            length = int(episode.get("length", actions.shape[0]))
            if length < 2:
                continue
            self.transition_index.extend((ep_idx, t) for t in range(length - 1))

    def __len__(self) -> int:
        return len(self.transition_index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_idx, t = self.transition_index[idx]
        episode = self.episodes[ep_idx]
        obs = episode["obs"]
        actions = episode["actions"]
        length = int(episode.get("length", actions.shape[0]))

        obs_seq = flatten_obs(obs, self.obs_keys)
        if self.obs_keys is None and isinstance(obs, Mapping):
            self.obs_keys = list(obs.keys())
        obs_seq = obs_seq[:length]
        actions = actions[:length].reshape(length, -1)

        return {
            "state_t": obs_seq[t],
            "action_t": actions[t],
            "state_tp1": obs_seq[t + 1],
        }


def collate_inverse_dynamics(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "state_t": torch.stack([item["state_t"] for item in batch], dim=0),
        "action_t": torch.stack([item["action_t"] for item in batch], dim=0),
        "state_tp1": torch.stack([item["state_tp1"] for item in batch], dim=0),
    }


def _build_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation '{name}'.")


class ForwardDynamicsResidualMLP(nn.Module):
    """Simple MLP that predicts residual (state_tp1 - state_t) from (state_t, action_t)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 4,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2 for an input and output projection.")

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        input_dim = self.state_dim + self.action_dim
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(_build_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.state_dim))
        self.network = nn.Sequential(*layers)
        self.register_buffer("state_mean", torch.zeros(self.state_dim, dtype=torch.float32))
        self.register_buffer("state_std", torch.ones(self.state_dim, dtype=torch.float32))
        self.register_buffer("action_mean", torch.zeros(self.action_dim, dtype=torch.float32))
        self.register_buffer("action_std", torch.ones(self.action_dim, dtype=torch.float32))

    def set_input_normalization(
        self,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
    ) -> None:
        state_mean_buf = cast(torch.Tensor, self._buffers["state_mean"])
        state_std_buf = cast(torch.Tensor, self._buffers["state_std"])
        action_mean_buf = cast(torch.Tensor, self._buffers["action_mean"])
        action_std_buf = cast(torch.Tensor, self._buffers["action_std"])
        state_mean_buf.copy_(state_mean.to(device=state_mean_buf.device, dtype=state_mean_buf.dtype))
        state_std_buf.copy_(state_std.to(device=state_std_buf.device, dtype=state_std_buf.dtype))
        action_mean_buf.copy_(action_mean.to(device=action_mean_buf.device, dtype=action_mean_buf.dtype))
        action_std_buf.copy_(action_std.to(device=action_std_buf.device, dtype=action_std_buf.dtype))

    def get_input_normalization_payload(self) -> dict[str, list[float]]:
        state_mean_buf = cast(torch.Tensor, self._buffers["state_mean"])
        state_std_buf = cast(torch.Tensor, self._buffers["state_std"])
        action_mean_buf = cast(torch.Tensor, self._buffers["action_mean"])
        action_std_buf = cast(torch.Tensor, self._buffers["action_std"])
        return {
            "state_mean": state_mean_buf.detach().cpu().tolist(),
            "state_std": state_std_buf.detach().cpu().tolist(),
            "action_mean": action_mean_buf.detach().cpu().tolist(),
            "action_std": action_std_buf.detach().cpu().tolist(),
        }

    def forward(self, state_t: torch.Tensor, action_t: torch.Tensor) -> torch.Tensor:
        state_mean_buf = cast(torch.Tensor, self._buffers["state_mean"])
        state_std_buf = cast(torch.Tensor, self._buffers["state_std"])
        action_mean_buf = cast(torch.Tensor, self._buffers["action_mean"])
        action_std_buf = cast(torch.Tensor, self._buffers["action_std"])
        state_norm = (state_t - state_mean_buf) / state_std_buf
        action_norm = (action_t - action_mean_buf) / action_std_buf
        x = torch.cat([state_norm, action_norm], dim=-1)
        return self.network(x)

