"""Episode storage for completed rollouts."""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Mapping, Optional, cast

import torch


class EpisodeStorage:
    """Stores full episodes and flushes them to disk."""

    def __init__(
        self,
        max_num_episodes: int,
        save_dir: Optional[str] = None,
        file_prefix: str = "episodes",
    ) -> None:
        """Initialize episode storage."""
        self.max_num_episodes = max_num_episodes
        self.save_dir = save_dir
        self.file_prefix = file_prefix
        self.episodes: list[Dict[str, torch.Tensor | Dict[str, torch.Tensor] | int | None]] = []
        self.save_index = 0
        self.total_episodes = 0
        self.saved_episodes = 0

    def add_episode(
        self,
        rollouts: Mapping[str, torch.Tensor | Mapping[str, torch.Tensor]],
        env_ids: Optional[Iterable[int] | torch.Tensor] = None,
    ) -> None:
        """Add completed episodes to storage."""
        lengths = cast(torch.Tensor, rollouts["lengths"])
        obs_rollouts = rollouts["obs"]
        actions = cast(torch.Tensor, rollouts["actions"])
        rewards = cast(torch.Tensor, rollouts["rewards"])
        dones = cast(torch.Tensor, rollouts["dones"])
        states = rollouts.get("states")
        physics = rollouts.get("physics")

        env_id_list = self._normalize_env_ids(env_ids, int(lengths.shape[0]))
        for rollout_idx, env_id in enumerate(env_id_list):
            length = int(lengths[rollout_idx].item())
            if length == 0:
                continue
            if isinstance(obs_rollouts, Mapping):
                obs_mapping = cast(Mapping[str, torch.Tensor], obs_rollouts)
                # Advanced indexing to slice per-episode obs.
                obs_episode = {
                    key: value[rollout_idx, :length].detach().cpu() for key, value in obs_mapping.items()
                }
            else:
                obs_tensor = cast(torch.Tensor, obs_rollouts)
                # Advanced indexing to slice per-episode obs.
                obs_episode = obs_tensor[rollout_idx, :length].detach().cpu()
            episode = {
                "obs": obs_episode,
                "actions": actions[rollout_idx, :length].detach().cpu(),
                "rewards": rewards[rollout_idx, :length].detach().cpu(),
                "dones": dones[rollout_idx, :length].detach().cpu(),
                "length": length,
                "env_id": env_id,
            }
            if states is not None:
                episode["states"] = self._slice_env_data(states, rollout_idx)
            if physics is not None:
                episode["physics"] = self._slice_env_data(physics, rollout_idx)
            self.episodes.append(episode)
            self.total_episodes += 1
            if len(self.episodes) >= self.max_num_episodes:
                if self.save_dir is None:
                    raise RuntimeError("save_dir must be set to flush episodes.")
                self.save(self.save_dir)

    def save(self, save_dir: str) -> str:
        """Persist current episodes to disk and clear the buffer."""
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{self.file_prefix}_{self.save_index:06d}.pt"
        save_path = os.path.join(save_dir, filename)
        torch.save({"episodes": self.episodes}, save_path)
        self.saved_episodes += len(self.episodes)
        self.episodes = []
        self.save_index += 1
        return save_path

    def _normalize_env_ids(
        self, env_ids: Optional[Iterable[int] | torch.Tensor], num_episodes: int
    ) -> list[Optional[int]]:
        if env_ids is None:
            return [None] * num_episodes
        if isinstance(env_ids, torch.Tensor):
            return [int(value) for value in env_ids.detach().cpu().tolist()]
        return [int(value) for value in env_ids]

    def _slice_env_data(self, data: Any, rollout_idx: int) -> Any:
        if isinstance(data, Mapping):
            return {key: self._slice_env_data(value, rollout_idx) for key, value in data.items()}
        if isinstance(data, torch.Tensor):
            # Advanced indexing to slice per-episode extras.
            return data[rollout_idx].detach().cpu()
        return data
