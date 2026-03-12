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
        num_similar_trajectories: int | None = None,
    ) -> None:
        """Initialize episode storage."""
        self.max_num_episodes = max_num_episodes
        self.save_dir = save_dir
        self.file_prefix = file_prefix
        parsed_num_similar_trajectories = (
            int(num_similar_trajectories) if num_similar_trajectories is not None else None
        )
        if parsed_num_similar_trajectories is not None:
            assert parsed_num_similar_trajectories > 0, "num_similar_trajectories must be positive."
            self.num_similar_trajectories = (
                parsed_num_similar_trajectories if parsed_num_similar_trajectories > 1 else None
            )
        else:
            self.num_similar_trajectories = None
        self.episodes: list[Dict[str, torch.Tensor | Dict[str, torch.Tensor] | int | None]] = []
        self.episode_groups: list[list[Dict[str, torch.Tensor | Dict[str, torch.Tensor] | int | None]]] = []
        self._pending_episode_groups_by_env: dict[int, list[Dict[str, torch.Tensor | Dict[str, torch.Tensor] | int | None]]] = {}
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
        raw_states = rollouts.get("raw_states")
        success = rollouts.get("success")

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
            assert states is not None, "states must be provided for episode storage"
            assert physics is not None, "physics must be provided for episode storage"
            episode["states"] = self._slice_env_data(states, rollout_idx)
            episode["physics"] = self._slice_env_data(physics, rollout_idx)
            if raw_states is not None:
                episode["raw_states"] = self._slice_env_data(raw_states, rollout_idx)
            if success is not None:
                episode["success"] = self._slice_env_data(success, rollout_idx)
            self._append_episode(episode, env_id)
            self.total_episodes += 1
            if self._num_buffered_episodes() >= self.max_num_episodes:
                if self.save_dir is None:
                    raise RuntimeError("save_dir must be set to flush episodes.")
                self.save(self.save_dir)

    def save(self, save_dir: str) -> str:
        """Persist current episodes to disk and clear the buffer."""
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{self.file_prefix}_{self.save_index:06d}.pt"
        save_path = os.path.join(save_dir, filename)
        payload = self._build_save_payload(include_partial_groups=False)
        torch.save(payload, save_path)
        self.saved_episodes += int(payload["num_episodes"])
        self.episodes = []
        self.episode_groups = []
        self.save_index += 1
        return save_path

    def force_save(self) -> None:
        """Force save current episodes to disk and clear the buffer."""
        if self.save_dir is None:
            raise RuntimeError("save_dir must be set to flush episodes.")
        if self._num_buffered_episodes() == 0 and not self._pending_episode_groups_by_env:
            return
        os.makedirs(self.save_dir, exist_ok=True)
        filename = f"{self.file_prefix}_{self.save_index:06d}.pt"
        save_path = os.path.join(self.save_dir, filename)
        payload = self._build_save_payload(include_partial_groups=True)
        torch.save(payload, save_path)
        self.saved_episodes += int(payload["num_episodes"])
        self.episodes = []
        self.episode_groups = []
        self._pending_episode_groups_by_env.clear()
        self.save_index += 1

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
        if isinstance(data, list):
            return data[rollout_idx]
        return data

    def _append_episode(
        self,
        episode: Dict[str, torch.Tensor | Dict[str, torch.Tensor] | int | None],
        env_id: Optional[int],
    ) -> None:
        if self.num_similar_trajectories is None:
            self.episodes.append(episode)
            return
        assert env_id is not None, "Grouped episode storage requires env_id."
        current_group = self._pending_episode_groups_by_env.setdefault(env_id, [])
        current_group.append(episode)
        if len(current_group) >= self.num_similar_trajectories:
            self.episode_groups.append(current_group.copy())
            self._pending_episode_groups_by_env[env_id] = []

    def _num_buffered_episodes(self) -> int:
        if self.num_similar_trajectories is None:
            return len(self.episodes)
        return sum(len(group) for group in self.episode_groups)

    def _build_save_payload(self, include_partial_groups: bool) -> dict[str, Any]:
        if self.num_similar_trajectories is None:
            return {"episodes": self.episodes, "num_episodes": len(self.episodes)}
        episode_groups = list(self.episode_groups)
        if include_partial_groups:
            for group in self._pending_episode_groups_by_env.values():
                if group:
                    episode_groups.append(group.copy())
        num_episodes = sum(len(group) for group in episode_groups)
        return {
            "episode_groups": episode_groups,
            "num_episodes": num_episodes,
            "num_similar_trajectories": self.num_similar_trajectories,
        }
