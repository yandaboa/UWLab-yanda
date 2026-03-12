"""Rollout pair storage utilities."""

from __future__ import annotations

import os
from typing import Any, Mapping, cast

import torch


class RolloutPairStorage:
    """Stores rollout pairs and flushes them to disk."""

    def __init__(self, max_num_pairs: int, save_dir: str, file_prefix: str = "rollout_pairs") -> None:
        self.max_num_pairs = max_num_pairs
        self.save_dir = save_dir
        self.file_prefix = file_prefix
        self.pairs: list[dict[str, Any]] = []
        self.save_index = 0
        self.total_pairs = 0
        self.saved_pairs = 0

    def add_pairs(self, pairs: list[dict[str, Any]]) -> None:
        if not pairs:
            return
        self.pairs.extend(pairs)
        self.total_pairs += len(pairs)
        if len(self.pairs) >= self.max_num_pairs:
            self.save()

    def save(self) -> str:
        os.makedirs(self.save_dir, exist_ok=True)
        filename = f"{self.file_prefix}_{self.save_index:06d}.pt"
        save_path = os.path.join(self.save_dir, filename)
        torch.save({"pairs": self.pairs}, save_path)
        self.saved_pairs += len(self.pairs)
        self.pairs = []
        self.save_index += 1
        print(f"[INFO]: Saved {len(self.pairs)} rollout pairs to {save_path}")
        return save_path

    def force_save(self) -> None:
        if self.pairs:
            self.save()


def infer_episode_length(episode: Mapping[str, Any]) -> int:
    length = episode.get("length")
    if isinstance(length, int):
        return length
    obs = episode.get("obs")
    if isinstance(obs, Mapping):
        return int(next(iter(obs.values())).shape[0])
    if isinstance(obs, torch.Tensor):
        return int(obs.shape[0])
    raise ValueError("Unable to infer demo episode length.")


def detach_to_cpu(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: detach_to_cpu(item) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def slice_env_data(data: Any, rollout_idx: int) -> Any:
    if isinstance(data, Mapping):
        return {key: slice_env_data(value, rollout_idx) for key, value in data.items()}
    if isinstance(data, torch.Tensor):
        # Slice per-episode entries from per-env buffers.
        return data[rollout_idx].detach().cpu()
    return data


def rollout_to_episode(
    rollouts: Mapping[str, Any],
    rollout_idx: int,
    env_id: int,
) -> dict[str, Any]:
    lengths = cast(torch.Tensor, rollouts["lengths"])
    length = int(lengths[rollout_idx].item())
    obs_rollouts = rollouts["obs"]
    if isinstance(obs_rollouts, Mapping):
        # Slice per-episode obs sequences from batched rollouts.
        obs_episode = {
            key: value[rollout_idx, :length].detach().cpu() for key, value in obs_rollouts.items()
        }
    else:
        # Slice per-episode obs sequence from batched rollouts.
        obs_episode = obs_rollouts[rollout_idx, :length].detach().cpu()
    episode = {
        "obs": obs_episode,
        "actions": rollouts["actions"][rollout_idx, :length].detach().cpu(),
        "rewards": rollouts["rewards"][rollout_idx, :length].detach().cpu(),
        "dones": rollouts["dones"][rollout_idx, :length].detach().cpu(),
        "length": length,
        "env_id": env_id,
    }
    if "states" in rollouts:
        episode["states"] = slice_env_data(rollouts["states"], rollout_idx)
    if "physics" in rollouts:
        episode["physics"] = slice_env_data(rollouts["physics"], rollout_idx)
    return episode


def demo_to_episode(demo_context: Any, episode_index: int, env_id: int) -> dict[str, Any]:
    episode = demo_context.episodes[episode_index]
    length = infer_episode_length(episode)
    obs = episode.get("obs")
    actions = episode.get("actions")
    rewards = episode.get("rewards")
    dones = episode.get("dones")
    demo_episode = {
        "obs": detach_to_cpu(obs),
        "actions": detach_to_cpu(actions),
        "rewards": detach_to_cpu(rewards),
        "dones": detach_to_cpu(dones) if dones is not None else None,
        "length": length,
        "env_id": env_id,
        "episode_index": episode_index,
        "states": detach_to_cpu(episode.get("states")),
        "physics": detach_to_cpu(episode.get("physics")),
    }
    return demo_episode
