from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset

from uwlab_rl.rsl_rl.discrete_action_utils import actions_to_indices


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def flatten_obs(obs: Any, obs_keys: list[str] | None, exclude_keys: str = "debug") -> torch.Tensor:
    if isinstance(obs, Mapping):
        keys = obs_keys or list(obs.keys())
        flat_terms = []
        for key in keys:
            value = obs[key]
            if exclude_keys in key:
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Observation key '{key}' is not a tensor.")
            flat_terms.append(value.reshape(value.shape[0], -1))
        return torch.cat(flat_terms, dim=-1)
    if isinstance(obs, torch.Tensor):
        return obs.reshape(obs.shape[0], -1)
    raise TypeError(f"Unsupported obs type: {type(obs)}")


def coerce_rewards(rewards: Any, length: int) -> torch.Tensor:
    if rewards is None:
        return torch.zeros((length, 1))
    if isinstance(rewards, torch.Tensor):
        if rewards.ndim == 1:
            return rewards[:length].unsqueeze(-1)
        return rewards[:length].reshape(length, -1)
    raise TypeError("Episode rewards must be a tensor or None.")


def pad_sequence_list(sequences: list[torch.Tensor], max_len: int, device: torch.device) -> torch.Tensor:
    if not sequences:
        raise ValueError("Cannot pad empty sequence list.")
    padded = []
    for seq in sequences:
        seq = seq.to(device)
        if seq.shape[0] < max_len:
            pad = torch.zeros((max_len - seq.shape[0], *seq.shape[1:]), device=device, dtype=seq.dtype)
            seq = torch.cat([seq, pad], dim=0)
        padded.append(seq)
    return torch.stack(padded, dim=0)


def collate_episodes(
    batch: list[dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    obs_list = [item["obs"] for item in batch]
    actions_list = [item["actions"] for item in batch]
    rewards_list = [item["rewards"] for item in batch]
    lengths = [int(item["length"].item()) for item in batch]
    max_len = max(lengths)
    return {
        "demo_obs": pad_sequence_list(obs_list, max_len, device),
        "demo_actions": pad_sequence_list(actions_list, max_len, device),
        "demo_rewards": pad_sequence_list(rewards_list, max_len, device),
        "demo_lengths": torch.tensor(lengths, device=device, dtype=torch.long).unsqueeze(-1),
    }


def resolve_distributed(distributed: bool) -> tuple[bool, int, int]:
    if not distributed:
        return False, 1, 0
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size() > 1, dist.get_world_size(), dist.get_rank()
    return False, 1, 0


def reduce_loss_if_needed(loss: torch.Tensor, is_multi_gpu: bool) -> torch.Tensor:
    if not is_multi_gpu:
        return loss
    if not (dist.is_available() and dist.is_initialized()):
        return loss
    reduced = loss.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced / float(dist.get_world_size())


def sequence_loss_mse(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (preds - targets) ** 2
    masked = diff.mean(dim=-1) * mask
    denom = mask.sum().clamp_min(1.0)
    return masked.sum() / denom


def sequence_loss_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    action_bins: list[int],
) -> torch.Tensor:
    batch, seq_len, _ = logits.shape
    losses = []
    start = 0
    for dim, bins in enumerate[int](action_bins):
        end = start + bins
        per_dim_logits = logits[..., start:end].reshape(batch * seq_len, bins)
        per_dim_target = targets[..., dim].reshape(batch * seq_len)
        per_dim_loss = F.cross_entropy(per_dim_logits, per_dim_target, reduction="none")
        per_dim_loss = per_dim_loss.reshape(batch, seq_len)
        losses.append(per_dim_loss)
        start = end
    mean_loss = torch.stack(losses, dim=0).mean(dim=0)
    masked = mean_loss * mask
    denom = mask.sum().clamp_min(1.0)
    return masked.sum() / denom


def load_action_discretization_spec(
    episode_paths: Iterable[str],
    spec_path: str | None = None,
) -> dict[str, Any] | None:
    if spec_path is not None and spec_path != "":
        candidate = Path(spec_path)
    else:
        first_path = next(iter(episode_paths), None)
        if first_path is None:
            return None
        candidate = Path(first_path).parent / "action_discretization_spec.json"
    if not candidate.exists():
        return None
    return json.loads(candidate.read_text())


def resolve_action_bins(
    spec: dict[str, Any],
    num_actions: int,
) -> list[int]:
    bin_values = spec["bin_values"]
    num_bins = int(spec["num_bins"])
    assert isinstance(bin_values, list) and len(bin_values) == num_actions
    return [num_bins for _ in range(num_actions)]


def resolve_action_bin_values(
    spec: dict[str, Any],
    num_actions: int,
) -> list[torch.Tensor]:
    """Resolve per-action bin values from the canonical action discretization spec."""
    bin_values = spec["bin_values"]
    num_bins = int(spec["num_bins"])
    assert isinstance(bin_values, list) and len(bin_values) == num_actions
    values = [torch.as_tensor(per_dim, dtype=torch.float32).reshape(-1) for per_dim in bin_values]
    assert all(v.numel() == num_bins for v in values)
    return values


def concat_context_and_rollout(
    demo_seq: torch.Tensor,
    demo_lengths: torch.Tensor,
    rollout_seq: torch.Tensor,
    rollout_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = demo_seq.device
    batch = demo_seq.shape[0]
    demo_lengths = demo_lengths.to(dtype=torch.long)
    rollout_lengths = rollout_lengths.to(dtype=torch.long)
    max_len = int((demo_lengths + rollout_lengths).max().item())
    combined = torch.zeros((batch, max_len, demo_seq.shape[-1]), device=device, dtype=demo_seq.dtype)
    combined_lengths = demo_lengths + rollout_lengths
    for idx in range(batch):
        d_len = int(demo_lengths[idx].item())
        r_len = int(rollout_lengths[idx].item())
        if d_len > 0:
            combined[idx, :d_len] = demo_seq[idx, :d_len]
        if r_len > 0:
            combined[idx, d_len : d_len + r_len] = rollout_seq[idx, :r_len]
    return combined, combined_lengths


def append_current_trajectory_prefix(
    context_obs: torch.Tensor,
    context_actions: torch.Tensor,
    context_rewards: torch.Tensor,
    current_traj_obs: torch.Tensor,
    current_traj_actions: torch.Tensor,
    current_traj_rewards: torch.Tensor,
    query_timestep: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Append current trajectory prefix [0, query_timestep) to context sequences."""
    assert context_obs.ndim == 2 and context_actions.ndim == 2 and context_rewards.ndim == 2
    assert current_traj_obs.ndim == 2 and current_traj_actions.ndim == 2 and current_traj_rewards.ndim == 2
    assert int(current_traj_obs.shape[0]) == int(current_traj_actions.shape[0]) == int(current_traj_rewards.shape[0])
    assert 0 <= query_timestep <= int(current_traj_obs.shape[0])
    if query_timestep == 0:
        return context_obs, context_actions, context_rewards
    obs = torch.cat([context_obs, current_traj_obs[:query_timestep]], dim=0)
    actions = torch.cat([context_actions, current_traj_actions[:query_timestep]], dim=0)
    rewards = torch.cat([context_rewards, current_traj_rewards[:query_timestep]], dim=0)
    return obs, actions, rewards


class ContextEpisodeDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset of context episodes stored in .pt files."""

    def __init__(self, episode_paths: Iterable[str], obs_keys: list[str] | None) -> None:
        self.obs_keys = obs_keys
        self.episodes: list[dict[str, Any]] = []
        for path in episode_paths:
            data = torch.load(path, map_location="cpu")
            episodes = data.get("episodes", [])
            self.episodes.extend(episodes)

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        episode = self.episodes[idx]
        obs = episode["obs"]
        actions = episode["actions"]
        rewards = episode.get("rewards")
        length = int(episode.get("length", actions.shape[0]))
        obs_seq = flatten_obs(obs, self.obs_keys, exclude_keys="debug")
        if self.obs_keys is None and isinstance(obs, Mapping):
            self.obs_keys = list[Any](obs.keys())
        return {
            "obs": obs_seq[:length],
            "actions": actions[:length].reshape(length, -1),
            "rewards": coerce_rewards(rewards, length),
            "length": torch.tensor(length, dtype=torch.long),
        }


class ContextStepDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset that yields per-step samples with full episode context."""

    def __init__(
        self,
        episode_paths: Iterable[str],
        obs_keys: list[str] | None,
        include_data_augs: bool = False,
        include_current_trajectory: bool = False,
    ) -> None:
        self.obs_keys = obs_keys
        self.include_data_augs = bool(include_data_augs)
        self.include_current_trajectory = bool(include_current_trajectory)
        print(f"INFO: initializing ContextStepDataset with include_data_augs={self.include_data_augs} and include_current_trajectory={self.include_current_trajectory}")
        assert not (self.include_data_augs and self.include_current_trajectory), (
            "ContextStepDataset does not support include_data_augs=True with "
            "include_current_trajectory=True."
        )
        self.episodes: list[dict[str, Any]] = []
        self.step_index: list[tuple[int, Literal["real", "aug"], int]] = []
        self.real_sample_count = 0
        self.aug_sample_count = 0
        for path in episode_paths:
            data = torch.load(path, map_location="cpu")
            print("Loaded episode data from ", path)
            episodes = data.get("episodes", [])
            self.episodes.extend(episodes)
        for ep_idx, episode in enumerate(self.episodes):
            actions = episode["actions"]
            length = int(episode.get("length", actions.shape[0]))
            self.step_index.extend([(ep_idx, "real", t) for t in range(length)])
            self.real_sample_count += length
            if self.include_data_augs:
                ccil_flat = episode.get("ccil_synthetic_flat")
                if isinstance(ccil_flat, Mapping):
                    aug_state = ccil_flat.get("state")
                    aug_action = ccil_flat.get("action")
                    aug_next = ccil_flat.get("next_state")
                    assert isinstance(aug_state, torch.Tensor)
                    assert isinstance(aug_action, torch.Tensor)
                    assert isinstance(aug_next, torch.Tensor)
                    aug_len = int(aug_state.shape[0])
                    assert int(aug_action.shape[0]) == aug_len
                    assert int(aug_next.shape[0]) == aug_len
                    self.step_index.extend([(ep_idx, "aug", t) for t in range(aug_len)])
                    self.aug_sample_count += aug_len

    def __len__(self) -> int:
        return len(self.step_index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_idx, sample_kind, t = self.step_index[idx]
        episode = self.episodes[ep_idx]
        obs = episode["obs"]
        actions = episode["actions"]
        rewards = episode.get("rewards")
        length = int(episode.get("length", actions.shape[0]))
        obs_seq = flatten_obs(obs, self.obs_keys, exclude_keys="debug")
        if self.obs_keys is None and isinstance(obs, Mapping):
            self.obs_keys = list(obs.keys())
        obs_seq = obs_seq[:length]
        actions = actions[:length].reshape(length, -1)
        rewards = coerce_rewards(rewards, length)
        current_obs = obs_seq[t] if sample_kind == "real" else episode["ccil_synthetic_flat"]["state"][t]
        target_action = actions[t] if sample_kind == "real" else episode["ccil_synthetic_flat"]["action"][t]
        context_obs = obs_seq
        context_actions = actions
        context_rewards = rewards
        if self.include_current_trajectory:
            assert sample_kind == "real", (
                "include_current_trajectory=True is only supported for real samples "
                "(data augmentations must be disabled)."
            )
            context_obs, context_actions, context_rewards = append_current_trajectory_prefix(
                context_obs,
                context_actions,
                context_rewards,
                obs_seq,
                actions,
                rewards,
                query_timestep=t,
            )
        return {
            "obs": context_obs,
            "actions": context_actions,
            "rewards": context_rewards,
            "length": torch.tensor(int(context_obs.shape[0]), dtype=torch.long),
            "current_obs": current_obs,
            "target_action": target_action,
        }


class EpisodeGroupDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset for grouped demonstrations with variable multi-episode context.

    Each sample is anchored by a query timestep from one episode in a group. The
    context is built from a random permutation prefix of *other* episodes in the
    group, and the query is taken from the held-out episode at the next position.
    The supervision target is the action at the query timestep.
    """

    def __init__(
        self,
        episode_paths: Iterable[str],
        obs_keys: list[str] | None,
        randomize_context: bool = True,
        base_seed: int = 0,
        num_context_episodes: int | None = None,
        minimum_num_context_trajs: int = 1,
        include_current_trajectory: bool = False,
    ) -> None:
        self.obs_keys = obs_keys
        self.randomize_context = bool(randomize_context)
        self.base_seed = int(base_seed)
        self.num_context_episodes = None if num_context_episodes is None else int(num_context_episodes)
        self.minimum_num_context_trajs = int(minimum_num_context_trajs)
        self.include_current_trajectory = bool(include_current_trajectory)
        print(f"INFO: include_current_trajectory={self.include_current_trajectory}")
        if self.minimum_num_context_trajs <= 0:
            raise ValueError("minimum_num_context_trajs must be positive.")
        self.groups: list[list[dict[str, torch.Tensor]]] = []
        self.step_index: list[tuple[int, int, int]] = []
        self.max_context_length = 0
        for path in episode_paths:
            data = torch.load(path, map_location="cpu")
            loaded_groups = data.get("episode_groups", [])
            assert isinstance(loaded_groups, list), f"Expected 'episode_groups' list in {path}."
            for raw_group in loaded_groups:
                assert isinstance(raw_group, list) and len(raw_group) > 0, f"Expected non-empty group in {path}."
                group: list[dict[str, torch.Tensor]] = []
                for episode in raw_group:
                    obs = episode["obs"]
                    actions = episode["actions"]
                    rewards = episode.get("rewards")
                    length = int(episode.get("length", actions.shape[0]))
                    obs_seq = flatten_obs(obs, self.obs_keys, exclude_keys="debug")
                    if self.obs_keys is None and isinstance(obs, Mapping):
                        self.obs_keys = list(obs.keys())
                    actions_seq = actions[:length].reshape(length, -1)
                    rewards_seq = coerce_rewards(rewards, length)
                    group.append(
                        {
                            "obs": obs_seq[:length],
                            "actions": actions_seq,
                            "rewards": rewards_seq,
                            "length": torch.tensor(length, dtype=torch.long),
                        }
                    )
                if len(group) <= 1:
                    # Skip singleton groups: no held-out query episode can be paired with non-empty context.
                    continue
                group_idx = len(self.groups)
                self.groups.append(group)
                group_total_len = 0
                min_ep_len = None
                for ep_idx, ep in enumerate(group):
                    length = int(ep["length"].item())
                    group_total_len += length
                    min_ep_len = length if min_ep_len is None else min(min_ep_len, length)
                    self.step_index.extend([(group_idx, ep_idx, t) for t in range(length)])
                assert min_ep_len is not None
                # Max context excludes the held-out query episode unless current trajectory prefix is appended.
                if self.include_current_trajectory:
                    self.max_context_length = max(self.max_context_length, max(group_total_len - 1, 0))
                else:
                    self.max_context_length = max(self.max_context_length, group_total_len - min_ep_len)

    def __len__(self) -> int:
        return len(self.step_index)

    def _sample_context_episode_indices(self, group_size: int, query_ep_idx: int, sample_idx: int) -> list[int]:
        if group_size <= 1:
            return []
        available = [idx for idx in range(group_size) if idx != query_ep_idx]
        max_context_eps = len(available)
        assert max_context_eps > 0
        if self.num_context_episodes is not None:
            if self.num_context_episodes > max_context_eps:
                return available
            num_context_eps = self.num_context_episodes
        elif self.randomize_context:
            if self.minimum_num_context_trajs > max_context_eps:
                raise ValueError(
                    f"minimum_num_context_trajs={self.minimum_num_context_trajs} exceeds "
                    f"available context episodes ({max_context_eps}) for held-out grouped sampling."
                )
            num_context_eps = random.randint(self.minimum_num_context_trajs, max_context_eps)
        else:
            generator = torch.Generator()
            generator.manual_seed(self.base_seed + sample_idx)
            if self.minimum_num_context_trajs > max_context_eps:
                raise ValueError(
                    f"minimum_num_context_trajs={self.minimum_num_context_trajs} exceeds "
                    f"available context episodes ({max_context_eps}) for held-out grouped sampling."
                )
            num_context_eps = int(
                torch.randint(
                    self.minimum_num_context_trajs,
                    max_context_eps + 1,
                    (1,),
                    generator=generator,
                ).item()
            )
        if self.randomize_context:
            perm = torch.randperm(max_context_eps).tolist()
        else:
            generator = torch.Generator()
            generator.manual_seed(self.base_seed + sample_idx)
            perm = torch.randperm(max_context_eps, generator=generator).tolist()
        return [available[idx] for idx in perm[:num_context_eps]]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        group_idx, query_ep_idx, t = self.step_index[idx]
        group = self.groups[group_idx]
        selected_ep_indices = self._sample_context_episode_indices(
            len(group), query_ep_idx, sample_idx=idx
        )
        query_episode = group[query_ep_idx]
        assert len(selected_ep_indices) > 0, "Expected at least one context episode for grouped samples."
        selected_obs = [group[ep_idx]["obs"] for ep_idx in selected_ep_indices]
        selected_actions = [group[ep_idx]["actions"] for ep_idx in selected_ep_indices]
        selected_rewards = [group[ep_idx]["rewards"] for ep_idx in selected_ep_indices]
        obs_seq = torch.cat(selected_obs, dim=0)
        actions_seq = torch.cat(selected_actions, dim=0)
        rewards_seq = torch.cat(selected_rewards, dim=0)
        if self.include_current_trajectory:
            obs_seq, actions_seq, rewards_seq = append_current_trajectory_prefix(
                obs_seq,
                actions_seq,
                rewards_seq,
                query_episode["obs"],
                query_episode["actions"],
                query_episode["rewards"],
                query_timestep=t,
            )
        return {
            "obs": obs_seq,
            "actions": actions_seq,
            "rewards": rewards_seq,
            "length": torch.tensor(int(obs_seq.shape[0]), dtype=torch.long),
            "current_obs": query_episode["obs"][t],
            "target_action": query_episode["actions"][t],
        }


class SplineQueryEpisodeGroupDataset(Dataset[dict[str, torch.Tensor]]):
    """Grouped dataset with query state/action from per-group spline reference.

    Each sample is:
      - context: concatenation of one or more noisy rollout episodes from the group
      - query state: from the group's spline/reference trajectory at timestep t
      - target action: from the same spline/reference timestep
    """

    def __init__(
        self,
        episode_paths: Iterable[str],
        obs_keys: list[str] | None,
        randomize_context: bool = True,
        base_seed: int = 0,
        num_context_episodes: int | None = None,
        minimum_num_context_trajs: int = 1,
        include_current_trajectory: bool = False,
    ) -> None:
        _ = obs_keys
        self.obs_keys = None
        self.randomize_context = bool(randomize_context)
        self.base_seed = int(base_seed)
        self.num_context_episodes = None if num_context_episodes is None else int(num_context_episodes)
        self.minimum_num_context_trajs = int(minimum_num_context_trajs)
        self.include_current_trajectory = bool(include_current_trajectory)
        if self.minimum_num_context_trajs <= 0:
            raise ValueError("minimum_num_context_trajs must be positive.")

        self.groups: list[list[dict[str, torch.Tensor]]] = []
        self.group_queries: list[dict[str, torch.Tensor]] = []
        self.step_index: list[tuple[int, int]] = []
        self.max_context_length = 0

        for path in episode_paths:
            data = torch.load(path, map_location="cpu")
            loaded_groups = data.get("episode_groups", [])
            loaded_splines = data.get("group_splines", [])
            assert isinstance(loaded_groups, list), f"Expected 'episode_groups' list in {path}."
            assert isinstance(loaded_splines, list), f"Expected 'group_splines' list in {path}."
            assert len(loaded_groups) == len(loaded_splines), (
                f"Mismatched grouped payload in {path}: "
                f"{len(loaded_groups)} episode groups vs {len(loaded_splines)} group_splines."
            )

            for raw_group, raw_spline in zip(loaded_groups, loaded_splines):
                assert isinstance(raw_group, list) and len(raw_group) > 0, f"Expected non-empty group in {path}."
                group: list[dict[str, torch.Tensor]] = []
                group_total_len = 0
                for episode in raw_group:
                    obs = episode["obs"]
                    actions = episode["actions"]
                    rewards = episode.get("rewards")
                    length = int(episode.get("length", actions.shape[0]))
                    obs_seq = flatten_obs(obs, None, exclude_keys="debug")
                    actions_seq = actions[:length].reshape(length, -1)
                    rewards_seq = coerce_rewards(rewards, length)
                    group.append(
                        {
                            "obs": obs_seq[:length],
                            "actions": actions_seq,
                            "rewards": rewards_seq,
                            "length": torch.tensor(length, dtype=torch.long),
                        }
                    )
                    group_total_len += length

                if len(group) == 0:
                    continue

                query_obs, query_actions = self._build_query_from_spline(raw_spline)
                query_len = int(min(query_obs.shape[0], query_actions.shape[0]))
                if query_len <= 0:
                    continue
                query_obs = query_obs[:query_len]
                query_actions = query_actions[:query_len]

                group_idx = len(self.groups)
                self.groups.append(group)
                self.group_queries.append({"obs": query_obs, "actions": query_actions})
                self.step_index.extend([(group_idx, t) for t in range(query_len)])
                if self.include_current_trajectory:
                    self.max_context_length = max(self.max_context_length, group_total_len + max(query_len - 1, 0))
                else:
                    self.max_context_length = max(self.max_context_length, group_total_len)

    def _build_query_from_spline(
        self,
        spline: Mapping[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(spline, Mapping):
            raise TypeError("Each group spline entry must be a mapping.")
        if "obs" not in spline or "actions" not in spline:
            raise ValueError(
                "Each group_splines entry must contain 'obs' and 'actions' "
                "saved in episode format."
            )

        query_obs = flatten_obs(spline["obs"], None, exclude_keys="debug")
        query_actions = torch.as_tensor(spline["actions"]).reshape(spline["actions"].shape[0], -1)

        return query_obs.to(dtype=torch.float32), query_actions.to(dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.step_index)

    def _sample_context_episode_indices(self, group_size: int, sample_idx: int) -> list[int]:
        if group_size <= 0:
            return []
        if group_size == 1:
            if self.num_context_episodes is not None and self.num_context_episodes != 1:
                raise ValueError(
                    f"Requested num_context_episodes={self.num_context_episodes}, "
                    "but group has only one episode."
                )
            return [0]
        available = list(range(group_size))
        if self.num_context_episodes is not None:
            if self.num_context_episodes > group_size:
                raise ValueError(
                    f"Requested num_context_episodes={self.num_context_episodes}, "
                    f"but group has only {group_size} episodes."
                )
            num_context_eps = self.num_context_episodes
        elif self.randomize_context:
            if self.minimum_num_context_trajs > group_size:
                raise ValueError(
                    f"minimum_num_context_trajs={self.minimum_num_context_trajs} exceeds "
                    f"group size ({group_size}) for grouped sampling."
                )
            num_context_eps = random.randint(self.minimum_num_context_trajs, group_size)
        else:
            generator = torch.Generator()
            generator.manual_seed(self.base_seed + sample_idx)
            if self.minimum_num_context_trajs > group_size:
                raise ValueError(
                    f"minimum_num_context_trajs={self.minimum_num_context_trajs} exceeds "
                    f"group size ({group_size}) for grouped sampling."
                )
            num_context_eps = int(
                torch.randint(
                    self.minimum_num_context_trajs,
                    group_size + 1,
                    (1,),
                    generator=generator,
                ).item()
            )
        if self.randomize_context:
            perm = torch.randperm(group_size).tolist()
        else:
            generator = torch.Generator()
            generator.manual_seed(self.base_seed + sample_idx)
            perm = torch.randperm(group_size, generator=generator).tolist()
        return [available[idx] for idx in perm[:num_context_eps]]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        group_idx, t = self.step_index[idx]
        group = self.groups[group_idx]
        query = self.group_queries[group_idx]

        selected_ep_indices = self._sample_context_episode_indices(len(group), sample_idx=idx)
        assert len(selected_ep_indices) > 0, "Expected at least one context episode."
        selected_obs = [group[ep_idx]["obs"] for ep_idx in selected_ep_indices]
        selected_actions = [group[ep_idx]["actions"] for ep_idx in selected_ep_indices]
        selected_rewards = [group[ep_idx]["rewards"] for ep_idx in selected_ep_indices]
        obs_seq = torch.cat(selected_obs, dim=0)
        actions_seq = torch.cat(selected_actions, dim=0)
        rewards_seq = torch.cat(selected_rewards, dim=0)
        if self.include_current_trajectory:
            query_rewards = torch.zeros(
                (query["obs"].shape[0], rewards_seq.shape[-1]),
                dtype=rewards_seq.dtype,
            )
            obs_seq, actions_seq, rewards_seq = append_current_trajectory_prefix(
                obs_seq,
                actions_seq,
                rewards_seq,
                query["obs"],
                query["actions"],
                query_rewards,
                query_timestep=t,
            )
        return {
            "obs": obs_seq,
            "actions": actions_seq,
            "rewards": rewards_seq,
            "length": torch.tensor(int(obs_seq.shape[0]), dtype=torch.long),
            "current_obs": query["obs"][t],
            "target_action": query["actions"][t],
        }


def collate_context_steps(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    obs_list = [item["obs"] for item in batch]
    actions_list = [item["actions"] for item in batch]
    rewards_list = [item["rewards"] for item in batch]
    lengths = [int(item["length"].item()) for item in batch]
    current_obs = torch.stack([item["current_obs"] for item in batch], dim=0)
    target_action = torch.stack([item["target_action"] for item in batch], dim=0)
    max_len = max(lengths)
    cpu_device = torch.device("cpu")
    return {
        "demo_obs": pad_sequence_list(obs_list, max_len, cpu_device),
        "demo_actions": pad_sequence_list(actions_list, max_len, cpu_device),
        "demo_rewards": pad_sequence_list(rewards_list, max_len, cpu_device),
        "demo_lengths": torch.tensor(lengths, device=cpu_device, dtype=torch.long).unsqueeze(-1),
        "current_obs": current_obs,
        "target_action": target_action,
    }
