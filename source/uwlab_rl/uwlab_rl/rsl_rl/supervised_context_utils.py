from __future__ import annotations

import json
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
    ) -> None:
        self.obs_keys = obs_keys
        self.include_data_augs = bool(include_data_augs)
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
        return {
            "obs": obs_seq,
            "actions": actions,
            "rewards": rewards,
            "length": torch.tensor(length, dtype=torch.long),
            "current_obs": obs_seq[t] if sample_kind == "real" else episode["ccil_synthetic_flat"]["state"][t],
            "target_action": actions[t] if sample_kind == "real" else episode["ccil_synthetic_flat"]["action"][t],
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
