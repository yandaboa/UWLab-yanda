"""Utilities shared by demo-tracking eval scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.nn.utils.rnn import pad_sequence

from metalearning.tools.visualization_utils import load_pairs
from metalearning.tools.visualize_trajectory import plot_episode_rewards, visualize_context_rollout_3d


def flatten_debug_obs(debug_obs: Any) -> dict[str, torch.Tensor]:
    if isinstance(debug_obs, Mapping):
        return {f"debug/{key}": value.detach().clone() for key, value in debug_obs.items()}
    if isinstance(debug_obs, torch.Tensor):
        return {"debug": debug_obs.detach().clone()}
    return {}


def update_demo_snapshot(
    demo_context: Any,
    env_ids: torch.Tensor,
    demo_obs_snapshot: dict[str, torch.Tensor] | None,
    demo_lengths_snapshot: torch.Tensor,
) -> None:
    if env_ids.numel() == 0:
        return
    env_ids_cpu = env_ids.detach().cpu()
    demo_lengths_snapshot[env_ids_cpu] = demo_context.demo_obs_lengths[env_ids].detach().cpu()
    if demo_obs_snapshot is None:
        return
    demo_obs_dict = demo_context.demo_obs_dict
    if demo_obs_dict is None:
        return
    for key, value in demo_obs_dict.items():
        demo_obs_snapshot[key][env_ids_cpu] = value[env_ids].detach().cpu()


def resolve_demo_actions(demo_actions: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(demo_actions, dict):
        if "demo" in demo_actions:
            return demo_actions["demo"]
        raise RuntimeError("Demo actions not found under 'demo' key.")
    return demo_actions


def get_replay_actions(demo_context: Any, num_envs: int, episode_length_buf: torch.Tensor) -> torch.Tensor:
    demo_actions = resolve_demo_actions(demo_context.demo_actions)
    lengths = demo_context.demo_obs_lengths.to(dtype=torch.long)
    end_idx = (lengths - 1).clamp(min=0)
    t = torch.minimum(episode_length_buf.long(), end_idx)
    env_ids = torch.arange(num_envs, device=demo_actions.device)
    # Index per-env, per-timestep action from the stored demo buffer.
    return demo_actions[env_ids, t, :]


def build_grouped_context_tensors(
    demo_context: Any,
    obs_keys: list[str],
    num_context_trajs: int,
    device: torch.device,
    env_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert num_context_trajs >= 1, "eval_num_context_trajs must be >= 1."
    episode_to_group = getattr(demo_context, "episode_to_group_index", None)
    episode_groups = getattr(demo_context, "episode_groups", None)
    if not isinstance(episode_to_group, list) or not isinstance(episode_groups, list) or len(episode_groups) == 0:
        raise RuntimeError("Grouped eval requires demo_context.episode_to_group_index and demo_context.episode_groups.")
    if demo_context.demo_obs_dict is None:
        raise RuntimeError("Grouped eval requires demo_obs_dict in demo context.")

    if env_ids is None:
        selected_env_ids = torch.arange(
            int(demo_context.episode_indices.shape[0]),
            device=demo_context.episode_indices.device,
            dtype=torch.long,
        )
    else:
        selected_env_ids = env_ids.to(device=demo_context.episode_indices.device, dtype=torch.long)
    assert selected_env_ids.numel() > 0, "Expected at least one env id when building grouped context tensors."
    env_id_list = [int(idx) for idx in selected_env_ids.detach().cpu().tolist()]
    env_count = len(env_id_list)
    grouped_obs: list[torch.Tensor] = []
    grouped_actions: list[torch.Tensor] = []
    grouped_rewards: list[torch.Tensor] = []
    grouped_lengths: list[int] = []

    for env_id in env_id_list:
        anchor_episode_idx = int(demo_context.episode_indices[env_id].item())
        assert anchor_episode_idx >= 0, "Context episode index is not initialized."
        group_idx = int(episode_to_group[anchor_episode_idx])
        group = episode_groups[group_idx]
        assert len(group) >= num_context_trajs, (
            f"Requested eval_num_context_trajs={num_context_trajs}, but selected group has only {len(group)} episodes."
        )
        candidate_indices = [idx for idx in group if idx != anchor_episode_idx]
        selected_indices = [anchor_episode_idx]
        if num_context_trajs > 1:
            assert len(candidate_indices) >= (num_context_trajs - 1), (
                f"Requested eval_num_context_trajs={num_context_trajs}, but only {len(candidate_indices)} "
                "episodes available in selected group."
            )
            perm = torch.randperm(len(candidate_indices)).tolist()
            selected_indices.extend([candidate_indices[idx] for idx in perm[: num_context_trajs - 1]])
        seq_obs: list[torch.Tensor] = []
        seq_actions: list[torch.Tensor] = []
        seq_rewards: list[torch.Tensor] = []
        total_len = 0
        for episode_idx in selected_indices:
            episode = demo_context.episodes[episode_idx]
            obs_terms = [episode["obs"][key].to(device).flatten(start_dim=1) for key in obs_keys]
            obs_tensor = torch.cat(obs_terms, dim=-1)
            actions_tensor = episode["actions"].to(device).reshape(episode["actions"].shape[0], -1)
            rewards_tensor = episode["rewards"].to(device).reshape(episode["rewards"].shape[0], -1)
            assert obs_tensor.shape[0] == actions_tensor.shape[0] == rewards_tensor.shape[0], (
                "Grouped context episode tensors must share the same sequence length."
            )
            seq_obs.append(obs_tensor)
            seq_actions.append(actions_tensor)
            seq_rewards.append(rewards_tensor)
            total_len += int(obs_tensor.shape[0])
        grouped_obs.append(torch.cat(seq_obs, dim=0))
        grouped_actions.append(torch.cat(seq_actions, dim=0))
        grouped_rewards.append(torch.cat(seq_rewards, dim=0))
        grouped_lengths.append(total_len)

    padded_obs = pad_sequence(grouped_obs, batch_first=True)
    padded_actions = pad_sequence(grouped_actions, batch_first=True)
    padded_rewards = pad_sequence(grouped_rewards, batch_first=True)
    return (
        padded_obs,
        padded_actions,
        padded_rewards,
        torch.tensor(grouped_lengths, device=device, dtype=torch.long).unsqueeze(-1),
    )


def resolve_context_log_dir(env_cfg: Any, retrieve_file_path_fn: Any) -> str | None:
    context_cfg = getattr(env_cfg, "context")
    if isinstance(context_cfg, dict):
        episode_paths = context_cfg.get("episode_paths")
    else:
        episode_paths = getattr(context_cfg, "episode_paths", None)
    assert episode_paths is not None
    if isinstance(episode_paths, str):
        episode_paths = [episode_paths]
    episode_list = list(episode_paths)
    resolved = retrieve_file_path_fn(str(episode_list[0]))
    return os.path.dirname(resolved)


def select_context_episode(pair: Mapping[str, Any]) -> Mapping[str, Any]:
    if "context" in pair:
        return pair["context"]
    if "demo" in pair:
        return pair["demo"]
    raise KeyError("Pair does not contain 'context' or 'demo' episode.")


def save_rollout_visualizations(rollout_dir: str) -> None:
    rollout_root = Path(rollout_dir)
    pair_paths = sorted(rollout_root.glob("rollout_pairs_*.pt"))
    if not pair_paths:
        print(f"[WARN] No rollout pair files found in {rollout_root}; skipping visualization.")
        return
    out_dir = rollout_root / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    rollout_episodes: list[Mapping[str, Any]] = []
    rollout_labels: list[str] = []
    global_pair_idx = 0
    for pair_path in pair_paths:
        pairs = load_pairs(pair_path)
        for pair in pairs:
            context_episode = select_context_episode(pair)
            rollout_episode = pair["rollout"]
            eef_out = out_dir / f"pair_{global_pair_idx:04d}_eef.png"
            try:
                visualize_context_rollout_3d(
                    context_episode,
                    rollout_episode,
                    out_path=eef_out,
                    backend="matplotlib",
                )
            except Exception as exc:
                print(f"[WARN] Failed to render pair {global_pair_idx} trajectory: {exc}")
            rollout_episodes.append(rollout_episode)
            rollout_labels.append(f"rollout_{global_pair_idx:04d}")
            global_pair_idx += 1

    if rollout_episodes:
        reward_out = out_dir / "pairs_rollout_rewards.png"
        try:
            plot_episode_rewards(
                rollout_episodes,
                rollout_labels,
                reward_out,
                backend="matplotlib",
            )
        except Exception as exc:
            print(f"[WARN] Failed to render rollout rewards plot: {exc}")

    print(f"[INFO] Saved rollout visualizations to: {out_dir}")
