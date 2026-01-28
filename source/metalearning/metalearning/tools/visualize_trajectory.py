"""Visualize end effector and action trajectories from episode files.

Example usage:
    python -m metalearning.tools.visualize_trajectory /path/to/episodes_000000.pt
    python -m metalearning.tools.visualize_trajectory /path/to/episodes_000000.pt --episode-idx 3
    python -m metalearning.tools.visualize_trajectory /path/to/episodes_000000.pt --out-dir /tmp/plots
    python -m metalearning.tools.visualize_trajectory /path/to/rollout_pairs_000000.pt --episode-idx 3
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from .visualization_utils import (
    get_pose_obs,
    load_episodes,
    load_pairs,
    plot_series,
    plot_traj3d,
    plot_traj3d_pair,
    select_episode,
    select_pair,
    trim_to_length,
)


def _select_context_episode(pair: Mapping[str, Any]) -> Mapping[str, Any]:
    if "context" in pair:
        return pair["context"]
    if "demo" in pair:
        return pair["demo"]
    raise KeyError("Pair does not contain 'context' or 'demo' episode.")


def _get_debug_pose_obs(
    obs: Mapping[str, Any], obs_key: Optional[str] = None
) -> Tuple[torch.Tensor, str]:
    if obs_key is not None:
        return get_pose_obs(obs, obs_key)
    if "debug/end_effector_pose" in obs:
        value = obs["debug/end_effector_pose"]
        if not isinstance(value, torch.Tensor):
            raise TypeError("obs key 'debug/end_effector_pose' is not a tensor.")
        return value, "debug/end_effector_pose"
    debug_keys = [key for key in obs.keys() if key.startswith("debug/")]
    if debug_keys:
        key = debug_keys[0]
        value = obs[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"obs key '{key}' is not a tensor.")
        return value, key
    return get_pose_obs(obs, obs_key)


def _plot_traj3d_multi(
    trajectories: Sequence[torch.Tensor],
    labels: Sequence[str],
    title: str,
    out_path: Optional[Path],
) -> None:
    traj_np = []
    for traj in trajectories:
        data = traj.detach().cpu().numpy()
        if data.ndim != 2 or data.shape[1] < 3:
            raise ValueError(f"Expected (T, >=3) data, got {data.shape}.")
        data = data[:, :3]
        data = data[np.isfinite(data).all(axis=1)]
        if data.shape[0] < 2:
            raise ValueError("Not enough finite points to plot.")
        traj_np.append(data)
    all_points = np.concatenate(traj_np, axis=0)
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.cm.tab20(np.linspace(0, 1, len(traj_np)))
    for idx, data in enumerate(traj_np):
        ax.plot(data[:, 0], data[:, 1], data[:, 2], linewidth=1.6, color=colors[idx], label=labels[idx])
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    try:
        ax.set_box_aspect((maxs - mins))
    except Exception:
        pass
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=25, azim=45)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def visualize_context_rollout_3d(
    context_episode: Mapping[str, Any],
    rollout_episode: Mapping[str, Any],
    obs_key: Optional[str] = None,
    out_path: Optional[Path] = None,
) -> Tuple[str, str]:
    """Visualize context and rollout trajectories in a shared 3D plot."""
    context_length = int(context_episode["length"]) if "length" in context_episode else None
    rollout_length = int(rollout_episode["length"]) if "length" in rollout_episode else None
    context_obs_key = obs_key or "end_effector_pose"
    context_obs, context_key = get_pose_obs(context_episode["obs"], context_obs_key)
    rollout_obs, rollout_key = _get_debug_pose_obs(rollout_episode["obs"], obs_key)
    context_obs = trim_to_length(context_obs, context_length)
    rollout_obs = trim_to_length(rollout_obs, rollout_length)
    if context_obs.shape[-1] < 3 or rollout_obs.shape[-1] < 3:
        raise ValueError("Pose obs last dim must be at least 3.")
    context_eef = context_obs[..., :3]
    rollout_eef = rollout_obs[..., :3]
    plot_traj3d_pair(context_eef, rollout_eef, "Context vs Rollout", out_path)
    return context_key, rollout_key


def main() -> None:
    """Run the visualization script."""
    parser = argparse.ArgumentParser(description="Visualize episode trajectories from .pt files.")
    parser.add_argument("path", type=Path, help="Path to a .pt episode file.")
    parser.add_argument("--episode-idx", type=int, default=0, help="Episode index to visualize.")
    parser.add_argument(
        "--episode-idxs",
        type=str,
        default=None,
        help="Comma-separated list of episode indices to overlay.",
    )
    parser.add_argument("--obs-key", type=str, default=None, help="Override key for debug obs.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to save plots (default: <episode_dir>/visualizations).",
    )
    parser.add_argument("--plot-actions", action="store_true", help="Plot actions.")
    args = parser.parse_args()

    out_dir = args.out_dir or (args.path.parent / "visualizations")
    out_dir.mkdir(parents=True, exist_ok=True)
    data = torch.load(args.path, map_location="cpu")
    if args.episode_idxs is not None:
        episode_idxs = [int(value) for value in args.episode_idxs.split(",") if value.strip()]
    else:
        episode_idxs = [args.episode_idx]

    if isinstance(data, dict) and "pairs" in data:
        pairs = load_pairs(args.path)
        if len(episode_idxs) == 1:
            pair = select_pair(pairs, episode_idxs[0])
            context_episode = _select_context_episode(pair)
            rollout_episode = pair["rollout"]
            eef_out = out_dir / f"pair_{episode_idxs[0]:04d}_eef.png"
            context_key, rollout_key = visualize_context_rollout_3d(
                context_episode, rollout_episode, obs_key=args.obs_key, out_path=eef_out
            )
            print(f"[INFO] Saved plot: {eef_out}")
            if args.plot_actions and isinstance(rollout_episode.get("actions"), torch.Tensor):
                actions = rollout_episode["actions"]
                rollout_length = int(rollout_episode["length"]) if "length" in rollout_episode else None
                actions = trim_to_length(actions, rollout_length)
                action_out = out_dir / f"pair_{episode_idxs[0]:04d}_actions.png"
                plot_series(actions, f"Rollout Actions ({rollout_key})", "action", action_out)
                print(f"[INFO] Saved plot: {action_out}")
        else:
            context_trajs = []
            rollout_trajs = []
            combined_trajs = []
            combined_labels = []
            context_key = "end_effector_pose"
            rollout_key = "debug/end_effector_pose"
            for idx in episode_idxs:
                pair = select_pair(pairs, idx)
                context_episode = _select_context_episode(pair)
                rollout_episode = pair["rollout"]
                context_length = int(context_episode["length"]) if "length" in context_episode else None
                rollout_length = int(rollout_episode["length"]) if "length" in rollout_episode else None
                context_obs, context_key = get_pose_obs(
                    context_episode["obs"], args.obs_key or "end_effector_pose"
                )
                rollout_obs, rollout_key = _get_debug_pose_obs(rollout_episode["obs"], args.obs_key)
                context_obs = trim_to_length(context_obs, context_length)
                rollout_obs = trim_to_length(rollout_obs, rollout_length)
                context_traj = context_obs[..., :3]
                rollout_traj = rollout_obs[..., :3]
                context_trajs.append(context_traj)
                rollout_trajs.append(rollout_traj)
                combined_trajs.extend([context_traj, rollout_traj])
                combined_labels.extend([f"context_{idx:04d}", f"rollout_{idx:04d}"])
            combined_out = out_dir / "pairs_context_rollout_multi_eef.png"
            _plot_traj3d_multi(
                combined_trajs, combined_labels, "Context + Rollout (multiple episodes)", combined_out
            )
            print(f"[INFO] Saved plot: {combined_out}")
    else:
        episodes = load_episodes(args.path)
        if len(episode_idxs) == 1:
            episode = select_episode(episodes, episode_idxs[0])
            length = int(episode["length"]) if "length" in episode else None
            obs = episode["obs"]
            debug_obs, debug_key = get_pose_obs(obs, args.obs_key)
            debug_obs = trim_to_length(debug_obs, length)
            if debug_obs.shape[-1] < 3:
                raise ValueError(f"debug_obs has last dim {debug_obs.shape[-1]}, expected at least 3.")
            eef_obs = debug_obs[..., :3]
            eef_out = out_dir / f"episode_{episode_idxs[0]:04d}_eef.png"
            plot_traj3d(eef_obs, f"End Effector ({debug_key}[:3])", eef_out)
            print(f"[INFO] Saved plot: {eef_out}")
            if args.plot_actions:
                actions = episode.get("actions")
                if isinstance(actions, torch.Tensor):
                    actions = trim_to_length(actions, length)
                    action_out = out_dir / f"episode_{episode_idxs[0]:04d}_actions.png"
                    plot_series(actions, "Actions", "action", action_out)
                    print(f"[INFO] Saved plot: {action_out}")
        else:
            eef_trajs = []
            labels = []
            debug_key = ""
            for idx in episode_idxs:
                episode = select_episode(episodes, idx)
                length = int(episode["length"]) if "length" in episode else None
                obs = episode["obs"]
                debug_obs, debug_key = get_pose_obs(obs, args.obs_key)
                debug_obs = trim_to_length(debug_obs, length)
                if debug_obs.shape[-1] < 3:
                    raise ValueError(f"debug_obs has last dim {debug_obs.shape[-1]}, expected at least 3.")
                eef_trajs.append(debug_obs[..., :3])
                labels.append(f"episode_{idx:04d}")
            eef_out = out_dir / "episodes_multi_eef.png"
            _plot_traj3d_multi(eef_trajs, labels, f"End Effector ({debug_key}[:3])", eef_out)
            print(f"[INFO] Saved plot: {eef_out}")


if __name__ == "__main__":
    main()
