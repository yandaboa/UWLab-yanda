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
    show_legend: bool = True,
    show_markers: bool = True,
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
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(traj_np)))
    for idx, data in enumerate(traj_np):
        ax.plot(data[:, 0], data[:, 1], data[:, 2], linewidth=1.6, color=colors[idx], label=labels[idx])
        if show_markers:
            ax.scatter(
                data[0, 0],
                data[0, 1],
                data[0, 2],
                s=35,
                marker="o",
                color=colors[idx],
                alpha=0.6,
            )
            ax.scatter(
                data[-1, 0],
                data[-1, 1],
                data[-1, 2],
                s=35,
                marker="^",
                color=colors[idx],
                alpha=0.9,
            )
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
    if show_legend:
        ax.legend()
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_traj3d_multi(
    trajectories: Sequence[torch.Tensor],
    labels: Sequence[str],
    title: str,
    out_path: Optional[Path],
    show_legend: bool = True,
    show_markers: bool = True,
) -> None:
    """Public wrapper for plotting multiple 3D trajectories (matplotlib)."""
    _plot_traj3d_multi(
        trajectories,
        labels,
        title,
        out_path,
        show_legend=show_legend,
        show_markers=show_markers,
    )


def _to_xyz_numpy(traj: torch.Tensor) -> np.ndarray:
    data = traj.detach().cpu().numpy()
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"Expected (T, >=3) data, got {data.shape}.")
    data = data[:, :3]
    data = data[np.isfinite(data).all(axis=1)]
    if data.shape[0] < 2:
        raise ValueError("Not enough finite points to plot.")
    return data


def _get_plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "Plotly backend requested but 'plotly' is not installed. "
            "Install it (e.g., `pip install plotly`) or use `--backend matplotlib`."
        ) from exc
    return go


def _plot_traj3d_plotly(
    trajectory: torch.Tensor,
    title: str,
    out_path: Optional[Path],
    label: str = "trajectory",
) -> None:
    go = _get_plotly()
    data = _to_xyz_numpy(trajectory)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=data[:, 0],
            y=data[:, 1],
            z=data[:, 2],
            mode="lines",
            name=label,
            line={"width": 4},
        )
    )
    fig.update_layout(
        title=title,
        scene={
            "xaxis_title": "x",
            "yaxis_title": "y",
            "zaxis_title": "z",
            "aspectmode": "data",
        },
        template="plotly_white",
    )
    if out_path is not None:
        fig.write_html(str(out_path), include_plotlyjs=True, full_html=True)
    else:
        fig.show()


def _plot_traj3d_pair_plotly(
    context_traj: torch.Tensor,
    rollout_traj: torch.Tensor,
    title: str,
    out_path: Optional[Path],
) -> None:
    go = _get_plotly()
    context_data = _to_xyz_numpy(context_traj)
    rollout_data = _to_xyz_numpy(rollout_traj)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=context_data[:, 0],
            y=context_data[:, 1],
            z=context_data[:, 2],
            mode="lines",
            name="context",
            line={"width": 4},
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=rollout_data[:, 0],
            y=rollout_data[:, 1],
            z=rollout_data[:, 2],
            mode="lines",
            name="rollout",
            line={"width": 4},
        )
    )
    fig.update_layout(
        title=title,
        scene={
            "xaxis_title": "x",
            "yaxis_title": "y",
            "zaxis_title": "z",
            "aspectmode": "data",
        },
        template="plotly_white",
    )
    if out_path is not None:
        fig.write_html(str(out_path), include_plotlyjs=True, full_html=True)
    else:
        fig.show()


def _plot_traj3d_multi_plotly(
    trajectories: Sequence[torch.Tensor],
    labels: Sequence[str],
    title: str,
    out_path: Optional[Path],
    show_legend: bool = True,
    show_markers: bool = True,
) -> None:
    go = _get_plotly()
    fig = go.Figure()
    for traj, label in zip(trajectories, labels):
        data = _to_xyz_numpy(traj)
        fig.add_trace(
            go.Scatter3d(
                x=data[:, 0],
                y=data[:, 1],
                z=data[:, 2],
                mode="lines",
                name=label,
                line={"width": 3},
                showlegend=show_legend,
            )
        )
        if show_markers:
            fig.add_trace(
                go.Scatter3d(
                    x=[data[0, 0]],
                    y=[data[0, 1]],
                    z=[data[0, 2]],
                    mode="markers",
                    name=f"{label}_start",
                    marker={"size": 4, "symbol": "circle"},
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[data[-1, 0]],
                    y=[data[-1, 1]],
                    z=[data[-1, 2]],
                    mode="markers",
                    name=f"{label}_end",
                    marker={"size": 4, "symbol": "diamond"},
                    showlegend=False,
                )
            )
    fig.update_layout(
        title=title,
        scene={
            "xaxis_title": "x",
            "yaxis_title": "y",
            "zaxis_title": "z",
            "aspectmode": "data",
        },
        template="plotly_white",
    )
    if out_path is not None:
        fig.write_html(str(out_path), include_plotlyjs=True, full_html=True)
    else:
        fig.show()


def plot_traj3d_multi_plotly(
    trajectories: Sequence[torch.Tensor],
    labels: Sequence[str],
    title: str,
    out_path: Optional[Path],
    show_legend: bool = True,
    show_markers: bool = True,
) -> None:
    """Public wrapper for plotting multiple 3D trajectories (plotly)."""
    _plot_traj3d_multi_plotly(
        trajectories,
        labels,
        title,
        out_path,
        show_legend=show_legend,
        show_markers=show_markers,
    )


def _plot_series_plotly(
    series: torch.Tensor,
    title: str,
    ylabel: str,
    out_path: Optional[Path],
) -> None:
    go = _get_plotly()
    data = series.detach().cpu().numpy()
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2:
        raise ValueError(f"Expected 1D or 2D actions, got shape {data.shape}.")
    fig = go.Figure()
    steps = np.arange(data.shape[0])
    for idx in range(data.shape[1]):
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=data[:, idx],
                mode="lines",
                name=f"{ylabel}_{idx}",
                line={"width": 2},
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="step",
        yaxis_title=ylabel,
        template="plotly_white",
    )
    if out_path is not None:
        fig.write_html(str(out_path), include_plotlyjs=True, full_html=True)
    else:
        fig.show()


def _plot_suffix(backend: str) -> str:
    return ".html" if backend == "plotly" else ".png"


def _episode_success_flag(episode: Mapping[str, Any]) -> bool:
    success = episode.get("success")
    if isinstance(success, torch.Tensor):
        if success.numel() == 0:
            return False
        return bool(success.reshape(-1)[0].item())
    if isinstance(success, (bool, int, float)):
        return bool(success)
    return False


def _coerce_rewards(rewards: torch.Tensor) -> torch.Tensor:
    if rewards.ndim == 1:
        return rewards
    if rewards.ndim == 2:
        if rewards.shape[1] == 1:
            return rewards[:, 0]
        return rewards.sum(dim=1)
    raise ValueError(f"Expected 1D or 2D rewards, got shape {rewards.shape}.")


def _plot_rewards_matplotlib(
    reward_series: Sequence[torch.Tensor],
    labels: Sequence[str],
    success_flags: Sequence[bool],
    title: str,
    out_path: Optional[Path],
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for rewards, label, success in zip(reward_series, labels, success_flags):
        color = "tab:blue" if success else "tab:red"
        rewards_np = rewards.detach().cpu().numpy()
        ax.plot(np.arange(rewards_np.shape[0]), rewards_np, color=color, linewidth=1.6, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("reward")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def _plot_rewards_plotly(
    reward_series: Sequence[torch.Tensor],
    labels: Sequence[str],
    success_flags: Sequence[bool],
    title: str,
    out_path: Optional[Path],
) -> None:
    go = _get_plotly()
    fig = go.Figure()
    for rewards, label, success in zip(reward_series, labels, success_flags):
        rewards_np = rewards.detach().cpu().numpy()
        color = "blue" if success else "red"
        fig.add_trace(
            go.Scatter(
                x=np.arange(rewards_np.shape[0]),
                y=rewards_np,
                mode="lines",
                name=label,
                line={"width": 2, "color": color},
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="step",
        yaxis_title="reward",
        template="plotly_white",
    )
    if out_path is not None:
        fig.write_html(str(out_path), include_plotlyjs=True, full_html=True)
    else:
        fig.show()


def plot_episode_rewards(
    episodes: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    out_path: Optional[Path],
    backend: str = "plotly",
) -> None:
    reward_series = []
    success_flags = []
    for episode in episodes:
        rewards = episode.get("rewards")
        if not isinstance(rewards, torch.Tensor):
            raise TypeError("Episode rewards are missing or not a tensor.")
        length = int(episode["length"]) if "length" in episode else None
        rewards = trim_to_length(rewards, length)
        reward_series.append(_coerce_rewards(rewards))
        success_flags.append(_episode_success_flag(episode))
    if backend == "plotly":
        _plot_rewards_plotly(reward_series, labels, success_flags, "Episode Rewards", out_path)
    else:
        _plot_rewards_matplotlib(reward_series, labels, success_flags, "Episode Rewards", out_path)


def visualize_context_rollout_3d(
    context_episode: Mapping[str, Any],
    rollout_episode: Mapping[str, Any],
    obs_key: Optional[str] = None,
    out_path: Optional[Path] = None,
    backend: str = "plotly",
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
    if backend == "plotly":
        _plot_traj3d_pair_plotly(context_eef, rollout_eef, "Context vs Rollout", out_path)
    else:
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
    parser.add_argument(
        "--backend",
        type=str,
        choices=("plotly", "matplotlib"),
        default="matplotlib",
        help="Plotting backend to use (default: plotly).",
    )
    parser.add_argument("--plot-actions", action="store_true", help="Plot actions.")
    parser.add_argument("--plot-rewards", action="store_true", help="Plot per-step rewards.")
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
            eef_out = out_dir / f"pair_{episode_idxs[0]:04d}_eef{_plot_suffix(args.backend)}"
            context_key, rollout_key = visualize_context_rollout_3d(
                context_episode,
                rollout_episode,
                obs_key=args.obs_key,
                out_path=eef_out,
                backend=args.backend,
            )
            print(f"[INFO] Saved plot: {eef_out}")
            if args.plot_actions and isinstance(rollout_episode.get("actions"), torch.Tensor):
                actions = rollout_episode["actions"]
                rollout_length = int(rollout_episode["length"]) if "length" in rollout_episode else None
                actions = trim_to_length(actions, rollout_length)
                action_out = out_dir / f"pair_{episode_idxs[0]:04d}_actions{_plot_suffix(args.backend)}"
                if args.backend == "plotly":
                    _plot_series_plotly(actions, f"Rollout Actions ({rollout_key})", "action", action_out)
                else:
                    plot_series(actions, f"Rollout Actions ({rollout_key})", "action", action_out)
                print(f"[INFO] Saved plot: {action_out}")
            if args.plot_rewards:
                reward_out = out_dir / f"pair_{episode_idxs[0]:04d}_rewards{_plot_suffix(args.backend)}"
                plot_episode_rewards(
                    [rollout_episode],
                    [f"rollout_{episode_idxs[0]:04d}"],
                    reward_out,
                    backend=args.backend,
                )
                print(f"[INFO] Saved plot: {reward_out}")
        else:
            context_trajs = []
            rollout_trajs = []
            combined_trajs = []
            combined_labels = []
            rollout_episodes = []
            rollout_labels = []
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
                rollout_episodes.append(rollout_episode)
                rollout_labels.append(f"rollout_{idx:04d}")
            combined_out = out_dir / f"pairs_context_rollout_multi_eef{_plot_suffix(args.backend)}"
            if args.backend == "plotly":
                _plot_traj3d_multi_plotly(
                    combined_trajs, combined_labels, "Context + Rollout (multiple episodes)", combined_out
                )
            else:
                _plot_traj3d_multi(
                    combined_trajs, combined_labels, "Context + Rollout (multiple episodes)", combined_out
                )
            print(f"[INFO] Saved plot: {combined_out}")
            if args.plot_rewards and rollout_episodes:
                reward_out = out_dir / f"pairs_rollout_rewards{_plot_suffix(args.backend)}"
                plot_episode_rewards(rollout_episodes, rollout_labels, reward_out, backend=args.backend)
                print(f"[INFO] Saved plot: {reward_out}")
    else:
        episodes = load_episodes(args.path)
        if len(episode_idxs) == 1:
            episode = select_episode(episodes, episode_idxs[0])
            length = int(episode["length"]) if "length" in episode else None
            obs = episode["obs"]
            obs_key = args.obs_key or "end_effector_pose"
            debug_obs, debug_key = get_pose_obs(obs, obs_key)
            debug_obs = trim_to_length(debug_obs, length)
            if debug_obs.shape[-1] < 3:
                raise ValueError(f"debug_obs has last dim {debug_obs.shape[-1]}, expected at least 3.")
            eef_obs = debug_obs[..., :3]
            eef_out = out_dir / f"episode_{episode_idxs[0]:04d}_eef{_plot_suffix(args.backend)}"
            if args.backend == "plotly":
                _plot_traj3d_plotly(eef_obs, f"End Effector ({debug_key}[:3])", eef_out)
            else:
                plot_traj3d(eef_obs, f"End Effector ({debug_key}[:3])", eef_out)
            print(f"[INFO] Saved plot: {eef_out}")
            if args.plot_actions:
                actions = episode.get("actions")
                if isinstance(actions, torch.Tensor):
                    actions = trim_to_length(actions, length)
                    action_out = out_dir / f"episode_{episode_idxs[0]:04d}_actions{_plot_suffix(args.backend)}"
                    if args.backend == "plotly":
                        _plot_series_plotly(actions, "Actions", "action", action_out)
                    else:
                        plot_series(actions, "Actions", "action", action_out)
                    print(f"[INFO] Saved plot: {action_out}")
            if args.plot_rewards:
                reward_out = out_dir / f"episode_{episode_idxs[0]:04d}_rewards{_plot_suffix(args.backend)}"
                plot_episode_rewards(
                    [episode],
                    [f"episode_{episode_idxs[0]:04d}"],
                    reward_out,
                    backend=args.backend,
                )
                print(f"[INFO] Saved plot: {reward_out}")
        else:
            eef_trajs = []
            labels = []
            debug_key = ""
            reward_episodes = []
            reward_labels = []
            for idx in episode_idxs:
                episode = select_episode(episodes, idx)
                length = int(episode["length"]) if "length" in episode else None
                obs = episode["obs"]
                obs_key = args.obs_key or "end_effector_pose"
                debug_obs, debug_key = get_pose_obs(obs, obs_key)
                debug_obs = trim_to_length(debug_obs, length)
                if debug_obs.shape[-1] < 3:
                    raise ValueError(f"debug_obs has last dim {debug_obs.shape[-1]}, expected at least 3.")
                eef_trajs.append(debug_obs[..., :3])
                labels.append(f"episode_{idx:04d}")
                reward_episodes.append(episode)
                reward_labels.append(f"episode_{idx:04d}")
            eef_out = out_dir / f"episodes_multi_eef{_plot_suffix(args.backend)}"
            if args.backend == "plotly":
                _plot_traj3d_multi_plotly(eef_trajs, labels, f"End Effector ({debug_key}[:3])", eef_out)
            else:
                _plot_traj3d_multi(eef_trajs, labels, f"End Effector ({debug_key}[:3])", eef_out)
            print(f"[INFO] Saved plot: {eef_out}")
            if args.plot_rewards and reward_episodes:
                reward_out = out_dir / f"episodes_multi_rewards{_plot_suffix(args.backend)}"
                plot_episode_rewards(reward_episodes, reward_labels, reward_out, backend=args.backend)
                print(f"[INFO] Saved plot: {reward_out}")


if __name__ == "__main__":
    main()
