"""Visualize end effector and action trajectories from episode files.

Example usage:
    python -m metalearning.tools.visualize_trajectory /path/to/episodes_000000.pt
    python -m metalearning.tools.visualize_trajectory /path/to/episodes_000000.pt --episode-idx 3
    python -m metalearning.tools.visualize_trajectory /path/to/episodes_000000.pt --out-dir /tmp/plots
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def _load_episodes(path: Path) -> list[dict[str, Any]]:
    """Load episodes saved by EpisodeStorage."""
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict) and "episodes" in data:
        episodes = data["episodes"]
    elif isinstance(data, list):
        episodes = data
    elif isinstance(data, dict) and "obs" in data:
        episodes = [data]
    else:
        raise ValueError(f"Unsupported file format in {path}.")
    if not episodes:
        raise ValueError(f"No episodes found in {path}.")
    return episodes


def _select_episode(episodes: list[dict[str, Any]], index: int) -> dict[str, Any]:
    """Select a single episode by index."""
    if index < 0 or index >= len(episodes):
        raise IndexError(f"Episode index {index} out of range (0..{len(episodes) - 1}).")
    return episodes[index]


def _get_debug_obs(
    obs: torch.Tensor | Mapping[str, Any], obs_key: Optional[str]
) -> Tuple[torch.Tensor, str]:
    """Extract debug observations from an episode."""
    if isinstance(obs, Mapping):
        if "debug" in obs:
            value = obs["debug"]
            if not isinstance(value, torch.Tensor):
                raise TypeError("obs key 'debug' is not a tensor.")
            return value, "debug"
        if obs_key is not None:
            if obs_key not in obs:
                raise KeyError(f"obs key '{obs_key}' not found.")
            value = obs[obs_key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"obs key '{obs_key}' is not a tensor.")
            return value, obs_key
        raise ValueError("No debug obs tensor found.")
    if not isinstance(obs, torch.Tensor):
        raise TypeError("obs is not a tensor.")
    return obs, "obs"


def _trim_to_length(tensor: torch.Tensor, length: Optional[int]) -> torch.Tensor:
    """Trim a tensor to an episode length if available."""
    if length is None:
        return tensor
    if tensor.shape[0] <= length:
        return tensor
    return tensor[:length]


def _plot_series(data: torch.Tensor, title: str, y_label: str, out_path: Optional[Path]) -> None:
    """Plot time series data for each dimension."""
    data_np = data.detach().cpu().numpy()
    if data_np.ndim == 1:
        data_np = data_np[:, None]
    num_dims = data_np.shape[1]
    fig, axes = plt.subplots(num_dims, 1, sharex=True, figsize=(8, 2.4 * num_dims))
    if num_dims == 1:
        axes = [axes]
    for dim in range(num_dims):
        axes[dim].plot(data_np[:, dim])
        axes[dim].set_ylabel(f"{y_label}[{dim}]")
    axes[-1].set_xlabel("t")
    fig.suptitle(title)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def _plot_traj3d(data: torch.Tensor, title: str, out_path: Optional[Path]) -> None:
    """Plot a 3D trajectory from (T, 3+) data."""
    data_np = data.detach().cpu().numpy()
    if data_np.ndim != 2 or data_np.shape[1] < 3:
        raise ValueError(f"Expected (T, >=3) data, got {data_np.shape}.")
    data_np = data_np[:, :3]
    mask = np.isfinite(data_np).all(axis=1)
    data_np = data_np[mask]
    if data_np.shape[0] < 2:
        raise ValueError("Not enough finite points to plot.")
    x, y, z = data_np.T
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z, linewidth=2.0)
    ax.scatter(x[0], y[0], z[0], s=60, marker="o")
    ax.scatter(x[-1], y[-1], z[-1], s=60, marker="^")
    mins = data_np.min(axis=0)
    maxs = data_np.max(axis=0)
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
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def main() -> None:
    """Run the visualization script."""
    parser = argparse.ArgumentParser(description="Visualize episode trajectories from .pt files.")
    parser.add_argument("path", type=Path, help="Path to a .pt episode file.")
    parser.add_argument("--episode-idx", type=int, default=0, help="Episode index to visualize.")
    parser.add_argument("--obs-key", type=str, default=None, help="Override key for debug obs.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to save plots (default: <episode_dir>/visualizations).",
    )
    parser.add_argument("--plot-actions", action="store_true", help="Plot actions.")
    args = parser.parse_args()

    episodes = _load_episodes(args.path)
    episode = _select_episode(episodes, args.episode_idx)

    length = int(episode["length"]) if "length" in episode else None
    obs = episode["obs"]
    debug_obs, debug_key = _get_debug_obs(obs, args.obs_key)
    debug_obs = _trim_to_length(debug_obs, length)
    if debug_obs.shape[-1] < 3:
        raise ValueError(f"debug_obs has last dim {debug_obs.shape[-1]}, expected at least 3.")
    eef_obs = debug_obs[..., :3]

    out_dir = args.out_dir or (args.path.parent / "visualizations")
    out_dir.mkdir(parents=True, exist_ok=True)
    eef_out = out_dir / f"episode_{args.episode_idx:04d}_eef.png"
    _plot_traj3d(eef_obs, f"End Effector ({debug_key}[:3])", eef_out)

    if args.plot_actions:
        actions = episode.get("actions")
        if isinstance(actions, torch.Tensor):
            actions = _trim_to_length(actions, length)
            action_out = out_dir / f"episode_{args.episode_idx:04d}_actions.png"
            _plot_series(actions, "Actions", "action", action_out)


if __name__ == "__main__":
    main()
