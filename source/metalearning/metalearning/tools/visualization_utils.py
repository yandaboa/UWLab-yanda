"""Shared utilities for episode visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch


def load_episodes(path: Path) -> list[dict[str, Any]]:
    """Load episodes saved by EpisodeStorage."""
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict) and "pairs" in data:
        raise ValueError("Found paired rollouts; use _load_pairs instead.")
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


def select_episode(episodes: list[dict[str, Any]], index: int) -> dict[str, Any]:
    """Select a single episode by index."""
    if index < 0 or index >= len(episodes):
        raise IndexError(f"Episode index {index} out of range (0..{len(episodes) - 1}).")
    return episodes[index]


def load_pairs(path: Path) -> list[dict[str, Any]]:
    """Load paired rollouts saved by RolloutPairStorage."""
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict) and "pairs" in data:
        pairs = data["pairs"]
    elif isinstance(data, list):
        pairs = data
    else:
        raise ValueError(f"Unsupported paired file format in {path}.")
    if not pairs:
        raise ValueError(f"No paired rollouts found in {path}.")
    return pairs


def select_pair(pairs: list[dict[str, Any]], index: int) -> dict[str, Any]:
    """Select a single rollout pair by index."""
    if index < 0 or index >= len(pairs):
        raise IndexError(f"Pair index {index} out of range (0..{len(pairs) - 1}).")
    return pairs[index]


def get_pose_obs(
    obs: torch.Tensor | Mapping[str, Any], obs_key: Optional[str]
) -> Tuple[torch.Tensor, str]:
    """Extract pose observations from an episode."""
    if isinstance(obs, Mapping):
        if obs_key is not None:
            if obs_key not in obs:
                raise KeyError(f"obs key '{obs_key}' not found.")
            value = obs[obs_key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"obs key '{obs_key}' is not a tensor.")
            return value, obs_key
        if "debug/end_effector_pose" in obs:
            value = obs["debug/end_effector_pose"]
            if not isinstance(value, torch.Tensor):
                raise TypeError("obs key 'debug/end_effector_pose' is not a tensor.")
            return value, "debug/end_effector_pose"
        if "end_effector_pose" in obs:
            value = obs["end_effector_pose"]
            if not isinstance(value, torch.Tensor):
                raise TypeError("obs key 'end_effector_pose' is not a tensor.")
            return value, "end_effector_pose"
        if "debug" in obs:
            value = obs["debug"]
            if not isinstance(value, torch.Tensor):
                raise TypeError("obs key 'debug' is not a tensor.")
            return value, "debug"
        debug_keys = [key for key in obs.keys() if key.startswith("debug/")]
        if debug_keys:
            key = debug_keys[0]
            value = obs[key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"obs key '{key}' is not a tensor.")
            return value, key
        raise ValueError("No pose obs tensor found.")
    if not isinstance(obs, torch.Tensor):
        raise TypeError("obs is not a tensor.")
    return obs, "obs"


def trim_to_length(tensor: torch.Tensor, length: Optional[int]) -> torch.Tensor:
    """Trim a tensor to an episode length if available."""
    if length is None:
        return tensor
    if tensor.shape[0] <= length:
        return tensor
    return tensor[:length]


def _blend_color(color: Any, target: Tuple[float, float, float], amount: float) -> Tuple[float, float, float]:
    rgb = np.array(mcolors.to_rgb(color))
    tgt = np.array(target)
    blended = rgb * (1.0 - amount) + tgt * amount
    return tuple(blended.tolist())


def plot_series(data: torch.Tensor, title: str, y_label: str, out_path: Optional[Path]) -> None:
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


def plot_traj3d(data: torch.Tensor, title: str, out_path: Optional[Path]) -> None:
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
    line = ax.plot(x, y, z, linewidth=2.0)[0]
    line_color = line.get_color()
    start_color = _blend_color(line_color, (1.0, 1.0, 1.0), 0.35)
    end_color = _blend_color(line_color, (0.0, 0.0, 0.0), 0.25)
    ax.scatter(x[0], y[0], z[0], s=60, marker="o", color=start_color)
    ax.scatter(x[-1], y[-1], z[-1], s=60, marker="^", color=end_color)
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


def plot_traj3d_pair(
    demo: torch.Tensor,
    rollout: torch.Tensor,
    title: str,
    out_path: Optional[Path],
    colors: tuple[str, str] = ("tab:blue", "tab:orange"),
) -> None:
    """Plot paired 3D trajectories from (T, 3+) data."""
    demo_np = demo.detach().cpu().numpy()
    rollout_np = rollout.detach().cpu().numpy()
    if demo_np.ndim != 2 or demo_np.shape[1] < 3:
        raise ValueError(f"Expected demo (T, >=3) data, got {demo_np.shape}.")
    if rollout_np.ndim != 2 or rollout_np.shape[1] < 3:
        raise ValueError(f"Expected rollout (T, >=3) data, got {rollout_np.shape}.")
    demo_np = demo_np[:, :3]
    rollout_np = rollout_np[:, :3]
    demo_np = demo_np[np.isfinite(demo_np).all(axis=1)]
    rollout_np = rollout_np[np.isfinite(rollout_np).all(axis=1)]
    if demo_np.shape[0] < 2 or rollout_np.shape[0] < 2:
        raise ValueError("Not enough finite points to plot.")
    demo_x, demo_y, demo_z = demo_np.T
    roll_x, roll_y, roll_z = rollout_np.T
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(demo_x, demo_y, demo_z, linewidth=2.0, color=colors[0], label="demo")
    ax.plot(roll_x, roll_y, roll_z, linewidth=2.0, color=colors[1], label="rollout")
    demo_start = _blend_color(colors[0], (1.0, 1.0, 1.0), 0.35)
    demo_end = _blend_color(colors[0], (0.0, 0.0, 0.0), 0.25)
    roll_start = _blend_color(colors[1], (1.0, 1.0, 1.0), 0.35)
    roll_end = _blend_color(colors[1], (0.0, 0.0, 0.0), 0.25)
    ax.scatter(demo_x[0], demo_y[0], demo_z[0], s=50, marker="o", color=demo_start)
    ax.scatter(demo_x[-1], demo_y[-1], demo_z[-1], s=50, marker="^", color=demo_end)
    ax.scatter(roll_x[0], roll_y[0], roll_z[0], s=50, marker="o", color=roll_start)
    ax.scatter(roll_x[-1], roll_y[-1], roll_z[-1], s=50, marker="^", color=roll_end)
    mins = np.minimum(demo_np.min(axis=0), rollout_np.min(axis=0))
    maxs = np.maximum(demo_np.max(axis=0), rollout_np.max(axis=0))
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
