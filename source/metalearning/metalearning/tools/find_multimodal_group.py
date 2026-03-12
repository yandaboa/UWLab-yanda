"""Find episodes with similar initial EE state and joint angles, then plot."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from .visualize_trajectory import plot_traj3d_multi, plot_traj3d_multi_plotly
from .visualization_utils import load_episodes, trim_to_length


def _resolve_episode_paths(paths: Sequence[str]) -> list[Path]:
    resolved: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            resolved.extend(sorted(path.glob("episodes_*.pt")))
        elif any(char in raw for char in ["*", "?", "["]):
            if path.is_absolute():
                resolved.extend(sorted(path.parent.glob(path.name)))
            else:
                resolved.extend(sorted(Path().glob(raw)))
        else:
            resolved.append(path)
    return resolved


def _extract_eef_pose(
    obs: Mapping[str, Any] | torch.Tensor, obs_key: str
) -> torch.Tensor:
    if not isinstance(obs, Mapping):
        raise TypeError("Expected obs mapping when extracting end effector pose.")
    if obs_key not in obs:
        raise KeyError(f"obs key '{obs_key}' not found.")
    value = obs[obs_key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"obs key '{obs_key}' is not a tensor.")
    return value


def _extract_joint_angles(
    episode: Mapping[str, Any],
    obs: Mapping[str, Any] | torch.Tensor,
    joint_key: str,
) -> torch.Tensor:
    if not isinstance(obs, Mapping):
        raise TypeError("Expected obs mapping when extracting joint angles.")
    if joint_key not in obs:
        raise KeyError(f"obs key '{joint_key}' not found.")
    value = obs[joint_key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"obs key '{joint_key}' is not a tensor.")
    return value


def _compute_episode_signature(
    episode: Mapping[str, Any],
    obs_key: str,
    joint_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    obs = episode["obs"]
    length = int(episode.get("length", 0)) or None
    eef_obs = _extract_eef_pose(obs, obs_key)
    eef_obs = trim_to_length(eef_obs, length)
    if eef_obs.shape[-1] < 3:
        raise ValueError(f"EEF obs last dim {eef_obs.shape[-1]} < 3.")
    eef0 = eef_obs[0, :3].detach().cpu().numpy()

    joint_obs = _extract_joint_angles(episode, obs, joint_key)
    joint_obs = trim_to_length(joint_obs, length)
    joint0 = joint_obs[0].detach().cpu().numpy()
    return eef0, joint0


def _cluster_episodes(
    episodes: list[tuple[Path, int, Mapping[str, Any]]],
    obs_key: str,
    joint_key: str,
    eef_tol: float,
    joint_tol: float,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for count, (path, idx, episode) in enumerate(episodes, start=1):
        if count == 1 or count % 500 == 0:
            print(f"[INFO] Processing episode {count}/{len(episodes)}")
        eef0, joint0 = _compute_episode_signature(episode, obs_key, joint_key)
        placed = False
        for group in groups:
            eef_ref = group["eef_ref"]
            joint_ref = group["joint_ref"]
            eef_err = np.linalg.norm(eef0 - eef_ref)
            joint_err = np.max(np.abs(joint0 - joint_ref))
            if eef_err <= eef_tol and joint_err <= joint_tol:
                group["items"].append((path, idx, episode))
                group["eef_points"].append(eef0)
                placed = True
                break
        if not placed:
            groups.append(
                {
                    "eef_ref": eef0,
                    "joint_ref": joint0,
                    "items": [(path, idx, episode)],
                    "eef_points": [eef0],
                }
            )
    return groups


def _find_multimodal_group(
    groups: list[dict[str, Any]],
) -> list[tuple[Path, int, Mapping[str, Any]]]:
    multimodal = [group for group in groups if len(group["items"]) > 1]
    if not multimodal:
        return []
    multimodal.sort(key=lambda group: len(group["items"]), reverse=True)
    return multimodal[0]["items"]


def _plot_group(
    group: list[tuple[Path, int, Mapping[str, Any]]],
    obs_key: str,
    out_path: Path,
    backend: str,
) -> None:
    trajectories: list[torch.Tensor] = []
    labels: list[str] = []
    for path, idx, episode in group:
        obs = episode["obs"]
        length = int(episode.get("length", 0)) or None
        eef_obs = _extract_eef_pose(obs, obs_key)
        eef_obs = trim_to_length(eef_obs, length)
        traj = eef_obs[..., :3]
        data = traj.detach().cpu().numpy()
        if data.ndim != 2 or data.shape[1] < 3:
            continue
        data = data[np.isfinite(data).all(axis=1)]
        if data.shape[0] < 2:
            continue
        trajectories.append(traj)
        labels.append(f"{path.name}:ep{idx:04d}")
    if len(trajectories) < 2:
        print("[INFO] Not enough valid trajectories to plot.")
        return
    title = "Episodes with matching initial EE + joint angles"
    if backend == "plotly":
        plot_traj3d_multi_plotly(
            trajectories,
            labels,
            title,
            out_path,
            show_legend=False,
            show_markers=True,
        )
    else:
        plot_traj3d_multi(
            trajectories,
            labels,
            title,
            out_path,
            show_legend=False,
            show_markers=True,
        )


def _plot_cluster_scatter(
    groups: list[dict[str, Any]],
    out_path: Path,
) -> None:
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(1, len(groups))))
    fig, ax = plt.subplots(figsize=(6, 5))
    total_count = sum(len(group["items"]) for group in groups) if groups else 0
    multi_count = sum(len(group["items"]) for group in groups if len(group["items"]) > 1)
    percent_multi = (100.0 * multi_count / total_count) if total_count else 0.0
    for idx, group in enumerate(groups):
        points = np.array(group["eef_points"], dtype=float)
        if points.ndim != 2 or points.shape[1] < 2:
            continue
        xy = points[:, :2]
        if len(group["items"]) == 1:
            color = (0.6, 0.6, 0.6, 0.7)
        else:
            color = colors[idx]
        ax.scatter(xy[:, 0], xy[:, 1], s=20, color=color)
    ax.set_xlabel("eef_x")
    ax.set_ylabel("eef_y")
    ax.set_title(f"Initial EE clusters (xy) | >=2: {percent_multi:.1f}%")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _groups_filename(eef_tol: float, joint_tol: float) -> str:
    eef_str = f"{eef_tol:.3f}".replace(".", "p")
    joint_str = f"{joint_tol:.3f}".replace(".", "p")
    return f"groups_{eef_str}_eepose_{joint_str}_joint_pose.pt"


def _serialize_groups(
    groups: list[dict[str, Any]],
    episode_paths: Sequence[Path],
    obs_key: str,
    joint_key: str,
    eef_tol: float,
    joint_tol: float,
) -> dict[str, Any]:
    packed = []
    for group in groups:
        items = [{"path": str(path), "idx": int(idx)} for path, idx, _ in group["items"]]
        packed.append(
            {
                "items": items,
                "eef_points": np.array(group["eef_points"], dtype=float),
            }
        )
    return {
        "episode_paths": [str(path) for path in episode_paths],
        "obs_key": obs_key,
        "joint_key": joint_key,
        "eef_tol": float(eef_tol),
        "joint_tol": float(joint_tol),
        "groups": packed,
    }


def _load_groups(path: Path) -> dict[str, Any]:
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict) or "groups" not in data:
        raise ValueError(f"Invalid groups file: {path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find episodes with similar initial EE state and joint angles."
    )
    parser.add_argument(
        "--episode-paths",
        type=str,
        nargs="*",
        default=["/gscratch/weirdlab/yanda/lti/UWLab-yanda/episodes/episodes_*.pt"],
        help="Episode .pt files, directories, or glob patterns.",
    )
    parser.add_argument(
        "--obs-key",
        type=str,
        default="end_effector_pose",
        help="Obs key for EE pose (default: end_effector_pose).",
    )
    parser.add_argument(
        "--joint-key",
        type=str,
        default="joint_pos",
        help="Obs key for joint positions (default: joint_pos).",
    )
    parser.add_argument("--eef-tol", type=float, default=0.02, help="EEF position tolerance.")
    parser.add_argument("--joint-tol", type=float, default=10.0, help="Joint angle tolerance.")
    parser.add_argument(
        "--backend",
        type=str,
        choices=("plotly", "matplotlib"),
        default="plotly",
        help="Plotting backend to use (default: plotly).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to save plots (default: ./visualizations).",
    )
    parser.add_argument(
        "--groups-file",
        type=Path,
        default=None,
        help="Path to a saved groups file to replot without recomputing.",
    )
    args = parser.parse_args()

    groups_data: dict[str, Any] | None = None
    if args.groups_file is not None:
        groups_data = _load_groups(args.groups_file)
        episode_paths = [Path(path) for path in groups_data["episode_paths"]]
        print(f"[INFO] Loaded groups from {args.groups_file}")
    else:
        episode_paths = _resolve_episode_paths(args.episode_paths)
        if not episode_paths:
            raise FileNotFoundError("No episode files found.")

    episodes: list[tuple[Path, int, Mapping[str, Any]]] = []
    for path in episode_paths:
        print(f"[INFO] Loading episodes from {path}")
        loaded = load_episodes(path)
        print(f"[INFO] Loaded {len(loaded)} episodes from {path}")
        for idx, episode in enumerate(loaded):
            episodes.append((path, idx, episode))

    print(f"[INFO] Total episodes: {len(episodes)}")
    if groups_data is None:
        groups = _cluster_episodes(
            episodes,
            obs_key=args.obs_key,
            joint_key=args.joint_key,
            eef_tol=args.eef_tol,
            joint_tol=args.joint_tol,
        )
    else:
        groups = []
        episode_lookup = {(path, idx): episode for path, idx, episode in episodes}
        for group in groups_data["groups"]:
            items = []
            for entry in group["items"]:
                path = Path(entry["path"])
                idx = int(entry["idx"])
                episode = episode_lookup.get((path, idx))
                if episode is None:
                    raise KeyError(f"Episode not found for {path} index {idx}")
                items.append((path, idx, episode))
            groups.append(
                {
                    "items": items,
                    "eef_points": np.array(group["eef_points"], dtype=float),
                }
            )
    group = _find_multimodal_group(groups)
    if not group:
        print("[INFO] No episodes matched the requested tolerances.")
        # Still generate the cluster scatter to show singleton groups.
        groups = groups or []

    out_dir = args.out_dir or episode_paths[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".html" if args.backend == "plotly" else ".png"
    out_path = out_dir / f"multimodal_group_eef{suffix}"
    if group:
        _plot_group(group, args.obs_key, out_path, args.backend)

    print(f"[INFO] Found group size {len(group)}.")
    for path, idx, _ in group:
        print(f"[INFO]  - {path} (episode {idx})")
    if group:
        print(f"[INFO] Saved plot: {out_path}")
    cluster_out = out_dir / "multimodal_cluster_scatter.png"
    _plot_cluster_scatter(groups, cluster_out)
    print(f"[INFO] Saved cluster scatter: {cluster_out}")

    if groups_data is None:
        groups_file = out_dir / _groups_filename(args.eef_tol, args.joint_tol)
        torch.save(
            _serialize_groups(
                groups,
                episode_paths,
                args.obs_key,
                args.joint_key,
                args.eef_tol,
                args.joint_tol,
            ),
            groups_file,
        )
        print(f"[INFO] Saved groups file: {groups_file}")


if __name__ == "__main__":
    main()
