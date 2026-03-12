#!/usr/bin/env python3
"""Quick 3D visualization for receptive/insertive pose distributions in reset-state .pt datasets."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use("Agg")


def quat_to_yaw(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternions (w, x, y, z) to yaw angles in radians."""
    w = quat_wxyz[:, 0]
    x = quat_wxyz[:, 1]
    y = quat_wxyz[:, 2]
    z = quat_wxyz[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def to_pose_array(root_pose_value: torch.Tensor | list[torch.Tensor]) -> np.ndarray:
    if isinstance(root_pose_value, list):
        if len(root_pose_value) == 0:
            raise ValueError("root_pose list is empty")
        root_pose = torch.stack(root_pose_value, dim=0).cpu().numpy()
    elif isinstance(root_pose_value, torch.Tensor):
        root_pose = root_pose_value.cpu().numpy()
    else:
        raise TypeError(f"Unsupported root_pose type: {type(root_pose_value)}")

    if root_pose.ndim != 2 or root_pose.shape[1] < 7:
        raise ValueError(f"Expected shape [N,7+] for root_pose, got {root_pose.shape}")
    return root_pose


def load_pose_pair(path: str) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected top-level dict in {path}, got {type(payload)}")

    try:
        rigid = payload["initial_state"]["rigid_object"]
        receptive_pose = to_pose_array(rigid["receptive_object"]["root_pose"])
        insertive_pose = to_pose_array(rigid["insertive_object"]["root_pose"])
    except Exception as exc:
        raise KeyError(
            "Could not find expected keys under initial_state/rigid_object/{receptive_object,insertive_object}/root_pose"
        ) from exc

    return receptive_pose, insertive_pose


def save_plot(
    xyz: np.ndarray,
    quat: np.ndarray,
    *,
    color_by: str,
    point_size: float,
    object_name: str,
    out_path: str,
) -> None:
    if color_by == "yaw":
        color = quat_to_yaw(quat)
        cbar_label = "yaw (rad)"
    else:
        color = xyz[:, 2]
        cbar_label = "z"

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=color, s=point_size, alpha=0.6, cmap="viridis")
    fig.colorbar(sc, ax=ax, label=cbar_label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"{object_name} pose distribution ({xyz.shape[0]} samples)")
    ax.set_box_aspect((1.0, 1.0, 1.0))
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_compare_plot(
    xyz_a_only: np.ndarray,
    xyz_overlap: np.ndarray,
    xyz_b_only: np.ndarray,
    *,
    object_name: str,
    label_a: str,
    label_b: str,
    out_path: str,
) -> None:
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    if xyz_a_only.shape[0] > 0:
        ax.scatter(
            xyz_a_only[:, 0],
            xyz_a_only[:, 1],
            xyz_a_only[:, 2],
            s=2.0,
            alpha=0.35,
            c="tab:blue",
            label=f"{label_a} only ({xyz_a_only.shape[0]})",
        )
    if xyz_overlap.shape[0] > 0:
        ax.scatter(
            xyz_overlap[:, 0],
            xyz_overlap[:, 1],
            xyz_overlap[:, 2],
            s=8.0,
            alpha=0.95,
            c="tab:green",
            label=f"overlap ({xyz_overlap.shape[0]})",
        )
    if xyz_b_only.shape[0] > 0:
        ax.scatter(
            xyz_b_only[:, 0],
            xyz_b_only[:, 1],
            xyz_b_only[:, 2],
            s=3.0,
            alpha=0.8,
            c="tab:orange",
            label=f"{label_b} only ({xyz_b_only.shape[0]})",
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"{object_name} pose coverage comparison (A-only / overlap / B-only)")
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def print_xyz_stats(name: str, poses: np.ndarray) -> None:
    xyz = poses[:, :3]
    mean = xyz.mean(axis=0)
    std = xyz.std(axis=0)
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    print(
        f"{name} xyz mean=({mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}) "
        f"std=({std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}) "
        f"min=({mins[0]:.4f}, {mins[1]:.4f}, {mins[2]:.4f}) "
        f"max=({maxs[0]:.4f}, {maxs[1]:.4f}, {maxs[2]:.4f})"
    )


def split_overlap(
    xyz_a: np.ndarray, xyz_b: np.ndarray, *, round_decimals: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    a_q = np.round(xyz_a, decimals=round_decimals)
    b_q = np.round(xyz_b, decimals=round_decimals)

    b_set = {tuple(row) for row in b_q}
    a_in_b = np.array([tuple(row) in b_set for row in a_q], dtype=bool)

    a_set = {tuple(row) for row in a_q}
    b_in_a = np.array([tuple(row) in a_set for row in b_q], dtype=bool)

    xyz_a_only = xyz_a[~a_in_b]
    xyz_overlap = xyz_a[a_in_b]
    xyz_b_only = xyz_b[~b_in_a]
    return xyz_a_only, xyz_overlap, xyz_b_only, int((~a_in_b).sum()), int(a_in_b.sum()), int((~b_in_a).sum())


def downsample_xyz(xyz: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or xyz.shape[0] <= max_points:
        return xyz
    idx = np.random.choice(xyz.shape[0], size=max_points, replace=False)
    return xyz[idx]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot receptive + insertive pose distributions (single dataset) or coverage comparison (two datasets)."
    )
    parser.add_argument("pt_files", type=str, nargs="+", help="One or two dataset .pt files.")
    parser.add_argument("--max-points", type=int, default=20000, help="Max points to plot.")
    parser.add_argument("--point-size", type=float, default=2.0, help="Scatter marker size.")
    parser.add_argument(
        "--color-by",
        type=str,
        default="yaw",
        choices=["yaw", "z"],
        help="Color points by yaw (from quat) or z position.",
    )
    parser.add_argument("--out-dir", type=str, default=None, help="Directory to save output PNG files.")
    parser.add_argument(
        "--compare-round-decimals",
        type=int,
        default=4,
        help="Rounding decimals for overlap matching in comparison mode.",
    )
    args = parser.parse_args()

    if len(args.pt_files) not in (1, 2):
        raise ValueError("Provide one dataset path (single plot mode) or two dataset paths (comparison mode).")

    if args.out_dir is None:
        out_dir = os.path.join(tempfile.gettempdir(), "object_pose_plots")
    else:
        out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    ts = int(time.time())

    if len(args.pt_files) == 1:
        pt_file = args.pt_files[0]
        receptive_poses, insertive_poses = load_pose_pair(pt_file)
        print_xyz_stats("Receptive (full data)", receptive_poses)
        print_xyz_stats("Insertive (full data)", insertive_poses)
        n = min(receptive_poses.shape[0], insertive_poses.shape[0])

        if args.max_points > 0 and n > args.max_points:
            idx = np.random.choice(n, size=args.max_points, replace=False)
        else:
            idx = np.arange(n)

        receptive_poses = receptive_poses[idx]
        insertive_poses = insertive_poses[idx]
        print_xyz_stats("Receptive (plotted sample)", receptive_poses)
        print_xyz_stats("Insertive (plotted sample)", insertive_poses)

        stem = os.path.splitext(os.path.basename(pt_file))[0]
        receptive_out = os.path.join(out_dir, f"{stem}_receptive_pose_dist_{args.color_by}_{ts}.png")
        insertive_out = os.path.join(out_dir, f"{stem}_insertive_pose_dist_{args.color_by}_{ts}.png")

        save_plot(
            receptive_poses[:, :3],
            receptive_poses[:, 3:7],
            color_by=args.color_by,
            point_size=args.point_size,
            object_name="Receptive object",
            out_path=receptive_out,
        )
        save_plot(
            insertive_poses[:, :3],
            insertive_poses[:, 3:7],
            color_by=args.color_by,
            point_size=args.point_size,
            object_name="Insertive object",
            out_path=insertive_out,
        )

        print(f"Saved receptive plot to: {receptive_out}")
        print(f"Saved insertive plot to: {insertive_out}")
    else:
        a_path, b_path = args.pt_files
        a_receptive, a_insertive = load_pose_pair(a_path)
        b_receptive, b_insertive = load_pose_pair(b_path)
        print_xyz_stats("Dataset A receptive (full data)", a_receptive)
        print_xyz_stats("Dataset A insertive (full data)", a_insertive)
        print_xyz_stats("Dataset B receptive (full data)", b_receptive)
        print_xyz_stats("Dataset B insertive (full data)", b_insertive)

        n_a = min(a_receptive.shape[0], a_insertive.shape[0])
        n_b = min(b_receptive.shape[0], b_insertive.shape[0])
        a_receptive = a_receptive[:n_a]
        a_insertive = a_insertive[:n_a]
        b_receptive = b_receptive[:n_b]
        b_insertive = b_insertive[:n_b]

        a_label = os.path.basename(a_path)
        b_label = os.path.basename(b_path)
        compare_stem = f"{os.path.splitext(a_label)[0]}_vs_{os.path.splitext(b_label)[0]}"
        receptive_out = os.path.join(out_dir, f"{compare_stem}_receptive_coverage_{ts}.png")
        insertive_out = os.path.join(out_dir, f"{compare_stem}_insertive_coverage_{ts}.png")

        r_a_only, r_overlap, r_b_only, r_n_a_only, r_n_overlap, r_n_b_only = split_overlap(
            a_receptive[:, :3], b_receptive[:, :3], round_decimals=args.compare_round_decimals
        )
        i_a_only, i_overlap, i_b_only, i_n_a_only, i_n_overlap, i_n_b_only = split_overlap(
            a_insertive[:, :3], b_insertive[:, :3], round_decimals=args.compare_round_decimals
        )

        r_a_only_plot = downsample_xyz(r_a_only, args.max_points)
        r_overlap_plot = downsample_xyz(r_overlap, args.max_points)
        r_b_only_plot = downsample_xyz(r_b_only, args.max_points)
        i_a_only_plot = downsample_xyz(i_a_only, args.max_points)
        i_overlap_plot = downsample_xyz(i_overlap, args.max_points)
        i_b_only_plot = downsample_xyz(i_b_only, args.max_points)

        save_compare_plot(
            r_a_only_plot,
            r_overlap_plot,
            r_b_only_plot,
            object_name="Receptive object",
            label_a=a_label,
            label_b=b_label,
            out_path=receptive_out,
        )
        save_compare_plot(
            i_a_only_plot,
            i_overlap_plot,
            i_b_only_plot,
            object_name="Insertive object",
            label_a=a_label,
            label_b=b_label,
            out_path=insertive_out,
        )

        print(
            "Comparison mode: plotting A-only (blue), overlap (green), B-only (orange); "
            "ignoring --color-by yaw."
        )
        print(
            f"Receptive split counts: A-only={r_n_a_only}, overlap={r_n_overlap}, B-only={r_n_b_only} "
            f"(round_decimals={args.compare_round_decimals})"
        )
        print(
            f"Insertive split counts: A-only={i_n_a_only}, overlap={i_n_overlap}, B-only={i_n_b_only} "
            f"(round_decimals={args.compare_round_decimals})"
        )
        print(f"Saved receptive comparison plot to: {receptive_out}")
        print(f"Saved insertive comparison plot to: {insertive_out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
