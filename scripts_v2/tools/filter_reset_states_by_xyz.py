#!/usr/bin/env python3
"""Filter reset-state .pt files by hard-coded xyz ranges for receptive and insertive objects."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Set ranges here (inclusive). Edit these constants directly.
# ---------------------------------------------------------------------------
RECEPTIVE_X_RANGE = (0.352, 0.4971)
RECEPTIVE_Y_RANGE = (-0.0156, 0.214)
RECEPTIVE_Z_RANGE = (0.0, 0.1)

# RECEPTIVE_X_RANGE = (0.20, 0.60)
# RECEPTIVE_Y_RANGE = (-0.30, 0.30)
# RECEPTIVE_Z_RANGE = (-0.10, 0.20)

INSERTIVE_X_RANGE = (0.3398, 0.5062)
INSERTIVE_Y_RANGE = (0.021, 0.3798)
INSERTIVE_Z_RANGE = (-0.10, 0.10)


def resolve_range(arg_value: list[float] | None, default_value: tuple[float, float]) -> tuple[float, float]:
    if arg_value is None:
        return default_value
    lo, hi = float(arg_value[0]), float(arg_value[1])
    if lo > hi:
        raise ValueError(f"Invalid range with lo > hi: ({lo}, {hi})")
    return (lo, hi)


def to_pose_tensor(value: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    """Convert root_pose leaf into tensor of shape [N, 7+]."""
    if isinstance(value, torch.Tensor):
        pose = value
    elif isinstance(value, list):
        if not value:
            raise ValueError("Encountered empty root_pose list.")
        pose = torch.stack(value, dim=0)
    else:
        raise TypeError(f"Unsupported root_pose type: {type(value)}")

    if pose.ndim != 2 or pose.shape[1] < 3:
        raise ValueError(f"Expected pose shape [N, >=3], got {tuple(pose.shape)}")
    return pose


def in_range(x: torch.Tensor, lo_hi: tuple[float, float]) -> torch.Tensor:
    lo, hi = lo_hi
    return (x >= lo) & (x <= hi)


def summarize_xyz(xyz: torch.Tensor, name: str) -> None:
    mean = xyz.mean(dim=0)
    std = xyz.std(dim=0, unbiased=False)
    mins = xyz.min(dim=0).values
    maxs = xyz.max(dim=0).values
    print(
        f"{name} xyz mean=({mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}) "
        f"std=({std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}) "
        f"min=({mins[0]:.4f}, {mins[1]:.4f}, {mins[2]:.4f}) "
        f"max=({maxs[0]:.4f}, {maxs[1]:.4f}, {maxs[2]:.4f})"
    )


def build_keep_mask(
    payload: dict[str, Any],
    *,
    receptive_x_range: tuple[float, float],
    receptive_y_range: tuple[float, float],
    receptive_z_range: tuple[float, float],
    insertive_x_range: tuple[float, float],
    insertive_y_range: tuple[float, float],
    insertive_z_range: tuple[float, float],
    filter_mode: str,
    invert_receptive: bool,
    invert_insertive: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        rigid = payload["initial_state"]["rigid_object"]
        receptive_pose = to_pose_tensor(rigid["receptive_object"]["root_pose"])
        insertive_pose = to_pose_tensor(rigid["insertive_object"]["root_pose"])
    except Exception as exc:
        raise KeyError(
            "Missing required keys under initial_state/rigid_object/{receptive_object,insertive_object}/root_pose"
        ) from exc

    if receptive_pose.shape[0] != insertive_pose.shape[0]:
        raise ValueError(
            f"Receptive and insertive sample counts differ: {receptive_pose.shape[0]} vs {insertive_pose.shape[0]}"
        )

    r_xyz = receptive_pose[:, :3]
    i_xyz = insertive_pose[:, :3]

    receptive_in = (
        in_range(r_xyz[:, 0], receptive_x_range)
        & in_range(r_xyz[:, 1], receptive_y_range)
        & in_range(r_xyz[:, 2], receptive_z_range)
    )
    insertive_in = (
        in_range(i_xyz[:, 0], insertive_x_range)
        & in_range(i_xyz[:, 1], insertive_y_range)
        & in_range(i_xyz[:, 2], insertive_z_range)
    )

    receptive_ok = ~receptive_in if invert_receptive else receptive_in
    insertive_ok = ~insertive_in if invert_insertive else insertive_in
    if filter_mode == "both":
        keep_mask = receptive_ok & insertive_ok
    elif filter_mode == "receptive":
        keep_mask = receptive_ok
    elif filter_mode == "insertive":
        keep_mask = insertive_ok
    else:
        raise ValueError(f"Unsupported filter_mode: {filter_mode}")

    return keep_mask, r_xyz, i_xyz


def filter_nested(obj: Any, keep_idx: torch.Tensor, expected_n: int) -> Any:
    """Recursively index every environment-correlated leaf with keep_idx."""
    if isinstance(obj, dict):
        return {k: filter_nested(v, keep_idx, expected_n) for k, v in obj.items()}

    if isinstance(obj, torch.Tensor):
        if obj.ndim > 0 and obj.shape[0] == expected_n:
            return obj[keep_idx]
        return obj

    if isinstance(obj, list):
        if len(obj) == expected_n:
            idx_list = keep_idx.tolist()
            return [obj[i] for i in idx_list]
        return obj

    return obj


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter reset-state .pt by hard-coded receptive/insertive xyz ranges.")
    parser.add_argument("input_pt", type=str, help="Input reset-state .pt file path.")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output .pt path. Default: auto path in a temp directory.",
    )
    parser.add_argument(
        "--filter-mode",
        type=str,
        default="both",
        choices=["both", "receptive", "insertive"],
        help="Which object ranges to enforce: both, receptive-only, or insertive-only.",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Flip both object filters: keep outside ranges for both receptive and insertive.",
    )
    parser.add_argument(
        "--invert-receptive",
        action="store_true",
        help="Flip only receptive filter: keep receptive states outside receptive ranges.",
    )
    parser.add_argument(
        "--invert-insertive",
        action="store_true",
        help="Flip only insertive filter: keep insertive states outside insertive ranges.",
    )
    parser.add_argument("--receptive-x", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--receptive-y", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--receptive-z", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--insertive-x", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--insertive-y", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--insertive-z", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
    args = parser.parse_args()

    if not os.path.isfile(args.input_pt):
        raise FileNotFoundError(f"Input file not found: {args.input_pt}")

    payload = torch.load(args.input_pt, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected top-level dict, got {type(payload)}")

    receptive_x_range = resolve_range(args.receptive_x, RECEPTIVE_X_RANGE)
    receptive_y_range = resolve_range(args.receptive_y, RECEPTIVE_Y_RANGE)
    receptive_z_range = resolve_range(args.receptive_z, RECEPTIVE_Z_RANGE)
    insertive_x_range = resolve_range(args.insertive_x, INSERTIVE_X_RANGE)
    insertive_y_range = resolve_range(args.insertive_y, INSERTIVE_Y_RANGE)
    insertive_z_range = resolve_range(args.insertive_z, INSERTIVE_Z_RANGE)

    invert_receptive = args.invert or args.invert_receptive
    invert_insertive = args.invert or args.invert_insertive

    keep_mask, r_xyz, i_xyz = build_keep_mask(
        payload,
        receptive_x_range=receptive_x_range,
        receptive_y_range=receptive_y_range,
        receptive_z_range=receptive_z_range,
        insertive_x_range=insertive_x_range,
        insertive_y_range=insertive_y_range,
        insertive_z_range=insertive_z_range,
        filter_mode=args.filter_mode,
        invert_receptive=invert_receptive,
        invert_insertive=invert_insertive,
    )
    n_total = int(keep_mask.numel())
    keep_idx = torch.where(keep_mask)[0]
    n_keep = int(keep_idx.numel())

    print("Using ranges (inclusive):")
    print(f"  filter_mode={args.filter_mode}")
    print(
        "  invert_receptive="
        f"{invert_receptive} ({'outside' if invert_receptive else 'inside'} receptive ranges kept)"
    )
    print(
        "  invert_insertive="
        f"{invert_insertive} ({'outside' if invert_insertive else 'inside'} insertive ranges kept)"
    )
    print(f"  receptive x={receptive_x_range}, y={receptive_y_range}, z={receptive_z_range}")
    print(f"  insertive x={insertive_x_range}, y={insertive_y_range}, z={insertive_z_range}")
    summarize_xyz(r_xyz, "Input receptive")
    summarize_xyz(i_xyz, "Input insertive")

    filtered = filter_nested(payload, keep_idx, n_total)

    if args.out is None:
        out_dir = os.path.join(tempfile.gettempdir(), "filtered_reset_states")
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(args.input_pt))[0]
        if invert_receptive and invert_insertive:
            mode = "outside_both"
        elif invert_receptive:
            mode = "outside_receptive"
        elif invert_insertive:
            mode = "outside_insertive"
        else:
            mode = "inside"
        args.out = os.path.join(out_dir, f"{stem}_filtered_{mode}_{n_keep}_of_{n_total}_{int(time.time())}.pt")
    else:
        out_dir = os.path.dirname(os.path.abspath(args.out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    torch.save(filtered, args.out)
    print(f"Saved filtered dataset to: {args.out}")
    print(f"Kept {n_keep}/{n_total} states ({(100.0 * n_keep / n_total) if n_total > 0 else 0.0:.2f}%)")
    if n_keep > 0:
        summarize_xyz(r_xyz[keep_idx], "Filtered receptive")
        summarize_xyz(i_xyz[keep_idx], "Filtered insertive")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
