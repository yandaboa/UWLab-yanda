#!/usr/bin/env python3
# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Any, Iterable

import torch

RESET_KEYS = ("states", "physics", "raw_states")
SEQUENCE_KEYS = ("obs", "actions", "rewards", "dones")


def _expand_inputs(inputs: Iterable[str]) -> list[Path]:
    expanded: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.exists():
            if path.is_dir():
                expanded.extend(sorted(path.glob("*.pt")))
            else:
                expanded.append(path)
            continue
        matches = [Path(match) for match in glob.glob(item)]
        expanded.extend(sorted(matches))
    seen = set()
    unique: list[Path] = []
    for path in expanded:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _normalize_episodes(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict) and "episodes" in data:
        return list(data.get("episodes") or [])
    if isinstance(data, list):
        return list(data)
    if isinstance(data, dict):
        return [data]
    raise TypeError(f"Unsupported episode data type: {type(data)}")


def _strip_episode(episode: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(episode)
    for key in RESET_KEYS:
        cleaned.pop(key, None)
    return cleaned


def _output_path(input_path: Path, output_dir: Path | None, suffix: str) -> Path:
    if output_dir is not None:
        return output_dir / input_path.name
    return input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")


def _extract_length(episode: dict[str, Any]) -> int:
    length = episode.get("length")
    if isinstance(length, torch.Tensor):
        return int(length.item())
    if isinstance(length, int):
        return length
    actions = episode.get("actions")
    if isinstance(actions, torch.Tensor):
        return int(actions.shape[0])
    return 0


def _collect_tensor_shapes(value: Any, prefix: str, out: list[tuple[str, tuple[int, ...]]]) -> None:
    if isinstance(value, torch.Tensor):
        out.append((prefix, tuple(value.shape)))
        return
    if isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _collect_tensor_shapes(item, next_prefix, out)
        return
    if isinstance(value, list):
        list_prefix = f"{prefix}[]" if prefix else "[]"
        for item in value:
            _collect_tensor_shapes(item, list_prefix, out)


def _check_sequence_lengths(episode: dict[str, Any], length: int) -> list[str]:
    issues: list[str] = []
    for key in SEQUENCE_KEYS:
        value = episode.get(key)
        if value is None:
            continue
        if isinstance(value, dict):
            for sub_key, tensor in value.items():
                if isinstance(tensor, torch.Tensor) and tensor.shape[0] != length:
                    issues.append(f"{key}.{sub_key}: expected {length}, got {tensor.shape[0]}")
        elif isinstance(value, torch.Tensor):
            if value.shape[0] != length:
                issues.append(f"{key}: expected {length}, got {value.shape[0]}")
    return issues


def _compare_shape_reference(
    key: str,
    shape: tuple[int, ...],
    reference: dict[str, tuple[int, ...]],
    mismatches: list[str],
) -> None:
    is_sequence = key.split(".")[0] in SEQUENCE_KEYS
    compare_shape = shape[1:] if is_sequence and shape else shape
    if key not in reference:
        reference[key] = compare_shape
        return
    if reference[key] != compare_shape:
        mismatches.append(f"{key}: expected {reference[key]}, got {compare_shape}")


def strip_reset_states(paths: Iterable[Path], output_dir: Path | None, suffix: str) -> None:
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        data = torch.load(path, map_location="cpu")
        episodes = _normalize_episodes(data)
        cleaned = [_strip_episode(ep) for ep in episodes]
        if isinstance(data, dict) and "episodes" in data:
            new_data = dict(data)
            new_data["episodes"] = cleaned
        else:
            new_data = {"episodes": cleaned}
        out_path = _output_path(path, output_dir, suffix)
        torch.save(new_data, out_path)
        removed = sum(1 for ep in episodes for key in RESET_KEYS if key in ep)
        print(f"[INFO] Wrote {out_path} (removed keys count: {removed})")


def check_shapes(paths: Iterable[Path], max_episodes: int | None) -> None:
    for path in paths:
        data = torch.load(path, map_location="cpu")
        episodes = _normalize_episodes(data)
        if max_episodes is not None:
            episodes = episodes[:max_episodes]
        if not episodes:
            print(f"[WARN] No episodes in {path}")
            continue
        reference: dict[str, tuple[int, ...]] = {}
        mismatches: list[str] = []
        length_issues: list[str] = []
        for idx, episode in enumerate(episodes):
            length = _extract_length(episode)
            length_issues.extend([f"episode {idx}: {issue}" for issue in _check_sequence_lengths(episode, length)])
            shapes: list[tuple[str, tuple[int, ...]]] = []
            _collect_tensor_shapes(episode, "", shapes)
            for key, shape in shapes:
                _compare_shape_reference(key, shape, reference, mismatches)
        print(f"[INFO] {path}: episodes={len(episodes)}")
        if length_issues:
            print(f"[WARN] Sequence length mismatches: {len(length_issues)}")
            for issue in length_issues:
                print(f"  - {issue}")
        if mismatches:
            print(f"[WARN] Shape mismatches vs reference: {len(mismatches)}")
            for issue in mismatches:
                print(f"  - {issue}")
        if not length_issues and not mismatches:
            print("[INFO] Shapes look consistent.")


def subset_episodes(
    paths: Iterable[Path],
    output_dir: Path | None,
    num_episodes: int,
) -> None:
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_subset_{num_episodes}"
    for path in paths:
        data = torch.load(path, map_location="cpu")
        episodes = _normalize_episodes(data)
        subset = episodes[:num_episodes]
        if isinstance(data, dict) and "episodes" in data:
            new_data = dict(data)
            new_data["episodes"] = subset
        else:
            new_data = {"episodes": subset}
        out_path = _output_path(path, output_dir, suffix)
        torch.save(new_data, out_path)
        print(f"[INFO] Wrote {out_path} (episodes: {len(subset)}/{len(episodes)})")


def random_subset_episodes(
    paths: Iterable[Path],
    output_dir: Path | None,
    num_episodes: int,
    seed: int,
) -> None:
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_subsetx"
    for path in paths:
        data = torch.load(path, map_location="cpu")
        episodes = _normalize_episodes(data)
        total = len(episodes)
        if total == 0:
            subset: list[dict[str, Any]] = []
        else:
            keep = min(num_episodes, total)
            generator = torch.Generator().manual_seed(seed)
            perm = torch.randperm(total, generator=generator).tolist()
            subset_indices = perm[:keep]
            subset = [episodes[i] for i in subset_indices]
        if isinstance(data, dict) and "episodes" in data:
            new_data = dict(data)
            new_data["episodes"] = subset
        else:
            new_data = {"episodes": subset}
        out_path = _output_path(path, output_dir, suffix)
        torch.save(new_data, out_path)
        print(f"[INFO] Wrote {out_path} (episodes: {len(subset)}/{len(episodes)})")


def split_episodes_train_val(
    path: Path,
    output_dir: Path | None,
    val_fraction: float = 0.05,
) -> tuple[Path, Path]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1).")
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    data = torch.load(path, map_location="cpu")
    episodes = _normalize_episodes(data)
    if len(episodes) < 2:
        raise ValueError("Need at least two episodes to create a train/val split.")
    generator = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(episodes), generator=generator)
    num_val = max(1, int(len(episodes) * val_fraction))
    val_indices = perm[:num_val].tolist()
    train_indices = perm[num_val:].tolist()
    train_episodes = [episodes[i] for i in train_indices]
    val_episodes = [episodes[i] for i in val_indices]
    train_path = _output_path(path, output_dir, "_train")
    val_path = _output_path(path, output_dir, "_val")
    if isinstance(data, dict) and "episodes" in data:
        train_data = dict(data)
        train_data["episodes"] = train_episodes
        val_data = dict(data)
        val_data["episodes"] = val_episodes
    else:
        train_data = {"episodes": train_episodes}
        val_data = {"episodes": val_episodes}
    torch.save(train_data, train_path)
    torch.save(val_data, val_path)
    print(
        f"[INFO] Wrote {train_path} (episodes: {len(train_episodes)}/{len(episodes)})"
    )
    print(
        f"[INFO] Wrote {val_path} (episodes: {len(val_episodes)}/{len(episodes)})"
    )
    return train_path, val_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset utilities for demo episode files.")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Episode files, directories, or glob patterns.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--strip-reset-states",
        action="store_true",
        help="Remove reset/raw state entries and save a smaller dataset.",
    )
    mode.add_argument(
        "--check-shapes",
        action="store_true",
        help="Check tensor shapes across episodes.",
    )
    mode.add_argument(
        "--subset-episodes",
        action="store_true",
        help="Write a dataset with only the first N episodes.",
    )
    mode.add_argument(
        "--random-subset-episodes",
        action="store_true",
        help="Write a dataset with N randomly sampled episodes using postfix _subsetx.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for modified datasets.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_no_reset",
        help="Suffix used when writing alongside inputs.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Limit episodes checked per file.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Number of episodes to keep when creating a subset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for random subset sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = _expand_inputs(args.inputs)
    if not paths:
        raise FileNotFoundError("No input episode files found.")
    output_dir = Path(args.output_dir) if args.output_dir else None
    if args.strip_reset_states:
        strip_reset_states(paths, output_dir, args.suffix)
    elif args.check_shapes:
        check_shapes(paths, args.max_episodes)
    elif args.subset_episodes:
        if args.num_episodes is None or args.num_episodes <= 0:
            raise ValueError("--num-episodes must be a positive integer when using --subset-episodes.")
        subset_episodes(paths, output_dir, args.num_episodes)
    elif args.random_subset_episodes:
        if args.num_episodes is None or args.num_episodes <= 0:
            raise ValueError("--num-episodes must be a positive integer when using --random-subset-episodes.")
        random_subset_episodes(paths, output_dir, args.num_episodes, args.seed)


if __name__ == "__main__":
    main()
