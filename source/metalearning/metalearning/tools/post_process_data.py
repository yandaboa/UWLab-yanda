"""Post-process episode files by trimming failed rollouts to closest approach."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import torch

from .visualization_utils import load_episodes, trim_to_length


def _episode_success_flag(episode: Mapping[str, Any]) -> bool:
    success = episode.get("success")
    if isinstance(success, torch.Tensor):
        if success.numel() == 0:
            return False
        return bool(success.reshape(-1)[0].item())
    if isinstance(success, (bool, int, float)):
        return bool(success)
    return False


def _get_obs_tensor(obs: Mapping[str, Any], keys: list[str]) -> Optional[torch.Tensor]:
    for key in keys:
        value = obs.get(key)
        if isinstance(value, torch.Tensor):
            return value
    return None


def _distance_series_from_obs(obs: Mapping[str, Any]) -> torch.Tensor:
    relative = _get_obs_tensor(obs, ["insertive_asset_in_receptive_asset_frame"])
    if relative is not None:
        if relative.shape[-1] < 3:
            raise ValueError(f"Relative pose has last dim {relative.shape[-1]}, expected at least 3.")
        return torch.linalg.norm(relative[..., :3], dim=-1)
    insertive = _get_obs_tensor(obs, ["insertive_asset_pose", "debug/insertive_asset_pose"])
    receptive = _get_obs_tensor(obs, ["receptive_asset_pose", "debug/receptive_asset_pose"])
    if insertive is None or receptive is None:
        raise KeyError(
            "Missing insertive/receptive pose tensors in obs. "
            "Expected keys like 'insertive_asset_pose' or 'insertive_asset_in_receptive_asset_frame'."
        )
    if insertive.shape[-1] < 3 or receptive.shape[-1] < 3:
        raise ValueError(
            f"Pose tensors must have last dim >= 3, got {insertive.shape[-1]} and {receptive.shape[-1]}."
        )
    return torch.linalg.norm(insertive[..., :3] - receptive[..., :3], dim=-1)


def _trim_raw_states(raw_states: Any, new_length: int) -> Any:
    if not isinstance(raw_states, list):
        return raw_states
    trimmed = []
    for entry in raw_states:
        if not isinstance(entry, dict):
            trimmed.append(entry)
            continue
        timestep = entry.get("timestep")
        if timestep is None or int(timestep) <= new_length - 1:
            trimmed.append(entry)
    return trimmed


def _set_done_terminal(dones: torch.Tensor) -> torch.Tensor:
    if dones.numel() == 0:
        return dones
    dones = dones.clone()
    if dones.dtype == torch.bool:
        dones[-1] = True
    else:
        dones[-1] = torch.ones((), dtype=dones.dtype)
    return dones


def _trim_episode(episode: Mapping[str, Any], new_length: int) -> dict[str, Any]:
    obs = episode.get("obs")
    if isinstance(obs, Mapping):
        obs_trimmed = {key: trim_to_length(value, new_length) for key, value in obs.items()}
    elif isinstance(obs, torch.Tensor):
        obs_trimmed = trim_to_length(obs, new_length)
    else:
        raise TypeError("Episode obs is missing or has unsupported type.")

    trimmed: dict[str, Any] = dict(episode)
    trimmed["obs"] = obs_trimmed
    for key in ("actions", "rewards", "dones"):
        value = episode.get(key)
        if isinstance(value, torch.Tensor):
            trimmed[key] = trim_to_length(value, new_length)
    if isinstance(trimmed.get("dones"), torch.Tensor):
        trimmed["dones"] = _set_done_terminal(trimmed["dones"])
    if "raw_states" in trimmed:
        trimmed["raw_states"] = _trim_raw_states(trimmed["raw_states"], new_length)
    trimmed["length"] = int(new_length)
    return trimmed


def filter_trim_failed_to_closest(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    processed = []
    for episode in episodes:
        if _episode_success_flag(episode):
            processed.append(episode)
            continue
        obs = episode.get("obs")
        if not isinstance(obs, Mapping):
            raise TypeError("Episode obs must be a mapping to compute distances.")
        length = int(episode["length"]) if "length" in episode else None
        distances = _distance_series_from_obs(obs)
        distances = trim_to_length(distances, length)
        if distances.numel() == 0:
            raise ValueError("Distance series is empty; cannot trim episode.")
        min_index = int(torch.argmin(distances).item())
        new_length = max(1, min_index + 1)
        processed.append(_trim_episode(episode, new_length))
    return processed


def apply_episode_filters(
    episodes: list[dict[str, Any]], filters: list[Callable[[list[dict[str, Any]]], list[dict[str, Any]]]]
) -> list[dict[str, Any]]:
    for filter_fn in filters:
        episodes = filter_fn(episodes)
    return episodes


def filter_only_successes(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [episode for episode in episodes if _episode_success_flag(episode)]


def split_train_eval(
    episodes: list[dict[str, Any]], train_ratio: float, seed: int
) -> Tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if train_ratio <= 0.0 or train_ratio >= 1.0:
        raise ValueError("train_ratio must be in (0, 1).")
    if not episodes:
        return [], []
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(episodes), generator=generator).tolist()
    train_count = max(1, int(round(len(episodes) * train_ratio)))
    train_idx = set(indices[:train_count])
    train = [episode for idx, episode in enumerate(episodes) if idx in train_idx]
    eval_episodes = [episode for idx, episode in enumerate(episodes) if idx not in train_idx]
    return train, eval_episodes


def _collect_episode_files(path: Path, pattern: str) -> list[Path]:
    return sorted([item for item in path.iterdir() if item.is_file() and item.match(pattern)])


def _resolve_out_path(
    input_path: Path, out_path: Optional[Path], out_dir: Optional[Path], suffix: str = "_post"
) -> Path:
    if out_path is not None:
        return out_path
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"{input_path.stem}{suffix}{input_path.suffix}"
    return input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")


def _build_suffix_parts(filters: Sequence[str], split_ratio: Optional[float]) -> str:
    parts = []
    if filters:
        parts.extend(filters)
    if split_ratio is not None:
        parts.append(f"train{int(round(split_ratio * 100))}")
    if not parts:
        return "_post"
    return "_" + "_".join(parts)


def _process_file(
    input_path: Path,
    out_path: Optional[Path],
    out_dir: Optional[Path],
    filters: list[Callable[[list[dict[str, Any]]], list[dict[str, Any]]]],
    filter_tags: Sequence[str],
    split_ratio: Optional[float],
    split_seed: int,
) -> list[Path]:
    episodes = load_episodes(input_path)
    processed = apply_episode_filters(episodes, filters)
    saved_paths: list[Path] = []
    if split_ratio is not None:
        train, eval_episodes = split_train_eval(processed, split_ratio, split_seed)
        base_suffix = _build_suffix_parts(filter_tags, split_ratio)
        train_path = _resolve_out_path(input_path, out_path, out_dir, suffix=f"{base_suffix}_train")
        eval_path = _resolve_out_path(input_path, None, out_dir, suffix=f"{base_suffix}_eval")
        torch.save({"episodes": train}, train_path)
        torch.save({"episodes": eval_episodes}, eval_path)
        saved_paths.extend([train_path, eval_path])
    else:
        suffix = _build_suffix_parts(filter_tags, None)
        resolved_out = _resolve_out_path(input_path, out_path, out_dir, suffix=suffix)
        torch.save({"episodes": processed}, resolved_out)
        saved_paths.append(resolved_out)
    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trim failed episodes to the closest insertive/receptive state."
    )
    parser.add_argument("path", type=Path, help="Path to a .pt episode file or directory.")
    parser.add_argument(
        "--pattern",
        type=str,
        default="episodes_*.pt",
        help="Glob pattern for episode files when path is a directory.",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=None,
        help="Optional output path (default: <input>_post.pt).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional output directory when processing a directory.",
    )
    parser.add_argument(
        "--trim-failed-to-closest",
        action="store_true",
        help="Trim failed episodes at closest insertive/receptive distance.",
    )
    parser.add_argument(
        "--only-successes",
        action="store_true",
        help="Filter out episodes that are not successful.",
    )
    parser.add_argument(
        "--train-eval-split",
        type=float,
        default=None,
        help="Optional train split ratio (0-1); saves train/eval files when provided.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=7,
        help="Seed for train/eval splitting (default: 7).",
    )
    args = parser.parse_args()

    filters = []
    filter_tags = []
    if args.trim_failed_to_closest:
        filters.append(filter_trim_failed_to_closest)
        filter_tags.append("trim")
    if args.only_successes:
        filters.append(filter_only_successes)
        filter_tags.append("successes")
    if not filters:
        filters = [filter_trim_failed_to_closest]
        filter_tags = ["trim"]
    if args.path.is_dir():
        episode_files = _collect_episode_files(args.path, args.pattern)
        if not episode_files:
            raise FileNotFoundError(f"No episode files matched {args.pattern} in {args.path}.")
        out_dir = args.out_dir or args.path
        for episode_path in episode_files:
            saved_paths = _process_file(
                episode_path,
                None,
                out_dir,
                filters,
                filter_tags,
                args.train_eval_split,
                args.split_seed,
            )
            for saved_path in saved_paths:
                print(f"[INFO] Saved post-processed episodes: {saved_path}")
    else:
        saved_paths = _process_file(
            args.path,
            args.out_path,
            None,
            filters,
            filter_tags,
            args.train_eval_split,
            args.split_seed,
        )
        for saved_path in saved_paths:
            print(f"[INFO] Saved post-processed episodes: {saved_path}")


if __name__ == "__main__":
    main()
