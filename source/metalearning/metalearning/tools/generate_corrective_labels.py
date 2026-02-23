"""Generate corrective action labels from collected demo episodes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from metalearning.corrective_labels import (
    CorrectiveLabeler,
    CorrectiveLabelerConfig,
    sample_perturbations_obs_space,
    validate_quat_convention,
)
from metalearning.tools.visualization_utils import get_pose_obs, load_episodes, trim_to_length


def _parse_floats(raw: str, expected_len: int) -> tuple[float, ...]:
    values = tuple(float(x.strip()) for x in raw.split(","))
    if len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} comma-separated values, got {len(values)}: '{raw}'")
    return values


def _parse_optional_pair(raw: str | None) -> tuple[float, float] | None:
    if raw is None:
        return None
    values = _parse_floats(raw, expected_len=2)
    return values[0], values[1]


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


def _iter_episode_steps(
    episode: Mapping[str, Any],
    obs_key: str | None,
    max_steps_per_episode: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    obs = episode["obs"]
    actions = episode["actions"]
    if not isinstance(actions, torch.Tensor):
        raise TypeError("Episode actions must be a torch.Tensor.")
    if actions.ndim == 3:
        if actions.shape[0] == 1:
            actions = actions[0]
        elif actions.shape[1] == 1:
            actions = actions[:, 0, :]
        else:
            # Some files store [num_envs, T, D]; flatten to per-step samples.
            actions = actions.reshape(-1, actions.shape[-1])
    if actions.ndim != 2 or actions.shape[-1] < 7:
        raise ValueError(f"Expected actions shape [T,>=7], got {tuple(actions.shape)}.")

    try:
        pose_obs, resolved_obs_key = get_pose_obs(obs, obs_key)
    except KeyError:
        # Allow robust fallback when requested key is absent in older episode files.
        pose_obs, resolved_obs_key = get_pose_obs(obs, None)
        print(f"[WARN] obs key '{obs_key}' missing; using '{resolved_obs_key}' instead.")
    if not isinstance(pose_obs, torch.Tensor):
        raise TypeError("Pose observation is not a tensor.")
    if pose_obs.ndim == 3:
        if pose_obs.shape[0] == 1:
            pose_obs = pose_obs[0]
        elif pose_obs.shape[1] == 1:
            pose_obs = pose_obs[:, 0, :]
        else:
            pose_obs = pose_obs.reshape(-1, pose_obs.shape[-1])
    if pose_obs.ndim != 2 or pose_obs.shape[-1] < 6:
        raise ValueError(f"Expected pose observation shape [T,>=6], got {tuple(pose_obs.shape)}.")

    length = int(episode.get("length", 0)) if "length" in episode else None
    pose_obs = trim_to_length(pose_obs, length)
    actions = trim_to_length(actions, length)

    steps = min(pose_obs.shape[0], actions.shape[0])
    if max_steps_per_episode is not None:
        steps = min(steps, max_steps_per_episode)
    if steps <= 0:
        return torch.empty(0, 6), torch.empty(0, 7)

    return pose_obs[:steps, :6], actions[:steps, :7]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate corrective labels for perturbed EE observations.")
    parser.add_argument(
        "--episode-paths",
        type=str,
        nargs="+",
        required=True,
        help="Episode files, directories, or glob patterns.",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        required=True,
        help="Output .pt file containing generated tuples.",
    )
    parser.add_argument("--obs-key", type=str, default="end_effector_pose", help="Observation key for EE pose.")
    parser.add_argument("--k", type=int, default=4, help="Perturbation samples per demo step.")
    parser.add_argument("--sigma-p", type=float, default=0.01, help="Position perturbation std in meters.")
    parser.add_argument("--sigma-r", type=float, default=0.05, help="Axis-angle perturbation std in radians.")
    parser.add_argument(
        "--q-grip",
        type=str,
        default="0.5,0.5,0.5,0.5",
        help="Gripper offset quaternion from metadata, comma-separated.",
    )
    parser.add_argument(
        "--q-aro",
        type=str,
        default="0,0,0,1",
        help="Action root offset quaternion from metadata, comma-separated.",
    )
    parser.add_argument(
        "--quat-order",
        type=str,
        choices=("auto", "xyzw", "wxyz"),
        default="auto",
        help="Metadata quaternion convention for q-grip/q-aro.",
    )
    parser.add_argument(
        "--scale",
        type=str,
        default="0.02,0.02,0.02,0.02,0.02,0.2",
        help="OSC action scale values.",
    )
    parser.add_argument(
        "--input-clip",
        type=str,
        default=None,
        help="Optional input clip range 'min,max' after scaling.",
    )
    parser.add_argument(
        "--action-clip",
        type=str,
        default="-1,1",
        help="Final corrective arm action clip range 'min,max'. Use 'none' to disable.",
    )
    parser.add_argument("--rot-clip", type=float, default=None, help="Optional axis-angle magnitude clip in radians.")
    parser.add_argument(
        "--invalid-handling",
        type=str,
        choices=("assert", "skip"),
        default="skip",
        help="Whether to assert on invalid rows or skip them.",
    )
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=None,
        help="Optional cap per episode for quick runs.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for perturbation sampling.")
    parser.add_argument(
        "--save-target",
        action="store_true",
        help="Also save p_target and q_target for each generated sample.",
    )
    args = parser.parse_args()

    episode_paths = _resolve_episode_paths(args.episode_paths)
    if not episode_paths:
        raise FileNotFoundError("No episode files found.")

    device = torch.device("cpu")
    q_grip_in = torch.tensor(_parse_floats(args.q_grip, 4), dtype=torch.float32, device=device)
    q_aro_in = torch.tensor(_parse_floats(args.q_aro, 4), dtype=torch.float32, device=device)
    q_grip_wxyz, inferred_grip = validate_quat_convention(q_grip_in, args.quat_order)
    q_aro_wxyz, inferred_aro = validate_quat_convention(q_aro_in, args.quat_order)

    input_clip = _parse_optional_pair(args.input_clip)
    if args.action_clip is None or str(args.action_clip).lower() == "none":
        action_clip = None
    else:
        action_clip = _parse_optional_pair(args.action_clip)

    cfg = CorrectiveLabelerConfig(
        scale=_parse_floats(args.scale, 6),  # type: ignore[arg-type]
        input_clip=input_clip,
        action_clip=action_clip,
        rot_clip=args.rot_clip,
        invalid_handling=args.invalid_handling,
    )
    labeler = CorrectiveLabeler(cfg=cfg, q_grip_wxyz=q_grip_wxyz, q_action_root_offset_wxyz=q_aro_wxyz, device=device)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    all_s_g: list[torch.Tensor] = []
    all_a_g: list[torch.Tensor] = []
    all_valid: list[torch.Tensor] = []
    all_source: list[torch.Tensor] = []
    all_indices: list[torch.Tensor] = []
    all_targets_pos: list[torch.Tensor] = []
    all_targets_quat: list[torch.Tensor] = []

    for file_idx, path in enumerate(episode_paths):
        episodes = load_episodes(path)
        for ep_idx, episode in enumerate(episodes):
            obs_t, a_t = _iter_episode_steps(episode, args.obs_key, args.max_steps_per_episode)
            if obs_t.numel() == 0:
                continue

            obs_g = sample_perturbations_obs_space(
                obs_t,
                num_samples_per_step=args.k,
                sigma_p=args.sigma_p,
                sigma_r=args.sigma_r,
                generator=generator,
            )
            obs_t_rep = obs_t.repeat_interleave(args.k, dim=0)
            a_t_rep = a_t.repeat_interleave(args.k, dim=0)
            a_g, valid_mask = labeler.label(obs_t_rep, a_t_rep, obs_g, return_valid_mask=True)  # type: ignore[assignment]

            step_idx = torch.arange(obs_t.shape[0], dtype=torch.int64).repeat_interleave(args.k)
            source_index = torch.stack(
                [
                    torch.full_like(step_idx, file_idx),
                    torch.full_like(step_idx, ep_idx),
                    step_idx,
                ],
                dim=-1,
            )

            all_s_g.append(obs_g.cpu())
            all_a_g.append(a_g.cpu())
            all_valid.append(valid_mask.cpu())
            all_source.append(obs_t_rep.cpu())
            all_indices.append(source_index.cpu())

            if args.save_target:
                p_target, q_target = labeler.compute_target_pose(obs_t_rep, a_t_rep[:, :6])
                all_targets_pos.append(p_target.cpu())
                all_targets_quat.append(q_target.cpu())

    if not all_s_g:
        raise RuntimeError("No valid samples generated.")

    s_g = torch.cat(all_s_g, dim=0)
    a_g = torch.cat(all_a_g, dim=0)
    valid_mask = torch.cat(all_valid, dim=0)
    source_obs_t = torch.cat(all_source, dim=0)
    source_indices = torch.cat(all_indices, dim=0)

    a_abs = torch.abs(a_g[:, :6])
    qtiles = torch.tensor([0.5, 0.9, 0.99], dtype=torch.float32)
    stats = {
        "num_samples": int(s_g.shape[0]),
        "num_valid": int(valid_mask.sum().item()),
        "num_invalid": int((~valid_mask).sum().item()),
        "arm_abs_median": torch.quantile(a_abs, qtiles[0], dim=0),
        "arm_abs_p90": torch.quantile(a_abs, qtiles[1], dim=0),
        "arm_abs_p99": torch.quantile(a_abs, qtiles[2], dim=0),
        "inferred_quat_order_grip": inferred_grip,
        "inferred_quat_order_aro": inferred_aro,
    }

    payload: dict[str, Any] = {
        "s_g": s_g,
        "a_g": a_g,
        "valid_mask": valid_mask,
        "source_obs_t": source_obs_t,
        "source_indices": source_indices,  # [file_idx, episode_idx, timestep]
        "meta": {
            "k": args.k,
            "sigma_p": args.sigma_p,
            "sigma_r": args.sigma_r,
            "scale": cfg.scale,
            "input_clip": cfg.input_clip,
            "action_clip": cfg.action_clip,
            "rot_clip": cfg.rot_clip,
            "invalid_handling": cfg.invalid_handling,
            "q_grip_wxyz": q_grip_wxyz.cpu(),
            "q_aro_wxyz": q_aro_wxyz.cpu(),
            "episode_paths": [str(p) for p in episode_paths],
        },
        "stats": stats,
    }
    if args.save_target:
        payload["target_pos"] = torch.cat(all_targets_pos, dim=0)
        payload["target_quat"] = torch.cat(all_targets_quat, dim=0)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out_path)

    print(f"[INFO] Saved corrective labels to: {args.out_path}")
    print(f"[INFO] Samples: {stats['num_samples']} (valid={stats['num_valid']}, invalid={stats['num_invalid']})")
    print(f"[INFO] Arm |a_g| p90: {stats['arm_abs_p90'].tolist()}")


if __name__ == "__main__":
    main()

