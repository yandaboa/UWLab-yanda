#!/usr/bin/env python3
"""Run dm_control point-mass inference with a supervised context checkpoint."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Set before importing dm_control so backend selection is headless-safe.
os.environ.setdefault("MUJOCO_GL", "egl")

from dm_control import suite


ROOT_DIR = Path(__file__).resolve().parents[3]
SOURCE_DIR = ROOT_DIR / "source"
PKG_DIRS = [
    ROOT_DIR,
    SOURCE_DIR,
    SOURCE_DIR / "uwlab_rl",
    SOURCE_DIR / "uwlab_tasks",
]
for path in PKG_DIRS:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for supervised context model on point-mass.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model_*.pt checkpoint.")
    parser.add_argument(
        "--context-path",
        type=str,
        required=True,
        help="Path to local rollout file (.pt) containing 'episode_groups'.",
    )
    parser.add_argument("--task", choices=["easy", "hard"], default="easy")
    parser.add_argument("--time-limit-sec", type=float, default=5.0)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=None, help="Optional hard cap per episode.")
    parser.add_argument(
        "--group-index",
        type=int,
        default=-1,
        help="Episode-group index to use. Set to -1 to sample a random group each episode.",
    )
    parser.add_argument(
        "--num-context-trajs",
        type=int,
        default=2,
        help="How many noisy rollouts from the selected group to concatenate into context.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb-project", type=str, default="point_mass_supervised_inference")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument(
        "--open-loop",
        action="store_true",
        help=(
            "Use grouped reference spline states (group_splines) as query states, "
            "predict all actions in one batch, then execute them open-loop."
        ),
    )
    parser.add_argument(
        "--save-rollouts-path",
        type=str,
        default=None,
        help="Output .pt path for inferred rollouts. Defaults to checkpoint directory when unset.",
    )
    return parser.parse_args()


def _init_wandb(args: argparse.Namespace, checkpoint_path: Path) -> Any | None:
    try:
        import wandb  # type: ignore
    except ImportError:
        return None
    run_name = args.wandb_run_name or f"infer_{checkpoint_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "checkpoint": str(checkpoint_path),
            "context_path": args.context_path,
            "task": args.task,
            "num_episodes": args.num_episodes,
            "open_loop": bool(args.open_loop),
            "num_context_trajs": args.num_context_trajs,
        },
    )


def _load_grouped_payload(
    context_path: str,
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]] | None]:
    payload = torch.load(context_path, map_location="cpu")
    episode_groups = payload["episode_groups"]
    group_splines = payload.get("group_splines")
    if group_splines is not None:
        assert isinstance(group_splines, list), "Expected 'group_splines' to be a list when present."
        assert len(group_splines) == len(episode_groups), (
            "Expected len(group_splines) == len(episode_groups), "
            f"got {len(group_splines)} vs {len(episode_groups)}."
        )
    return episode_groups, group_splines


def _episode_to_tensors(
    episode: dict[str, Any], obs_keys: list[str], reward_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    obs_dict = episode["obs"]
    actions = episode["actions"]

    obs_parts: list[torch.Tensor] = []
    for key in obs_keys:
        obs_parts.append(torch.as_tensor(obs_dict[key], dtype=torch.float32).flatten(start_dim=1))
    obs = torch.cat(obs_parts, dim=-1)

    action_tensor = torch.as_tensor(actions, dtype=torch.float32).reshape(obs.shape[0], -1)
    seq_len = min(obs.shape[0], action_tensor.shape[0], int(episode["length"]))
    obs = obs[:seq_len]
    action_tensor = action_tensor[:seq_len]
    reward_tensor = torch.zeros((seq_len, reward_dim), dtype=torch.float32)
    return obs, action_tensor, reward_tensor


def _build_context_batch(
    group: list[dict[str, Any]],
    obs_keys: list[str],
    reward_dim: int,
    num_context_trajs: int,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    assert num_context_trajs >= 1
    selected = rng.permutation(len(group))[:num_context_trajs].tolist()
    context_obs: list[torch.Tensor] = []
    context_actions: list[torch.Tensor] = []
    context_rewards: list[torch.Tensor] = []
    for idx in selected:
        obs, actions, rewards = _episode_to_tensors(group[idx], obs_keys, reward_dim)
        context_obs.append(obs)
        context_actions.append(actions)
        context_rewards.append(rewards)

    demo_obs = torch.cat(context_obs, dim=0).unsqueeze(0).to(device)
    demo_actions = torch.cat(context_actions, dim=0).unsqueeze(0).to(device)
    demo_rewards = torch.cat(context_rewards, dim=0).unsqueeze(0).to(device)
    demo_lengths = torch.tensor([[demo_obs.shape[1]]], dtype=torch.long, device=device)
    return demo_obs, demo_actions, demo_rewards, demo_lengths, selected


def _extract_context_episodes_for_save(
    group: list[dict[str, Any]], selected_idxs: list[int], reward_dim: int
) -> list[dict[str, Any]]:
    context_episodes: list[dict[str, Any]] = []
    for idx in selected_idxs:
        episode = group[idx]
        seq_len = int(episode["length"])
        obs_dict = {
            key: torch.as_tensor(value, dtype=torch.float32)[:seq_len].cpu()
            for key, value in episode["obs"].items()
        }
        actions = torch.as_tensor(episode["actions"], dtype=torch.float32)[:seq_len].cpu()
        if "rewards" in episode:
            rewards = torch.as_tensor(episode["rewards"], dtype=torch.float32)[:seq_len].reshape(seq_len, -1).cpu()
        else:
            rewards = torch.zeros((seq_len, reward_dim), dtype=torch.float32)
        context_episodes.append(
            {
                "context_index": idx,
                "obs": obs_dict,
                "actions": actions,
                "rewards": rewards,
                "length": seq_len,
            }
        )
    return context_episodes


def _get_current_obs_vector(env: Any, time_step: Any, obs_keys: list[str]) -> np.ndarray:
    obs_dict = time_step.observation
    parts: list[np.ndarray] = []
    assert "position" in obs_keys and "velocity" in obs_keys, "Position and velocity must be in obs_keys."
    for key in obs_keys:
        # Prefer physics for position/velocity so manual state resets are reflected immediately.
        if key == "position":
            parts.append(np.asarray(env.physics.position(), dtype=np.float32).reshape(-1))
            continue
        if key == "velocity":
            parts.append(np.asarray(env.physics.velocity(), dtype=np.float32).reshape(-1))
            continue
        parts.append(np.asarray(obs_dict[key], dtype=np.float32).reshape(-1))
    return np.concatenate(parts, axis=0).astype(np.float32, copy=False)


def _obs_dict_to_vector(obs_dict: dict[str, Any], obs_keys: list[str], index: int) -> np.ndarray:
    parts: list[np.ndarray] = []
    for key in obs_keys:
        value = np.asarray(obs_dict[key][index], dtype=np.float32).reshape(-1)
        parts.append(value)
    return np.concatenate(parts, axis=0).astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if args.save_rollouts_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_rollouts_path = str(
            checkpoint_path.parent / f"infer_point_mass_rollouts_{timestamp}.pt"
        )

    from uwlab_rl.rsl_rl.context_sequence_policy import ContextSequencePolicy

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model, meta = ContextSequencePolicy.from_checkpoint(checkpoint, device=device)
    model.eval()
    wandb_run = _init_wandb(args, checkpoint_path)
    obs_keys = list(model.cfg.data.obs_keys or [])
    reward_dim = int(meta["reward_dim"])

    episode_groups, group_splines = _load_grouped_payload(args.context_path)
    print(f"[INFO] Loaded {len(episode_groups)} context groups from: {args.context_path}")
    if args.open_loop and group_splines is None:
        raise ValueError(
            "--open-loop requires grouped payload with 'group_splines'. "
            f"Missing group_splines in {args.context_path}."
        )

    env = suite.load(
        domain_name="point_mass",
        task_name=args.task,
        task_kwargs={"random": args.seed, "time_limit": args.time_limit_sec},
    )

    all_rollouts: list[dict[str, Any]] = []
    with torch.no_grad():
        for episode_idx in range(args.num_episodes):
            if args.group_index < 0:
                group_idx = int(rng.integers(0, len(episode_groups)))
            else:
                group_idx = args.group_index
            group = episode_groups[group_idx]
            group_spline = None if group_splines is None else group_splines[group_idx]

            demo_obs, demo_actions, demo_rewards, demo_lengths, selected_idxs = _build_context_batch(
                group=group,
                obs_keys=obs_keys,
                reward_dim=reward_dim,
                num_context_trajs=args.num_context_trajs,
                rng=rng,
                device=device,
            )

            time_step = env.reset()
            # Align initial state with chosen reference source.
            if args.open_loop:
                assert group_spline is not None
                if not (isinstance(group_spline, dict) and "obs" in group_spline):
                    raise ValueError("Each group_splines entry must be a dict with key 'obs'.")
                anchor_obs = group_spline["obs"]
            else:
                anchor_obs = group[selected_idxs[0]]["obs"]
            with env.physics.reset_context():
                pos0 = np.asarray(anchor_obs["position"][0], dtype=np.float32).reshape(-1)
                env.physics.named.data.qpos["root_x"] = float(pos0[0])
                env.physics.named.data.qpos["root_y"] = float(pos0[1])
                vel0 = np.asarray(anchor_obs["velocity"][0], dtype=np.float32).reshape(-1)
                env.physics.named.data.qvel["root_x"] = float(vel0[0])
                env.physics.named.data.qvel["root_y"] = float(vel0[1])

            obs_log: list[np.ndarray] = []
            action_log: list[np.ndarray] = []
            reward_log: list[float] = []
            total_reward = 0.0
            t = 0
            if args.open_loop:
                assert group_spline is not None
                if not (isinstance(group_spline, dict) and "obs" in group_spline):
                    raise ValueError("Each group_splines entry must be a dict with key 'obs'.")
                spline_obs_dict = group_spline["obs"]
                if not isinstance(spline_obs_dict, dict):
                    raise ValueError("group_splines[*]['obs'] must be a dict.")
                missing_keys = [key for key in obs_keys if key not in spline_obs_dict]
                if missing_keys:
                    raise ValueError(
                        f"group_splines[{group_idx}]['obs'] missing required keys for obs encoding: {missing_keys}"
                    )
                spline_len = min(len(spline_obs_dict[key]) for key in obs_keys)
                if args.max_steps is not None:
                    spline_len = min(spline_len, int(args.max_steps))
                if spline_len <= 0:
                    raise ValueError(f"group_splines[{group_idx}] has no valid timesteps.")

                spline_obs_np = np.stack(
                    [_obs_dict_to_vector(spline_obs_dict, obs_keys, index=i) for i in range(spline_len)],
                    axis=0,
                )
                spline_obs_tensor = torch.from_numpy(spline_obs_np).to(device)
                demo_obs_batch = demo_obs.expand(spline_len, -1, -1)
                demo_actions_batch = demo_actions.expand(spline_len, -1, -1)
                demo_rewards_batch = demo_rewards.expand(spline_len, -1, -1)
                demo_lengths_batch = demo_lengths.expand(spline_len, -1)
                planned_actions = model.act(
                    demo_obs=demo_obs_batch,
                    demo_actions=demo_actions_batch,
                    demo_rewards=demo_rewards_batch,
                    demo_lengths=demo_lengths_batch,
                    current_obs=spline_obs_tensor,
                )
                planned_actions_np = planned_actions.detach().cpu().numpy().astype(np.float32, copy=False)

                for i in range(spline_len):
                    current_obs = _get_current_obs_vector(env, time_step, obs_keys)
                    obs_log.append(current_obs)
                    action_np = planned_actions_np[i]
                    action_log.append(action_np)
                    time_step = env.step(action_np)
                    reward = 0.0 if time_step.reward is None else float(time_step.reward)
                    reward_log.append(reward)
                    total_reward += reward
                    t += 1
                    if time_step.last():
                        break
            else:
                while True:
                    current_obs = _get_current_obs_vector(env, time_step, obs_keys)
                    obs_log.append(current_obs)
                    current_obs_tensor = torch.from_numpy(current_obs).unsqueeze(0).to(device)
                    action = model.act(
                        demo_obs=demo_obs,
                        demo_actions=demo_actions,
                        demo_rewards=demo_rewards,
                        demo_lengths=demo_lengths,
                        current_obs=current_obs_tensor,
                    )
                    action_np = action.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
                    action_log.append(action_np)

                    time_step = env.step(action_np)
                    reward = 0.0 if time_step.reward is None else float(time_step.reward)
                    reward_log.append(reward)
                    total_reward += reward
                    t += 1

                    if time_step.last():
                        break
                    if args.max_steps is not None and t >= args.max_steps:
                        break

            rollout = {
                "group_index": group_idx,
                "selected_context_indices": selected_idxs,
                "open_loop": bool(args.open_loop),
                "context_episodes": _extract_context_episodes_for_save(
                    group=group,
                    selected_idxs=selected_idxs,
                    reward_dim=reward_dim,
                ),
                "obs": np.stack(obs_log, axis=0),
                "actions": np.stack(action_log, axis=0),
                "rewards": np.asarray(reward_log, dtype=np.float32),
                "length": len(action_log),
                "total_reward": total_reward,
            }
            final_pos_l2_error = None
            if group_spline is not None:
                spline_obs = group_spline.get("obs") if isinstance(group_spline, dict) else None
                if isinstance(spline_obs, dict) and "position" in spline_obs:
                    true_final_position = np.asarray(spline_obs["position"][-1], dtype=np.float32).reshape(-1)
                    rollout_final_position = np.asarray(env.physics.position(), dtype=np.float32).reshape(-1)
                    final_pos_l2_error = float(np.linalg.norm(rollout_final_position - true_final_position, ord=2))
                    rollout["final_position_l2_error"] = final_pos_l2_error
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "eval/final_position_l2_error": final_pos_l2_error,
                                "eval/episode_idx": episode_idx,
                                "eval/group_idx": group_idx,
                            }
                        )
            all_rollouts.append(rollout)
            print(
                f"[EP {episode_idx:03d}] group={group_idx} "
                f"context_ids={selected_idxs} length={rollout['length']} total_reward={total_reward:.4f}"
            )
            if final_pos_l2_error is not None:
                print(f"         final_position_l2_error={final_pos_l2_error:.6f}")

    out_path = Path(args.save_rollouts_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "rollouts": all_rollouts,
            "context_path": args.context_path,
            "checkpoint_path": args.checkpoint,
        },
        out_path,
    )
    print(f"[INFO] Saved rollouts to: {out_path}")

    mean_return = float(np.mean([ep["total_reward"] for ep in all_rollouts])) if all_rollouts else 0.0
    print(f"[DONE] Ran {len(all_rollouts)} episodes. Mean return: {mean_return:.4f}")
    valid_l2_errors = [
        float(ep["final_position_l2_error"])
        for ep in all_rollouts
        if isinstance(ep, dict) and "final_position_l2_error" in ep
    ]
    if valid_l2_errors:
        mean_final_pos_l2_error = float(np.mean(valid_l2_errors))
        print(f"[DONE] Mean final_position_l2_error: {mean_final_pos_l2_error:.6f}")
        if wandb_run is not None:
            wandb_run.log(
                {
                    "eval/mean_return": mean_return,
                    "eval/mean_final_position_l2_error": mean_final_pos_l2_error,
                }
            )
    elif wandb_run is not None:
        wandb_run.log({"eval/mean_return": mean_return})
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()

