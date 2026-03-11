#!/usr/bin/env python3
"""Run dm_control point-mass inference with a supervised context checkpoint."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Set before importing dm_control so backend selection is headless-safe.
os.environ.setdefault("MUJOCO_GL", "egl")
ROOT_DIR = Path(__file__).resolve().parents[3]
SOURCE_DIR = ROOT_DIR / "source"
DM_CONTROL_REPO_DIR = ROOT_DIR.parent / "dm_control"
PKG_DIRS = [
    ROOT_DIR,
    SOURCE_DIR,
    SOURCE_DIR / "uwlab_rl",
    SOURCE_DIR / "uwlab_tasks",
]
for path in PKG_DIRS:
    path_str = str(path)
    if path_str in sys.path:
        # Reinsert local package roots with highest import priority.
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)

from dm_control import suite

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
        default=3,
        help="How many noisy rollouts from the selected group to concatenate into context.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb-project", type=str, default="point_mass_supervised_inference")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument(
        "--max-plot-rollouts",
        type=int,
        default=8,
        help="Optional max rollout plots to render and upload to W&B.",
    )
    parser.add_argument(
        "--open-loop",
        action="store_true",
        help=(
            "Use grouped reference spline states (group_splines) as query states, "
            "predict all actions in one batch, then execute them open-loop."
        ),
    )
    parser.add_argument(
        "--open-loop-action-playback",
        action="store_true",
        help=(
            "Replay spline actions in the environment instead of inferred actions. "
            "Useful as a sanity-check baseline."
        ),
    )
    parser.add_argument(
        "--save-rollouts-path",
        type=str,
        default=None,
        help="Output .pt path for inferred rollouts. Defaults to checkpoint directory when unset.",
    )
    parser.add_argument(
        "--local-point-mass-file",
        type=str,
        default=str(DM_CONTROL_REPO_DIR / "dm_control" / "suite" / "point_mass.py"),
        help=(
            "Path to a local point_mass.py override. "
            "Set empty string to disable and use library dm_control suite domain."
        ),
    )
    return parser.parse_args()


def _install_local_point_mass_override(local_point_mass_file: str) -> None:
    override = local_point_mass_file.strip()
    if not override:
        print("[INFO] Using library dm_control.suite.point_mass (no local override).")
        return
    point_mass_path = Path(override).expanduser().resolve()
    assert point_mass_path.is_file(), f"Local point_mass override not found: {point_mass_path}"
    spec = importlib.util.spec_from_file_location("local_point_mass_override", str(point_mass_path))
    assert spec is not None and spec.loader is not None, f"Failed to load module spec for: {point_mass_path}"
    local_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(local_module)
    assert hasattr(local_module, "SUITE"), f"Local point_mass module does not define SUITE: {point_mass_path}"
    domains = suite.__dict__.get("_DOMAINS")
    assert isinstance(domains, dict), "dm_control.suite internals changed: missing _DOMAINS registry."
    domains["point_mass"] = local_module
    suite.point_mass = local_module
    print(f"[INFO] Overrode dm_control point_mass domain from: {point_mass_path}")


def _init_wandb(args: argparse.Namespace, checkpoint_path: Path, context_path: Path) -> Any | None:
    try:
        import wandb  # type: ignore
    except ImportError:
        return None
    run_name = args.wandb_run_name or f"infer_{checkpoint_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb_api: Any = wandb
    return wandb_api.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "checkpoint": str(checkpoint_path),
            "context_path": str(context_path),
            "task": args.task,
            "num_episodes": args.num_episodes,
            "open_loop": bool(args.open_loop),
            "open_loop_action_playback": bool(args.open_loop_action_playback),
            "num_context_trajs": args.num_context_trajs,
            "time_limit_sec": args.time_limit_sec,
            "group_index": args.group_index,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "device": args.device,
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


def _to_serializable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(v) for v in value]
    if hasattr(value, "__dict__"):
        return {str(k): _to_serializable(v) for k, v in vars(value).items()}
    return repr(value)


def _render_rollout_plots(rollouts_path: Path, max_rollouts: int | None) -> tuple[Path | None, list[Path]]:
    viz_script = ROOT_DIR.parent / "dm_control" / "visualize_point_mass_infer_rollouts.py"
    if not viz_script.is_file():
        print(f"[WARN] Visualization script not found, skipping plots: {viz_script}")
        return None, []
    out_dir = rollouts_path.parent / f"{rollouts_path.stem}_plots"
    cmd = [
        sys.executable,
        str(viz_script),
        "--rollouts-path",
        str(rollouts_path),
        "--out-dir",
        str(out_dir),
        "--include-reference",
    ]
    if max_rollouts is not None:
        cmd.extend(["--max-rollouts", str(int(max_rollouts))])
    try:
        subprocess.run(cmd, check=True)
    except Exception as exc:
        print(f"[WARN] Failed to render rollout plots: {exc}")
        return None, []
    images = sorted(out_dir.glob("rollout_*.png"))
    print(f"[INFO] Rendered {len(images)} rollout plot(s) to: {out_dir}")
    return out_dir, images


def _render_open_loop_action_comparison_plots(
    rollouts: list[dict[str, Any]],
    rollouts_path: Path,
    max_rollouts: int | None,
) -> tuple[Path | None, list[Path]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = rollouts_path.parent / f"{rollouts_path.stem}_open_loop_action_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    images: list[Path] = []
    plotted = 0
    for rollout_idx, rollout in enumerate(rollouts):
        if not bool(rollout.get("open_loop", False)):
            continue
        inferred = rollout.get("open_loop_inferred_actions")
        spline = rollout.get("open_loop_spline_actions")
        if inferred is None or spline is None:
            continue
        inferred_np = np.asarray(inferred, dtype=np.float32)
        spline_np = np.asarray(spline, dtype=np.float32)
        if inferred_np.ndim != 2 or spline_np.ndim != 2:
            continue
        steps = min(inferred_np.shape[0], spline_np.shape[0])
        action_dim = min(inferred_np.shape[1], spline_np.shape[1])
        if steps <= 0 or action_dim <= 0:
            continue

        fig, axes = plt.subplots(action_dim, 1, figsize=(8, max(2.5 * action_dim, 3.0)), sharex=True)
        if action_dim == 1:
            axes = [axes]
        xs = np.arange(steps)
        for dim in range(action_dim):
            ax = axes[dim]
            ax.plot(xs, inferred_np[:steps, dim], label="inferred", linewidth=1.6)
            ax.plot(xs, spline_np[:steps, dim], label="spline", linewidth=1.2, alpha=0.85)
            ax.set_ylabel(f"a[{dim}]")
            ax.grid(True, linewidth=0.35, alpha=0.5)
            if dim == 0:
                ax.legend(loc="best", fontsize=8)
        axes[-1].set_xlabel("timestep")
        fig.suptitle(f"Open-loop actions rollout {rollout_idx}", fontsize=10)
        fig.tight_layout()
        out_file = out_dir / f"rollout_{rollout_idx:05d}_actions.png"
        fig.savefig(out_file, dpi=180)
        plt.close(fig)
        images.append(out_file)
        plotted += 1
        if max_rollouts is not None and plotted >= int(max_rollouts):
            break
    if not images:
        return None, []
    print(f"[INFO] Rendered {len(images)} open-loop action plot(s) to: {out_dir}")
    return out_dir, images


def _append_prefix_to_demo(
    demo_obs: torch.Tensor,
    demo_actions: torch.Tensor,
    demo_rewards: torch.Tensor,
    demo_lengths: torch.Tensor,
    prefix_obs: torch.Tensor,
    prefix_actions: torch.Tensor,
    prefix_rewards: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Append rollout prefix [0, t) to demo context along time axis."""
    prefix_len = int(prefix_obs.shape[0])
    if prefix_len == 0:
        return demo_obs, demo_actions, demo_rewards, demo_lengths
    assert demo_obs.shape[0] == demo_actions.shape[0] == demo_rewards.shape[0] == demo_lengths.shape[0] == 1
    prefix_obs = prefix_obs.unsqueeze(0).to(device=demo_obs.device, dtype=demo_obs.dtype)
    prefix_actions = prefix_actions.unsqueeze(0).to(device=demo_actions.device, dtype=demo_actions.dtype)
    prefix_rewards = prefix_rewards.unsqueeze(0).to(device=demo_rewards.device, dtype=demo_rewards.dtype)
    return (
        torch.cat([demo_obs, prefix_obs], dim=1),
        torch.cat([demo_actions, prefix_actions], dim=1),
        torch.cat([demo_rewards, prefix_rewards], dim=1),
        demo_lengths + prefix_len,
    )


def main() -> None:
    args = parse_args()
    _install_local_point_mass_override(args.local_point_mass_file)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    context_path = Path(args.context_path).expanduser().resolve()
    args.context_path = str(context_path)
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
    model_cfg = _to_serializable(getattr(model, "cfg", {}))
    print(f"[INFO] Loaded checkpoint: {checkpoint_path}")
    print(f"[INFO] Evaluating context file: {context_path}")
    print("[INFO] Loaded model config:")
    print(json.dumps(model_cfg, indent=2, sort_keys=True))
    wandb_run = _init_wandb(args, checkpoint_path, context_path)
    if wandb_run is not None:
        wandb_run.config.update(
            {
                "model_cfg": model_cfg,
                "checkpoint_path": str(checkpoint_path),
            },
            allow_val_change=True,
        )
        wandb_run.log(
            {
                "eval/context_path": str(context_path),
                "eval/checkpoint_path": str(checkpoint_path),
            }
        )
    obs_keys = list(model.cfg.data.obs_keys or [])
    include_current_trajectory = bool(getattr(getattr(model.cfg, "input", None), "include_current_trajectory", False))
    reward_dim = int(meta["reward_dim"])
    action_dim = int(meta["action_dim"])
    print(f"[INFO] include_current_trajectory={include_current_trajectory}")

    episode_groups, group_splines = _load_grouped_payload(args.context_path)
    print(f"[INFO] Loaded {len(episode_groups)} context groups from: {args.context_path}")
    assert not (args.open_loop and args.open_loop_action_playback), ("--open-loop and --open-loop-action-playback are mutually exclusive modes.")
    assert not ((args.open_loop or args.open_loop_action_playback) and group_splines is None), ("--open-loop and --open-loop-action-playback require grouped payload with 'group_splines'. Missing group_splines in {args.context_path}.")

    env = suite.load(
        domain_name="point_mass",
        task_name=args.task,
        task_kwargs={"random": args.seed, "time_limit": args.time_limit_sec},
    )

    all_rollouts: list[dict[str, Any]] = []
    with torch.no_grad():
        for episode_idx in range(args.num_episodes):
            use_spline_mode = bool(args.open_loop or args.open_loop_action_playback)
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
            if use_spline_mode:
                assert group_spline is not None
                assert isinstance(group_spline, dict) and "obs" in group_spline, ("Each group_splines entry must be a dict with key 'obs'.")
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
            prefix_obs_log: list[np.ndarray] = []
            prefix_action_log: list[np.ndarray] = []
            prefix_reward_log: list[float] = []
            inferred_action_log: list[np.ndarray] = []
            spline_action_log: list[np.ndarray] = []
            total_reward = 0.0
            t = 0
            if use_spline_mode:
                assert group_spline is not None
                assert isinstance(group_spline, dict) and "obs" in group_spline, ("Each group_splines entry must be a dict with key 'obs'.")
                spline_obs_dict = group_spline["obs"]
                assert isinstance(spline_obs_dict, dict), "group_splines[*]['obs'] must be a dict."
                missing_keys = [key for key in obs_keys if key not in spline_obs_dict]
                assert not missing_keys, (f"group_splines[{group_idx}]['obs'] missing required keys for obs encoding: {missing_keys}")
                spline_len = min(len(spline_obs_dict[key]) for key in obs_keys)
                if args.max_steps is not None:
                    spline_len = min(spline_len, int(args.max_steps))
                assert spline_len > 0, f"group_splines[{group_idx}] has no valid timesteps."

                spline_obs_np = np.stack(
                    [_obs_dict_to_vector(spline_obs_dict, obs_keys, index=i) for i in range(spline_len)],
                    axis=0,
                )
                spline_actions_raw = group_spline.get("actions") if isinstance(group_spline, dict) else None
                assert spline_actions_raw is not None, ("Expected group_splines[*]['actions'] for spline-based open-loop modes, but it is missing.")
                spline_actions_np = np.asarray(spline_actions_raw, dtype=np.float32).reshape(-1, action_dim)
                assert spline_actions_np.shape[0] >= spline_len, (f"group_splines[{group_idx}]['actions'] has length {spline_actions_np.shape[0]}, but at least {spline_len} is required.")
                spline_actions_np = spline_actions_np[:spline_len]

                if args.open_loop and include_current_trajectory:
                    for i in range(spline_len):
                        prefix_obs = torch.from_numpy(spline_obs_np[:i])
                        prefix_actions = torch.from_numpy(spline_actions_np[:i])
                        prefix_rewards = torch.zeros((i, reward_dim), dtype=torch.float32)
                        demo_obs_i, demo_actions_i, demo_rewards_i, demo_lengths_i = _append_prefix_to_demo(
                            demo_obs=demo_obs,
                            demo_actions=demo_actions,
                            demo_rewards=demo_rewards,
                            demo_lengths=demo_lengths,
                            prefix_obs=prefix_obs,
                            prefix_actions=prefix_actions,
                            prefix_rewards=prefix_rewards,
                        )
                        current_obs_tensor = torch.from_numpy(spline_obs_np[i]).unsqueeze(0).to(device)
                        action = model.act(
                            demo_obs=demo_obs_i,
                            demo_actions=demo_actions_i,
                            demo_rewards=demo_rewards_i,
                            demo_lengths=demo_lengths_i,
                            current_obs=current_obs_tensor,
                        )
                        inferred_action_np = action.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
                        spline_action_log.append(spline_actions_np[i].astype(np.float32, copy=False))
                        inferred_action_log.append(inferred_action_np)
                        action_np = inferred_action_np
                        current_obs = _get_current_obs_vector(env, time_step, obs_keys)
                        obs_log.append(current_obs)
                        action_log.append(action_np)
                        time_step = env.step(action_np)
                        reward = 0.0 if time_step.reward is None else float(time_step.reward)
                        reward_log.append(reward)
                        total_reward += reward
                        t += 1
                        if time_step.last():
                            break
                elif args.open_loop:
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
                    inferred_action_log.extend([planned_actions_np[i] for i in range(spline_len)])
                    spline_action_log.extend([spline_actions_np[i] for i in range(spline_len)])

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
                    for i in range(spline_len):
                        current_obs = _get_current_obs_vector(env, time_step, obs_keys)
                        obs_log.append(current_obs)
                        action_np = spline_actions_np[i].astype(np.float32, copy=False)
                        spline_action_log.append(action_np)
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
                    if include_current_trajectory:
                        prefix_obs = torch.from_numpy(np.asarray(prefix_obs_log, dtype=np.float32)).reshape(-1, current_obs.shape[0])
                        prefix_actions = torch.from_numpy(np.asarray(prefix_action_log, dtype=np.float32)).reshape(-1, action_dim)
                        prefix_rewards = torch.from_numpy(np.asarray(prefix_reward_log, dtype=np.float32)).reshape(-1, 1)
                        prefix_rewards = prefix_rewards.expand(-1, reward_dim) if reward_dim > 1 else prefix_rewards
                        demo_obs_i, demo_actions_i, demo_rewards_i, demo_lengths_i = _append_prefix_to_demo(
                            demo_obs=demo_obs,
                            demo_actions=demo_actions,
                            demo_rewards=demo_rewards,
                            demo_lengths=demo_lengths,
                            prefix_obs=prefix_obs,
                            prefix_actions=prefix_actions,
                            prefix_rewards=prefix_rewards,
                        )
                    else:
                        demo_obs_i, demo_actions_i, demo_rewards_i, demo_lengths_i = (
                            demo_obs,
                            demo_actions,
                            demo_rewards,
                            demo_lengths,
                        )
                    action = model.act(
                        demo_obs=demo_obs_i,
                        demo_actions=demo_actions_i,
                        demo_rewards=demo_rewards_i,
                        demo_lengths=demo_lengths_i,
                        current_obs=current_obs_tensor,
                    )
                    action_np = action.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
                    action_log.append(action_np)

                    time_step = env.step(action_np)
                    reward = 0.0 if time_step.reward is None else float(time_step.reward)
                    reward_log.append(reward)
                    total_reward += reward
                    t += 1

                    if include_current_trajectory:
                        prefix_obs_log.append(current_obs)
                        prefix_action_log.append(action_np)
                        prefix_reward_log.append(reward)

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
                "action_source": (
                    "spline_playback"
                    if (args.open_loop_action_playback and use_spline_mode)
                    else "model"
                ),
                "open_loop_action_playback": bool(args.open_loop_action_playback),
            }
            if args.open_loop and inferred_action_log and spline_action_log:
                compare_len = min(len(inferred_action_log), len(spline_action_log))
                rollout["open_loop_inferred_actions"] = np.stack(inferred_action_log[:compare_len], axis=0)
                rollout["open_loop_spline_actions"] = np.stack(spline_action_log[:compare_len], axis=0)
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
    plot_dir, plot_images = _render_rollout_plots(
        rollouts_path=out_path,
        max_rollouts=args.max_plot_rollouts,
    )
    action_plot_dir, action_plot_images = _render_open_loop_action_comparison_plots(
        rollouts=all_rollouts,
        rollouts_path=out_path,
        max_rollouts=args.max_plot_rollouts,
    )

    mean_return = float(np.mean([ep["total_reward"] for ep in all_rollouts])) if all_rollouts else 0.0
    if wandb_run is not None:
        wandb_run.log({"eval/num_rollouts": len(all_rollouts), "eval/save_rollouts_path": str(out_path)})
        if plot_dir is not None:
            wandb_run.log({"eval/plots_dir": str(plot_dir)})
        if action_plot_dir is not None:
            wandb_run.log({"eval/open_loop_action_plots_dir": str(action_plot_dir)})
        if plot_images:
            selected_images = plot_images
            if selected_images:
                try:
                    import wandb  # type: ignore
                except ImportError:
                    wandb = None
                if wandb is not None:
                    wandb_api: Any = wandb
                    wandb_run.log(
                        {
                            "eval/rollout_plots": [
                                wandb_api.Image(str(image_path), caption=image_path.name)
                                for image_path in selected_images
                            ]
                        }
                    )
        if action_plot_images:
            try:
                import wandb  # type: ignore
            except ImportError:
                wandb = None
            if wandb is not None:
                wandb_api_actions: Any = wandb
                wandb_run.log(
                    {
                        "eval/open_loop_action_plots": [
                            wandb_api_actions.Image(str(image_path), caption=image_path.name)
                            for image_path in action_plot_images
                        ]
                    }
                )
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

