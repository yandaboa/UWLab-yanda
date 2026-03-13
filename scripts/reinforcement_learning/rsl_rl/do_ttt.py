#!/usr/bin/env python3
# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run test-time training for a supervised MLP policy inside Isaac Lab.

Flow:
    load supervised policy ->
    run one SYNCHRONIZED round (all envs complete exactly one episode each) ->
    collect successful prefixes from all envs in the round ->
    if any successes occurred, do TTT on the collected prefixes ->
    repeat for max_num_episodes rounds

TTT fires at most once per round, after ALL environments have finished their
current episode. This ensures:
  - The policy is never updated mid-episode.
  - All environments always run under the same policy version.
  - Round boundaries are well-defined and reproducible.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Run test-time training for a supervised MLP policy.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during rollout.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time if possible.")

parser.add_argument(
    "--supervised_checkpoint",
    type=str,
    required=True,
    help="Path to a supervised MLP checkpoint trained by the supervised trainer.",
)
parser.add_argument(
    "--success_reward_threshold",
    type=float,
    default=0.1,
    help="A step is marked successful if reward > this threshold.",
)
parser.add_argument(
    "--max_num_episodes",
    type=int,
    default=10,
    help="Number of synchronized rounds (episodes) to run.",
)
parser.add_argument(
    "--ttt_learning_rate",
    type=float,
    default=1e-4,
    help="Learning rate for TTT.",
)
parser.add_argument(
    "--ttt_num_epochs",
    type=int,
    default=5,
    help="Number of optimization epochs for each TTT update.",
)
parser.add_argument(
    "--ttt_batch_size",
    type=int,
    default=1024,
    help="Batch size for each TTT update.",
)
parser.add_argument(
    "--ttt_weight_decay",
    type=float,
    default=0.0,
    help="Weight decay for TTT.",
)
parser.add_argument(
    "--ttt_save_path",
    type=str,
    default=None,
    help="Where to save the final adapted supervised checkpoint. Defaults to <env_log_dir>/ttt_adapted_supervised_model.pt",
)
parser.add_argument(
    "--wandb_project",
    type=str,
    default=None,
    help="Weights & Biases project name. If not set, wandb logging is disabled.",
)
parser.add_argument(
    "--wandb_run_name",
    type=str,
    default=None,
    help="Weights & Biases run name. Defaults to wandb auto-generated name.",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Rest of imports after app launch
# -----------------------------------------------------------------------------

import gymnasium as gym
import wandb

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
try:
    from uwlab_rl.rsl_rl.supervised_mlp_policy import SupervisedMLPPolicy, load_supervised_policy_checkpoint
except ModuleNotFoundError:
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    _source_root = os.path.join(_repo_root, "source", "uwlab_rl")
    if _source_root not in sys.path:
        sys.path.append(_source_root)
    from uwlab_rl.rsl_rl.supervised_mlp_policy import SupervisedMLPPolicy, load_supervised_policy_checkpoint
from uwlab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


def get_module_device(model: SupervisedMLPPolicy) -> torch.device:
    return next(model.parameters()).device


def supervised_policy_action(model: SupervisedMLPPolicy, obs, stochastic: bool = False) -> torch.Tensor:
    """Run the supervised policy and return actions in environment scale."""
    device = get_module_device(model)
    states = _extract_policy_state(obs).to(device)
    use_stochastic_sampling = model.loss_type == "gaussian_nll" and stochastic
    return model.act(states, stochastic=use_stochastic_sampling)


def do_ttt(
    model: SupervisedMLPPolicy,
    states: torch.Tensor,
    actions: torch.Tensor,
    num_epochs: int,
    learning_rate: float,
    batch_size: int,
    weight_decay: float,
) -> dict[str, float]:
    """Fine-tune the supervised policy on newly collected successful state-action prefixes."""
    if states.ndim != 2:
        raise ValueError(f"Expected states shape [N, state_dim], got {tuple(states.shape)}")
    if actions.ndim != 2:
        raise ValueError(f"Expected actions shape [N, action_dim], got {tuple(actions.shape)}")
    if states.shape[0] != actions.shape[0]:
        raise ValueError(f"State/action sample mismatch: {states.shape[0]} vs {actions.shape[0]}")
    if states.shape[0] == 0:
        raise ValueError("Cannot run TTT with zero samples.")
    if num_epochs <= 0:
        raise ValueError(f"num_epochs must be > 0, got {num_epochs}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    device = get_module_device(model)
    dataset = TensorDataset(states.float(), actions.float())
    effective_batch_size = min(batch_size, len(dataset))
    data_loader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train()
    final_loss = float("nan")

    pbar = tqdm(range(num_epochs), desc="TTT", unit="epoch", leave=False)
    for epoch_idx in pbar:
        total_loss = 0.0
        total_count = 0

        for batch_states, batch_actions in data_loader:
            batch_states = batch_states.to(device, non_blocking=True)
            batch_actions = batch_actions.to(device, non_blocking=True)

            loss = model.compute_loss(batch_states, batch_actions)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite TTT loss detected at epoch {epoch_idx + 1}: {loss.item()}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_n = batch_states.shape[0]
            total_loss += float(loss.item()) * batch_n
            total_count += batch_n

        final_loss = total_loss / max(1, total_count)
        pbar.set_postfix(loss=f"{final_loss:.6f}")

    model.eval()
    return {
        "num_samples": float(states.shape[0]),
        "final_loss": float(final_loss),
    }


# -----------------------------------------------------------------------------
# Observation helpers
# -----------------------------------------------------------------------------

def _flatten_observation_value(value) -> torch.Tensor:
    """Flatten a tensor-like observation structure into [num_envs, flat_dim]."""
    if isinstance(value, torch.Tensor):
        return value.reshape(value.shape[0], -1)

    if isinstance(value, Mapping) or (hasattr(value, "keys") and hasattr(value, "__getitem__")):
        flat_chunks = [_flatten_observation_value(value[key]) for key in value.keys()]
        if not flat_chunks:
            raise ValueError("Observation mapping is empty and cannot be flattened.")
        return torch.cat(flat_chunks, dim=-1)

    tensor_value = torch.as_tensor(value)
    return tensor_value.reshape(tensor_value.shape[0], -1)


def _extract_policy_state(obs) -> torch.Tensor:
    """Extract and flatten policy observations into [num_envs, state_dim]."""
    if isinstance(obs, Mapping) or (hasattr(obs, "keys") and hasattr(obs, "__getitem__")):
        if "policy" in obs:
            state = obs["policy"]
        else:
            first_key = next(iter(obs))
            state = obs[first_key]
    else:
        state = obs
    return _flatten_observation_value(state)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Run synchronized test-time training with a supervised MLP policy.

    Each round collects exactly one episode per environment, then optionally
    runs TTT on the successful prefixes from that round before the next round
    begins. The policy is never updated mid-episode.
    """
    if args_cli.max_num_episodes <= 0:
        raise ValueError(f"--max_num_episodes must be > 0, got {args_cli.max_num_episodes}")
    if args_cli.ttt_num_epochs <= 0:
        raise ValueError(f"--ttt_num_epochs must be > 0, got {args_cli.ttt_num_epochs}")
    if args_cli.ttt_batch_size <= 0:
        raise ValueError(f"--ttt_batch_size must be > 0, got {args_cli.ttt_batch_size}")

    # Override cfg with CLI
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Logging dir
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    os.makedirs(log_root_path, exist_ok=True)
    print(f"[INFO] Using log directory: {log_root_path}")
    env_cfg.log_dir = log_root_path

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, "videos", "ttt"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video during rollout.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Initial observations / env dims
    obs = env.get_observations()
    num_envs = env.num_envs
    state_dim = _extract_policy_state(obs).shape[1]
    action_dim = env.num_actions
    device = env.unwrapped.device

    # Load supervised checkpoint
    supervised_ckpt_path = os.path.abspath(args_cli.supervised_checkpoint)
    print(f"[INFO] Loading supervised checkpoint from: {supervised_ckpt_path}")
    model, source_checkpoint = load_supervised_policy_checkpoint(supervised_ckpt_path, device=device)

    if model.state_dim != state_dim:
        raise ValueError(
            f"Supervised checkpoint expects state_dim={model.state_dim}, but env policy obs dim is {state_dim}."
        )
    if model.action_dim != action_dim:
        raise ValueError(
            f"Supervised checkpoint expects action_dim={model.action_dim}, but env action dim is {action_dim}."
        )

    max_episode_steps = getattr(env.unwrapped, "max_episode_length", None)
    if max_episode_steps is None:
        raise RuntimeError("This script requires env.unwrapped.max_episode_length.")
    max_episode_steps = int(max_episode_steps)

    dt = env.unwrapped.step_dt

    # ------------------------------------------------------------------
    # Weights & Biases
    # ------------------------------------------------------------------
    use_wandb = args_cli.wandb_project is not None
    if use_wandb:
        # Serialize configs to plain dicts for wandb. dataclasses/OmegaConf
        # objects are converted via their __dict__ if available.
        def _cfg_to_dict(cfg) -> dict:
            if hasattr(cfg, "__dict__"):
                return {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
            return {}

        wandb.init(
            project=args_cli.wandb_project,
            name=args_cli.wandb_run_name,
            config={
                "task": args_cli.task,
                "num_envs": num_envs,
                "supervised_checkpoint": supervised_ckpt_path,
                "max_num_episodes": args_cli.max_num_episodes,
                "success_reward_threshold": args_cli.success_reward_threshold,
                "ttt_learning_rate": args_cli.ttt_learning_rate,
                "ttt_num_epochs": args_cli.ttt_num_epochs,
                "ttt_batch_size": args_cli.ttt_batch_size,
                "ttt_weight_decay": args_cli.ttt_weight_decay,
                "model": {
                    "state_dim": model.state_dim,
                    "action_dim": model.action_dim,
                    "hidden_dims": model.hidden_dims,
                    "loss_type": model.loss_type,
                    "log_std_min": model.log_std_min,
                    "log_std_max": model.log_std_max,
                },
                "env_cfg": _cfg_to_dict(env_cfg),
                "agent_cfg": _cfg_to_dict(agent_cfg),
            },
        )
        print(f"[INFO] wandb run initialized: {wandb.run.url}")
    else:
        print("[INFO] wandb logging disabled (pass --wandb_project to enable).")

    total_successful_episodes = 0
    history: list[dict[str, Any]] = []

    video_timestep = 0

    # ------------------------------------------------------------------
    # Outer loop: one iteration = one synchronized round
    # All num_envs environments run in parallel until every env has
    # emitted done=True exactly once, then TTT fires (if any successes).
    # ------------------------------------------------------------------
    rounds_pbar = tqdm(range(args_cli.max_num_episodes), desc="Rounds", unit="round")

    for round_idx in rounds_pbar:
        if not simulation_app.is_running():
            break

        # Per-env buffers for this round.
        # We guard against the rare case where the env fires done one step
        # late by clamping step_ids to max_episode_steps - 1.
        episode_lengths = torch.zeros(num_envs, device=device, dtype=torch.long)
        first_success_step = torch.full((num_envs,), fill_value=-1, device=device, dtype=torch.long)
        episode_done = torch.zeros(num_envs, device=device, dtype=torch.bool)

        episode_states = torch.empty((num_envs, max_episode_steps, state_dim), device=device, dtype=torch.float32)
        episode_actions = torch.empty((num_envs, max_episode_steps, action_dim), device=device, dtype=torch.float32)

        round_successful_state_chunks: list[torch.Tensor] = []
        round_successful_action_chunks: list[torch.Tensor] = []

        # ------------------------------------------------------------------
        # Inner loop: step until every env has finished its episode.
        # ------------------------------------------------------------------
        while not episode_done.all():
            if not simulation_app.is_running():
                break

            start_time = time.time()

            with torch.inference_mode():
                state_t = _extract_policy_state(obs).detach()
                actions = supervised_policy_action(model, obs, stochastic=True)
                flat_actions = actions.reshape(actions.shape[0], -1).detach()

                obs, rewards, dones, _ = env.step(actions)

                rewards = rewards.detach()
                dones = dones.detach()

                # Only record steps for envs that haven't finished yet this round.
                # Once an env emits done=True its episode is over; ignore
                # subsequent steps from the auto-reset until the next round.
                active = ~episode_done

                if active.any():
                    # .squeeze(-1) collapses a [1] tensor to a scalar, breaking
                    # indexing when num_envs == 1. Use .reshape(-1) instead.
                    active_ids = active.nonzero(as_tuple=False).reshape(-1)

                    # Clamp step index to avoid out-of-bounds writes if the
                    # environment fires done one step later than expected.
                    step_ids = episode_lengths[active_ids].clamp(max=max_episode_steps - 1)

                    episode_states[active_ids, step_ids] = state_t[active_ids]
                    episode_actions[active_ids, step_ids] = flat_actions[active_ids]

                    # Mark first success (reward threshold check on active envs only).
                    new_success_mask = active & (first_success_step < 0) & (rewards > args_cli.success_reward_threshold)
                    if new_success_mask.any():
                        success_ids = new_success_mask.nonzero(as_tuple=False).reshape(-1)
                        first_success_step[success_ids] = episode_lengths[success_ids].clamp(max=max_episode_steps - 1)

                    episode_lengths[active_ids] += 1

                # Mark envs whose episode just ended (first done signal only).
                newly_done = dones & ~episode_done
                episode_done |= newly_done

            if args_cli.video:
                video_timestep += 1
                if video_timestep >= args_cli.video_length:
                    break

            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

        # ------------------------------------------------------------------
        # Round complete: all envs have finished their episode.
        # Collect successful prefixes, then do one TTT update if any exist.
        # ------------------------------------------------------------------
        for env_id in range(num_envs):
            success_step = int(first_success_step[env_id].item())
            if success_step >= 0:
                prefix_len = success_step + 1
                round_successful_state_chunks.append(episode_states[env_id, :prefix_len].detach().cpu())
                round_successful_action_chunks.append(episode_actions[env_id, :prefix_len].detach().cpu())
                total_successful_episodes += 1

        num_successes_this_round = len(round_successful_state_chunks)
        rounds_pbar.set_postfix(successes=num_successes_this_round)

        if round_successful_state_chunks:
            ttt_states = torch.cat(round_successful_state_chunks, dim=0)
            ttt_actions = torch.cat(round_successful_action_chunks, dim=0)

            print(
                f"[INFO] Round {round_idx + 1}/{args_cli.max_num_episodes}: "
                f"TTT on {num_successes_this_round}/{num_envs} successful envs, "
                f"{ttt_states.shape[0]} samples."
            )

            ttt_stats = do_ttt(
                model=model,
                states=ttt_states,
                actions=ttt_actions,
                num_epochs=args_cli.ttt_num_epochs,
                learning_rate=args_cli.ttt_learning_rate,
                batch_size=args_cli.ttt_batch_size,
                weight_decay=args_cli.ttt_weight_decay,
            )

            print(
                f"[INFO] Round {round_idx + 1} TTT done: "
                f"final_loss={ttt_stats['final_loss']:.6f}"
            )
        else:
            ttt_stats = None
            print(
                f"[INFO] Round {round_idx + 1}/{args_cli.max_num_episodes}: "
                f"no successes — skipping TTT."
            )

        history.append(
            {
                "round": round_idx + 1,
                "successful_episodes_in_round": num_successes_this_round,
                "samples_in_round": int(ttt_states.shape[0]) if round_successful_state_chunks else 0,
                "final_loss": float(ttt_stats["final_loss"]) if ttt_stats is not None else None,
                "total_successful_episodes": int(total_successful_episodes),
            }
        )

        if use_wandb:
            wandb_log: dict[str, Any] = {
                "round": round_idx + 1,
                "success_rate": num_successes_this_round / num_envs,
                "successful_envs": num_successes_this_round,
                "ttt_samples": int(ttt_states.shape[0]) if round_successful_state_chunks else 0,
                "total_successful_episodes": int(total_successful_episodes),
            }
            if ttt_stats is not None:
                wandb_log["ttt/final_loss"] = float(ttt_stats["final_loss"])
                wandb_log["ttt/num_samples"] = float(ttt_stats["num_samples"])
            wandb.log(wandb_log, step=round_idx + 1)

    rounds_pbar.close()

    # Save final adapted checkpoint
    save_path = args_cli.ttt_save_path
    if save_path is None:
        save_path = os.path.join(log_root_path, "ttt_adapted_supervised_model.pt")
    save_path = os.path.abspath(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "state_dim": model.state_dim,
            "action_dim": model.action_dim,
            "hidden_dims": model.hidden_dims,
            "loss_type": model.loss_type,
            "log_std_min": model.log_std_min,
            "log_std_max": model.log_std_max,
            "action_norm_mean": model.action_norm_mean.detach().cpu(),
            "action_norm_std": model.action_norm_std.detach().cpu(),
            "source_checkpoint": supervised_ckpt_path,
            "source_checkpoint_epoch": source_checkpoint.get("epoch"),
            "max_num_episodes": int(args_cli.max_num_episodes),
            "success_reward_threshold": float(args_cli.success_reward_threshold),
            "ttt_num_epochs": int(args_cli.ttt_num_epochs),
            "ttt_learning_rate": float(args_cli.ttt_learning_rate),
            "ttt_batch_size": int(args_cli.ttt_batch_size),
            "ttt_weight_decay": float(args_cli.ttt_weight_decay),
            "num_envs": int(num_envs),
            "total_successful_episodes": int(total_successful_episodes),
            "ttt_rounds": int(len(history)),
            "history": history,
        },
        save_path,
    )

    history_path = os.path.join(os.path.dirname(save_path), "ttt_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_checkpoint": supervised_ckpt_path,
                "save_path": save_path,
                "num_envs": int(num_envs),
                "max_num_episodes": int(args_cli.max_num_episodes),
                "total_successful_episodes": int(total_successful_episodes),
                "ttt_rounds": int(len(history)),
                "history": history,
            },
            f,
            indent=2,
        )

    print(f"[INFO] Saved adapted supervised model to: {save_path}")
    print(f"[INFO] Saved TTT history to: {history_path}")
    print(f"[INFO] Total successful episodes used for TTT: {total_successful_episodes}")

    if use_wandb:
        wandb.finish()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()