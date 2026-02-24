# Copyright (c) 2024-2025, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for wandb logging during demo-tracking eval (reward, tracking error)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import torch

# Name of the command term that provides demo tracking metrics (joint_tracking_error, ee_tracking_error)
TRACKING_METRICS_COMMAND_NAME = "tracking_metrics_command"


def get_tracking_metrics_term(manager_env: Any) -> Any:
    """Return the DemoTrackingMetricsCommand term if present, else None."""
    if not hasattr(manager_env, "command_manager"):
        return None
    try:
        return manager_env.command_manager.get_term(TRACKING_METRICS_COMMAND_NAME)
    except Exception:
        return None


def init_eval_wandb(use_wandb: bool, agent_cfg: Any, log_dir: str, task_name: str) -> tuple[Any, Any]:
    """Initialize wandb for eval logging when requested. Returns (wandb_run, wandb_module) or (None, None)."""
    if not use_wandb:
        return None, None
    try:
        import wandb  # type: ignore
    except ImportError:
        print("[INFO] Wandb requested but not installed; skipping eval logging.")
        return None, None
    project = getattr(agent_cfg, "experiment_name", None) or "eval_demo_tracking"
    run_name = getattr(agent_cfg, "run_name", None) or datetime.now().strftime("%Y%m%d_%H%M%S")
    run = wandb.init(project=project, name=run_name, dir=log_dir, config={"task": task_name})
    return run, wandb


class EvalWandbLogger:
    """Logs eval metrics to wandb: per-step tracking error, and on episode end: return and length.
    All logs use total environment steps as the wandb step.
    """

    def __init__(self, wandb_run: Any) -> None:
        self._run = wandb_run

    def log_step(self, tracking_term: Any, total_env_steps: int) -> None:
        """Log tracking error at the current step (mean over envs). Call after env.step()."""
        if self._run is None:
            return
        joint = tracking_term.metrics["joint_tracking_error"].mean().item()
        ee = tracking_term.metrics["ee_position_tracking_error"].mean().item()
        ee_rot = tracking_term.metrics["ee_orientation_tracking_error"].mean().item()
        self._run.log(
            {
                "eval/joint_tracking_error": joint,
                "eval/ee_tracking_error": ee,
                "eval/ee_orientation_tracking_error": ee_rot,
            },
            step=total_env_steps,
        )

    def log_episode_batch(
        self,
        done_env_ids: torch.Tensor,
        rollouts: dict[str, Any],
        total_env_steps: int,
    ) -> None:
        """Log mean return and length for completed episodes."""
        if self._run is None:
            return
        rewards_batch = rollouts["rewards"]
        episode_returns = rewards_batch.sum(dim=1).detach().cpu().tolist()
        n_done = done_env_ids.numel()
        lengths_tensor = rollouts["lengths"]
        self._run.log(
            {
                "episode/avg_total_return": sum(episode_returns) / n_done if n_done else 0.0,
                "episode/avg_length": lengths_tensor.float().mean().item(),
            },
            step=total_env_steps,
        )

    def finish(self) -> None:
        """Close the wandb run."""
        if self._run is not None:
            self._run.finish()
            self._run = None
