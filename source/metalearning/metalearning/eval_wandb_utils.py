# Copyright (c) 2024-2025, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for wandb logging during demo-tracking eval (reward, tracking error)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import torch

# Name of the command term that provides demo tracking metrics (joint_tracking_error, ee_tracking_error)
TRACKING_METRICS_COMMAND_NAMES = ("tracking_command", "tracking_metrics_command")
TASK_METRICS_COMMAND_NAME = "task_command"


def get_tracking_metrics_term(manager_env: Any) -> Any:
    """Return the DemoTrackingMetricsCommand term if present, else None."""
    if not hasattr(manager_env, "command_manager"):
        return None
    for term_name in TRACKING_METRICS_COMMAND_NAMES:
        try:
            return manager_env.command_manager.get_term(term_name)
        except Exception:
            continue
    return None


def get_task_metrics_term(manager_env: Any) -> Any:
    """Return the task command term if present, else None."""
    if not hasattr(manager_env, "command_manager"):
        return None
    try:
        return manager_env.command_manager.get_term(TASK_METRICS_COMMAND_NAME)
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

    def log_step(self, tracking_term: Any, total_env_steps: int, task_term: Any | None = None) -> None:
        """Log tracking and task metrics at the current step (mean over envs)."""
        if self._run is None:
            return
        payload: dict[str, float] = {}
        if tracking_term is not None and hasattr(tracking_term, "metrics"):
            tracking_metrics = tracking_term.metrics
            if "joint_tracking_error" in tracking_metrics:
                payload["eval/joint_tracking_error"] = tracking_metrics["joint_tracking_error"].mean().item()
            if "ee_position_tracking_error" in tracking_metrics:
                payload["eval/ee_tracking_error"] = tracking_metrics["ee_position_tracking_error"].mean().item()
            if "ee_orientation_tracking_error" in tracking_metrics:
                payload["eval/ee_orientation_tracking_error"] = tracking_metrics[
                    "ee_orientation_tracking_error"
                ].mean().item()
            if "demo_success_match_rate" in tracking_metrics:
                payload["eval/demo_success_match_rate"] = tracking_metrics["demo_success_match_rate"].mean().item()
        if task_term is not None and hasattr(task_term, "metrics"):
            task_metrics = task_term.metrics
            if "average_rot_align_error" in task_metrics:
                payload["eval/task_average_rot_align_error"] = task_metrics["average_rot_align_error"].mean().item()
            if "average_pos_align_error" in task_metrics:
                payload["eval/task_average_pos_align_error"] = task_metrics["average_pos_align_error"].mean().item()
            if "end_of_episode_rot_align_error" in task_metrics:
                payload["eval/task_end_of_episode_rot_align_error"] = task_metrics[
                    "end_of_episode_rot_align_error"
                ].mean().item()
            if "end_of_episode_pos_align_error" in task_metrics:
                payload["eval/task_end_of_episode_pos_align_error"] = task_metrics[
                    "end_of_episode_pos_align_error"
                ].mean().item()
            if "end_of_episode_success_rate" in task_metrics:
                payload["eval/task_end_of_episode_success_rate"] = task_metrics[
                    "end_of_episode_success_rate"
                ].mean().item()
        if payload:
            self._run.log(payload, step=total_env_steps)

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
