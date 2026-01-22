"""Wandb logging helpers for metalearning."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


class WandbNoiseLogger:
    """Wandb logger for environment noise statistics."""

    def __init__(
        self,
        enable: bool,
        agent_cfg: Mapping[str, Any] | Any,
        log_dir: str,
        task_name: str,
        log_interval: int = 100,
        project_name: str | None = None,
        run_name: str | None = None,
    ) -> None:
        self._log_interval = log_interval
        self._wandb = None
        self._run = None
        if not enable:
            return
        try:
            import wandb  # type: ignore
        except ImportError:
            print("[INFO] Wandb requested but not installed.")
            return
        self._wandb = wandb
        project, run = self._resolve_names(agent_cfg, project_name, run_name)
        self._run = wandb.init(project=project, name=run, dir=log_dir, config={"task": task_name})

    def log(self, environment_noise: Any, step: int) -> None:
        """Log noise statistics when due."""
        if self._run is None or self._wandb is None or environment_noise is None:
            return
        if step % self._log_interval != 0:
            return
        stats = environment_noise.get_statistics()
        log_payload = {
            "noise/avg_noise_applications_per_reset": stats["avg_noise_applications_per_reset"],
            "noise/avg_noise_magnitude": stats["avg_noise_magnitude"],
            "noise/noise_apply_rate": stats["noise_apply_rate"],
            "noise/current_frequency_mean": stats["current_noise_frequency_mean"],
            "noise/current_frequency_std": stats["current_noise_frequency_std"],
        }
        if stats["resample_frequency_history"]:
            fig = self._make_histogram_figure(
                stats["resample_frequency_history"],
                "Resample Frequency",
                "Resample Frequency",
                stats["current_noise_frequency_mean"],
                stats["current_noise_frequency_std"],
            )
            log_payload["noise/resample_frequency_plot"] = self._wandb.Image(fig)
            plt.close(fig)
        if stats["noise_magnitude_history"]:
            fig = self._make_histogram_figure(
                stats["noise_magnitude_history"],
                "Noise Magnitude",
                "Noise Magnitude",
                stats["avg_noise_magnitude"],
                float(np.std(stats["noise_magnitude_history"])),
            )
            log_payload["noise/noise_magnitude_plot"] = self._wandb.Image(fig)
            plt.close(fig)
        if stats["noise_application_count_history"]:
            fig = self._make_histogram_figure(
                stats["noise_application_count_history"],
                "Noise Applications Per Reset",
                "Noise Applications",
                stats["avg_noise_applications_per_reset"],
                float(np.std(stats["noise_application_count_history"])),
            )
            log_payload["noise/noise_application_count_plot"] = self._wandb.Image(fig)
            plt.close(fig)
        self._run.log(log_payload, step=step)

    def finish(self) -> None:
        """Close the wandb run."""
        if self._run is None:
            return
        self._run.finish()
        self._run = None

    @property
    def run(self):
        return self._run

    @property
    def wandb(self):
        return self._wandb

    def _resolve_names(
        self,
        agent_cfg: Mapping[str, Any] | Any,
        project_name: str | None,
        run_name: str | None,
    ) -> tuple[str | None, str | None]:
        if isinstance(agent_cfg, Mapping):
            project = project_name or agent_cfg.get("wandb_project") or agent_cfg.get("experiment_name")
            run = run_name or agent_cfg.get("run_name")
            return project, run
        project = project_name or getattr(agent_cfg, "wandb_project", None) or getattr(agent_cfg, "experiment_name", None)
        run = run_name or getattr(agent_cfg, "run_name", None)
        return project, run

    def _make_histogram_figure(
        self,
        values: Sequence[float],
        title: str,
        xlabel: str,
        mean_value: float,
        std_value: float,
    ):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(values, bins=30, alpha=0.8, color="#4C78A8", edgecolor="white")
        ax.axvline(mean_value, color="#E45756", linestyle="--", linewidth=2, label=f"mean={mean_value:.3f}")
        ax.axvline(
            mean_value + std_value, color="#72B7B2", linestyle=":", linewidth=2, label=f"+1 std={std_value:.3f}"
        )
        ax.axvline(mean_value - std_value, color="#72B7B2", linestyle=":", linewidth=2, label="-1 std")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.legend(loc="best")
        fig.tight_layout()
        return fig


class WandbEpisodeLogger:
    """Wandb logger for episodic success and return statistics."""

    def __init__(
        self,
        enable: bool,
        agent_cfg: Mapping[str, Any] | Any,
        log_dir: str,
        task_name: str,
        log_interval: int = 100,
        project_name: str | None = None,
        run_name: str | None = None,
        run=None,
        wandb_module=None,
    ) -> None:
        self._log_interval = log_interval
        self._wandb = wandb_module
        self._run = run
        self._episode_returns: list[float] = []
        self._episode_success: list[float] = []
        self._track_success = False
        if not enable:
            return
        if self._run is not None:
            return
        try:
            import wandb  # type: ignore
        except ImportError:
            print("[INFO] Wandb requested but not installed.")
            return
        self._wandb = wandb
        project, run = self._resolve_names(agent_cfg, project_name, run_name)
        self._run = wandb.init(project=project, name=run, dir=log_dir, config={"task": task_name})

    def log(self, episode_returns: Sequence[float], episode_success: Sequence[float] | None, step: int) -> None:
        """Log episodic statistics when due."""
        if self._run is None:
            return
        if episode_returns:
            self._episode_returns.extend(float(value) for value in episode_returns)
        if episode_success is not None:
            self._track_success = True
            self._episode_success.extend(float(value) for value in episode_success)
        self._flush_batches(step)

    def flush(self, step: int) -> None:
        """Log any remaining episodic statistics."""
        if self._run is None:
            return
        if not self._episode_returns:
            return
        self._log_batch(step, len(self._episode_returns))

    def _flush_batches(self, step: int) -> None:
        while len(self._episode_returns) >= self._log_interval:
            self._log_batch(step, self._log_interval)

    def _log_batch(self, step: int, batch_size: int) -> None:
        if self._run is None:
            return
        run = self._run
        returns = self._episode_returns[:batch_size]
        del self._episode_returns[:batch_size]
        log_payload = {"episode/avg_total_return": float(np.mean(returns))}
        if self._track_success and len(self._episode_success) >= batch_size:
            successes = self._episode_success[:batch_size]
            del self._episode_success[:batch_size]
            log_payload["episode/task_success_rate"] = float(np.mean(successes))
        run.log(log_payload, step=step)

    def _resolve_names(
        self,
        agent_cfg: Mapping[str, Any] | Any,
        project_name: str | None,
        run_name: str | None,
    ) -> tuple[str | None, str | None]:
        if isinstance(agent_cfg, Mapping):
            project = project_name or agent_cfg.get("wandb_project") or agent_cfg.get("experiment_name")
            run = run_name or agent_cfg.get("run_name")
            return project, run
        project = project_name or getattr(agent_cfg, "wandb_project", None) or getattr(agent_cfg, "experiment_name", None)
        run = run_name or getattr(agent_cfg, "run_name", None)
        return project, run
