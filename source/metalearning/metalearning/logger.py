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
