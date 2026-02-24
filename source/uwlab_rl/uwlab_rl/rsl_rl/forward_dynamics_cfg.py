from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


def _update_dataclass_from_dict(target: object, values: dict) -> None:
    for key, value in values.items():
        if not hasattr(target, key):
            continue
        current = getattr(target, key)
        if isinstance(value, dict) and hasattr(current, "__dataclass_fields__"):
            _update_dataclass_from_dict(current, value)
        else:
            setattr(target, key, value)


@dataclass
class ForwardDynamicsDataCfg:
    """Dataset configuration for forward dynamics training."""

    train_episode_paths: list[str] | None = field(
        default_factory=lambda: [
            "episodes/20260222_155251/episodes_000001_trim.pt",
            "episodes/20260222_155251/episodes_000002_trim.pt",
            "episodes/20260222_155251/episodes_000003_trim.pt",
            "episodes/20260222_155251/episodes_000004_trim.pt",
            "episodes/20260222_155251/episodes_000005_trim.pt",
            "episodes/20260222_155251/episodes_000006_trim.pt",
            "episodes/20260222_155251/episodes_000007_trim.pt",
            "episodes/20260222_155251/episodes_000008_trim.pt",
            "episodes/20260222_155251/episodes_000009_trim.pt",
        ]
    )
    """List of episode .pt files or glob patterns for training."""

    validation_episode_paths: list[str] | None = field(
        default_factory=lambda: [
            "episodes/20260222_155251/episodes_000000_trim_subsetx.pt",
        ]
    )
    """Optional list of episode .pt files or glob patterns for validation."""

    obs_keys: list[str] | None = field(
        default_factory=lambda: [
            "joint_pos",
            "end_effector_pose",
            "insertive_asset_pose",
            "receptive_asset_pose",
            "insertive_asset_in_receptive_asset_frame",
        ]
    )
    """Optional ordered obs keys for dict observations."""

    batch_size: int = 1024
    num_workers: int = 4
    shuffle: bool = True


@dataclass
class ForwardDynamicsModelCfg:
    """MLP architecture configuration for forward residual dynamics."""

    hidden_dim: int = 512
    num_layers: int = 4
    activation: Literal["relu", "gelu", "tanh"] = "relu"
    dropout: float = 0.0
    normalize_state_action: bool = True
    norm_num_batches: int = 32
    norm_min_std: float = 1.0e-6


@dataclass
class ForwardDynamicsOptimizationCfg:
    """Optimization configuration for forward dynamics training."""

    num_steps: int = 150000
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    betas: tuple[float, float] = (0.9, 0.99)
    eps: float = 1.0e-8
    max_grad_norm: float = 1.0
    optimizer_class: str = "AdamW"
    use_amp: bool = True


@dataclass
class ForwardDynamicsLoggingCfg:
    """Logging and checkpoint configuration."""

    experiment_name: str = "forward_dynamics"
    run_name: str = ""
    log_interval: int = 1
    save_interval: int = 5000
    val_interval: int = 1000
    log_project_name: str | None = None
    use_wandb: bool = True


@dataclass
class ForwardDynamicsTrainerCfg:
    """Top-level forward dynamics trainer configuration."""

    data: ForwardDynamicsDataCfg = field(default_factory=ForwardDynamicsDataCfg)
    model: ForwardDynamicsModelCfg = field(default_factory=ForwardDynamicsModelCfg)
    optim: ForwardDynamicsOptimizationCfg = field(default_factory=ForwardDynamicsOptimizationCfg)
    logging: ForwardDynamicsLoggingCfg = field(default_factory=ForwardDynamicsLoggingCfg)

    def from_dict(self, values: dict) -> None:
        _update_dataclass_from_dict(self, values)
