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
class SupervisedContextDataCfg:
    """Dataset configuration for supervised context training."""

    train_episode_paths: list[str] | None = field(
        default_factory=lambda: [
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
            "episodes/20260222_155251/episodes_000000_trim.pt",
        ]
    )
    """Optional list of episode .pt files or glob patterns for validation."""

    episode_paths: list[str] = field(
        default_factory=lambda: [
            # "episodes/20260218_023144/episodes_000000.pt",
            # "episodes/20260218_023144/episodes_000001.pt",
            # "episodes/20260218_023144/episodes_000002.pt",
            # "episodes/20260218_023144/episodes_000003.pt",
            # "episodes/20260218_023144/episodes_000004.pt",
            # "episodes/20260218_023144/episodes_000005.pt",
            # "episodes/20260218_023144/episodes_000006.pt",    
            # "episodes/20260218_023144/episodes_000007.pt",
            # "episodes/20260218_023144/episodes_000008.pt",
            # "episodes/20260218_023144/episodes_000009.pt",
        ]
    )
    """Deprecated single dataset list (used when train paths are unset)."""

    obs_keys: list[str] | None = field(default_factory=lambda: ["joint_pos", "end_effector_pose", "insertive_asset_pose", "receptive_asset_pose", "insertive_asset_in_receptive_asset_frame"])
    """Optional ordered obs keys for dict observations."""

    max_context_length: int | None = None
    """Optional cap on context length per episode."""

    batch_size: int = 256
    """Batch size for training."""

    num_workers: int = 4
    """Data loader workers."""

    shuffle: bool = True
    """Shuffle episodes in the dataset."""


@dataclass
class SupervisedContextModelCfg:
    """Model configuration for supervised context training."""

    num_actions: int = 7  # type: ignore
    """Action dimension (always inferred from data or checkpoint state, overrides this value. This is just a placeholder)."""

    action_distribution: Literal["normal", "categorical"] = "normal"
    """Action distribution type."""

    action_discretization_spec_path: str = ""
    """Optional path to action discretization spec (defaults to episode folder)."""

    context_token_layout: str = "merged"
    """Token layout: merged, state_action, state_only."""

    include_actions_in_context: bool = False
    """Include action terms in merged context tokens."""

    include_rewards_in_context: bool = True
    """Include reward terms in merged context tokens."""

    share_current_and_context_obs_projection: bool = True
    """Reuse one projection for current_obs and context_obs; merged layout pads current_obs if needed."""

    encoding_projection_hidden_dim: int | None = None
    """Optional hidden size for obs encoders (in->hidden->embedding instead of single linear)."""

    embedding_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 4
    num_heads: int = 4
    embedding_dropout: float = 0.0
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0


@dataclass
class SupervisedContextOptimizationCfg:
    """Optimization configuration for supervised context training."""

    num_steps: int = 100000
    """Total optimizer steps."""

    learning_rate: float = 3.0e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.99)
    eps: float = 1.0e-8
    max_grad_norm: float = 1.0
    optimizer_class: str = "AdamW"
    lr_warmup_steps: int = 200
    lr_schedule: str | None = "cosine_annealing_with_warmup"

    use_amp: bool = True
    """Use automatic mixed precision when CUDA is available."""


@dataclass
class SupervisedContextInputCfg:
    """Input configuration for supervised context training."""

    include_current_trajectory: bool = False
    """Whether to append current rollout to context before prediction (disabled)."""


@dataclass
class SupervisedContextDistributedCfg:
    """Distributed configuration for supervised context training."""

    distributed: bool = False
    """Enable multi-GPU distributed training."""


@dataclass
class SupervisedContextLoggingCfg:
    """Logging configuration for supervised context training."""

    experiment_name: str = "supervised_context"
    run_name: str = ""
    log_interval: int = 1
    save_interval: int = 5000
    log_project_name: str | None = None
    use_wandb: bool = True
    val_interval: int = 1000
    """Validation cadence in steps (0 disables validation)."""


@dataclass
class SupervisedContextTrainerCfg:
    """Top-level configuration for supervised context training."""

    data: SupervisedContextDataCfg = field(default_factory=SupervisedContextDataCfg)
    model: SupervisedContextModelCfg = field(default_factory=SupervisedContextModelCfg)
    optim: SupervisedContextOptimizationCfg = field(default_factory=SupervisedContextOptimizationCfg)
    input: SupervisedContextInputCfg = field(default_factory=SupervisedContextInputCfg)
    distributed: SupervisedContextDistributedCfg = field(default_factory=SupervisedContextDistributedCfg)
    logging: SupervisedContextLoggingCfg = field(default_factory=SupervisedContextLoggingCfg)

    def from_dict(self, values: dict) -> None:
        _update_dataclass_from_dict(self, values)
