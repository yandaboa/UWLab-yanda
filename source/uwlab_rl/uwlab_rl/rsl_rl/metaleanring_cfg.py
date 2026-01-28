from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg


@configclass
class BCFromContextWarmStartCfg:
    """Configuration for context-conditioned behavior cloning warm start."""

    num_steps: int = 200
    """Number of optimizer steps to run."""

    num_episodes_per_batch: int = 8
    """Number of demo episodes sampled per step."""

    num_minibatches: int = 4
    """Number of minibatches per step."""

    minibatch_size: int | None = None
    """Minibatch size (defaults to batch_size // num_minibatches)."""

    learning_rate: float = 3.0e-4
    """Learning rate for warm-start optimizer."""

    lr_warmup_steps: int = 0
    """Number of warmup steps for the learning rate."""

    lr_warmup_start_ratio: float = 0.1
    """Warmup start ratio relative to learning_rate."""

    weight_decay: float = 0.0
    """Weight decay for warm-start optimizer."""

    betas: tuple[float, float] = (0.9, 0.99)
    """Optimizer betas."""

    eps: float = 1.0e-8
    """Optimizer epsilon."""

    max_grad_norm: float = 1.0
    """Maximum gradient norm for warm-start updates."""

    optimizer_class: str = "AdamW"
    """Optimizer class name."""

    use_amp: bool = True
    """Whether to enable AMP when CUDA is available."""


@configclass
class RslRlPpoAlgorithmWarmStartCfg(RslRlPpoAlgorithmCfg):
    """PPO configuration with optional BC warm start."""

    bc_warmstart_cfg: BCFromContextWarmStartCfg | None = None
    """Optional context-conditioned BC warm-start configuration."""
