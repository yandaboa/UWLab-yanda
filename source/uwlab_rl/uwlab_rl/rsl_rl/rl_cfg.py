# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg  # noqa: F401


@configclass
class BehaviorCloningCfg:
    experts_path: list[str] = MISSING  # type: ignore
    """The path to the expert data."""

    experts_loader: callable = "torch.jit.load"
    """The function to construct the expert. Default is None, for which is loaded in the same way student is loaded."""

    experts_env_mapping_func: callable = None
    """The function to map the expert to env_ids. Default is None, for which is mapped to all env_ids"""

    experts_observation_group_cfg: str | None = None
    """The observation group of the expert which may be different from student"""

    experts_observation_func: callable = None
    """The function that returns expert observation data, default is None, same as student observation."""

    learn_std: bool = False
    """Whether to learn the standard deviation of the expert policy."""

    cloning_loss_coeff: float = MISSING  # type: ignore
    """The coefficient for the cloning loss."""

    loss_decay: float = 1.0
    """The decay for the cloning loss coefficient. default to 1, no decay."""


@configclass
class OffPolicyAlgorithmCfg:
    """Configuration for the off-policy algorithm."""

    update_frequencies: float = 1
    """The frequency to update relative to online update."""

    batch_size: int | None = None
    """The batch size for the offline algorithm update, default to None, same of online size."""

    num_learning_epochs: int | None = None
    """The number of learning epochs for the offline algorithm update."""

    behavior_cloning_cfg: BehaviorCloningCfg | None = None
    """The configuration for the offline behavior cloning(dagger)."""


@configclass
class RslRlFancyActorCriticCfg(RslRlPpoActorCriticCfg):
    """Configuration for the fancy actor-critic networks."""

    state_dependent_std: bool = False
    """Whether to use state-dependent standard deviation."""

    noise_std_type: Literal["scalar", "log", "gsde"] = "scalar"
    """The type of noise standard deviation for the policy. Default is scalar."""


@configclass
class TransformerOptimizerCfg:
    """Configuration for the transformer optimizer."""
    learning_rate: float = 1.0e-4
    """The learning rate for the transformer optimizer. Default is 1.0e-4."""

    weight_decay: float = 0.00
    """The weight decay for the transformer optimizer."""

    betas: tuple[float, float] = (0.9, 0.99)
    """The betas for the transformer optimizer."""

    eps: float = 1.0e-8
    """The epsilon for the transformer optimizer."""

    max_grad_norm: float = 1.0
    """The maximum gradient norm for the transformer optimizer."""

    optimizer_class: str = "AdamW"
    """The class name of the transformer optimizer."""

    lr_warmup_steps: int = 0
    """The number of warmup steps for the learning rate scheduler."""

    lr_schedule: str | None = None
    """The learning rate schedule function name."""


@configclass
class RslRlFancyTransformerHistoryActorCriticCfg(RslRlFancyActorCriticCfg):
    """Configuration for actor-critic networks with transformer history."""

    embedding_dim: int = 128
    """The embedding dimension for the transformer history actor-critic."""

    hidden_dim: int = 256
    """The hidden dimension for the transformer history actor-critic."""

    num_layers: int = 2
    """The number of layers for the transformer history actor-critic."""

    num_heads: int = 4
    """The number of heads for the transformer history actor-critic."""

    embedding_dropout: float = 0.1
    """The dropout for the transformer history actor-critic."""

    attention_dropout: float = 0.1
    """The attention dropout for the transformer history actor-critic."""

    residual_dropout: float = 0.1
    """The residual dropout for the transformer history actor-critic."""

    max_num_episodes: int = 1
    """The maximum number of episodes for the transformer history actor-critic."""

    context_length_override: int | None = None
    """The context length override for the transformer history actor-critic."""

    cross_attention_merge: bool = False
    """Whether to merge obs and context via cross-attention."""

    obs_token_count: int = 4
    """The number of obs query tokens for cross-attention."""

    transformer_actor_class_name: str | None = None
    """Transformer actor class name for transformer-only policy.

    Valid options: "MergedTokenTransformerActor", "StateActionTransformerActor".
    """

    action_distribution: Literal["normal", "categorical"] = "normal"
    """Action distribution type for the policy."""

    optimizer: TransformerOptimizerCfg = TransformerOptimizerCfg()
    """The optimizer for the transformer history actor-critic."""

    log_attention_entropy: bool = False
    """Whether to log transformer attention entropy during training."""

    attention_entropy_interval: int = 0
    """Log attention entropy every N policy updates (0 disables)."""


@configclass
class RslRlFancyPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the PPO algorithm."""

    behavior_cloning_cfg: BehaviorCloningCfg | None = None
    """The configuration for the online behavior cloning."""

    offline_algorithm_cfg: OffPolicyAlgorithmCfg | None = None
    """The configuration for the offline algorithms."""
