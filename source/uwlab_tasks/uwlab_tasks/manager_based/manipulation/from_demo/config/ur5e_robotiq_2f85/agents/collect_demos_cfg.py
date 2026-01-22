# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from uwlab_rl.rsl_rl.rl_cfg import RslRlFancyActorCriticCfg
from ..noise_cfg import EnvironmentNoiseCfg

@configclass
class Base_CollectDemosPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    noise: EnvironmentNoiseCfg = EnvironmentNoiseCfg()

@configclass
class CollectDemosPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 40000
    save_interval = 100
    resume = False
    experiment_name = "ur5e_robotiq_2f85_from_demo_collect_demos"
    logger = "wandb"
    policy = RslRlFancyActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128, 64],
        critic_hidden_dims=[512, 256, 128, 64],
        activation="elu",
        noise_std_type="gsde",
        state_dependent_std=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        normalize_advantage_per_mini_batch=False,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    noise: EnvironmentNoiseCfg = EnvironmentNoiseCfg(
        noise_frequency_distribution="uniform",
        noise_magnitude_distribution="uniform",
        max_noise_frequency = 1.0,
        min_noise_frequency = 0.0,
        mean_noise_magnitude = 0.0,
        std_noise_magnitude = 4.0,
        min_noise_magnitude = 8.0,
        max_noise_magnitude = 16.0,
    )