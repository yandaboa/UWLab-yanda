# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from uwlab_rl.rsl_rl.rl_cfg import RslRlFancyTransformerHistoryActorCriticCfg, TransformerOptimizerCfg

def my_experts_observation_func(env):
    obs = env.unwrapped.obs_buf["expert_obs"]
    return obs


@configclass
class PPOWithContextRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 10000
    save_interval = 100
    resume = False
    experiment_name = "ur5e_robotiq_2f85_from_demo"
    logger = "wandb"
    policy = RslRlFancyTransformerHistoryActorCriticCfg(
        class_name="LongContextActorCritic",
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128, 64],
        critic_hidden_dims=[512, 256, 128, 64],
        activation="elu",
        noise_std_type="gsde",
        state_dependent_std=False,

        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        num_heads=4,
        embedding_dropout=0.1,
        attention_dropout=0.1,
        residual_dropout=0.1,
        max_num_episodes=1, # not actually implemented yet lol
        context_length_override=None, # i dont think this does anything either
        optimizer=TransformerOptimizerCfg(
            learning_rate=1.0e-4,
            weight_decay=0.00,
            betas=(0.9, 0.99),
            eps=1.0e-8,
            max_grad_norm=1.0,
            optimizer_class="AdamW",
        ),
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPOWithLongContext",
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
