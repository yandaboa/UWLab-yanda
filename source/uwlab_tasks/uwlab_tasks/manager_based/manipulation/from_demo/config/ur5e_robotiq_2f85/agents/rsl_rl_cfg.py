# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

from uwlab_rl.rsl_rl.rl_cfg import RslRlFancyTransformerHistoryActorCriticCfg, TransformerOptimizerCfg
from uwlab_rl.rsl_rl.metaleanring_cfg import RslRlPpoAlgorithmWarmStartCfg
from .trajectory_viz_cfg import TrajectoryVizCfg


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
        init_noise_std=0.5,
        actor_obs_normalization=False,
        critic_obs_normalization=True,
        actor_hidden_dims=[512],
        # actor_hidden_dims=[256, 128],
        critic_hidden_dims=[512, 256, 128, 64],
        activation="elu",
        noise_std_type="gsde",
        state_dependent_std=False,

        embedding_dim=256,
        hidden_dim=512,
        num_layers=4,
        num_heads=4,
        embedding_dropout=0.0,
        attention_dropout=0.0,
        residual_dropout=0.0,

        context_token_layout="merged",
        include_actions_in_context=False,
        include_rewards_in_context=True,

        action_distribution="normal",
        share_current_and_context_obs_projection=True,
        encoding_projection_hidden_dim=None,
        model_finetune_ckpt="logs/rsl_rl/supervised_context/2026-02-22_18-04-35/model_020000.pt",
        log_attention_entropy=True,
        attention_entropy_interval=10,

        cross_attention_merge=True,
        obs_token_count=1,
        max_num_episodes=1,  # not actually implemented yet lol
        context_length_override=None,  # i dont think this does anything either
        optimizer=TransformerOptimizerCfg(
            learning_rate=1.0e-4,
            weight_decay=0.00,
            betas=(0.9, 0.95),
            eps=1.0e-8,
            max_grad_norm=1.0,
            optimizer_class="AdamW",
            lr_warmup_steps=200,
            # lr_schedule="cosine_annealing_with_warmup",
            lr_schedule=None,
        ),
    )
    algorithm = RslRlPpoAlgorithmWarmStartCfg(
        class_name="PPOWithLongContext",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        normalize_advantage_per_mini_batch=False,
        clip_param=0.5,
        entropy_coef=0.0001,
        num_learning_epochs=1,
        num_mini_batches=16,
        num_minibatches_per_update=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        bc_warmstart_cfg=None,
        # bc_warmstart_cfg=BCFromContextWarmStartCfg(
        #     # lr_warmup_steps=100,
        #     lr_warmup_steps=500,
        #     lr_warmup_start_ratio=0.1,
        #     # num_steps=150,
        #     num_steps=1500,
        #     num_episodes_per_batch=8,
        #     num_minibatches=4,
        #     minibatch_size=None,
        #     learning_rate=1.0e-4,
        #     weight_decay=0.0,
        #     betas=(0.9, 0.99),
        #     eps=1.0e-8,
        #     max_grad_norm=1.0,
        #     optimizer_class="AdamW",
        #     use_amp=True,
        # ),
    )
    trajectory_viz: TrajectoryVizCfg = TrajectoryVizCfg(
        enable=False,
        log_every_iters=50,
        max_pairs_per_log=4,
        obs_key="debug/end_effector_pose",
        demo_obs_key="end_effector_pose",
        save_pairs=False,
    )