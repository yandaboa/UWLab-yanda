# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from uwlab_rl.rsl_rl.rl_cfg import RslRlFancyTransformerHistoryActorCriticCfg, TransformerOptimizerCfg
from uwlab_rl.rsl_rl.metaleanring_cfg import RslRlPpoAlgorithmWarmStartCfg, BCFromContextWarmStartCfg
from uwlab_rl.rsl_rl.rl_cfg import RslRlFancyActorCriticCfg

from .trajectory_viz_cfg import TrajectoryVizCfg


def my_experts_observation_func(env):
    obs = env.unwrapped.obs_buf["expert_obs"]
    return obs


@configclass
class PPOWithPriviledgedRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 10000
    save_interval = 100
    resume = False
    experiment_name = "ur5e_robotiq_2f85_from_demo_priviledged"
    logger = "wandb"
    policy = RslRlFancyActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128, 64],
        critic_hidden_dims=[512, 256, 128, 64],
        activation="elu",
        noise_std_type="gsde",
        state_dependent_std=False
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
    trajectory_viz = TrajectoryVizCfg(
        enable=True,
        log_every_iters=100,
        max_pairs_per_log=4,
        obs_key="debug/end_effector_pose",
        demo_obs_key="end_effector_pose",
        save_pairs=False,
    )
