from dataclasses import MISSING

from isaaclab.utils import configclass

from uwlab_rl.rsl_rl.supervised_context_cfg import (
    SupervisedContextDataCfg,
    SupervisedContextDistributedCfg,
    SupervisedContextInputCfg,
    SupervisedContextLoggingCfg,
    SupervisedContextModelCfg,
    SupervisedContextOptimizationCfg,
    SupervisedContextTrainerCfg,
)


@configclass
class SupervisedContextRunnerCfg(SupervisedContextTrainerCfg):
    """Default config for supervised context training."""

    data: SupervisedContextDataCfg = SupervisedContextDataCfg(
        episode_paths=[
            "episodes/20260208_011257/episodes_000000.pt",
            "episodes/20260208_011257/episodes_000001.pt",
            "episodes/20260208_011257/episodes_000002.pt",
            "episodes/20260208_011257/episodes_000003.pt",
            "episodes/20260208_011257/episodes_000004.pt",
            "episodes/20260208_011257/episodes_000005.pt",
            "episodes/20260208_011257/episodes_000006.pt",
            "episodes/20260208_011257/episodes_000007.pt",
            "episodes/20260208_011257/episodes_000008.pt",
            "episodes/20260208_011257/episodes_000009.pt",
        ],
        obs_keys=["joint_pos", "end_effector_pose"],
        max_context_length=None,
        batch_size=64,
        num_workers=2,
        shuffle=True,
    )
    model: SupervisedContextModelCfg = SupervisedContextModelCfg(
        action_distribution="normal",
        action_discretization_spec_path=None,
        context_token_layout="state_only",
        include_actions_in_context=True,
        include_rewards_in_context=True,
        share_current_and_context_obs_projection=True,
        encoding_projection_hidden_dim=None,
        embedding_dim=256,
        hidden_dim=512,
        num_layers=4,
        num_heads=4,
        embedding_dropout=0.1,
        attention_dropout=0.1,
        residual_dropout=0.1,
    )
    optim: SupervisedContextOptimizationCfg = SupervisedContextOptimizationCfg(
        num_steps=100000,
        learning_rate=3.0e-4,
        weight_decay=0.0,
        betas=(0.9, 0.99),
        eps=1.0e-8,
        max_grad_norm=1.0,
        optimizer_class="AdamW",
        lr_warmup_steps=0,
        lr_schedule="cosine_annealing_with_warmup",
        use_amp=True,
    )
    input: SupervisedContextInputCfg = SupervisedContextInputCfg(
        include_current_trajectory=False,
    )
    distributed: SupervisedContextDistributedCfg = SupervisedContextDistributedCfg(
        distributed=False,
    )
    logging: SupervisedContextLoggingCfg = SupervisedContextLoggingCfg(
        experiment_name="same_obs_projector",
        run_name=None,
        log_interval=50,
        save_interval=200,
        log_project_name=None,
        use_wandb=True,
    )
