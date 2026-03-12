from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

from uwlab_rl.rsl_rl.rl_cfg import RslRlFancyActorCriticCfg, RslRlFancyTransformerHistoryActorCriticCfg
from uwlab_rl.rsl_rl.rl_cfg import TransformerOptimizerCfg

@configclass
class DistillationAlgorithmCfg:
    class_name = "DaggerDistillation"
    num_learning_epochs = 10
    gradient_length = 8
    learning_rate = 1.0e-4
    max_grad_norm = 1.0
    loss_type = "mse"
    optimizer = "adam"
    dagger_buffer_steps = 8196
    dagger_storage_device = "cpu"


@configclass
class LongContextStudentTeacherCfg:
    class_name = "LongContextStudentTeacher"
    student_cfg: RslRlFancyTransformerHistoryActorCriticCfg = RslRlFancyTransformerHistoryActorCriticCfg(
        class_name="LongContextActorCritic",
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=True,
        actor_hidden_dims=[512],
        critic_hidden_dims=[512, 256, 128, 64],
        activation="elu",
        noise_std_type="scalar",
        state_dependent_std=False,
        embedding_dim=256,
        hidden_dim=256,
        num_layers=6,
        num_heads=4,
        embedding_dropout=0.0,
        attention_dropout=0.0,
        residual_dropout=0.0,
        context_token_layout="state_action",
        action_distribution="normal",
        share_current_and_context_obs_projection=False,
        encoding_projection_hidden_dim=None,
        cross_attention_merge=True,
        obs_token_count=1,
        max_num_episodes=1,
        context_length_override=None,
        optimizer=TransformerOptimizerCfg(
            learning_rate=1.0e-4,
            weight_decay=0.00,
            betas=(0.9, 0.95),
            eps=1.0e-8,
            max_grad_norm=1.0,
            optimizer_class="AdamW",
            lr_warmup_steps=50,
            lr_schedule="cosine_annealing_with_warmup",
        ),
    )
    teacher_cfg: RslRlFancyActorCriticCfg = RslRlFancyActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128, 64],
        critic_hidden_dims=[512, 256, 128, 64],
        activation="elu",
        noise_std_type="gsde",
        state_dependent_std=False,
    )


@configclass
class DistillationLongContextRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "DistillationRunner"
    num_steps_per_env = 8
    max_iterations = 10000
    save_interval = 100
    resume = False
    load_expert = "/gscratch/weirdlab/yanda/lti/UWLab-yanda/less_overfit_peg_tracking_expert.pt"
    # load_expert = "/gscratch/weirdlab/yanda/lti/UWLab-yanda/logs/rsl_rl/ur5e_robotiq_2f85_from_demo_priviledged/2026-02-02_20-13-17/model_9000.pt"
    experiment_name = "ur5e_robotiq_2f85_from_demo_distill"
    logger = "wandb"
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
        "teacher": ["critic"],
    }
    policy: LongContextStudentTeacherCfg = LongContextStudentTeacherCfg()
    algorithm: DistillationAlgorithmCfg = DistillationAlgorithmCfg()  # type: ignore[assignment[]]
