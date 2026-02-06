from isaaclab.utils import configclass


from uwlab_assets.robots.ur5e_robotiq_gripper import (
    EXPLICIT_UR5E_ROBOTIQ_2F85,
)
from .rl_state_cfg import (
    DemoContextPriviledgedCfg,
    FromDemoTrainEventCfg,
    FromDemoTrainTerminationsCfg,
    FromDemoRewardsCfg,
    PriviledgedFromDemoRewardsCfg,
    PriviledgedTrainingObservationsCfg,
    TrackingCommandsCfg,
    Ur5eRobotiq2f85RelativeOSCAction,
    Ur5eRobotiq2f85RlStateCfg,
    DebugObservationsCfg,
    POMDPPolicyCfg,
    PrivilegedPolicyCfg,
    TrainingObservationsCfg,
    DemoContextCfg,
)

@configclass
class DistillationObservationsCfg(TrainingObservationsCfg):
    """Observations for distillation."""

    policy: POMDPPolicyCfg = POMDPPolicyCfg()
    critic: PrivilegedPolicyCfg = PrivilegedPolicyCfg()


@configclass
class Ur5eRobotiq2f85RelJointPosFromDemoDistillationCfg(Ur5eRobotiq2f85RlStateCfg):
    """From-demo configuration for student-teacher distillation."""

    events: FromDemoTrainEventCfg = FromDemoTrainEventCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()
    rewards: FromDemoRewardsCfg = FromDemoRewardsCfg()
    observations: DistillationObservationsCfg = DistillationObservationsCfg()
    context: DemoContextCfg = DemoContextCfg()
    commands: TrackingCommandsCfg = TrackingCommandsCfg()
    terminations: FromDemoTrainTerminationsCfg = FromDemoTrainTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = EXPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")
