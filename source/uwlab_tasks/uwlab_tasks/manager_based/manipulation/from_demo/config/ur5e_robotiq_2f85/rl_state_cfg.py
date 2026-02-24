# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import CommandTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.envs.mdp import events as mdp_events

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR
from uwlab_assets.robots.ur5e_robotiq_gripper import (
    EXPLICIT_UR5E_ROBOTIQ_2F85,
    IMPLICIT_UR5E_ROBOTIQ_2F85,
    Ur5eRobotiq2f85RelativeJointPositionAction,
)

from uwlab_tasks.manager_based.manipulation.reset_states.config.ur5e_robotiq_2f85.actions import (
    Ur5eRobotiq2f85RelativeOSCAction,
)

from ....reset_states import mdp as omni_reset_mdp
from ... import mdp as from_demo_mdp

LOCAL_DATASETS_DIR = "reset_state_datasets"


@configclass
class ActionDiscretizationCfg:
    """Configuration for discretizing continuous actions."""

    min_actions: tuple[float, ...] | float = (-25.0, -25.0, -25.0, -25.0, -25.0, -25.0, -25.0)
    max_actions: tuple[float, ...] | float = (25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0)
    num_bins: int = 201


@configclass
class RlStateSceneCfg(InteractiveSceneCfg):
    """Scene configuration for RL state environment."""

    robot = EXPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")

    insertive_object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/InsertiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Peg/peg.usd",
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    receptive_object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ReceptiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/PegHole/peg_hole.usd",
            scale=(1, 1, 1),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                # receptive object does not move
                kinematic_enabled=True,
            ),
            # since kinematic_enabled=True, mass does not matter
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    receptive_object_grip_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/ReceptiveObject",
        history_length=1,
        # Filter contact reports to gripper links, so this tracks EE/receptive-object interaction only.
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Robot/robotiq_base_link",
            "{ENV_REGEX_NS}/Robot/.*finger.*",
            "{ENV_REGEX_NS}/Robot/.*inner.*",
            "{ENV_REGEX_NS}/Robot/.*outer.*",
        ],
    )

    # Environment
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, -0.881), rot=(0.707, 0.0, 0.0, -0.707)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Mounts/UWPatVention/pat_vention.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    ur5_metal_support = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/UR5MetalSupport",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0, -0.013), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Mounts/UWPatVention2/Ur5MetalSupport/ur5plate.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.868)),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=10000.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class BaseEventCfg:
    """Configuration for events."""

    # mode: startup (randomize dynamics)
    robot_material = EventTerm(
        func=omni_reset_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            # "static_friction_range": (0.3, 1.2),
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.6, 0.6),
            # "dynamic_friction_range": (0.2, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
            "asset_cfg": SceneEntityCfg("robot"),
            "make_consistent": True,
        },
    )

    # use large friction to avoid slipping
    insertive_object_material = EventTerm(
        func=omni_reset_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            # "static_friction_range": (1.0, 2.0),
            "static_friction_range": (1.5, 1.5),
            # "dynamic_friction_range": (0.9, 1.9),
            "dynamic_friction_range": (1.4, 1.4),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
            "asset_cfg": SceneEntityCfg("insertive_object"),
            "make_consistent": True,
        },
    )

    # use large friction to avoid slipping
    receptive_object_material = EventTerm(
        func=omni_reset_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            # "static_friction_range": (1.0, 2.0),
            "static_friction_range": (1.5, 1.5),
            # "dynamic_friction_range": (0.9, 1.9),
            "dynamic_friction_range": (1.4, 1.4),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
            "asset_cfg": SceneEntityCfg("receptive_object"),
            "make_consistent": True,
        },
    )

    table_material = EventTerm(
        func=omni_reset_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            # "static_friction_range": (0.3, 0.6),
            "static_friction_range": (0.45, 0.45),
            # "dynamic_friction_range": (0.2, 0.5),
            "dynamic_friction_range": (0.35, 0.35),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
            "asset_cfg": SceneEntityCfg("table"),
            "make_consistent": True,
        },
    )

    randomize_robot_mass = EventTerm(
        func=omni_reset_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (1.0, 1.0),
            # "mass_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    randomize_insertive_object_mass = EventTerm(
        func=omni_reset_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("insertive_object"),
            # we assume insertive object is somewhere between 20g and 200g
            # "mass_distribution_params": (0.02, 0.2),
            "mass_distribution_params": (0.11, 0.11),
            "operation": "abs",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    randomize_receptive_object_mass = EventTerm(
        func=omni_reset_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("receptive_object"),
            "mass_distribution_params": (0.75, 0.75),
            # "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    randomize_table_mass = EventTerm(
        func=omni_reset_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("table"),
            "mass_distribution_params": (0.75, 0.75),
            # "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    randomize_robot_joint_parameters = EventTerm(
        func=omni_reset_mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*", "finger_joint"]),
            # "friction_distribution_params": (0.25, 4.0),
            "friction_distribution_params": (0.325, 0.325),
            # "armature_distribution_params": (0.25, 4.0),
            "armature_distribution_params": (0.325, 0.325),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    randomize_gripper_actuator_parameters = EventTerm(
        func=omni_reset_mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
            # "stiffness_distribution_params": (0.5, 2.0),
            "stiffness_distribution_params": (1.25, 1.25),
            # "damping_distribution_params": (0.5, 2.0),
            "damping_distribution_params": (1.25, 1.25),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    # mode: reset
    reset_everything = EventTerm(func=omni_reset_mdp.reset_scene_to_default, mode="reset", params={})

# @configclass
# class DefaultEventCfg:
    # reset_everything = EventTerm(func=omni_reset_mdp.reset_scene_to_default, mode="reset", params={})

@configclass
class BaseFromDemoEventCfg(BaseEventCfg):

    reset_everything = EventTerm(func=omni_reset_mdp.reset_scene_to_default, mode="reset", params={})

    reset_from_reset_states = EventTerm(
        func=omni_reset_mdp.MultiResetManager,
        mode="reset",
        params={
            "base_paths": [
                f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/Resets/ObjectPairs/ObjectAnywhereEEAnywhere",
                # f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/Resets/ObjectPairs/ObjectRestingEEGrasped",
                # f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/Resets/ObjectPairs/ObjectAnywhereEEGrasped",
                # f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/Resets/ObjectPairs/ObjectPartiallyAssembledEEGrasped",
            ],
            "probs": [1.0],
            # "probs": [0.25, 0.25, 0.25, 0.25],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )


@configclass
class FromDemoTrainEventCfg(BaseFromDemoEventCfg):
    """Training events with from-demo context."""

    # random_pushes = EventTerm(
    #     func=mdp_events.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(2.0, 4.0),
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
    #         "velocity_range": {"x": (-0.4, 0.4), "y": (-0.4, 0.4)},
    #     },
    # )

    resample_episode = EventTerm(
        func=from_demo_mdp.resample_episode,
        mode="reset",
        params={},
    )

@configclass
class FromDemoPriviledgedTrainEventCfg(FromDemoTrainEventCfg):
    """Training events with from-demo context."""

    # randomly sample pushes to the robot end effector
    random_pushes = EventTerm(
        func=mdp_events.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 2.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
            "velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        },
    )

@configclass
class FromDemoEvalEventCfg(BaseFromDemoEventCfg):
    """Evaling models to track a demo."""

    resample_episode = EventTerm(
        func=from_demo_mdp.resample_episode,
        mode="reset",
        params={},
    )

@configclass
class EvalEventCfg(BaseEventCfg):
    """Configuration for evaluation events."""

    reset_from_reset_states = EventTerm(
        func=omni_reset_mdp.MultiResetManager,
        mode="reset",
        params={
            "base_paths": [
                f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/Resets/ObjectPairs/ObjectAnywhereEEAnywhere",
                # f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/Resets/ObjectPairs/ObjectRestingEEGrasped",
                # f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/Resets/ObjectPairs/ObjectAnywhereEEGrasped",
                # f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/Resets/ObjectPairs/ObjectPartiallyAssembledEEGrasped",
            ],
            "probs": [1.0],
            # "probs": [0.25, 0.25, 0.25, 0.25],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )

@configclass
class DemoCollectEventCfg(EvalEventCfg):
    """Configuration for demo collection events."""

    resample_environment_noise = EventTerm(
        func=from_demo_mdp.resample_environment_noise,
        mode="post_reset",
        params={},
    )

@configclass
class DemoTrackingMetricsCommandCfg(CommandTermCfg):
    """Configuration for the demo tracking metrics command term."""

    class_type: type = from_demo_mdp.DemoTrackingMetricsCommand
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    resampling_time_range: tuple[float, float] = (0.01, 0.01)

@configclass
class GoalCommandsCfg:
    """Command specifications for the MDP."""

    task_command = omni_reset_mdp.TaskCommandCfg(
        asset_cfg=SceneEntityCfg("robot", body_names="body"),
        resampling_time_range=(1e6, 1e6),
        success_position_threshold=0.005,
        success_orientation_threshold=0.025,
        insertive_asset_cfg=SceneEntityCfg("insertive_object"),
        receptive_asset_cfg=SceneEntityCfg("receptive_object"),
    )
@configclass
class TrackingCommandsCfg:
    """Command specifications for the tracking metrics command term."""

    tracking_metrics_command = DemoTrackingMetricsCommandCfg()

@configclass
class CriticCfg(ObsGroup):
    """Critic observations for policy group."""

    prev_actions = ObsTerm(func=omni_reset_mdp.last_action)

    joint_pos = ObsTerm(func=omni_reset_mdp.joint_pos)

    end_effector_pose = ObsTerm(
        func=omni_reset_mdp.target_asset_pose_in_root_asset_frame_with_metadata,
        params={
            "target_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
            "root_asset_cfg": SceneEntityCfg("robot"),
            "target_asset_offset_metadata_key": "gripper_offset",
            "root_asset_offset_metadata_key": "offset",
            "rotation_repr": "axis_angle",
        },
    )

    insertive_asset_pose = ObsTerm(
        func=omni_reset_mdp.target_asset_pose_in_root_asset_frame_with_metadata,
        params={
            "target_asset_cfg": SceneEntityCfg("insertive_object"),
            "root_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
            "root_asset_offset_metadata_key": "gripper_offset",
            "rotation_repr": "axis_angle",
            "object_type": "insertive",
        },
    )

    receptive_asset_pose = ObsTerm(
        func=omni_reset_mdp.target_asset_pose_in_root_asset_frame,
        params={
            "target_asset_cfg": SceneEntityCfg("receptive_object"),
            "root_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
            "rotation_repr": "axis_angle",
            "ood_offset": 1.0,
            "object_type": "receptive",
        },
    )

    insertive_asset_in_receptive_asset_frame: ObsTerm = ObsTerm(
        func=omni_reset_mdp.target_asset_pose_in_root_asset_frame,
        params={
            "target_asset_cfg": SceneEntityCfg("insertive_object"),
            "root_asset_cfg": SceneEntityCfg("receptive_object"),
            "rotation_repr": "axis_angle",
            "ood_offset": 1.0,
            "object_type": "insertive",
        },
    )

    # privileged observations
    time_left = ObsTerm(func=omni_reset_mdp.time_left)

    joint_vel = ObsTerm(func=omni_reset_mdp.joint_vel)

    end_effector_vel_lin_ang_b = ObsTerm(
        func=omni_reset_mdp.asset_link_velocity_in_root_asset_frame,
        params={
            "target_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
            "root_asset_cfg": SceneEntityCfg("robot"),
        },
    )

    robot_material_properties = ObsTerm(
        func=omni_reset_mdp.get_material_properties, params={"asset_cfg": SceneEntityCfg("robot")}
    )

    insertive_object_material_properties = ObsTerm(
        func=omni_reset_mdp.get_material_properties, params={"asset_cfg": SceneEntityCfg("insertive_object")}
    )

    receptive_object_material_properties = ObsTerm(
        func=omni_reset_mdp.get_material_properties, params={"asset_cfg": SceneEntityCfg("receptive_object")}
    )

    table_material_properties = ObsTerm(
        func=omni_reset_mdp.get_material_properties, params={"asset_cfg": SceneEntityCfg("table")}
    )

    robot_mass = ObsTerm(func=omni_reset_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("robot")})

    insertive_object_mass = ObsTerm(
        func=omni_reset_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("insertive_object")}
    )

    receptive_object_mass = ObsTerm(
        func=omni_reset_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("receptive_object")}
    )

    table_mass = ObsTerm(func=omni_reset_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("table")})

    robot_joint_friction = ObsTerm(func=omni_reset_mdp.get_joint_friction, params={"asset_cfg": SceneEntityCfg("robot")})

    robot_joint_armature = ObsTerm(func=omni_reset_mdp.get_joint_armature, params={"asset_cfg": SceneEntityCfg("robot")})

    robot_joint_stiffness = ObsTerm(
        func=omni_reset_mdp.get_joint_stiffness, params={"asset_cfg": SceneEntityCfg("robot")}
    )

    robot_joint_damping = ObsTerm(func=omni_reset_mdp.get_joint_damping, params={"asset_cfg": SceneEntityCfg("robot")})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 1

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        prev_actions = ObsTerm(func=omni_reset_mdp.last_action)

        joint_pos = ObsTerm(func=omni_reset_mdp.joint_pos)

        end_effector_pose = ObsTerm(
            func=omni_reset_mdp.target_asset_pose_in_root_asset_frame_with_metadata,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
                "root_asset_cfg": SceneEntityCfg("robot"),
                "target_asset_offset_metadata_key": "gripper_offset",
                "root_asset_offset_metadata_key": "offset",
                "rotation_repr": "axis_angle",
            },
        )

        insertive_asset_pose = ObsTerm(
            func=omni_reset_mdp.target_asset_pose_in_root_asset_frame_with_metadata,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
                "root_asset_offset_metadata_key": "gripper_offset",
                "rotation_repr": "axis_angle",
                "object_type": "insertive",
            },
        )

        receptive_asset_pose = ObsTerm(
            func=omni_reset_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("receptive_object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
                "rotation_repr": "axis_angle",
                "object_type": "receptive",
            },
        )

        insertive_asset_in_receptive_asset_frame: ObsTerm = ObsTerm(
            func=omni_reset_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("receptive_object"),
                "rotation_repr": "axis_angle",
                "object_type": "insertive",
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            # self.history_length = 1
            self.history_length = 5

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

@configclass
class DebugObservationsCfg(ObservationsCfg):
    
    @configclass
    class DebugCfg(ObsGroup):
        """Observations for debug group."""

        end_effector_pose = ObsTerm(
            func=omni_reset_mdp.target_asset_pose_in_root_asset_frame_with_metadata,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
                "root_asset_cfg": SceneEntityCfg("robot"),
                "target_asset_offset_metadata_key": "gripper_offset",
                "root_asset_offset_metadata_key": "offset",
                "rotation_repr": "axis_angle",
            },
        )

        insertive_asset_pose = ObsTerm(
            func=omni_reset_mdp.target_asset_pose_in_root_asset_frame_with_metadata,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
                "root_asset_offset_metadata_key": "gripper_offset",
                "rotation_repr": "axis_angle",
                "object_type": "insertive",
            },
        )

        receptive_asset_pose = ObsTerm(
            func=omni_reset_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("receptive_object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
                "rotation_repr": "axis_angle",
                "object_type": "receptive",
            },
        )

        receptive_object_gripped = ObsTerm(
            func=from_demo_mdp.receptive_object_gripped_by_ee,
            params={
                "sensor_cfg": SceneEntityCfg("receptive_object_grip_contact"),
                "force_threshold": 1.0,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False
            self.history_length = 1
    
    debug: DebugCfg = DebugCfg()


@configclass
class DemoCfg(ObsGroup):
    """Observations for demo group."""

    joint_pos = ObsTerm(func=omni_reset_mdp.joint_pos)

    demo_link_quats = ObsTerm(func=from_demo_mdp.demo_link_quats)

    end_effector_pose = ObsTerm(
        func=omni_reset_mdp.target_asset_pose_in_root_asset_frame_with_metadata,
        params={
            "target_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
            "root_asset_cfg": SceneEntityCfg("robot"),
            "target_asset_offset_metadata_key": "gripper_offset",
            "root_asset_offset_metadata_key": "offset",
            "rotation_repr": "axis_angle",
        },
    )

    insertive_asset_pose = ObsTerm(
        func=omni_reset_mdp.target_asset_pose_in_root_asset_frame_with_metadata,
        params={
            "target_asset_cfg": SceneEntityCfg("insertive_object"),
            "root_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
            "root_asset_offset_metadata_key": "gripper_offset",
            "rotation_repr": "axis_angle",
        },
    )

    receptive_asset_pose = ObsTerm(
        func=omni_reset_mdp.target_asset_pose_in_root_asset_frame,
        params={
            "target_asset_cfg": SceneEntityCfg("receptive_object"),
            "root_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
            "rotation_repr": "axis_angle",
        },
    )

    insertive_asset_in_receptive_asset_frame: ObsTerm = ObsTerm(
        func=omni_reset_mdp.target_asset_pose_in_root_asset_frame,
        params={
            "target_asset_cfg": SceneEntityCfg("insertive_object"),
            "root_asset_cfg": SceneEntityCfg("receptive_object"),
            "rotation_repr": "axis_angle",
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False
        self.history_length = 1

@configclass
class SupervisedEvalObsCfg(ObsGroup):
    """Observations for supervised evaluation."""

    joint_pos = ObsTerm(func=omni_reset_mdp.joint_pos)

    end_effector_pose = ObsTerm(
        func=omni_reset_mdp.target_asset_pose_in_root_asset_frame_with_metadata,
        params={
            "target_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
            "root_asset_cfg": SceneEntityCfg("robot"),
            "target_asset_offset_metadata_key": "gripper_offset",
            "root_asset_offset_metadata_key": "offset",
            "rotation_repr": "axis_angle",
        },
    )

    insertive_asset_pose = ObsTerm(
        func=omni_reset_mdp.target_asset_pose_in_root_asset_frame_with_metadata,
        params={
            "target_asset_cfg": SceneEntityCfg("insertive_object"),
            "root_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
            "root_asset_offset_metadata_key": "gripper_offset",
            "rotation_repr": "axis_angle",
        },
    )

    receptive_asset_pose = ObsTerm(
        func=omni_reset_mdp.target_asset_pose_in_root_asset_frame,
        params={
            "target_asset_cfg": SceneEntityCfg("receptive_object"),
            "root_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
            "rotation_repr": "axis_angle",
        },
    )

    insertive_asset_in_receptive_asset_frame: ObsTerm = ObsTerm(
        func=omni_reset_mdp.target_asset_pose_in_root_asset_frame,
        params={
            "target_asset_cfg": SceneEntityCfg("insertive_object"),
            "root_asset_cfg": SceneEntityCfg("receptive_object"),
            "rotation_repr": "axis_angle",
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False
        self.history_length = 1
@configclass
class DataCollectObservationsCfg(DebugObservationsCfg):

    demo: DemoCfg = DemoCfg()

@configclass
class POMDPPolicyCfg(ObsGroup):
    """Needs to be the same as demo if we want to do BC"""

    joint_pos = ObsTerm(func=omni_reset_mdp.joint_pos)

    # demo_link_quats = ObsTerm(func=from_demo_mdp.demo_link_quats)

    end_effector_pose = ObsTerm(
        func=omni_reset_mdp.target_asset_pose_in_root_asset_frame_with_metadata,
        params={
            "target_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
            "root_asset_cfg": SceneEntityCfg("robot"),
            "target_asset_offset_metadata_key": "gripper_offset",
            "root_asset_offset_metadata_key": "offset",
            "rotation_repr": "axis_angle",
        },
    )

    insertive_asset_pose = ObsTerm(
        func=omni_reset_mdp.target_asset_pose_in_root_asset_frame_with_metadata,
        params={
            "target_asset_cfg": SceneEntityCfg("insertive_object"),
            "root_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
            "root_asset_offset_metadata_key": "gripper_offset",
            "rotation_repr": "axis_angle",
        },
    )

    # receptive_asset_pose = ObsTerm(
    #     func=omni_reset_mdp.target_asset_pose_in_root_asset_frame,
    #     params={
    #         "target_asset_cfg": SceneEntityCfg("receptive_object"),
    #         "root_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
    #         "rotation_repr": "axis_angle",
    #     },
    # )

    # insertive_asset_in_receptive_asset_frame: ObsTerm = ObsTerm(
    #     func=omni_reset_mdp.target_asset_pose_in_root_asset_frame,
    #     params={
    #         "target_asset_cfg": SceneEntityCfg("insertive_object"),
    #         "root_asset_cfg": SceneEntityCfg("receptive_object"),
    #         "rotation_repr": "axis_angle",
    #         "object_type": "insertive",
    #     },
    # )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 1
@configclass
class TrainingObservationsCfg(DebugObservationsCfg):
    """Observations for training."""

    @configclass
    class TrackingCriticCfg(CriticCfg):
        """Observations for tracking critic group."""
        context_obs = ObsTerm(
            func=from_demo_mdp.get_supervision_demo_obs,
            params={"observation_keys":["joint_pos", "end_effector_pose", "insertive_asset_pose"]
            },
        )
        context_actions = ObsTerm(
            func=from_demo_mdp.get_supervision_demo_actions,
        )
        context_rewards = ObsTerm(
            func=from_demo_mdp.get_last_demo_rewards,
        )

    @configclass
    class ContextCfg(ObsGroup):

        context_obs = ObsTerm(
            func=from_demo_mdp.get_demo_obs,
            params={
                "observation_keys": [
                    "joint_pos",
                    "end_effector_pose",
                    "insertive_asset_pose"
                ],
            },
        )
        context_actions = ObsTerm(
            func=from_demo_mdp.get_demo_actions,
        )
        context_rewards = ObsTerm(
            func=from_demo_mdp.get_demo_rewards,
        )
        context_lengths = ObsTerm(
            func=from_demo_mdp.get_demo_lengths,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False
            self.history_length = 1
    
    @configclass
    class EndEffectorPoseCfg(ObsGroup):
        """Observations for end effector pose."""
        
        end_effector_pose = ObsTerm(
            func=omni_reset_mdp.target_asset_pose_in_root_asset_frame_with_metadata,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
                "root_asset_cfg": SceneEntityCfg("robot"),
                "target_asset_offset_metadata_key": "gripper_offset",
                "root_asset_offset_metadata_key": "offset",
                "rotation_repr": "axis_angle",
            },
        )
    
    critic: TrackingCriticCfg = TrackingCriticCfg()
    context: ContextCfg = ContextCfg()
    ee_pose: EndEffectorPoseCfg = EndEffectorPoseCfg()
    policy: POMDPPolicyCfg = POMDPPolicyCfg()

@configclass
class SupervisedEvalObservationsCfg(TrainingObservationsCfg):

    policy: SupervisedEvalObsCfg = SupervisedEvalObsCfg()

@configclass
class PrivilegedPolicyCfg(ObsGroup):
    joint_pos = ObsTerm(func=omni_reset_mdp.joint_pos)

    # demo_link_quats = ObsTerm(func=from_demo_mdp.demo_link_quats)

    end_effector_pose = ObsTerm(
        func=omni_reset_mdp.target_asset_pose_in_root_asset_frame_with_metadata,
        params={
            "target_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
            "root_asset_cfg": SceneEntityCfg("robot"),
            "target_asset_offset_metadata_key": "gripper_offset",
            "root_asset_offset_metadata_key": "offset",
            "rotation_repr": "axis_angle",
        },
    )

    context_obs = ObsTerm(
        func=from_demo_mdp.get_supervision_demo_obs,
        params={"observation_keys":["end_effector_pose"]
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 1

@configclass
class PriviledgedTrainingObservationsCfg(TrainingObservationsCfg):
    """Observations for privileged training."""
    
    @configclass
    class PriviledgedTrackingCriticCfg(CriticCfg):
        """Observations for tracking critic group."""
        context_obs = ObsTerm(
            func=from_demo_mdp.get_supervision_demo_obs,
            params={"observation_keys":["end_effector_pose", "demo_link_quats", "joint_pos"]
            },
        )
        context_actions = ObsTerm(
            func=from_demo_mdp.get_supervision_demo_actions,
        )
        end_effector_vel_lin_ang_b = ObsTerm(
            func=omni_reset_mdp.asset_link_velocity_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
                "root_asset_cfg": SceneEntityCfg("robot"),
            },
        )
        # context_rewards = ObsTerm(
        #     func=from_demo_mdp.get_last_demo_rewards,
        # )
    
    critic: PriviledgedTrackingCriticCfg = PriviledgedTrackingCriticCfg()
    policy: PrivilegedPolicyCfg = PrivilegedPolicyCfg()

@configclass
class EvalObservationsCfg(DebugObservationsCfg):
    """Observations for evaluation."""

    @configclass
    class ContextCfg(ObsGroup):

        context_obs = ObsTerm(
            func=from_demo_mdp.get_demo_obs,
        )
        context_actions = ObsTerm(
            func=from_demo_mdp.get_demo_actions,
        )
        context_rewards = ObsTerm(
            func=from_demo_mdp.get_demo_rewards,
        )
        context_lengths = ObsTerm(
            func=from_demo_mdp.get_demo_lengths,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False
            self.history_length = 1
    
    @configclass
    class EndEffectorPoseCfg(ObsGroup):
        """Observations for end effector pose."""
        
        end_effector_pose = ObsTerm(
            func=omni_reset_mdp.target_asset_pose_in_root_asset_frame_with_metadata,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
                "root_asset_cfg": SceneEntityCfg("robot"),
                "target_asset_offset_metadata_key": "gripper_offset",
                "root_asset_offset_metadata_key": "offset",
                "rotation_repr": "axis_angle",
            },
        )

    context: ContextCfg = ContextCfg()
    ee_pose: EndEffectorPoseCfg = EndEffectorPoseCfg()

@configclass
class RewardsCfg:

    # safety rewards

    action_magnitude = RewTerm(func=omni_reset_mdp.action_l2_clamped, weight=-1e-4)

    action_rate = RewTerm(func=omni_reset_mdp.action_rate_l2_clamped, weight=-1e-4)

    joint_vel = RewTerm(
        func=omni_reset_mdp.joint_vel_l2_clamped,
        weight=-1e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"])},
    )

    abnormal_robot = RewTerm(func=omni_reset_mdp.abnormal_robot_state, weight=-100.0)

    # task rewards

    progress_context = RewTerm(
        func=omni_reset_mdp.ProgressContext,  # type: ignore
        weight=0.1,
        params={
            "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
            "receptive_asset_cfg": SceneEntityCfg("receptive_object"),
        },
    )

    ee_asset_distance = RewTerm(
        func=omni_reset_mdp.ee_asset_distance_tanh,
        weight=0.1,
        params={
            "root_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
            "target_asset_cfg": SceneEntityCfg("insertive_object"),
            "root_asset_offset_metadata_key": "gripper_offset",
            "std": 1.0,
        },
    )

    dense_success_reward = RewTerm(func=omni_reset_mdp.dense_success_reward, weight=0.1, params={"std": 1.0})

    success_reward = RewTerm(func=omni_reset_mdp.success_reward, weight=1.0)

@configclass
class FromDemoRewardsCfg:

    # safety rewards

    # action_magnitude = RewTerm(func=omni_reset_mdp.action_l2_clamped, weight=-1e-4)

    # action_rate = RewTerm(func=omni_reset_mdp.action_rate_l2_clamped, weight=-1e-4)

    # joint_vel = RewTerm(
    #     func=omni_reset_mdp.joint_vel_l2_clamped,
    #     weight=-1e-3,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"])},
    # )

    abnormal_robot = RewTerm(func=omni_reset_mdp.abnormal_robot_state, weight=-100.0)

    # task rewards

    # tracking_joint_angle = RewTerm(
    #     func=from_demo_mdp.pose_quat_tracking_reward,
    #     weight=5.0,
    #     params={
    #         "k": 2.0,
    #     },
    # )

    tracking_end_effector = RewTerm(
        func=from_demo_mdp.tracking_end_effector_reward,
        weight=10.0,
        params={
            "k": 1.0,
            "tolerance": 0.1,
        },
    )

    tracking_end_effector_orientation = RewTerm(
        func=from_demo_mdp.tracking_end_effector_orientation_reward,
        weight=10.0,
        params={
            "k": 5.0,
            "tolerance": 1.0,
        },
    )

    # tracking_action = RewTerm(
    #     func=from_demo_mdp.tracking_action_reward,
    #     weight=1.0,
    # )

    # just for making events work
    progress_context = RewTerm(
        func=omni_reset_mdp.ProgressContext,  # type: ignore
        weight=0.0,
        params={
            "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
            "receptive_asset_cfg": SceneEntityCfg("receptive_object"),
        },
    )

    demo_success = RewTerm(
        func=from_demo_mdp.demo_success_reward,
        weight=10.0,
    )

    demo_dense_success = RewTerm(
        func=from_demo_mdp.demo_dense_success_reward,
        weight=20.0,
        params={
            "std": 1.0,
        },
    )


@configclass
class PriviledgedFromDemoRewardsCfg(FromDemoRewardsCfg):
    tracking_joint_angle = RewTerm(
        func=from_demo_mdp.pose_quat_tracking_reward,
        weight=100.0,
        params={
            "k": 2.0,
        },
    )

@configclass
class DemoContextCfg:
    # episode_paths: str = "episodes/20260208_025923/episodes_000000.pt"
    episode_paths: list[str] = [
        # "episodes/20260222_164621/episodes_000000_trim_train90_train.pt",
        "episodes/20260222_155251/episodes_000001_trim.pt",
    ]
    # episode_paths: list[str] = [ # peg tracking
    #     "episodes/20260202_173353/episodes_000000.pt",
    #     "episodes/20260202_173353/episodes_000001.pt",
    #     "episodes/20260202_173353/episodes_000002.pt",
    #     "episodes/20260202_173353/episodes_000003.pt",
    #     "episodes/20260202_173353/episodes_000004.pt",
    #     "episodes/20260202_173353/episodes_000005.pt",
    #     "episodes/20260202_173353/episodes_000006.pt",
    #     "episodes/20260202_173353/episodes_000007.pt",
    #     "episodes/20260202_173353/episodes_000008.pt",
    #     "episodes/20260202_173353/episodes_000009.pt",
    # ]
    state_noise_scale: float = 0.0
    download_dir: str | None = None
    use_raw_states: bool = False

@configclass
class DemoContextPriviledgedCfg(DemoContextCfg):
    """Context for privileged training."""
    # episode_paths: str = "episodes/20260208_011257/episodes_000000.pt"
    episode_paths: list[str] = [
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
    ]
    use_raw_states=False

@configclass
class DemoEvalContextCfg:
    # episode_paths: str = "episodes/20260217_124614/episodes_000000.pt" # single episode of peg insertion
    # episode_paths: str = "episodes/20260222_155251/episodes_000002_trim_subsetx.pt"
    episode_paths: str = "episodes/20260222_155251/episodes_000000_trim_subsetx.pt"
    # episode_paths: str = "episodes/20260222_164621/episodes_000000_trim_train90_train.pt"
    # episode_paths: list[str] = [
    #     "episodes/20260128_011438/episodes_000000.pt",
    # ]
    state_noise_scale: float = 0.0
    download_dir: str | None = None
    use_raw_states: bool = False

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=omni_reset_mdp.time_out, time_out=True)

    abnormal_robot = DoneTerm(func=omni_reset_mdp.abnormal_robot_state)

@configclass
class FromDemoCollectTerminationsCfg:
    time_out = DoneTerm(func=omni_reset_mdp.time_out, time_out=True)

    abnormal_robot = DoneTerm(func=omni_reset_mdp.abnormal_robot_state)

    success = DoneTerm(func=omni_reset_mdp.terminate_on_success, params={"delay_steps": 0})

@configclass
class FromDemoTrainTerminationsCfg:
    abnormal_robot = DoneTerm(func=omni_reset_mdp.abnormal_robot_state)
    end_of_demo = DoneTerm(func=from_demo_mdp.end_of_demo)

@configclass
class EpisodeTimeoutTerminationsCfg:
    # time_out = DoneTerm(func=omni_reset_mdp.time_out, time_out=True)

    end_of_demo = DoneTerm(func=from_demo_mdp.end_of_demo)

def make_insertive_object(usd_path: str):
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/InsertiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )


def make_receptive_object(usd_path: str):
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ReceptiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=(1, 1, 1),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )


variants = {
    "scene.insertive_object": {
        "fbleg": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/SquareLeg/square_leg.usd"),
        "fbdrawerbottom": make_insertive_object(
            f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/DrawerBottom/drawer_bottom.usd"
        ),
        "peg": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Peg/peg.usd"),
        "cupcake": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/CupCake/cupcake.usd"),
        "cube": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/InsertiveCube/insertive_cube.usd"),
        "rectangle": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Rectangle/rectangle.usd"),
    },
    "scene.receptive_object": {
        "fbtabletop": make_receptive_object(
            f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/SquareTableTop/square_table_top.usd"
        ),
        "fbdrawerbox": make_receptive_object(
            f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/DrawerBox/drawer_box.usd"
        ),
        "peghole": make_receptive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/PegHole/peg_hole.usd"),
        "plate": make_receptive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Plate/plate.usd"),
        "cube": make_receptive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/ReceptiveCube/receptive_cube.usd"),
        "wall": make_receptive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Wall/wall.usd"),
    },
}


@configclass
class Ur5eRobotiq2f85RlStateCfg(ManagerBasedRLEnvCfg):
    scene: RlStateSceneCfg = RlStateSceneCfg(num_envs=32, env_spacing=1.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: BaseEventCfg = MISSING
    commands: GoalCommandsCfg = GoalCommandsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(2.0, 0.0, 0.75), origin_type="world", env_index=0, asset_name="robot")
    variants = variants

    def __post_init__(self):
        self.decimation = 12
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 1 / 120.0

        # Contact and solver settings
        self.sim.physx.solver_type = 1
        self.sim.physx.max_position_iteration_count = 192
        self.sim.physx.max_velocity_iteration_count = 1
        self.sim.physx.bounce_threshold_velocity = 0.02
        self.sim.physx.friction_offset_threshold = 0.01
        self.sim.physx.friction_correlation_distance = 0.0005

        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**23
        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.sim.physx.gpu_max_rigid_patch_count = 2**23
        self.sim.physx.gpu_collision_stack_size = 2**31

        # Render settings
        self.sim.render.enable_dlssg = True
        self.sim.render.enable_ambient_occlusion = True
        self.sim.render.enable_reflections = True
        self.sim.render.enable_dl_denoiser = True


# Training configurations

@configclass
class Ur5eRobotiq2f85RelJointPosDemoCollectCfg(Ur5eRobotiq2f85RlStateCfg):
    """Demo collection configuration for Relative Joint Position action space."""

    events: DemoCollectEventCfg = DemoCollectEventCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()
    action_discretization: ActionDiscretizationCfg = ActionDiscretizationCfg()
    terminations: FromDemoCollectTerminationsCfg = FromDemoCollectTerminationsCfg()
    observations: DataCollectObservationsCfg = DataCollectObservationsCfg()
    commands: GoalCommandsCfg = GoalCommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = EXPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.events.randomize_robot_actuator_parameters = EventTerm(
            func=omni_reset_mdp.randomize_operational_space_controller_gains,
            mode="reset",
            params={
                "action_name": "arm",
                "stiffness_distribution_params": (0.7, 1.3),
                "damping_distribution_params": (0.9, 1.1),
                "operation": "scale",
                "distribution": "uniform",
            },
        )

@configclass
class Ur5eRobotiq2f85RelJointPosFromDemoTrainCfg(Ur5eRobotiq2f85RlStateCfg):
    """Demo collection configuration for Relative Joint Position action space."""

    events: FromDemoTrainEventCfg = FromDemoTrainEventCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()
    rewards: FromDemoRewardsCfg = FromDemoRewardsCfg()
    observations: TrainingObservationsCfg = TrainingObservationsCfg()
    context: DemoContextCfg = DemoContextCfg()
    commands: TrackingCommandsCfg = TrackingCommandsCfg()
    terminations: FromDemoTrainTerminationsCfg = FromDemoTrainTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = EXPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # self.events.randomize_robot_actuator_parameters = EventTerm(
        #     func=omni_reset_mdp.randomize_operational_space_controller_gains,
        #     mode="reset",
        #     params={
        #         "action_name": "arm",
        #         "stiffness_distribution_params": (0.7, 1.3),
        #         "damping_distribution_params": (0.9, 1.1),
        #         "operation": "scale",
        #         "distribution": "uniform",
        #     },
        # )

@configclass
class Ur5eRobotiq2f85RelJointPosFromDemoSupervisedEvalCfg(Ur5eRobotiq2f85RelJointPosFromDemoTrainCfg):

    observations: SupervisedEvalObservationsCfg = SupervisedEvalObservationsCfg()
    context: DemoEvalContextCfg = DemoEvalContextCfg()
    terminations: EpisodeTimeoutTerminationsCfg = EpisodeTimeoutTerminationsCfg()

@configclass
class Ur5eRobotiq2f85RelJointPosFromDemoPriviledgedTrainCfg(Ur5eRobotiq2f85RlStateCfg):
    """Demo collection configuration for Relative Joint Position action space."""

    events: FromDemoPriviledgedTrainEventCfg = FromDemoPriviledgedTrainEventCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()
    rewards: PriviledgedFromDemoRewardsCfg = PriviledgedFromDemoRewardsCfg()
    observations: PriviledgedTrainingObservationsCfg = PriviledgedTrainingObservationsCfg()
    context: DemoContextPriviledgedCfg = DemoContextPriviledgedCfg()
    commands: TrackingCommandsCfg = TrackingCommandsCfg()
    terminations: FromDemoTrainTerminationsCfg = FromDemoTrainTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = EXPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class Ur5eRobotiq2f85RelJointPosFromDemoEvalCfg(Ur5eRobotiq2f85RlStateCfg):
    """Demo collection configuration for Relative Joint Position action space."""

    events: FromDemoEvalEventCfg = FromDemoEvalEventCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()
    rewards: FromDemoRewardsCfg = FromDemoRewardsCfg()
    observations: TrainingObservationsCfg = TrainingObservationsCfg()
    context: DemoEvalContextCfg = DemoEvalContextCfg()
    commands: TrackingCommandsCfg = TrackingCommandsCfg()
    terminations: FromDemoTrainTerminationsCfg = FromDemoTrainTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = EXPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # self.events.randomize_robot_actuator_parameters = EventTerm(
        #     func=omni_reset_mdp.randomize_operational_space_controller_gains,
        #     mode="reset",
        #     params={
        #         "action_name": "arm",
        #         "stiffness_distribution_params": (0.7, 1.3),
        #         "damping_distribution_params": (0.9, 1.1),
        #         "operation": "scale",
        #         "distribution": "uniform",
        #     },
        # )
