# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset states tasks for IsaacLab."""

import gymnasium as gym

from . import agents

# Register the demo collect environment
gym.register(
    id="OmniFromDemo-UR5eRobotiq2f85-CollectDemos-v0",
    entry_point="uwlab_tasks.manager_based.manipulation.from_demo.env:FromDemoEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelJointPosDemoCollectCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.collect_demos_cfg:CollectDemosPolicyRunnerCfg",
    },
)

gym.register(
    id="OmniFromDemo-UR5eRobotiq2f85-Train-v0",
    entry_point="uwlab_tasks.manager_based.manipulation.from_demo.env:FromDemoEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelJointPosFromDemoTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:PPOWithContextRunnerCfg",
    },
)

gym.register(
    id="OmniFromDemo-UR5eRobotiq2f85-PriviledgedTrain-v0",
    entry_point="uwlab_tasks.manager_based.manipulation.from_demo.env:FromDemoEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelJointPosFromDemoPriviledgedTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.priviledged_rsl_rl_cfg:PPOWithPriviledgedRunnerCfg",
    },
)

gym.register(
    id="OmniFromDemo-UR5eRobotiq2f85-Eval-v0",
    entry_point="uwlab_tasks.manager_based.manipulation.from_demo.env:FromDemoEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelJointPosFromDemoTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:PPOWithContextRunnerCfg",
    },
)

gym.register(
    id="OmniFromDemo-UR5eRobotiq2f85-PriviledgedEval-v0",
    entry_point="uwlab_tasks.manager_based.manipulation.from_demo.env:FromDemoEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelJointPosFromDemoPriviledgedTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.priviledged_rsl_rl_cfg:PPOWithPriviledgedRunnerCfg",
    },
)

gym.register(
    id="OmniFromDemo-UR5eRobotiq2f85-Distillation-v0",
    entry_point="uwlab_tasks.manager_based.manipulation.from_demo.env:FromDemoEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.distillation_rl_state_cfg:Ur5eRobotiq2f85RelJointPosFromDemoDistillationCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.distillation_rsl_rl_cfg:DistillationLongContextRunnerCfg"
        ),
    },
)