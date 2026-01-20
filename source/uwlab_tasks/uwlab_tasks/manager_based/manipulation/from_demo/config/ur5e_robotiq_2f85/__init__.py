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
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelJointPosDemoCollectCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.collect_demos_cfg:CollectDemosPolicyRunnerCfg",
    },
)