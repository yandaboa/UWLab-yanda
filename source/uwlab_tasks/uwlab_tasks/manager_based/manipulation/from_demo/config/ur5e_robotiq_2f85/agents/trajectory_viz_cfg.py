# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass


@configclass
class TrajectoryVizCfg:
    """Configuration for demo-vs-rollout trajectory logging."""

    enable: bool = False
    log_every_iters: int = 100
    max_pairs_per_log: int = 4
    obs_key: str = "debug/end_effector_pose"
    demo_obs_key: str = "end_effector_pose"
    save_pairs: bool = False
