# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command terms that track demo errors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
import isaaclab.utils.math as math_utils

from uwlab_tasks.manager_based.manipulation.reset_states import mdp as reset_states_mdp
from .context import (
    tracking_end_effector_orientation_error,
    tracking_end_effector_position_error,
)
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class DemoTrackingMetricsCommand(CommandTerm):
    """Command term that logs demo tracking errors."""

    cfg: Any

    def __init__(self, cfg: Any, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_cfg.name]
        self.metrics["joint_tracking_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["ee_position_tracking_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["ee_orientation_tracking_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["demo_success_match_rate"] = torch.zeros(self.num_envs, device=self.device)
        self._command = torch.zeros((self.num_envs, 1), device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Return a dummy command tensor for compatibility."""
        return self._command

    def _update_metrics(self):
        env = cast("ManagerBasedRLEnv", self._env)
        context_term = _get_tracking_context(env)
        demo_obs_dict = context_term.demo_obs_dict
        demo_lengths = context_term.demo_obs_lengths
        end_idx = (demo_lengths - 1).clamp(min=0)
        t = torch.minimum(env.episode_length_buf.long(), end_idx)

        demo_q_flat = demo_obs_dict["demo_link_quats"]
        env_ids = torch.arange(env.num_envs, device=demo_q_flat.device)
        # Per-env advanced indexing for timestep-aligned demo targets.
        q_ref_flat = demo_q_flat[env_ids, t]

        q_cur = self.robot.data.body_link_quat_w
        q_root = self.robot.data.root_link_quat_w
        # Root-relative link quats to mirror demo preprocessing.
        q_cur = math_utils.quat_mul(
            math_utils.quat_conjugate(q_root)[:, None, :].expand(q_cur.shape[0], q_cur.shape[1], 4),
            q_cur,
        )
        B = q_cur.shape[1]
        q_ref = q_ref_flat.view(q_ref_flat.shape[0], B, 4)
        q_ref = math_utils.normalize(q_ref)
        q_cur = math_utils.normalize(q_cur)

        # Relative rotation: q_ref ⊗ q_cur^{-1}.
        q_rel = math_utils.quat_mul(q_ref, math_utils.quat_conjugate(q_cur))
        ang = _quat_angle_from_rel(q_rel)
        self.metrics["joint_tracking_error"][:] = torch.sum(ang * ang, dim=-1)

        self.metrics["ee_position_tracking_error"][:] = tracking_end_effector_position_error(env)
        self.metrics["ee_orientation_tracking_error"][:] = tracking_end_effector_orientation_error(env)

        demo_success = getattr(context_term, "demo_success", None)
        assert demo_success is not None, "Demo context is missing demo_success."
        sim_success = reset_states_mdp.calculate_successes(env).to(torch.bool)
        if torch.any(demo_success):
            matched = torch.logical_and(demo_success, sim_success)
            rate = matched.sum(dtype=torch.float32) / demo_success.sum(dtype=torch.float32)
        else:
            rate = torch.zeros((), device=self.device)
        self.metrics["demo_success_match_rate"][:] = rate

    def _resample_command(self, env_ids):
        pass

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass


def _get_tracking_context(env: Any):
    context = getattr(env, "context", None)
    if context is None:
        raise AttributeError("Env has no demo context. Expected env.context to be initialized.")
    return context


def _quat_angle_from_rel(q_rel_wxyz: torch.Tensor) -> torch.Tensor:
    """q_rel: (...,4) wxyz -> angle in radians (...,)"""
    q_rel = math_utils.normalize(q_rel_wxyz)
    w = torch.clamp(torch.abs(q_rel[..., 0]), 0.0, 1.0)
    return 2.0 * torch.acos(w)
