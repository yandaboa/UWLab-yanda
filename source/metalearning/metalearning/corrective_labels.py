"""Corrective action label generation for relative OSC actions.

This module reconstructs the instantaneous OSC target implied by (obs_t, a_t)
and computes the corrective action a_g from a perturbed observation obs_g.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

try:
    import isaaclab.utils.math as math_utils
except ModuleNotFoundError:
    class _TorchMathUtils:
        """Minimal torch fallback for quaternion math used in this module."""

        @staticmethod
        def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
            return x / torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(eps)

        @staticmethod
        def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
            w1, x1, y1, z1 = q1.unbind(dim=-1)
            w2, x2, y2, z2 = q2.unbind(dim=-1)
            return torch.stack(
                [
                    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                ],
                dim=-1,
            )

        @staticmethod
        def quat_inv(q: torch.Tensor) -> torch.Tensor:
            qn = _TorchMathUtils.normalize(q)
            out = qn.clone()
            out[..., 1:] = -out[..., 1:]
            return out

        @staticmethod
        def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
            half = 0.5 * angle
            sin_h = torch.sin(half)
            cos_h = torch.cos(half)
            axis_n = _TorchMathUtils.normalize(axis)
            return torch.cat([cos_h.unsqueeze(-1), axis_n * sin_h.unsqueeze(-1)], dim=-1)

        @staticmethod
        def axis_angle_from_quat(q: torch.Tensor) -> torch.Tensor:
            qn = _TorchMathUtils.normalize(q)
            w = qn[..., 0].clamp(-1.0, 1.0)
            xyz = qn[..., 1:]
            xyz_norm = torch.linalg.norm(xyz, dim=-1, keepdim=True)
            angle = 2.0 * torch.atan2(xyz_norm.squeeze(-1), w)
            axis = xyz / xyz_norm.clamp_min(1e-8)
            return axis * angle.unsqueeze(-1)

        @staticmethod
        def matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
            qn = _TorchMathUtils.normalize(q)
            w, x, y, z = qn.unbind(dim=-1)
            xx, yy, zz = x * x, y * y, z * z
            xy, xz, yz = x * y, x * z, y * z
            wx, wy, wz = w * x, w * y, w * z

            m00 = 1 - 2 * (yy + zz)
            m01 = 2 * (xy - wz)
            m02 = 2 * (xz + wy)
            m10 = 2 * (xy + wz)
            m11 = 1 - 2 * (xx + zz)
            m12 = 2 * (yz - wx)
            m20 = 2 * (xz - wy)
            m21 = 2 * (yz + wx)
            m22 = 1 - 2 * (xx + yy)
            return torch.stack(
                [
                    torch.stack([m00, m01, m02], dim=-1),
                    torch.stack([m10, m11, m12], dim=-1),
                    torch.stack([m20, m21, m22], dim=-1),
                ],
                dim=-2,
            )

    math_utils = _TorchMathUtils()


def xyzw_to_wxyz(q_xyzw: torch.Tensor) -> torch.Tensor:
    """Convert quaternion tensor from xyzw to wxyz ordering."""
    if q_xyzw.shape[-1] != 4:
        raise ValueError(f"Expected last dim=4, got {q_xyzw.shape}.")
    return torch.cat([q_xyzw[..., 3:4], q_xyzw[..., 0:3]], dim=-1)


def wxyz_to_xyzw(q_wxyz: torch.Tensor) -> torch.Tensor:
    """Convert quaternion tensor from wxyz to xyzw ordering."""
    if q_wxyz.shape[-1] != 4:
        raise ValueError(f"Expected last dim=4, got {q_wxyz.shape}.")
    return torch.cat([q_wxyz[..., 1:4], q_wxyz[..., 0:1]], dim=-1)


def validate_quat_convention(
    quat: torch.Tensor,
    metadata_order: Literal["auto", "xyzw", "wxyz"] = "auto",
) -> tuple[torch.Tensor, Literal["xyzw", "wxyz"]]:
    """Return a quaternion in wxyz format and the inferred source order."""
    if quat.ndim != 1 or quat.numel() != 4:
        raise ValueError(f"Expected shape [4], got {tuple(quat.shape)}.")
    if not torch.isfinite(quat).all():
        raise ValueError("Quaternion contains non-finite values.")

    if metadata_order == "wxyz":
        q_wxyz = quat
        inferred = "wxyz"
    elif metadata_order == "xyzw":
        q_wxyz = xyzw_to_wxyz(quat)
        inferred = "xyzw"
    else:
        # Heuristic: metadata identity often appears as [0,0,0,1] (xyzw).
        if torch.abs(quat[3] - 1.0) < 1e-3 and torch.linalg.norm(quat[:3]) < 1e-3:
            q_wxyz = xyzw_to_wxyz(quat)
            inferred = "xyzw"
        else:
            q_wxyz = quat
            inferred = "wxyz"

    q_wxyz = math_utils.normalize(q_wxyz.unsqueeze(0)).squeeze(0)
    return q_wxyz, inferred


@dataclass(frozen=True)
class CorrectiveLabelerConfig:
    """Configuration for corrective label generation."""

    scale: tuple[float, float, float, float, float, float] = (0.02, 0.02, 0.02, 0.02, 0.02, 0.2)
    input_clip: tuple[float, float] | None = None
    action_clip: tuple[float, float] | None = (-1.0, 1.0)
    rot_clip: float | None = None
    eps: float = 1e-8
    invalid_handling: Literal["assert", "skip"] = "assert"


class CorrectiveLabeler:
    """Compute corrective labels for a relative OSC arm + gripper action."""

    def __init__(
        self,
        cfg: CorrectiveLabelerConfig,
        q_grip_wxyz: torch.Tensor,
        q_action_root_offset_wxyz: torch.Tensor | None = None,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        if q_grip_wxyz.shape != (4,):
            raise ValueError(f"Expected q_grip_wxyz shape [4], got {tuple(q_grip_wxyz.shape)}.")

        resolved_device = device if device is not None else q_grip_wxyz.device
        self._device = torch.device(resolved_device)
        self.cfg = cfg

        self.scale = torch.tensor(cfg.scale, dtype=torch.float32, device=self._device)
        self.input_clip = (
            torch.tensor(cfg.input_clip, dtype=torch.float32, device=self._device) if cfg.input_clip is not None else None
        )
        self.action_clip = (
            torch.tensor(cfg.action_clip, dtype=torch.float32, device=self._device)
            if cfg.action_clip is not None
            else None
        )
        self.rot_clip = cfg.rot_clip
        self.eps = cfg.eps
        self.invalid_handling = cfg.invalid_handling

        self.q_grip = math_utils.normalize(q_grip_wxyz.to(self._device, dtype=torch.float32).unsqueeze(0)).squeeze(0)
        if q_action_root_offset_wxyz is None:
            self.q_aro = None
        else:
            if q_action_root_offset_wxyz.shape != (4,):
                raise ValueError(
                    f"Expected q_action_root_offset_wxyz shape [4], got {tuple(q_action_root_offset_wxyz.shape)}."
                )
            self.q_aro = math_utils.normalize(
                q_action_root_offset_wxyz.to(self._device, dtype=torch.float32).unsqueeze(0)
            ).squeeze(0)

    @property
    def device(self) -> torch.device:
        return self._device

    def _assert_finite(self, name: str, x: torch.Tensor) -> None:
        if not torch.isfinite(x).all():
            raise ValueError(f"{name} contains non-finite values.")

    def _quat_from_axis_angle_vec(self, axis_angle: torch.Tensor) -> torch.Tensor:
        angle = torch.linalg.norm(axis_angle, dim=-1).clamp_min(self.eps)
        axis = axis_angle / angle.unsqueeze(-1)
        q = math_utils.quat_from_angle_axis(angle, axis)
        return math_utils.normalize(q)

    def _axis_angle_from_quat_safe(self, quat: torch.Tensor) -> torch.Tensor:
        quat = math_utils.normalize(quat)
        axis_angle = math_utils.axis_angle_from_quat(quat)
        return axis_angle

    def _rotate_vectors(self, rot_mats: torch.Tensor, vecs: torch.Tensor) -> torch.Tensor:
        return torch.bmm(rot_mats, vecs.unsqueeze(-1)).squeeze(-1)

    def _make_batched_quat(self, quat: torch.Tensor, batch_size: int) -> torch.Tensor:
        return quat.unsqueeze(0).expand(batch_size, 4)

    def _convert_obs_to_ctrl_quat(self, obs_pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos_obs = obs_pose[:, :3]
        aa_obs = obs_pose[:, 3:6]
        q_obs = self._quat_from_axis_angle_vec(aa_obs)
        q_obs = math_utils.normalize(q_obs)
        q_grip_inv = math_utils.quat_inv(self._make_batched_quat(self.q_grip, obs_pose.shape[0]))
        q_ctrl = math_utils.quat_mul(q_obs, q_grip_inv)
        q_ctrl = math_utils.normalize(q_ctrl)
        return pos_obs, q_ctrl

    def _to_standard_delta(self, arm_action_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scaled_offset = arm_action_raw * self.scale.unsqueeze(0)
        if self.input_clip is not None:
            scaled_offset = torch.clamp(scaled_offset, min=self.input_clip[0], max=self.input_clip[1])

        dp_off = scaled_offset[:, :3]
        dr_off = scaled_offset[:, 3:6]
        if self.q_aro is None:
            return dp_off, dr_off

        q_aro_b = self._make_batched_quat(self.q_aro, arm_action_raw.shape[0])
        r_off_to_std = math_utils.matrix_from_quat(math_utils.quat_inv(q_aro_b))
        dp_std = self._rotate_vectors(r_off_to_std, dp_off)
        dr_std = self._rotate_vectors(r_off_to_std, dr_off)
        return dp_std, dr_std

    def _to_offset_delta(self, dp_std: torch.Tensor, dr_std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.q_aro is None:
            return dp_std, dr_std
        q_aro_b = self._make_batched_quat(self.q_aro, dp_std.shape[0])
        r_std_to_off = math_utils.matrix_from_quat(q_aro_b)
        dp_off = self._rotate_vectors(r_std_to_off, dp_std)
        dr_off = self._rotate_vectors(r_std_to_off, dr_std)
        return dp_off, dr_off

    def compute_target_pose(self, obs_t: torch.Tensor, a_t_arm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute instantaneous target pose implied by (obs_t, a_t_arm)."""
        if obs_t.ndim != 2 or obs_t.shape[1] != 6:
            raise ValueError(f"Expected obs_t shape [N,6], got {tuple(obs_t.shape)}.")
        if a_t_arm.ndim != 2 or a_t_arm.shape[1] != 6:
            raise ValueError(f"Expected a_t_arm shape [N,6], got {tuple(a_t_arm.shape)}.")

        obs_t = obs_t.to(self.device, dtype=torch.float32)
        a_t_arm = a_t_arm.to(self.device, dtype=torch.float32)
        self._assert_finite("obs_t", obs_t)
        self._assert_finite("a_t_arm", a_t_arm)

        p_ctrl_t, q_ctrl_t = self._convert_obs_to_ctrl_quat(obs_t)
        dp_std, dr_std = self._to_standard_delta(a_t_arm)
        dq_std = self._quat_from_axis_angle_vec(dr_std)

        p_target = p_ctrl_t + dp_std
        q_target = math_utils.normalize(math_utils.quat_mul(q_ctrl_t, dq_std))

        self._assert_finite("p_target", p_target)
        self._assert_finite("q_target", q_target)
        return p_target, q_target

    def compute_corrective_action(
        self,
        obs_g: torch.Tensor,
        p_target: torch.Tensor,
        q_target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute corrective 6D arm action from perturbed observation to target."""
        if obs_g.ndim != 2 or obs_g.shape[1] != 6:
            raise ValueError(f"Expected obs_g shape [N,6], got {tuple(obs_g.shape)}.")
        if p_target.ndim != 2 or p_target.shape[1] != 3:
            raise ValueError(f"Expected p_target shape [N,3], got {tuple(p_target.shape)}.")
        if q_target.ndim != 2 or q_target.shape[1] != 4:
            raise ValueError(f"Expected q_target shape [N,4], got {tuple(q_target.shape)}.")

        obs_g = obs_g.to(self.device, dtype=torch.float32)
        p_target = p_target.to(self.device, dtype=torch.float32)
        q_target = math_utils.normalize(q_target.to(self.device, dtype=torch.float32))
        self._assert_finite("obs_g", obs_g)
        self._assert_finite("p_target", p_target)
        self._assert_finite("q_target", q_target)

        p_ctrl_g, q_ctrl_g = self._convert_obs_to_ctrl_quat(obs_g)
        dp_needed_std = p_target - p_ctrl_g
        dq_needed_std = math_utils.normalize(math_utils.quat_mul(math_utils.quat_inv(q_ctrl_g), q_target))
        dr_needed_std = self._axis_angle_from_quat_safe(dq_needed_std)

        dp_needed_off, dr_needed_off = self._to_offset_delta(dp_needed_std, dr_needed_std)
        if self.rot_clip is not None:
            dr_norm = torch.linalg.norm(dr_needed_off, dim=-1, keepdim=True)
            scale = torch.clamp(self.rot_clip / (dr_norm + self.eps), max=1.0)
            dr_needed_off = dr_needed_off * scale

        a_arm = torch.cat(
            [
                dp_needed_off / self.scale[:3].unsqueeze(0),
                dr_needed_off / self.scale[3:6].unsqueeze(0),
            ],
            dim=-1,
        )
        if self.action_clip is not None:
            a_arm = torch.clamp(a_arm, min=self.action_clip[0], max=self.action_clip[1])

        self._assert_finite("a_arm", a_arm)
        return a_arm

    def label(
        self,
        obs_t: torch.Tensor,
        a_t: torch.Tensor,
        obs_g: torch.Tensor,
        *,
        return_valid_mask: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute full 7D corrective action `[arm6, gripper1]`."""
        if a_t.ndim != 2 or a_t.shape[1] != 7:
            raise ValueError(f"Expected a_t shape [N,7], got {tuple(a_t.shape)}.")
        if obs_t.shape[0] != a_t.shape[0] or obs_g.shape[0] != a_t.shape[0]:
            raise ValueError("obs_t, a_t, and obs_g must have the same batch size.")

        obs_t = obs_t.to(self.device, dtype=torch.float32)
        obs_g = obs_g.to(self.device, dtype=torch.float32)
        a_t = a_t.to(self.device, dtype=torch.float32)
        valid_mask = torch.isfinite(obs_t).all(dim=-1) & torch.isfinite(obs_g).all(dim=-1) & torch.isfinite(a_t).all(dim=-1)

        if self.invalid_handling == "assert":
            if not valid_mask.all():
                raise ValueError("Non-finite input rows detected.")
            p_target, q_target = self.compute_target_pose(obs_t, a_t[:, :6])
            a_arm = self.compute_corrective_action(obs_g, p_target, q_target)
            a_g = torch.cat([a_arm, a_t[:, 6:7]], dim=-1)
            if return_valid_mask:
                return a_g, valid_mask
            return a_g

        # skip mode: compute on valid rows only, zero-fill invalid rows.
        a_g = torch.zeros_like(a_t)
        if torch.any(valid_mask):
            idx = torch.nonzero(valid_mask, as_tuple=True)[0]
            p_target, q_target = self.compute_target_pose(obs_t[idx], a_t[idx, :6])
            a_arm = self.compute_corrective_action(obs_g[idx], p_target, q_target)
            a_g[idx, :6] = a_arm
            a_g[idx, 6:7] = a_t[idx, 6:7]
        if return_valid_mask:
            return a_g, valid_mask
        return a_g


def sample_perturbations_obs_space(
    obs_t: torch.Tensor,
    *,
    num_samples_per_step: int,
    sigma_p: float = 0.01,
    sigma_r: float = 0.05,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample perturbed end-effector observations in obs space.

    Args:
        obs_t: Tensor [N, 6] with [pos(3), axis_angle(3)].
        num_samples_per_step: Number of perturbation samples K per row.
        sigma_p: Position noise std (meters).
        sigma_r: Axis-angle noise std (radians).
    Returns:
        obs_g: Tensor [N*K, 6].
    """
    if obs_t.ndim != 2 or obs_t.shape[1] != 6:
        raise ValueError(f"Expected obs_t shape [N,6], got {tuple(obs_t.shape)}.")
    if num_samples_per_step <= 0:
        raise ValueError("num_samples_per_step must be > 0.")

    obs_t = obs_t.to(dtype=torch.float32)
    base = obs_t.repeat_interleave(num_samples_per_step, dim=0)
    pos_noise = torch.randn(base.shape[0], 3, device=base.device, generator=generator) * sigma_p
    rot_noise = torch.randn(base.shape[0], 3, device=base.device, generator=generator) * sigma_r
    obs_g = base.clone()
    obs_g[:, :3] = obs_g[:, :3] + pos_noise
    obs_g[:, 3:6] = obs_g[:, 3:6] + rot_noise
    return obs_g

