"""Offline tests for corrective label generation."""

from __future__ import annotations

import argparse

import torch

from metalearning.corrective_labels import (
    CorrectiveLabeler,
    CorrectiveLabelerConfig,
    sample_perturbations_obs_space,
    validate_quat_convention,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)


def _test_quat_helpers() -> None:
    q_xyzw = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
    q_wxyz = xyzw_to_wxyz(q_xyzw)
    assert torch.allclose(q_wxyz, torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32), atol=1e-6)
    q_xyzw_roundtrip = wxyz_to_xyzw(q_wxyz)
    assert torch.allclose(q_xyzw, q_xyzw_roundtrip, atol=1e-6)

    q_wxyz_auto, inferred_auto = validate_quat_convention(q_xyzw, metadata_order="auto")
    assert inferred_auto == "xyzw"
    assert torch.allclose(q_wxyz_auto, q_wxyz, atol=1e-6)


def _build_labeler() -> CorrectiveLabeler:
    q_grip_xyzw = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32)
    q_aro_xyzw = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
    q_grip_wxyz, _ = validate_quat_convention(q_grip_xyzw, metadata_order="xyzw")
    q_aro_wxyz, _ = validate_quat_convention(q_aro_xyzw, metadata_order="xyzw")
    cfg = CorrectiveLabelerConfig(
        scale=(0.02, 0.02, 0.02, 0.02, 0.02, 0.2),
        input_clip=None,
        action_clip=None,
        rot_clip=None,
        invalid_handling="assert",
    )
    return CorrectiveLabeler(cfg=cfg, q_grip_wxyz=q_grip_wxyz, q_action_root_offset_wxyz=q_aro_wxyz)


def _test_shapes_and_finite(num_rows: int, k: int) -> None:
    torch.manual_seed(0)
    labeler = _build_labeler()
    obs_t = torch.randn(num_rows, 6, dtype=torch.float32) * 0.05
    a_t = torch.randn(num_rows, 7, dtype=torch.float32) * 0.1
    a_t[:, 6] = torch.sign(a_t[:, 6])

    obs_g = sample_perturbations_obs_space(obs_t, num_samples_per_step=k, sigma_p=0.01, sigma_r=0.05)
    obs_t_rep = obs_t.repeat_interleave(k, dim=0)
    a_t_rep = a_t.repeat_interleave(k, dim=0)
    a_g = labeler.label(obs_t_rep, a_t_rep, obs_g)

    assert a_g.shape == a_t_rep.shape
    assert torch.isfinite(a_g).all()


def _test_identity_perturbation_consistency(num_rows: int) -> None:
    torch.manual_seed(1)
    labeler = _build_labeler()
    obs_t = torch.randn(num_rows, 6, dtype=torch.float32) * 0.03
    a_t = torch.randn(num_rows, 7, dtype=torch.float32) * 0.05
    a_t[:, 6] = torch.sign(a_t[:, 6])

    # If obs_g == obs_t and no clipping is active, corrective arm action
    # should recover the original raw arm action.
    a_g = labeler.label(obs_t, a_t, obs_t.clone())
    arm_err = torch.max(torch.abs(a_g[:, :6] - a_t[:, :6])).item()
    grip_err = torch.max(torch.abs(a_g[:, 6] - a_t[:, 6])).item()
    assert arm_err < 5e-4, f"Arm consistency error too large: {arm_err}"
    assert grip_err < 1e-6, f"Gripper consistency error too large: {grip_err}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline corrective-label tests.")
    parser.add_argument("--num-rows", type=int, default=64)
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    _test_quat_helpers()
    _test_shapes_and_finite(args.num_rows, args.k)
    _test_identity_perturbation_consistency(args.num_rows)
    print("[INFO] All corrective-label tests passed.")


if __name__ == "__main__":
    main()

