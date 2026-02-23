import torch
from typing import cast
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor
import isaaclab.utils.math as math_utils

def demo_link_quats(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    root_relative: bool = True,
    flatten: bool = True,
) -> torch.Tensor:
    """
    Returns reference quaternions for ALL links of the robot.

    Output:
      - if flatten=True:  [num_envs, num_bodies*4]
      - else:            [num_envs, num_bodies, 4]

    Quat format is wxyz (matches IsaacLab's body_link_quat_w).
    """
    asset: Articulation = env.scene[asset_cfg.name]

    q_link_w = asset.data.body_link_quat_w  # [N, B, 4] wxyz

    if root_relative:
        q_root = asset.data.root_link_quat_w  # [N, 4]
        q_root_inv = math_utils.quat_conjugate(q_root)  # [N, 4]

        # Manually broadcast to match [N, B, 4] (since quat_mul checks shape equality)
        B = q_link_w.shape[1]
        q_root_inv_nb = q_root_inv[:, None, :].expand(-1, B, -1)  # [N, B, 4]

        # q_rel = q_root^{-1} ⊗ q_link
        q_link_w = math_utils.quat_mul(q_root_inv_nb, q_link_w)  # [N, B, 4]

    q_link_w = math_utils.normalize(q_link_w)

    if flatten:
        return q_link_w.reshape(q_link_w.shape[0], -1)
    return q_link_w


def end_effector_quat(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="robotiq_base_link"),
    root_relative: bool = True,
) -> torch.Tensor:
    """Returns end-effector rotation as a quaternion (wxyz)."""
    asset: Articulation = env.scene[asset_cfg.name]

    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        body_idx = 0
    elif isinstance(body_ids, (list, tuple)):
        assert len(body_ids) == 1, "end_effector_quat expects a single body id"
        body_idx = body_ids[0]
    else:
        body_idx = body_ids

    q_link_w = asset.data.body_link_quat_w[:, body_idx].view(-1, 4)

    if root_relative:
        q_root = asset.data.root_link_quat_w
        q_root_inv = math_utils.quat_conjugate(q_root)
        q_link_w = math_utils.quat_mul(q_root_inv, q_link_w)

    return math_utils.normalize(q_link_w)


def receptive_object_gripped_by_ee(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("receptive_object_grip_contact"),
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Binary contact flag for receptive-object contact with gripper/EE links."""
    contact_sensor = cast(ContactSensor, env.scene.sensors[sensor_cfg.name])
    force_matrix_w = contact_sensor.data.force_matrix_w
    assert force_matrix_w is not None, "Contact sensor must define filter_prim_paths_expr for grip detection."
    # force_matrix_w has shape [num_envs, num_sensor_bodies, num_filters, 3]
    in_contact = torch.linalg.norm(force_matrix_w, dim=-1) > force_threshold
    return in_contact.any(dim=(1, 2)).to(force_matrix_w.dtype).unsqueeze(-1)