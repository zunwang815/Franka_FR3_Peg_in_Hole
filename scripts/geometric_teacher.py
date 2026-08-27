"""Shared geometric teacher action rules for the Pose6D peg-in-hole task.

The teacher consumes the same relative quantities exposed to the policy:
peg-to-hole displacement, insertion depth, peg tilt and an orientation error.
It never needs the hidden mount joint or an absolute world pose.
"""

from __future__ import annotations

import math

import torch


def compute_geometric_action(
    delta_xy: torch.Tensor,
    depth: torch.Tensor,
    tilt: torch.Tensor,
    rot_error: torch.Tensor,
    *,
    position_scale: float = 0.005,
    kp_position: float = 0.8,
    kp_orientation: float = 0.8,
    approach_depth_mm: float = -10.0,
    insert_depth_mm: float = 30.0,
    alignment_gate_mm: float = 1.0,
    tilt_gate_deg: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return bounded 6-D OSC action and the insertion-phase mask.

    ``delta_xy`` is the vector from the measured peg tip to the measured hole
    center. ``depth`` is positive below the hole surface. ``rot_error`` is the
    desired small-angle orientation correction in the OSC task frame.
    """

    if delta_xy.ndim != 2 or delta_xy.shape[-1] != 2:
        raise ValueError("delta_xy must have shape (N, 2)")
    if depth.ndim != 1 or tilt.ndim != 1:
        raise ValueError("depth and tilt must have shape (N,)")
    if rot_error.ndim != 2 or rot_error.shape[-1] != 3:
        raise ValueError("rot_error must have shape (N, 3)")
    if not position_scale > 0.0:
        raise ValueError("position_scale must be positive")

    radial = torch.linalg.vector_norm(delta_xy, dim=-1)
    insert_mask = (radial <= alignment_gate_mm / 1000.0) & (
        tilt <= tilt_gate_deg * math.pi / 180.0
    )
    target_depth = torch.where(
        insert_mask,
        torch.full_like(depth, insert_depth_mm / 1000.0),
        torch.full_like(depth, approach_depth_mm / 1000.0),
    )

    # Current depth minus target depth is the desired tip displacement along
    # the action Z axis: negative means moving down toward the fixture.
    pos_delta = torch.cat(
        (
            kp_position * delta_xy,
            (kp_position * (depth - target_depth)).unsqueeze(-1),
        ),
        dim=-1,
    )
    pos_action = torch.clamp(pos_delta / float(position_scale), -1.0, 1.0)
    rot_action = torch.clamp(kp_orientation * rot_error, -0.25, 0.25)
    action = torch.cat((pos_action, rot_action), dim=-1)
    action[insert_mask, 2] = torch.clamp(action[insert_mask, 2], -0.8, 0.8)
    return action, insert_mask


def upright_axis_error(axis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return tilt angle and minimum-angle rotation vector to upright the peg.

    The sign of the target axis follows the current axis, so a downward-facing
    peg is kept downward and yaw remains unconstrained for the cylindrical peg.
    """

    if axis.ndim != 2 or axis.shape[-1] != 3:
        raise ValueError("axis must have shape (N, 3)")
    axis = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(1.0e-8)
    world_z = torch.zeros_like(axis)
    world_z[:, 2] = 1.0
    target_axis = world_z * torch.where(axis[:, 2:3] >= 0.0, 1.0, -1.0)
    cross = torch.linalg.cross(axis, target_axis)
    sin_angle = torch.linalg.vector_norm(cross, dim=-1)
    cos_angle = torch.sum(axis * target_axis, dim=-1).clamp(-1.0, 1.0)
    tilt = torch.atan2(sin_angle, cos_angle)
    rot_axis = cross / sin_angle.unsqueeze(-1).clamp_min(1.0e-8)
    return tilt, rot_axis * tilt.unsqueeze(-1)


def action_from_policy_observation(
    observations: torch.Tensor,
    *,
    position_scale: float = 0.005,
    kp_position: float = 0.8,
    kp_orientation: float = 0.8,
    approach_depth_mm: float = -10.0,
    insert_depth_mm: float = 30.0,
    alignment_gate_mm: float = 1.0,
    tilt_gate_deg: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the teacher action from the Pose6D relative policy observation.

    The shared feature layout is q(9), qdot(9), peg-to-hole(3), peg tilt(3);
    optional target-id and previous-action features may follow.
    """

    if observations.ndim != 2 or observations.shape[-1] < 24:
        raise ValueError("Pose6D policy observations must have at least 24 features")
    delta_xy = observations[:, 18:20]
    peg_to_hole_z = observations[:, 20]
    axis = observations[:, 21:24]
    depth = peg_to_hole_z + 0.010
    tilt, rot_error = upright_axis_error(axis)
    return compute_geometric_action(
        delta_xy,
        depth,
        tilt,
        rot_error,
        position_scale=position_scale,
        kp_position=kp_position,
        kp_orientation=kp_orientation,
        approach_depth_mm=approach_depth_mm,
        insert_depth_mm=insert_depth_mm,
        alignment_gate_mm=alignment_gate_mm,
        tilt_gate_deg=tilt_gate_deg,
    )
