"""Unified geometry constants and helpers for the Peg-in-Hole task.

ALL modules (observations, rewards, terminations, events) MUST use these
functions to compute peg and hole geometry. This ensures consistent reference
frames, avoids hardcoded offsets, and eliminates the multi-reference-frame bugs
documented in the project evaluation.

Key constants:
    PEG_RADIUS = 0.0100 m  (10.0 mm radius → 20.0 mm diameter)
    HOLE_RADIUS = 0.0115 m (11.5 mm radius → 23.0 mm diameter)
    CLEARANCE = 0.0015 m   (1.5 mm nominal radial clearance)
    SUCCESS_RADIAL_TOL = 0.0013 m (1.3 mm with 0.2 mm safety margin)
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.sensors import FrameTransformer
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ──── Physical Constants ────

PEG_RADIUS = 0.0100          # 10.0 mm
PEG_LENGTH = 0.1000          # 100.0 mm
HOLE_RADIUS = 0.0115         # 11.5 mm
HOLE_DEPTH = 0.0400          # 40.0 mm
SUCCESS_DEPTH = 0.0300       # 30.0 mm insertion required
SUCCESS_MAX_DEPTH = HOLE_DEPTH  # peg tip must remain inside the physical sleeve
OVER_INSERTION_TOL = 0.0020  # 2mm numerical tolerance below the sleeve bottom

CLEARANCE = HOLE_RADIUS - PEG_RADIUS           # 1.5 mm nominal
SUCCESS_RADIAL_TOL = 0.0013                     # 1.3 mm (0.2 mm safety)
SUCCESS_TILT_TOL_DEG = 2.0                      # 2 degrees
SUCCESS_TILT_TOL_RAD = SUCCESS_TILT_TOL_DEG * torch.pi / 180.0

# Offset from FINGER MIDPOINT to peg tip (consistent with sync_peg_to_ee):
# sync puts peg CENTER at finger_mid + (0,0,0.07); cylinder half-height = 0.05
# → peg TIP at finger_mid + (0,0,0.12) along finger local Z.
PEG_TIP_OFFSET_FROM_FINGERS = (0.0, 0.0, 0.12)


# ──── Public Geometry Helpers ────

def _finger_midpoint(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (finger_mid_world, finger_quat_world) from articulation body state.

    Uses the SAME reference as sync_peg_to_ee — guarantees the reward's peg
    position matches the VISUAL peg position exactly.
    """
    from isaaclab.assets import Articulation
    robot: Articulation = env.scene[robot_cfg.name]

    def find(name):
        indices, _ = robot.find_bodies(name)
        return indices[0] if len(indices) > 0 else None

    l_idx = find("panda_leftfinger")
    r_idx = find("panda_rightfinger")
    if l_idx is not None and r_idx is not None:
        l_pos = robot.data.body_state_w[:, l_idx, :3]
        r_pos = robot.data.body_state_w[:, r_idx, :3]
        l_quat = robot.data.body_state_w[:, l_idx, 3:7]
        return (l_pos + r_pos) / 2.0, l_quat
    # Fallback: panda_hand
    h_idx = find("panda_hand") or find("panda_link8")
    return (robot.data.body_state_w[:, h_idx, :3],
            robot.data.body_state_w[:, h_idx, 3:7])


def get_peg_tip(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (peg_tip_world, peg_quat_world) from finger midpoint.

    Peg tip = finger_mid + (0,0,0.12) along finger local Z — IDENTICAL to the
    visual peg synced by sync_peg_to_ee (center at +0.07, half-height 0.05).
    """
    # Preferred physical model: Peg is a rigid link inside the robot
    # articulation, connected to panda_hand by a FixedJoint.
    from isaaclab.assets import Articulation
    robot: Articulation = env.scene[robot_cfg.name]
    try:
        peg_indices, _ = robot.find_bodies("peg")
    except ValueError:
        # Legacy array environments use a separate RigidObject peg rather
        # than embedding the peg as an articulation link.
        peg_indices = []
    if len(peg_indices) == 1:
        peg_idx = peg_indices[0]
        peg_center = robot.data.body_state_w[:, peg_idx, :3]
        peg_quat = robot.data.body_state_w[:, peg_idx, 3:7]
        half_offset = torch.tensor(
            (0.0, 0.0, PEG_LENGTH / 2.0),
            device=peg_center.device,
            dtype=peg_center.dtype,
        )
        peg_tip = peg_center + quat_apply(peg_quat, half_offset.expand(peg_center.shape[0], -1))
        return peg_tip, peg_quat

    # Legacy array path: the peg is a standalone RigidObject synchronized to
    # the gripper by an interval event. Its root pose is in world coordinates.
    try:
        from isaaclab.assets import RigidObject

        peg: RigidObject = env.scene["peg"]
        peg_center = peg.data.root_pos_w[:, :3]
        peg_quat = peg.data.root_quat_w[:, :4]
        half_offset = torch.tensor(
            (0.0, 0.0, PEG_LENGTH / 2.0),
            device=peg_center.device,
            dtype=peg_center.dtype,
        )
        peg_tip = peg_center + quat_apply(peg_quat, half_offset.expand(peg_center.shape[0], -1))
        return peg_tip, peg_quat
    except (KeyError, AttributeError):
        pass

    # Legacy fallback for checkpoints/assets created before the FixedJoint tool.
    ee_pos, ee_quat = _finger_midpoint(env, robot_cfg)
    offset = torch.tensor(PEG_TIP_OFFSET_FROM_FINGERS, device=ee_pos.device, dtype=ee_pos.dtype)
    peg_tip = ee_pos + quat_apply(ee_quat, offset.expand(ee_pos.shape[0], -1))
    return peg_tip, ee_quat


def get_peg_center(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return peg center world position (midpoint of cylinder axis)."""
    peg_tip, peg_quat = get_peg_tip(env, robot_cfg)
    half_len = PEG_LENGTH / 2.0
    offset = torch.tensor((0.0, 0.0, -half_len), device=peg_tip.device, dtype=peg_tip.dtype)
    return peg_tip + quat_apply(peg_quat, offset.expand(peg_tip.shape[0], -1))


def get_peg_axis(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return peg axis unit vector in world frame (shape: (N, 3))."""
    _, peg_quat = get_peg_tip(env, robot_cfg)
    # Cylinder axis is local +Z, rotated by quaternion
    z_local = torch.tensor((0.0, 0.0, 1.0), device=peg_quat.device, dtype=peg_quat.dtype)
    return quat_apply(peg_quat, z_local.expand(peg_quat.shape[0], -1))


def get_hole_center(
    env: ManagerBasedRLEnv,
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole_board"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (hole_center_world, hole_quat_world) in WORLD frame.

    CRITICAL: RigidObject.root_pos_w is env-LOCAL (relative to env origin),
    while Articulation.body_state_w is WORLD. To compare against the robot,
    we must add the env origin offset (from scene.env_origins).
    """
    from isaaclab.assets import RigidObject
    hole: RigidObject = env.scene[hole_cfg.name]
    pos_w = hole.data.root_pos_w[:, :3] + env.scene.env_origins
    quat_w = hole.data.root_quat_w[:, :4]
    return pos_w, quat_w


def get_radial_error(
    peg_tip: torch.Tensor,
    hole_pos: torch.Tensor,
) -> torch.Tensor:
    """Radial (XY) distance from peg tip to hole center. Shape (N,)."""
    return torch.linalg.vector_norm(peg_tip[:, :2] - hole_pos[:, :2], dim=-1)


def get_tilt_angle(
    peg_quat: torch.Tensor,
) -> torch.Tensor:
    """Angle between peg axis (tool Z) and world Z (vertical). Shape (N,).

    Returns angle in radians. 0 = perfectly vertical.
    """
    # World Z axis
    world_z = torch.tensor((0.0, 0.0, 1.0), device=peg_quat.device, dtype=peg_quat.dtype)
    # Peg local Z rotated by quaternion
    peg_axis = quat_apply(peg_quat, world_z.expand(peg_quat.shape[0], -1))
    # Dot product gives cos(theta)
    cos_theta = torch.abs(torch.sum(peg_axis * world_z, dim=-1))
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    return torch.acos(cos_theta)


def get_insertion_depth(
    peg_tip: torch.Tensor,
    hole_pos: torch.Tensor,
) -> torch.Tensor:
    """Depth of peg tip below hole surface. Positive = inserted. Shape (N,).

    hole_z_surface = hole_pos.z + 0.01 (approximate surface height)
    depth = hole_z_surface - peg_tip.z
    """
    hole_z_surface = hole_pos[:, 2] + 0.01
    return hole_z_surface - peg_tip[:, 2]


def is_in_hole(
    peg_tip: torch.Tensor,
    hole_pos: torch.Tensor,
    peg_quat: torch.Tensor | None = None,
    radial_tol: float | None = None,
    depth_required: float | None = None,
    max_depth: float | None = None,
    tilt_tol: float | None = None,
) -> torch.Tensor:
    """Success condition: aligned, upright, and inside the physical depth window.

    Args:
        peg_tip: Peg tip world position (N, 3).
        hole_pos: Hole center world position (N, 3).
        peg_quat: Peg orientation quaternion (N, 4). If None, skip tilt check.
        radial_tol: Maximum allowed radial error (default 1.3 mm).
        depth_required: Minimum insertion depth (default 30 mm).
        max_depth: Maximum insertion depth (default physical hole depth, 40 mm).
        tilt_tol: Maximum tilt angle in radians (default 2°).

    Returns:
        Boolean tensor (N,).
    """
    # Resolve defaults at call time.  Curriculum configurations used to mutate
    # module globals, while Python had already captured the old values in the
    # function defaults.  Keeping the fallback dynamic also preserves backwards
    # compatibility for diagnostic scripts.
    radial_tol = SUCCESS_RADIAL_TOL if radial_tol is None else radial_tol
    depth_required = SUCCESS_DEPTH if depth_required is None else depth_required
    max_depth = SUCCESS_MAX_DEPTH if max_depth is None else max_depth
    tilt_tol = SUCCESS_TILT_TOL_RAD if tilt_tol is None else tilt_tol

    radial_err = get_radial_error(peg_tip, hole_pos)
    depth = get_insertion_depth(peg_tip, hole_pos)

    in_xy = radial_err <= radial_tol
    deep_enough = depth >= depth_required
    not_through_bottom = depth <= max_depth

    if peg_quat is not None:
        tilt = get_tilt_angle(peg_quat)
        upright = tilt <= tilt_tol
        return in_xy & deep_enough & not_through_bottom & upright

    return in_xy & deep_enough & not_through_bottom


def get_xy_distance(
    peg_tip: torch.Tensor,
    hole_pos: torch.Tensor,
) -> torch.Tensor:
    """XY distance from peg tip to hole center (without Z). Shape (N,)."""
    return torch.linalg.vector_norm(peg_tip[:, :2] - hole_pos[:, :2], dim=-1)


def get_3d_distance(
    peg_tip: torch.Tensor,
    hole_pos: torch.Tensor,
) -> torch.Tensor:
    """3D Euclidean distance from peg tip to hole center. Shape (N,)."""
    return torch.linalg.vector_norm(peg_tip - hole_pos, dim=-1)
