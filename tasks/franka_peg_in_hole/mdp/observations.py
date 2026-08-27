"""Observation functions for the Franka Peg-in-Hole task.

Uses unified geometry helpers from mdp.geometry.
All spatial observations are relative to FR3 base frame for env-independent learning.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

from . import geometry as geo

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _env_origin(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get FR3 base (env origin) position for converting to base frame.

    Robot base is at (0,0,0) in env-local but may have env_spacing offset.
    Using the robot base link gives the correct reference for each env.
    """
    robot: Articulation = env.scene["robot"]
    # panda_link0 is the base — its world position is the env origin
    return robot.data.body_state_w[:, 0, :3]


def joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Current joint positions (arm 7 + finger 2 = 9)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids]


def joint_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Current joint velocities."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids]


def canonical_joint_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Default posture used by the offset5 policy's proprioceptive interface.

    Target-conditioned IK deliberately changes the physical arm posture to
    reach a displaced hole. This observation keeps the input distribution
    compatible with the offset5 checkpoint while the simulator still uses
    the actual joint state for dynamics and OSC control.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.default_joint_pos[:, asset_cfg.joint_ids]


def canonical_joint_vel(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Zero velocity counterpart to :func:`canonical_joint_pos`."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.zeros_like(asset.data.joint_vel[:, asset_cfg.joint_ids])


def ee_position(
    env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    """End-effector position in world frame."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_pos_w[..., 0, :]


def ee_orientation(
    env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    """End-effector orientation as quaternion in world frame."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_quat_w[..., 0, :]


def peg_to_hole_vector(
    env: ManagerBasedRLEnv,
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole_board"),
) -> torch.Tensor:
    """Vector from peg tip to hole center, in ROBOT BASE FRAME. Shape (N, 3)."""
    peg_tip, _ = geo.get_peg_tip(env)
    hole_pos, _ = geo.get_hole_center(env, hole_cfg)
    origin = _env_origin(env)

    # Convert to base-relative for env-independent representation
    peg_tip_rel = peg_tip - origin
    hole_pos_rel = hole_pos - origin
    return hole_pos_rel - peg_tip_rel  # (N, 3)


def hole_position(
    env: ManagerBasedRLEnv,
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole_board"),
) -> torch.Tensor:
    """Hole center position in ROBOT BASE FRAME. Shape (N, 3)."""
    hole_pos, _ = geo.get_hole_center(env, hole_cfg)
    origin = _env_origin(env)
    return hole_pos - origin


def peg_tilt_vector(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Peg axis unit vector in world frame. Shape (N, 3). Captures tilt."""
    return geo.get_peg_axis(env)


def last_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Last action taken by the agent."""
    return env.action_manager.action


def hole_id_onehot(env: ManagerBasedRLEnv, num_holes: int = 1) -> torch.Tensor:
    """One-hot encoding of target hole ID (for phase 2)."""
    target_ids = getattr(
        env,
        "_target_hole_id",
        torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
    ).clamp(0, num_holes - 1)
    one_hot = torch.zeros(env.num_envs, num_holes, device=env.device)
    one_hot.scatter_(1, target_ids.unsqueeze(-1), 1.0)
    return one_hot


def generated_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Generated commands from the command manager."""
    return env.command_manager.get_command(command_name)
