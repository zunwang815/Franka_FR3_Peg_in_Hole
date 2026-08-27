"""Termination conditions for the Franka Peg-in-Hole task.

Uses unified geometry helpers from mdp.geometry.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from . import geometry as geo

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate episodes that have reached the maximum time steps."""
    return env.episode_length_buf >= env.max_episode_length - 1


def success_insertion(
    env: ManagerBasedRLEnv,
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole_board"),
    radial_tol: float = geo.SUCCESS_RADIAL_TOL,
    depth_required: float = geo.SUCCESS_DEPTH,
    max_depth: float = geo.SUCCESS_MAX_DEPTH,
    tilt_tol: float = geo.SUCCESS_TILT_TOL_RAD,
) -> torch.Tensor:
    """Terminate when the peg is aligned inside the accepted depth window."""
    peg_tip, peg_quat = geo.get_peg_tip(env)
    hole_pos, _ = geo.get_hole_center(env, hole_cfg)
    success = geo.is_in_hole(
        peg_tip, hole_pos, peg_quat,
        radial_tol=radial_tol,
        depth_required=depth_required,
        max_depth=max_depth,
        tilt_tol=tilt_tol,
    )
    # Preserve the exact pre-auto-reset reason and terminal geometry. Consumers
    # must not equate generic ``terminated`` with success now that physical
    # failure terminations also exist.
    env._success_termination_mask = success.detach().clone()
    env._termination_depth = geo.get_insertion_depth(peg_tip, hole_pos).detach().clone()
    env._termination_radial_error = geo.get_radial_error(peg_tip, hole_pos).detach().clone()
    env._termination_tilt = geo.get_tilt_angle(peg_quat).detach().clone()
    if getattr(env, "_capture_terminal_dynamics", False):
        terminal_mask = success | (env.episode_length_buf >= env.max_episode_length - 1)
        if terminal_mask.any():
            robot: Articulation = env.scene["robot"]
            arm_ids, _ = robot.find_joints("panda_joint.*", preserve_order=True)
            hand_idx = robot.find_bodies("panda_hand")[0][0]
            jac_body_idx = hand_idx - 1 if robot.is_fixed_base else hand_idx
            jacobian = robot.root_physx_view.get_jacobians()[
                :, jac_body_idx, :, arm_ids
            ]
            singular = torch.linalg.svdvals(jacobian[terminal_mask])
            sigma_min = singular[:, -1]
            condition = singular[:, 0] / sigma_min.clamp_min(1.0e-8)
            joint_pos = robot.data.joint_pos[:, arm_ids]
            limits = robot.data.soft_joint_pos_limits[:, arm_ids]
            joint_range = (limits[:, :, 1] - limits[:, :, 0]).clamp_min(1.0e-8)
            normalized_margin = torch.minimum(
                joint_pos - limits[:, :, 0], limits[:, :, 1] - joint_pos
            ) / joint_range
            min_joint_margin = normalized_margin.amin(dim=-1)
            if not hasattr(env, "_termination_sigma_min"):
                nan = torch.full((env.num_envs,), float("nan"), device=env.device)
                env._termination_sigma_min = nan.clone()
                env._termination_jacobian_condition = nan.clone()
                env._termination_ee_z_velocity = nan.clone()
                env._termination_joint_limit_margin = nan.clone()
                env._termination_joint_margins = torch.full(
                    (env.num_envs, len(arm_ids)), float("nan"), device=env.device
                )
                env._termination_joint_positions = torch.full(
                    (env.num_envs, len(arm_ids)), float("nan"), device=env.device
                )
            env._termination_sigma_min[terminal_mask] = sigma_min.detach()
            env._termination_jacobian_condition[terminal_mask] = condition.detach()
            env._termination_ee_z_velocity[terminal_mask] = (
                robot.data.body_vel_w[terminal_mask, hand_idx, 2].detach()
            )
            env._termination_joint_limit_margin[terminal_mask] = (
                min_joint_margin[terminal_mask].detach()
            )
            env._termination_joint_margins[terminal_mask] = (
                normalized_margin[terminal_mask].detach()
            )
            env._termination_joint_positions[terminal_mask] = (
                joint_pos[terminal_mask].detach()
            )
    # Opt-in instrumentation used by the process-isolated visualizer. This is
    # evaluated before ManagerBasedRLEnv auto-resets successful environments,
    # so it preserves the exact accepted state without changing termination.
    if getattr(env, "_capture_visualization_success", False) and success.any():
        env._visualize_success_joint_pos = env.scene["robot"].data.joint_pos.detach().clone()
        env._visualize_success_metrics = (
            geo.get_radial_error(peg_tip, hole_pos).detach().clone(),
            geo.get_insertion_depth(peg_tip, hole_pos).detach().clone(),
            geo.get_tilt_angle(peg_quat).detach().clone(),
        )
    return success


def over_insertion(
    env: ManagerBasedRLEnv,
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole_board"),
    max_depth: float = geo.SUCCESS_MAX_DEPTH + geo.OVER_INSERTION_TOL,
) -> torch.Tensor:
    """Fail when the peg tip passes below the physical sleeve bottom."""
    peg_tip, _ = geo.get_peg_tip(env)
    hole_pos, _ = geo.get_hole_center(env, hole_cfg)
    failed = geo.get_insertion_depth(peg_tip, hole_pos) > max_depth
    env._over_insertion_termination_mask = failed.detach().clone()
    return failed


def workspace_violation(
    env: ManagerBasedRLEnv,
    bounds: tuple[float, float, float] = (0.25, 0.25, 0.25),
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole_board"),
) -> torch.Tensor:
    """Terminate when peg leaves workspace around hole."""
    peg_tip, _ = geo.get_peg_tip(env)
    hole_pos, _ = geo.get_hole_center(env, hole_cfg)
    rel_pos = peg_tip - hole_pos

    bx, by, bz = bounds
    out = (rel_pos[:, 0].abs() > bx) | (rel_pos[:, 1].abs() > by) | (rel_pos[:, 2].abs() > bz)
    return out


def peg_left_cylinder(
    env: ManagerBasedRLEnv,
    radius: float = 0.15,
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole_board"),
) -> torch.Tensor:
    """Terminate when peg tip leaves the constraint cylinder (XY radius around hole)."""
    peg_tip, _ = geo.get_peg_tip(env)
    hole_pos, _ = geo.get_hole_center(env, hole_cfg)
    dist_xy = geo.get_xy_distance(peg_tip, hole_pos)
    return dist_xy > radius


def joint_limits_violation(
    env: ManagerBasedRLEnv,
    margin: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when any joint is near its limit."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    lower = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    upper = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]

    near_lower = (joint_pos - lower) < margin
    near_upper = (upper - joint_pos) < margin

    return (near_lower | near_upper).any(dim=-1)  # Fixed: parentheses around OR
