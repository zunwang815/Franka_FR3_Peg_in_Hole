"""Event functions for environment reset and randomization.

Key randomizations (per reset):
1. Hole position: Random within 10x10 cm area on the table
2. Robot joint positions: Small random perturbation around default pose (using built-in)
3. Peg grasp uncertainty: simulated through hole position variation relative to peg
4. Peg-to-EE sync: Update peg rigid body pose to follow end-effector (every step)
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def set_joint_position_targets_to_default(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Initialize implicit-actuator PD targets to the reset posture.

    The OSC action writes effort targets only.  When weak implicit joint
    stiffness is also used for posture stabilization, an uninitialized position
    target defaults to zero and violently pulls the arm away from its reset
    pose.  Persist the articulation's configured default targets instead.
    """
    from isaaclab.assets import Articulation

    robot: Articulation = env.scene[robot_cfg.name]
    target = robot.data.default_joint_pos[env_ids].clone()
    robot.set_joint_position_target(target, env_ids=env_ids)


def randomize_fixed_peg_mount(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    x_range: tuple[float, float] = (-0.002, 0.002),
    y_range: tuple[float, float] = (-0.002, 0.002),
):
    """Randomize physical X/Y mount joints independently per environment.

    This changes the peg relative to the flange/hand; it does not move the hole
    and therefore represents grasp/tool mounting uncertainty as a distinct
    random variable.
    """
    rand_x = math_utils.sample_uniform(x_range[0], x_range[1], (len(env_ids),), device=env.device)
    rand_y = math_utils.sample_uniform(y_range[0], y_range[1], (len(env_ids),), device=env.device)
    offsets = torch.stack((rand_x, rand_y), dim=-1)

    from isaaclab.assets import Articulation
    robot: Articulation = env.scene["robot"]
    mount_ids, mount_names = robot.find_joints("peg_mount_joint_.*")
    if len(mount_ids) != 2:
        raise RuntimeError(f"Expected two peg mount joints, found {mount_names}")
    # Resolve semantic X/Y order even if PhysX changes articulation ordering.
    name_to_id = dict(zip(mount_names, mount_ids))
    ordered_ids = [name_to_id["peg_mount_joint_x"], name_to_id["peg_mount_joint_y"]]
    joint_pos = robot.data.joint_pos[env_ids][:, ordered_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    joint_pos[:, 0] = rand_x
    joint_pos[:, 1] = rand_y
    robot.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=ordered_ids, env_ids=env_ids)
    robot.set_joint_position_target(joint_pos, joint_ids=ordered_ids, env_ids=env_ids)

    # Retain sampled ground truth for diagnostics only. It is not included in
    # policy observations.
    if not hasattr(env, "_peg_mount_offset_xy"):
        env._peg_mount_offset_xy = torch.zeros((env.num_envs, 2), device=env.device)
    env._peg_mount_offset_xy[env_ids] = offsets


def hold_fixed_peg_mount(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Hold the sampled peg-mount offset after each control step.

    The two prismatic joints represent a static tool-mount uncertainty, not a
    compliant contact mechanism.  With a light mount body, contact impulses
    can otherwise drive the joints to their +/-5 mm limits even though their
    PD targets remain at the sampled reset offset.  Rewriting the joint state
    at the control boundary keeps the sampled offset fixed while preserving
    its randomization across episodes.  This is an opt-in course event; the
    historical tasks leave it disabled for reproducibility.
    """
    from isaaclab.assets import Articulation

    robot: Articulation = env.scene[robot_cfg.name]
    mount_ids, mount_names = robot.find_joints("peg_mount_joint_.*")
    if len(mount_ids) != 2:
        raise RuntimeError(f"Expected two peg mount joints, found {mount_names}")
    name_to_id = dict(zip(mount_names, mount_ids))
    ordered_ids = [name_to_id["peg_mount_joint_x"], name_to_id["peg_mount_joint_y"]]

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if not hasattr(env, "_peg_mount_offset_xy"):
        env._peg_mount_offset_xy = torch.zeros((env.num_envs, 2), device=env.device)
    offsets = env._peg_mount_offset_xy[env_ids].clone()
    robot.write_joint_state_to_sim(
        offsets,
        torch.zeros_like(offsets),
        joint_ids=ordered_ids,
        env_ids=env_ids,
    )
    robot.set_joint_position_target(offsets, joint_ids=ordered_ids, env_ids=env_ids)


def target_conditioned_arm_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole_board"),
    iterations: int = 5,
):
    """Use a short damped 6-D Jacobian solve to initialize the arm over the hole.

    This runs only during reset.  XY translation is the dominant task and a
    lightly weighted orientation term preserves the near-vertical peg pose.
    It removes the large long-range XY burden from PPO without replacing any
    actions during the episode.
    """
    from isaaclab.assets import Articulation
    from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul
    from . import geometry

    robot: Articulation = env.scene[robot_cfg.name]
    hole = env.scene[hole_cfg.name]
    arm_ids, _ = robot.find_joints("panda_joint.*", preserve_order=True)
    hand_idx = robot.find_bodies("panda_hand")[0][0]
    jac_body_idx = hand_idx - 1 if robot.is_fixed_base else hand_idx
    limits = robot.data.soft_joint_pos_limits[env_ids][:, arm_ids]

    if len(env_ids) == 0 or iterations <= 0:
        return

    # Isaac/PhysX is reliable for a full initial reset, but repeatedly calling
    # sim.forward() from a partial reset while other environments are active
    # can terminate the Kit process.  Cache the solved pose/Jacobian from the
    # first full reset and use a single linear XY update for later resets.
    cache = getattr(env, "_target_init_cache", None)
    if cache is not None:
        target_xy = hole.data.root_pos_w[env_ids, :2].clone()
        q_opt = cache["q_arm"][env_ids].clone()
        base_hole = cache["hole_xy"][env_ids]
        jac_xy = cache["jac_xy"][env_ids]
        err_xy = target_xy - base_hole
        gram = jac_xy @ jac_xy.transpose(1, 2)
        eye_xy = torch.eye(2, device=env.device, dtype=q_opt.dtype).expand(len(env_ids), -1, -1)
        dq = jac_xy.transpose(1, 2) @ torch.linalg.solve(
            gram + 1.0e-4 * eye_xy, err_xy.unsqueeze(-1)
        )
        dq = torch.clamp(dq.squeeze(-1), -0.08, 0.08)
        q_opt = torch.maximum(torch.minimum(q_opt + 0.7 * dq, limits[..., 1]), limits[..., 0])
        q_state = robot.data.joint_pos[env_ids].clone()
        q_state[:, arm_ids] = q_opt
        robot.write_joint_state_to_sim(q_state, torch.zeros_like(q_state), env_ids=env_ids)
        robot.set_joint_position_target(q_state, env_ids=env_ids)
    cache["q_arm"][env_ids] = q_opt
    cache["hole_xy"][env_ids] = target_xy
    return

    # The first reset should include all environments.  If a caller invokes a
    # partial reset before initialization, leave the default pose untouched
    # rather than attempting a forward pass in a partially active simulator.
    if len(env_ids) != env.num_envs:
        return

    q_all = robot.data.joint_pos[env_ids].clone()
    q_opt = q_all[:, arm_ids].clone()

    def write_and_forward(q_arm: torch.Tensor):
        q_state = q_all.clone()
        q_state[:, arm_ids] = q_arm
        robot.write_joint_state_to_sim(q_state, torch.zeros_like(q_state), env_ids=env_ids)
        robot.set_joint_position_target(q_state, env_ids=env_ids)
        env.sim.forward()
        env.scene.update(env.cfg.sim.dt)

    write_and_forward(q_opt)
    desired_quat = geometry.get_peg_tip(env, robot_cfg)[1][env_ids].detach().clone()
    target_xy = hole.data.root_pos_w[env_ids, :2].clone()
    rot_weight = 0.05
    eye = torch.eye(6, device=env.device, dtype=q_opt.dtype).expand(len(env_ids), -1, -1)

    for _ in range(iterations):
        tip, quat = geometry.get_peg_tip(env, robot_cfg)
        tip = tip[env_ids]
        quat = quat[env_ids]
        # Slice the environment and joint dimensions separately.  Combining a
        # tensor of env indices with a Python joint-index list in one advanced
        # index can produce an invalid broadcast shape in PhysX tensor views.
        jac_all = robot.root_physx_view.get_jacobians()
        jac = jac_all[env_ids][:, jac_body_idx, :, :][:, :, arm_ids].clone()
        err_xy = target_xy - tip[:, :2]
        rot_err = axis_angle_from_quat(quat_mul(desired_quat, quat_conjugate(quat)))
        task_err = torch.cat(
            (err_xy, torch.zeros((len(env_ids), 1), device=env.device, dtype=q_opt.dtype),
             rot_weight * rot_err), dim=-1
        )
        jac[:, 3:, :] *= rot_weight
        gram = jac @ jac.transpose(1, 2)
        dq = jac.transpose(1, 2) @ torch.linalg.solve(
            gram + 1.0e-4 * eye, task_err.unsqueeze(-1)
        )
        dq = torch.clamp(dq.squeeze(-1), -0.08, 0.08)
        q_opt = torch.maximum(torch.minimum(q_opt + 0.7 * dq, limits[..., 1]), limits[..., 0])
        write_and_forward(q_opt)

    # Leave the solved pose in the simulator and initialize actuator targets.
    write_and_forward(q_opt)
    jac_xy = jac[:, :2, :].detach().clone()
    cache_q = torch.zeros((env.num_envs, len(arm_ids)), device=env.device, dtype=q_opt.dtype)
    cache_hole = torch.zeros((env.num_envs, 2), device=env.device, dtype=q_opt.dtype)
    cache_jac = torch.zeros(
        (env.num_envs, 2, len(arm_ids)), device=env.device, dtype=q_opt.dtype
    )
    cache_q[env_ids] = q_opt.detach()
    cache_hole[env_ids] = target_xy.detach()
    cache_jac[env_ids] = jac_xy
    env._target_init_cache = {"q_arm": cache_q, "hole_xy": cache_hole, "jac_xy": cache_jac}


def target_conditioned_arm_pose_online(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    iterations: int = 15,
    target_tip_z: float = 0.393592,
    preserve_mount_residual: bool = False,
    target_blend: float = 1.0,
):
    """Reset-time target-conditioned IK with fixed tip height.

    The first full reset solves a damped 6-D position/orientation task. Later
    partial resets use a cached position Jacobian for a single linear update,
    avoiding repeated ``sim.forward`` calls during asynchronous resets.
    """
    from isaaclab.assets import Articulation
    from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul
    from . import geometry

    robot: Articulation = env.scene[robot_cfg.name]
    arm_ids, _ = robot.find_joints("panda_joint.*", preserve_order=True)
    hand_idx = robot.find_bodies("panda_hand")[0][0]
    jac_body_idx = hand_idx - 1 if robot.is_fixed_base else hand_idx
    if len(env_ids) == 0 or iterations <= 0:
        return

    hole_pos, _ = geometry.get_hole_center(env)
    # A full target solve minimizes the residual XY error but changes the
    # arm's proprioceptive state substantially at the workspace edge.  Allow
    # a partial solve for transfer diagnostics: the IK removes only a chosen
    # fraction of the initial XY error, leaving the remainder for the policy.
    # This keeps the reset state closer to the offset5 training distribution.
    blend = float(max(0.0, min(1.0, target_blend)))
    tip_seed, _ = geometry.get_peg_tip(env, robot_cfg)
    target_pos = tip_seed[env_ids].clone()
    desired_xy = hole_pos[env_ids, :2].clone()
    if preserve_mount_residual and hasattr(env, "_peg_mount_offset_xy"):
        # Match the offset5 policy's initial geometry: the peg tip retains the
        # sampled physical mount offset relative to the hole after arm IK.
        desired_xy += env._peg_mount_offset_xy[env_ids]
    target_pos[:, :2] += blend * (desired_xy - target_pos[:, :2])
    target_pos[:, 2] = target_tip_z
    limits = robot.data.soft_joint_pos_limits[env_ids][:, arm_ids]
    cache = getattr(env, "_online_target_init_cache", None)

    if cache is not None:
        # Reset events have already restored the default posture.  Use the
        # cached central Jacobian for a single target-dependent correction.
        tip_now, _ = geometry.get_peg_tip(env, robot_cfg)
        q_opt = robot.data.joint_pos[env_ids][:, arm_ids].clone()
        jac_pos = cache["jac_pos"][env_ids]
        err_pos = target_pos - tip_now[env_ids]
        gram = jac_pos @ jac_pos.transpose(1, 2)
        eye = torch.eye(3, device=env.device, dtype=q_opt.dtype).expand(len(env_ids), -1, -1)
        dq = jac_pos.transpose(1, 2) @ torch.linalg.solve(
            gram + 1.0e-4 * eye, err_pos.unsqueeze(-1)
        )
        dq = torch.clamp(dq.squeeze(-1), -0.08, 0.08)
        q_opt = torch.maximum(torch.minimum(q_opt + 0.7 * dq, limits[..., 1]), limits[..., 0])
        q_state = robot.data.joint_pos[env_ids].clone()
        q_state[:, arm_ids] = q_opt
        robot.write_joint_state_to_sim(q_state, torch.zeros_like(q_state), env_ids=env_ids)
        robot.set_joint_position_target(q_state, env_ids=env_ids)
        return

    # The first reset must solve all environments together so that one safe
    # simulator forward pass initializes the reusable Jacobian cache.
    if len(env_ids) != env.num_envs:
        return

    q_all = robot.data.joint_pos.clone()
    q_opt = q_all[env_ids][:, arm_ids].clone()

    def write_and_forward(q_arm: torch.Tensor):
        q_state = q_all.clone()
        q_selected = q_state[env_ids].clone()
        q_selected[:, arm_ids] = q_arm
        q_state[env_ids] = q_selected
        robot.write_joint_state_to_sim(q_state, torch.zeros_like(q_state))
        robot.set_joint_position_target(q_state)
        env.sim.forward()
        env.scene.update(env.cfg.sim.dt)

    write_and_forward(q_opt)
    desired_quat = geometry.get_peg_tip(env, robot_cfg)[1][env_ids].detach().clone()
    eye6 = torch.eye(6, device=env.device, dtype=q_opt.dtype).expand(len(env_ids), -1, -1)
    for _ in range(iterations):
        tip, quat = geometry.get_peg_tip(env, robot_cfg)
        tip = tip[env_ids]
        quat = quat[env_ids]
        jac_all = robot.root_physx_view.get_jacobians()
        jac = jac_all[env_ids][:, jac_body_idx, :, :][:, :, arm_ids].clone()
        pos_err = target_pos - tip
        rot_err = axis_angle_from_quat(
            quat_mul(desired_quat, quat_conjugate(quat))
        )
        rot_weight = 0.05
        task_err = torch.cat((pos_err, rot_weight * rot_err), dim=-1)
        jac[:, 3:, :] *= rot_weight
        dq = jac.transpose(1, 2) @ torch.linalg.solve(
            jac @ jac.transpose(1, 2) + 1.0e-4 * eye6,
            task_err.unsqueeze(-1),
        )
        dq = torch.clamp(dq.squeeze(-1), -0.08, 0.08)
        q_opt = torch.maximum(torch.minimum(q_opt + 0.7 * dq, limits[..., 1]), limits[..., 0])
        write_and_forward(q_opt)

    # Cache the final position Jacobian.  It remains well-conditioned across
    # the +/-20 mm workspace and is used only for later partial resets.
    jac_all = robot.root_physx_view.get_jacobians()
    jac_final = jac_all[env_ids][:, jac_body_idx, :3, :][:, :, arm_ids].detach().clone()
    cache_jac = torch.zeros(
        (env.num_envs, 3, len(arm_ids)), device=env.device, dtype=q_opt.dtype
    )
    cache_q = torch.zeros(
        (env.num_envs, len(arm_ids)), device=env.device, dtype=q_opt.dtype
    )
    cache_jac[env_ids] = jac_final
    cache_q[env_ids] = q_opt.detach()
    env._online_target_init_cache = {"jac_pos": cache_jac, "q_arm": cache_q}


def set_target_conditioned_bank_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole_board"),
):
    """Reset-time lookup of offline IK poses without any simulator forward pass."""
    from isaaclab.assets import Articulation
    from .target_pose_bank import TARGET_POSE_BANK

    robot: Articulation = env.scene[robot_cfg.name]
    hole = env.scene[hole_cfg.name]
    arm_ids, _ = robot.find_joints("panda_joint.*", preserve_order=True)
    if len(env_ids) == 0:
        return

    keys = tuple(TARGET_POSE_BANK.keys())
    if not hasattr(env, "_target_pose_bank_tensors"):
        env._target_pose_bank_tensors = (
            torch.tensor(keys, device=env.device, dtype=torch.float32),
            torch.tensor(tuple(TARGET_POSE_BANK[key] for key in keys),
                         device=env.device, dtype=torch.float32),
        )
    bank_xy, bank_q = env._target_pose_bank_tensors
    raw_xy = hole.data.root_pos_w[env_ids, :2]
    origins = env.scene.env_origins[env_ids, :2]
    hole_local = raw_xy - origins if raw_xy.abs().max() > 2.0 else raw_xy
    nominal_xy = torch.tensor((0.249418, 0.001268), device=env.device, dtype=raw_xy.dtype)
    offset_mm = (hole_local - nominal_xy) * 1000.0
    nearest = torch.linalg.vector_norm(offset_mm[:, None, :] - bank_xy[None, :, :], dim=-1).argmin(dim=1)
    q_target = bank_q[nearest].to(dtype=robot.data.joint_pos.dtype)
    q_state = robot.data.joint_pos[env_ids].clone()
    q_state[:, arm_ids] = q_target
    robot.write_joint_state_to_sim(q_state, torch.zeros_like(q_state), env_ids=env_ids)
    robot.set_joint_position_target(q_state, env_ids=env_ids)


def reset_hole_position_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    x_range: tuple[float, float] = (-0.05, 0.05),
    y_range: tuple[float, float] = (-0.05, 0.05),
    table_z: float = 0.025,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("hole_board"),
    outer_probability: float = 0.0,
    outer_min_abs: float = 0.0,
    inner_probability: float = 0.0,
    inner_abs: float = 0.0,
    radial_min: float = 0.0,
    radial_max: float = 0.0,
    array_hole_spacing: float = 0.0,
    array_columns: int = 1,
    array_rows: int = 1,
):
    """Reset hole board AND its walls to random XY position on table.

    This function moves the hole_board marker AND the 4 physical walls
    that form the insertion hole, maintaining their relative positions.

    Args:
        env: The environment.
        env_ids: Environment IDs to reset.
        x_range: (min, max) for hole x position offset in meters (default: ±5cm).
        y_range: (min, max) for hole y position offset in meters (default: ±5cm).
        table_z: Height of the table surface in meters.
        asset_cfg: Scene entity config for the hole board marker.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    if radial_min < 0.0 or radial_max < 0.0:
        raise ValueError("radial annulus bounds must be non-negative")
    if radial_max > 0.0 and radial_max < radial_min:
        raise ValueError("radial_max must be >= radial_min")

    # Get default root state and compute random XY offset
    root_states = asset.data.default_root_state[env_ids].clone()
    rand_x = math_utils.sample_uniform(x_range[0], x_range[1], (len(env_ids),), device=env.device)
    rand_y = math_utils.sample_uniform(y_range[0], y_range[1], (len(env_ids),), device=env.device)
    if radial_min > 0.0 or radial_max > 0.0:
        # Rejection-sample an annulus inside the configured rectangular
        # workspace.  This keeps the original square bounds while providing
        # a controllable edge-focused curriculum.
        r_max = radial_max if radial_max > 0.0 else float("inf")
        for _ in range(12):
            radius = torch.sqrt(rand_x.square() + rand_y.square())
            valid = (radius >= radial_min) & (radius <= r_max)
            if bool(valid.all()):
                break
            count = int((~valid).sum().item())
            rand_x[~valid] = math_utils.sample_uniform(
                x_range[0], x_range[1], (count,), device=env.device
            )
            rand_y[~valid] = math_utils.sample_uniform(
                y_range[0], y_range[1], (count,), device=env.device
            )
    if inner_probability > 0.0:
        if not 0.0 <= inner_probability <= 1.0:
            raise ValueError("inner_probability must be in [0, 1]")
        if inner_abs <= 0.0 or inner_abs > min(abs(x_range[0]), abs(x_range[1]), abs(y_range[0]), abs(y_range[1])):
            raise ValueError("inner_abs must lie inside the outer workspace")
        inner_mask = torch.rand(len(env_ids), device=env.device) < inner_probability
        inner_count = int(inner_mask.sum().item())
        if inner_count > 0:
            rand_x[inner_mask] = math_utils.sample_uniform(-inner_abs, inner_abs, (inner_count,), device=env.device)
            rand_y[inner_mask] = math_utils.sample_uniform(-inner_abs, inner_abs, (inner_count,), device=env.device)
    if outer_probability > 0.0:
        outer_mask = torch.rand(len(env_ids), device=env.device) < outer_probability
        outer_count = int(outer_mask.sum().item())
        if outer_count > 0:
            axis_x = torch.rand(outer_count, device=env.device) < 0.5
            signs = torch.where(
                torch.rand(outer_count, device=env.device) < 0.5,
                -torch.ones(outer_count, device=env.device),
                torch.ones(outer_count, device=env.device),
            )
            outer_max = min(abs(x_range[0]), abs(x_range[1]), abs(y_range[0]), abs(y_range[1]))
            magnitudes = math_utils.sample_uniform(
                outer_min_abs, outer_max, (outer_count,), device=env.device
            )
            outer_values = signs * magnitudes
            rand_x[outer_mask] = torch.where(axis_x, outer_values, rand_x[outer_mask])
            rand_y[outer_mask] = torch.where(axis_x, rand_y[outer_mask], outer_values)
    # Keep the sampled array-origin offset separate from the active target
    # hole offset.  Every physical sleeve follows the origin; only the marker
    # used by the task/reward follows the selected target center.
    array_origin_x = rand_x.clone()
    array_origin_y = rand_y.clone()
    # The array task samples the array origin in the requested workspace and
    # offsets the active physical target hole by its 3cm grid coordinate.
    if array_hole_spacing > 0.0:
        if array_columns <= 0 or array_rows <= 0:
            raise ValueError("array_columns/array_rows must be positive")
        target_ids = getattr(
            env,
            "_target_hole_id",
            torch.zeros(len(env_ids), dtype=torch.long, device=env.device),
        )[env_ids]
        target_ids = target_ids.clamp(0, array_columns * array_rows - 1)
        col = target_ids.remainder(array_columns).to(rand_x.dtype)
        row = torch.div(target_ids, array_columns, rounding_mode="floor").to(rand_y.dtype)
        rand_x = rand_x + (col - 0.5 * (array_columns - 1.0)) * array_hole_spacing
        rand_y = rand_y + (row - 0.5 * (array_rows - 1.0)) * array_hole_spacing

    if not hasattr(env, "_hole_offset_xy"):
        env._hole_offset_xy = torch.zeros((env.num_envs, 2), device=env.device)
    env._hole_offset_xy[env_ids] = torch.stack((rand_x, rand_y), dim=-1)
    if array_hole_spacing > 0.0:
        if not hasattr(env, "_array_origin_offset_xy"):
            env._array_origin_offset_xy = torch.zeros((env.num_envs, 2), device=env.device)
        env._array_origin_offset_xy[env_ids] = torch.stack((array_origin_x, array_origin_y), dim=-1)

    # Apply offset to default position
    root_states[:, 0] += rand_x
    root_states[:, 1] += rand_y
    root_states[:, 2] = table_z

    # Fixed orientation (hole pointing up)
    root_states[:, 3] = 1.0  # w
    root_states[:, 4] = 0.0  # x
    root_states[:, 5] = 0.0  # y
    root_states[:, 6] = 0.0  # z

    # Write hole_board to simulation
    asset.write_root_pose_to_sim(root_states[:, :7], env_ids=env_ids)
    zeros_vel = torch.zeros(len(env_ids), 6, device=env.device)
    asset.write_root_velocity_to_sim(zeros_vel, env_ids=env_ids)

    # === CRITICAL FIX: Also move the 4 walls that form the physical hole ===
    # Phase 1 uses 4 separate wall cuboids. They must move with the hole_board
    # marker so that the physical hole geometry matches the reward target.
    # Curriculum fixtures may use a denser 36-segment ring.  Missing entities
    # are intentionally ignored so the same reset function serves every stage.
    if array_hole_spacing > 0.0:
        wall_names = [
            f"array_hole_{hole_index}_wall_{wall_index}"
            for hole_index in range(array_columns * array_rows)
            for wall_index in range(36)
        ]
        wall_offset_x, wall_offset_y = array_origin_x, array_origin_y
    else:
        wall_names = [f"hole_wall_{i}" for i in range(36)]
        wall_offset_x, wall_offset_y = rand_x, rand_y
    for wall_name in wall_names:
        try:
            wall: RigidObject = env.scene[wall_name]
            wall_states = wall.data.default_root_state[env_ids].clone()
            wall_states[:, 0] += wall_offset_x
            wall_states[:, 1] += wall_offset_y
            # Preserve the configured segment orientation and Z height.
            wall.write_root_pose_to_sim(wall_states[:, :7], env_ids=env_ids)
            wall.write_root_velocity_to_sim(zeros_vel, env_ids=env_ids)
        except (KeyError, AttributeError):
            pass  # Wall not in this scene config.

    # Move the four top-plate slabs with the marker and sleeve. Leaving these
    # at their default location would make randomized reward geometry diverge
    # from the physical opening.
    fixture_names = ("fixture_left", "fixture_right", "fixture_front", "fixture_back")
    for fixture_name in fixture_names:
        try:
            fixture: RigidObject = env.scene[fixture_name]
            fixture_states = fixture.data.default_root_state[env_ids].clone()
            fixture_states[:, 0] += rand_x
            fixture_states[:, 1] += rand_y
            fixture.write_root_pose_to_sim(fixture_states[:, :7], env_ids=env_ids)
            fixture.write_root_velocity_to_sim(zeros_vel, env_ids=env_ids)
        except (KeyError, AttributeError):
            pass


def apply_hole_offsets(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    offsets_xy: torch.Tensor,
    table_z: float | None = None,
):
    """Apply deterministic hole XY offsets for reachability-grid audits."""
    if offsets_xy.shape != (len(env_ids), 2):
        raise ValueError("offsets_xy must have shape (len(env_ids), 2)")
    asset: RigidObject = env.scene["hole_board"]
    if not hasattr(env, "_hole_offset_xy"):
        env._hole_offset_xy = torch.zeros((env.num_envs, 2), device=env.device)
    # `default_root_state` is stored in environment-local coordinates and may
    # already contain the reset-time random offset. Apply only the requested
    # delta so a grid audit is deterministic rather than random+grid.
    previous_offsets = env._hole_offset_xy[env_ids].clone()
    delta_xy = offsets_xy - previous_offsets
    root_states = asset.data.default_root_state[env_ids].clone()
    root_states[:, :2] -= previous_offsets
    root_states[:, :2] += delta_xy
    if table_z is not None:
        root_states[:, 2] = table_z
    root_states[:, 3:7] = torch.tensor(
        (1.0, 0.0, 0.0, 0.0), device=env.device, dtype=root_states.dtype
    )
    asset.write_root_pose_to_sim(root_states[:, :7], env_ids=env_ids)
    zeros_vel = torch.zeros(len(env_ids), 6, device=env.device)
    asset.write_root_velocity_to_sim(zeros_vel, env_ids=env_ids)
    env._hole_offset_xy[env_ids] = offsets_xy

    for index in range(36):
        try:
            wall: RigidObject = env.scene[f"hole_wall_{index}"]
            states = wall.data.default_root_state[env_ids].clone()
            states[:, :2] -= previous_offsets
            states[:, :2] += delta_xy
            states[:, 3:7] = torch.tensor(
                (1.0, 0.0, 0.0, 0.0), device=env.device, dtype=states.dtype
            )
            wall.write_root_pose_to_sim(states[:, :7], env_ids=env_ids)
            wall.write_root_velocity_to_sim(zeros_vel, env_ids=env_ids)
        except (KeyError, AttributeError):
            pass
    for name in ("fixture_left", "fixture_right", "fixture_front", "fixture_back"):
        try:
            fixture: RigidObject = env.scene[name]
            states = fixture.data.default_root_state[env_ids].clone()
            states[:, :2] -= previous_offsets
            states[:, :2] += delta_xy
            fixture.write_root_pose_to_sim(states[:, :7], env_ids=env_ids)
            fixture.write_root_velocity_to_sim(zeros_vel, env_ids=env_ids)
        except (KeyError, AttributeError):
            pass


def select_target_hole(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    num_holes: int = 6,
    target_hole_id: int = -1,
):
    """Select the target hole for an array episode.

    ``target_hole_id=-1`` samples uniformly for training. A value in
    ``[0, num_holes)`` makes the episode deterministic, enabling per-hole
    evaluation without changing the rest of the reset distribution.
    """
    if num_holes <= 0:
        raise ValueError("num_holes must be positive")
    if target_hole_id >= num_holes:
        raise ValueError("target_hole_id must be -1 or less than num_holes")
    if not hasattr(env, "_target_hole_id"):
        env._target_hole_id = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
    fixed_target_hole_id = int(getattr(env, "_fixed_target_hole_id", -1))
    if fixed_target_hole_id >= 0:
        target_hole_id = fixed_target_hole_id
    if target_hole_id < 0:
        selected = torch.randint(
            0, num_holes, (len(env_ids),), device=env.device, dtype=torch.long
        )
    else:
        selected = torch.full(
            (len(env_ids),), target_hole_id, device=env.device, dtype=torch.long
        )
    env._target_hole_id[env_ids] = selected


def sync_peg_to_ee(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    peg_offset: tuple[float, float, float] = (0.0, 0.0, -0.10),
    ee_body_name: str = "panda_hand",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
):
    """Sync the peg RigidObject to the robot's gripper center.

    Called every physics step as an interval event. Strategy (tried in order):
    1. Midpoint of panda_leftfinger + panda_rightfinger (ideal: grasp center)
    2. Single-body fallback chain: ee_body_name → panda_hand → panda_link8 → panda_link7

    The peg offset places the peg center relative to the tracked point.
    Default: (0, 0, -0.10) = peg center 10cm below, tip at 15cm (matching reward).

    Args:
        env: The environment.
        env_ids: Environment IDs to sync (None = all envs).
        peg_offset: (x,y,z) offset from tracked point to peg center, in local frame.
        ee_body_name: Preferred single-body name for fallback tracking.
        robot_cfg: Scene entity config for the robot articulation.
        peg_cfg: Scene entity config for the peg rigid object.
    """
    from isaaclab.assets import Articulation

    robot: Articulation = env.scene[robot_cfg.name]
    peg: RigidObject = env.scene[peg_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    ee_pos = None
    ee_quat = None
    tracked_name = "none"

    # --- Strategy 1: midpoint of left + right fingers ---
    for l_name, r_name in [("panda_leftfinger", "panda_rightfinger"),
                            ("leftfinger", "rightfinger")]:
        l_idx = _find_body_index(robot, l_name)
        r_idx = _find_body_index(robot, r_name)
        if l_idx is not None and r_idx is not None:
            left_pos = robot.data.body_state_w[:, l_idx, :3]
            right_pos = robot.data.body_state_w[:, r_idx, :3]
            left_quat = robot.data.body_state_w[:, l_idx, 3:7]
            ee_pos = (left_pos + right_pos) / 2.0
            ee_quat = left_quat
            tracked_name = f"finger_mid({l_name},{r_name})"
            break
        elif l_idx is not None:
            ee_pos = robot.data.body_state_w[:, l_idx, :3]
            ee_quat = robot.data.body_state_w[:, l_idx, 3:7]
            tracked_name = l_name
            break

    # --- Strategy 2: single-body fallback chain ---
    if ee_pos is None:
        for name in [ee_body_name, "panda_hand", "panda_link8", "panda_link7"]:
            idx = _find_body_index(robot, name)
            if idx is not None:
                ee_pos = robot.data.body_state_w[:, idx, :3]
                ee_quat = robot.data.body_state_w[:, idx, 3:7]
                tracked_name = name
                break

    if ee_pos is None:
        raise RuntimeError(
            f"Cannot find any end-effector body. "
            f"Available bodies: {robot.body_names}"
        )

    # Compute peg center position in world frame
    offset = torch.tensor(peg_offset, device=ee_pos.device, dtype=ee_pos.dtype)
    peg_center_world = ee_pos + quat_apply(ee_quat, offset.expand(ee_pos.shape[0], -1))

    # One-time debug: print body list, tracked name, and positions
    global _sync_peg_debug_done
    if not _sync_peg_debug_done:
        _sync_peg_debug_done = True
        print(f"\n[sync_peg_to_ee] Bodies ({len(robot.body_names)}):")
        for i, bname in enumerate(robot.body_names):
            print(f"  [{i}] {bname}")
        print(f"[sync_peg_to_ee] TRACKING: '{tracked_name}'")
        print(f"[sync_peg_to_ee] Tracked pos (env 0):   {ee_pos[0].cpu().numpy()}")
        print(f"[sync_peg_to_ee] Peg offset:             {peg_offset}")
        print(f"[sync_peg_to_ee] Peg center (env 0):     {peg_center_world[0].cpu().numpy()}")

    # Write peg root state (dynamic + high mass: collision impulse is negligible)
    root_state = peg.data.root_state_w.clone()
    root_state[:, :3] = peg_center_world
    root_state[:, 3:7] = ee_quat
    root_state[:, 7:13] = 0.0

    peg.write_root_state_to_sim(root_state, env_ids=env_ids)


def reset_robot_above_hole(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    height_above: float = 0.12,
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole_board"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Warm-start: use the environment's own IK controller to position hand above hole.

    Uses the SAME IK controller (from the action space) to solve for joint positions,
    ensuring consistency between reset and action execution.
    """
    from isaaclab.assets import Articulation

    robot: Articulation = env.scene[robot_cfg.name]
    hole: RigidObject = env.scene[hole_cfg.name]

    # Access the environment's IK controller (same as the action space uses)
    try:
        ik_ctrl = env.action_manager._terms["arm_action"]._ik_controller
    except (KeyError, AttributeError):
        # Fallback: no IK controller, use default reset
        return

    # Get hole position and set target: PEG TIP above hole surface at given height
    # (the IK controls panda_hand, peg tip is ~17cm below hand along tool Z)
    hole_pos = hole.data.root_pos_w[env_ids, :3]
    # Hand target = peg tip target + 17cm up along hand Z (approx, for vertical hand)
    target_pos = hole_pos + torch.tensor([0.0, 0.0, height_above + 0.17], device=env.device)

    body_idx = robot.find_bodies("panda_hand")[0][0]

    # For position-only IK: set desired position (orientation stays current)
    ik_ctrl.ee_pos_des[:] = robot.data.body_state_w[:, body_idx, :3]
    ik_ctrl.ee_pos_des[env_ids] = target_pos

    # Get Jacobian and current joint positions (arm joints only)
    jacobian = robot.root_physx_view.get_jacobians()[:, body_idx, :, robot_cfg.joint_ids]
    joint_pos = robot.data.joint_pos[:, robot_cfg.joint_ids].clone()

    # Numerical IK iteration with fresh Jacobian each step
    for _ in range(50):
        body_pos = robot.data.body_state_w[:, body_idx, :3]
        body_quat = robot.data.body_state_w[:, body_idx, 3:7]
        new_joint_pos = ik_ctrl.compute(body_pos, body_quat, jacobian, joint_pos)
        # Write back and simulate forward to refresh state
        robot.write_joint_position_to_sim(new_joint_pos[env_ids][:, :7],
            joint_ids=list(range(7)), env_ids=env_ids)
        # Refresh jacobian after write (PhysX updates on next buffer read)
        if _ % 10 == 9:
            jacobian = robot.root_physx_view.get_jacobians()[:, body_idx, :, robot_cfg.joint_ids]
        joint_pos = new_joint_pos

    # Final write
    robot.write_joint_position_to_sim(joint_pos[env_ids][:, :7],
        joint_ids=list(range(7)), env_ids=env_ids)


# Module-level flag for one-time debug
_sync_peg_debug_done = False


def _find_body_index(robot, name_pattern: str) -> int | None:
    """Find a body index in the articulation by name pattern. Returns None if not found."""
    try:
        indices, _ = robot.find_bodies(name_pattern)
    except ValueError:
        return None
    return indices[0] if len(indices) > 0 else None
