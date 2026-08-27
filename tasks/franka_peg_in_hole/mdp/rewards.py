"""Reward functions for the Franka Peg-in-Hole task.

Uses unified geometry helpers from mdp.geometry — single source of truth
for all peg/hole measurements.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from . import geometry as geo

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_debug_done = False


def approach_xy(
    env: ManagerBasedRLEnv,
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole_board"),
) -> torch.Tensor:
    """Dense linear XY-distance reward: 1 at 0 → 0 at 1m. Always provides gradient."""
    peg_tip, _ = geo.get_peg_tip(env)
    hole_pos, _ = geo.get_hole_center(env, hole_cfg)
    dist = geo.get_3d_distance(peg_tip, hole_pos)
    global _debug_done
    if not _debug_done:
        _debug_done = True
        print(f"[approach_xy] env0 peg_tip={peg_tip[0].cpu().numpy()}")
        print(f"[approach_xy] env0 hole={hole_pos[0].cpu().numpy()}")
        print(f"[approach_xy] env0 dist={dist[0].item():.3f}  "
              f"mean_dist={dist.mean().item():.3f}  min_dist={dist.min().item():.3f}")
    return torch.clamp(1.0 - dist / 1.00, min=0.0)


def approach_xy_progress(
    env: ManagerBasedRLEnv,
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole_board"),
) -> torch.Tensor:
    """Progress reward: reward REDUCTION in XY distance from previous step.

    r = prev_dist_xy - current_dist_xy

    Positive when getting closer, negative when moving away.
    Eliminates the "stay near hole to farm reward" local optimum.
    Weight: high enough to encourage approach.
    """
    # Store previous distance in env extras
    peg_tip, _ = geo.get_peg_tip(env)
    hole_pos, _ = geo.get_hole_center(env, hole_cfg)
    dist_xy = geo.get_xy_distance(peg_tip, hole_pos)

    # Get previous distance (stored from last step)
    key = "_prev_dist_xy"
    if key not in env.extras:
        env.extras[key] = dist_xy.clone()
    prev_dist = env.extras[key]
    # A vectorized environment resets individual environments asynchronously.
    # Do not compare a new episode's first distance with the previous episode.
    reset_mask = env.episode_length_buf == 0
    prev_dist = torch.where(reset_mask, dist_xy, prev_dist)
    env.extras[key] = dist_xy.clone()

    return prev_dist - dist_xy


def fine_alignment(
    env: ManagerBasedRLEnv,
    sigma: float = 0.003,
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole_board"),
) -> torch.Tensor:
    """Improvement in the fine-alignment potential.

    The potential exp(-0.5*(radial/sigma)^2) is concentrated at 0--3 mm,
    but only its step-to-step increase is rewarded.  Returning the absolute
    potential allowed a centered policy to farm alignment reward until timeout
    instead of completing insertion.
    """
    peg_tip, _ = geo.get_peg_tip(env)
    hole_pos, _ = geo.get_hole_center(env, hole_cfg)
    radial_err = geo.get_radial_error(peg_tip, hole_pos)
    potential = torch.exp(-0.5 * (radial_err / sigma) ** 2)

    key = "_prev_fine_alignment"
    if key not in env.extras:
        env.extras[key] = potential.clone()
    prev_potential = env.extras[key]
    reset_mask = env.episode_length_buf == 0
    prev_potential = torch.where(reset_mask, potential, prev_potential)
    env.extras[key] = potential.clone()
    return potential - prev_potential


def multi_scale_alignment_progress(
    env: ManagerBasedRLEnv,
    broad_sigma: float = 0.015,
    medium_sigma: float = 0.007,
    fine_sigma: float = 0.0025,
    broad_weight: float = 0.50,
    medium_weight: float = 0.30,
    fine_weight: float = 0.20,
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole_board"),
) -> torch.Tensor:
    """Dense radial-alignment progress with broad-to-fine scales.

    The original 3 mm Gaussian is effectively zero for most hole20 starts.
    This potential keeps a useful gradient at 10--20 mm while retaining a
    sharper term near the final insertion gate.  Only the change in potential
    is returned, so standing still cannot farm reward.
    """
    peg_tip, _ = geo.get_peg_tip(env)
    hole_pos, _ = geo.get_hole_center(env, hole_cfg)
    radial = geo.get_radial_error(peg_tip, hole_pos)
    potential = (
        broad_weight * torch.exp(-0.5 * (radial / broad_sigma) ** 2)
        + medium_weight * torch.exp(-0.5 * (radial / medium_sigma) ** 2)
        + fine_weight * torch.exp(-0.5 * (radial / fine_sigma) ** 2)
    )
    key = "_prev_multiscale_alignment"
    if key not in env.extras:
        env.extras[key] = potential.clone()
    prev = env.extras[key]
    reset_mask = env.episode_length_buf == 0
    prev = torch.where(reset_mask, potential, prev)
    env.extras[key] = potential.clone()
    return potential - prev


def approach_depth_progress(
    env: ManagerBasedRLEnv,
    start_depth: float = -0.050,
    gate_depth: float = -0.010,
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole_board"),
) -> torch.Tensor:
    """Reward approaching the surface before alignment-gated insertion.

    The potential rises from the nominal -50 mm reset depth to the -10 mm
    alignment gate, but is bounded above so it does not reward blind insertion.
    It is also bounded below to make large upward retreats costly.
    """
    peg_tip, _ = geo.get_peg_tip(env)
    hole_pos, _ = geo.get_hole_center(env, hole_cfg)
    depth = geo.get_insertion_depth(peg_tip, hole_pos)
    width = max(gate_depth - start_depth, 1.0e-6)
    potential = torch.clamp((depth - start_depth) / width, min=-1.0, max=1.0)
    key = "_prev_approach_depth"
    if key not in env.extras:
        env.extras[key] = potential.clone()
    prev = env.extras[key]
    reset_mask = env.episode_length_buf == 0
    prev = torch.where(reset_mask, potential, prev)
    env.extras[key] = potential.clone()
    return potential - prev


def insertion_progress(
    env: ManagerBasedRLEnv,
    threshold: float = 0.03,
    tilt_gate_deg: float = 50.0,
    radial_gate: float = 0.0025,
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole_board"),
) -> torch.Tensor:
    """Gated, normalized increase in insertion depth.

    Gate: radial_error < 2.5 mm AND tilt < tilt_gate_deg.
    Default tilt gate 50°: baseline uses 3D position control which cannot
    change orientation; the Franka's natural hand tilt is 30-50°, so a 2°
    gate would NEVER open. Tighten the gate in later curriculum stages
    when orientation control is added.
    """
    peg_tip, peg_quat = geo.get_peg_tip(env)
    hole_pos, _ = geo.get_hole_center(env, hole_cfg)

    radial_err = geo.get_radial_error(peg_tip, hole_pos)
    tilt = geo.get_tilt_angle(peg_quat)
    depth = geo.get_insertion_depth(peg_tip, hole_pos)

    tilt_tol = tilt_gate_deg * torch.pi / 180.0
    gate = (radial_err < radial_gate) & (tilt < tilt_tol)

    key = "_prev_insertion_depth"
    if key not in env.extras:
        env.extras[key] = depth.clone()
    prev_depth = env.extras[key]
    # Never compare the first state of a reset episode with the terminal state
    # of the preceding episode in a vectorized environment.
    reset_mask = env.episode_length_buf == 0
    prev_depth = torch.where(reset_mask, depth, prev_depth)
    env.extras[key] = depth.clone()

    # Positive only when the tip moves deeper. Normalizing by the nominal
    # insertion distance keeps the signal useful at millimetre-scale motion.
    progress = torch.clamp((depth - prev_depth) / threshold, min=-1.0, max=1.0)
    return progress * gate.float()


def jam_penalty(
    env: ManagerBasedRLEnv,
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole_board"),
    radial_tol: float = geo.SUCCESS_RADIAL_TOL,
) -> torch.Tensor:
    """Penalty for pushing against walls or surface without proper alignment.

    1.0 when peg is outside hole XY AND pushing down (likely jamming).
    """
    peg_tip, _ = geo.get_peg_tip(env)
    hole_pos, _ = geo.get_hole_center(env, hole_cfg)

    radial_err = geo.get_radial_error(peg_tip, hole_pos)
    pushing_down = peg_tip[:, 2] < (hole_pos[:, 2] + 0.005)

    out_of_hole = radial_err > radial_tol
    return (out_of_hole & pushing_down).float()


def tilt_penalty(
    env: ManagerBasedRLEnv,
    tilt_tol: float = geo.SUCCESS_TILT_TOL_RAD,
) -> torch.Tensor:
    """Penalty for excessive tilt: linear from 2° to 10°."""
    _, peg_quat = geo.get_peg_tip(env)
    tilt = geo.get_tilt_angle(peg_quat)
    # 0 at 2°, 1 at 10°+
    return torch.clamp((tilt - tilt_tol) / (8.0 * torch.pi / 180.0), min=0.0, max=1.0)


def success_bonus(
    env: ManagerBasedRLEnv,
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole_board"),
    radial_tol: float = geo.SUCCESS_RADIAL_TOL,
    depth_required: float = geo.SUCCESS_DEPTH,
    max_depth: float = geo.SUCCESS_MAX_DEPTH,
    tilt_tol: float = geo.SUCCESS_TILT_TOL_RAD,
) -> torch.Tensor:
    """Sparse success bonus: 1.0 when fully inserted (geometry.is_in_hole)."""
    peg_tip, peg_quat = geo.get_peg_tip(env)
    hole_pos, _ = geo.get_hole_center(env, hole_cfg)
    return geo.is_in_hole(
        peg_tip, hole_pos, peg_quat,
        radial_tol=radial_tol,
        depth_required=depth_required,
        max_depth=max_depth,
        tilt_tol=tilt_tol,
    ).float()


def over_insertion_penalty(
    env: ManagerBasedRLEnv,
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole_board"),
    max_depth: float = geo.SUCCESS_MAX_DEPTH + geo.OVER_INSERTION_TOL,
) -> torch.Tensor:
    """Unit terminal penalty when the peg passes below the sleeve bottom."""
    peg_tip, _ = geo.get_peg_tip(env)
    hole_pos, _ = geo.get_hole_center(env, hole_cfg)
    return (geo.get_insertion_depth(peg_tip, hole_pos) > max_depth).float()


def deep_insertion_braking_penalty(
    env: ManagerBasedRLEnv,
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole_board"),
    start_depth: float = 0.035,
    terminal_depth: float = geo.SUCCESS_MAX_DEPTH + geo.OVER_INSERTION_TOL,
) -> torch.Tensor:
    """Continuous warning signal before the terminal over-insertion boundary.

    The sparse terminal cost arrives too late to teach a policy to decelerate.
    This term ramps linearly from zero at ``start_depth`` to one at the
    over-insertion boundary, providing a gradient throughout the 35--42 mm
    braking zone.  The reward configuration supplies the negative weight.
    """
    peg_tip, _ = geo.get_peg_tip(env)
    hole_pos, _ = geo.get_hole_center(env, hole_cfg)
    depth = geo.get_insertion_depth(peg_tip, hole_pos)
    width = max(terminal_depth - start_depth, 1.0e-6)
    return torch.clamp((depth - start_depth) / width, min=0.0, max=1.0)


def time_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Unit per-step cost; the reward configuration supplies a negative weight."""
    return torch.ones(env.num_envs, device=env.device)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large action changes (CLIPPED to prevent value explosion).

    If the actor network diverges it can output actions of magnitude 1e9;
    squaring gives 1e18 and starts a value-function death spiral.
    Actions are bounded to [-1,1] by the policy, so clip the diff to ±2
    and cap the penalty at 20.
    """
    diff = env.action_manager.action - env.action_manager.prev_action
    diff_clipped = torch.clamp(diff, min=-2.0, max=2.0)
    penalty = torch.sum(torch.square(diff_clipped), dim=1)
    return torch.clamp(penalty, max=20.0)


def joint_vel_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize large joint velocities (CLIPPED to prevent value explosion).

    PhysX contact spikes (100kg peg vs kinematic walls) can produce joint
    velocities of 1e9 rad/s; squaring gives 1e18 → value function diverges.
    Clip velocities to ±10 rad/s first, then compute L2 penalty capped at 100.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    vel_clipped = torch.clamp(vel, min=-10.0, max=10.0)
    penalty = torch.sum(torch.square(vel_clipped), dim=1)
    return torch.clamp(penalty, max=100.0)  # Cap at 100 (weight -0.00005 → -0.005/step)
