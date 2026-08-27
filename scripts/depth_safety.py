"""Runtime action safety filters shared by training and evaluation."""

from __future__ import annotations


def wrap_with_predictive_depth_barrier(
    env,
    *,
    barrier_mm: float | None,
    position_scale,
    gym,
    torch,
    geometry,
    insertion_gate_radial_mm: float | None = None,
    insertion_gate_tilt_deg: float | None = None,
    insertion_gate_hysteresis: float = 1.0,
    insertion_gate_start_depth_mm: float = 0.0,
    insertion_soft_gate_min_scale: float | None = None,
    approach_xy_assist_blend: float | None = None,
    approach_xy_assist_until_depth_mm: float = -10.0,
    approach_xy_assist_radial_mm: float = 2.0,
    approach_z_guard_depth_mm: float | None = None,
    approach_z_guard_min_downward: float = 0.0,
    orientation_upright_assist: bool = False,
    orientation_assist_gain: float = 1.0,
    orientation_assist_clip: float = 0.25,
):
    """Return a vector-env wrapper that constrains unsafe downward insertion.

    Only the task-space Z action is changed. XY/orientation corrections and
    upward withdrawal remain available. The optional alignment gate opens with
    radial/tilt thresholds and closes with wider hysteresis thresholds.
    """

    if isinstance(position_scale, (tuple, list)):
        z_scale = float(position_scale[2])
    else:
        z_scale = float(position_scale)
    barrier = None if barrier_mm is None else float(barrier_mm) / 1000.0
    gate_radial = (
        None if insertion_gate_radial_mm is None
        else float(insertion_gate_radial_mm) / 1000.0
    )
    gate_tilt = (
        None if insertion_gate_tilt_deg is None
        else float(insertion_gate_tilt_deg) * torch.pi / 180.0
    )
    close_radial = None if gate_radial is None else gate_radial * insertion_gate_hysteresis
    close_tilt = None if gate_tilt is None else gate_tilt * insertion_gate_hysteresis
    gate_start_depth = float(insertion_gate_start_depth_mm) / 1000.0

    class PredictiveDepthBarrier(gym.Wrapper):
        def __init__(self, wrapped_env):
            super().__init__(wrapped_env)
            self._previous_depth = None
            self._insertion_enabled = None

        def _geometry(self):
            peg_tip, peg_quat = geometry.get_peg_tip(self.unwrapped)
            hole_pos, _ = geometry.get_hole_center(self.unwrapped)
            depth = geometry.get_insertion_depth(peg_tip, hole_pos)
            radial = geometry.get_radial_error(peg_tip, hole_pos)
            tilt = geometry.get_tilt_angle(peg_quat)
            return depth, radial, tilt, hole_pos[:, :2] - peg_tip[:, :2]

        def _update_gate(self, radial, tilt):
            if gate_radial is None or gate_tilt is None:
                return torch.ones_like(radial, dtype=torch.bool)
            aligned = (radial <= gate_radial) & (tilt <= gate_tilt)
            if self._insertion_enabled is None:
                self._insertion_enabled = aligned.clone()
            still_aligned = (radial <= close_radial) & (tilt <= close_tilt)
            self._insertion_enabled = torch.where(
                self._insertion_enabled, still_aligned, aligned
            )
            return self._insertion_enabled

        def reset(self, **kwargs):
            result = self.env.reset(**kwargs)
            depth, radial, tilt, _ = self._geometry()
            self._previous_depth = depth.clone()
            self._insertion_enabled = None
            self._update_gate(radial, tilt)
            return result

        def step(self, actions):
            current_depth, radial, tilt, peg_to_hole_xy = self._geometry()
            if self._previous_depth is None:
                self._previous_depth = current_depth.clone()
            insertion_enabled = self._update_gate(radial, tilt)
            safe_actions = actions.clone()
            if approach_z_guard_depth_mm is not None:
                far_above_surface = current_depth < approach_z_guard_depth_mm / 1000.0
                max_z_action = torch.full_like(
                    current_depth, -float(approach_z_guard_min_downward)
                )
                guarded_z = torch.minimum(safe_actions[:, 2], max_z_action)
                safe_actions[:, 2] = torch.where(
                    far_above_surface, guarded_z, safe_actions[:, 2]
                )
            if approach_xy_assist_blend is not None:
                assist_active = (
                    (current_depth < approach_xy_assist_until_depth_mm / 1000.0)
                    & (radial > approach_xy_assist_radial_mm / 1000.0)
                )
                target_xy_action = (peg_to_hole_xy / z_scale).clamp(-1.0, 1.0)
                blended_xy = (
                    (1.0 - float(approach_xy_assist_blend)) * safe_actions[:, :2]
                    + float(approach_xy_assist_blend) * target_xy_action
                )
                safe_actions[:, :2] = torch.where(
                    assist_active.unsqueeze(-1), blended_xy, safe_actions[:, :2]
                )
            if orientation_upright_assist and safe_actions.shape[-1] >= 6:
                # Diagnostic-only stabilizer matching the verified oracle:
                # preserve yaw while driving the peg axis toward world Z.
                from isaaclab.utils.math import (
                    axis_angle_from_quat,
                    quat_apply,
                    quat_conjugate,
                    quat_from_angle_axis,
                    quat_mul,
                )
                _, peg_quat = geometry.get_peg_tip(self.unwrapped)
                world_z = torch.zeros_like(peg_quat[:, :3])
                world_z[:, 2] = 1.0
                current_axis = quat_apply(peg_quat, world_z)
                target_axis = world_z * torch.sign(current_axis[:, 2:3])
                correction_axis = torch.linalg.cross(current_axis, target_axis)
                sin_angle = torch.linalg.vector_norm(correction_axis, dim=-1)
                cos_angle = torch.sum(current_axis * target_axis, dim=-1).clamp(-1.0, 1.0)
                correction_axis = correction_axis / sin_angle.unsqueeze(-1).clamp_min(1.0e-8)
                correction_angle = torch.atan2(sin_angle, cos_angle)
                correction = quat_from_angle_axis(correction_angle, correction_axis)
                quat_error = quat_mul(correction, peg_quat)
                rot_error = axis_angle_from_quat(quat_mul(quat_error, quat_conjugate(peg_quat)))
                assist = (
                    float(orientation_assist_gain) * rot_error
                ).clamp(-float(orientation_assist_clip), float(orientation_assist_clip))
                safe_actions[:, 3:6] = assist
            if insertion_soft_gate_min_scale is not None:
                radial_scale = torch.clamp(gate_radial / radial.clamp_min(1.0e-6), max=1.0)
                tilt_scale = torch.clamp(gate_tilt / tilt.clamp_min(1.0e-6), max=1.0)
                alignment_scale = torch.minimum(radial_scale, tilt_scale).clamp(
                    min=float(insertion_soft_gate_min_scale)
                )
                alignment_scale = torch.where(
                    current_depth >= gate_start_depth,
                    alignment_scale,
                    torch.ones_like(alignment_scale),
                )
                safe_actions[:, 2] = torch.where(
                    safe_actions[:, 2] < 0.0,
                    safe_actions[:, 2] * alignment_scale,
                    safe_actions[:, 2],
                )

            min_safe_z_action = torch.full_like(current_depth, -torch.inf)
            if barrier is not None:
                downward_momentum = torch.clamp(
                    current_depth - self._previous_depth, min=0.0
                )
                remaining = barrier - current_depth - downward_momentum
                min_safe_z_action = -remaining / z_scale
            if insertion_soft_gate_min_scale is None:
                gate_allows_descent = insertion_enabled | (current_depth < gate_start_depth)
                min_safe_z_action = torch.where(
                    gate_allows_descent, min_safe_z_action, torch.zeros_like(min_safe_z_action)
                )
            safe_actions[:, 2] = torch.maximum(
                safe_actions[:, 2], min_safe_z_action
            ).clamp(max=1.0)

            result = self.env.step(safe_actions)
            _, _, terminated, truncated, _ = result
            self._previous_depth = current_depth.clone()
            done = terminated.bool() | truncated.bool()
            if done.any():
                reset_depth, reset_radial, reset_tilt, _ = self._geometry()
                self._previous_depth[done] = reset_depth[done]
                if self._insertion_enabled is not None and gate_radial is not None:
                    reset_aligned = (reset_radial <= gate_radial) & (reset_tilt <= gate_tilt)
                    self._insertion_enabled[done] = reset_aligned[done]
            return result

    return PredictiveDepthBarrier(env)
