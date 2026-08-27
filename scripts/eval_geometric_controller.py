#!/usr/bin/env python3
"""Evaluate a deterministic Cartesian geometric insertion controller.

This deliberately bypasses PPO.  The controller uses the measured peg tip and
the known hole center to form a bounded relative-pose command for the existing
6-D OSC action term.  It is intended as a dynamics/reachability baseline for
the hole20 and later hole50/23mm stages.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_phase1 import apply_verified_physics
from scripts.geometric_teacher import compute_geometric_action


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--task",
        default="Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom20mm-MountStableReward-v0",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--geometric_teacher_residual", action="store_true")
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target_hole_id",
        type=int,
        default=-1,
        help="For array tasks, evaluate one fixed hole (0-5); -1 keeps reset randomization.",
    )
    parser.add_argument(
        "--array_origin_zero",
        action="store_true",
        help="For array tasks, disable whole-array XY randomization for staged reachability audits.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument(
        "--episode_length_s",
        type=float,
        default=None,
        help="Optional environment episode length override; 20s gives a 600-step budget.",
    )
    parser.add_argument("--position_scale", type=float, default=None)
    parser.add_argument("--kp_position", type=float, default=0.8)
    parser.add_argument("--kp_orientation", type=float, default=0.8)
    parser.add_argument("--approach_depth_mm", type=float, default=-10.0)
    parser.add_argument("--insert_depth_mm", type=float, default=30.0)
    parser.add_argument("--alignment_gate_mm", type=float, default=1.0)
    parser.add_argument("--tilt_gate_deg", type=float, default=2.0)
    parser.add_argument(
        "--hole_xy_noise_std_mm",
        type=float,
        default=0.0,
        help="Gaussian XY error in the measured hole center (not physical geometry).",
    )
    parser.add_argument(
        "--hole_xy_bias_std_mm",
        type=float,
        default=0.0,
        help="Fixed-per-episode Gaussian XY bias in the measured hole center.",
    )
    parser.add_argument(
        "--tip_position_noise_std_mm",
        type=float,
        default=0.0,
        help="Gaussian XYZ error in the measured peg-tip position.",
    )
    parser.add_argument(
        "--tip_orientation_noise_std_deg",
        type=float,
        default=0.0,
        help="Gaussian small-angle error in the measured peg orientation.",
    )
    parser.add_argument(
        "--measurement_ema_alpha",
        type=float,
        default=1.0,
        help="EMA weight for noisy tip/hole measurements; 1.0 disables filtering.",
    )
    parser.add_argument(
        "--action_noise_std",
        type=float,
        default=0.0,
        help="Gaussian normalized action noise applied before the optional delay.",
    )
    parser.add_argument(
        "--action_delay_steps",
        type=int,
        default=0,
        help="Number of control steps by which commands are delayed.",
    )
    parser.add_argument(
        "--action_gain_noise_std_pct",
        type=float,
        default=0.0,
        help="Per-episode percentage std of the actuator/control gain multiplier.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path for the aggregate and per-episode report.",
    )
    args = parser.parse_args()
    if args.episodes <= 0 or args.num_envs <= 0 or args.max_steps <= 0:
        parser.error("episodes/num_envs/max_steps must be positive")
    if args.episode_length_s is not None and args.episode_length_s <= 0.0:
        parser.error("episode_length_s must be positive")
    for name in (
        "hole_xy_noise_std_mm",
        "hole_xy_bias_std_mm",
        "tip_position_noise_std_mm",
        "tip_orientation_noise_std_deg",
        "action_noise_std",
        "action_gain_noise_std_pct",
    ):
        if getattr(args, name) < 0.0:
            parser.error(f"{name} must be non-negative")
    if args.action_delay_steps < 0:
        parser.error("action_delay_steps must be non-negative")
    if args.target_hole_id < -1 or args.target_hole_id >= 6:
        parser.error("target_hole_id must be -1 or in [0, 5]")
    if not 0.0 < args.measurement_ema_alpha <= 1.0:
        parser.error("measurement_ema_alpha must be in (0, 1]")
    if args.geometric_teacher_residual and args.checkpoint is None:
        parser.error("--geometric_teacher_residual requires --checkpoint")
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.expanduser().resolve()
        if not args.checkpoint.is_file():
            parser.error(f"checkpoint does not exist: {args.checkpoint}")

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, device=args.device).app
    env = None
    try:
        import gymnasium as gym
        import torch
        from isaaclab.utils.math import (
            axis_angle_from_quat,
            quat_apply,
            quat_conjugate,
            quat_from_angle_axis,
            quat_mul,
        )
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

        import tasks.franka_peg_in_hole.config.franka  # noqa: F401
        from tasks.franka_peg_in_hole.mdp import geometry as geo

        cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        cfg.seed = args.seed
        cfg.scene.num_envs = args.num_envs
        cfg.observations.policy.enable_corruption = False
        select_target = getattr(cfg.events, "select_target_hole", None)
        if args.target_hole_id >= 0 and select_target is not None:
            select_target.params["target_hole_id"] = args.target_hole_id
        randomize_hole = getattr(cfg.events, "randomize_hole", None)
        if args.array_origin_zero and randomize_hole is not None:
            if "array_hole_spacing" in randomize_hole.params:
                randomize_hole.params["x_range"] = (0.0, 0.0)
                randomize_hole.params["y_range"] = (0.0, 0.0)
        apply_verified_physics(cfg)
        if args.episode_length_s is not None:
            cfg.episode_length_s = args.episode_length_s
        cfg.actions.arm_action.controller_cfg.nullspace_stiffness = 20.0
        scale = float(args.position_scale or cfg.actions.arm_action.position_scale)
        cfg.actions.arm_action.position_scale = scale
        env = gym.make(args.task, cfg=cfg)
        uenv = env.unwrapped
        step_env = env
        if args.target_hole_id >= 0:
            uenv._fixed_target_hole_id = args.target_hole_id
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        obs, _ = env.reset(seed=args.seed)
        policy = None
        if args.checkpoint is not None:
            agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
            if args.geometric_teacher_residual:
                from scripts.residual_policy import GeometricTeacherResidualActorCritic

                GeometricTeacherResidualActorCritic.default_teacher_alignment_gate_mm = float(
                    args.alignment_gate_mm
                )
                model = GeometricTeacherResidualActorCritic(
                    num_actor_obs=obs["policy"].shape[-1],
                    num_critic_obs=obs["policy"].shape[-1],
                    num_actions=6,
                    actor_hidden_dims=list(agent_cfg.policy.actor_hidden_dims),
                    critic_hidden_dims=list(agent_cfg.policy.critic_hidden_dims),
                    activation=agent_cfg.policy.activation,
                    init_noise_std=float(agent_cfg.policy.init_noise_std),
                ).to(uenv.device)
            else:
                raise ValueError("--checkpoint currently requires --geometric_teacher_residual")
            checkpoint_data = torch.load(
                str(args.checkpoint), map_location=uenv.device, weights_only=False
            )
            model.load_state_dict(checkpoint_data["model_state_dict"], strict=False)
            model.eval()
            policy = model.act_inference

        def enforce_fixed_array_target(env_ids):
            """Reapply a fixed target after ManagerBasedRLEnv auto-resets."""
            if args.target_hole_id < 0 or select_target is None:
                return
            from tasks.franka_peg_in_hole.mdp import events as array_events

            array_events.select_target_hole(
                uenv, env_ids, num_holes=6, target_hole_id=args.target_hole_id
            )
            randomize = getattr(cfg.events, "randomize_hole", None)
            if randomize is not None:
                array_events.reset_hole_position_uniform(
                    uenv, env_ids, **randomize.params
                )

        if args.target_hole_id >= 0:
            enforce_fixed_array_target(
                torch.arange(args.num_envs, device=uenv.device, dtype=torch.long)
            )

        def vertical_target_quat(current_quat):
            world_z = torch.zeros_like(current_quat[:, :3])
            world_z[:, 2] = 1.0
            current_axis = quat_apply(current_quat, world_z)
            target_axis = world_z * torch.where(
                current_axis[:, 2:3] >= 0.0, 1.0, -1.0
            )
            correction_axis = torch.linalg.cross(current_axis, target_axis)
            correction_sin = torch.linalg.vector_norm(correction_axis, dim=-1)
            correction_cos = torch.sum(current_axis * target_axis, dim=-1)
            correction_axis = correction_axis / correction_sin.unsqueeze(-1).clamp_min(1.0e-8)
            correction = quat_from_angle_axis(
                torch.atan2(correction_sin, correction_cos), correction_axis
            )
            return quat_mul(correction, current_quat)

        # All perturbations below are injected at the controller interface.
        # The simulator geometry and termination criteria remain untouched, so
        # a noisy result is directly comparable with the clean oracle baseline.
        noise_device = uenv.device
        action_history = [
            torch.zeros((args.num_envs, 6), device=noise_device)
            for _ in range(args.action_delay_steps)
        ]
        action_gain = torch.ones(args.num_envs, device=noise_device)
        hole_xy_bias = torch.zeros((args.num_envs, 2), device=noise_device)

        def sample_episode_gain(env_ids=None):
            if args.action_gain_noise_std_pct == 0.0:
                value = torch.ones(
                    args.num_envs if env_ids is None else len(env_ids),
                    device=noise_device,
                )
            else:
                count = args.num_envs if env_ids is None else len(env_ids)
                value = 1.0 + torch.randn(count, device=noise_device) * (
                    args.action_gain_noise_std_pct / 100.0
                )
                value = value.clamp_min(0.05)
            if env_ids is None:
                action_gain[:] = value
            else:
                action_gain[env_ids] = value

        sample_episode_gain()

        def sample_hole_xy_bias(env_ids=None):
            count = args.num_envs if env_ids is None else int(env_ids.numel())
            value = torch.randn((count, 2), device=noise_device) * (
                args.hole_xy_bias_std_mm / 1000.0
            )
            if env_ids is None:
                hole_xy_bias[:] = value
            else:
                hole_xy_bias[env_ids] = value

        sample_hole_xy_bias()

        def reset_targets():
            tip, quat = geo.get_peg_tip(uenv)
            hole, _ = geo.get_hole_center(uenv)
            surface_z = hole[:, 2] + 0.010
            return tip.detach().clone(), quat.detach().clone(), hole.detach().clone(), surface_z

        # The single-hole fixed-peg oracle benefits from a 20-step OSC warmup.
        # The array task synchronizes its standalone peg during reset; sending
        # zero OSC commands here would instead let its null-space dynamics
        # drift the carefully selected array initial posture before control.
        if select_target is None:
            zero_action = torch.zeros((args.num_envs, 6), device=noise_device)
            for _ in range(20):
                obs, _, _, _, _ = step_env.step(zero_action)
        tip, current_quat, hole, surface_z = reset_targets()
        desired_quat = vertical_target_quat(current_quat)
        filtered_tip = tip.detach().clone()
        filtered_hole = hole.detach().clone()
        filtered_rot_error = torch.zeros((args.num_envs, 3), device=noise_device)
        initial_radial = torch.linalg.vector_norm(tip[:, :2] - hole[:, :2], dim=-1)
        initial_hole_offset = getattr(
            uenv,
            "_hole_offset_xy",
            torch.zeros((args.num_envs, 2), device=noise_device),
        ).detach().clone()
        completed = success = timeout = over = 0
        steps = torch.zeros(args.num_envs, dtype=torch.long, device=uenv.device)
        terminal_depth: list[float] = []
        terminal_radial: list[float] = []
        terminal_tilt: list[float] = []
        episode_records: list[dict] = []
        phase_counts = {"approach": 0, "align": 0, "insert": 0}

        while completed < args.episodes:
            tip, quat = geo.get_peg_tip(uenv)
            hole, _ = geo.get_hole_center(uenv)
            surface_z = hole[:, 2] + 0.010

            measured_tip = tip
            if args.tip_position_noise_std_mm > 0.0:
                measured_tip = measured_tip + torch.randn_like(measured_tip) * (
                    args.tip_position_noise_std_mm / 1000.0
                )
            measured_hole = hole
            if args.hole_xy_noise_std_mm > 0.0:
                measured_hole = measured_hole.clone()
                measured_hole[:, :2] += torch.randn_like(measured_hole[:, :2]) * (
                    args.hole_xy_noise_std_mm / 1000.0
                )
            if args.hole_xy_bias_std_mm > 0.0:
                measured_hole = measured_hole.clone()
                measured_hole[:, :2] += hole_xy_bias
            if args.measurement_ema_alpha < 1.0:
                alpha = args.measurement_ema_alpha
                filtered_tip.mul_(1.0 - alpha).add_(alpha * measured_tip)
                filtered_hole.mul_(1.0 - alpha).add_(alpha * measured_hole)
                measured_tip = filtered_tip
                measured_hole = filtered_hole

            depth = surface_z - measured_tip[:, 2]
            radial = torch.linalg.vector_norm(
                measured_tip[:, :2] - measured_hole[:, :2], dim=-1
            )
            tilt = geo.get_tilt_angle(quat)
            rot_error = axis_angle_from_quat(
                quat_mul(desired_quat, quat_conjugate(quat))
            )
            if args.tip_orientation_noise_std_deg > 0.0:
                rot_error = rot_error + torch.randn_like(rot_error) * (
                    args.tip_orientation_noise_std_deg * math.pi / 180.0
                )
            if args.measurement_ema_alpha < 1.0:
                alpha = args.measurement_ema_alpha
                filtered_rot_error.mul_(1.0 - alpha).add_(alpha * rot_error)
                rot_error = filtered_rot_error
            if policy is None:
                actions, insert_mask = compute_geometric_action(
                    measured_hole[:, :2] - measured_tip[:, :2],
                    depth,
                    tilt,
                    rot_error,
                    position_scale=scale,
                    kp_position=args.kp_position,
                    kp_orientation=args.kp_orientation,
                    approach_depth_mm=args.approach_depth_mm,
                    insert_depth_mm=args.insert_depth_mm,
                    alignment_gate_mm=args.alignment_gate_mm,
                    tilt_gate_deg=args.tilt_gate_deg,
                )
            else:
                policy_obs = obs["policy"] if isinstance(obs, dict) else obs
                policy_obs = policy_obs.clone()
                if args.hole_xy_bias_std_mm > 0.0:
                    # The measured hole is nominal hole + episode bias.
                    policy_obs[:, 18:20] += hole_xy_bias
                with torch.inference_mode():
                    actions = policy(policy_obs)
                insert_mask = torch.zeros(
                    args.num_envs, dtype=torch.bool, device=noise_device
                )

            if args.action_noise_std > 0.0:
                actions = actions + torch.randn_like(actions) * args.action_noise_std
            actions = (actions * action_gain.unsqueeze(-1)).clamp(-1.0, 1.0)
            if action_history:
                action_history.append(actions)
                applied_actions = action_history.pop(0)
            else:
                applied_actions = actions

            phase_counts["insert"] += int(insert_mask.sum().item())
            phase_counts["align"] += int(((~insert_mask) & (depth >= args.approach_depth_mm / 1000.0)).sum().item())
            phase_counts["approach"] += int((depth < args.approach_depth_mm / 1000.0).sum().item())

            obs, _, terminated, truncated, _ = step_env.step(applied_actions)
            steps += 1
            done = terminated.bool() | truncated.bool()
            if not bool(done.any()):
                # A hard per-episode timeout is handled by the task's normal
                # timeout; this guard only prevents a malformed config from
                # looping forever.
                if int(steps.max().item()) >= args.max_steps:
                    done = steps >= args.max_steps
                else:
                    continue

            ids = torch.nonzero(done).flatten()
            take = min(args.episodes - completed, int(ids.numel()))
            ids = ids[:take]
            sm = uenv._success_termination_mask.bool()[ids]
            over_mask = getattr(
                uenv,
                "_over_insertion_termination_mask",
                torch.zeros(args.num_envs, dtype=torch.bool, device=uenv.device),
            )
            om = over_mask.bool()[ids]
            tm = truncated.bool()[ids] & ~terminated.bool()[ids]
            success += int(sm.sum().item())
            over += int(om.sum().item())
            timeout += int(tm.sum().item())
            depths = uenv._termination_depth[ids].detach().cpu().tolist()
            radials = uenv._termination_radial_error[ids].detach().cpu().tolist()
            tilts = getattr(
                uenv,
                "_termination_tilt",
                torch.zeros(args.num_envs, device=uenv.device),
            )[ids].detach().cpu().tolist()
            terminal_depth.extend(depths)
            terminal_radial.extend(radials)
            terminal_tilt.extend(tilts)
            for index, env_id in enumerate(ids.detach().cpu().tolist()):
                if bool(sm[index]):
                    reason = "success"
                elif bool(om[index]):
                    reason = "over_insertion"
                elif bool(tm[index]):
                    reason = "timeout"
                else:
                    reason = "failure"
                episode_records.append(
                    {
                        "episode": completed + index,
                        "env_id": env_id,
                        "reason": reason,
                        "steps": int(steps[env_id].item()),
                        "initial_radial_mm": float(initial_radial[env_id].item() * 1000.0),
                        "hole_offset_xy_mm": [
                            float(value * 1000.0)
                            for value in initial_hole_offset[env_id].detach().cpu().tolist()
                        ],
                        "terminal_depth_mm": float(depths[index] * 1000.0),
                        "terminal_radial_mm": float(radials[index] * 1000.0),
                        "terminal_tilt_deg": float(tilts[index] * 180.0 / math.pi),
                    }
                )
            completed += take
            steps[ids] = 0

            enforce_fixed_array_target(ids)

            if action_history:
                for queued in action_history:
                    queued[ids] = 0.0
            sample_episode_gain(ids)
            sample_hole_xy_bias(ids)

            # env.step() has already reset completed environments. Capture a
            # fresh upright reference orientation for the next episode.
            new_tip, new_quat = geo.get_peg_tip(uenv)
            desired_quat[ids] = vertical_target_quat(new_quat[ids])
            filtered_tip[ids] = new_tip[ids]
            filtered_hole[ids] = geo.get_hole_center(uenv)[0][ids]
            filtered_rot_error[ids] = 0.0
            reset_hole_offset = getattr(
                uenv,
                "_hole_offset_xy",
                torch.zeros((args.num_envs, 2), device=noise_device),
            )
            initial_radial[ids] = torch.linalg.vector_norm(
                new_tip[ids, :2] - geo.get_hole_center(uenv)[0][ids, :2], dim=-1
            )
            initial_hole_offset[ids] = reset_hole_offset[ids]

        n = completed
        d = [x * 1000.0 for x in terminal_depth]
        r = [x * 1000.0 for x in terminal_radial]
        t = [x * 180.0 / math.pi for x in terminal_tilt]
        reason_counts = {
            "success": success,
            "timeout": timeout,
            "over_insertion": over,
            "failure": n - success - timeout - over,
        }
        result = {
            "schema_version": 1,
            "task": args.task,
            "seed": args.seed,
            "episodes": n,
            "num_envs": args.num_envs,
            "max_steps": args.max_steps,
            "controller": {
                key: value
                for key, value in vars(args).items()
                if key not in {"task", "episodes", "num_envs", "seed", "device", "max_steps", "output"}
            },
            "result": {
                "success": success,
                "timeout": timeout,
                "over_insertion": over,
                "failure": n - success - timeout - over,
                "success_rate": success / n,
                "reason_counts": reason_counts,
            },
            "terminal": {
                "depth_mm": {"mean": sum(d) / n, "min": min(d), "max": max(d)},
                "radial_mm": {"mean": sum(r) / n, "min": min(r), "max": max(r)},
                "tilt_deg": {"mean": sum(t) / n, "min": min(t), "max": max(t)},
            },
            "phase_counts": phase_counts,
            "episodes_detail": episode_records,
        }
        print(f"GEOM_RESULT success={success}/{n} rate={success / n:.4f} timeout={timeout} over={over}")
        print(f"GEOM_TERMINAL depth_mm mean/min/max={sum(d)/n:.3f}/{min(d):.3f}/{max(d):.3f}")
        print(f"GEOM_TERMINAL radial_mm mean/min/max={sum(r)/n:.3f}/{min(r):.3f}/{max(r):.3f}")
        print(f"GEOM_TERMINAL tilt_deg mean/min/max={sum(t)/n:.3f}/{min(t):.3f}/{max(t):.3f}")
        print(f"GEOM_PHASE_COUNTS {phase_counts}")
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            print(f"GEOM_REPORT {args.output}")
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
