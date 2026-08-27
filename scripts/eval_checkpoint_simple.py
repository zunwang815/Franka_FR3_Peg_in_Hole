#!/usr/bin/env python3
"""Small, robust checkpoint evaluator for debugging Isaac Lab runs."""

from __future__ import annotations

import argparse
import math
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_phase1 import apply_teacher_observation_noise, apply_verified_physics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--task", default="Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom20mm-MountStable-v0")
    p.add_argument("--episodes", type=int, default=32)
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--episode_length_s",
        type=float,
        default=None,
        help="Optional environment episode length override; 20s gives a 600-step budget.",
    )
    p.add_argument("--nullspace_stiffness", type=float, default=75.0)
    p.add_argument(
        "--barrier_mm", type=float, default=None,
        help="Optional predictive insertion-depth safety barrier; disabled for matched stress evaluation",
    )
    p.add_argument("--gate_radial_mm", type=float, default=None)
    p.add_argument("--gate_tilt_deg", type=float, default=None)
    p.add_argument("--gate_start_depth_mm", type=float, default=-10.0)
    p.add_argument(
        "--teacher_alignment_gate_mm", type=float, default=1.0,
        help="Alignment gate used by the geometric teacher residual policy",
    )
    p.add_argument(
        "--ik_blend", type=float, default=None,
        help="For an online-IK task, solve only this fraction of the initial XY error",
    )
    p.add_argument(
        "--residual_adapter", action="store_true",
        help="Load a frozen-teacher checkpoint with its trainable residual head",
    )
    p.add_argument(
        "--geometric_teacher_residual", action="store_true",
        help="Load a checkpoint containing the geometric-teacher residual head",
    )
    p.add_argument("--teacher_observation_noise_mm", type=float, default=0.5)
    p.add_argument("--teacher_tilt_noise_deg", type=float, default=0.25)
    p.add_argument(
        "--hole_xy_bias_std_mm", type=float, default=0.0,
        help="Fixed per-episode Gaussian XY hole-position estimate bias",
    )
    p.add_argument(
        "--action_noise_std", type=float, default=0.0,
        help="Zero-delay per-control-step normalized action noise",
    )
    p.add_argument(
        "--action_gain_noise_std_pct", type=float, default=0.0,
        help="Zero-delay per-episode control gain noise percentage",
    )
    p.add_argument(
        "--reset_warmup_steps", type=int, default=20,
        help="Zero-action OSC settling steps after reset for single-hole tasks",
    )
    p.add_argument(
        "--xy_assist_blend", type=float, default=None,
        help="Blend policy XY action with direct hole-centering action",
    )
    p.add_argument(
        "--xy_assist_until_depth_mm", type=float, default=-10.0,
        help="Apply XY assist while insertion depth is below this value",
    )
    p.add_argument(
        "--xy_assist_radial_mm", type=float, default=2.0,
        help="Only apply XY assist when radial error exceeds this value",
    )
    p.add_argument("--orientation_upright_assist", action="store_true")
    p.add_argument(
        "--xy_bins_mm",
        action="store_true",
        help="Report success/timeout counts by radial XY target-offset bin",
    )
    p.add_argument(
        "--canonicalize_proprioception",
        action="store_true",
        help="Replace policy joint position/velocity observations with the default offset5 posture",
    )
    args = p.parse_args()
    if args.ik_blend is not None and not 0.0 <= args.ik_blend <= 1.0:
        p.error("--ik_blend must be in [0, 1]")
    if args.residual_adapter and args.geometric_teacher_residual:
        p.error("residual adapter modes are mutually exclusive")
    if args.action_noise_std < 0.0 or args.action_gain_noise_std_pct < 0.0:
        p.error("action perturbation values must be non-negative")
    if args.hole_xy_bias_std_mm < 0.0:
        p.error("hole_xy_bias_std_mm must be non-negative")
    if args.reset_warmup_steps < 0:
        p.error("reset_warmup_steps must be non-negative")
    if args.episode_length_s is not None and args.episode_length_s <= 0.0:
        p.error("episode_length_s must be positive")
    if args.teacher_alignment_gate_mm <= 0.0:
        p.error("--teacher_alignment_gate_mm must be positive")

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, device=args.device).app
    env = None
    try:
        import gymnasium as gym
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg, load_cfg_from_registry
        from rsl_rl.runners import OnPolicyRunner
        from rsl_rl.runners import on_policy_runner as runner_module
        import tasks.franka_peg_in_hole.config.franka  # noqa: F401
        from tasks.franka_peg_in_hole.mdp import geometry as geo
        from scripts.depth_safety import wrap_with_predictive_depth_barrier

        cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        cfg.seed = args.seed
        cfg.scene.num_envs = args.num_envs
        cfg.observations.policy.enable_corruption = True
        apply_verified_physics(cfg)
        if args.episode_length_s is not None:
            cfg.episode_length_s = args.episode_length_s
        if args.geometric_teacher_residual:
            apply_teacher_observation_noise(
                cfg,
                args.teacher_observation_noise_mm,
                args.teacher_tilt_noise_deg,
            )
        if args.ik_blend is not None:
            online_term = getattr(cfg.events, "target_conditioned_arm_pose_online", None)
            if online_term is None:
                raise ValueError("--ik_blend requires an online-IK task")
            online_term.params["target_blend"] = args.ik_blend
        cfg.actions.arm_action.controller_cfg.nullspace_stiffness = args.nullspace_stiffness
        env = gym.make(args.task, cfg=cfg)
        action_perturbation_enabled = (
            args.action_noise_std > 0.0 or args.action_gain_noise_std_pct > 0.0
        )
        if action_perturbation_enabled:
            from scripts.action_perturbation import wrap_with_action_perturbation

            env = wrap_with_action_perturbation(
                env,
                gym=gym,
                torch=torch,
                action_noise_std=args.action_noise_std,
                action_gain_noise_std_pct=args.action_gain_noise_std_pct,
                reset_warmup_steps=args.reset_warmup_steps,
            )
        if args.barrier_mm is not None:
            env = wrap_with_predictive_depth_barrier(
                env,
                barrier_mm=args.barrier_mm,
                position_scale=cfg.actions.arm_action.position_scale,
                gym=gym,
                torch=torch,
                geometry=geo,
                insertion_gate_radial_mm=args.gate_radial_mm,
                insertion_gate_tilt_deg=args.gate_tilt_deg,
                insertion_gate_start_depth_mm=args.gate_start_depth_mm,
                approach_xy_assist_blend=args.xy_assist_blend,
                approach_xy_assist_until_depth_mm=args.xy_assist_until_depth_mm,
                approach_xy_assist_radial_mm=args.xy_assist_radial_mm,
                orientation_upright_assist=args.orientation_upright_assist,
            )
        if args.hole_xy_bias_std_mm > 0.0:
            from scripts.observation_perturbation import wrap_with_hole_xy_bias

            env = wrap_with_hole_xy_bias(
                env,
                gym=gym,
                torch=torch,
                bias_std_mm=args.hole_xy_bias_std_mm,
            )
        uenv = env.unwrapped
        wrapped = RslRlVecEnvWrapper(env)
        agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        if args.geometric_teacher_residual:
            from scripts.residual_policy import GeometricTeacherResidualActorCritic

            runner_module.GeometricTeacherResidualActorCritic = GeometricTeacherResidualActorCritic
            GeometricTeacherResidualActorCritic.default_teacher_alignment_gate_mm = (
                float(args.teacher_alignment_gate_mm)
            )
            agent_cfg.policy.class_name = "GeometricTeacherResidualActorCritic"
        elif args.residual_adapter:
            from scripts.residual_policy import ResidualActorCritic

            runner_module.ResidualActorCritic = ResidualActorCritic
            agent_cfg.policy.class_name = "ResidualActorCritic"
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir="/tmp/peg_simple_eval", device=args.device)
        runner.load(str(args.checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device=args.device)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        # Match eval_phase1's deterministic reset path.  Manager-based event
        # randomization uses the vector wrapper seed, not only cfg.seed.
        wrapped.seed(args.seed)
        obs, _ = wrapped.reset()
        if getattr(cfg.events, "select_target_hole", None) is None and not action_perturbation_enabled:
            zero_action = torch.zeros(
                args.num_envs,
                wrapped.action_space.shape[-1],
                device=wrapped.device,
            )
            for _ in range(args.reset_warmup_steps):
                obs_dict, _, _, _, _ = env.step(zero_action)
                obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
        policy_joint_ids, _ = uenv.scene["robot"].find_joints(
            ["panda_joint.*", "panda_finger_joint.*"], preserve_order=True
        )
        canonical_q = uenv.scene["robot"].data.default_joint_pos[:, policy_joint_ids].clone()

        def adapt_obs(policy_obs):
            if not args.canonicalize_proprioception:
                return policy_obs
            policy_obs = policy_obs.clone()
            policy_obs[:, :9] = canonical_q
            policy_obs[:, 9:18] = 0.0
            return policy_obs

        obs = adapt_obs(obs["policy"] if isinstance(obs, dict) else obs)
        # Capture the target offset before each action.  The environment may
        # reset completed instances inside env.step(), so reading the hole
        # position only after the step would associate an episode with the
        # next target.
        hole_pos0, _ = geo.get_hole_center(uenv)
        nominal_xy = torch.as_tensor(
            cfg.scene.hole_board.init_state.pos[:2],
            device=wrapped.device,
            dtype=hole_pos0.dtype,
        ).expand_as(hole_pos0[:, :2]) + uenv.scene.env_origins[:, :2]
        target_offset = hole_pos0[:, :2] - nominal_xy
        bin_edges_mm = [0.0, 5.0, 10.0, 15.0, 20.0, float("inf")]
        bin_success = [0] * (len(bin_edges_mm) - 1)
        bin_total = [0] * (len(bin_edges_mm) - 1)
        bin_timeout = [0] * (len(bin_edges_mm) - 1)
        bin_over = [0] * (len(bin_edges_mm) - 1)
        completed = success = timeout = over = 0
        terminal_depths = []
        terminal_radials = []
        terminal_tilts = []
        steps = torch.zeros(args.num_envs, dtype=torch.long, device=wrapped.device)
        while completed < args.episodes:
            episode_target_offset = target_offset.clone()
            with torch.inference_mode():
                actions = policy(obs)
            obs_dict, _, terminated, truncated, _ = env.step(actions)
            obs = adapt_obs(obs_dict["policy"])
            # Refresh offsets for any newly reset environments for the next
            # loop iteration; done episodes use episode_target_offset above.
            next_hole_pos, _ = geo.get_hole_center(uenv)
            target_offset = next_hole_pos[:, :2] - nominal_xy
            steps += 1
            done = terminated.bool() | truncated.bool()
            if not bool(done.any()):
                continue
            ids = torch.nonzero(done).flatten()
            take = min(args.episodes - completed, int(ids.numel()))
            ids = ids[:take]
            sm = uenv._success_termination_mask.bool()[ids]
            over_mask = getattr(uenv, "_over_insertion_termination_mask", None)
            if over_mask is None:
                om = torch.zeros_like(sm, dtype=torch.bool)
            else:
                om = over_mask.bool()[ids]
            tm = truncated.bool()[ids] & ~terminated.bool()[ids]
            if args.xy_bins_mm:
                target_r = torch.linalg.vector_norm(
                    episode_target_offset[ids], dim=-1
                ) * 1000.0
                target_r_list = target_r.detach().cpu().tolist()
                sm_list = sm.detach().cpu().tolist()
                tm_list = tm.detach().cpu().tolist()
                om_list = om.detach().cpu().tolist()
                for radius, ok, to, ov in zip(target_r_list, sm_list, tm_list, om_list):
                    b = next(
                        i for i in range(len(bin_edges_mm) - 1)
                        if bin_edges_mm[i] <= radius < bin_edges_mm[i + 1]
                    )
                    bin_total[b] += 1
                    bin_success[b] += int(ok)
                    bin_timeout[b] += int(to)
                    bin_over[b] += int(ov)
            success += int(sm.sum().item())
            over += int(om.sum().item())
            timeout += int(tm.sum().item())
            terminal_depths += uenv._termination_depth[ids].detach().cpu().tolist()
            terminal_radials += uenv._termination_radial_error[ids].detach().cpu().tolist()
            terminal_tilts += uenv._termination_tilt[ids].detach().cpu().tolist()
            completed += take
            steps[ids] = 0
        n = completed
        d = [x * 1000.0 for x in terminal_depths]
        r = [x * 1000.0 for x in terminal_radials]
        t = [x * 180.0 / math.pi for x in terminal_tilts]
        print(f"SIMPLE_RESULT success={success}/{n} rate={success/n:.4f} timeout={timeout} over={over}")
        print(f"TERMINAL depth_mm mean/min/max={sum(d)/n:.3f}/{min(d):.3f}/{max(d):.3f}")
        print(f"TERMINAL radial_mm mean/min/max={sum(r)/n:.3f}/{min(r):.3f}/{max(r):.3f}")
        print(f"TERMINAL tilt_deg mean/min/max={sum(t)/n:.3f}/{min(t):.3f}/{max(t):.3f}")
        if args.xy_bins_mm:
            labels = ["0-5", "5-10", "10-15", "15-20", ">=20"]
            print("XY_TARGET_BINS radial_mm success/total timeout over")
            for label, total, ok, to, ov in zip(
                labels, bin_total, bin_success, bin_timeout, bin_over
            ):
                rate = ok / total if total else float("nan")
                print(f"XY_BIN {label} {ok}/{total} rate={rate:.4f} timeout={to} over={ov}")
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
