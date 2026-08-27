#!/usr/bin/env python3
"""Capture a verified policy under a named report protocol, then replay it."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_phase1 import apply_teacher_observation_noise

CURRICULUM_TASKS = {
    "baseline": "Isaac-PegInHole-Franka-OSC-Pose6D-Baseline-v0",
    "offset5": "Isaac-PegInHole-Franka-OSC-Pose6D-PegOffset5mm-v0",
    "hole_xy5": "Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom5mm-v0",
    "hole10": "Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom10mm-v0",
    "hole10_outer": "Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom10mmOuterMix-v0",
    "hole15": "Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom15mm-v0",
    "hole15_mix": "Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom15mmMix-v0",
    "hole20": "Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom20mm-v0",
    "hole20_mount_stable": "Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom20mm-MountStable-v0",
    "hole20_mount_stable_ik": "Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom20mm-MountStableIK-v0",
    "hole20_online_ik_canonical": "Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom20mm-MountStableOnlineIKCanonical-v0",
    "hole30": "Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom30mm-v0",
    "hole50": "Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom50mm-v0",
    "hole25": "Isaac-PegInHole-Franka-OSC-Pose6D-Hole25mm-v0",
    "hole23": "Isaac-PegInHole-Franka-OSC-Pose6D-Hole23mm-v0",
}

REPORT_PROTOCOLS = {
    "task1_basic": {
        "checkpoint": PROJECT_ROOT / "runs/ppo/hole20_reward/20260818_130108/model_49.pt",
        "task": "Isaac-PegInHole-Franka-OSC-Pose6D-Hole23mm-v0",
        "episode_length_s": 20.0,
        "nullspace_stiffness": 20.0,
        "teacher_alignment_gate_mm": 1.0,
        "teacher_observation_noise_mm": 0.5,
        "teacher_tilt_noise_deg": 0.25,
        "hole_xy_bias_std_mm": 0.0,
        "action_noise_std": 0.0,
        "action_gain_noise_std_pct": 0.0,
        "reset_warmup_steps": 20,
        "barrier_mm": 38.0,
        "observation_corruption": True,
    },
    "task1_stress": {
        "checkpoint": PROJECT_ROOT / "runs/ppo/stress/custom/20260818_164739/model_499.pt",
        "task": "Isaac-PegInHole-Franka-OSC-Pose6D-Hole23mm-v0",
        "episode_length_s": 20.0,
        "nullspace_stiffness": 20.0,
        "teacher_alignment_gate_mm": 1.0,
        "teacher_observation_noise_mm": 0.0,
        "teacher_tilt_noise_deg": 0.0,
        "hole_xy_bias_std_mm": 0.5,
        "action_noise_std": 0.02,
        "action_gain_noise_std_pct": 5.0,
        "reset_warmup_steps": 20,
        "barrier_mm": None,
        "observation_corruption": True,
    },
    "task2_basic": {
        "checkpoint": PROJECT_ROOT / "runs/ppo/custom/20260818_141826/model_49.pt",
        "task": "Isaac-PegInHoleArray-Franka-OSC-Pose6D-v0",
        "episode_length_s": 20.0,
        "nullspace_stiffness": 20.0,
        "teacher_alignment_gate_mm": 1.0,
        "teacher_observation_noise_mm": 0.5,
        "teacher_tilt_noise_deg": 0.25,
        "hole_xy_bias_std_mm": 0.0,
        "action_noise_std": 0.0,
        "action_gain_noise_std_pct": 0.0,
        "reset_warmup_steps": 0,
        "barrier_mm": None,
        "observation_corruption": True,
    },
    "task2_fair": {
        "checkpoint": PROJECT_ROOT / "runs/ppo/custom/20260818_232714/model_499.pt",
        "task": "Isaac-PegInHoleArray-Franka-OSC-Pose6D-v0",
        "episode_length_s": 20.0,
        "nullspace_stiffness": 20.0,
        "teacher_alignment_gate_mm": 2.0,
        "teacher_observation_noise_mm": 0.0,
        "teacher_tilt_noise_deg": 0.0,
        "hole_xy_bias_std_mm": 0.5,
        "action_noise_std": 0.0305,
        "action_gain_noise_std_pct": 5.0,
        "reset_warmup_steps": 0,
        "barrier_mm": None,
        "observation_corruption": True,
    },
}

CUSTOM_PROTOCOL_DEFAULTS = {
    "checkpoint": None,
    "task": None,
    "episode_length_s": None,
    "nullspace_stiffness": None,
    "teacher_alignment_gate_mm": 1.0,
    "teacher_observation_noise_mm": 0.5,
    "teacher_tilt_noise_deg": 0.25,
    "hole_xy_bias_std_mm": 0.0,
    "action_noise_std": 0.0,
    "action_gain_noise_std_pct": 0.0,
    "reset_warmup_steps": 0,
    "barrier_mm": None,
    "observation_corruption": True,
}


def apply_verified_physics(env_cfg) -> None:
    """Apply the physics settings shared by training and evaluation."""
    env_cfg.sim.dt = 1.0 / 120.0
    env_cfg.decimation = 4
    env_cfg.sim.render_interval = env_cfg.decimation
    env_cfg.sim.physx.enable_enhanced_determinism = True
    articulation = env_cfg.scene.robot.spawn.articulation_props
    articulation.solver_position_iteration_count = 16
    articulation.solver_velocity_iteration_count = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a verified Pose6D PPO policy using process-isolated replay",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--protocol", choices=("custom", *REPORT_PROTOCOLS), default="custom",
        help="Named protocol matching one row of 任务1_任务2_总结报告.md",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Optional override; report protocols select their accepted checkpoint automatically",
    )
    parser.add_argument("--stage", choices=tuple(CURRICULUM_TASKS), default="baseline")
    parser.add_argument("--task", default=None, help="Optional report-protocol task override")
    parser.add_argument("--episode_length_s", type=float, default=None)
    parser.add_argument("--nullspace_stiffness", type=float, default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument(
        "--capture_num_envs", type=int, default=128,
        help="Parallel headless environments used to match formal evaluation physics",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--replay_fps", type=float, default=60.0)
    parser.add_argument(
        "--initial_hold_seconds", type=float, default=3.0,
        help="Seconds to display the initial pose before trajectory playback",
    )
    parser.add_argument("--hold_seconds", type=float, default=5.0)
    parser.add_argument(
        "--allow_stage_mismatch", action="store_true",
        help="Allow a checkpoint directory whose curriculum stage differs from --stage",
    )
    parser.add_argument(
        "--observation_corruption", action=argparse.BooleanOptionalAction, default=None,
    )
    parser.add_argument(
        "--geometric_teacher_residual", action=argparse.BooleanOptionalAction, default=None,
        help="Load a checkpoint containing the geometric-teacher residual head",
    )
    parser.add_argument("--teacher_alignment_gate_mm", type=float, default=None)
    parser.add_argument("--teacher_observation_noise_mm", type=float, default=None)
    parser.add_argument("--teacher_tilt_noise_deg", type=float, default=None)
    parser.add_argument("--hole_xy_bias_std_mm", type=float, default=None)
    parser.add_argument("--action_noise_std", type=float, default=None)
    parser.add_argument("--action_gain_noise_std_pct", type=float, default=None)
    parser.add_argument("--reset_warmup_steps", type=int, default=None)
    parser.add_argument(
        "--barrier_mm", type=float, default=None,
        help="Optional predictive insertion-depth barrier; task1_basic uses 38 mm",
    )
    parser.add_argument(
        "--fixed_target_hole_id", type=int, default=None,
        help="Array replay: force every reset to target this hole id",
    )
    parser.add_argument(
        "--show_table", action=argparse.BooleanOptionalAction, default=False,
        help="Replay-only: add the original SeattleLabTable visual asset (no training physics)",
    )
    parser.add_argument(
        "--show_fixture_plate", action=argparse.BooleanOptionalAction, default=True,
        help="Replay-only: show a visual thin plate with a circular opening (no collision)",
    )
    parser.add_argument(
        "--screenshot_dir", type=Path, default=None,
        help="Optional replay screenshots (initial/final) for scene visibility checks",
    )
    parser.add_argument(
        "--trajectory_output", type=Path, default=None,
        help="Optional persistent .pt trajectory and adjacent .json protocol record",
    )
    # Internal arguments used by the parent process. They are deliberately
    # hidden so the public interface remains a single command.
    parser.add_argument("--_mode", choices=("auto", "capture", "replay"), default="auto", help=argparse.SUPPRESS)
    parser.add_argument("--_trajectory_file", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    defaults = dict(CUSTOM_PROTOCOL_DEFAULTS)
    if args.protocol != "custom":
        defaults.update(REPORT_PROTOCOLS[args.protocol])
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.geometric_teacher_residual is None:
        args.geometric_teacher_residual = args.protocol != "custom"
    if args.checkpoint is None:
        parser.error("--checkpoint is required when --protocol custom is used")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if (
        args.episodes <= 0 or args.capture_num_envs <= 0 or args.speed <= 0
        or args.replay_fps <= 0 or args.initial_hold_seconds < 0 or args.hold_seconds < 0
    ):
        parser.error("episodes/speed/replay_fps must be positive and hold_seconds non-negative")
    if args.episode_length_s is not None and args.episode_length_s <= 0.0:
        parser.error("--episode_length_s must be positive")
    if args.nullspace_stiffness is not None and args.nullspace_stiffness < 0.0:
        parser.error("--nullspace_stiffness must be non-negative")
    for name in (
        "teacher_observation_noise_mm", "teacher_tilt_noise_deg",
        "hole_xy_bias_std_mm", "action_noise_std", "action_gain_noise_std_pct",
    ):
        if getattr(args, name) < 0.0:
            parser.error(f"--{name} must be non-negative")
    if args.teacher_alignment_gate_mm <= 0.0:
        parser.error("--teacher_alignment_gate_mm must be positive")
    if args.reset_warmup_steps < 0:
        parser.error("--reset_warmup_steps must be non-negative")
    if args.barrier_mm is not None and args.barrier_mm <= 0.0:
        parser.error("--barrier_mm must be positive")
    if args.fixed_target_hole_id is not None and not 0 <= args.fixed_target_hole_id < 6:
        parser.error("--fixed_target_hole_id must be in [0, 5]")
    if args._mode != "auto" and args._trajectory_file is None:
        parser.error("internal worker mode requires --_trajectory_file")
    if args.trajectory_output is not None:
        args.trajectory_output = args.trajectory_output.expanduser().resolve()
    stage_dir = args.checkpoint.parent.parent.name if len(args.checkpoint.parents) >= 2 else ""
    if (
        args.task is None
        and stage_dir in CURRICULUM_TASKS
        and stage_dir != args.stage
        and not args.allow_stage_mismatch
    ):
        parser.error(
            f"checkpoint belongs to stage '{stage_dir}', but --stage is '{args.stage}'. "
            f"Use --stage {stage_dir}, select a {args.stage} checkpoint, or explicitly pass "
            "--allow_stage_mismatch for an intentional cross-stage test."
        )
    return args


def worker_command(args: argparse.Namespace, mode: str, trajectory_file: Path) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--checkpoint", str(args.checkpoint),
        "--protocol", args.protocol,
        "--stage", args.stage,
        "--episodes", str(args.episodes),
        "--capture_num_envs", str(args.capture_num_envs),
        "--seed", str(args.seed),
        "--device", args.device,
        "--speed", str(args.speed),
        "--replay_fps", str(args.replay_fps),
        "--initial_hold_seconds", str(args.initial_hold_seconds),
        "--hold_seconds", str(args.hold_seconds),
        "--_mode", mode,
        "--_trajectory_file", str(trajectory_file),
        "--observation_corruption" if args.observation_corruption else "--no-observation_corruption",
        "--geometric_teacher_residual" if args.geometric_teacher_residual else "--no-geometric_teacher_residual",
        "--teacher_observation_noise_mm", str(args.teacher_observation_noise_mm),
        "--teacher_tilt_noise_deg", str(args.teacher_tilt_noise_deg),
        "--teacher_alignment_gate_mm", str(args.teacher_alignment_gate_mm),
        "--hole_xy_bias_std_mm", str(args.hole_xy_bias_std_mm),
        "--action_noise_std", str(args.action_noise_std),
        "--action_gain_noise_std_pct", str(args.action_gain_noise_std_pct),
        "--reset_warmup_steps", str(args.reset_warmup_steps),
    ]
    if args.episode_length_s is not None:
        command.extend(("--episode_length_s", str(args.episode_length_s)))
    if args.nullspace_stiffness is not None:
        command.extend(("--nullspace_stiffness", str(args.nullspace_stiffness)))
    if args.barrier_mm is not None:
        command.extend(("--barrier_mm", str(args.barrier_mm)))
    if args.show_table:
        command.append("--show_table")
    if args.show_fixture_plate:
        command.append("--show_fixture_plate")
    else:
        command.append("--no-show_fixture_plate")
    if args.screenshot_dir is not None:
        command.extend(("--screenshot_dir", str(args.screenshot_dir)))
    if args.allow_stage_mismatch:
        command.append("--allow_stage_mismatch")
    if args.task is not None:
        command.extend(("--task", args.task))
    if args.fixed_target_hole_id is not None:
        command.extend(("--fixed_target_hole_id", str(args.fixed_target_hole_id)))
    return command


def run_orchestrator(args: argparse.Namespace) -> None:
    with tempfile.TemporaryDirectory(prefix="peg_visualize_", dir="/tmp") as temp_dir:
        trajectory_file = Path(temp_dir) / "verified_trajectory.pt"
        print("[VIS] Phase 1/2: capturing strict-success trajectory headlessly", flush=True)
        subprocess.run(worker_command(args, "capture", trajectory_file), check=True, cwd=PROJECT_ROOT)
        status_file = trajectory_file.with_suffix(".json")
        if not status_file.is_file():
            raise RuntimeError("Headless capture exited without writing a validation status file")
        status = json.loads(status_file.read_text(encoding="utf-8"))
        if status.get("captured") != status.get("expected") or not trajectory_file.is_file():
            raise RuntimeError(
                f"Headless capture produced {status.get('captured', 0)}/"
                f"{status.get('expected', args.episodes)} strict successes. "
                "GUI replay was cancelled."
            )
        if args.trajectory_output is not None:
            args.trajectory_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(trajectory_file, args.trajectory_output)
            shutil.copy2(status_file, args.trajectory_output.with_suffix(".json"))
            print(
                f"[VIS] Persistent trajectory/protocol record: {args.trajectory_output}",
                flush=True,
            )
        print("[VIS] Headless Isaac process exited cleanly", flush=True)
        print("[VIS] Phase 2/2: opening a fresh Isaac GUI for replay", flush=True)
        subprocess.run(worker_command(args, "replay", trajectory_file), check=True, cwd=PROJECT_ROOT)


def run_capture(args: argparse.Namespace) -> None:
    """Run policy physics without a viewport and persist only verified successes."""
    task = args.task or CURRICULUM_TASKS[args.stage]
    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher(headless=True, device=args.device).app
    env = None
    try:
        import gymnasium as gym
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg
        from rsl_rl.runners import OnPolicyRunner
        from rsl_rl.runners import on_policy_runner as runner_module

        import tasks.franka_peg_in_hole.config.franka  # noqa: F401
        from tasks.franka_peg_in_hole.mdp import geometry as geo

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        num_envs = max(args.capture_num_envs, args.episodes)
        env_cfg = parse_env_cfg(task, device=args.device, num_envs=num_envs)
        env_cfg.seed = args.seed
        env_cfg.scene.num_envs = num_envs
        env_cfg.observations.policy.enable_corruption = args.observation_corruption
        apply_verified_physics(env_cfg)
        if args.episode_length_s is not None:
            env_cfg.episode_length_s = args.episode_length_s
        if args.nullspace_stiffness is not None:
            env_cfg.actions.arm_action.controller_cfg.nullspace_stiffness = (
                args.nullspace_stiffness
            )
        if args.geometric_teacher_residual:
            apply_teacher_observation_noise(
                env_cfg,
                args.teacher_observation_noise_mm,
                args.teacher_tilt_noise_deg,
            )

        success_params = dict(env_cfg.terminations.success.params)
        radial_tol = float(success_params["radial_tol"])
        depth_required = float(success_params["depth_required"])
        max_depth_allowed = float(success_params["max_depth"])
        tilt_tol = float(success_params["tilt_tol"])
        print("\n" + "=" * 72)
        print("  Headless policy trajectory capture")
        print(f"  Protocol   : {args.protocol}")
        print(f"  Task       : {task}")
        print(f"  Checkpoint : {args.checkpoint}")
        print(f"  Environments: {num_envs} (matching batched evaluation)")
        print(
            "  Disturbance: "
            f"hole_bias={args.hole_xy_bias_std_mm:g}mm, "
            f"action_noise={args.action_noise_std:g}, "
            f"gain_noise={args.action_gain_noise_std_pct:g}%, delay=0"
        )
        print(
            "  Controller : "
            f"episode={env_cfg.episode_length_s:g}s, "
            f"teacher_gate={args.teacher_alignment_gate_mm:g}mm, "
            f"nullspace_K={env_cfg.actions.arm_action.controller_cfg.nullspace_stiffness:g}, "
            f"barrier={args.barrier_mm if args.barrier_mm is not None else 'off'}"
        )
        print(
            f"  Success    : radial<={radial_tol*1000:.2f}mm, "
            f"depth=[{depth_required*1000:.2f}, {max_depth_allowed*1000:.2f}]mm, "
            f"tilt<={tilt_tol*180/math.pi:.2f}deg"
        )
        print("=" * 72, flush=True)

        env = gym.make(task, cfg=env_cfg)
        action_perturbation = None
        hole_bias_perturbation = None
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
            action_perturbation = env
        if args.barrier_mm is not None:
            from scripts.depth_safety import wrap_with_predictive_depth_barrier

            env = wrap_with_predictive_depth_barrier(
                env,
                barrier_mm=args.barrier_mm,
                position_scale=env_cfg.actions.arm_action.position_scale,
                gym=gym,
                torch=torch,
                geometry=geo,
            )
        if args.hole_xy_bias_std_mm > 0.0:
            from scripts.observation_perturbation import wrap_with_hole_xy_bias

            env = wrap_with_hole_xy_bias(
                env,
                gym=gym,
                torch=torch,
                bias_std_mm=args.hole_xy_bias_std_mm,
            )
            hole_bias_perturbation = env
        uenv = env.unwrapped
        if args.fixed_target_hole_id is not None:
            uenv._fixed_target_hole_id = args.fixed_target_hole_id
            print(f"[CAPTURE] Fixed target hole id: {args.fixed_target_hole_id}", flush=True)
        uenv._capture_visualization_success = True
        robot = uenv.scene["robot"]
        sensor = uenv.scene.sensors.get("peg_contact")
        wrapped_env = RslRlVecEnvWrapper(env)
        agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
        if args.geometric_teacher_residual:
            from scripts.residual_policy import GeometricTeacherResidualActorCritic

            runner_module.GeometricTeacherResidualActorCritic = GeometricTeacherResidualActorCritic
            GeometricTeacherResidualActorCritic.default_teacher_alignment_gate_mm = float(
                args.teacher_alignment_gate_mm
            )
            agent_cfg.policy.class_name = "GeometricTeacherResidualActorCritic"
        with tempfile.TemporaryDirectory(prefix="peg_capture_log_", dir="/tmp") as log_dir:
            runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=log_dir, device=args.device)
            runner.load(str(args.checkpoint), load_optimizer=False)
            policy = runner.get_inference_policy(device=args.device)

            # Runner construction performs a reset. Re-seed afterwards so this
            # path is reproducible and agrees with deterministic evaluation.
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            wrapped_env.seed(args.seed)

            obs_dict, _ = env.reset()
            if (
                args.reset_warmup_steps > 0
                and not action_perturbation_enabled
                and getattr(env_cfg.events, "select_target_hole", None) is None
            ):
                zero_action = torch.zeros(
                    num_envs,
                    env.action_space.shape[-1],
                    device=uenv.device,
                )
                for _ in range(args.reset_warmup_steps):
                    obs_dict, _, _, _, _ = env.step(zero_action)
            history = [robot.data.joint_pos.detach().cpu().clone()]
            fixture_states = {
                name: asset.data.root_state_w.detach().cpu().clone()
                for name, asset in uenv.scene.rigid_objects.items()
            }
            max_contact = torch.zeros(num_envs, device=uenv.device)
            peg_tip, _ = geo.get_peg_tip(uenv)
            hole_pos, _ = geo.get_hole_center(uenv)
            previous_depth = geo.get_insertion_depth(peg_tip, hole_pos)
            min_depth = previous_depth.clone()
            max_depth = previous_depth.clone()
            max_z_step = torch.zeros(num_envs, device=uenv.device)
            already_captured = torch.zeros(num_envs, dtype=torch.bool, device=uenv.device)
            captured = []
            uenv._visualize_success_joint_pos = None
            uenv._visualize_success_metrics = None
            print(
                f"[CAPTURE] Searching {num_envs} evaluation-equivalent environments "
                f"for {args.episodes} strict success(es)",
                flush=True,
            )

            for step in range(uenv.max_episode_length):
                target_hole_ids = getattr(uenv, "_target_hole_id", None)
                target_hole_ids_before_step = (
                    target_hole_ids.detach().clone() if target_hole_ids is not None else None
                )
                with torch.inference_mode():
                    action = policy(obs_dict["policy"])
                obs_dict, _, terminated, _, _ = env.step(action)
                if sensor is not None:
                    forces = torch.linalg.vector_norm(sensor.data.net_forces_w, dim=-1).amax(dim=-1)
                    max_contact = torch.maximum(max_contact, forces)
                terminal_depth = uenv._termination_depth
                min_depth = torch.minimum(min_depth, terminal_depth)
                max_depth = torch.maximum(max_depth, terminal_depth)
                max_z_step = torch.maximum(max_z_step, (terminal_depth - previous_depth).abs())
                previous_depth = terminal_depth.clone()

                success_mask = uenv._success_termination_mask.bool()
                new_success_ids = torch.nonzero(success_mask & ~already_captured).squeeze(-1)
                if new_success_ids.numel() > 0:
                    cached_joint_pos = uenv._visualize_success_joint_pos
                    cached_metrics = uenv._visualize_success_metrics
                    if cached_joint_pos is None or cached_metrics is None:
                        raise RuntimeError("Success terminated without a cached pre-reset state")
                    radial, depth, tilt = cached_metrics
                    for env_id_tensor in new_success_ids:
                        env_id = int(env_id_tensor.item())
                        if len(captured) >= args.episodes:
                            break
                        trajectory = torch.stack(
                            [frame[env_id] for frame in history]
                            + [cached_joint_pos[env_id].detach().cpu().clone()]
                        )
                        replay_fixtures = {}
                        for name, states in fixture_states.items():
                            state = states[env_id].clone()
                            # Rigid-object state tensors for these replicated
                            # fixture prims are already expressed in their
                            # environment-local frame.  Store them unchanged:
                            # adding or subtracting scene.env_origins would
                            # inject the parallel capture grid displacement
                            # into the single-environment GUI replay.
                            replay_fixtures[name] = state
                        result = {
                            "success": True,
                            "steps": len(trajectory) - 1,
                            "source_env": env_id,
                            "radial": radial[env_id].item(),
                            "depth": depth[env_id].item(),
                            "tilt": tilt[env_id].item(),
                            "max_contact": max_contact[env_id].item(),
                            "min_depth": min_depth[env_id].item(),
                            "max_depth": max_depth[env_id].item(),
                            "max_abs_z_step": max_z_step[env_id].item(),
                            "target_hole_id": (
                                int(target_hole_ids_before_step[env_id].item())
                                if target_hole_ids_before_step is not None else None
                            ),
                            "perturbation": {
                                "hole_xy_bias_mm": (
                                    (
                                        hole_bias_perturbation._last_policy_bias[env_id]
                                        .detach().cpu() * 1000.0
                                    ).tolist()
                                    if hole_bias_perturbation is not None
                                    and hole_bias_perturbation._last_policy_bias is not None
                                    else [0.0, 0.0]
                                ),
                                "action_gain": (
                                    float(
                                        action_perturbation._last_applied_gain[env_id].item()
                                    )
                                    if action_perturbation is not None
                                    and action_perturbation._last_applied_gain is not None
                                    else 1.0
                                ),
                                "terminal_action_noise": (
                                    action_perturbation._last_action_noise[env_id]
                                    .detach().cpu().tolist()
                                    if action_perturbation is not None
                                    and action_perturbation._last_action_noise is not None
                                    else [0.0] * 6
                                ),
                            },
                        }
                        captured.append({
                            "trajectory": trajectory,
                            "fixture_states": replay_fixtures,
                            "result": result,
                        })
                        print(
                            f"[CAPTURE] SUCCESS {len(captured)}/{args.episodes}: env={env_id} "
                            f"steps={result['steps']} radial={result['radial']*1000:.2f}mm "
                            f"depth={result['depth']*1000:.2f}mm "
                            f"tilt={result['tilt']*180/math.pi:.2f}deg "
                            f"peak_force={result['max_contact']:.2f}N "
                            f"max_dz={result['max_abs_z_step']*1000:.2f}mm",
                            flush=True,
                        )
                    already_captured[new_success_ids] = True
                if len(captured) >= args.episodes:
                    break
                history.append(robot.data.joint_pos.detach().cpu().clone())

            payload = {
                "task": task,
                "seed": args.seed,
                "policy_dt": env_cfg.sim.dt * env_cfg.decimation,
                "fixture_state_frame": "env_local",
                "protocol": {
                    "name": args.protocol,
                    "episode_length_s": float(env_cfg.episode_length_s),
                    "teacher_alignment_gate_mm": args.teacher_alignment_gate_mm,
                    "teacher_observation_noise_mm": args.teacher_observation_noise_mm,
                    "teacher_tilt_noise_deg": args.teacher_tilt_noise_deg,
                    "hole_xy_bias_std_mm": args.hole_xy_bias_std_mm,
                    "action_noise_std": args.action_noise_std,
                    "action_gain_noise_std_pct": args.action_gain_noise_std_pct,
                    "action_delay_steps": 0,
                    "reset_warmup_steps": args.reset_warmup_steps,
                    "nullspace_stiffness": float(
                        env_cfg.actions.arm_action.controller_cfg.nullspace_stiffness
                    ),
                    "barrier_mm": args.barrier_mm,
                },
                "success_params": {
                    "radial_tol": radial_tol,
                    "depth_required": depth_required,
                    "max_depth": max_depth_allowed,
                    "tilt_tol": tilt_tol,
                },
                "episodes": captured,
            }
            torch.save(payload, args._trajectory_file)
            args._trajectory_file.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "expected": args.episodes,
                        "captured": len(captured),
                        "protocol": payload["protocol"],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            if len(captured) != args.episodes:
                raise RuntimeError(
                    f"Headless capture produced {len(captured)}/{args.episodes} strict successes; "
                    "GUI replay was cancelled. Verify this checkpoint with "
                    "scripts/eval_checkpoint_simple.py under the same protocol."
                )
            print(f"[CAPTURE] Saved {len(captured)} verified trajectory/trajectories", flush=True)
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


def run_replay(args: argparse.Namespace) -> None:
    """Open a fresh GUI and display recorded states without running the policy."""
    status_file = args._trajectory_file.with_suffix(".json")
    if not status_file.is_file():
        raise RuntimeError("Replay refused: capture validation status is missing")
    status = json.loads(status_file.read_text(encoding="utf-8"))
    if status.get("captured", 0) <= 0 or status.get("captured") != status.get("expected"):
        raise RuntimeError(
            f"Replay refused: capture contains {status.get('captured', 0)}/"
            f"{status.get('expected', 0)} strict successes"
        )
    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher(headless=False, device=args.device).app
    env = None
    try:
        import gymnasium as gym
        import omni.timeline
        import torch
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        import tasks.franka_peg_in_hole.config.franka  # noqa: F401

        payload = torch.load(args._trajectory_file, map_location="cpu", weights_only=False)
        task = payload["task"]
        protocol = payload.get("protocol", {"name": "legacy/unrecorded"})
        print(
            "[REPLAY] Captured protocol: "
            + json.dumps(protocol, ensure_ascii=False, sort_keys=True),
            flush=True,
        )
        env_cfg = parse_env_cfg(task, device=args.device, num_envs=1)
        env_cfg.seed = int(payload["seed"])
        env_cfg.scene.num_envs = 1
        env_cfg.observations.policy.enable_corruption = False
        apply_verified_physics(env_cfg)
        if args.show_table:
            # Pose6D training deliberately disables SeattleLabTable because
            # its collision mesh covered the robot base.  For GUI replay only,
            # restore the original visual asset; the captured policy physics
            # remains the four-fixture setup and is never re-evaluated here.
            from isaaclab.assets import AssetBaseCfg
            from isaaclab.sim.spawners.from_files import UsdFileCfg
            from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

            env_cfg.scene.table = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                spawn=UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
                ),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(0.525, 0.0, 0.0),
                    rot=(0.70711, 0.0, 0.0, 0.70711),
                ),
            )
            print("[REPLAY] SeattleLabTable visual asset enabled", flush=True)
        env_cfg.viewer.eye = (0.75, 0.45, 0.70)
        env_cfg.viewer.lookat = (0.2494, 0.0013, 0.36)

        env = gym.make(task, cfg=env_cfg)
        uenv = env.unwrapped
        robot = uenv.scene["robot"]
        table_extra = uenv.scene.extras.get("table") if args.show_table else None
        table_pose = None
        if table_extra is not None:
            table_pose = table_extra.get_world_poses()
            print("[REPLAY] Table is an AssetBase extra; preserving its XForm pose", flush=True)

        # The training fixture is intentionally collision-safe: its top plate
        # is represented by four cuboids and ``hole_board`` is only an
        # invisible reference marker.  That is physically correct, but it is
        # easy to mistake for a missing plate in a GUI replay when a timeline
        # refresh temporarily drops the rigid-object visuals.  Add a
        # collision-free USD mesh for the replay only.  It has a square outer
        # boundary and a genuinely circular opening, and is repositioned from
        # the recorded fixture states on every frame.
        visual_plate = None
        if args.show_fixture_plate:
            try:
                from pxr import Gf, UsdGeom
                import omni.usd

                stage = omni.usd.get_context().get_stage()
                # Keep replay-only geometry outside the cloned environment
                # namespace. Prims authored below env_0 can be discarded when
                # Isaac/Fabric refreshes that cloned scene during playback.
                UsdGeom.Xform.Define(stage, "/World/ReplayVisuals")
                plate_path = "/World/ReplayVisuals/FixturePlate"
                mesh = UsdGeom.Mesh.Define(stage, plate_path)
                mesh.CreateSubdivisionSchemeAttr().Set("none")
                mesh.CreateOrientationAttr().Set("rightHanded")

                segments = 64
                half_outer = 0.150
                inner_radius = 0.014
                half_thickness = 0.010
                points = []
                for i in range(segments):
                    angle = 2.0 * math.pi * i / segments
                    c, s = math.cos(angle), math.sin(angle)
                    boundary_radius = half_outer / max(abs(c), abs(s))
                    points.extend((
                        (boundary_radius * c, boundary_radius * s, half_thickness),
                        (inner_radius * c, inner_radius * s, half_thickness),
                        (boundary_radius * c, boundary_radius * s, -half_thickness),
                        (inner_radius * c, inner_radius * s, -half_thickness),
                    ))
                face_counts = []
                face_indices = []
                for i in range(segments):
                    j = (i + 1) % segments
                    bi, ii, bo, io = 4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3
                    bj, ij, bo_j, io_j = 4 * j, 4 * j + 1, 4 * j + 2, 4 * j + 3
                    # Top, bottom, outer wall, and circular inner wall.
                    for face in (
                        (bi, bj, ij, ii),
                        (bo, io, io_j, bo_j),
                        (bi, bo, bo_j, bj),
                        (ii, ij, io_j, io),
                    ):
                        face_counts.append(4)
                        face_indices.extend(face)
                mesh.CreatePointsAttr().Set([Gf.Vec3f(*point) for point in points])
                mesh.CreateFaceVertexCountsAttr().Set(face_counts)
                mesh.CreateFaceVertexIndicesAttr().Set(face_indices)
                mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(0.30, 0.34, 0.40)])
                mesh.CreateDisplayOpacityAttr().Set([1.0])
                xform = UsdGeom.Xformable(mesh.GetPrim())
                plate_translate = xform.AddTranslateOp()

                def update_visual_plate(fixture_states) -> None:
                    left = fixture_states.get("fixture_left")
                    right = fixture_states.get("fixture_right")
                    front = fixture_states.get("fixture_front")
                    back = fixture_states.get("fixture_back")
                    if left is None or right is None or front is None or back is None:
                        return
                    center = 0.25 * (left[:3] + right[:3] + front[:3] + back[:3])
                    # Lift by a fraction of a millimetre to avoid z-fighting
                    # with the collision fixture's own visual materials.
                    plate_translate.Set(
                        Gf.Vec3d(float(center[0]), float(center[1]), float(center[2]) + 0.0003)
                    )

                visual_plate = update_visual_plate
                print(
                    "[REPLAY] Collision-free circular-hole fixture plate enabled",
                    flush=True,
                )
            except Exception as exc:
                print(f"[REPLAY] Fixture plate visual skipped: {exc}", flush=True)
        visual_array = None
        if args.show_fixture_plate:
            try:
                from pxr import Gf, UsdGeom
                import omni.usd

                has_array_walls = any(
                    name.startswith("array_hole_0_wall_")
                    for name in payload["episodes"][0]["fixture_states"]
                )
                if has_array_walls:
                    stage = omni.usd.get_context().get_stage()
                    UsdGeom.Xform.Define(stage, "/World/ReplayVisuals")
                    array_path = "/World/ReplayVisuals/SixHoleSleeves"
                    mesh = UsdGeom.Mesh.Define(stage, array_path)
                    mesh.CreateSubdivisionSchemeAttr().Set("none")
                    mesh.CreateOrientationAttr().Set("rightHanded")

                    segments = 32
                    outer_radius = 0.012
                    inner_radius = 0.0102
                    half_height = 0.020
                    points = []
                    face_counts = []
                    face_indices = []
                    for hole_index in range(6):
                        column = hole_index % 3
                        row = hole_index // 3
                        center_x = (column - 1) * 0.030
                        center_y = (0.5 - row) * 0.030
                        base = len(points)
                        for segment in range(segments):
                            angle = 2.0 * math.pi * segment / segments
                            c, s = math.cos(angle), math.sin(angle)
                            points.extend(
                                (
                                    (center_x + outer_radius * c, center_y + outer_radius * s, half_height),
                                    (center_x + inner_radius * c, center_y + inner_radius * s, half_height),
                                    (center_x + outer_radius * c, center_y + outer_radius * s, -half_height),
                                    (center_x + inner_radius * c, center_y + inner_radius * s, -half_height),
                                )
                            )
                        for segment in range(segments):
                            next_segment = (segment + 1) % segments
                            bi, ii, bo, io = base + 4 * segment, base + 4 * segment + 1, base + 4 * segment + 2, base + 4 * segment + 3
                            bj, ij, bo_j, io_j = base + 4 * next_segment, base + 4 * next_segment + 1, base + 4 * next_segment + 2, base + 4 * next_segment + 3
                            for face in (
                                (bi, bj, ij, ii),
                                (bo, io, io_j, bo_j),
                                (bi, bo, bo_j, bj),
                                (ii, ij, io_j, io),
                            ):
                                face_counts.append(4)
                                face_indices.extend(face)
                    mesh.CreatePointsAttr().Set([Gf.Vec3f(*point) for point in points])
                    mesh.CreateFaceVertexCountsAttr().Set(face_counts)
                    mesh.CreateFaceVertexIndicesAttr().Set(face_indices)
                    mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(0.95, 0.70, 0.08)])
                    mesh.CreateDisplayOpacityAttr().Set([1.0])
                    array_translate = UsdGeom.Xformable(mesh.GetPrim()).AddTranslateOp()
                    outer_translates = []
                    inner_translates = []
                    for hole_index in range(6):
                        outer = UsdGeom.Cylinder.Define(
                            stage, f"/World/ReplayVisuals/HoleOuter{hole_index}"
                        )
                        # Slightly enlarge this color shell relative to the
                        # combined sleeve mesh.  Equal, coplanar surfaces
                        # z-fight and can hide the red target-hole highlight.
                        outer.CreateRadiusAttr().Set(0.0122)
                        outer.CreateHeightAttr().Set(0.0404)
                        outer_color = outer.CreateDisplayColorAttr()
                        outer_color.Set([Gf.Vec3f(0.95, 0.70, 0.08)])
                        outer_translates.append(UsdGeom.Xformable(outer.GetPrim()).AddTranslateOp())
                        inner = UsdGeom.Cylinder.Define(
                            stage, f"/World/ReplayVisuals/HoleInner{hole_index}"
                        )
                        inner.CreateRadiusAttr().Set(0.0102)
                        inner.CreateHeightAttr().Set(0.041)
                        inner.CreateDisplayColorAttr().Set([Gf.Vec3f(0.08, 0.08, 0.08)])
                        inner_translates.append(UsdGeom.Xformable(inner.GetPrim()).AddTranslateOp())

                    # A dedicated, slightly larger red sleeve is moved onto
                    # the selected target.  Recoloring an already-rendered
                    # cylinder is not reliably propagated by Hydra in Isaac
                    # Sim 4.5, while a fixed-color prim remains deterministic.
                    target_outer = UsdGeom.Cylinder.Define(
                        stage, "/World/ReplayVisuals/TargetHoleOuter"
                    )
                    target_outer.CreateRadiusAttr().Set(0.0145)
                    target_outer.CreateHeightAttr().Set(0.044)
                    target_outer.CreateDisplayColorAttr().Set([Gf.Vec3f(0.92, 0.15, 0.10)])
                    target_outer_translate = UsdGeom.Xformable(
                        target_outer.GetPrim()
                    ).AddTranslateOp()
                    target_inner = UsdGeom.Cylinder.Define(
                        stage, "/World/ReplayVisuals/TargetHoleInner"
                    )
                    target_inner.CreateRadiusAttr().Set(0.0102)
                    target_inner.CreateHeightAttr().Set(0.0444)
                    target_inner.CreateDisplayColorAttr().Set([Gf.Vec3f(0.08, 0.08, 0.08)])
                    target_inner_translate = UsdGeom.Xformable(
                        target_inner.GetPrim()
                    ).AddTranslateOp()
                    hidden_target_position = Gf.Vec3d(0.0, 0.0, -10.0)
                    target_outer_translate.Set(hidden_target_position)
                    target_inner_translate.Set(hidden_target_position)

                    def update_visual_array(fixture_states, target_hole_id=None) -> None:
                        hole_centers = []
                        for hole_index in range(6):
                            wall_positions = [
                                state[:3]
                                for name, state in fixture_states.items()
                                if name.startswith(f"array_hole_{hole_index}_wall_")
                            ]
                            if wall_positions:
                                hole_centers.append(torch.stack(wall_positions).mean(dim=0))
                        if hole_centers:
                            center = torch.stack(hole_centers).mean(dim=0)
                            array_translate.Set(
                                Gf.Vec3d(float(center[0]), float(center[1]), float(center[2]))
                            )
                            for hole_index, hole_center in enumerate(hole_centers):
                                position = Gf.Vec3d(
                                    float(hole_center[0]),
                                    float(hole_center[1]),
                                    float(hole_center[2]),
                                )
                                outer_translates[hole_index].Set(position)
                                inner_translates[hole_index].Set(position)
                            if (
                                target_hole_id is not None
                                and 0 <= int(target_hole_id) < len(hole_centers)
                            ):
                                target_center = hole_centers[int(target_hole_id)]
                                target_position = Gf.Vec3d(
                                    float(target_center[0]),
                                    float(target_center[1]),
                                    float(target_center[2]),
                                )
                                target_outer_translate.Set(target_position)
                                target_inner_translate.Set(target_position)
                            else:
                                target_outer_translate.Set(hidden_target_position)
                                target_inner_translate.Set(hidden_target_position)

                    visual_array = update_visual_array
                    print("[REPLAY] Persistent six-hole visual sleeves enabled", flush=True)
            except Exception as exc:
                print(f"[REPLAY] Six-hole visual sleeves skipped: {exc}", flush=True)
        finger_ids, finger_names = robot.find_joints("panda_finger_joint.*", preserve_order=True)
        if len(finger_ids) != 2:
            raise RuntimeError(f"Expected two Franka finger joints, found {finger_names}")

        timeline = omni.timeline.get_timeline_interface()
        env0 = torch.tensor([0], device=uenv.device)
        zero_vel = torch.zeros((1, robot.num_joints), device=uenv.device)
        closed = torch.zeros((1, len(finger_ids)), device=uenv.device)
        policy_dt = float(payload["policy_dt"])
        interpolation_steps = max(1, round(args.replay_fps * policy_dt / args.speed))

        def normalize_fixture_states(episode) -> dict[str, torch.Tensor]:
            """Return fixture poses in replay env-0 coordinates.

            Trajectories captured before ``fixture_state_frame=env_local`` may
            contain the source environment's grid displacement.  Recover the
            local frame from the exact 2 m environment spacing so existing
            saved visualizations remain usable.
            """
            fixture_states = {
                name: state.detach().cpu().clone()
                for name, state in episode["fixture_states"].items()
            }
            if payload.get("fixture_state_frame") == "env_local":
                return fixture_states

            reference = fixture_states.get("hole_board")
            if reference is None:
                return fixture_states
            spacing = float(env_cfg.scene.env_spacing)
            grid_offset = torch.round(reference[:2] / spacing) * spacing
            if torch.any(grid_offset.abs() > 0.5 * spacing):
                for state in fixture_states.values():
                    state[:2] -= grid_offset
                print(
                    "[REPLAY] Legacy fixture coordinates normalized by "
                    f"({float(grid_offset[0]):.3f}, {float(grid_offset[1]):.3f}) m",
                    flush=True,
                )
            return fixture_states

        def present(sample, fixture_states, delay: float, target_hole_id=None) -> None:
            # Keep the static fixture/table slabs pinned while the GUI timeline
            # is running.  Isaac Sim 4.5 may refresh kinematic Fabric views on
            # a timeline tick; writing them only once before timeline.play()
            # can make the desktop appear to vanish during replay.
            for name, root_state in fixture_states.items():
                # The six-hole wall segments are static. Rewriting all 216
                # objects every frame can make their Fabric visuals disappear
                # in Isaac Sim 4.5, so they are written only before playback.
                if name.startswith("array_hole_"):
                    continue
                if name in uenv.scene.rigid_objects:
                    uenv.scene.rigid_objects[name].write_root_state_to_sim(
                        root_state.to(uenv.device).unsqueeze(0), env_ids=env0
                    )
            if table_extra is not None and table_pose is not None:
                table_extra.set_world_poses(
                    table_pose[0].to(uenv.device), table_pose[1].to(uenv.device)
                )
            joint_pos = sample.to(uenv.device).unsqueeze(0).clone()
            joint_pos[:, finger_ids] = 0.0
            robot.set_joint_position_target(closed, joint_ids=finger_ids, env_ids=env0)
            robot.write_joint_state_to_sim(joint_pos, zero_vel, env_ids=env0)
            uenv.sim.forward()
            simulation_app.update()
            # Author replay-only visuals after both the simulator/Fabric
            # forward pass and the live application update.  The latter is
            # where Isaac Sim refreshes cloned transforms for the viewport.
            if visual_plate is not None:
                visual_plate(fixture_states)
            if visual_array is not None:
                visual_array(fixture_states, target_hole_id)
            uenv.sim.render()
            time.sleep(max(0.0, delay))

        def capture_view(name: str) -> None:
            if args.screenshot_dir is None:
                return
            try:
                import omni.kit.viewport.utility as viewport_utility

                viewport = viewport_utility.get_active_viewport()
                if viewport is None:
                    return
                args.screenshot_dir.mkdir(parents=True, exist_ok=True)
                screenshot_path = args.screenshot_dir / f"{name}.png"
                capture = viewport_utility.capture_viewport_to_file(
                    viewport, str(screenshot_path)
                )
                # Screenshot capture is asynchronous.  Drive Kit's event loop
                # until the result is complete so the final frame is not lost
                # when the replay worker exits immediately afterwards.
                capture_future = asyncio.ensure_future(capture.wait_for_result())
                capture_deadline = time.monotonic() + 15.0
                while time.monotonic() < capture_deadline:
                    simulation_app.update()
                    if capture_future.done():
                        capture_future.result()
                        break
                else:
                    capture_future.cancel()
                    raise TimeoutError(f"Timed out writing screenshot: {screenshot_path}")
                print(f"[REPLAY] Screenshot {screenshot_path}", flush=True)
            except Exception as exc:
                print(f"[REPLAY] Screenshot skipped: {exc}", flush=True)

        print("\n[REPLAY] GUI contains recorded states only; no policy or control physics is running", flush=True)
        for index, episode in enumerate(payload["episodes"]):
            if not simulation_app.is_running():
                break
            env.reset()
            timeline.pause()
            replay_fixtures = normalize_fixture_states(episode)
            if index == 0:
                print(
                    "[REPLAY] Restoring static rigid objects: "
                    + ", ".join(sorted(replay_fixtures)),
                    flush=True,
                )
            for name, root_state in replay_fixtures.items():
                if name in uenv.scene.rigid_objects:
                    uenv.scene.rigid_objects[name].write_root_state_to_sim(
                        root_state.to(uenv.device).unsqueeze(0), env_ids=env0
                    )

            # Isaac Sim 4.5 only propagates articulation tensor writes to the
            # viewport's Fabric transforms on a live timeline update. Replay
            # is safe because every displayed frame overwrites the state.
            timeline.play()

            trajectory = episode["trajectory"]
            result = episode["result"]
            display_frames = 1 + (len(trajectory) - 1) * interpolation_steps
            print(
                f"[REPLAY] Episode {index + 1}/{len(payload['episodes'])}: "
                f"{display_frames} frames at {args.speed:.2f}x / {args.replay_fps:.0f} FPS; "
                f"verified depth={result['depth']*1000:.2f}mm; "
                f"target_hole={result.get('target_hole_id', 'legacy')}; "
                f"perturbation={result.get('perturbation', 'legacy/unrecorded')}",
                flush=True,
            )
            target_hole_id = result.get("target_hole_id")
            present(
                trajectory[0], replay_fixtures,
                1.0 / args.replay_fps, target_hole_id,
            )
            capture_view(f"episode_{index + 1}_initial")
            if args.initial_hold_seconds > 0.0:
                print(
                    f"[REPLAY] Holding initial pose for "
                    f"{args.initial_hold_seconds:.1f}s",
                    flush=True,
                )
                initial_deadline = time.time() + args.initial_hold_seconds
                while simulation_app.is_running() and time.time() < initial_deadline:
                    present(
                        trajectory[0], replay_fixtures,
                        1.0 / args.replay_fps, target_hole_id,
                    )
            for previous, current in zip(trajectory[:-1], trajectory[1:]):
                if not simulation_app.is_running():
                    break
                for substep in range(1, interpolation_steps + 1):
                    alpha = substep / interpolation_steps
                    present(
                        previous + alpha * (current - previous),
                        replay_fixtures,
                        1.0 / args.replay_fps,
                        target_hole_id,
                    )

            deadline = time.time() + args.hold_seconds
            while simulation_app.is_running() and time.time() < deadline:
                present(
                    trajectory[-1], replay_fixtures,
                    1.0 / args.replay_fps, target_hole_id,
                )
            capture_view(f"episode_{index + 1}_final")
            timeline.pause()

        print(f"\nVisualization summary: replayed {len(payload['episodes'])} verified successes")
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


def main() -> None:
    args = parse_args()
    if args._mode == "auto":
        run_orchestrator(args)
    elif args._mode == "capture":
        run_capture(args)
    else:
        run_replay(args)


if __name__ == "__main__":
    main()
