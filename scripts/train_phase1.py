#!/usr/bin/env python3
"""Train the verified Pose6D Franka peg-in-hole curriculum with RSL-RL PPO."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import io
import json
import os
import statistics
import sys
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

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
    "hole20_reward": "Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom20mm-MountStableReward-v0",
    "hole20_reward_mix": "Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom20mm-MountStableRewardMix-v0",
    "hole20_reward_edge": "Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom20mm-MountStableRewardEdge-v0",
    "hole20_reward_edge_anchor": "Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom20mm-MountStableRewardEdgeAnchor-v0",
    "hole20_online_ik": "Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom20mm-MountStableOnlineIK-v0",
    "hole20_online_ik_canonical": "Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom20mm-MountStableOnlineIKCanonical-v0",
    "hole20_online_ik_offset5": "Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom20mm-MountStableOnlineIKOffset5Residual-v0",
    "hole20_mount_stable_ik": "Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom20mm-MountStableIK-v0",
    "hole30": "Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom30mm-v0",
    "hole50": "Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom50mm-v0",
    "hole25": "Isaac-PegInHole-Franka-OSC-Pose6D-Hole25mm-v0",
    "hole23": "Isaac-PegInHole-Franka-OSC-Pose6D-Hole23mm-v0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the verified Franka Pose6D peg-in-hole PPO curriculum",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage", choices=tuple(CURRICULUM_TASKS), default="baseline",
        help="Curriculum stage; ignored when --task is supplied",
    )
    parser.add_argument(
        "--task", default=None,
        help="Explicit Gym task ID (advanced override)",
    )
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument(
        "--episode_length_s", type=float, default=None,
        help="Override environment episode length; 20s gives a 600-step budget at 30Hz",
    )
    parser.add_argument(
        "--num_steps_per_env", type=int, default=None,
        help="Override PPO rollout length; use >=64 for full insertion credit assignment",
    )
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Checkpoint to resume or transfer from the preceding curriculum stage",
    )
    parser.add_argument(
        "--load_mode", choices=("resume", "transfer"), default="transfer",
        help="Resume optimizer/iteration state or transfer policy weights into a new stage",
    )
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--log_root", type=Path, default=PROJECT_ROOT / "runs" / "ppo")
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_learning_epochs", type=int, default=None)
    parser.add_argument("--entropy_coef", type=float, default=None)
    parser.add_argument(
        "--init_noise_std", type=float, default=None,
        help="Override PPO exploration std; residual stress training can use 0.03",
    )
    parser.add_argument(
        "--nullspace_stiffness", type=float, default=None,
        help="Override OSC joint-position nullspace stiffness",
    )
    parser.add_argument(
        "--console", choices=("progress", "verbose", "quiet"), default="progress",
        help="Progress bar, original RSL-RL iteration report, or no per-iteration output",
    )
    parser.add_argument(
        "--headless", action=argparse.BooleanOptionalAction, default=True,
        help="Disable the GUI for training (recommended)",
    )
    parser.add_argument(
        "--residual_adapter", action=argparse.BooleanOptionalAction, default=False,
        help="Freeze the transferred actor and train only a bounded residual action head",
    )
    parser.add_argument(
        "--residual_weight_decay", type=float, default=0.0,
        help="L2/weight-decay regularization for residual transfer (0 disables it)",
    )
    parser.add_argument(
        "--residual_penalty_coef", type=float, default=0.0,
        help="Teacher-action residual MSE coefficient (enables ResidualPPO when > 0)",
    )
    parser.add_argument(
        "--residual_freeze_critic", action=argparse.BooleanOptionalAction, default=False,
        help="Freeze the transferred critic as well as the teacher actor",
    )
    parser.add_argument(
        "--geometric_teacher_residual", action="store_true",
        help="Train a bounded residual around the relative-observation geometric teacher",
    )
    parser.add_argument(
        "--teacher_observation_noise_mm", type=float, default=0.5,
        help="Uniform relative-position observation noise for geometric-teacher training",
    )
    parser.add_argument(
        "--teacher_tilt_noise_deg", type=float, default=0.25,
        help="Uniform peg-axis observation noise in degrees for geometric-teacher training",
    )
    parser.add_argument(
        "--hole_xy_bias_std_mm", type=float, default=0.0,
        help="Fixed per-episode Gaussian XY hole-position estimate bias",
    )
    parser.add_argument(
        "--action_noise_std", type=float, default=0.0,
        help="Zero-delay per-control-step normalized action noise for stress training",
    )
    parser.add_argument(
        "--action_gain_noise_std_pct", type=float, default=0.0,
        help="Zero-delay per-episode control gain noise percentage for stress training",
    )
    parser.add_argument(
        "--reset_warmup_steps", type=int, default=0,
        help="Zero-action OSC settling steps after reset; use the same value for teacher/PPO",
    )
    parser.add_argument(
        "--depth_safety_barrier_mm", type=float, default=None,
        help="Predictive insertion-depth barrier applied to the task-space Z action",
    )
    parser.add_argument("--insertion_gate_radial_mm", type=float, default=None)
    parser.add_argument("--insertion_gate_tilt_deg", type=float, default=None)
    parser.add_argument("--insertion_gate_hysteresis", type=float, default=4.0 / 3.0)
    parser.add_argument("--insertion_gate_start_depth_mm", type=float, default=0.0)
    parser.add_argument("--insertion_soft_gate_min_scale", type=float, default=None)
    parser.add_argument(
        "--teacher_alignment_gate_mm", type=float, default=1.0,
        help="Alignment gate used by the geometric teacher residual policy",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Quick integration test: 64 envs and 10 iterations unless explicitly overridden",
    )
    args = parser.parse_args()

    if args.num_envs <= 0:
        parser.error("--num_envs must be positive")
    if args.episode_length_s is not None and args.episode_length_s <= 0.0:
        parser.error("--episode_length_s must be positive")
    if args.max_iterations is not None and args.max_iterations <= 0:
        parser.error("--max_iterations must be positive")
    if args.num_steps_per_env is not None and args.num_steps_per_env <= 0:
        parser.error("--num_steps_per_env must be positive")
    if args.save_interval is not None and args.save_interval <= 0:
        parser.error("--save_interval must be positive")
    if args.learning_rate is not None and args.learning_rate <= 0.0:
        parser.error("--learning_rate must be positive")
    if args.num_learning_epochs is not None and args.num_learning_epochs <= 0:
        parser.error("--num_learning_epochs must be positive")
    if args.entropy_coef is not None and args.entropy_coef < 0.0:
        parser.error("--entropy_coef must be non-negative")
    if args.init_noise_std is not None and args.init_noise_std < 0.0:
        parser.error("--init_noise_std must be non-negative")
    if args.nullspace_stiffness is not None and args.nullspace_stiffness < 0.0:
        parser.error("--nullspace_stiffness must be non-negative")
    if args.residual_weight_decay < 0.0:
        parser.error("--residual_weight_decay must be non-negative")
    if args.residual_penalty_coef < 0.0:
        parser.error("--residual_penalty_coef must be non-negative")
    if args.teacher_observation_noise_mm < 0.0 or args.teacher_tilt_noise_deg < 0.0:
        parser.error("teacher observation noise values must be non-negative")
    if args.hole_xy_bias_std_mm < 0.0:
        parser.error("--hole_xy_bias_std_mm must be non-negative")
    if args.action_noise_std < 0.0 or args.action_gain_noise_std_pct < 0.0:
        parser.error("action perturbation values must be non-negative")
    if args.reset_warmup_steps < 0:
        parser.error("reset_warmup_steps must be non-negative")
    if args.teacher_alignment_gate_mm <= 0.0:
        parser.error("--teacher_alignment_gate_mm must be positive")
    if args.geometric_teacher_residual and args.residual_adapter:
        parser.error("geometric teacher residual cannot be combined with --residual_adapter")
    if args.geometric_teacher_residual and args.checkpoint is not None:
        parser.error("geometric teacher residual starts from its analytic teacher; omit --checkpoint")
    if args.depth_safety_barrier_mm is not None and args.depth_safety_barrier_mm <= 0.0:
        parser.error("--depth_safety_barrier_mm must be positive")
    gate_values = (args.insertion_gate_radial_mm, args.insertion_gate_tilt_deg)
    if (gate_values[0] is None) != (gate_values[1] is None):
        parser.error("both insertion gate thresholds must be supplied together")
    if any(value is not None and value <= 0.0 for value in gate_values):
        parser.error("insertion gate thresholds must be positive")
    if args.insertion_gate_hysteresis < 1.0:
        parser.error("--insertion_gate_hysteresis must be >= 1")
    if args.insertion_soft_gate_min_scale is not None:
        if args.insertion_gate_radial_mm is None:
            parser.error("soft gate requires insertion gate thresholds")
        if not 0.0 < args.insertion_soft_gate_min_scale <= 1.0:
            parser.error("--insertion_soft_gate_min_scale must be in (0, 1]")
    if args.smoke:
        # Preserve explicit non-default values while making the common smoke
        # command safe and quick.
        if args.num_envs == parser.get_default("num_envs"):
            args.num_envs = 64
        if args.max_iterations is None:
            args.max_iterations = 10
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.expanduser().resolve()
        if not args.checkpoint.is_file():
            parser.error(f"checkpoint does not exist: {args.checkpoint}")
    args.log_root = args.log_root.expanduser().resolve()
    return args


def apply_verified_physics(env_cfg) -> None:
    """Match the 128-env Pose6D Oracle acceptance physics exactly."""
    env_cfg.sim.dt = 1.0 / 120.0
    env_cfg.decimation = 4
    env_cfg.sim.render_interval = env_cfg.decimation
    env_cfg.sim.physx.enable_enhanced_determinism = True
    articulation = env_cfg.scene.robot.spawn.articulation_props
    articulation.solver_position_iteration_count = 16
    articulation.solver_velocity_iteration_count = 4


def apply_teacher_observation_noise(env_cfg, position_noise_mm: float, tilt_noise_deg: float) -> None:
    """Enable the bounded observation noise used by geometric-teacher training."""
    from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

    env_cfg.observations.policy.enable_corruption = True
    env_cfg.observations.policy.peg_to_hole_vec.noise = Unoise(
        n_min=-float(position_noise_mm) / 1000.0,
        n_max=float(position_noise_mm) / 1000.0,
    )
    tilt_noise_rad = float(tilt_noise_deg) * 3.141592653589793 / 180.0
    env_cfg.observations.policy.peg_tilt.noise = Unoise(
        n_min=-tilt_noise_rad,
        n_max=tilt_noise_rad,
    )


def install_console_logger(runner, mode: str, total_iterations: int, description: str):
    """Keep TensorBoard logging while replacing RSL-RL's verbose console report."""
    if mode == "verbose":
        return None

    original_log = runner.log
    progress = None
    if mode == "progress":
        from tqdm.auto import tqdm

        progress = tqdm(
            total=total_iterations,
            desc=description,
            unit="iter",
            dynamic_ncols=True,
            leave=True,
        )

    def compact_log(locs: dict) -> None:
        # original_log performs every SummaryWriter.add_scalar call. Suppress
        # only its multi-line printout so TensorBoard remains unchanged.
        with contextlib.redirect_stdout(io.StringIO()):
            original_log(locs)

        if progress is None:
            return
        duration = locs["collection_time"] + locs["learn_time"]
        fps = int(runner.num_steps_per_env * runner.env.num_envs / max(duration, 1.0e-9))
        reward = None
        if len(locs["rewbuffer"]) > 0:
            reward = float(statistics.mean(locs["rewbuffer"]))

        termination_totals = {"success": 0.0, "time_out": 0.0, "over_insertion": 0.0}
        for episode_info in locs["ep_infos"]:
            for key, value in episode_info.items():
                termination_name = None
                if key.endswith("Termination/success"):
                    termination_name = "success"
                elif key.endswith("Termination/time_out"):
                    termination_name = "time_out"
                elif key.endswith("Termination/over_insertion"):
                    termination_name = "over_insertion"
                if termination_name is not None:
                    try:
                        amount = float(value.float().mean().item())
                    except AttributeError:
                        amount = float(value)
                    termination_totals[termination_name] += amount
        termination_count = sum(termination_totals.values())
        success = (
            termination_totals["success"] / termination_count
            if termination_count > 0.0 else None
        )

        postfix = {
            "fps": fps,
            "reward": f"{reward:.2f}" if reward is not None else "--",
            "success": f"{100.0 * success:.1f}%" if success is not None else "--",
            "noise": f"{runner.alg.policy.action_std.mean().item():.2f}",
        }
        progress.set_postfix(postfix, refresh=False)
        progress.update(1)

    runner.log = compact_log
    return progress


def main() -> None:
    args = parse_args()
    task = args.task or CURRICULUM_TASKS[args.stage]

    # Isaac/Omniverse modules must only be imported after AppLauncher.
    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher(headless=args.headless, device=args.device).app
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
        from scripts.depth_safety import wrap_with_predictive_depth_barrier

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        env_cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
        env_cfg.seed = args.seed
        env_cfg.scene.num_envs = args.num_envs
        apply_verified_physics(env_cfg)
        if args.episode_length_s is not None:
            env_cfg.episode_length_s = args.episode_length_s
        if args.geometric_teacher_residual:
            apply_teacher_observation_noise(
                env_cfg,
                args.teacher_observation_noise_mm,
                args.teacher_tilt_noise_deg,
            )
        if args.nullspace_stiffness is not None:
            env_cfg.actions.arm_action.controller_cfg.nullspace_stiffness = (
                args.nullspace_stiffness
            )

        agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
        if args.geometric_teacher_residual:
            from scripts.residual_policy import GeometricTeacherResidualActorCritic

            runner_module.GeometricTeacherResidualActorCritic = GeometricTeacherResidualActorCritic
            GeometricTeacherResidualActorCritic.default_freeze_critic = bool(
                args.residual_freeze_critic
            )
            GeometricTeacherResidualActorCritic.default_teacher_alignment_gate_mm = (
                float(args.teacher_alignment_gate_mm)
            )
            agent_cfg.policy.class_name = "GeometricTeacherResidualActorCritic"
            # The analytic teacher already provides a directed action. Keep
            # exploration small so the smoke test measures residual learning,
            # not an unnecessarily noisy initial policy.
            agent_cfg.policy.init_noise_std = 0.1
            if args.residual_penalty_coef > 0.0:
                from scripts.residual_ppo import ResidualPPO

                ResidualPPO.default_residual_penalty_coef = args.residual_penalty_coef
                runner_module.PPO = ResidualPPO
                agent_cfg.algorithm.class_name = "PPO"
        elif args.residual_adapter:
            from scripts.residual_policy import ResidualActorCritic

            # OnPolicyRunner resolves policy class names in its own module
            # namespace via eval(). Register the local residual class there.
            runner_module.ResidualActorCritic = ResidualActorCritic
            ResidualActorCritic.default_freeze_critic = bool(args.residual_freeze_critic)
            agent_cfg.policy.class_name = "ResidualActorCritic"
            if args.residual_penalty_coef > 0.0:
                from scripts.residual_ppo import ResidualPPO

                ResidualPPO.default_residual_penalty_coef = args.residual_penalty_coef
                # OnPolicyRunner uses the literal "PPO" to select RL mode,
                # then eval()s that name from its own module namespace.
                runner_module.PPO = ResidualPPO
                agent_cfg.algorithm.class_name = "PPO"
        agent_cfg.seed = args.seed
        if args.max_iterations is not None:
            agent_cfg.max_iterations = args.max_iterations
        if args.num_steps_per_env is not None:
            agent_cfg.num_steps_per_env = args.num_steps_per_env
        if args.save_interval is not None:
            agent_cfg.save_interval = args.save_interval
        if args.learning_rate is not None:
            agent_cfg.algorithm.learning_rate = args.learning_rate
        if args.num_learning_epochs is not None:
            agent_cfg.algorithm.num_learning_epochs = args.num_learning_epochs
        if args.entropy_coef is not None:
            agent_cfg.algorithm.entropy_coef = args.entropy_coef
        if args.init_noise_std is not None:
            agent_cfg.policy.init_noise_std = args.init_noise_std
        agent_cfg.experiment_name = "franka_peg_in_hole_pose6d"
        stage_name = "custom" if args.task else args.stage
        agent_cfg.run_name = args.run_name or stage_name

        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = args.log_root / stage_name / timestamp
        run_dir.mkdir(parents=True, exist_ok=False)
        over_insertion_term = getattr(env_cfg.terminations, "over_insertion", None)
        metadata = {
            "task": task,
            "stage": stage_name,
            "num_envs": args.num_envs,
            "max_iterations": agent_cfg.max_iterations,
            "num_steps_per_env": agent_cfg.num_steps_per_env,
            "seed": args.seed,
            "device": args.device,
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "load_mode": args.load_mode if args.checkpoint else None,
            "console": args.console,
            "ppo_overrides": {
                "learning_rate": agent_cfg.algorithm.learning_rate,
                "num_learning_epochs": agent_cfg.algorithm.num_learning_epochs,
                "entropy_coef": agent_cfg.algorithm.entropy_coef,
                "init_noise_std": agent_cfg.policy.init_noise_std,
                "residual_adapter": bool(args.residual_adapter),
                "residual_weight_decay": args.residual_weight_decay,
                "residual_penalty_coef": args.residual_penalty_coef,
                "residual_freeze_critic": bool(args.residual_freeze_critic),
                "geometric_teacher_residual": bool(args.geometric_teacher_residual),
                "teacher_observation_noise_mm": args.teacher_observation_noise_mm,
                "teacher_tilt_noise_deg": args.teacher_tilt_noise_deg,
                "hole_xy_bias_std_mm": args.hole_xy_bias_std_mm,
                "action_noise_std": args.action_noise_std,
                "action_gain_noise_std_pct": args.action_gain_noise_std_pct,
                "reset_warmup_steps": args.reset_warmup_steps,
                "teacher_alignment_gate_mm": args.teacher_alignment_gate_mm,
            },
            "episode_length_s": float(env_cfg.episode_length_s),
            "physics": {
                "dt": env_cfg.sim.dt,
                "decimation": env_cfg.decimation,
                "policy_dt": env_cfg.sim.dt * env_cfg.decimation,
                "solver_position_iterations": 16,
                "solver_velocity_iterations": 4,
                "enhanced_determinism": True,
            },
            "task_safety": {
                "depth_safety_barrier_mm": args.depth_safety_barrier_mm,
                "insertion_gate_radial_mm": args.insertion_gate_radial_mm,
                "insertion_gate_tilt_deg": args.insertion_gate_tilt_deg,
                "insertion_gate_hysteresis": args.insertion_gate_hysteresis,
                "insertion_gate_start_depth_mm": args.insertion_gate_start_depth_mm,
                "insertion_soft_gate_min_scale": args.insertion_soft_gate_min_scale,
                "nullspace_stiffness": float(
                    env_cfg.actions.arm_action.controller_cfg.nullspace_stiffness
                ),
                "position_scale_m": (
                    list(env_cfg.actions.arm_action.position_scale)
                    if isinstance(env_cfg.actions.arm_action.position_scale, (tuple, list))
                    else float(env_cfg.actions.arm_action.position_scale)
                ),
                "success_depth_range_m": [
                    float(env_cfg.terminations.success.params["depth_required"]),
                    float(env_cfg.terminations.success.params["max_depth"]),
                ],
                "over_insertion_depth_m": (
                    float(over_insertion_term.params["max_depth"])
                    if over_insertion_term is not None
                    else None
                ),
            },
        }
        (run_dir / "run_config.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        print("\n" + "=" * 68)
        print("  Franka Peg-in-Hole Pose6D PPO Training")
        print(f"  Stage / task : {stage_name} / {task}")
        print(f"  Environments : {args.num_envs}")
        print(f"  Iterations   : {agent_cfg.max_iterations}")
        print(f"  Seed / device: {args.seed} / {args.device}")
        print(
            f"  PPO update   : lr={agent_cfg.algorithm.learning_rate:g}, "
            f"epochs={agent_cfg.algorithm.num_learning_epochs}, "
            f"entropy={agent_cfg.algorithm.entropy_coef:g}"
        )
        print(
            f"  Physics      : 120Hz, decimation={env_cfg.decimation}, "
            "solver=16/4, enhanced"
        )
        print(f"  XYZ scale    : {env_cfg.actions.arm_action.position_scale}")
        print(
            "  Nullspace K  : "
            f"{env_cfg.actions.arm_action.controller_cfg.nullspace_stiffness}"
        )
        print(
            "  Depth barrier: "
            + (
                f"{args.depth_safety_barrier_mm:.2f}mm"
                if args.depth_safety_barrier_mm is not None
                else "disabled"
            )
        )
        print(f"  Output       : {run_dir}")
        if args.checkpoint:
            print(f"  Checkpoint   : {args.checkpoint} ({args.load_mode})")
        print("=" * 68 + "\n")

        env = gym.make(task, cfg=env_cfg)
        if args.action_noise_std > 0.0 or args.action_gain_noise_std_pct > 0.0:
            from scripts.action_perturbation import wrap_with_action_perturbation

            env = wrap_with_action_perturbation(
                env,
                gym=gym,
                torch=torch,
                action_noise_std=args.action_noise_std,
                action_gain_noise_std_pct=args.action_gain_noise_std_pct,
                reset_warmup_steps=args.reset_warmup_steps,
            )
        if args.hole_xy_bias_std_mm > 0.0:
            from scripts.observation_perturbation import wrap_with_hole_xy_bias

            env = wrap_with_hole_xy_bias(
                env,
                gym=gym,
                torch=torch,
                bias_std_mm=args.hole_xy_bias_std_mm,
            )
        if args.depth_safety_barrier_mm is not None:
            env = wrap_with_predictive_depth_barrier(
                env,
                barrier_mm=args.depth_safety_barrier_mm,
                position_scale=env_cfg.actions.arm_action.position_scale,
                gym=gym,
                torch=torch,
                geometry=geo,
                insertion_gate_radial_mm=args.insertion_gate_radial_mm,
                insertion_gate_tilt_deg=args.insertion_gate_tilt_deg,
                insertion_gate_hysteresis=args.insertion_gate_hysteresis,
                insertion_gate_start_depth_mm=args.insertion_gate_start_depth_mm,
                insertion_soft_gate_min_scale=args.insertion_soft_gate_min_scale,
            )
        print(f"[INFO] Observation space: {env.observation_space}")
        print(f"[INFO] Action space: {env.action_space}")
        wrapped_env = RslRlVecEnvWrapper(env)
        print(
            f"[INFO] Runner classes: policy={agent_cfg.policy.class_name}, "
            f"algorithm={agent_cfg.algorithm.class_name}",
            flush=True,
        )
        runner = OnPolicyRunner(
            wrapped_env, agent_cfg.to_dict(), log_dir=str(run_dir), device=args.device
        )
        if args.residual_adapter and args.residual_weight_decay > 0.0:
            # The residual actor is the only trainable actor component.  The
            # optimizer also contains the adaptive critic, so this applies a
            # conservative coupled L2 decay to both trainable parts.
            for param_group in runner.alg.optimizer.param_groups:
                param_group["weight_decay"] = args.residual_weight_decay
            print(
                f"[INFO] Residual transfer weight decay: {args.residual_weight_decay:g}",
                flush=True,
            )
        if args.checkpoint:
            print(f"[INFO] Loading checkpoint ({args.load_mode}): {args.checkpoint}")
            runner.load(
                str(args.checkpoint), load_optimizer=(args.load_mode == "resume")
            )
            if args.load_mode == "transfer":
                # RSL-RL restores the source iteration even when the optimizer
                # is intentionally omitted. A new curriculum stage should log
                # and checkpoint from iteration zero.
                runner.current_learning_iteration = 0

        print("[INFO] Starting PPO training", flush=True)
        progress = install_console_logger(
            runner,
            mode=args.console,
            total_iterations=agent_cfg.max_iterations,
            description=f"PPO {stage_name}",
        )
        try:
            runner.learn(
                num_learning_iterations=agent_cfg.max_iterations,
                init_at_random_ep_len=True,
            )
        finally:
            if progress is not None:
                progress.close()
        print(f"[INFO] Training complete: {run_dir}")
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
