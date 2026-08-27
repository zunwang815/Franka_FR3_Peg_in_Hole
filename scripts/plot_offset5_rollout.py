#!/usr/bin/env python3
"""Plot one verified offset5 policy rollout.

The plot is intentionally numerical rather than a rendered viewport: it is
reproducible on headless Isaac Sim and shows the quantities used by the C0
acceptance criteria (radial error, insertion depth, tilt, and actions).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_phase1 import apply_verified_physics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/tmp/offset5_model499_rollout.png"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=240)
    args = parser.parse_args()

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, device=args.device).app
    env = None
    try:
        import gymnasium as gym
        import matplotlib.pyplot as plt
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg
        from rsl_rl.runners import OnPolicyRunner

        import tasks.franka_peg_in_hole.config.franka  # noqa: F401
        from tasks.franka_peg_in_hole.mdp import geometry as geo

        task = "Isaac-PegInHole-Franka-OSC-Pose6D-PegOffset5mm-v0"
        cfg = parse_env_cfg(task, device=args.device, num_envs=1)
        cfg.seed = args.seed
        cfg.scene.num_envs = 1
        cfg.observations.policy.enable_corruption = False
        apply_verified_physics(cfg)
        env = gym.make(task, cfg=cfg)
        wrapped = RslRlVecEnvWrapper(env)
        agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
        runner = OnPolicyRunner(
            wrapped, agent_cfg.to_dict(), log_dir="/tmp/offset5_rollout", device=args.device
        )
        runner.load(str(args.checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device=args.device)
        wrapped.seed(args.seed)
        obs, _ = wrapped.reset()

        uenv = env.unwrapped
        radial = []
        depth = []
        tilt = []
        actions = []
        t = []
        success_step = None
        for step in range(args.max_steps):
            peg_tip, peg_quat = geo.get_peg_tip(uenv)
            hole_pos, _ = geo.get_hole_center(uenv)
            radial.append(float(geo.get_radial_error(peg_tip, hole_pos)[0].item() * 1000.0))
            depth.append(float(geo.get_insertion_depth(peg_tip, hole_pos)[0].item() * 1000.0))
            tilt.append(float(geo.get_tilt_angle(peg_quat)[0].item() * 180.0 / math.pi))
            t.append(step * float(cfg.sim.dt * cfg.decimation))

            with torch.inference_mode():
                action = policy(obs["policy"] if isinstance(obs, dict) else obs)
            actions.append(action[0].detach().cpu().tolist())
            obs, _, terminated, truncated, _ = env.step(action)
            done = bool((terminated | truncated)[0].item())
            if done:
                # Manager-based environments reset completed instances inside
                # step(); use the cached terminal values instead of plotting
                # the next episode's reset state.
                radial.append(float(uenv._termination_radial_error[0].item() * 1000.0))
                depth.append(float(uenv._termination_depth[0].item() * 1000.0))
                tilt.append(float(uenv._termination_tilt[0].item() * 180.0 / math.pi))
                t.append((step + 1) * float(cfg.sim.dt * cfg.decimation))
                success_step = step + 1 if bool(uenv._success_termination_mask[0].item()) else None
                break

        if len(actions) == args.max_steps and success_step is None:
            # No terminal event: append the actual current state only in the
            # timeout/truncated case.
            peg_tip, peg_quat = geo.get_peg_tip(uenv)
            hole_pos, _ = geo.get_hole_center(uenv)
            radial.append(float(geo.get_radial_error(peg_tip, hole_pos)[0].item() * 1000.0))
            depth.append(float(geo.get_insertion_depth(peg_tip, hole_pos)[0].item() * 1000.0))
            tilt.append(float(geo.get_tilt_angle(peg_quat)[0].item() * 180.0 / math.pi))
            t.append(len(actions) * float(cfg.sim.dt * cfg.decimation))

        actions_tensor = torch.tensor(actions, dtype=torch.float32).numpy()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
        axes[0].plot(t, radial, color="#1769aa", linewidth=2, label="radial error")
        axes[0].axhline(2.0, color="#d62728", linestyle="--", label="success gate 2 mm")
        axes[0].set_ylabel("Radial error (mm)")
        axes[0].legend(loc="best")
        axes[0].grid(alpha=0.25)

        axes[1].plot(t, depth, color="#2ca02c", linewidth=2, label="insertion depth")
        axes[1].axhspan(15.0, 40.0, color="#2ca02c", alpha=0.10, label="success depth 15--40 mm")
        axes[1].axhline(15.0, color="#2ca02c", linestyle=":")
        axes[1].axhline(40.0, color="#d62728", linestyle=":")
        axes[1].set_ylabel("Depth (mm)")
        axes[1].legend(loc="best")
        axes[1].grid(alpha=0.25)

        axes[2].plot(t, tilt, color="#9467bd", linewidth=2, label="tilt")
        axes[2].axhline(2.0, color="#d62728", linestyle="--", label="tilt gate 2 deg")
        axes[2].set_ylabel("Tilt (deg)")
        axes[2].set_xlabel("Time (s)")
        axes[2].legend(loc="best")
        axes[2].grid(alpha=0.25)

        status = "SUCCESS" if success_step is not None else "NOT SUCCESS"
        fig.suptitle(f"Offset5 model_499 rollout | {status} | steps={len(actions)}")
        fig.savefig(args.output, dpi=160)
        plt.close(fig)

        npz_path = args.output.with_suffix(".npz")
        import numpy as np

        np.savez(
            npz_path,
            time_s=np.asarray(t),
            radial_mm=np.asarray(radial),
            depth_mm=np.asarray(depth),
            tilt_deg=np.asarray(tilt),
            actions=actions_tensor,
            success_step=-1 if success_step is None else success_step,
        )
        print(f"ROLLOUT status={status} steps={len(actions)}")
        print(f"PLOT {args.output}")
        print(f"DATA {npz_path}")
        print(
            f"FINAL radial/depth/tilt={radial[-1]:.3f}mm/{depth[-1]:.3f}mm/{tilt[-1]:.3f}deg"
        )
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
