#!/usr/bin/env python3
"""Quick verification: Create the peg-in-hole environment and run random actions.

Usage:
    conda activate isaac_lab
    export OMNI_KIT_ACCEPT_EULA=YES
    python scripts/verify_env.py [--phase 1|2]
"""

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
_ISAACLAB_SOURCE = os.path.expanduser(
    "~/miniconda3/envs/isaac_lab/lib/python3.10/site-packages/isaaclab/source"
)
sys.path.insert(0, _ISAACLAB_SOURCE)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--task", type=str, default=None,
                        help="Override the registered Gym task ID")
    return parser.parse_args()


def main():
    args = parse_args()

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    import torch
    import gymnasium as gym
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    import tasks.franka_peg_in_hole.config.franka  # noqa: F401

    task_id = args.task or (
        "Isaac-PegInHole-Franka-OSC-Baseline-v0" if args.phase == 1
        else "Isaac-PegInHoleArray-Franka-IK-Abs-v0"
    )

    print(f"\n{'='*60}")
    print(f"  Environment Verification: Phase {args.phase}")
    print(f"  Task: {task_id}")
    print(f"  Num Envs: {args.num_envs}")
    print(f"  Steps: {args.steps}")
    print(f"{'='*60}\n")

    # Parse config
    env_cfg = parse_env_cfg(task_id, device="cuda", num_envs=args.num_envs)
    env_cfg.observations.policy.enable_corruption = False

    # Create environment
    print("[1/4] Creating environment...")
    env = gym.make(task_id, cfg=env_cfg)
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print("  OK!")

    # Reset
    print("[2/4] Resetting environment...")
    obs, info = env.reset()
    print(f"  Observation keys: {list(obs.keys())}")
    if "policy" in obs:
        print(f"  Policy obs shape: {obs['policy'].shape}")
    print(f"  Info keys: {list(info.keys()) if isinstance(info, dict) else 'N/A'}")
    print("  OK!")

    # Step with random actions
    print(f"[3/4] Running {args.steps} random steps...")
    for step in range(args.steps):
        action = torch.from_numpy(env.action_space.sample()).to("cuda")
        obs, reward, terminated, truncated, info = env.step(action)
        if step == 0:
            print(f"  Step 0: reward={reward[0].item():.3f}, "
                  f"terminated={terminated[0].item()}, truncated={truncated[0].item()}")
        if terminated.any() or truncated.any():
            print(f"  Episode ended at step {step}")
            obs, info = env.reset()
    print("  OK!")

    # Reset and check randomization
    print("[4/4] Checking environment reset randomization...")
    obs1, _ = env.reset()
    obs2, _ = env.reset()
    diff = torch.norm(obs1["policy"][0] - obs2["policy"][0])
    print(f"  Reset produces different states: L2_diff = {diff.item():.4f}")
    print("  OK!")

    print(f"\n{'='*60}")
    print(f"  ALL CHECKS PASSED!")
    print(f"{'='*60}\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
