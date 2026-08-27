#!/usr/bin/env python3
"""Compute a joint configuration that places the hand above the fixed hole.

Runs the differential IK offline and prints joint angles to hardcode
into the baseline config for reliable warm-start.
"""
import sys
from pathlib import Path

# Make the project importable without depending on one developer's filesystem.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app

import torch
import gymnasium as gym
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
import tasks.franka_peg_in_hole.config.franka

task = 'Isaac-PegInHole-Franka-IK-Rel-Baseline-v0'
cfg = parse_env_cfg(task, device='cuda', num_envs=1)
cfg.scene.num_envs = 1
env = gym.make(task, cfg=cfg)
obs, _ = env.reset()
uenv = env.unwrapped

# Read hole position (ground truth, fixed in baseline)
from isaaclab.assets import RigidObject
hole: RigidObject = uenv.scene["hole_board"]
hole_pos = hole.data.root_pos_w[0, :3].clone()

# Target: hand such that peg tip is ~10cm above hole surface.
# Peg tip = hand + 0.17 along tool Z (approximately world Z for vertical hand)
# hand_z = hole_z + 0.10 + 0.17 = hole_z + 0.27
target = hole_pos + torch.tensor([0.0, 0.0, 0.27], device='cuda')

# Access IK controller from the action space
ik = uenv.action_manager._terms["arm_action"]._ik_controller
robot = uenv.scene["robot"]
body_idx = robot.find_bodies("panda_hand")[0][0]

# Start from default joint positions
joint_pos = robot.data.default_joint_pos[:, :7].clone()

# Set IK desired position
ik.ee_pos_des[0] = target

# Iterate IK with fresh Jacobian
for it in range(200):
    # Get jacobian for arm joints (0-6)
    jacobian = robot.root_physx_view.get_jacobians()[:, body_idx, :, 0:7]
    body_pos = robot.data.body_state_w[:, body_idx, :3]
    body_quat = robot.data.body_state_w[:, body_idx, 3:7]
    new_jp = ik.compute(body_pos, body_quat, jacobian, joint_pos)
    joint_pos = new_jp
    # Write to sim to update Jacobian
    robot.write_joint_position_to_sim(joint_pos[0, :7].unsqueeze(0), joint_ids=list(range(7)))
    # Refresh state
    env.step(torch.zeros(1, 3, device='cuda'))
    body_pos = robot.data.body_state_w[:, body_idx, :3]
    err = torch.norm(body_pos[0] - target).item()
    if it % 20 == 0:
        print(f"  iter {it}: joint_pos={joint_pos[0].cpu().numpy().round(4)}, err={err:.4f}m")

print(f"\nTarget: {target.cpu().numpy()}")
print(f"Final hand pos: {body_pos[0].cpu().numpy()}")
print(f"Final error: {err:.4f}m")
print(f"\nJOINT ANGLES (hardcode into config):")
print(f"[{', '.join(f'{x:.4f}' for x in joint_pos[0].cpu().numpy())}]")

env.close()
app.close()
