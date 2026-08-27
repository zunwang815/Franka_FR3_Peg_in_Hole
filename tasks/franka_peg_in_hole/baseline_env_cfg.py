"""Simplified PPO baseline per review Phase 2.

30mm hole, fixed position, no noise, 3D position control.
"""
import math
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, CollisionPropertiesCfg
from isaaclab.utils import configclass

from .config.franka.ik_rel_env_cfg import FrankaPegInHoleRelEnvCfg


@configclass
class Baseline30mmEnvCfg(FrankaPegInHoleRelEnvCfg):
    """Easy baseline: 30mm hole, fixed position, no noise."""

    def __post_init__(self):
        super().__post_init__()

        # VERTICAL DEFAULT POSE: hand points straight down (tilt=0°)
        # Found via scripts/search_vertical_pose.py grid search.
        import torch as _torch
        _q = _torch.tensor([0.0, -0.9, 0.0, -2.4, 0.0, 1.5, 0.8, 0.04, 0.04],
                           dtype=_torch.float32)
        self.scene.robot.init_state.joint_pos = {
            f"panda_joint{i+1}": _q[i].item() for i in range(7)}
        self.scene.robot.init_state.joint_pos["panda_finger_joint1"] = 0.04
        self.scene.robot.init_state.joint_pos["panda_finger_joint2"] = 0.04

        # Override scene: 30mm hole (r=15mm), pillars at r=17.5mm
        cx = self.scene.hole_board.init_state.pos[0]  # 0.525
        for i in range(12):
            angle = 2 * math.pi * i / 12
            x = cx + 0.0175 * math.cos(angle)
            y = 0.0175 * math.sin(angle)
            name = f"hole_wall_{i}"
            wall = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/HoleWall{i}",
                spawn=sim_utils.CuboidCfg(
                    size=(0.005, 0.005, 0.02),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.85, 0.1), roughness=0.3, metallic=0.1),
                    rigid_props=RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
                    collision_props=CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0.0)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, 0.02)))
            setattr(self.scene, name, wall)

        # Disable observation noise
        for term_name in ["joint_pos", "joint_vel", "peg_to_hole_vec", "peg_tilt"]:
            term = getattr(self.observations.policy, term_name, None)
            if term is not None:
                term.noise = None

        # SMALL PLATFORM at VERTICAL-POSE hand position.
        # Vertical pose (tilt=0°): q=(0,-0.9,0,-2.4,0,1.5,0.8), hand at
        # (0.269, 0, 0.538), peg tip ≈ hand-0.12 = (0.269, 0, 0.418).
        # Hole surface at tip - 5cm = 0.368; platform top at 0.363.
        cx, cy = 0.269, 0.0
        hole_z = 0.368
        platform_z = hole_z - 0.010  # platform top at 0.358 → max depth 20mm > 15mm required
        self.scene.table = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Platform",
            spawn=sim_utils.CuboidCfg(
                size=(0.30, 0.30, 0.02),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.5, 0.5, 0.5), roughness=0.5),
                rigid_props=RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
                collision_props=CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0.0)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(cx, cy, platform_z)))

        self.scene.hole_board.init_state.pos = (cx, cy, hole_z)
        self.scene.peg.init_state.pos = (cx, cy, hole_z)
        # Move walls to match new hole position
        for i in range(12):
            angle = 2 * math.pi * i / 12
            x = cx + 0.0175 * math.cos(angle)
            y = cy + 0.0175 * math.sin(angle)
            getattr(self.scene, f"hole_wall_{i}").init_state.pos = (x, y, hole_z)

        # Fixed hole position (no randomization), raised to platform height
        self.events.randomize_hole.params["x_range"] = (0.0, 0.0)
        self.events.randomize_hole.params["y_range"] = (0.0, 0.0)
        self.events.randomize_hole.params["table_z"] = 0.368  # hole surface z

        # EXACT vertical pose every reset: joint scaling ±5% destroys the
        # carefully-found tilt=0° configuration (±5% → tilt 6.7°, tip +10cm).
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

        # Disable cylinder termination (configclass: set to None, don't del)
        self.terminations.peg_left_cylinder = None

        # Easier success criteria — tilt relaxed because 3D position control
        # cannot change orientation (Franka natural tilt is 30-50°).
        # Depth 15mm: pillars are 20mm tall, peg tip at 15mm depth is still
        # 5mm above the table surface (physically achievable).
        criteria = {"radial_tol": 0.002, "depth_required": 0.015,
                    "tilt_tol": 50.0 * math.pi / 180.0}
        self.rewards.success_bonus.params.update(criteria)
        self.terminations.success.params.update(criteria)
        self.rewards.jam_penalty.params["radial_tol"] = criteria["radial_tol"]
        self.rewards.tilt_penalty.params["tilt_tol"] = criteria["tilt_tol"]
