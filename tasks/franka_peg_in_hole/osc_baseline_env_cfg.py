"""OSC baseline: 30mm hole, vertical pose, OSC position control.

Same simplified baseline as Baseline30mmEnvCfg but with the Operational
Space Controller instead of Differential IK — avoids wrist singularity
at the vertical pose.
"""
import math
from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, CollisionPropertiesCfg
from isaaclab.utils import configclass

from .config.franka.osc_env_cfg import FrankaPegInHoleOscEnvCfg


@configclass
class OscBaseline30mmEnvCfg(FrankaPegInHoleOscEnvCfg):
    """OSC easy baseline: 30mm hole, fixed position, vertical pose, no noise."""

    def __post_init__(self):
        super().__post_init__()

        fixed_peg_usd = Path(__file__).resolve().parents[2] / "assets" / "panda_with_fixed_peg.usda"
        if fixed_peg_usd.is_file():
            # The peg is now an articulation link. Remove the legacy standalone
            # 100kg rigid body and its periodic pose overwrite event.
            self.scene.peg = None
            self.events.sync_peg_pose = None
            self.scene.peg_contact = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/peg",
                update_period=0.0,
                history_length=3,
                track_air_time=True,
                force_threshold=1.0,
            )

        # VERTICAL DEFAULT POSE: hand points straight down (tilt=0°)
        import torch as _torch
        _q = _torch.tensor([0.0, -0.9, 0.0, -2.4, 0.0, 1.5, 0.8, 0.04, 0.04],
                           dtype=_torch.float32)
        self.scene.robot.init_state.joint_pos = {
            f"panda_joint{i+1}": _q[i].item() for i in range(7)}
        self.scene.robot.init_state.joint_pos["panda_finger_joint1"] = 0.04
        self.scene.robot.init_state.joint_pos["panda_finger_joint2"] = 0.04

        # 30mm circular sleeve approximation.  Thirty-six thin, tangentially
        # rotated collision segments reduce the radial error of the old 12
        # axis-aligned blocks.  There is deliberately no solid plate below the
        # opening, so insertion cannot require tunnelling through a platform.
        cx = self.scene.hole_board.init_state.pos[0]
        for i in range(36):
            angle = 2 * math.pi * i / 36
            x = cx + 0.0160 * math.cos(angle)
            y = 0.0160 * math.sin(angle)
            rot = (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))
            setattr(self.scene, f"hole_wall_{i}", RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/HoleWall{i}",
                spawn=sim_utils.CuboidCfg(
                    size=(0.0020, 0.0028, 0.040),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.85, 0.1), roughness=0.3, metallic=0.1),
                    rigid_props=RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
                    collision_props=CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0.0)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, 0.02), rot=rot)))

        # Disable observation noise
        for term_name in ["joint_pos", "joint_vel", "peg_to_hole_vec", "peg_tilt"]:
            term = getattr(self.observations.policy, term_name, None)
            if term is not None:
                term.noise = None

        # Reachable insertion fixture.  The previous ground-level fixture put
        # the hole surface at 35mm while the vertical peg tip started at 418mm,
        # demanding ~38cm of straight-down travel at fixed orientation.  That
        # drives this pose toward its workspace/joint boundary and makes OSC
        # escape laterally.  Start 50mm above a table-height fixture instead.
        cx, cy = 0.269, 0.0
        fixture_surface_z = 0.368
        platform_z = fixture_surface_z - 0.010  # 20mm plate center
        # geometry.get_insertion_depth defines surface as hole marker z + 10mm.
        hole_z = fixture_surface_z - 0.010
        # Four slabs form the top plate while leaving a 40x40mm true opening.
        # The circular sleeve sits inside that opening.
        self.scene.table = None
        slab_specs = {
            "fixture_left": ((0.130, 0.300, 0.020), (cx - 0.085, cy, platform_z)),
            "fixture_right": ((0.130, 0.300, 0.020), (cx + 0.085, cy, platform_z)),
            "fixture_front": ((0.040, 0.130, 0.020), (cx, cy + 0.085, platform_z)),
            "fixture_back": ((0.040, 0.130, 0.020), (cx, cy - 0.085, platform_z)),
        }
        for name, (size, pos) in slab_specs.items():
            setattr(self.scene, name, RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{name}",
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.5, 0.5, 0.5), roughness=0.5),
                    rigid_props=RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
                    collision_props=CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0.0)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos)))

        self.scene.hole_board.init_state.pos = (cx, cy, hole_z)
        if self.scene.peg is not None:
            self.scene.peg.init_state.pos = (cx, cy, hole_z)
        for i in range(36):
            angle = 2 * math.pi * i / 36
            x = cx + 0.0160 * math.cos(angle)
            y = cy + 0.0160 * math.sin(angle)
            wall = getattr(self.scene, f"hole_wall_{i}")
            # Sleeve extends downward from the fixture surface.
            wall.init_state.pos = (x, y, fixture_surface_z - 0.020)
            wall.init_state.rot = (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))

        # Fixed hole, exact vertical pose every reset
        self.events.randomize_hole.params["x_range"] = (0.0, 0.0)
        self.events.randomize_hole.params["y_range"] = (0.0, 0.0)
        self.events.randomize_hole.params["table_z"] = hole_z
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

        # Disable cylinder termination
        self.terminations.peg_left_cylinder = None

        # Explicit curriculum criteria.  Never mutate geometry module globals:
        # reward, termination and evaluation must all receive the same values.
        criteria = {"radial_tol": 0.002, "depth_required": 0.015, "max_depth": 0.040,
                    "tilt_tol": 50.0 * math.pi / 180.0}
        self.rewards.success_bonus.params.update(criteria)
        self.terminations.success.params.update(criteria)
        self.rewards.jam_penalty.params["radial_tol"] = criteria["radial_tol"]
        self.rewards.tilt_penalty.params["tilt_tol"] = criteria["tilt_tol"]
