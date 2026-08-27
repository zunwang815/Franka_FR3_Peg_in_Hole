"""Full-pose OSC validation environment built on the physical C0 fixture."""

import math

from isaaclab.utils import configclass

from .osc_baseline_env_cfg import OscBaseline30mmEnvCfg


@configclass
class OscPose6DBaselineEnvCfg(OscBaseline30mmEnvCfg):
    """30 mm fixed-hole baseline with all six Cartesian axes controlled."""

    def __post_init__(self):
        super().__post_init__()

        # Candidate 2 from search_manipulable_vertical_pose.py: nearly the
        # best full-Jacobian conditioning (sigma_min=0.22799), 0.711 deg peg
        # tilt, and useful clearance from the arm joint limits.
        q = (0.010158, -0.956863, -0.009616, -2.485868,
             0.002040, 1.500947, 0.830643)
        self.scene.robot.init_state.joint_pos.update(
            {f"panda_joint{i + 1}": value for i, value in enumerate(q)}
        )

        # Keep the same 50 mm approach distance as C0. These are the settled
        # simulator coordinates after storing the candidate at six decimals;
        # using the raw search coordinates left a systematic 2.8 mm XY error.
        cx, cy = 0.249418, 0.001268
        fixture_surface_z = 0.343591
        platform_z = fixture_surface_z - 0.010
        hole_z = fixture_surface_z - 0.010

        slab_specs = {
            "fixture_left": (cx - 0.085, cy, platform_z),
            "fixture_right": (cx + 0.085, cy, platform_z),
            "fixture_front": (cx, cy + 0.085, platform_z),
            "fixture_back": (cx, cy - 0.085, platform_z),
        }
        for name, pos in slab_specs.items():
            getattr(self.scene, name).init_state.pos = pos

        self.scene.hole_board.init_state.pos = (cx, cy, hole_z)
        for i in range(36):
            angle = 2.0 * math.pi * i / 36
            wall = getattr(self.scene, f"hole_wall_{i}")
            wall.init_state.pos = (
                cx + 0.0160 * math.cos(angle),
                cy + 0.0160 * math.sin(angle),
                fixture_surface_z - 0.020,
            )

        self.events.randomize_hole.params["table_z"] = hole_z

        # The selected configuration has a well-conditioned full 6x7
        # Jacobian, so translation and tool-axis orientation can be controlled
        # together without the old wrist singularity.
        controller = self.actions.arm_action.controller_cfg
        # Control all six axes. Although a centered cylindrical peg is
        # geometrically yaw-invariant, a physical +/-5mm mount offset makes
        # free wrist yaw sweep the peg tip around the tool axis. Locking yaw
        # is therefore required for a stationary XY target under mounting
        # uncertainty. The searched pose has a full-rank 6x7 Jacobian.
        controller.motion_control_axes_task = (1, 1, 1, 1, 1, 1)
        controller.partial_inertial_dynamics_decoupling = False
        controller.motion_stiffness_task = (
            500.0, 500.0, 500.0, 400.0, 400.0, 400.0
        )
        controller.motion_damping_ratio_task = (1.0,) * 6
        # Restore the first verified single-environment configuration. The
        # projected posture objective prevents redundant-joint drift without
        # relying on implicit actuator springs.
        controller.nullspace_control = "position"
        controller.nullspace_stiffness = 20.0
        controller.nullspace_damping_ratio = 1.0
        self.actions.arm_action.nullspace_joint_pos_target = "default"

        # Full pose OSC now owns wrist orientation.  Retain only light joint
        # damping; implicit position springs would fight the Cartesian task.
        self.scene.robot.actuators["panda_shoulder"].stiffness = 0.0
        self.scene.robot.actuators["panda_forearm"].stiffness = 0.0

        criteria = {"radial_tol": 0.002, "depth_required": 0.015, "max_depth": 0.040,
                    "tilt_tol": 2.0 * math.pi / 180.0}
        self.rewards.success_bonus.params.update(criteria)
        self.terminations.success.params.update(criteria)
        self.rewards.tilt_penalty.params["tilt_tol"] = criteria["tilt_tol"]
