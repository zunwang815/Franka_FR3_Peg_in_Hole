"""Six-hole array with 6D relative-pose operational-space control."""

import math
from pathlib import Path

from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

from ...peg_in_hole_array_env_cfg import PegInHoleArrayEnvCfg


@configclass
class FrankaPegInHoleArrayOscPose6DEnvCfg(PegInHoleArrayEnvCfg):
    """Array task used by the analytic teacher and residual policy.

    The array uses the same fixed Peg articulation chain as the verified
    single-hole OSC task. The arm is controlled by a six-axis relative-pose
    OSC action, so its six action values match the teacher interface directly.
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = FRANKA_PANDA_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
        )
        fixed_peg_usd = Path(__file__).resolve().parents[4] / "assets" / "panda_with_fixed_peg.usda"
        if fixed_peg_usd.is_file():
            self.scene.robot.spawn.usd_path = str(fixed_peg_usd)
            self.scene.robot.spawn.activate_contact_sensors = True
            self.scene.robot.actuators["peg_mount"] = IdealPDActuatorCfg(
                joint_names_expr=["peg_mount_joint_.*"],
                effort_limit=500.0,
                effort_limit_sim=500.0,
                velocity_limit=10.0,
                velocity_limit_sim=10.0,
                stiffness=1.0e4,
                damping=200.0,
                armature=1.0,
            )
            self.scene.robot.init_state.joint_pos["peg_mount_joint_.*"] = 0.0
            # Reuse task-1's fixed Peg kinematic chain. The standalone Peg
            # synchronization path is deliberately disabled for this OSC
            # variant because it has different dynamics from the accepted
            # baseline.
            self.scene.peg = None
            self.events.sync_peg_pose = None
            self.events.sync_peg_pose_reset = None
        # Well-conditioned upright posture used by the verified Pose6D OSC
        # baseline.  The array task must not inherit the legacy broad joint
        # randomization, which can place the full-pose Jacobian near a wrist
        # singularity before the teacher has issued its first action.
        initial_q = (
            0.010158, -0.956863, -0.009616, -2.485868,
            0.002040, 1.500947, 0.830643,
        )
        self.scene.robot.init_state.joint_pos.update(
            {f"panda_joint{i + 1}": value for i, value in enumerate(initial_q)}
        )
        self.scene.robot.spawn.rigid_props.disable_gravity = True
        # The six sleeves are the physical fixture.  The legacy table asset
        # sits at a different height and would introduce an unrelated plate
        # collision into the array audit.
        self.scene.table = None
        self.scene.robot.actuators["panda_shoulder"].stiffness = 0.0
        self.scene.robot.actuators["panda_shoulder"].damping = 0.0
        self.scene.robot.actuators["panda_forearm"].stiffness = 0.0
        self.scene.robot.actuators["panda_forearm"].damping = 0.0
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="ee_tool",
                ),
            ],
        )

        self.actions.arm_action = OperationalSpaceControllerActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller_cfg=OperationalSpaceControllerCfg(
                target_types=["pose_rel"],
                motion_control_axes_task=(1, 1, 1, 1, 1, 1),
                impedance_mode="fixed",
                inertial_dynamics_decoupling=True,
                partial_inertial_dynamics_decoupling=False,
                gravity_compensation=False,
                motion_stiffness_task=(500.0, 500.0, 500.0, 400.0, 400.0, 400.0),
                motion_damping_ratio_task=(1.0,) * 6,
                nullspace_control="position",
                nullspace_stiffness=20.0,
                nullspace_damping_ratio=1.0,
            ),
            nullspace_joint_pos_target="default",
            position_scale=0.005,
            orientation_scale=1.0,
            stiffness_scale=1.0,
        )
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        fixture_surface_z = 0.343591
        hole_z = fixture_surface_z - 0.010
        array_center_x, array_center_y = 0.249418, 0.001268
        self.scene.hole_board.init_state.pos = (array_center_x, array_center_y, hole_z)
        for hole_index in range(6):
            col = hole_index % 3
            row = hole_index // 3
            dx = (col - 1.0) * 0.03
            dy = (row - 0.5) * 0.03
            for wall_index in range(36):
                angle = 2.0 * math.pi * wall_index / 36.0
                wall = getattr(self.scene, f"array_hole_{hole_index}_wall_{wall_index}")
                wall.init_state.pos = (
                    array_center_x + dx + 0.0125 * math.cos(angle),
                    array_center_y + dy + 0.0125 * math.sin(angle),
                    fixture_surface_z - 0.020,
                )
                wall.init_state.rot = (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0))
        self.events.randomize_hole.params["table_z"] = hole_z
        # The legacy all-joint check treats the intentionally closed/near-limit
        # finger joints as a failure.  The array arm action only controls the
        # seven arm joints, so keep the verified arm posture and disable that
        # unrelated termination for this OSC task.
        self.terminations.joint_limits = None

        # Pose6D teacher observation: q(9), qdot(9), relative position(3),
        # upright-axis error(3), target id(6), previous action(6).
        self.observations.policy.ee_position = None
        self.observations.policy.ee_orientation = None
        self.observations.policy.hole_position = None

        self.decimation = 4
        self.episode_length_s = 12.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.solver_position_iteration_count = 16
        self.sim.physx.solver_velocity_iteration_count = 4
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_correlation_distance = 0.005
        self.sim.use_fabric = False


@configclass
class FrankaPegInHoleArrayOscPose6DEnvCfg_PLAY(FrankaPegInHoleArrayOscPose6DEnvCfg):
    """Single-environment rendering configuration."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.0
        self.observations.policy.enable_corruption = False
