"""Franka peg-in-hole with Operational Space Controller (OSC).

OSC directly computes joint TORQUES from task-space errors — unlike
Differential IK, it does not suffer from singularities at the vertical
wrist configuration. Per review recommendation 4.2.
"""

from pathlib import Path

from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

from . import joint_pos_env_cfg


@configclass
class FrankaPegInHoleOscEnvCfg(joint_pos_env_cfg.FrankaPegInHoleEnvCfg):
    """Franka peg-in-hole with OSC relative-pose control.

    Action: (dx, dy, dz) in task frame, position_scale=0.005 → ±5mm/step.
    The pose_rel interface is 6-D, but the baseline activates XYZ translation
    only; its three rotation slots are retained as zeros.
    """

    def __post_init__(self):
        super().__post_init__()

        # Effort-control robot: zero PD gains (OSC computes torques directly)
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
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
                # Explicit PD avoids the unstable PhysX implicit drive on the
                # nested X/Y prismatic mount. The higher contact stiffness and
                # virtual inertia keep the sampled mounting offset rigid under
                # the ~20N loads observed in the final 23mm sleeve.
                stiffness=1.0e4,
                damping=200.0,
                armature=1.0,
            )
            self.scene.robot.init_state.joint_pos["peg_mount_joint_.*"] = 0.0
        # Hybrid stabilization: OSC controls XYZ while a weak implicit joint
        # spring keeps the wrist near the known vertical posture. This avoids
        # the singular full-orientation Jacobian without leaving orientation
        # completely free (which previously let tilt grow toward 90 degrees).
        self.scene.robot.actuators["panda_shoulder"].stiffness = 20.0
        self.scene.robot.actuators["panda_shoulder"].damping = 4.0
        self.scene.robot.actuators["panda_forearm"].stiffness = 10.0
        self.scene.robot.actuators["panda_forearm"].damping = 2.0
        self.scene.robot.spawn.rigid_props.disable_gravity = True

        self.actions.arm_action = OperationalSpaceControllerActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller_cfg=OperationalSpaceControllerCfg(
                target_types=["pose_rel"],
                # The selected vertical wrist posture is close to a rotational
                # singularity. Control translation only for the baseline and
                # avoid a six-axis inverse/null-space projection here.
                motion_control_axes_task=(1, 1, 1, 0, 0, 0),
                impedance_mode="fixed",
                # Decouple translation and rotation. Only the well-conditioned
                # 3x3 translational operational inertia is used by the active
                # XYZ axes, avoiding the singular full 6x6 wrist inverse while
                # converting Cartesian acceleration into meaningful force.
                inertial_dynamics_decoupling=True,
                partial_inertial_dynamics_decoupling=True,
                gravity_compensation=False,
                # Translational stiffness must overcome the actuator damping
                # and move through the 50mm approach within one episode.
                motion_stiffness_task=(500.0, 500.0, 500.0, 40.0, 40.0, 40.0),
                motion_damping_ratio_task=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                nullspace_control="none",
                nullspace_stiffness=0.0,
            ),
            nullspace_joint_pos_target="none",
            # A 20mm command could jump from a valid insertion to below the
            # 40mm sleeve bottom in one control step.  Five millimetres keeps
            # contact and the accepted depth window observable while still
            # covering the 50mm approach in a small fraction of an episode.
            position_scale=0.005,   # ±5mm max per step
            orientation_scale=1.0,
            stiffness_scale=1.0,
        )


@configclass
class FrankaPegInHoleOscEnvCfg_PLAY(FrankaPegInHoleOscEnvCfg):
    """Play config: single env, rendering."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.0
        self.observations.policy.enable_corruption = False
