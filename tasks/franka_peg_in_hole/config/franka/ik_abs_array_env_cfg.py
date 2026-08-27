"""Franka Peg-in-Hole Array with IK absolute pose control."""

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import joint_pos_array_env_cfg


@configclass
class FrankaPegInHoleArrayEnvCfg(joint_pos_array_env_cfg.FrankaPegInHoleArrayEnvCfg):
    """Franka peg-in-hole array with IK absolute control."""

    def __post_init__(self):
        super().__post_init__()

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
            ),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=[0.0, 0.0, 0.107],
            ),
        )


@configclass
class FrankaPegInHoleArrayEnvCfg_PLAY(FrankaPegInHoleArrayEnvCfg):
    """Play mode: fewer envs, no noise."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.0
        self.observations.policy.enable_corruption = False
