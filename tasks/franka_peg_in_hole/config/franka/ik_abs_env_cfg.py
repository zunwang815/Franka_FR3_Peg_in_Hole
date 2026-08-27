"""Franka-specific peg-in-hole environment with IK absolute control."""

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import joint_pos_env_cfg


@configclass
class FrankaPegInHoleEnvCfg(joint_pos_env_cfg.FrankaPegInHoleEnvCfg):
    """Franka peg-in-hole with Differential IK absolute pose control."""

    def __post_init__(self):
        super().__post_init__()

        # Use IK controller for precise end-effector control
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
                pos=[0.0, 0.0, 0.107],  # Offset from hand to tool center point
            ),
        )


@configclass
class FrankaPegInHoleEnvCfg_PLAY(FrankaPegInHoleEnvCfg):
    """Configuration for evaluation/play with fewer envs and no domain randomization."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.0
        self.observations.policy.enable_corruption = False
