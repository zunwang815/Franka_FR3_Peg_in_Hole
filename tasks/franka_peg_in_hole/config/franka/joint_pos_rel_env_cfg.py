"""Franka peg-in-hole with DIRECT joint position control.

Policy outputs TARGET joint positions (9D). The PD controller tracks them.
This is the simplest, most common RL control mode for manipulation.
"""

from isaaclab.utils import configclass

from . import joint_pos_env_cfg


@configclass
class FrankaPegInHoleJointEnvCfg(joint_pos_env_cfg.FrankaPegInHoleEnvCfg):
    """Franka peg-in-hole with joint position control (recommended for RL)."""

    def __post_init__(self):
        super().__post_init__()
        # joint_pos_env_cfg already sets self.actions.arm_action = JointPositionActionCfg
        # Increase action scale for visible motion
        self.actions.arm_action.scale = 1.0  # Full range


@configclass
class FrankaPegInHoleJointEnvCfg_PLAY(FrankaPegInHoleJointEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.0
        self.observations.policy.enable_corruption = False
