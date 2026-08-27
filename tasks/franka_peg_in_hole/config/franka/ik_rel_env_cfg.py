"""Franka peg-in-hole with Differential IK position control.

Two variants:
- FrankaPegInHoleRelEnvCfg: 3D position delta (recommended per review)
- FrankaPegInHoleRel6DEnvCfg: 6D pose delta (legacy, kept for reference)
"""

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import joint_pos_env_cfg


@configclass
class FrankaPegInHoleRelEnvCfg(joint_pos_env_cfg.FrankaPegInHoleEnvCfg):
    """Franka peg-in-hole with 3D position-only Differential IK.

    Action: (dx, dy, dz) in EE-local frame, scale=0.01 → max ±1cm/step.
    Orientation: fixed at current pose (hand naturally ~45° downward).
    """

    def __post_init__(self):
        super().__post_init__()

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="position",  # 3D position only
                use_relative_mode=True,   # Delta commands
                ik_method="dls",
            ),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=[0.0, 0.0, 0.107],
            ),
            scale=0.02,  # Max ±2cm per step (balanced exploration/precision)
        )


@configclass
class FrankaPegInHoleRel6DEnvCfg(joint_pos_env_cfg.FrankaPegInHoleEnvCfg):
    """Franka peg-in-hole with 6D pose Differential IK (legacy).

    Action: (dx, dy, dz, droll, dpitch, dyaw), scale=0.05 → max ±5cm/step.
    """

    def __post_init__(self):
        super().__post_init__()

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
            ),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=[0.0, 0.0, 0.107],
            ),
            scale=0.05,
        )


@configclass
class FrankaPegInHoleRelEnvCfg_PLAY(FrankaPegInHoleRelEnvCfg):
    """Play config: single env, rendering enabled, same MDP as training."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.0
        self.observations.policy.enable_corruption = False


@configclass
class FrankaPegInHoleRel6DEnvCfg_PLAY(FrankaPegInHoleRel6DEnvCfg):
    """Play config for legacy 6D control."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.0
        self.observations.policy.enable_corruption = False
