"""Franka Peg-in-Hole base configuration using joint position control.

This is the base config that the IK config inherits from.
"""

from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from ...peg_in_hole_env_cfg import PegInHoleEnvCfg
from ... import mdp


@configclass
class FrankaPegInHoleEnvCfg(PegInHoleEnvCfg):
    """Franka peg-in-hole environment with joint position control.

    This is the base config. The IK config inherits from this and overrides
    the action space with Differential IK.
    """

    def __post_init__(self):
        super().__post_init__()

        # --- Scene setup ---
        # Franka Panda as the robot (FR3 is functionally equivalent)
        # Restored to original Franka default pose for reliable kinematics
        import torch
        init_joint_pos = torch.tensor(
            [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741, 0.04, 0.04],
            dtype=torch.float32,
        )
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "panda_joint1": init_joint_pos[0],
                    "panda_joint2": init_joint_pos[1],
                    "panda_joint3": init_joint_pos[2],
                    "panda_joint4": init_joint_pos[3],
                    "panda_joint5": init_joint_pos[4],
                    "panda_joint6": init_joint_pos[5],
                    "panda_joint7": init_joint_pos[6],
                    "panda_finger_joint1": init_joint_pos[7],
                    "panda_finger_joint2": init_joint_pos[8],
                },
            ),
        )

        # End-effector frame sensor (attached to panda_hand)
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="ee_tool",
                ),
            ],
        )

        # --- Action setup ---
        # Default: joint position control (will be overridden by IK config)
        self.actions.arm_action = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=0.5,
            use_default_offset=True,
        )
