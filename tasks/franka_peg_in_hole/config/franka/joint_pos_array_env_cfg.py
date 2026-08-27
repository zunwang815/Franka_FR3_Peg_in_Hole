"""Franka Peg-in-Hole Array base configuration (joint position control)."""

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from ...peg_in_hole_array_env_cfg import PegInHoleArrayEnvCfg


@configclass
class FrankaPegInHoleArrayEnvCfg(PegInHoleArrayEnvCfg):
    """Franka peg-in-hole array environment (base)."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
        )

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="ee_tool",
                ),
            ],
        )

        self.actions.arm_action = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=0.5,
            use_default_offset=True,
        )
