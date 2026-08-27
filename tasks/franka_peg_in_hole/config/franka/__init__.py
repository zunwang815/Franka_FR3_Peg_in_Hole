"""Register Franka Peg-in-Hole environments with gym."""

import gymnasium as gym

from . import agents

##
# Phase 1: Single hole
##

gym.register(
    id="Isaac-PegInHole-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaPegInHoleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPegInHolePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PegInHole-Franka-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:FrankaPegInHoleEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPegInHolePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PegInHole-Franka-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:FrankaPegInHoleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPegInHolePPORunnerCfg",
    },
)

##
# Inverse Kinematics - Relative Pose Control (delta commands)
##

gym.register(
    id="Isaac-PegInHole-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:FrankaPegInHoleRelEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPegInHoleRelPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PegInHole-Franka-IK-Rel-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:FrankaPegInHoleRelEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPegInHolePPORunnerCfg",
    },
)

##
# OSC: Operational Space Controller (3D position, singularity-free)
##

gym.register(
    id="Isaac-PegInHole-Franka-OSC-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.osc_env_cfg:FrankaPegInHoleOscEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPegInHoleRelPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PegInHole-Franka-OSC-Baseline-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "tasks.franka_peg_in_hole.osc_baseline_env_cfg:OscBaseline30mmEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPegInHoleRelPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PegInHole-Franka-OSC-Pose6D-Baseline-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "tasks.franka_peg_in_hole.osc_pose6d_env_cfg:OscPose6DBaselineEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPegInHoleRelPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PegInHole-Franka-OSC-PegOffset2mm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "tasks.franka_peg_in_hole.osc_curriculum_env_cfg:OscPegOffset2mmEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPegInHoleRelPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PegInHole-Franka-OSC-PegOffset5mm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "tasks.franka_peg_in_hole.osc_curriculum_env_cfg:OscPegOffset5mmEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPegInHoleRelPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PegInHole-Franka-OSC-HoleRandom10mm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "tasks.franka_peg_in_hole.osc_curriculum_env_cfg:OscHoleRandom10mmEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPegInHoleRelPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PegInHole-Franka-OSC-HoleRandom50mm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "tasks.franka_peg_in_hole.osc_curriculum_env_cfg:OscHoleRandom50mmEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPegInHoleRelPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PegInHole-Franka-OSC-Hole25mm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "tasks.franka_peg_in_hole.osc_curriculum_env_cfg:OscHole25mmEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPegInHoleRelPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PegInHole-Franka-OSC-Hole23mm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "tasks.franka_peg_in_hole.osc_curriculum_env_cfg:OscHole23mmEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPegInHoleRelPPORunnerCfg",
    },
)

for task_id, cfg_class in (
    ("Isaac-PegInHole-Franka-OSC-Pose6D-PegOffset5mm-v0", "OscPose6DPegOffset5mmEnvCfg"),
    ("Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom5mm-v0", "OscPose6DHoleRandom5mmEnvCfg"),
    ("Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom10mm-v0", "OscPose6DHoleRandom10mmEnvCfg"),
    ("Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom10mmOuterMix-v0", "OscPose6DHoleRandom10mmOuterMixEnvCfg"),
    ("Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom15mm-v0", "OscPose6DHoleRandom15mmEnvCfg"),
    ("Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom15mmMix-v0", "OscPose6DHoleRandom15mmMixEnvCfg"),
    ("Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom20mm-v0", "OscPose6DHoleRandom20mmEnvCfg"),
    ("Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom20mm-MountStable-v0", "OscPose6DHoleRandom20mmMountStableEnvCfg"),
    ("Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom20mm-MountStableReward-v0", "OscPose6DHoleRandom20mmMountStableRewardEnvCfg"),
    ("Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom20mm-MountStableRewardMix-v0", "OscPose6DHoleRandom20mmMountStableRewardMixEnvCfg"),
    ("Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom20mm-MountStableRewardEdge-v0", "OscPose6DHoleRandom20mmMountStableRewardEdgeEnvCfg"),
    ("Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom20mm-MountStableRewardEdgeAnchor-v0", "OscPose6DHoleRandom20mmMountStableRewardEdgeAnchorEnvCfg"),
    ("Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom20mm-MountStableOnlineIK-v0", "OscPose6DHoleRandom20mmMountStableOnlineIKEnvCfg"),
    ("Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom20mm-MountStableOnlineIKCanonical-v0", "OscPose6DHoleRandom20mmMountStableOnlineIKCanonicalEnvCfg"),
    ("Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom20mm-MountStableOnlineIKOffset5Residual-v0", "OscPose6DHoleRandom20mmMountStableOnlineIKOffset5ResidualEnvCfg"),
    ("Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom20mm-MountStableIK-v0", "OscPose6DHoleRandom20mmMountStableIKEnvCfg"),
    ("Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom30mm-v0", "OscPose6DHoleRandom30mmEnvCfg"),
    ("Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom50mm-v0", "OscPose6DHoleRandom50mmEnvCfg"),
    ("Isaac-PegInHole-Franka-OSC-Pose6D-Hole25mm-v0", "OscPose6DHole25mmEnvCfg"),
    ("Isaac-PegInHole-Franka-OSC-Pose6D-Hole23mm-v0", "OscPose6DHole23mmEnvCfg"),
):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                "tasks.franka_peg_in_hole.osc_curriculum_env_cfg:" + cfg_class
            ),
            "rsl_rl_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPegInHoleRelPPORunnerCfg"
            ),
        },
    )

##
# Baseline: Simplified 30mm hole (Phase 2 of review roadmap)
##

gym.register(
    id="Isaac-PegInHole-Franka-IK-Rel-Baseline-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "tasks.franka_peg_in_hole.baseline_env_cfg:Baseline30mmEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPegInHoleRelPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PegInHole-Franka-IK-Rel-Baseline-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "tasks.franka_peg_in_hole.baseline_env_cfg:Baseline30mmEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPegInHolePPORunnerCfg",
    },
)

##
# Phase 2: Six-hole array
##

gym.register(
    id="Isaac-PegInHoleArray-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_array_env_cfg:FrankaPegInHoleArrayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPegInHolePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PegInHoleArray-Franka-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_abs_array_env_cfg:FrankaPegInHoleArrayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPegInHolePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PegInHoleArray-Franka-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_abs_array_env_cfg:FrankaPegInHoleArrayEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPegInHolePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PegInHoleArray-Franka-OSC-Pose6D-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.osc_array_env_cfg:FrankaPegInHoleArrayOscPose6DEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPegInHoleRelPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PegInHoleArray-Franka-OSC-Pose6D-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.osc_array_env_cfg:FrankaPegInHoleArrayOscPose6DEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPegInHoleRelPPORunnerCfg",
    },
)
