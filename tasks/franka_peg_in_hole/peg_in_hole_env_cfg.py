"""Peg-in-Hole task environment configuration.

Phase 1: Single hole with random position in 10x10 cm area.
Peg: 2cm diameter cylinder, Hole: 2.3cm diameter.
Grasp uncertainty: ±0.5cm offset in x,y.
"""

from dataclasses import MISSING
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, CollisionPropertiesCfg, MassPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from isaaclab.envs.mdp import events as isaaclab_mdp_events


##
# Scene definition
##


@configclass
class PegInHoleSceneCfg(InteractiveSceneCfg):
    """Configuration for the peg-in-hole scene."""

    # Robot: Franka FR3 (using Panda as equivalent)
    robot: ArticulationCfg = MISSING

    # End-effector frame sensor
    ee_frame: FrameTransformerCfg = MISSING

    # Ground plane (well below table level so table is visibly above it)
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # Table (at 0.525m from Franka base for proper mounting alignment)
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.525, 0.0, 0.0),
            rot=(0.70711, 0.0, 0.0, 0.70711),
        ),
    )

    # ===== PEG (dynamic, synced to finger midpoint, high-mass anti-jitter) =====
    # Dynamic so write_root_state_to_sim works. High mass + damping minimize
    # contact-induced jitter: collision impulses barely move a 100kg peg.
    peg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Peg",
        spawn=sim_utils.CylinderCfg(
            radius=0.01,
            height=0.10,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=True,
                linear_damping=0.99,
                angular_damping=0.99,
                max_depenetration_velocity=5.0,
            ),
            mass_props=MassPropertiesCfg(
                mass=100.0,
            ),
            collision_props=CollisionPropertiesCfg(
                contact_offset=0.0005,
                rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.525, 0.0, 0.02),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # ===== ROUND HOLE: 12 cuboid pillars forming a ~2.3cm diameter ring =====
    # Hole center (0.525, 0, 0.02). Hole radius = 1.15cm.
    # 12 pillars at radius 1.4cm from center, 30° apart, each 5×5×20mm.
    # Inner edge at 1.15cm → gap between pillars < 1mm → peg (2cmφ) can't escape.
    # Kinematic so they stay fixed during robot interaction.

    hole_wall_0: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HoleWall0",
        spawn=sim_utils.CuboidCfg(size=(0.005, 0.005, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.1), roughness=0.3, metallic=0.1),
            rigid_props=RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
            collision_props=CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.53900, 0.00000, 0.02)))
    hole_wall_1: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HoleWall1",
        spawn=sim_utils.CuboidCfg(size=(0.005, 0.005, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.1), roughness=0.3, metallic=0.1),
            rigid_props=RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
            collision_props=CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.53712, 0.00700, 0.02)))
    hole_wall_2: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HoleWall2",
        spawn=sim_utils.CuboidCfg(size=(0.005, 0.005, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.1), roughness=0.3, metallic=0.1),
            rigid_props=RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
            collision_props=CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.53200, 0.01212, 0.02)))
    hole_wall_3: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HoleWall3",
        spawn=sim_utils.CuboidCfg(size=(0.005, 0.005, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.1), roughness=0.3, metallic=0.1),
            rigid_props=RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
            collision_props=CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.52500, 0.01400, 0.02)))
    hole_wall_4: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HoleWall4",
        spawn=sim_utils.CuboidCfg(size=(0.005, 0.005, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.1), roughness=0.3, metallic=0.1),
            rigid_props=RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
            collision_props=CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.51800, 0.01212, 0.02)))
    hole_wall_5: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HoleWall5",
        spawn=sim_utils.CuboidCfg(size=(0.005, 0.005, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.1), roughness=0.3, metallic=0.1),
            rigid_props=RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
            collision_props=CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.51288, 0.00700, 0.02)))
    hole_wall_6: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HoleWall6",
        spawn=sim_utils.CuboidCfg(size=(0.005, 0.005, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.1), roughness=0.3, metallic=0.1),
            rigid_props=RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
            collision_props=CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.51100, 0.00000, 0.02)))
    hole_wall_7: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HoleWall7",
        spawn=sim_utils.CuboidCfg(size=(0.005, 0.005, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.1), roughness=0.3, metallic=0.1),
            rigid_props=RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
            collision_props=CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.51288, -0.00700, 0.02)))
    hole_wall_8: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HoleWall8",
        spawn=sim_utils.CuboidCfg(size=(0.005, 0.005, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.1), roughness=0.3, metallic=0.1),
            rigid_props=RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
            collision_props=CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.51800, -0.01212, 0.02)))
    hole_wall_9: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HoleWall9",
        spawn=sim_utils.CuboidCfg(size=(0.005, 0.005, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.1), roughness=0.3, metallic=0.1),
            rigid_props=RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
            collision_props=CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.52500, -0.01400, 0.02)))
    hole_wall_10: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HoleWall10",
        spawn=sim_utils.CuboidCfg(size=(0.005, 0.005, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.1), roughness=0.3, metallic=0.1),
            rigid_props=RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
            collision_props=CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.53200, -0.01212, 0.02)))
    hole_wall_11: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HoleWall11",
        spawn=sim_utils.CuboidCfg(size=(0.005, 0.005, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.1), roughness=0.3, metallic=0.1),
            rigid_props=RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
            collision_props=CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.53712, -0.00700, 0.02)))

    # Invisible reference marker at hole center (for reward functions)
    hole_board: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HoleBoard",
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 0.01, 0.01),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.1)),
            rigid_props=RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
            collision_props=CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.525, 0.0, 0.02)),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTermCfg = MISSING
    gripper_action: ActionTermCfg | None = None


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP.

    Design (per review):
    - Joint state (q, q̇) for proprioception
    - Peg-to-hole vector in ROBOT BASE frame (no absolute world coords)
    - Peg tilt vector (captures orientation without noisy quaternion)
    - Last action for temporal context
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        # Exclude the hidden peg-mount DOFs: exposing their positions would
        # give the policy direct access to the simulated grasp uncertainty.
        joint_pos = ObsTerm(func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=["panda_joint.*", "panda_finger_joint.*"], preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=["panda_joint.*", "panda_finger_joint.*"], preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01))
        peg_to_hole_vec = ObsTerm(func=mdp.peg_to_hole_vector,
            noise=Unoise(n_min=-0.003, n_max=0.003),
            params={"hole_cfg": SceneEntityCfg("hole_board")})
        peg_tilt = ObsTerm(func=mdp.peg_tilt_vector,
            noise=Unoise(n_min=-0.01, n_max=0.01))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms — progress-based per review recommendations.

    r = w_progress * (prev_dist - cur_dist)          [encourages approach]
      + w_fine * exp(-(radial/0.003)²)               [precise alignment at 0-3mm]
      + w_insert * gate * insertion_progress          [gated: radial<2.5mm & tilt<2°]
      - w_jam * jam_contact                           [penalty for pushing walls]
      - w_tilt * tilt_penalty                         [penalty for excessive tilt]
      - w_time * 1                                    [small per-step cost]
      + w_success * success * 3000                    [dominant sparse bonus]
    """

    # Progress is potential-based: standing still cannot farm approach reward.
    approach_xy = None
    approach_xy_progress = RewTerm(func=mdp.approach_xy_progress, weight=80.0,
        params={"hole_cfg": SceneEntityCfg("hole_board")})
    fine_alignment = RewTerm(func=mdp.fine_alignment, weight=20.0,
        params={"sigma": 0.003, "hole_cfg": SceneEntityCfg("hole_board")})
    # Optional broad-to-fine potential used by the hole20 reward experiment.
    # Kept disabled in historical stages to preserve their checkpoints.
    multi_scale_alignment_progress = None
    approach_depth_progress = None
    insertion_progress = RewTerm(func=mdp.insertion_progress, weight=40.0,
        params={"threshold": 0.03, "radial_gate": 0.0025,
                "hole_cfg": SceneEntityCfg("hole_board")})
    jam_penalty = RewTerm(func=mdp.jam_penalty, weight=-8.0,
        params={"hole_cfg": SceneEntityCfg("hole_board"), "radial_tol": 0.0013})
    tilt_penalty = RewTerm(func=mdp.tilt_penalty, weight=-3.0,
        params={"tilt_tol": 2.0 * math.pi / 180.0})
    success_bonus = RewTerm(func=mdp.success_bonus, weight=500.0,
        params={"hole_cfg": SceneEntityCfg("hole_board"), "radial_tol": 0.0013,
                "depth_required": 0.030, "max_depth": 0.040,
                "tilt_tol": 2.0 * math.pi / 180.0})
    over_insertion_penalty = RewTerm(func=mdp.over_insertion_penalty, weight=-100.0,
        params={"hole_cfg": SceneEntityCfg("hole_board"), "max_depth": 0.042})
    deep_insertion_braking = None
    time_penalty = RewTerm(func=mdp.time_penalty, weight=-0.05)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.00005,
        params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    success = DoneTerm(func=mdp.success_insertion,
        params={"hole_cfg": SceneEntityCfg("hole_board"), "radial_tol": 0.0013,
                "depth_required": 0.030, "max_depth": 0.040,
                "tilt_tol": 2.0 * math.pi / 180.0})

    over_insertion = DoneTerm(func=mdp.over_insertion,
        params={"hole_cfg": SceneEntityCfg("hole_board"), "max_depth": 0.042})

    # Terminate if peg drifts more than 8cm from hole in XY
    peg_left_cylinder = DoneTerm(func=mdp.peg_left_cylinder,
        params={"radius": 0.15, "hole_cfg": SceneEntityCfg("hole_board")})


@configclass
class EventCfg:
    """Configuration for events (reset-time randomization).

    ORDER MATTERS: Isaac Lab executes events in config declaration order.
    1. randomize_hole FIRST (move hole + walls to random position)
    2. reset_robot_joints SECOND (reset robot to default forward pose)
    """

    # Step 1: Randomize hole position in 10x10 cm workspace
    randomize_hole = EventTerm(
        func=mdp.reset_hole_position_uniform,
        mode="reset",
        params={
            "x_range": (-0.05, 0.05),
            "y_range": (-0.05, 0.05),
            "table_z": 0.025,
            "asset_cfg": SceneEntityCfg("hole_board"),
        },
    )

    # Step 2: Reset robot to forward-reaching default pose (modest perturbation)
    reset_robot_joints = EventTerm(
        func=isaaclab_mdp_events.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.95, 1.05),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Step 3: initialize implicit actuator targets after writing joint state.
    # Required by the OSC baseline's weak posture-holding joint springs.
    set_joint_targets = EventTerm(
        func=mdp.set_joint_position_targets_to_default,
        mode="reset",
        params={"robot_cfg": SceneEntityCfg("robot")},
    )

    # Interval: Sync peg rigid body to finger midpoint each physics step
    sync_peg_pose = EventTerm(
        func=mdp.sync_peg_to_ee,
        mode="interval",
        interval_range_s=(0.0167, 0.0167),
        params={
            "peg_offset": (0.0, 0.0, 0.07),
            "ee_body_name": "panda_hand",
            "robot_cfg": SceneEntityCfg("robot"),
            "peg_cfg": SceneEntityCfg("peg"),
        },
    )


##
# Environment configuration
##


@configclass
class PegInHoleEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the peg-in-hole insertion environment (Phase 1)."""

    # Scene settings
    scene: PegInHoleSceneCfg = PegInHoleSceneCfg(num_envs=2048, env_spacing=2.0)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.decimation = 2           # Decimation factor for rendering
        self.episode_length_s = 8.0  # 8s = ~240 steps per episode, more time to insert

        # Simulation settings
        self.sim.dt = 1.0 / 60.0      # 60Hz physics
        self.sim.render_interval = self.decimation
        self.sim.use_fabric = False    # Disable Fabric PhysX (version mismatch)

        # PhysX settings for contact-rich tasks
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_correlation_distance = 0.005
        self.sim.physx.gpu_max_rigid_contact_count = 2**23

        # Viewer settings
        self.viewer.eye = (1.5, 1.5, 1.0)
        self.viewer.lookat = (0.625, 0.0, 0.15)
