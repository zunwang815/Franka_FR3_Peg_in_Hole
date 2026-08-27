"""Peg-in-Hole Phase 2: Six-hole array environment configuration.

2x3 hole array, 3cm spacing between hole centers.
Entire array randomly placed in 10x10 cm workspace.
Environment specifies which hole (0-5) is the target.
"""

import math
from dataclasses import MISSING

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
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, CollisionPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from isaaclab.envs.mdp import events as isaaclab_mdp_events


@configclass
class PegInHoleArraySceneCfg(InteractiveSceneCfg):
    """Configuration for the six-hole array scene."""

    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.45, 0.0, 0.0),
            rot=(0.70711, 0.0, 0.0, 0.70711),
        ),
    )

    # ===== PEG (separate RigidObject, synced to finger midpoint) =====
    peg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Peg",
        spawn=sim_utils.CylinderCfg(
            radius=0.01,
            height=0.10,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            collision_props=CollisionPropertiesCfg(
                collision_enabled=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # The verified upright OSC posture places the peg tip near 0.56m;
            # keep the six-hole fixture 60mm below that pose for approach.
            pos=(0.254, 0.0, 0.493),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # Six-hole board: block with 6 holes in 2x3 grid
    hole_board: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HoleBoard",
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 0.01, 0.01),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.5)),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=True,
            ),
            collision_props=CollisionPropertiesCfg(
                collision_enabled=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # Geometry helpers define the fixture surface as hole_z + 10mm.
            pos=(0.254, 0.0, 0.493),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # Six physical sleeves. The array origin is randomized at reset while each
    # sleeve keeps its fixed 2x3, 30mm-center-spacing layout.
    for _array_hole_index in range(6):
        _array_col = _array_hole_index % 3
        _array_row = _array_hole_index // 3
        _array_dx = (_array_col - 1.0) * 0.03
        _array_dy = (_array_row - 0.5) * 0.03
        for _wall_index in range(36):
            _angle = 2.0 * math.pi * _wall_index / 36.0
            _center_radius = 0.0125  # 11.5mm inner radius + 1mm half thickness
            locals()[f"array_hole_{_array_hole_index}_wall_{_wall_index}"] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/ArrayHole{_array_hole_index}Wall{_wall_index}",
            spawn=sim_utils.CuboidCfg(
                size=(0.0020, 0.0028, 0.040),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.8, 0.6, 0.1), roughness=0.35
                ),
                rigid_props=RigidBodyPropertiesCfg(
                    disable_gravity=True, kinematic_enabled=True
                ),
                collision_props=CollisionPropertiesCfg(
                    contact_offset=0.0005, rest_offset=0.0
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(
                    0.254 + _array_dx + _center_radius * math.cos(_angle),
                    _array_dy + _center_radius * math.sin(_angle),
                    0.483,
                ),
                rot=(math.cos(_angle / 2.0), 0.0, 0.0, math.sin(_angle / 2.0)),
            ),
        )

    del _array_hole_index, _array_col, _array_row, _array_dx, _array_dy
    del _wall_index, _angle, _center_radius

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ArrayActionsCfg:
    """Action specifications (same as phase 1)."""

    arm_action: ActionTermCfg = MISSING
    gripper_action: ActionTermCfg | None = None


@configclass
class ArrayObservationsCfg:
    """Observation specifications with added goal hole information."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*", "panda_finger_joint.*"])},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*", "panda_finger_joint.*"])},
        )
        ee_position = ObsTerm(
            func=mdp.ee_position,
            noise=Unoise(n_min=-0.003, n_max=0.003),
        )
        ee_orientation = ObsTerm(
            func=mdp.ee_orientation,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        peg_to_hole_vec = ObsTerm(
            func=mdp.peg_to_hole_vector,
            noise=Unoise(n_min=-0.003, n_max=0.003),
        )
        peg_tilt = ObsTerm(
            func=mdp.peg_tilt_vector,
            noise=Unoise(n_min=-0.004, n_max=0.004),
        )
        hole_position = ObsTerm(
            func=mdp.hole_position,
            noise=Unoise(n_min=-0.003, n_max=0.003),
        )
        # Goal hole one-hot encoding (6 holes)
        hole_id = ObsTerm(func=mdp.hole_id_onehot, params={"num_holes": 6})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class ArrayRewardsCfg:
    """Reward terms with additional penalty for wrong hole."""

    approach_xy = RewTerm(
        func=mdp.approach_xy,
        weight=2.0,
        params={
            "hole_cfg": SceneEntityCfg("hole_board"),
        },
    )
    alignment = RewTerm(
        func=mdp.fine_alignment,
        weight=20.0,
        params={
            "sigma": 0.003,
            "hole_cfg": SceneEntityCfg("hole_board"),
        },
    )
    insertion_depth = RewTerm(
        func=mdp.insertion_progress,
        weight=40.0,
        params={
            "threshold": 0.03,
            "radial_gate": 0.0025,
            "hole_cfg": SceneEntityCfg("hole_board"),
        },
    )
    success_bonus = RewTerm(
        func=mdp.success_bonus,
        weight=100.0,
        params={
            "radial_tol": 0.0013,
            "depth_required": 0.015,
            "max_depth": 0.040,
            "tilt_tol": 2.0 * 3.141592653589793 / 180.0,
            "hole_cfg": SceneEntityCfg("hole_board"),
        },
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class ArrayTerminationsCfg:
    """Termination conditions for array task."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=mdp.success_insertion,
        params={
            "radial_tol": 0.0013,
            "depth_required": 0.015,
            "max_depth": 0.040,
            "tilt_tol": 2.0 * 3.141592653589793 / 180.0,
            "hole_cfg": SceneEntityCfg("hole_board"),
        },
    )
    workspace_violation = DoneTerm(
        func=mdp.workspace_violation,
        params={
            "bounds": (0.3, 0.3, 0.4),
            "hole_cfg": SceneEntityCfg("hole_board"),
        },
    )
    joint_limits = DoneTerm(
        func=mdp.joint_limits_violation,
        params={"margin": 0.02, "asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class ArrayEventCfg:
    """Events with array-wide position randomization."""

    reset_robot_joints = EventTerm(
        func=isaaclab_mdp_events.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    select_target_hole = EventTerm(
        func=mdp.select_target_hole,
        mode="reset",
        params={"num_holes": 6, "target_hole_id": -1},
    )

    # Randomize the entire hole board position in 10x10 cm area
    randomize_hole = EventTerm(
        func=mdp.reset_hole_position_uniform,
        mode="reset",
        params={
            "x_range": (-0.05, 0.05),
            "y_range": (-0.05, 0.05),
            "table_z": 0.493,
            "asset_cfg": SceneEntityCfg("hole_board"),
            "array_hole_spacing": 0.03,
            "array_columns": 3,
            "array_rows": 2,
        },
    )

    # Sync peg rigid body to end-effector every physics step
    sync_peg_pose = EventTerm(
        func=mdp.sync_peg_to_ee,
        mode="interval",
        interval_range_s=(0.0167, 0.0167),
        params={
            "peg_offset": (0.0, 0.0, -0.10),
            "ee_body_name": "panda_hand",
            "robot_cfg": SceneEntityCfg("robot"),
            "peg_cfg": SceneEntityCfg("peg"),
        },
    )

    # Align the standalone visual/physics peg immediately after reset.  This
    # removes a one-step initialization mismatch before a controller reads
    # the peg pose; the interval event keeps it synchronized thereafter.
    sync_peg_pose_reset = EventTerm(
        func=mdp.sync_peg_to_ee,
        mode="reset",
        params={
            "peg_offset": (0.0, 0.0, -0.10),
            "ee_body_name": "panda_hand",
            "robot_cfg": SceneEntityCfg("robot"),
            "peg_cfg": SceneEntityCfg("peg"),
        },
    )


@configclass
class PegInHoleArrayEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the peg-in-hole array insertion environment (Phase 2)."""

    scene: PegInHoleArraySceneCfg = PegInHoleArraySceneCfg(num_envs=2048, env_spacing=2.0)
    observations: ArrayObservationsCfg = ArrayObservationsCfg()
    actions: ArrayActionsCfg = ArrayActionsCfg()
    rewards: ArrayRewardsCfg = ArrayRewardsCfg()
    terminations: ArrayTerminationsCfg = ArrayTerminationsCfg()
    events: ArrayEventCfg = ArrayEventCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 12.0  # Longer episodes for harder task
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        self.sim.use_fabric = False    # Disable Fabric PhysX (version mismatch)
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_correlation_distance = 0.005
        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.viewer.eye = (1.5, 1.5, 1.0)
        self.viewer.lookat = (0.254, 0.0, 0.45)
