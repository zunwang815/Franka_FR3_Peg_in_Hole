"""Physical mount-offset, workspace and hole-size curriculum configs."""

import math

from isaaclab.managers import EventTermCfg as EventTerm, RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .osc_baseline_env_cfg import OscBaseline30mmEnvCfg
from .osc_pose6d_env_cfg import OscPose6DBaselineEnvCfg


@configclass
class OscPegOffset2mmEnvCfg(OscBaseline30mmEnvCfg):
    """C1: 30mm fixed hole with independent +/-2mm physical peg mount offset."""

    def __post_init__(self):
        super().__post_init__()
        self.events.randomize_peg_mount = EventTerm(
            func=mdp.randomize_fixed_peg_mount,
            mode="reset",
            params={"x_range": (-0.002, 0.002), "y_range": (-0.002, 0.002)},
        )


@configclass
class OscPegOffset5mmEnvCfg(OscPegOffset2mmEnvCfg):
    """C4 mount component: teacher-specified independent +/-5mm offset."""

    def __post_init__(self):
        super().__post_init__()
        self.events.randomize_peg_mount.params["x_range"] = (-0.005, 0.005)
        self.events.randomize_peg_mount.params["y_range"] = (-0.005, 0.005)


@configclass
class OscHoleRandom10mmEnvCfg(OscPegOffset5mmEnvCfg):
    """C2: 30mm hole/fixture translated within a 20x20mm workspace."""

    def __post_init__(self):
        super().__post_init__()
        self.events.randomize_hole.params["x_range"] = (-0.010, 0.010)
        self.events.randomize_hole.params["y_range"] = (-0.010, 0.010)


@configclass
class OscHoleRandom50mmEnvCfg(OscPegOffset5mmEnvCfg):
    """C3: 30mm hole/fixture translated in the final 100x100mm workspace."""

    def __post_init__(self):
        super().__post_init__()
        self.events.randomize_hole.params["x_range"] = (-0.050, 0.050)
        self.events.randomize_hole.params["y_range"] = (-0.050, 0.050)


def _set_physical_hole_radius(cfg, hole_radius: float):
    """Move the 36 tangential sleeve segments to realize a true inner radius."""
    cx, cy, _ = cfg.scene.hole_board.init_state.pos
    segment_radial_thickness = 0.0020
    center_radius = hole_radius + segment_radial_thickness / 2.0
    for i in range(36):
        angle = 2.0 * math.pi * i / 36
        wall = getattr(cfg.scene, f"hole_wall_{i}")
        _, _, z = wall.init_state.pos
        wall.init_state.pos = (
            cx + center_radius * math.cos(angle),
            cy + center_radius * math.sin(angle),
            z,
        )


def _set_success_geometry(cfg, radial_tol: float, radial_gate: float):
    cfg.rewards.success_bonus.params["radial_tol"] = radial_tol
    cfg.terminations.success.params["radial_tol"] = radial_tol
    cfg.rewards.jam_penalty.params["radial_tol"] = radial_tol
    cfg.rewards.insertion_progress.params["radial_gate"] = radial_gate


def _enable_fixed_mount_hold(cfg):
    """Keep the reset-sampled mount offset fixed during each episode.

    This is intentionally opt-in so historical checkpoints retain their
    original dynamics.  The event runs once per control step, after physics,
    and removes the artificial prismatic-joint drift observed under contact.
    """
    cfg.events.hold_fixed_peg_mount = EventTerm(
        func=mdp.hold_fixed_peg_mount,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        is_global_time=False,
    )


def _enable_target_conditioned_init(cfg):
    """Initialize each reset from the offline target-conditioned pose bank."""
    cfg.events.target_conditioned_arm_pose = EventTerm(
        func=mdp.set_target_conditioned_bank_pose,
        mode="reset",
        params={},
    )


def _enable_online_target_conditioned_init(cfg):
    """Initialize each reset with height-locked online target IK."""
    cfg.events.target_conditioned_arm_pose_online = EventTerm(
        func=mdp.target_conditioned_arm_pose_online,
        mode="reset",
        params={"iterations": 15, "target_tip_z": 0.393592},
    )


def _enable_online_target_conditioned_init_offset5(cfg):
    """Online IK while retaining the physical offset5 residual geometry."""
    cfg.events.target_conditioned_arm_pose_online = EventTerm(
        func=mdp.target_conditioned_arm_pose_online,
        mode="reset",
        params={
            "iterations": 15,
            "target_tip_z": 0.393592,
            "preserve_mount_residual": True,
        },
    )


@configclass
class OscHole25mmEnvCfg(OscHoleRandom10mmEnvCfg):
    """C3: physical 25mm hole, +/-10mm fixture, +/-5mm peg mount."""

    def __post_init__(self):
        super().__post_init__()
        _set_physical_hole_radius(self, hole_radius=0.0125)
        _set_success_geometry(self, radial_tol=0.0023, radial_gate=0.0030)


@configclass
class OscHole23mmEnvCfg(OscHoleRandom50mmEnvCfg):
    """C4 geometry: physical 23mm hole and final randomization ranges."""

    def __post_init__(self):
        super().__post_init__()
        _set_physical_hole_radius(self, hole_radius=0.0115)
        _set_success_geometry(self, radial_tol=0.0013, radial_gate=0.0020)


# Strict-orientation curriculum. Keep this branch separate from the historical
# XYZ-only tasks so both controller generations remain reproducible.


@configclass
class OscPose6DPegOffset5mmEnvCfg(OscPose6DBaselineEnvCfg):
    """Pose6D C1: physical +/-5mm peg-mount uncertainty."""

    def __post_init__(self):
        super().__post_init__()
        self.events.randomize_peg_mount = EventTerm(
            func=mdp.randomize_fixed_peg_mount,
            mode="reset",
            params={"x_range": (-0.005, 0.005), "y_range": (-0.005, 0.005)},
        )


@configclass
class OscPose6DHoleRandom5mmEnvCfg(OscPose6DPegOffset5mmEnvCfg):
    """Pose6D hole-position curriculum: hole center translated within +/-5mm."""

    def __post_init__(self):
        super().__post_init__()
        self.events.randomize_hole.params["x_range"] = (-0.005, 0.005)
        self.events.randomize_hole.params["y_range"] = (-0.005, 0.005)


@configclass
class OscPose6DHoleRandom10mmEnvCfg(OscPose6DPegOffset5mmEnvCfg):
    """Pose6D C2: 30mm hole translated within +/-10mm."""

    def __post_init__(self):
        super().__post_init__()
        self.events.randomize_hole.params["x_range"] = (-0.010, 0.010)
        self.events.randomize_hole.params["y_range"] = (-0.010, 0.010)


@configclass
class OscPose6DHoleRandom10mmOuterMixEnvCfg(OscPose6DHoleRandom10mmEnvCfg):
    """Hole10 training distribution with 40% targeted outer-workspace samples."""

    def __post_init__(self):
        super().__post_init__()
        self.events.randomize_hole.params["outer_probability"] = 0.40
        self.events.randomize_hole.params["outer_min_abs"] = 0.007


@configclass
class OscPose6DHoleRandom15mmEnvCfg(OscPose6DPegOffset5mmEnvCfg):
    """Pose6D intermediate workspace: 30mm hole translated within +/-15mm."""

    def __post_init__(self):
        super().__post_init__()
        self.events.randomize_hole.params["x_range"] = (-0.015, 0.015)
        self.events.randomize_hole.params["y_range"] = (-0.015, 0.015)
        # Match hole20's gentler curriculum penalty. The predictive barrier
        # remains the primary protection against unsafe insertion depth.
        self.rewards.over_insertion_penalty.weight = -25.0


@configclass
class OscPose6DHoleRandom15mmMixEnvCfg(OscPose6DHoleRandom15mmEnvCfg):
    """Hole15 mix: 50% +/-10mm retention and 50% +/-15mm expansion."""

    def __post_init__(self):
        super().__post_init__()
        self.events.randomize_hole.params["inner_probability"] = 0.50
        self.events.randomize_hole.params["inner_abs"] = 0.010


@configclass
class OscPose6DHoleRandom20mmEnvCfg(OscPose6DPegOffset5mmEnvCfg):
    """Pose6D intermediate workspace: 30mm hole translated within +/-20mm."""

    def __post_init__(self):
        super().__post_init__()
        self.events.randomize_hole.params["x_range"] = (-0.020, 0.020)
        self.events.randomize_hole.params["y_range"] = (-0.020, 0.020)
        # A gentler terminal penalty avoids erasing the transferred insertion
        # policy while it adapts to the wider hole-position distribution.
        self.rewards.over_insertion_penalty.weight = -25.0


@configclass
class OscPose6DHoleRandom20mmMountStableEnvCfg(OscPose6DHoleRandom20mmEnvCfg):
    """Hole20 with static, reset-sampled mount uncertainty.

    The XY offset is still randomized in +/-5 mm, but the physical mount is
    held at that sampled value instead of being allowed to drift under contact.
    """

    def __post_init__(self):
        super().__post_init__()
        _enable_fixed_mount_hold(self)


@configclass
class OscPose6DHoleRandom20mmMountStableRewardEnvCfg(
    OscPose6DHoleRandom20mmMountStableEnvCfg
):
    """Mount-stable hole20 with broad-to-fine alignment reward shaping."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.multi_scale_alignment_progress = RewTerm(
            func=mdp.multi_scale_alignment_progress,
            weight=20.0,
            params={
                "broad_sigma": 0.015,
                "medium_sigma": 0.007,
                "fine_sigma": 0.0025,
                "broad_weight": 0.50,
                "medium_weight": 0.30,
                "fine_weight": 0.20,
                "hole_cfg": SceneEntityCfg("hole_board"),
            },
        )
        self.rewards.approach_depth_progress = RewTerm(
            func=mdp.approach_depth_progress,
            weight=30.0,
            params={
                "start_depth": -0.050,
                "gate_depth": -0.010,
                "hole_cfg": SceneEntityCfg("hole_board"),
            },
        )
        self.rewards.deep_insertion_braking = RewTerm(
            func=mdp.deep_insertion_braking_penalty,
            weight=-8.0,
            params={
                "start_depth": 0.030,
                "terminal_depth": 0.042,
                "hole_cfg": SceneEntityCfg("hole_board"),
            },
        )
        # The previous hole20 transfer branch weakened this penalty to -25.
        # For from-scratch reward ablation, restore a stronger terminal cost.
        self.rewards.over_insertion_penalty.weight = -40.0


@configclass
class OscPose6DHoleRandom20mmMountStableRewardMixEnvCfg(
    OscPose6DHoleRandom20mmMountStableRewardEnvCfg
):
    """Reward-shaped hole20 with inner-success retention during expansion."""

    def __post_init__(self):
        super().__post_init__()
        # Keep frequent ±5 mm examples so PPO sees successful alignment while
        # retaining 40% full ±20 mm samples for workspace expansion.
        self.events.randomize_hole.params["inner_probability"] = 0.60
        self.events.randomize_hole.params["inner_abs"] = 0.005


@configclass
class OscPose6DHoleRandom20mmMountStableRewardEdgeEnvCfg(
    OscPose6DHoleRandom20mmMountStableRewardEnvCfg
):
    """Edge-focused annulus curriculum for the wide XY workspace.

    Samples targets with radial offset 10--28 mm inside the original
    +/-20 mm square.  A small inner retention fraction is intentionally not
    added here: this stage is used to repair the policy's edge behavior after
    the offset5 transfer, while validation remains on the full distribution.
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.randomize_hole.params["radial_min"] = 0.010
        self.events.randomize_hole.params["radial_max"] = 0.028


@configclass
class OscPose6DHoleRandom20mmMountStableRewardEdgeAnchorEnvCfg(
    OscPose6DHoleRandom20mmMountStableRewardEdgeEnvCfg
):
    """Edge curriculum retaining a small inner-workspace anchor fraction."""

    def __post_init__(self):
        super().__post_init__()
        self.events.randomize_hole.params["inner_probability"] = 0.20
        self.events.randomize_hole.params["inner_abs"] = 0.005


@configclass
class OscPose6DHoleRandom20mmMountStableOnlineIKEnvCfg(
    OscPose6DHoleRandom20mmMountStableRewardEnvCfg
):
    """Reward-shaped hole20 with height-locked online target IK reset."""

    def __post_init__(self):
        super().__post_init__()
        _enable_online_target_conditioned_init(self)


@configclass
class OscPose6DHoleRandom20mmMountStableOnlineIKCanonicalEnvCfg(
    OscPose6DHoleRandom20mmMountStableOnlineIKEnvCfg
):
    """Online IK reset with offset5-compatible canonical proprioception."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.joint_pos.func = mdp.canonical_joint_pos
        self.observations.policy.joint_vel.func = mdp.canonical_joint_vel
        # Match the evaluator's canonical adapter, which replaces noisy joint
        # observations with exact canonical values after assembly.
        self.observations.policy.joint_pos.noise = None
        self.observations.policy.joint_vel.noise = None


@configclass
class OscPose6DHoleRandom20mmMountStableOnlineIKOffset5ResidualEnvCfg(
    OscPose6DHoleRandom20mmMountStableRewardEnvCfg
):
    """Online IK reset preserving the offset5 initial peg-hole residual."""

    def __post_init__(self):
        super().__post_init__()
        _enable_online_target_conditioned_init_offset5(self)


@configclass
class OscPose6DHoleRandom20mmMountStableIKEnvCfg(OscPose6DHoleRandom20mmMountStableEnvCfg):
    """Mount-stable hole20 plus target-conditioned reset initialization."""

    def __post_init__(self):
        super().__post_init__()
        _enable_target_conditioned_init(self)


@configclass
class OscPose6DHoleRandom30mmEnvCfg(OscPose6DPegOffset5mmEnvCfg):
    """Pose6D intermediate workspace: 30mm hole translated within +/-30mm."""

    def __post_init__(self):
        super().__post_init__()
        self.events.randomize_hole.params["x_range"] = (-0.030, 0.030)
        self.events.randomize_hole.params["y_range"] = (-0.030, 0.030)


@configclass
class OscPose6DHoleRandom50mmEnvCfg(OscPose6DPegOffset5mmEnvCfg):
    """Pose6D C3: 30mm hole translated within the final +/-50mm workspace."""

    def __post_init__(self):
        super().__post_init__()
        self.events.randomize_hole.params["x_range"] = (-0.050, 0.050)
        self.events.randomize_hole.params["y_range"] = (-0.050, 0.050)


@configclass
class OscPose6DHole25mmEnvCfg(OscPose6DHoleRandom10mmEnvCfg):
    """Pose6D C4a: physical 25mm hole with strict tilt acceptance."""

    def __post_init__(self):
        super().__post_init__()
        _set_physical_hole_radius(self, hole_radius=0.0125)
        _set_success_geometry(self, radial_tol=0.0023, radial_gate=0.0030)


@configclass
class OscPose6DHole23mmEnvCfg(OscPose6DHoleRandom50mmEnvCfg):
    """Pose6D C4b: final physical 23mm hole and full randomization."""

    def __post_init__(self):
        super().__post_init__()
        controller = self.actions.arm_action.controller_cfg
        # At the final 1.5mm radial clearance, a ~1deg contact-induced tilt of
        # the 100mm peg is enough to geometrically jam even though the separate
        # radial/tilt acceptance bounds are met. Increase angular impedance for
        # the contact phase while retaining the verified translational gains.
        controller.motion_stiffness_task = (
            500.0, 500.0, 500.0, 800.0, 800.0, 800.0
        )
        controller.motion_damping_ratio_task = (
            1.0, 1.0, 1.0, 1.5, 1.5, 1.5
        )
        _set_physical_hole_radius(self, hole_radius=0.0115)
        _set_success_geometry(self, radial_tol=0.0013, radial_gate=0.0020)
