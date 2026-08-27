#!/usr/bin/env python3
"""Build a local Panda development asset with a physically fixed peg link.

The generated USD references Isaac Lab's Panda asset and adds a cylindrical
rigid body connected to ``panda_hand`` by a USD Physics FixedJoint.  It is a
development bridge until the official FR3 USD is imported; unlike the legacy
``sync_peg_to_ee`` event, it participates in articulation/contact dynamics.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "assets" / "panda_with_fixed_peg.usda"),
    )
    parser.add_argument("--headless", action="store_true", default=True)
    return parser.parse_args()


args = parse_args()
app = AppLauncher(headless=args.headless).app

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG


def main():
    source = FRANKA_PANDA_CFG.spawn.usd_path
    source_stage = Usd.Stage.Open(source)
    if source_stage is None:
        raise RuntimeError(f"Unable to open Panda USD: {source}")
    source_default = source_stage.GetDefaultPrim()
    if not source_default:
        raise RuntimeError(f"Panda USD has no default prim: {source}")

    hand_matches = [
        p for p in source_stage.Traverse()
        if p.GetName() == "panda_hand"
    ]
    if len(hand_matches) != 1:
        raise RuntimeError(f"Expected one panda_hand prim, found {len(hand_matches)}")
    hand_rel = hand_matches[0].GetPath().MakeRelativePath(source_default.GetPath())

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Rebuilding is intentional: this generated asset is reproducible and
    # should reflect the current mount calibration.
    if output.exists():
        output.unlink()
    stage = Usd.Stage.CreateNew(str(output))
    robot = UsdGeom.Xform.Define(stage, "/Robot").GetPrim()
    robot.GetReferences().AddReference(source, source_default.GetPath())
    stage.SetDefaultPrim(robot)

    # Calibrated against the articulation body state used by Isaac Lab.  A
    # 128mm joint-frame offset places the distal peg tip at about z=0.418m for
    # the configured vertical posture, i.e. 50mm above the fixture surface.
    mount_x_path = Sdf.Path("/Robot/peg_mount_x")
    mount_y_path = Sdf.Path("/Robot/peg_mount_y")
    for mount_path in (mount_x_path, mount_y_path):
        mount = UsdGeom.Xform.Define(stage, mount_path).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(mount)
        UsdPhysics.MassAPI.Apply(mount).CreateMassAttr(0.001)

    peg_path = Sdf.Path("/Robot/peg")
    peg = UsdGeom.Cylinder.Define(stage, peg_path)
    peg.CreateAxisAttr("Z")
    peg.CreateRadiusAttr(0.010)
    peg.CreateHeightAttr(0.100)
    peg.CreateDisplayColorAttr([Gf.Vec3f(0.8, 0.15, 0.15)])
    UsdPhysics.CollisionAPI.Apply(peg.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(peg.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(peg.GetPrim())
    mass.CreateMassAttr(0.25)

    hand_path = Sdf.Path("/Robot").AppendPath(hand_rel)
    joint_x = UsdPhysics.PrismaticJoint.Define(stage, "/Robot/peg_mount_joint_x")
    joint_x.CreateBody0Rel().SetTargets([hand_path])
    joint_x.CreateBody1Rel().SetTargets([mount_x_path])
    joint_x.CreateAxisAttr("X")
    joint_x.CreateLowerLimitAttr(-0.005)
    joint_x.CreateUpperLimitAttr(0.005)
    joint_x.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.128))
    joint_x.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint_x.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint_x.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    joint_y = UsdPhysics.PrismaticJoint.Define(stage, "/Robot/peg_mount_joint_y")
    joint_y.CreateBody0Rel().SetTargets([mount_x_path])
    joint_y.CreateBody1Rel().SetTargets([mount_y_path])
    joint_y.CreateAxisAttr("Y")
    joint_y.CreateLowerLimitAttr(-0.005)
    joint_y.CreateUpperLimitAttr(0.005)

    joint = UsdPhysics.FixedJoint.Define(stage, "/Robot/peg_fixed_joint")
    joint.CreateBody0Rel().SetTargets([mount_y_path])
    joint.CreateBody1Rel().SetTargets([peg_path])

    stage.GetRootLayer().Save()
    print(f"Built: {output}")
    print(f"Source: {source}")
    print(f"Hand body: {hand_path}")
    print("Peg: radius=10mm height=100mm mass=0.25kg nominal_offset_z=128mm")
    print("Mount DOFs: peg_mount_joint_x/y limits=-5..+5mm")


try:
    main()
finally:
    app.close()
