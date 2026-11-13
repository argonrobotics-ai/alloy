# SPDX-License-Identifier: Apache-2.0
"""``alloy.load`` / ``Scene.load`` / ``Scene.add_robot`` on the files people pass them (CPU).

A USD package (.usdz) opened in ``alloy.io.import_file`` and ``alloy.create`` (whose error sends robots to
``Scene.add_robot``), but ``add_robot`` refused the suffix, and ``alloy.load`` read any file with a suffix it did not
know as UTF-8 YAML: a .usdz or a binary STL raised a bare UnicodeDecodeError. Now a .usdz robot loads as any USD
robot, a mesh file gets a SceneError naming ``scene.add_mesh`` / ``alloy.create``, and any other file one listing the
kinds ``alloy.load`` reads.
"""

from __future__ import annotations

import struct

import pytest

import alloy.api as alloy


def _binary_stl(path):
    """One triangle as a binary STL (a float in its header region is not UTF-8)."""
    tri = struct.pack("<12fH", 0, 0, 1, 0, 0, 0, 0.1, 0, 0, 0, 0.1, 0, 0)
    path.write_bytes(b"\x80" * 80 + struct.pack("<I", 1) + tri)
    return path


@pytest.mark.parametrize("call", [alloy.load, alloy.Scene.load])
def test_a_mesh_file_names_the_calls_that_take_it(tmp_path, call):
    with pytest.raises(alloy.SceneError) as err:
        call(_binary_stl(tmp_path / "part.stl"))
    assert "is a mesh" in str(err.value) and "add_mesh" in err.value.fix and "alloy.create" in err.value.fix
    assert err.value.field == "source"


def test_an_unknown_or_binary_file_lists_what_load_reads(tmp_path):
    (tmp_path / "blob.bin").write_bytes(b"\x80\x81\x82")
    with pytest.raises(alloy.SceneError) as err:
        alloy.Scene.load(tmp_path / "blob.bin")
    assert "unknown suffix '.bin'" in str(err.value) and ".yaml" in err.value.fix and ".usdz" in err.value.fix
    (tmp_path / "scene.yaml").write_bytes(b"\x80\x81\x82")
    with pytest.raises(alloy.SceneError, match="not a text file"):
        alloy.Scene.load(tmp_path / "scene.yaml")
    (tmp_path / "scene.json").write_text("{not json")
    with pytest.raises(alloy.SceneError, match="not a Scene JSON file"):
        alloy.Scene.load(tmp_path / "scene.json")
    (tmp_path / "bad.yaml").write_text("schema: [unclosed\n")
    with pytest.raises(alloy.SceneError, match="not a readable scene card"):
        alloy.Scene.load(tmp_path / "bad.yaml")


def test_a_usdz_robot_is_taken_as_a_usd_robot(tmp_path):
    usdz = tmp_path / "arm.usdz"
    usdz.write_bytes(b"PK\x03\x04")                      # the suffix decides the importer; the build reads the file
    scene = alloy.Scene()
    scene.add_robot("r", usdz)
    assert scene.items["r"]["robot_kind"] == "usd"
    loaded = alloy.Scene.load(usdz)
    assert loaded.items["arm"]["robot_kind"] == "usd" and "ground" in loaded.items


def _usdz_arm(tmp_path):
    """A one-hinge arm with a drive, packaged as .usdz (pxr from the usd extra)."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

    usda = tmp_path / "arm.usda"
    stage = Usd.Stage.CreateNew(str(usda))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    for name, size, x, mass in (("Base", 0.1, 0.0, 1.0), ("Link", 0.06, 0.1, 0.3)):
        cube = UsdGeom.Cube.Define(stage, f"/Root/{name}")
        cube.CreateSizeAttr(size)
        cube.AddTranslateOp().Set(Gf.Vec3d(x, 0, 0.2))
        UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(mass)
    weld = UsdPhysics.FixedJoint.Define(stage, "/Root/Weld")
    weld.CreateBody1Rel().SetTargets(["/Root/Base"])
    weld.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0.2))
    hinge = UsdPhysics.RevoluteJoint.Define(stage, "/Root/J1")
    hinge.CreateBody0Rel().SetTargets(["/Root/Base"])
    hinge.CreateBody1Rel().SetTargets(["/Root/Link"])
    hinge.CreateAxisAttr("Y")
    hinge.CreateLocalPos0Attr(Gf.Vec3f(0.05, 0, 0))
    hinge.CreateLocalPos1Attr(Gf.Vec3f(-0.05, 0, 0))
    drive = UsdPhysics.DriveAPI.Apply(hinge.GetPrim(), "angular")
    drive.CreateStiffnessAttr(100.0)
    drive.CreateDampingAttr(10.0)
    stage.GetRootLayer().Save()
    out = tmp_path / "arm.usdz"
    UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(usda)), str(out))
    return out


@pytest.mark.compile_heavy
def test_a_usdz_robot_builds_and_follows_a_target(tmp_path):
    pytest.importorskip("pxr")
    pytest.importorskip("newton_usd_schemas")
    sim = alloy.load(_usdz_arm(tmp_path), device="cpu")
    sim.run(0.2, targets={"arm/J1": 0.4})
    assert float(sim.state(world=0)["arm/J1.q"]) == pytest.approx(0.35, abs=0.05)
    assert sim.check()["ok"]
