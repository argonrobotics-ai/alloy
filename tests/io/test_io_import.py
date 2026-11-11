# SPDX-License-Identifier: Apache-2.0
"""Imports: URDF (own reader), MJCF and USD (Newton's readers) into IOModels and cards, and card round trips
(cards -> file -> cards keep shapes, masses, friction and placements)."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from alloy.cards.library import CardLibrary, default_root
from alloy.cards.loader import load_object
from alloy.core.registry import isolated_registry
from alloy.io import export_object, export_scene, import_file
from alloy.io.cli import json_of, main as cli_main
from alloy.io.geometry import compose, quat_matrix
from alloy.io.importer import ImportFileError, read_file
from alloy.io.urdf_read import read_urdf
from alloy.io.urdf_write import write_urdf
from alloy.testing.io import arm_model, ramp_library

mujoco = pytest.importorskip("mujoco")


@pytest.fixture(autouse=True)
def _registry():
    with isolated_registry(empty=True):
        yield


def _same_pose(a, b, tol=1e-5):
    np.testing.assert_allclose(a[:3], b[:3], atol=tol)
    np.testing.assert_allclose(quat_matrix(a[3:]), quat_matrix(b[3:]), atol=tol)


def test_urdf_round_trip_keeps_poses_masses_limits(tmp_path):
    src = arm_model()
    index = write_urdf(src, tmp_path)
    g = index["groups"][0]
    io = read_urdf(tmp_path / g["file"], base_pose=tuple(g["base_pose"]))
    assert [j.kind for j in io.joints] == ["fixed", "revolute", "revolute"]
    for name in ("base", "link1", "link2"):
        b = io.body(name)
        j = src.joint_of(name)
        frame = src.body(name).pose if name == "base" else compose(src.body(name).pose, j.child_pose)
        _same_pose(b.pose, frame)                                  # URDF link frame = source joint frame
        assert b.mass == pytest.approx(src.body(name).mass, rel=1e-6)
    j2 = next(j for j in io.joints if j.name == "j2")
    assert (j2.lower, j2.upper, j2.effort, j2.velocity, j2.damping) == pytest.approx((-2.0, 2.0, 10.0, 4.0, 0.1))
    mat = io.materials[io.body("link1").shapes[0].material]
    assert (mat.mu_dynamic, mat.restitution, mat.torsional_lever) == pytest.approx((0.5, 0.3, 0.001))


def test_urdf_stl_mesh_through_package_path(tmp_path):
    tri = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1], [1, 0, 0],
                    [0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0]], np.float32).reshape(4, 3, 3)
    data = b"\0" * 80 + struct.pack("<I", 4) + b"".join(np.zeros(3, np.float32).tobytes() + t.tobytes() + b"\0\0"
                                                         for t in tri)
    (tmp_path / "pkg" / "meshes").mkdir(parents=True)
    (tmp_path / "pkg" / "meshes" / "tet.stl").write_bytes(data)
    (tmp_path / "pkg" / "tet.urdf").write_text(
        '<robot name="tet"><link name="tet"><inertial><mass value="0.5"/><inertia ixx="1e-3" iyy="1e-3" izz="1e-3"/>'
        '</inertial><collision><geometry><mesh filename="package://pkg/meshes/tet.stl" scale="0.1 0.1 0.1"/>'
        '</geometry></collision></link></robot>')
    io = read_urdf(tmp_path / "pkg" / "tet.urdf")
    shape = io.body("tet").shapes[0]
    assert shape.kind == "mesh" and len(shape.mesh.vertices) == 4 and shape.mesh.vertices.max() == pytest.approx(0.1)
    assert io.joints[0].kind == "free"                              # a single link without joints is a loose object


def test_object_card_urdf_round_trip(tmp_path):
    export_object("cube_beech_30", tmp_path / "x", formats=("urdf",))
    res = import_file(tmp_path / "x" / "urdf" / "cube_beech_30.urdf", tmp_path / "lib")
    card = load_object(tmp_path / "lib" / "objects" / f"{res.objects[0]}.yaml")
    assert card.shape.kind == "box" and card.shape.size == pytest.approx((0.015,) * 3)
    lib = CardLibrary([tmp_path / "lib", default_root()])
    mat = lib.material(card.material)
    assert mat.density.value == pytest.approx(720.0, rel=1e-5) and mat.density.source == "model"
    assert (mat.surface.mu_dynamic.value, mat.surface.restitution.value) == pytest.approx((0.3, 0.5))
    assert not [i for i in res.issues if i["level"] == "error"]


@pytest.mark.parametrize("fmt", ["mjcf", "usd"])
def test_scene_round_trip_keeps_cards(tmp_path, fmt):
    if fmt == "usd":
        pytest.importorskip("pxr")
        pytest.importorskip("newton_usd_schemas")
    lib, name = ramp_library(tmp_path / "lib")
    out = export_scene(name, tmp_path / "out", formats=(fmt,), library=lib)
    res = import_file(out.files[fmt], tmp_path / "imp", name="rt")
    lib2 = CardLibrary([tmp_path / "imp", default_root()])
    scene = lib2.scene("rt")
    by = {p.object.removeprefix("rt_").split("_")[0]: p for p in scene.objects}
    for obj in ("ramp", "block"):
        new, old = lib2.object(by[obj].object), lib.object(obj)
        assert new.shape == old.shape and new.motion == old.motion
        assert lib2.material(new.material).surface.mu_dynamic.value == pytest.approx(0.2)
        _same_pose(by[obj].pose, next(p.pose for p in lib.scene(name).objects if p.object == obj))
    assert lib2.material(lib2.object(by["block"].object).material).density.value == pytest.approx(1050.0, rel=1e-5)
    assert not [i for i in res.issues if i["level"] == "error"]


def test_robot_scene_import_extracts_robot(tmp_path):
    src = default_root() / "robots" / "emb_e2_panda" / "scene.xml"
    res = import_file(src, tmp_path, name="panda_scene")
    assert len(res.robots) == 1 and {"panda_scene_table", "panda_scene_mug"} <= set(res.objects)
    m = mujoco.MjModel.from_xml_path(res.robots[0]["path"])
    assert m.nu == 8 and m.njnt == 9
    assert not [i for i in res.issues if i["level"] == "error"]


def test_errors_say_what_to_change(tmp_path):
    bad = tmp_path / "x.obj"
    bad.write_text("v 0 0 0")
    with pytest.raises(ImportFileError, match="unknown suffix"):
        read_file(bad)
    (tmp_path / "r.urdf").write_text('<robot name="r"><link name="a"><visual><geometry>'
                                     '<mesh filename="m.dae"/></geometry></visual></link></robot>')
    (tmp_path / "m.dae").write_text("")
    with pytest.raises(ValueError, match="convert it to .obj or .stl"):
        read_file(tmp_path / "r.urdf")


def test_cli_import_json(tmp_path, capsys):
    export_object("cube_beech_30", tmp_path / "x", formats=("urdf",))
    code = cli_main(["import", str(tmp_path / "x" / "urdf" / "cube_beech_30.urdf"), "--out", str(tmp_path / "lib")])
    out = json_of(capsys.readouterr().out)
    assert code == 0 and out["ok"] and out["objects"] == ["cube_beech_30"]


def test_thin_part_gets_min_wall_and_lone_object_collides(tmp_path):
    res = export_object("fish_turner_blade", tmp_path / "x", formats=("mjcf",))
    m = mujoco.MjModel.from_xml_path(res.files["mjcf"])
    assert m.geom_contype[0] == 1 and m.geom_conaffinity[0] == 1      # a lone exported object still collides
    imp = import_file(res.files["mjcf"], tmp_path / "lib")
    card = load_object(tmp_path / "lib" / "objects" / f"{imp.objects[0]}.yaml")
    assert card.features.min_wall is not None and card.features.min_wall.value < 1e-3
    assert not [i for i in imp.issues if i["level"] == "error"]
