# SPDX-License-Identifier: Apache-2.0
"""Exports of assembled card scenes: MJCF loads in MuJoCo with Alloy's values, URDF and USD carry masses, friction
and joint frames that MuJoCo and Newton read back."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from alloy.core.registry import isolated_registry
from alloy.io import export_object, export_scene
from alloy.io.cli import json_of, main as cli_main
from alloy.testing import scene as S

mujoco = pytest.importorskip("mujoco")


@pytest.fixture(autouse=True)
def _registry():
    with isolated_registry(empty=True):
        yield


@pytest.fixture(scope="module")
def box_export(tmp_path_factory):
    out = tmp_path_factory.mktemp("box")
    card, lib = S.box_on_table(worlds=2)
    with isolated_registry(empty=True):
        res = export_scene(card, out, library=lib, name="box_on_table")
    return res, out


def test_mjcf_loads_with_alloy_values(box_export):
    res, _ = box_export
    m = mujoco.MjModel.from_xml_path(res.files["mjcf"])
    box = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "box")
    assert m.body_mass[box] == pytest.approx(1000.0 * 0.06 ** 3 * 1.05, rel=1e-6)   # box card: abs, 1050 kg/m^3
    assert m.opt.timestep == pytest.approx(0.00125)
    assert m.npair == 1 and m.pair_friction[0, 0] == pytest.approx(2 * 0.4 * 0.5 / 0.9, rel=1e-5)  # harmonic mu_d
    assert res.counts["mjcf"]["bodies"] == 1


def test_urdf_carries_mass_inertia_friction(box_export):
    res, out = box_export
    index = json.loads((out / "urdf" / "index.json").read_text())
    groups = {g["name"]: g for g in index["groups"]}
    assert groups["table"]["fixed_base"] and not groups["box"]["fixed_base"]
    assert groups["box"]["base_pose"][2] == pytest.approx(0.051)
    link = ET.parse(out / "urdf" / "box.urdf").getroot().find("link")
    assert float(link.find("inertial/mass").get("value")) == pytest.approx(0.2268, rel=1e-6)
    assert float(link.find("inertial/inertia").get("ixx")) == pytest.approx(0.2268 * 0.06 ** 2 / 6, rel=1e-5)
    assert float(link.find("contact/lateral_friction").get("value")) == pytest.approx(0.4)
    assert link.find("collision/geometry/box").get("size") == "0.06 0.06 0.06"


def test_urdf_loads_in_newton(box_export):
    newton = pytest.importorskip("alloy._vendor.newton")
    _, out = box_export
    b = newton.ModelBuilder()
    b.add_urdf(str(out / "urdf" / "box.urdf"), floating=True)
    assert b.body_count == 1 and b.body_mass[0] == pytest.approx(0.2268, rel=1e-5)


def test_usd_text_and_pxr(box_export):
    res, _ = box_export
    text = open(res.files["usd"]).read()
    for token in ("PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysicsCollisionAPI", "PhysicsMaterialAPI",
                  'upAxis = "Z"', "physics:dynamicFriction = 0.4"):
        assert token in text
    pxr = pytest.importorskip("pxr")
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(res.files["usd"])
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    assert len(bodies) == 1 and UsdPhysics.MassAPI(bodies[0]).GetMassAttr().Get() == pytest.approx(0.2268, rel=1e-6)
    assert pxr is not None


def test_usd_loads_in_newton(box_export):
    pytest.importorskip("pxr")
    pytest.importorskip("newton_usd_schemas")
    from alloy._vendor import newton

    res, _ = box_export
    b = newton.ModelBuilder()
    b.add_usd(res.files["usd"])
    assert b.body_count == 1 and b.body_mass[0] == pytest.approx(0.2268, rel=1e-5)
    assert sorted(np.round(b.shape_material_mu, 4)) == [0.4, 0.5]


def test_unknown_format_says_what_to_use(tmp_path):
    card, lib = S.box_on_table(worlds=1)
    with pytest.raises(ValueError, match="mjcf, urdf, usd"):
        export_scene(card, tmp_path, formats=("sdf",), library=lib)


def test_library_scene_with_mesh_weld_and_precision_pair(tmp_path):
    res = export_scene("demo_nut_on_bolt", tmp_path)
    assert res.counts["mjcf"] == {"bodies": 3, "geoms": 4, "joints": 3, "actuators": 0, "pairs": 5, "equalities": 1}
    index = json.loads((tmp_path / "urdf" / "index.json").read_text())
    assert {g["name"]: g["fixed_base"] for g in index["groups"]}["bolt_m10x60"]      # welded to the world
    items = [u.item for u in res.unmapped]
    assert "precision pair bolt_m10x60-nut_m10" in items


def test_robot_scene_keeps_actuators(tmp_path):
    res = export_scene("task_emb_e2_panda", tmp_path, formats=("mjcf",))
    assert res.counts["mjcf"]["actuators"] == 8 and res.counts["mjcf"]["joints"] == 12


def test_object_export_and_cli_json(tmp_path, capsys):
    res = export_object("cube_beech_30", tmp_path / "api", formats=("urdf",))
    assert set(res.files) == {"urdf"}
    code = cli_main(["export", "--object", "cube_beech_30", "--formats", "mjcf", "--out", str(tmp_path / "cli")])
    out = json_of(capsys.readouterr().out)
    assert code == 0 and out["ok"] and out["counts"]["mjcf"]["bodies"] == 1
    code = cli_main(["export", "--scene", "no_such_scene", "--out", str(tmp_path / "bad")])
    out = json_of(capsys.readouterr().out)
    assert code == 1 and not out["ok"] and "no library scene" in out["error"]


def test_static_only_and_deformable_only_objects_export(tmp_path):
    res = export_object("block_bore_d12_2", tmp_path / "static", formats=("mjcf", "usd"))
    assert res.counts["mjcf"]["geoms"] == 1 and res.counts["mjcf"]["bodies"] == 0
    res = export_object("dough_ball", tmp_path / "soft", formats=("mjcf",))
    assert any("not exported" in u.reason for u in res.unmapped)


def test_scene_saved_by_api_exports_and_stdout_stays_clean(tmp_path, capfd):
    import alloy.api

    scene = alloy.api.Scene()
    scene.add_ground()
    scene.add_box("a", size=(0.02,) * 3, pos=(0, 0, 0.1))
    card = scene.save(tmp_path / "saved")
    capfd.readouterr()
    res = export_scene(str(card), tmp_path / "exported", formats=("mjcf",))   # finds ground/a in saved/
    assert res.counts["mjcf"]["bodies"] == 1
    out = capfd.readouterr().out
    assert "Module " not in out and "Saved mujoco model" not in out


def test_static_mjcf_object_and_massless_link(tmp_path):
    from alloy.cards import types as ct
    from alloy.io.groups import split_groups
    from alloy.io.mjcf_write import write_group_mjcf
    from alloy.io.model import IOShape
    from alloy.testing.io import arm_model

    card = ct.ObjectCard("hinged_box_static", ct.MJCFSource("mjcf/hinged_box.xml"), "plywood_marine", motion="static")
    res = export_object(card, tmp_path / "static", formats=("mjcf",))     # no MuJoCo entry in Alloy for it
    assert res.counts["mjcf"]["geoms"] == 2
    io = arm_model()
    io.bodies[2] = type(io.bodies[2])(**{**io.bodies[2].__dict__, "shapes": (IOShape("tip", "sphere", (0.01,),
                                                                                    collide=False),)})
    path = write_group_mjcf(io, split_groups(io)[0], tmp_path / "arm.xml")    # link2 has no colliding volume
    assert mujoco.MjModel.from_xml_path(str(path)).body_mass[3] == pytest.approx(0.2)


def test_exported_mjcf_loads_back_into_alloy(tmp_path):
    import re

    from alloy.api import load

    res = export_scene("demo_peg_d12", tmp_path, formats=("mjcf",))
    text = open(res.files["mjcf"]).read()
    assert all(len(v.split()) == 2 for v in re.findall(r'solref="([^"]*)"', text))     # no "0.02" shorthand
    assert all(len(v.split()) == 5 for v in re.findall(r'solimp="([^"]*)"', text))
    load(res.files["mjcf"], device="cpu")                    # Alloy's MJCF route rejects dampratio 0 shorthand


def test_export_takes_an_alloy_scene_object(tmp_path):
    import alloy.api

    scene = alloy.api.Scene()
    scene.add_ground()
    scene.add_box("a", size=(0.02,) * 3, pos=(0, 0, 0.1))
    res = export_scene(scene, tmp_path / "exported", formats=("mjcf",))
    assert res.counts["mjcf"]["bodies"] == 1
    with pytest.raises(TypeError, match="scene.save"):
        export_scene(42, tmp_path / "bad")
