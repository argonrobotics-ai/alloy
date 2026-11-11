# SPDX-License-Identifier: Apache-2.0
"""RoboCasa importer end to end on synthetic assets: object and fixture models, checks, cards, records, CLI."""

from __future__ import annotations

import json

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("coacd")

from alloy.cards import CardLibrary, default_root  # noqa: E402
from alloy.importers.robocasa.batch import finalize, load_records  # noqa: E402
from alloy.importers.robocasa.checks import drop_test, fixture_test  # noqa: E402
from alloy.importers.robocasa.cli import main  # noqa: E402
from alloy.importers.robocasa.fixtures import import_fixture  # noqa: E402
from alloy.importers.robocasa.hulls import HullParams  # noqa: E402
from alloy.importers.robocasa.materials import MaterialTable  # noqa: E402
from alloy.importers.robocasa.meshio import read_msh  # noqa: E402
from alloy.importers.robocasa.objects import import_object  # noqa: E402
from alloy.importers.robocasa.pipeline import import_asset  # noqa: E402
from alloy.testing.imp_robocasa import BOX_HALF, write_fixture_asset, write_object_asset, write_pack_zip  # noqa: E402

FAST = HullParams(threshold=0.2, max_hulls=4)


def test_object_mass_hulls_and_model_agree(tmp_path):
    folder = write_object_asset(tmp_path / "src" / "Tupperware900")
    res = import_object(folder, "tupperware/Tupperware900", tmp_path / "out", params=FAST)
    rho = MaterialTable.load().density("pp_homopolymer").value
    assert res.mass_kg == pytest.approx(rho * 8 * np.prod(BOX_HALF), rel=1e-4)   # closed Clear part -> solid PP
    assert res.flags == [] and res.name == "robocasa_tupperware_900"
    model = tmp_path / "out" / res.model
    for hull in model.parent.glob("hulls/*.msh"):
        assert len(read_msh(hull)[0]) <= 64
    m = mujoco.MjModel.from_xml_path(str(model))
    assert m.body_mass[1] == pytest.approx(res.mass_kg, rel=1e-6)
    groups = np.bincount(m.geom_group, minlength=4)
    assert groups[2] == 1 and groups[3] == res.hulls                       # one visual, the hulls collide
    assert drop_test(model).ok


def test_implausible_mass_is_flagged(tmp_path):
    folder = write_object_asset(tmp_path / "src" / "SugarCube900", half=(0.05, 0.05, 0.05))   # a 10 cm sugar cube
    res = import_object(folder, "sugar_cube/SugarCube900", tmp_path / "out", params=FAST)
    assert any(f.startswith("size") for f in res.flags) and any(f.startswith("mass") for f in res.flags)


def test_fixture_keeps_joint_and_spreads_typical_mass(tmp_path):
    folder = write_fixture_asset(tmp_path / "src" / "Microwave900")
    res = import_fixture(folder, "microwaves/Microwave900", tmp_path / "out")
    assert res.joints == ["door_joint"] and res.bodies == 2
    m = mujoco.MjModel.from_xml_path(str(tmp_path / "out" / res.model))
    assert m.njnt == 1 and m.jnt_type[0] == mujoco.mjtJoint.mjJNT_HINGE
    assert m.body_mass[1:].sum() == pytest.approx(15.0, rel=1e-3)          # microwaves: typical 15 kg
    check = fixture_test(tmp_path / "out" / res.model)
    assert check.ok, check.failures


def test_pipeline_record_card_and_finalize(tmp_path, monkeypatch):
    import alloy.importers.robocasa.objects as objects

    monkeypatch.setattr(objects, "HullParams", lambda: FAST)
    original = objects.decompose
    monkeypatch.setattr(objects, "decompose", lambda v, f, p=None: original(v, f, FAST))
    a = write_object_asset(tmp_path / "src" / "GlassCup900")
    zpath = write_pack_zip(tmp_path / "pack.zip", "lightwheel", {"glass_cup/GlassCup900": a})
    out = tmp_path / "root"
    rec = import_asset("objects_lightwheel", "glass_cup/GlassCup900", out, zpath)
    assert rec["status"] in ("ok", "flagged") and not rec.get("error")
    assert rec["status"] == "ok" or rec["thumbnail"] is None                # flagged only without a GL backend
    assert rec["licence"] == "CC-BY-4.0" and rec["files"][rec["card"]]["sha256"]
    lib = CardLibrary([default_root(), out])
    card = lib.object("robocasa_glass_cup_900")
    assert card.material == "glass_soda_lime" and card.motion == "dynamic"
    assert "Licence: CC-BY-4.0" in (out / rec["card"]).read_text()
    report = finalize(out)
    assert report["assets"] == 1 and load_records(out)[0]["name"] == "robocasa_glass_cup_900"
    manifest = json.loads((out / "manifest.json").read_text())
    assert "robocasa_glass_cup_900" in manifest["assets"]
    assert "objects_lightwheel" in (out / "ATTRIBUTION.md").read_text()


def test_failed_asset_gives_a_failed_record(tmp_path):
    a = write_object_asset(tmp_path / "src" / "Thing900")
    zpath = write_pack_zip(tmp_path / "pack.zip", "lightwheel", {"unknown_category/Thing900": a})
    rec = import_asset("objects_lightwheel", "unknown_category/Thing900", tmp_path / "root", zpath)
    assert rec["status"] == "failed" and "categories.yaml" in rec["error"]


def test_cli_prints_json(capsys):
    assert main(["packs"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["objaverse"]["decision"] == "drop"
    assert main(["import", "--pack", "objaverse", "--asset", "x/y", "--out", "unused"]) == 2
    assert "PermissionError" in json.loads(capsys.readouterr().out)["error"]


def test_pending_records_complete_in_post_pass(tmp_path):
    from alloy.importers.robocasa.batch import post_pass

    a = write_fixture_asset(tmp_path / "src" / "Microwave900")
    zpath = write_pack_zip(tmp_path / "pack.zip", "fixtures", {"microwaves/Microwave900": a})
    out = tmp_path / "root"
    rec = import_asset("fixtures_lightwheel", "microwaves/Microwave900", out, zpath, validate=False, thumbs=False)
    assert rec["status"] == "pending" and rec["card_issues"] is None and rec["thumbnail"] == "pending"
    counts = post_pass(out)
    done = load_records(out)[0]
    assert done["card_issues"] is not None and done["status"] in ("ok", "flagged")
    assert sum(counts.values()) == 1


def test_composed_cabinets_have_working_doors_and_drawers(tmp_path):
    from alloy.importers.robocasa.cabinets import compose_all
    from alloy.testing.imp_robocasa import write_handle_asset, write_panel_asset

    root = tmp_path / "root"
    import_fixture(write_panel_asset(tmp_path / "src" / "CabinetDoorPanel900"),
                   "cabinets/cabinet_panels/CabinetDoorPanel900", root)
    import_fixture(write_handle_asset(tmp_path / "src" / "CabinetHandle900"), "handles/CabinetHandle900", root)
    counts = compose_all(root, layouts=("double", "drawers"))
    assert sum(counts.values()) == 2 and "failed" not in counts
    recs = {r["name"]: r for r in load_records(root)}
    double = recs["robocasa_cabinet_double_900_900"]
    assert double["joints"] == ["door_left_hinge", "door_right_hinge"] and double["check"]["ok"]
    drawers = recs["robocasa_cabinet_drawers_900_900"]
    assert len(drawers["joints"]) == 3 and drawers["check"]["ok"]
    assert 15.0 < drawers["mass_kg"] < 80.0                                  # a 60 cm drawer base cabinet
    m = mujoco.MjModel.from_xml_path(str(root / drawers["model"]))
    assert all(m.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE for j in range(m.njnt))


def test_overlapping_sibling_parts_get_contact_excludes(tmp_path):
    from alloy.importers.robocasa.fixtures import add_contact_excludes, overlapping_pairs

    xml = tmp_path / "model.xml"
    xml.write_text("""<mujoco><worldbody><body name="base"><geom type="box" size="0.2 0.2 0.2"/>
      <body name="knob"><joint name="k" type="hinge" axis="0 1 0" range="0 3"/>
        <geom type="cylinder" size="0.02 0.02" pos="0 -0.19 0" quat="0.7071068 0.7071068 0 0"/></body>
      <body name="panel"><geom type="box" size="0.1 0.01 0.1" pos="0 -0.2 0"/></body>
    </body></worldbody></mujoco>""")
    pairs = overlapping_pairs(xml)
    assert pairs == [("base", "knob"), ("knob", "panel")]   # siblings collide, and so do children of a welded root
    add_contact_excludes(xml, pairs)
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    assert d.ncon == 0
    with pytest.raises(ValueError, match="re-import"):
        add_contact_excludes(xml, pairs)


def test_textured_part_without_uvs_gets_the_texture_colour(tmp_path):
    from alloy.importers.robocasa.objects import dominant_colour

    folder = write_object_asset(tmp_path / "src" / "Tupperware901", uv_faces=0)
    res = import_object(folder, "tupperware/Tupperware901", tmp_path / "out", params=FAST)
    model = tmp_path / "out" / res.model
    text = model.read_text()
    assert "texture" not in text                                   # the atlas cannot be mapped: no texture bound
    m = mujoco.MjModel.from_xml_path(str(model))
    rgb = dominant_colour(folder / "visual" / "T_BC001.png")
    vis = [g for g in range(m.ngeom) if m.geom_group[g] == 2][0]
    np.testing.assert_allclose(m.mat_rgba[m.geom_matid[vis]][:3], rgb, atol=1e-6)
    assert m.mat_rgba[m.geom_matid[vis]][3] == pytest.approx(0.7)
    assert not (model.parent / "visual" / "T_BC001.png").exists()   # unused texture not copied
