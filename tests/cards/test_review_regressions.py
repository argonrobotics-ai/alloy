# SPDX-License-Identifier: Apache-2.0
"""Regressions for the 15 bugs an independent review reproduced on main@a9bf398 (one test or case each)."""

from __future__ import annotations

import dataclasses as dc
import math
from pathlib import Path

import numpy as np
import pytest

from alloy.cards import geometry as geo
from alloy.cards import meshgen as mg
from alloy.cards import types as ct
from alloy.cards.library import CardLibrary
from alloy.cards.loader import dump_card, loads_card
from alloy.cards.paths import PathError, get_value
from alloy.cards.validate import errors, validate_object, validate_scene
from alloy.core.errors import CardError
from alloy.core.quantity import Q

LIB = CardLibrary(assets=None)
O = "schema: alloy.object/1\nname: x\nmaterial: steel_carbon\n"
S = "schema: alloy.scene/1\nrobots: []\nobjects: []\n"
MAT = "schema: alloy.material/1\nname: m\nsurface: {mu_static: 0.5, mu_dynamic: 0.4, restitution: 0.5}\n"


@pytest.mark.parametrize("text, field", [                                                    # bug 1 (YAML side)
    (O + "shape: {curve: {points: [[0,0,0],[.nan,0,0]], radius: 0.001}}\n", "shape.points.1.0"),
    (O + "shape: {particles: {region: [[0,0,0],[.inf,1,1]], spacing: 0.01}}\n", "shape.region.1.0"),
    (O + "shape: {primitive: {kind: box, size: [0.01, 0.01, 0.01]}}\nfeatures: {precision_anchor: [0, .nan, 0]}\n",
     "features.precision_anchor.1"),
])
def test_non_finite_plain_floats_rejected_at_load(text: str, field: str) -> None:
    with pytest.raises(CardError) as exc:
        loads_card(text, ct.ObjectCard)
    assert exc.value.field == field


def test_non_finite_plain_floats_rejected_by_validation() -> None:                         # bug 1 (programmatic)
    card = dc.replace(LIB.object("cube_beech_30"), blade=ct.BladeSpec((0, 0, 0), (math.nan, 0, 0)),
                      topology=LIB.object("knife_blade_steel").topology)
    assert "blade.edge_b.0" in [i.field for i in errors(validate_object(card, LIB))]
    scene = ct.SceneCard((), (), emitters=(ct.EmitterSpec("emitter", "water", (0, 0, 0, 0, 0, 0, 1.0), (0.01,) * 3, 1e-5,
                                                          stop_s=math.nan),))
    assert "emitters.0.stop_s" in [i.field for i in errors(validate_scene(scene, LIB))]


def test_decimal_ints_and_bad_tagged_scalars() -> None:                                    # bug 2
    assert loads_card(S + "worlds: 010\n", ct.SceneCard).worlds == 10
    assert loads_card(S + "worlds: 09\n", ct.SceneCard).worlds == 9
    with pytest.raises(CardError):
        loads_card(MAT + "density: !!float abc\n", ct.MaterialCard)


def test_unknown_mjcf_body_is_a_card_error() -> None:                                      # bug 3
    c = loads_card(O + "shape: {mjcf: {path: mjcf/hinged_box.xml, body: nope}}\n", ct.ObjectCard)
    assert "shape.body" in [i.field for i in errors(validate_object(c, LIB))]
    with pytest.raises(CardError) as exc:
        LIB.resolve(c)
    assert exc.value.field == "shape.body"


def test_inside_out_mesh_is_not_watertight_and_wall_still_measured() -> None:              # bug 4
    plate = mg.extrude(mg.rect(0.05, 0.05), 0.0, 5e-4)
    inverted = geo.MeshData(plate.vertices, plate.faces[:, ::-1].copy())
    assert geo.is_watertight(plate) and not geo.is_watertight(inverted)
    assert geo.wall_thickness(inverted) == pytest.approx(5e-4)


def test_inside_out_mesh_card_is_rejected(tmp_library: Path) -> None:                     # bug 4 (card level)
    plate = mg.extrude(mg.rect(0.05, 0.05), 0.0, 5e-4)
    path = tmp_library / "meshes" / "inv.obj"
    mg.write_obj(path, geo.MeshData(plate.vertices, plate.faces[:, ::-1].copy()))
    lib = CardLibrary([tmp_library])
    c = ct.ObjectCard("x", ct.MeshSource("meshes/inv.obj", geo.sha256_file(path)), "steel_carbon")
    fields = [i.field for i in errors(validate_object(c, lib))]
    assert "shape.path" in fields and "features.min_wall" in fields


def test_rotated_round_mjcf_geoms_keep_their_size(tmp_path: Path) -> None:                 # bug 5
    xml = tmp_path / "r.xml"
    xml.write_text('<mujoco><worldbody><body name="b" euler="0 0 45"><geom type="sphere" size="0.02"/></body>'
                   '<body name="c" pos="1 0 0" euler="30 0 0"><geom type="cylinder" size="0.01 0.05"/></body></worldbody></mujoco>')
    lib = CardLibrary([tmp_path])
    sphere = ct.ObjectCard("s", ct.MJCFSource(str(xml), "b"), "steel_carbon")
    cyl = ct.ObjectCard("c", ct.MJCFSource(str(xml), "c"), "steel_carbon")
    from alloy.cards.validate_object import shape_extents
    assert shape_extents(sphere, lib.mesh_for(sphere)) == pytest.approx((0.04, 0.04, 0.04), rel=5e-3)
    assert shape_extents(cyl, lib.mesh_for(cyl)) == pytest.approx((0.1, 0.02, 0.02), rel=5e-3)


def test_rescan_drops_cached_meshes(tmp_library: Path) -> None:                            # bug 6
    lib = CardLibrary([tmp_library])
    path = tmp_library / "meshes" / "block_bore_d12_2.obj"
    assert errors(lib.object_issues("block_bore_d12_2")) == []
    m = geo.read_obj(path)
    mg.write_obj(path, geo.MeshData(m.vertices, m.faces[1:]))
    card = lib.peek_object("block_bore_d12_2")
    (tmp_library / "objects" / "block_bore_d12_2.yaml").write_text(
        dump_card(dc.replace(card, shape=dc.replace(card.shape, sha256=geo.sha256_file(path)))))
    lib.rescan()
    assert "shape.path" in [i.field for i in errors(lib.object_issues("block_bore_d12_2"))]


def test_huge_ints_are_card_errors() -> None:                                              # bug 7
    with pytest.raises(CardError) as exc:
        loads_card(MAT + "density: 1" + "0" * 400 + "\n", ct.MaterialCard)
    assert exc.value.field == "density"


def test_obj_face_index_out_of_range(tmp_library: Path) -> None:                          # bug 8
    (tmp_library / "meshes" / "bad.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 9\n")
    lib = CardLibrary([tmp_library])
    c = ct.ObjectCard("x", ct.MeshSource("meshes/bad.obj", geo.sha256_file(tmp_library / "meshes" / "bad.obj")), "steel_carbon")
    assert "shape.path" in [i.field for i in errors(validate_object(c, lib))]


def test_duplicate_key_names_the_dotted_field() -> None:                                  # bug 9
    with pytest.raises(CardError) as exc:
        loads_card(MAT.replace("restitution: 0.5}", "restitution: 0.5, mu_static: 0.6}") + "density: 1000\n", ct.MaterialCard)
    assert exc.value.field == "surface.mu_static"
    with pytest.raises(CardError) as exc:
        loads_card(S.replace("objects: []", "objects: [{object: a, pose: [0,0,0,0,0,0,1], object: b}]"), ct.SceneCard)
    assert exc.value.field == "objects.0.object"


def test_obb_of_isotropic_rotated_shapes() -> None:                                       # bug 10
    cube = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)])
    c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    assert geo.obb_extents(cube @ rz.T) == pytest.approx((1, 1, 1), rel=1e-6)
    peg = cube * [1, 1, 3]
    rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    assert geo.obb_extents(peg @ rz.T) == pytest.approx((3, 1, 1), rel=1e-6)
    assert geo.obb_extents(peg @ rx.T @ rz.T) == pytest.approx((3, 1, 1), rel=1e-6)


def test_rolling_warning_names_the_part_material(tmp_library: Path) -> None:              # bug 11
    lib = CardLibrary([tmp_library])
    m = lib.material("nitrile_70a")
    lib.put(dc.replace(m, name="pad_rolling", surface=dc.replace(m.surface, rolling_lever=Q(1e-4, "model", None, "x"))), reason="t")
    plug = dc.replace(lib.peek_object("usb_a_plug"), part_materials={"shell": "steel_nickel_plated", "insert": "pad_rolling",
                                                                    "overmold": "pvc_cable_jacket"})
    warns = [i.field for i in validate_object(plug, lib) if i.level == "warn"]
    assert "part_materials.insert.surface.rolling_lever" in warns


def test_material_names_are_file_safe(tmp_library: Path) -> None:                          # bug 12
    lib = CardLibrary([tmp_library])
    with pytest.raises(CardError) as exc:
        lib.put(dc.replace(lib.material("nitrile_70a"), name="sub/nitrile"), reason="x")
    assert exc.value.field == "name"


def test_broken_placed_object_is_an_issue_not_a_crash(tmp_library: Path) -> None:        # bug 13
    p = tmp_library / "objects" / "peg_d12.yaml"
    p.write_text(p.read_text() + "bogus: 1\n")
    lib = CardLibrary([tmp_library])
    fields = [i.field for i in errors(validate_scene(lib.peek_scene("demo_peg_d12"), lib))]
    assert "objects.2.object" in fields


@pytest.mark.parametrize("note", ["a\x85b", "a b", "tab\there", "bell\x07", "multi\nline\n", "trailing \n"])
def test_special_characters_round_trip(note: str) -> None:                                  # bug 14
    m = LIB.material("water")
    card = dc.replace(m, density=dc.replace(m.density, note=note))
    assert loads_card(dump_card(card), ct.MaterialCard) == card


def test_non_ascii_digits_are_path_errors() -> None:                                       # bug 15
    from alloy.cards.paths import CardView
    v = CardView(LIB.object("peg_d12"), LIB.material("stainless_303"))
    with pytest.raises(PathError):
        get_value(v, "shape.size.²")
    scene = loads_card(S.replace("objects: []", "objects: [{object: peg_d12, pose: [0,0,0,0,0,0,1], "
                                 "randomize: [{field: shape.size.0, lo: 0.005, hi: 0.007}]}]"), ct.SceneCard)
    bad = dc.replace(scene, objects=(dc.replace(scene.objects[0], randomize=(ct.Randomization("shape.size.²", 0.0, 1.0),)),))
    assert "objects.0.randomize.0.field" in [i.field for i in errors(validate_scene(bad, LIB))]
