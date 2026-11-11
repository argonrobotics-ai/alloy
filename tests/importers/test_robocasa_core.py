# SPDX-License-Identifier: Apache-2.0
"""RoboCasa importer: licence decisions, naming, mesh IO, hulls, material tables, pack reading (no downloads)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.cards import validate_material
from alloy.importers.robocasa import naming
from alloy.importers.robocasa.hulls import hull_mesh, hull_volume, reduce_hull
from alloy.importers.robocasa.materials import MaterialTable, part_role
from alloy.importers.robocasa.meshio import read_msh, read_obj_mesh, read_obj_mujoco, read_obj_uv, write_msh
from alloy.importers.robocasa.mjcf_in import read_model
from alloy.importers.robocasa.packs import PACKS, kept_packs, require_kept
from alloy.importers.robocasa.source import PackReader
from alloy.testing.imp_robocasa import box_obj, write_fixture_asset, write_object_asset, write_pack_zip

# Categories in the Lightwheel packs (zip listing of 2026-11-06); the table must cover every one.
LIGHTWHEEL_OBJECTS = (
    "aluminum_foil basket blender_jug cheese_grater chicken_drumstick cinnamon colander cookie_dough_ball "
    "cream_cheese_stick digital_scale dish_brush dish_rack flour_bag flower_vase fruit_bowl glass_cup honey_bottle "
    "hotdog_bun ice_cube ice_cube_tray jar juice kebab_skewer kettle knife_block lemon_wedge lettuce marshmallow "
    "mayonnaise measuring_cup mug_tree mustard oil_and_vinegar_bottle oven_tray pancake paper_towel_holder paprika "
    "peeler pickle_slice pitcher pizza pizza_cutter placemat plant pot reamer salt_and_pepper_shaker sandwich_bread "
    "saucepan shrimp soap_dispenser spray stool strainer straw sugar_cube syrup_bottle tiered_basket tiered_shelf "
    "tomato_slice tongs tray tupperware turkey_slice turmeric utensil_rack utensil_set whisk wooden_spoon").split()
LIGHTWHEEL_FIXTURES = ("blenders cabinets coffee_machines dishwashers electric_kettles fridges handles hoods microwaves "
                       "ovens sinks stand_mixers stovetops stoves toaster_ovens toasters windows").split()


def test_licence_decisions_keep_only_clear_cc_by_packs():
    assert [p.key for p in kept_packs()] == ["objects_lightwheel", "fixtures_lightwheel"]
    assert all(p.licence == "CC-BY-4.0" and p.attribution for p in kept_packs())
    assert PACKS["objaverse"].decision == "drop" and PACKS["aigen_objs"].decision == "hold"
    with pytest.raises(PermissionError, match="objaverse"):
        require_kept("objaverse")
    with pytest.raises(KeyError, match="use one of"):
        require_kept("nope")


@pytest.mark.parametrize("asset,name", [("glass_cup/GlassCup003", "robocasa_glass_cup_003"),
                                        ("saucepan/Saucepan002/Saucepan002_Lid", "robocasa_saucepan_002_lid"),
                                        ("fridges/Refrigerator031", "robocasa_refrigerator_031")])
def test_card_names_are_prefixed_snake_case(asset, name):
    assert naming.card_name(asset) == name
    assert naming.model_rel(asset) == f"mjcf/{asset.split('/')[0]}/{name}/model.xml"
    assert naming.card_rel(asset) == f"objects/{asset.split('/')[0]}/{name}.yaml"


def test_obj_to_msh_keeps_uvs_and_splits_seams(tmp_path):
    obj = tmp_path / "box.obj"
    obj.write_text(box_obj((0.1, 0.2, 0.3)))
    v, uv, f = read_obj_uv(obj)
    assert 8 < len(v) <= 24 and uv.shape == (len(v), 2) and f.shape == (12, 3)    # corners split at UV seams
    write_msh(tmp_path / "box.msh", v, f, uv)
    v2, uv2, f2 = read_msh(tmp_path / "box.msh")
    np.testing.assert_allclose(v2, v, atol=1e-7)
    np.testing.assert_allclose(uv2, uv, atol=1e-7)
    assert (f2 == f).all()


def test_partly_mapped_obj_reads_like_mujoco(tmp_path):
    obj = tmp_path / "box.obj"
    obj.write_text(box_obj((0.1, 0.2, 0.3), uv_faces=4))          # 2 of 6 faces written without vt
    v, uv, f = read_obj_uv(obj)
    assert uv is not None and uv.shape == (len(v), 2) and f.shape == (8, 3)   # as MuJoCo: UV-less faces dropped
    obj.write_text(box_obj((0.1, 0.2, 0.3), uv_faces=1))          # mostly unmapped: keep faces, discard UVs
    v, uv, f = read_obj_uv(obj)
    assert uv is None and f.shape == (12, 3)
    obj.write_text(box_obj((0.1, 0.2, 0.3), uv_faces=0))
    assert read_obj_uv(obj)[1] is None


def test_reduce_hull_caps_vertices_and_stays_inside():
    pts = np.random.default_rng(1).normal(size=(2000, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)                 # all on the hull
    full_v, full_f = hull_mesh(pts)
    small_v, small_f = hull_mesh(reduce_hull(pts, 64))
    assert len(full_v) > 64 >= len(small_v)
    assert 0.8 * hull_volume(full_v, full_f) < hull_volume(small_v, small_f) <= hull_volume(full_v, full_f) + 1e-12


def test_material_tables_resolve_and_new_cards_validate():
    table = MaterialTable.load()
    assert table.missing() == []
    assert set(LIGHTWHEEL_OBJECTS) <= set(table.objects)
    assert set(LIGHTWHEEL_FIXTURES) <= set(table.fixtures)
    for name, card in table.cards.items():
        assert not [i for i in validate_material(card) if i.level == "error"], name
    assert table.density("robocasa_ice").value == pytest.approx(917.0)


@pytest.mark.parametrize("stem,role", [("GlassCup003_Clear", "Clear"), ("Saucepan002", "body"), ("Lid", "Lid"),
                                       ("prop", "prop"), ("CreamCheese", "body"), ("Jar010_Prop", "Prop")])
def test_part_roles(stem, role):
    assert part_role(stem) == role


def test_reader_parses_objects_and_fixture_joints(tmp_path):
    obj = read_model(write_object_asset(tmp_path / "o" / "GlassCup900")).flattened()
    assert [v.mesh for v in obj.visuals] == ["visual/GlassCup900_Clear.obj"]
    assert len(obj.collisions) == 1 and [r.name for r in obj.regions] == ["reg_bbox"]
    fx = read_model(write_fixture_asset(tmp_path / "f" / "Cab900"))
    assert [(j.name, j.kind) for j in fx.joints()] == [("door_joint", "hinge")]
    with pytest.raises(ValueError, match="articulated"):
        fx.flattened()


def test_pack_reader_lists_extracts_and_skips_nested_assets(tmp_path):
    a = write_object_asset(tmp_path / "src" / "GlassCup900")
    lid = write_object_asset(tmp_path / "src" / "GlassCup900_Lid")
    zpath = write_pack_zip(tmp_path / "pack.zip", "lightwheel",
                           {"glass_cup/GlassCup900": a, "glass_cup/GlassCup900/GlassCup900_Lid": lid})
    reader = PackReader.open("objects_lightwheel", zpath)
    assert reader.assets() == ["glass_cup/GlassCup900", "glass_cup/GlassCup900/GlassCup900_Lid"]
    files = [i.filename for i in reader.files("glass_cup/GlassCup900")]
    assert not any("_Lid/" in f for f in files)
    out = reader.extract("glass_cup/GlassCup900", tmp_path / "work")
    assert (out / "model.xml").is_file() and not (out / "GlassCup900_Lid").exists()
    with pytest.raises(KeyError, match="PackReader.assets"):
        reader.files("glass_cup/Nope")


def test_obj_normals_reach_the_msh(tmp_path):
    obj = tmp_path / "tri.obj"
    obj.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nvn 0 0 2\nvn 1 0 0\n"
                   "f 1//1 3//1 2//1\nf 1//2 2//2 4//2\n")
    v, uv, n, f = read_obj_mesh(obj)
    assert uv is None and n.shape == (len(v), 3) and len(v) == 6          # vertex 1 and 2 split by normal
    write_msh(tmp_path / "tri.msh", v, f, uv, n)
    head = np.fromfile(tmp_path / "tri.msh", dtype=np.int32, count=4)
    assert head.tolist() == [6, 6, 0, 2]
    m = __import__("mujoco").MjModel.from_xml_string(
        f'<mujoco><asset><mesh name="t" file="{tmp_path / "tri.msh"}"/></asset></mujoco>')
    assert m.mesh_normalnum[0] == 6


def test_mujoco_reader_matches_the_file_frame(tmp_path):
    obj = tmp_path / "box.obj"
    obj.write_text(box_obj((0.1, 0.2, 0.3)))
    v, uv, n, f = read_obj_mujoco(obj)                         # MuJoCo recentres meshes; we map back
    np.testing.assert_allclose(v.min(axis=0), [-0.1, -0.2, -0.3], atol=1e-6)
    np.testing.assert_allclose(v.max(axis=0), [0.1, 0.2, 0.3], atol=1e-6)
    assert f.shape == (12, 3) and uv is not None and uv.shape == (len(v), 2)
    obj.write_text(box_obj((0.1, 0.2, 0.3), uv_faces=0))
    from alloy.importers.robocasa.objects import visual_mesh

    assert visual_mesh(obj)[1] is None                         # MuJoCo invents UVs; an unmapped OBJ stays unmapped
