# SPDX-License-Identifier: Apache-2.0
"""alloy.Scene: arguments, errors that say what to change, JSON round trip, and the cards it writes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from alloy.api.errors import SceneError
from alloy.api.scene import Scene

GRIPPER = ('<mujoco><worldbody><body name="b"><joint name="z" type="slide" axis="0 0 1"/>'
           '<geom type="box" size="0.01 0.01 0.01" mass="0.1"/></body></worldbody>'
           '<actuator><position joint="z" kp="100"/></actuator></mujoco>')


def _scene(worlds: int = 2) -> Scene:
    s = Scene(worlds=worlds, dt=0.01)
    s.add_ground()
    s.add_box("table", size=(1.0, 1.0, 0.75), pos=(0, 0, 0.375), static=True)
    s.add_box("cube", size=(0.1, 0.1, 0.1), pos=[[0, 0, 1], [0.01, 0, 1]][:worlds], mass=1.0)
    s.add_sphere("ball", 0.05, pos=(0.3, 0, 1), friction=0.9, rpy=(0, 0, 1.0))
    s.add_cloth("towel", size=(0.5, 0.4), pos=(0, 0, 1.2), mass=0.1, resolution=20)
    s.add_liquid("water", size=(0.05, 0.05, 0.08), pos=(0.3, 0.3, 0.2))
    s.add_robot("grip", mjcf=GRIPPER, pos=(0, 0.5, 0.3))
    return s


def test_json_round_trip():
    s = _scene()
    s.randomize("ball", "x", -0.01, 0.01)
    back = Scene.from_dict(json.loads(s.to_json()))
    assert back.to_dict() == s.to_dict()
    assert back.summary()["objects"] == {"ground": "ground", "table": "box", "cube": "box", "ball": "sphere",
                                         "towel": "cloth", "water": "liquid", "grip": "robot"}


@pytest.mark.parametrize("call, field", [
    (lambda s: s.add_box("a", size=(1, 2)), "size"),
    (lambda s: s.add_box("a", size=(1, 1, -1)), "size"),
    (lambda s: s.add_box("a", size=(1, 1, 1), half_extents=(1, 1, 1)), "size"),
    (lambda s: s.add_box("a", size=(1, 1, 1), material="steal"), "material"),
    (lambda s: s.add_box("1bad", size=(1, 1, 1)), "name"),
    (lambda s: s.add_box("cube", size=(1, 1, 1)), "name"),
    (lambda s: s.add_box("a", size=(1, 1, 1), mass=1, density=2), "mass"),
    (lambda s: s.add_box("a", size=(1, 1, 1), static=True, mass=1), "mass"),
    (lambda s: s.add_box("a", size=(1, 1, 1), pos=np.zeros((3, 3))), "pos"),
    (lambda s: s.add_sphere("a", 0.1, quat_wxyz=(1, 0, 0, 0), rpy=(0, 0, 0)), "rpy"),
    (lambda s: s.add_sphere("a", 0.1, restitution=2.0), "restitution"),
    (lambda s: s.add_robot("r", "missing.xml"), "path"),
    (lambda s: s.add_mesh("m", vertices=np.zeros((3, 3)), faces=np.array([[0, 1, 5]])), "faces"),
    (lambda s: s.randomize("nobody", "x", 0, 1), "name"),
])
def test_errors_name_the_field_and_a_fix(call, field):
    s = _scene()
    with pytest.raises(SceneError) as info:
        call(s)
    err = info.value
    assert err.field == field and err.fix and "fix:" in str(err)
    assert json.loads(json.dumps(err.to_dict()))["error"] == "SceneError"


def test_unknown_material_suggests_close_names():
    with pytest.raises(SceneError) as info:
        Scene().add_box("a", size=(1, 1, 1), material="steal")
    assert "steel_carbon" in info.value.fix


def test_orientation_forms_agree():
    s = Scene()
    s.add_sphere("a", 0.1, quat_wxyz=(0.0, 0.0, 0.0, 1.0))
    s.add_sphere("b", 0.1, quat_xyzw=(0.0, 0.0, 1.0, 0.0))
    s.add_sphere("c", 0.1, rpy=(0.0, 0.0, np.pi))
    qa, qb, qc = (np.array(s.items[n]["pose"][3:]) for n in "abc")
    assert np.allclose(qa, qb) and np.allclose(abs(qc @ qa), 1.0)


def test_cards_validate_and_hold_the_overrides(tmp_path):
    from alloy.cards.library import CardLibrary, default_root

    s = _scene()
    s.add_material("pad", density=1100.0, friction=1.2, restitution=0.2)
    s.add_box("block", size=(0.02, 0.02, 0.02), pos=(0, 0.2, 0.5), material="pad")
    s.set_friction("cube", "block", 1.3)
    s.save(tmp_path)
    lib = CardLibrary(roots=[tmp_path, default_root()], assets=None)
    card = lib.scene("scene")                                    # validates the scene and every object card
    assert card.worlds == 2 and card.control_dt == pytest.approx(0.01)
    cube = lib.material("abs_for_cube")
    assert cube.density.value == pytest.approx(1.0 / 0.1 ** 3)   # mass 1 kg in a 10 cm cube
    assert lib.material("abs_for_ball").surface.mu_static.value == pytest.approx(0.9)
    assert lib.object("towel").budget.particles == 20 * 16
    assert {(p.a, p.b) for p in card.pairs} == {("abs_for_cube", "pad_for_block")}
    assert lib.has_pair("abs_for_cube", "plywood_marine")     # sourced pairs follow a mass-only override


def test_set_friction_rejects_robots_and_shared_materials(tmp_path):
    s = _scene()
    s.set_friction("grip", "cube", 1.0)
    with pytest.raises(SceneError, match="robot"):
        s.save(tmp_path)


def test_mesh_from_arrays_is_written_with_its_hash(tmp_path):
    from alloy.api.meshes import open_box

    v, f = open_box((0.08, 0.08, 0.1), 0.004)
    s = Scene()
    s.add_mesh("cup", vertices=v, faces=f, collision="sdf", static=True)
    s.add_mesh("lid", vertices=v, faces=f, mass=0.05)
    s.save(tmp_path)
    assert (tmp_path / "meshes" / "cup.obj").is_file()
    from alloy.cards.library import CardLibrary, default_root

    lib = CardLibrary(roots=[tmp_path, default_root()], assets=None)
    lib.scene("scene")
    vol = 0.088 * 0.088 * 0.104 - 0.08 * 0.08 * 0.1
    assert lib.material("abs_for_lid").density.value == pytest.approx(0.05 / vol, rel=1e-6)


def test_scene_load_library_card_and_robot(tmp_path):
    from alloy.api.catalog import scenes
    from alloy.cards.library import CardLibrary, default_root

    assert "demo_peg_d12" in scenes()
    s = Scene.load("demo_peg_d12", worlds=3)
    assert s.worlds == 3 and "peg_d12" in s.names()
    s.randomize("peg_d12", "z", 0.04, 0.041)
    s.add_box("weight", size=(0.01, 0.01, 0.01), pos=(0.1, 0, 0.1))
    s.save(tmp_path / "a")
    card = CardLibrary(roots=[tmp_path / "a", default_root()], assets=None).scene("scene")
    assert card.worlds == 3 and len(card.objects) == 4
    xml = tmp_path / "arm.xml"
    xml.write_text(GRIPPER)
    r = Scene.load(xml)
    assert r.summary()["objects"] == {"ground": "ground", "arm": "robot"}
    with pytest.raises(SceneError, match="library scene"):
        Scene.load("no_such_scene")


def test_robot_file_with_its_own_floor_gets_no_second_ground(tmp_path):
    (tmp_path / "arm.xml").write_text(GRIPPER)
    (tmp_path / "scene.xml").write_text('<mujoco><include file="arm.xml"/><worldbody>'
                                        '<geom name="floor" type="plane" size="0 0 0.05"/></worldbody></mujoco>')
    assert Scene.load(tmp_path / "scene.xml").names() == ["scene"]
    assert Scene.load(tmp_path / "arm.xml").names() == ["ground", "arm"]
    from alloy.api._load import has_floor
    from alloy.api.catalog import library_root

    menagerie = library_root() / "robots" / "emb_e2_panda" / "scene.xml"
    if menagerie.is_file():
        assert has_floor(menagerie)


def test_imported_libraries_are_in_the_default_search(tmp_path, monkeypatch):
    from alloy.api import catalog

    lib = tmp_path / "assets"
    (lib / "imports" / "toyshop" / "objects").mkdir(parents=True)
    (lib / "imports" / "toyshop" / "materials").mkdir()
    (lib / "imports" / "toyshop" / "objects" / "toy_duck.yaml").write_text(
        "schema: alloy.object/1\nname: toy_duck\nshape: {primitive: {kind: sphere, size: [0.03]}}\nmaterial: abs\n")
    (lib / "imports" / "toyshop" / "materials" / "duck_vinyl.yaml").write_text("name: duck_vinyl\n")
    (lib / "imports" / "empty").mkdir()                              # no objects/: not a library
    monkeypatch.setenv("ALLOY_ASSETS", str(lib))
    catalog._names.cache_clear()
    assert [p.name for p in catalog.import_roots()] == ["toyshop"]
    assert "toy_duck" in catalog.objects() and catalog.import_root_of(catalog.card_path("objects", "toy_duck")).name \
        == "toyshop"
    assert "duck_vinyl" not in [m["name"] for m in catalog.materials()]
    assert "duck_vinyl" in [m["name"] for m in catalog.materials(imports=True)]
    s = Scene()
    s.add_object("duck", "toy_duck", pos=(0, 0, 0.1))
    monkeypatch.setenv("ALLOY_ASSETS", "")
    catalog._names.cache_clear()
    assert catalog.import_roots() == [] and catalog.assets_dir() is None


def test_assets_dir_matches_the_card_layer(monkeypatch):
    from alloy.api import catalog
    from alloy.cards.roots import assets_root

    monkeypatch.delenv("ALLOY_ASSETS", raising=False)
    assert catalog.assets_dir() == assets_root()


def test_wheel_layout_fallback(tmp_path, monkeypatch):
    from alloy.api import catalog

    data = tmp_path / "_data"
    (data / "imports" / "ycb_gso" / "objects").mkdir(parents=True)
    monkeypatch.setenv("ALLOY_ASSETS", "")                          # no checkout: the wheel's copy
    monkeypatch.setattr(catalog, "_package_data", lambda: data)
    assert [p.name for p in catalog.import_roots()] == ["ycb_gso"]


def test_card_roots_put_the_fetch_cache_before_each_import(tmp_path, monkeypatch):
    from alloy.api import catalog

    lib = tmp_path / "assets"
    for src in ("b_src", "a_src"):
        (lib / "imports" / src / "objects").mkdir(parents=True)
    monkeypatch.setenv("ALLOY_ASSETS", str(lib))
    monkeypatch.setenv("ALLOY_CACHE", str(tmp_path / "cache"))
    roots = catalog.card_roots(["mine"])
    assert roots[0] == Path("mine") and roots[1] == catalog.library_root()
    assert roots[2:] == [tmp_path / "cache" / "imports" / "a_src", tmp_path / "cache" / "imports" / "b_src",
                         lib.resolve() / "imports" / "a_src", lib.resolve() / "imports" / "b_src"]


def test_fast_catalog_mirrors_the_card_layer(monkeypatch):
    from alloy.api import catalog
    from alloy.cards import roots

    monkeypatch.delenv("ALLOY_ASSETS", raising=False)
    assert catalog.import_roots() == roots.import_roots()
    assert catalog.library_root() == roots.default_roots()[0]


def test_pour_sizes_reservoir_and_nozzle(tmp_path):
    from alloy.api.scene import SPH_SPACING as S

    s = Scene()
    s.add_pour("water", 200e-6, pos=(0, 0, 0.15), rate=2e-4, speed=0.5)
    p = s.items["water"]["pour"]
    assert p["volume"] == pytest.approx(200e-6, rel=0.03)
    k = round(2 * p["half_extents"][0] / S - 0.5)
    assert -p["velocity"][2] * (k * S) ** 2 == pytest.approx(2e-4)             # the nozzle carries the rate exactly
    with pytest.raises(SceneError, match="nozzle"):
        s.add_pour("big", 1e-3, pos=(0, 0, 1), rate=1.0, speed=0.1)
    s.save(tmp_path)


def test_scene_roots_argument_finds_own_cards(tmp_path):
    (tmp_path / "objects").mkdir()
    (tmp_path / "objects" / "my_cup.yaml").write_text(
        "schema: alloy.object/1\nname: my_cup\nshape: {primitive: {kind: cylinder, size: [0.04, 0.05]}}\nmaterial: abs\n")
    with pytest.raises(SceneError, match="roots="):
        Scene().add_object("cup", "my_cup")
    s = Scene(roots=[tmp_path])
    s.add_object("cup", "my_cup", pos=(0, 0, 0.1))
    assert s.to_dict()["roots"] == [str(tmp_path)]


def test_card_emitters_become_pours(tmp_path):
    from alloy.api.errors import SceneError as SE

    scene = Scene.load("demo_pour")                       # a library scene card with an emitter and a sink
    pours = {n: it["pour"] for n, it in scene.items.items() if it.get("pour")}
    assert list(pours) == ["water_pour_0"] and "emitters" not in scene.extra
    assert pours["water_pour_0"]["rate"] == pytest.approx(2e-5)
    card = (tmp_path / "scenes")
    card.mkdir()
    (card / "side.yaml").write_text("schema: alloy.scene/1\nrobots: []\nobjects: []\nemitters:\n  - kind: emitter\n"
                                    "    material: water\n    pose: [0, 0, 0.2, 0, 0, 0, 1]\n    half_extents: [0.005, "
                                    "0.005, 0.001]\n    rate: 1.0e-5\n    velocity: [0.5, 0, 0]\n    stop_s: 1\n")
    with pytest.raises(SE, match="downward"):
        Scene.load(str(card / "side.yaml"))
