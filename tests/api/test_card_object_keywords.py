# SPDX-License-Identifier: Apache-2.0
"""Library objects (``Scene.add_object``) and the objects of a loaded scene take what primitives take: the shared
``add_*`` keywords, ``Scene.set_friction`` and a randomized friction below their card's dynamic friction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import alloy
from alloy.api.errors import BuildError, SceneError
from alloy.api.scene import Scene


def _library(*roots: Path):
    from alloy.cards.library import CardLibrary, default_root

    return CardLibrary(roots=[*roots, default_root()], assets=None)


def _surface(lib, obj: str):
    mat = lib.material(lib.object(obj).material)
    return mat, mat.surface


# --- set_friction and friction ranges on library objects and loaded scenes ------------------------------------------


def test_set_friction_on_a_library_object_writes_its_pair(tmp_path):
    s = Scene()
    s.add_ground()
    s.add_object("cell", "aa_cell", pos=(0, 0, 0.03))
    s.set_friction("cell", "ground", 0.9)
    s.save(tmp_path)
    lib = _library(tmp_path)
    card = lib.scene("scene")                                     # validates every card it names
    mat, _ = _surface(lib, "cell")
    assert mat.name == "aa_cell_effective_for_cell" and mat.pairs_from == "aa_cell_effective"
    assert [(p.a, p.b, p.mu_static.value, p.mu_dynamic.value) for p in card.pairs] == [
        ("aa_cell_effective_for_cell", "plywood_marine_for_ground", 0.9, 0.9)]


def test_set_friction_on_a_loaded_scene_writes_its_pair(tmp_path):
    s = Scene()
    s.add_ground()
    s.add_box("b1", size=(0.04, 0.04, 0.04), pos=(0, 0, 0.02), material="abs")
    s.add_box("b2", size=(0.04, 0.04, 0.04), pos=(0.1, 0, 0.02), material="abs")
    loaded = Scene.load(s.save(tmp_path / "saved"))
    loaded.set_friction("b1", "b2", 0.8)
    loaded.save(tmp_path / "again")
    lib = _library(tmp_path / "again", tmp_path / "saved")
    card = lib.scene("scene")
    assert [(p.a, p.b, p.mu_static.value) for p in card.pairs] == [("abs_for_b1", "abs_for_b2", 0.8)]
    assert (lib.object("b1").material, lib.object("b2").material) == ("abs_for_b1", "abs_for_b2")
    assert [o.object for o in card.objects] == ["ground", "b1", "b2"]       # the same names: cards rewritten


@pytest.mark.parametrize("loaded", [False, True])
def test_friction_range_below_a_cards_dynamic_friction_caps_it(tmp_path, loaded):
    s = Scene(worlds=4)
    s.add_ground()
    s.add_object("aa_cell", "aa_cell", pos=(0, 0, 0.03))
    if loaded:
        s = Scene.load(s.save(tmp_path / "saved"))
    s.randomize("aa_cell", "friction", 0.05, 0.8)
    s.save(tmp_path / "out")
    lib = _library(tmp_path / "out", tmp_path / "saved")
    lib.scene("scene")                                            # the 0.05 end validates
    _, surface = _surface(lib, "aa_cell")
    assert surface.mu_dynamic.value == pytest.approx(0.05) and surface.mu_static.value == pytest.approx(0.4)


def test_a_shared_card_in_a_loaded_scene_refuses_a_friction_of_its_own(tmp_path):
    s = Scene()
    s.add_ground()
    s.add_object("cube_beech_30", "cube_beech_30", pos=(0, 0, 0.015))
    s.save(tmp_path / "saved")
    text = (tmp_path / "saved" / "scenes" / "scene.yaml").read_text()
    (tmp_path / "saved" / "scenes" / "twice.yaml").write_text(text.replace(
        "objects:\n", "objects:\n- object: cube_beech_30\n  pose: [0.1, 0, 0.015, 0, 0, 0, 1]\n"))
    loaded = Scene.load(tmp_path / "saved" / "scenes" / "twice.yaml")
    assert {"cube_beech_30", "cube_beech_30#1"} <= set(loaded.names())
    loaded.set_friction("cube_beech_30#1", "ground", 0.9)
    with pytest.raises(SceneError, match="shares object card") as info:
        loaded.save(tmp_path / "out")
    assert "objects/<new name>.yaml" in info.value.fix


def test_set_friction_on_a_robot_says_where_its_friction_is(tmp_path):
    s = Scene()
    s.add_object("cell", "aa_cell")
    s.add_robot("arm", mjcf='<mujoco><worldbody><body name="b"><joint type="hinge"/><geom type="sphere" size="0.01" '
                             'mass="0.1"/></body></worldbody></mujoco>')
    s.set_friction("cell", "arm", 0.5)
    with pytest.raises(SceneError, match="'arm' is a robot") as info:
        s.save(tmp_path)
    assert "MJCF <geom friction" in info.value.fix


# --- the shared add_* keywords on add_object --------------------------------------------------------------------------


def test_add_object_takes_the_shared_keywords(tmp_path):
    s = Scene()
    s.add_ground()
    s.add_object("hard", "sponge_kitchen", pos=(0, 0, 0.05), rigid=True)
    s.add_object("soft", "cube_beech_30", pos=(0.1, 0, 0.05), rigid=False)
    s.add_object("pusher", "cube_beech_30", pos=(0.2, 0, 0.015), kinematic=True)
    s.add_object("heavy", "cube_beech_30", pos=(0.3, 0, 0.015), mass=0.05, restitution=0.2)
    s.add_object("dense", "cube_beech_30", pos=(0.4, 0, 0.015), density=900.0, friction=0.7)
    s.add_object("steel", "cube_beech_30", pos=(0.5, 0, 0.015), material="steel_carbon")
    s.add_object("fixed", "aa_cell", pos=(0.6, 0, 0.03), static=True, friction=0.6)
    assert Scene.from_dict(json.loads(s.to_json())).to_dict() == s.to_dict()
    s.save(tmp_path)
    lib = _library(tmp_path)
    lib.scene("scene")
    assert lib.object("hard").route.route == "rigid" and lib.object("soft").route.route == "soft_vbd"
    assert lib.object("pusher").motion == "kinematic" and lib.object("fixed").motion == "static"
    heavy, surface = _surface(lib, "heavy")
    assert heavy.density.value * 0.03 ** 3 == pytest.approx(0.05) and surface.restitution.value == 0.2
    dense, surface = _surface(lib, "dense")
    assert dense.density.value == 900.0 and surface.mu_static.value == surface.mu_dynamic.value == 0.7
    assert dense.pairs_from is None                               # its own friction replaces the base's pair cards
    assert lib.object("steel").material == "steel_carbon"
    assert _surface(lib, "fixed")[1].mu_dynamic.value == 0.6
    assert lib.object("pusher").material == "wood_beech"          # no value of its own: the card's material


@pytest.mark.parametrize("kw, field", [
    ({"static": True, "kinematic": True}, "kinematic"),
    ({"static": True, "mass": 0.1}, "mass"),
    ({"mass": 0.1, "density": 500.0}, "mass"),
    ({"friction": -1.0}, "friction"),
    ({"restitution": 2.0}, "restitution"),
    ({"material": "steal"}, "material"),
    ({"rigid": "yes"}, "rigid"),
])
def test_add_object_keyword_errors_name_the_field_and_a_fix(kw, field):
    s = Scene()
    with pytest.raises(SceneError) as info:
        s.add_object("cube", "cube_beech_30", **kw)
    assert info.value.field == field and info.value.fix
    assert "cube" not in s.names()


def test_a_card_of_several_materials_refuses_a_value_for_one(tmp_path):
    s = Scene()
    s.add_object("driver", "screwdriver_pz2", friction=0.5)
    with pytest.raises(SceneError, match="several materials") as info:
        s.save(tmp_path)
    assert info.value.field == "friction=" and "roots=" in info.value.fix
    s = Scene()
    s.add_object("driver", "screwdriver_pz2", static=True, kinematic=False, rigid=True)
    s.save(tmp_path / "ok")                                       # motion and route are not material values
    assert _library(tmp_path / "ok").object("driver").part_materials["grip"] == "pp_homopolymer"


@pytest.mark.compile_heavy
@pytest.mark.parametrize("on", ["sponge", "dough"])
def test_the_rigid_fix_of_an_unsupported_pair_works_on_library_objects(on):
    def scene(**rigid):
        s = alloy.Scene(dt=0.01)
        s.add_ground()
        s.add_object("sponge", "sponge_kitchen", pos=(-0.15, 0.0, 0.05), **rigid.get("sponge", {}))
        s.add_object("dough", "dough_ball", pos=(0.15, 0.0, 0.05), **rigid.get("dough", {}))
        return s

    with pytest.raises(BuildError) as info:
        alloy.estimate(scene())
    assert "rigid=True" in info.value.fix
    routes = alloy.estimate(scene(**{on: {"rigid": True}}))["routes"]
    assert routes[on]["route"] == "rigid"


# --- in the engine ----------------------------------------------------------------------------------------------------


def _pushed(card: bool, mu: float) -> float:
    """x travel (m) of a 30 mm beech cube pushed for 0.3 s with half its weight, pair friction ``mu`` on the ground."""
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    if card:
        s.add_object("cube", "cube_beech_30", pos=(0, 0, 0.015))
    else:
        s.add_box("cube", size=(0.03, 0.03, 0.03), pos=(0, 0, 0.015), material="wood_beech")
    s.set_friction("cube", "ground", mu)
    sim = alloy.build(s, device="cpu")
    sim.settle(0.1)
    mass = float(sim.engine.model.body_mass.numpy()[0])
    x0 = float(sim.state(world=0)["cube.pos"][0])
    sim.apply_wrench("cube", force=(0.5 * mass * 9.81, 0.0, 0.0))
    sim.step(30)
    assert sim.check()["ok"]
    return float(sim.state(world=0)["cube.pos"][0]) - x0


@pytest.mark.compile_heavy
def test_a_library_objects_pair_friction_acts_as_a_primitives():
    slide = _pushed(True, 0.2)
    assert slide == pytest.approx(0.5 * 0.3 * 0.3 ** 2 * 9.81, rel=0.05)    # (0.5 - 0.2) g for 0.3 s
    assert slide == pytest.approx(_pushed(False, 0.2), abs=1e-6)
    assert abs(_pushed(True, 0.9)) < 1e-4                                   # held: 0.5 g is below 0.9 g


@pytest.mark.compile_heavy
def test_library_object_keywords_reach_the_engine(tmp_path):
    s = alloy.Scene(worlds=4, dt=0.01)
    s.add_ground()
    s.add_object("cube", "cube_beech_30", pos=(0, 0, 0.015), mass=0.05)
    s.add_object("cell", "aa_cell", pos=(0.1, 0, 0.03), mass=0.02)
    s.add_object("pusher", "cube_beech_30", pos=(0.3, 0, 0.015), kinematic=True)
    s.randomize("cell", "friction", 0.05, 0.8)
    sim = alloy.build(s, device="cpu", seed=0)
    assert sim.engine.model.body_mass.numpy()[:2] == pytest.approx([0.05, 0.02])
    sim.drive("pusher", velocity=(0.1, 0, 0), duration=0.2)
    x0 = sim.state(world=0)["pusher.pos"][0]
    sim.step(20)
    assert sim.state(world=0)["pusher.pos"][0] - x0 == pytest.approx(0.02, abs=1e-4)
    assert sim.check()["ok"]
