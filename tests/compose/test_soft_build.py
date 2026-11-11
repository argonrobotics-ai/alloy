# SPDX-License-Identifier: Apache-2.0
"""OSS-22 end to end on CPU: soft sentences become scene cards with generated soft cards (a resting liquid column, a
drape, a pile, a plugged cable, a pour with a held vessel), the router routes them, the saved folder loads and
passes the load check, and requests that do not fit are reported with what to change."""

from __future__ import annotations

import json

import numpy as np
import pytest

import alloy.check
from alloy.cards.library import CardLibrary, default_root
from alloy.compose.build import compose
from alloy.compose.catalog import Catalog
from alloy.compose.load import library


@pytest.fixture(scope="module")
def cat() -> Catalog:
    return Catalog(library=CardLibrary([default_root()], assets=None))


def _soft(result, kind: str) -> dict:
    return next(r for r in result.scenes[0].soft["items"] if r["kind"] == kind)


def test_water_in_a_mug_rests_on_its_floor_and_the_folder_loads(cat, tmp_path):
    r = compose("a mug with 200 ml of water", check="load", device="cpu", catalog=cat)
    assert r.ok, r.summary()
    row = _soft(r, "liquid")
    assert row["volume_ml"] == 200.0 and row["route"] == "liquid_sph" and row["in"] == "mug_stoneware"
    assert row["start"]["shape"] == "cylinder" and row["level_mm"] < row["capacity_ml"]
    mug, water = (o for o in r.scenes[0].objects)
    assert np.allclose(mug["pose"][:2], water["pose"][:2]) and water["pose"][2] > mug["pose"][2]
    r.save(tmp_path)
    assert (tmp_path / "objects" / f"{row['card']}.yaml").is_file()
    rep = alloy.check.check_load(str(tmp_path / "scenes" / "scene.yaml"), device="cpu",
                                 library=library([str(tmp_path)]))
    assert rep.ok, rep.summary()
    json.dumps(r.to_dict(), allow_nan=False)


def test_drape_and_pile_take_room_on_the_table(cat):
    """A drape and a pile each beside another object (a towel and sand cannot share a scene yet: see
    test_a_solver_pair_the_engine_refuses_is_split_before_any_build)."""
    r = compose("a towel draped over a mug and a jar", check="none", catalog=cat)
    assert r.ok, r.summary()
    objs = {o["item"]: o for o in r.scenes[0].objects}
    assert objs["towel"]["pose"][2] > 0.09 and objs["towel"]["on"] == "mug_stoneware"     # over the 96 mm mug
    r = compose("a pile of 50 g of sand and a mug", check="none", catalog=cat)
    assert r.ok, r.summary()
    pile = _soft(r, "grain")
    mat = cat.library.material(pile["material"])
    assert pile["form"] == "pile" and pile["angle_of_repose_deg"] == pytest.approx(
        np.degrees(mat.plastic.friction_angle.value), abs=0.01)                    # the material's sourced angle
    assert pile["volume_ml"] == pytest.approx(50.0 / (0.6 * mat.density.value) * 1e3, rel=1e-3)  # packed sand
    assert r.scenes[0].soft["settle_s"] >= 0.5


def _plugged_cable(cat) -> None:
    r = compose("a usb cable plugged into a jar", check="none", catalog=cat)
    assert r.ok, r.summary()
    (att,) = r.scenes[0].card.attachments
    assert att.kind == "weld" and att.a.endswith(":0") and att.b == "jar_glass_8oz"
    row = _soft(r, "cable")
    assert row["route"] == "chain" and row["links"] >= 3 and row["attached"] == ["jar_glass_8oz"]


def test_a_folded_towel_is_a_two_layer_mesh_with_self_contact(cat, tmp_path):
    r = compose("a towel folded in half", check="load", device="cpu", catalog=cat)
    assert r.ok, r.summary()
    row = _soft(r, "cloth")
    assert row["form"] == "folded" and row["layers"] == 2
    r.save(tmp_path)
    card = CardLibrary([str(tmp_path), default_root()], assets=None).object(row["card"])
    assert card.features.self_contact and card.shape.path.endswith(".obj")
    assert (tmp_path / card.shape.path).is_file()


def test_plugged_cable_is_welded_to_its_object(cat, monkeypatch):
    from alloy.compose.soft import cards as K

    real = K.route_of                     # a cold chain self-test takes minutes on CPU: route cables as a chain

    def route(lib, name):
        return ("chain", {"links": 14.0}, "") if "cable" in name else real(lib, name)

    monkeypatch.setattr(K, "route_of", route)
    _plugged_cable(cat)


def test_a_free_ended_cable_measures_its_asked_length_in_every_layout(cat, monkeypatch):
    """The engine caches its cable self-test (minutes on a CPU) by the cable's physics, length at 12 digits. A
    plugged end's drop to the table made a 1 m cable 1.0026 m, another length per layout, so every scene, variation
    and seed ran the self-test again (audit EXAMPLES-7: the cable sentence alone took 656 s on a GPU machine)."""
    import math

    from alloy.cards import types as ct
    from alloy.cards.loader import loads_card
    from alloy.compose.soft import cards as K

    real = K.route_of
    monkeypatch.setattr(K, "route_of", lambda lib, name: ("chain", {"links": 14.0}, "") if "cable" in name
                        else real(lib, name))
    seen = set()
    for seed in (0, 1, 2):
        r = compose("a usb cable plugged into a jar", check="none", catalog=cat, seed=seed)
        pts = loads_card(r.generated[_soft(r, "cable")["card"]], ct.ObjectCard).shape.points
        length = sum(math.dist(a, b) for a, b in zip(pts[:-1], pts[1:], strict=True))     # as the router sums it
        seen.add(f"{length:.12g}")
        assert _soft(r, "cable")["length_m"] == 1.0
    assert seen == {"1"}
    assert K.measured(((0, 0, 0), (0.5, 0, 0), (1.2, 0, 0)), 1.0) == ((0, 0, 0), (0.5, 0, 0), (1.0, 0.0, 0.0))    # cut
    assert K.measured(((0, 0, 0), (0.5, 0, 0)), 1.0) == ((0, 0, 0), (1.0, 0.0, 0.0))                   # extended


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_plugged_cable_routes_through_the_engine(cat):
    _plugged_cable(cat)


def test_a_covered_object_goes_down_first_and_a_cable_runs_under_the_cloth(cat, monkeypatch):
    """The jar a towel covers is placed before the larger mug, so the mug keeps out of the towel's spread, and a cable
    plugged into the jar may start under the towel (the sentence of examples/compose/soft_scene.py)."""
    from alloy.compose.soft import cards as K

    real = K.route_of                     # a cold chain self-test takes minutes on CPU: route cables as a chain
    monkeypatch.setattr(K, "route_of", lambda lib, name: ("chain", {"links": 14.0}, "") if "cable" in name
                        else real(lib, name))
    r = compose("a mug with 200 ml of water, a towel draped over a jar and a usb cable plugged into the jar",
                check="none", catalog=cat)
    assert r.ok, r.summary()
    objs = {o["item"]: o for o in r.scenes[0].objects}
    assert objs["towel"]["on"] == "jar_glass_8oz" and objs["mug"]["card"] == "mug_stoneware"
    assert _soft(r, "cable")["attached"] == ["jar_glass_8oz"] and _soft(r, "liquid")["volume_ml"] == 200.0


def test_pour_from_a_held_vessel_is_a_stream(cat):
    r = compose("pour 100 ml of water from a jar into a mug", check="load", device="cpu", catalog=cat)
    assert r.ok, r.summary()                                       # the still copies load (the checks see them)
    card = r.scenes[0].card
    (em,) = card.emitters
    assert em.kind == "emitter" and em.rate * em.stop_s == pytest.approx(1e-4)
    assert {pl.object for pl in card.objects} >= {"mug_stoneware"}            # a plain pour leaves objects free
    assert r.scenes[0].soft["moves_by_design"]
    vessel = next(o for o in r.scenes[0].objects if o["item"] == "jar")
    assert vessel["pose"][2] > em.pose[2]                                # held above the stream's start
    assert any("estimated" in k for k in r.scenes[0].soft) and r.scenes[0].soft["estimated"]


def test_a_saved_pour_loads_as_an_api_pour(cat, tmp_path):
    from alloy.compose.load import load_scene

    r = compose("pour 100 ml of water from a jar into a mug", check="none", catalog=cat)
    r.save(tmp_path)
    scene = load_scene(tmp_path / "scenes" / "scene.yaml")
    row = next(x for x in r.scenes[0].soft["items"] if x.get("form") == "pour")
    pour = scene.items[row["liquid"]]["pour"]
    assert pour["rate"] == pytest.approx(5e-5) and pour["volume"] == pytest.approx(1e-4, rel=0.15)
    assert "emitters" not in scene.extra                       # the card's emitter became the API's pour


def test_too_much_liquid_is_reported_with_the_capacity(cat):
    r = compose("a jar with 400 ml of water", check="none", catalog=cat)
    assert not r.ok
    (miss,) = r.scenes[0].missing
    assert "does not fit" in miss and "at most" in miss


def test_variations_move_the_container_and_keep_its_fill(cat):
    r = compose("a mug with 100 ml of water", variations=2, check="none", catalog=cat)
    assert r.ok and len(r.scenes) == 3
    poses = {tuple(np.round(s.objects[0]["pose"][:2], 4)) for s in r.scenes}
    assert len(poses) == 3
    assert all(any(o["item"] == "water" for o in s.objects) for s in r.scenes)
    assert len({n for s in r.scenes for n in s.cards}) == 3                # one fill card per scene


def test_a_soft_mix_the_engine_cannot_couple_is_reported_with_what_to_drop(cat):
    """Sand beside a towel is refused by the engine; the sand is left out and the water and the towel stay."""
    r = compose("a mug with 100 ml of water, a pile of 50 g of sand and a towel", check="load", device="cpu",
                catalog=cat)
    assert not r.ok
    assert any(m.startswith("sand: not placed: the sand cannot share a scene with the towel yet") and
               "ask for one of them per scene" in m for m in r.scenes[0].missing), r.scenes[0].missing
    assert {"water", "towel", "mug"} <= {o["item"] for o in r.scenes[0].objects}


def test_a_solver_pair_the_engine_refuses_is_split_before_any_build(cat, tmp_path):
    """Cloth beside grains is refused by the engine (``alloy.coupling.unsupported``): the builder plans the scene,
    leaves the rice out with why and what to ask for, and the scene it returns plans without the refusal."""
    from alloy.compose.soft import pairs

    r = compose("a bin of rice and a towel", check="none", catalog=cat)
    assert not r.ok
    scene = r.scenes[0]
    gone = [m for m in scene.missing if m.startswith("rice: not placed: the rice cannot share a scene with the towel")]
    assert gone and "ask for one of them per scene" in gone[0], scene.missing
    assert "not supported in one scene yet" in gone[0] and "Nearest supported" in gone[0]
    names = {o["item"] for o in scene.objects}
    assert {"towel", "bin"} <= names and "rice" not in names
    r.save(tmp_path)
    assert pairs.refused(scene.card, library([str(tmp_path)])) is None


def test_tilting_pour_holds_a_filled_vessel_and_returns_the_tilt(cat):
    r = compose("pour 150 ml of water from a jar tilted at 40 deg/s into a bin", check="none", catalog=cat)
    assert r.ok, r.summary()
    rows = r.scenes[0].soft["items"]
    plan = next(row for row in rows if "pivot" in row)
    vessel = next(row for row in rows if row["kind"] == "vessel")
    fill = next(row for row in rows if row.get("in") == vessel["name"])
    assert plan["vessel"] == vessel["name"] and plan["liquid"] == fill["name"] and fill["volume_ml"] == 150.0
    assert plan["duration_s"] == pytest.approx(3.0) and np.linalg.norm(plan["angular_velocity"]) == pytest.approx(
        np.radians(40.0), rel=1e-4)
    card = {pl.object: pl for pl in r.scenes[0].card.objects}
    bin_top = card["bin_pp_still"].pose[2] + 0.0762                # the bin is held still (a static copy)
    assert card[vessel["name"]].pose[2] > bin_top                   # held clear of the bin's rim
    assert not r.scenes[0].card.emitters and not r.scenes[0].soft["moves_by_design"]
    pivot, centre = np.asarray(plan["pivot"]), np.asarray(plan["opening_centre"])
    assert plan["into"] == "bin_pp" and centre[2] == pytest.approx(bin_top, abs=2e-3)     # rim height
    assert any("held still" in e for e in r.scenes[0].soft["estimated"])
    assert np.linalg.norm(pivot[:2] - centre[:2]) < plan["opening_radius_m"]             # the lip is over the opening


def test_tilt_starts_a_drive_about_the_lip():
    from types import SimpleNamespace as NS

    from alloy.compose import tilt

    calls = []
    sim = NS(drive=lambda name, **kw: calls.append((name, kw)) or "sim")
    plan = {"vessel": "jar", "angular_velocity": [0.0, 0.7, 0.0], "pivot": [0.1, 0.0, 0.2], "duration_s": 3.0}
    assert tilt(sim, plan) == "sim"
    assert calls == [("jar", {"angular_velocity": (0.0, 0.7, 0.0), "pivot": (0.1, 0.0, 0.2), "duration": 3.0})]
