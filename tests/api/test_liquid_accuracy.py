# SPDX-License-Identifier: Apache-2.0
"""``liquid_accuracy`` (SCR-WP9-LIQRES, user decision 2026-11-08): "default" 4 mm liquid particles, "super" 2 mm.
The Scene field and its round trip, the router's spacing per setting, the estimate's two rows, and the one-line warning:
it fires below each threshold (an opening under 8 particles across or under 24 mm; a pour stream under 2 across) and
stays silent at and above it."""

from __future__ import annotations

import logging

import pytest

import alloy
from alloy.api import _liquid_accuracy as LA
from alloy.api.errors import SceneError
from alloy.route.types import LIQUID_SPACING


def _items(opening: float | None = None, stream: float | None = None) -> dict:
    items: dict = {"water": {"kind": "liquid"}}
    if opening is not None:
        items["cup_floor"] = {"kind": "box", "fixture": "cup", "opening": opening}
        items["cup_wall0"] = {"kind": "box", "fixture": "cup", "opening": opening}
    if stream is not None:
        items["water"]["pour"] = {"rate": 0.5 * stream ** 2, "velocity": [0.0, 0.0, -0.5]}
    return items


def test_the_setting_is_a_scene_field_that_round_trips():
    assert alloy.Scene().liquid_accuracy == "default"
    sc = alloy.Scene(liquid_accuracy="super")
    assert alloy.Scene.from_dict(sc.to_dict()).liquid_accuracy == "super"
    assert alloy.Scene.from_dict({k: v for k, v in sc.to_dict().items() if k != "liquid_accuracy"}).liquid_accuracy \
        == "default"                                     # Scene JSON written before the field
    with pytest.raises(SceneError) as err:
        alloy.Scene(liquid_accuracy="ultra")
    assert err.value.field == "liquid_accuracy" and "super" in err.value.fix
    with pytest.raises(SceneError):
        sc.liquid_accuracy = "fine"                     # checked when set (OSS-17's review), not first at build
    assert sc.liquid_accuracy == "super"
    assert LA.SPACING == dict(LIQUID_SPACING) == {"default": 4.0e-3, "super": 2.0e-3}


@pytest.mark.law
@pytest.mark.parametrize(("acc", "silent_at", "warns_at"), [("default", 0.032, 0.0319), ("super", 0.024, 0.0239)])
def test_an_opening_warns_below_its_threshold_only(acc, silent_at, warns_at):
    """8 particles across at default (32 mm); at super the 24 mm floor binds (8 across would be 16 mm)."""
    s = LA.spacing(acc)
    assert LA.coarse(_items(opening=silent_at), s) == []
    found = LA.coarse(_items(opening=warns_at), s)
    assert [(c.item, c.what) for c in found] == [("cup", "opening")]           # once per fixture, not per part
    assert found[0].across == pytest.approx(warns_at / s, abs=0)


@pytest.mark.law
@pytest.mark.parametrize("acc", ["default", "super"])
def test_a_pour_stream_warns_below_two_particles_across_only(acc):
    s = LA.spacing(acc)
    assert LA.coarse(_items(stream=2.0 * s), s) == []
    assert [c.what for c in LA.coarse(_items(stream=1.99 * s), s)] == ["stream"]


def test_no_liquid_no_warning_and_cards_count():
    items = {"cup_floor": {"kind": "box", "fixture": "cup", "opening": 0.01}}
    assert LA.coarse(items, 4e-3) == []
    found = LA.coarse({"water": {"kind": "liquid"}}, 4e-3, cards=[("water's container", 0.02)])
    assert [(c.item, c.width) for c in found] == [("water's container", 0.02)]


def test_the_warning_is_one_line_naming_the_fix():
    found = LA.coarse(_items(opening=0.028, stream=0.005), LA.spacing("default"))
    msg = LA.message(found, "default")
    assert "\n" not in msg and msg.count("liquid_accuracy='super'") >= 2
    assert "cup: opening 28.0 mm = 7.0 particles across" in msg and "stream 5.0 mm" in msg
    under_floor = LA.message(LA.coarse(_items(opening=0.02), 2e-3), "super")
    assert "20.0 mm = 10.0 particles across (needs 24 mm): widen it to 24 mm" in under_floor     # not "needs 8"
    assert "(needs 8 across)" in msg
    assert LA.message(LA.coarse(_items(stream=0.003), 2e-3), "super").endswith("lower its speed")


def _cup_scene(acc: str, inner_radius: float) -> alloy.Scene:
    sc = alloy.Scene(dt=0.01, liquid_accuracy=acc)
    sc.add_ground()
    sc.add_open_cylinder("cup", inner_radius=inner_radius, depth=0.04)
    sc.add_liquid("water", size=(0.02, 0.02, 0.016), pos=(0.0, 0.0, 0.012), shape="cylinder")
    return sc


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if "too coarse" in r.getMessage()]


def test_a_build_routes_at_the_setting_and_warns_once(caplog):
    """A 28 mm cup: 7 particles across at default (one warning naming super), 14 at super (silent, r = 1 mm)."""
    caplog.set_level(logging.WARNING, logger="alloy.api")
    sim = alloy.build(_cup_scene("default", 0.014), device="cpu")
    numbers = sim.built.report.decisions["water"].numbers
    assert numbers["particle_radius"] == 2e-3 and numbers["liquid_spacing"] == 4e-3
    assert "liquid_accuracy 'default', spacing 4 mm" in sim.routes()["water"]["reasons"][-1]
    msgs = _warnings(caplog)
    assert len(msgs) == 1 and "cup: opening 28.0 mm = 7.0 particles across" in msgs[0], msgs
    sim.close()
    caplog.clear()
    sim = alloy.build(_cup_scene("super", 0.014), device="cpu")
    assert sim.built.report.decisions["water"].numbers["particle_radius"] == 1e-3
    assert _warnings(caplog) == []
    sim.close()


def test_estimate_shows_both_settings():
    est = alloy.estimate(_cup_scene("default", 0.02), 2, 0.5, device="cpu")
    rows = est["liquid"]
    assert est["liquid_accuracy"] == "default" and set(rows) == {"default", "super"}
    assert rows["default"]["spacing_mm"] == 4.0 and rows["super"]["spacing_mm"] == 2.0
    assert rows["super"]["particles"] > 6 * rows["default"]["particles"]
    # 2x the steps and 8x the particles, but each step's fixed SPH cost (FIX-LIST 7 fit) is shared: 6.6x on this
    # 2 cm cup (FIX-LIST 7 moved this pin from 8x), towards 16x as the particles grow
    assert rows["super"]["wall_time_s"] > 5 * rows["default"]["wall_time_s"]
    assert all(r["accuracy"] for r in rows.values())
    dry = alloy.Scene()
    dry.add_ground()
    dry.add_box("b", size=(0.1, 0.1, 0.1), pos=(0, 0, 0.05), mass=1.0)
    assert "liquid" not in alloy.estimate(dry, 1, 0.1, device="cpu")


def test_scene_json_is_unchanged_at_the_default():
    """Coordinator 02:58Z: to_dict omits the field at the default, so scene JSON (and anything hashing it) written
    before the switch is unchanged; super is written and read back."""
    sc = alloy.Scene(dt=0.01)
    sc.add_ground()
    sc.add_liquid("water", size=(0.02, 0.02, 0.02), pos=(0.0, 0.0, 0.05))
    d = sc.to_dict()
    assert "liquid_accuracy" not in d and "liquid_accuracy" not in sc.to_json()
    sc.liquid_accuracy = "super"
    assert sc.to_dict()["liquid_accuracy"] == "super"
    assert {k: v for k, v in sc.to_dict().items() if k != "liquid_accuracy"} == d


@pytest.mark.law
def test_a_pour_at_super_delivers_its_volume_into_the_cup(caplog):
    """WP2's review: add_pour sizes the reservoir and nozzle in whole 4 mm particles; the build re-sizes them for
    super's 2 mm (``_pour.at_spacing``: before, the reservoir held 10.65 ml for 8 ml and the emitter warned that its
    layers over-pack). 8 ml at 40 ml/s into a 40 mm cup: the reservoir holds 8 ml, the nozzle packs whole layers (no
    emitter warning), every particle is released and at least 97 % land in the cup, healthy; the run record names
    the setting."""
    caplog.set_level(logging.WARNING)
    sc = alloy.Scene(dt=0.01, liquid_accuracy="super")
    sc.add_ground()
    sc.add_open_cylinder("cup", inner_radius=0.02, depth=0.04)
    sc.add_pour("water", 8e-6, pos=(0.0, 0.0, 0.08), rate=4e-5, speed=0.5)
    sim = alloy.build(sc, device="cpu")
    assert sim.built.report.decisions["water"].numbers["particle_radius"] == 1e-3
    traj = sim.run(0.45, record=["water"], every=45)
    st = sim.state(["water"], world=0)
    pts, act = st["water.points"], st["water.active"].astype(bool)
    assert act.all()                                                           # the reservoir is fully poured
    inside = (pts[:, 0] ** 2 + pts[:, 1] ** 2 < 0.02 ** 2) & (pts[:, 2] < 0.004 + 0.04 + 0.002)
    assert inside.mean() >= 0.97, inside.mean()
    assert sim.liquid_volume("water") == pytest.approx(8e-6, rel=1e-9, abs=0)      # 10 x 10 x 10 particles of 2 mm
    assert not [r for r in caplog.records if "over/under-pack" in r.getMessage() or "not the 8 ml" in r.getMessage()]
    assert sim.check()["ok"] and traj.meta["liquid_accuracy"] == "super"
    assert sc.items["water"]["pour"]["volume"] == pytest.approx(8e-6, rel=1e-9, abs=0)  # the user's scene untouched
    sim.close()
