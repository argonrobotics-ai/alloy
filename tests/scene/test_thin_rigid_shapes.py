# SPDX-License-Identifier: Apache-2.0
"""Audit GAPS-7: a solid ``add_*`` shape under the thin-member limit says what builds, and the stiff-sheet gap page's
workaround (a rigid plate for a stiff sheet) runs on the page's own rig.

The card check refuses a shape under 2 x contact speed x rigid step (1.25 mm, whatever ``Scene(dt=)``) and names
``features.min_wall``. ``alloy.api._materialize`` raises a ``SceneError`` with the numbers first. The limit stays for
free parts: with the wall declared, a 0.5 mm plate dropped 5 mm onto a 0.5 mm shelf fell through it (CPU). A static
or kinematic wall builds down to 1 mm with nothing declared (liquid fix 3, physics 2026-11-17: the liquid holds in
1 mm glass, 0 leaks), and beside a free body or robot the build warns once with the speed that can pass it."""

from __future__ import annotations

import logging

import numpy as np
import pytest

import alloy
from alloy.api import _materialize
from alloy.api.errors import SceneError

TUNNEL = "can pass through it"          # the thin-wall warning's words


@pytest.fixture(autouse=True)
def _fresh_wall_warnings(monkeypatch):
    """The thin-wall warning is once per wall name per process: each law starts with none given."""
    monkeypatch.setattr(_materialize, "_WALL_WARNED", set())


def _warned(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "alloy.api" and TUNNEL in r.getMessage()]


@pytest.mark.parametrize("add, field, thin, need", [
    (lambda s: s.add_box("sheet", size=(0.2, 0.1, 0.0005), material="pet", rigid=True), "size", "0.5 mm", "1.25 mm"),
    (lambda s: s.add_cylinder("disc", 0.02, 0.001), "height", "1 mm", "1.25 mm"),
    (lambda s: s.add_box("plate", size=(0.05, 0.05, 0.001), rigid=True), "size", "1 mm", "1.25 mm"),   # free: 1.25
    (lambda s: s.add_sphere("bead", 0.0004), "radius", "0.8 mm", "1.25 mm"),
    (lambda s: s.add_open_box("tray", inner=(0.1, 0.1, 0.02), wall=0.0009), "size", "0.9 mm", "1 mm"),
    (lambda s: s.add_box("shelf", size=(0.1, 0.1, 0.0009), static=True), "size", "0.9 mm", "1 mm"),
])
def test_a_shape_under_the_thin_limit_says_what_builds(tmp_path, add, field, thin, need):
    s = alloy.Scene()
    s.add_ground()
    add(s)
    with pytest.raises(SceneError) as err:
        s.save(tmp_path)
    e = err.value
    assert e.field == field and f"is {thin} at its thinnest" in e.message and f"at least {need} (" in e.message
    assert "a moving part 1.25 mm" in e.message
    assert "mass=" in e.fix and "add_cloth" in e.fix and "dt" in e.fix
    assert "GOTCHAS" not in e.message + e.fix
    assert "rigid=True, card_extra={'features': {'min_wall'" in e.fix or "takes no card_extra" in e.fix


def test_shapes_at_the_thin_limit_or_declaring_a_wall_still_validate(tmp_path):
    from alloy.cards.library import CardLibrary, default_root

    s = alloy.Scene()
    s.add_ground()
    s.add_box("plate", size=(0.2, 0.1, 0.00125), pos=(0, 0, 0.06), material="pet", mass=0.0135, rigid=True)
    s.add_cloth("film", size=(0.2, 0.1), pos=(0, 0.3, 0.06), thickness=0.0005, resolution=10)
    wall = {"value": 0.0005, "source": "model", "note": "test: declared on the card"}
    s.add_box("shim", size=(0.02, 0.02, 0.0005), pos=(0.3, 0, 0.06), card_extra={"features": {"min_wall": wall}})
    s.save(tmp_path)
    CardLibrary(roots=[tmp_path, default_root()], assets=None).scene("scene")


@pytest.mark.parametrize("motion", ["static", "kinematic"])
def test_a_1mm_open_cylinder_builds_static_and_kinematic(motion, caplog):
    """1 mm glass, nothing declared: the cup builds and steps; with no free body beside it, no warning."""
    caplog.set_level(logging.WARNING, logger="alloy.api")
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_open_cylinder("vial", inner_radius=0.013, depth=0.06, wall=0.001, material="glass_soda_lime",
                        **({"kinematic": True} if motion == "kinematic" else {}))
    sim = alloy.build(s, device="cpu")
    sim.step(2)
    assert sim.check()["ok"], sim.check()["problems"]
    assert not _warned(caplog), _warned(caplog)


def _p1(s: alloy.Scene) -> None:
    """P1 ([internal note] notes/liquid-fixes-1116.md): a 20 ml vial, OD 28 mm, 1 mm glass, 61 mm tall, its axis 4 mm
    outboard of the source beaker's lip and its top 77 mm below it; the source a kinematic 100 ml beaker (inner
    radius 25 mm, inner height 70 mm, 2 mm glass, base at z = 66 mm) holding 60 ml of water, 30.6 mm deep."""
    s.add_ground()
    s.add_open_cylinder("vial", inner_radius=0.013, depth=0.06, wall=0.001, pos=(0.030, 0.0, 0.0),
                        material="glass_soda_lime")
    s.add_open_cylinder("beaker", inner_radius=0.025, depth=0.070, wall=0.002, pos=(0.0, 0.0, 0.066),
                        material="glass_soda_lime", kinematic=True)
    s.add_liquid("water", size=(0.05, 0.05, 0.0306), pos=(0.0, 0.0, 0.068 + 0.0153), shape="cylinder")


def test_the_p1_vial_builds_through_the_public_api(caplog):
    """Liquid fix 3's case with nothing but ``alloy.Scene`` calls: it builds and steps with health OK, and no
    thin-wall warning (no free body)."""
    caplog.set_level(logging.WARNING, logger="alloy.api")
    s = alloy.Scene(dt=0.01)
    _p1(s)
    sim = alloy.build(s, device="cpu")
    sim.step(3)
    assert sim.check()["ok"], sim.check()["problems"]
    assert sim.routes()["water"]["route"] == "liquid_sph" and sim.routes()["vial_floor"]["route"] == "static"
    assert not _warned(caplog), _warned(caplog)


@pytest.mark.parametrize("beside", ["box", "robot", "nothing", "static box", "kinematic box"])
def test_a_thin_wall_beside_a_free_body_warns_once_with_the_speed(tmp_path, caplog, beside):
    """Saved, not built (the cards are written the same way): a free body or robot beside 1 mm walls warns once per
    wall with thin / (2 x 1.25 ms) = 0.4 m/s and the known-gap page; static or kinematic neighbours say nothing."""
    caplog.set_level(logging.WARNING, logger="alloy.api")
    s = alloy.Scene(dt=0.01)
    _p1(s)
    if beside == "robot":
        s.add_robot("arm", mjcf=ARM, pos=(0.3, 0.0, 0.0))
    elif beside != "nothing":
        kw = {"static": True} if beside == "static box" else {"kinematic": True} if beside == "kinematic box" else {}
        s.add_box("block", size=(0.03, 0.03, 0.03), pos=(0.2, 0.0, 0.015), **kw)
    s.save(tmp_path / "a")
    s.save(tmp_path / "b")                                          # once per wall, not once per save
    got = _warned(caplog)
    if beside in ("box", "robot"):
        assert len(got) == 1 and got[0].startswith("vial: a wall 1 mm thick") and "0.4 m/s" in got[0], got
        assert "(health flags TUNNEL)" in got[0] and "docs/known_gaps/rigid-tunnelling.md" in got[0]
        assert ("block" if beside == "box" else "arm") in got[0]
    else:
        assert not got, got


def test_the_warning_promises_the_tunnel_flag_only_for_a_static_wall(tmp_path, caplog):
    """WP5's rc19 review: TUNNEL judges static shapes only (``alloy.precision.health_tunnel``), so a 2 mm steel ball
    dropped 1 m through a 1 mm kinematic shelf left ``sim.check()`` ok while the warning said health flags it. A
    kinematic wall's warning, a box or a fixture, says health does not flag a pass; a static wall's names TUNNEL."""
    caplog.set_level(logging.WARNING, logger="alloy.api")
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_box("shelf", size=(0.2, 0.2, 0.001), pos=(0.0, 0.0, 0.3), static=True)
    s.add_box("tray", size=(0.2, 0.2, 0.001), pos=(0.4, 0.0, 0.3), kinematic=True)
    s.add_open_cylinder("jug", inner_radius=0.03, depth=0.06, wall=0.001, pos=(-0.4, 0.0, 0.0), kinematic=True)
    s.add_sphere("ball", 0.001, pos=(0.0, 0.0, 1.3), material="steel_carbon")
    s.save(tmp_path)
    got = {m.split(":")[0]: m for m in _warned(caplog)}
    assert sorted(got) == ["jug", "shelf", "tray"], got
    assert "(health flags TUNNEL)" in got["shelf"] and "does not flag" not in got["shelf"], got["shelf"]
    for wall in ("tray", "jug"):
        assert "(health does not flag a pass through a kinematic wall)" in got[wall], got[wall]
        assert "TUNNEL" not in got[wall] and "0.4 m/s" in got[wall] and "ball" in got[wall], got[wall]


ARM = """<mujoco><worldbody><body name="link1" pos="0 0 0.1"><joint name="shoulder" type="hinge" axis="0 1 0"/>
  <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.02" mass="1"/></body></worldbody></mujoco>"""


@pytest.mark.compile_heavy
def test_a_stiff_sheet_as_a_rigid_plate_rests_on_two_round_supports():
    """Known gap cloth-stiff-sheet-sag's workaround on its own rig: its 0.5 mm PET sheet as a rigid box 1.25 mm thick
    (the thinnest that builds) with the sheet's mass, dropped 5 mm onto two 1 cm round supports 10 cm apart. It rests
    on the support tops, centred and still."""
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    for k, x in enumerate((-0.05, 0.05)):
        scene.add_cylinder(f"sup{k}", 0.01, 0.2, pos=(x, 0.0, 0.04), rpy=(np.pi / 2, 0.0, 0.0), static=True)
    scene.add_box("sheet", size=(0.2, 0.1, 0.00125), pos=(0.0, 0.0, 0.055625), material="pet", mass=0.0135,
                  rigid=True)
    sim = alloy.build(scene, device="cpu")
    tr = sim.run(1.0, record=["sheet"], every=5).world(0)
    p, t = tr["sheet.pos"], tr["t"]
    v = np.linalg.norm(np.diff(p, axis=0), axis=1) / np.diff(t)
    assert sim.routes()["sheet"]["route"] == "rigid" and sim.check()["ok"]
    assert p[-1, 2] == pytest.approx(0.050625, abs=5e-5) and abs(p[-1, 0]) < 1e-3   # on the tops (z 0.05), centred
    assert v[t[1:] >= 0.5].max() < 1e-3
    sim.close()
