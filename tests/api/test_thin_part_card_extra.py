# SPDX-License-Identifier: Apache-2.0
"""A part thinner than 1.25 mm from ``alloy.Scene``: the error's fix names the form that keeps it, pasted.

Laws (CPU): a 0.2 mm tab (``add_box``), a 0.61 mm wire (``add_capsule``), a 0.5 mm disc (``add_cylinder``) and a
0.3 mm film (``add_mesh``) fail the build as before; each fix holds ``rigid=True, card_extra={...}`` declaring the
part's thinnest member, and the same call with that pasted in builds, rests 100 periods on the ground as a rigid
body and passes ``sim.check()``. The speed the fix names holds: the tab dropped 2 mm (0.2 m/s, past the 0.08 m/s it
names) fails the check. No message or fix cites GOTCHAS (a file that does not ship). ``add_sphere`` and the
fixtures take no ``card_extra``, and their fix says so (``add_sphere(card_extra=...)`` is a ``TypeError``); a thin
``add_rod`` is told the radius that builds. Before, the primitives' fix named only a thicker part, and the mesh's
``BuildError`` said "change the field named in the message" with ``GOTCHAS:413``, though ``card_extra=`` (with
``rigid=True``: without it a part this thin is made a cloth sheet or a chain) was the only way through.
"""

from __future__ import annotations

import ast

import numpy as np
import pytest

import alloy.api as alloy

#: a 5 x 5 cm film 0.3 mm thick, faces counter-clockwise seen from outside
FILM_V = np.array([[0, 0, 0], [.05, 0, 0], [0, .05, 0], [.05, .05, 0],
                   [0, 0, 3e-4], [.05, 0, 3e-4], [0, .05, 3e-4], [.05, .05, 3e-4]])
FILM_F = np.array([[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 5], [0, 5, 4], [1, 3, 7], [1, 7, 5],
                   [3, 2, 6], [3, 6, 7], [2, 0, 4], [2, 4, 6]])

THIN = {                                                     # each resting on the ground
    "tab": lambda s, **k: s.add_box("tab", size=(0.02, 0.02, 0.0002), pos=(0, 0, 0.0001), mass=0.001, **k),
    "wire": lambda s, **k: s.add_capsule("wire", 0.000305, 0.05, pos=(0, 0, 0.000305), rpy=(0, 1.5708, 0),
                                         density=1100, **k),
    "disc": lambda s, **k: s.add_cylinder("disc", 0.02, 0.0005, pos=(0, 0, 0.00025), mass=0.001, **k),
    "film": lambda s, **k: s.add_mesh("film", vertices=FILM_V, faces=FILM_F, mass=0.001, **k),
}


def _scene(add, **kw) -> alloy.Scene:
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    add(scene, **kw)
    return scene


def _refused(scene: alloy.Scene) -> alloy.AlloyAPIError:
    with pytest.raises((alloy.SceneError, alloy.BuildError)) as err:
        alloy.build(scene, device="cpu")
    e = err.value
    assert "GOTCHAS" not in e.message + e.fix, e.message
    return e


def _pasted(fix: str) -> dict:
    """The ``card_extra={...}`` literal in ``fix``, parsed as the user would paste it."""
    start = fix.index("card_extra=") + len("card_extra=")
    depth = 0
    for i, ch in enumerate(fix[start:], start):
        depth += {"{": 1, "}": -1}.get(ch, 0)
        if depth == 0:
            return ast.literal_eval(fix[start:i + 1])
    raise AssertionError(f"no closed card_extra literal in {fix!r}")


def _pasted_form(fix: str) -> dict:
    """``rigid=True, card_extra={...}`` from ``fix`` as the keywords the user pastes into the call."""
    assert "rigid=True, card_extra={" in fix, fix
    return {"rigid": True, "card_extra": _pasted(fix)}


def _rests(scene: alloy.Scene, name: str) -> dict:
    sim = alloy.build(scene, device="cpu")
    try:
        assert sim.routes()[name]["route"] == "rigid"
        sim.step(100)
        return sim.check()
    finally:
        sim.close()


@pytest.mark.compile_heavy
@pytest.mark.parametrize("name", list(THIN))
def test_the_fix_pasted_into_the_same_call_builds_and_rests(name):
    form = _pasted_form(_refused(_scene(THIN[name])).fix)    # was: no card_extra named (GOTCHAS:413 for the film)
    assert form["card_extra"]["features"]["min_wall"]["source"] == "model"
    report = _rests(_scene(THIN[name], **form), name)
    assert report["ok"], report["problems"]


@pytest.mark.compile_heavy
def test_the_speed_the_fix_names_holds():
    fix = _refused(_scene(THIN["tab"])).fix
    assert "faster than 0.08 m/s" in fix                     # 0.2 mm / (2 x 1.25 ms)
    dropped = alloy.Scene(dt=0.01)
    dropped.add_ground()
    dropped.add_box("tab", size=(0.02, 0.02, 0.0002), pos=(0, 0, 0.0021), mass=0.001, **_pasted_form(fix))
    assert not _rests(dropped, "tab")["ok"]                  # 2 mm: 0.2 m/s at impact, 1 mm deep


def test_a_sphere_and_a_fixture_are_told_they_take_no_card_extra():
    bead = _refused(_scene(lambda s: s.add_sphere("bead", 0.0005, pos=(0, 0, 0.02), mass=0.0001)))
    assert "add_sphere takes no card_extra" in bead.fix and "card_extra={" not in bead.fix
    with pytest.raises(TypeError, match="card_extra"):
        alloy.Scene().add_sphere("bead", 0.0005, card_extra={})
    tray = _refused(_scene(lambda s: s.add_open_box("tray", inner=(0.1, 0.1, 0.02), wall=0.001)))
    assert "add_open_box / add_open_cylinder / add_block_with_hole takes no card_extra" in tray.fix


@pytest.mark.compile_heavy
def test_a_thin_rod_is_told_the_radius_that_builds():
    err = _refused(_scene(lambda s: s.add_rod("cord", radius=0.000305, length=0.1, pos=(0, 0, 0.05))))
    assert "add_rod: a radius of at least 0.625 mm" in err.fix
    assert "add_sphere, add_rod and the fixtures take no card_extra" in err.fix
