# SPDX-License-Identifier: Apache-2.0
"""Audit GAPS-7: a solid ``add_*`` shape under the thin-member limit says what builds, and the stiff-sheet gap page's
workaround (a rigid plate for a stiff sheet) runs on the page's own rig.

The card check refuses a shape under 2 x contact speed x rigid step (1.25 mm, whatever ``Scene(dt=)``) and names
``features.min_wall``, which ``add_box`` has no argument for. ``alloy.api._materialize`` raises a ``SceneError`` with
the numbers first. The limit stays: with the wall declared, a 0.5 mm plate dropped 5 mm onto a 0.5 mm shelf fell
through it (CPU). The same shapes fail and build as before; only the error changed."""

from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.api.errors import SceneError


@pytest.mark.parametrize("add, field, thin", [
    (lambda s: s.add_box("sheet", size=(0.2, 0.1, 0.0005), material="pet", rigid=True), "size", "0.5 mm"),
    (lambda s: s.add_cylinder("disc", 0.02, 0.001), "height", "1 mm"),
    (lambda s: s.add_sphere("bead", 0.0004), "radius", "0.8 mm"),
    (lambda s: s.add_open_box("tray", inner=(0.1, 0.1, 0.02), wall=0.001), "size", "1 mm"),
])
def test_a_shape_under_the_thin_limit_says_what_builds(tmp_path, add, field, thin):
    s = alloy.Scene()
    s.add_ground()
    add(s)
    with pytest.raises(SceneError) as err:
        s.save(tmp_path)
    e = err.value
    assert e.field == field and f"is {thin} at its thinnest" in e.message and "at least 1.25 mm" in e.message
    assert "mass=" in e.fix and "add_cloth" in e.fix and "dt" in e.fix
    assert "GOTCHAS" not in e.message + e.fix and "min_wall" not in e.message + e.fix


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
