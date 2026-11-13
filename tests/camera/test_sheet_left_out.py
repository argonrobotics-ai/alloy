# SPDX-License-Identifier: Apache-2.0
"""Law: the variants contact sheet names what it does not draw (Gemini audit S2-019).

``alloy.vary(..., contact_sheet=True)`` renders each tile from the base assembly, which holds the shapes of rigid
objects only, so cloth, film, soft bodies, MPM materials and liquids are on no tile. The sheet's ``left_out`` was
empty all the same: a sheet of a table with a towel and a block of water was pixel for pixel the sheet of the empty
table, with nothing reported. Now each such item is named in ``left_out`` (the variant files keep it; ``alloy render``
of a variant draws it). The rigid tiles are unchanged: the sheet of the scene with the towel and water is the sheet
of the scene without them, bit for bit, until the sheet draws particles (open: 0.1.x).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

SHEET = {"width": 96, "height": 72, "spp": 1, "quality": "fast"}

pytestmark = pytest.mark.timeout(600)


def _scene(particles: bool):
    import alloy
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_box("table", size=(0.55, 0.55, 0.40), pos=(0.0, 0.0, 0.20), static=True)
    s.add_box("marker1", size=(0.04, 0.04, 0.04), pos=(-0.22, -0.22, 0.42), mass=0.25)
    s.add_box("marker2", size=(0.04, 0.04, 0.04), pos=(0.22, 0.22, 0.42), mass=0.25)
    if particles:
        s.add_cloth("towel", size=(0.22, 0.22), pos=(-0.06, -0.04, 0.41), resolution=10, mass=0.1)
        s.add_liquid("puddle", size=(0.10, 0.10, 0.04), pos=(0.09, 0.08, 0.43), material="water")
    return s


def _sheet(tmp_path, particles: bool) -> tuple[list[str], np.ndarray]:
    """(left_out, contact sheet image) of one variant of the scene, nothing moved."""
    import alloy
    from PIL import Image
    out = tmp_path / ("with" if particles else "without")
    alloy.vary(_scene(particles), 1, pos=0.0, yaw_deg=0.0, textures="keep", tint=0.0, out=out, contact_sheet=True,
               sheet=SHEET)
    sheet = json.loads((out / "variants.json").read_text())["contact_sheet"]
    return sheet["left_out"], np.asarray(Image.open(out / sheet["file"]))


def test_the_contact_sheet_names_the_cloth_and_liquid_it_does_not_draw(tmp_path, caplog):
    left_out, img = _sheet(tmp_path, particles=True)
    named = {line.split(":")[0]: line for line in left_out}
    assert set(named) == {"towel", "puddle"}, left_out
    assert "cloth_vbd" in named["towel"] and "liquid_sph" in named["puddle"], left_out
    assert "stepped as particles are not drawn: towel, puddle" in caplog.text
    rigid_left_out, rigid = _sheet(tmp_path, particles=False)
    assert rigid_left_out == []                                    # a rigid scene: nothing left out
    assert np.array_equal(img, rigid)                              # the rigid tiles are unchanged
