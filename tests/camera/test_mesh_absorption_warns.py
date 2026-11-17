# SPDX-License-Identifier: Apache-2.0
"""Law: a look that tints a mesh by volume absorption warns once per material under ``quality="full"``.

The full-material camera draws ``absorption_color`` / ``absorption_m`` only inside simulated liquids; on a mesh (a
tinted glass, a dyed liquid modelled as a mesh) the material renders untinted. It said nothing about it. Now the
first material tree of such a look warns, naming the material, once per process, and the image is the untinted one
the warning describes. A liquid's own absorption, an untinted colour and a material without absorption stay quiet.
"""
from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.camera.liquid_looks import liquid_material
from alloy.camera.looks import material as material_mod
from alloy.camera.looks import material_tree

VIEW = ((0.3, -0.3, 0.2), (0.0, 0.0, 0.05))
DYED = {"preset": "glass_clear", "absorption_color": [0.9, 0.1, 0.1], "absorption_m": 0.01}
LOGGER = material_mod.__name__


@pytest.fixture(autouse=True)
def fresh_warnings(monkeypatch):
    monkeypatch.setattr(material_mod, "_ABSORPTION_WARNED", set())


@pytest.fixture(scope="module")
def sim():
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_box("cube", size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.05), mass=1.0)
    s = alloy.build(scene, device="cpu")
    yield s
    s.close()


def _render(sim, material: dict) -> np.ndarray:
    look = {"materials": {"dyed": material}, "assign": [{"match": "cube", "material": "dyed"}]}
    return sim.render(width=48, height=32, spp=4, quality="full", look=look, look_at=VIEW).astype(np.float64)


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == LOGGER and r.levelname == "WARNING"]


def test_a_look_that_tints_a_mesh_by_absorption_warns_once_and_renders_untinted(sim, caplog):
    with caplog.at_level("WARNING", logger=LOGGER):
        dyed = _render(sim, DYED)
        again = _render(sim, DYED)
    msgs = _warnings(caplog)
    assert len(msgs) == 1, msgs
    assert "'dyed'" in msgs[0] and "absorption" in msgs[0] and "untinted" in msgs[0]
    clear = _render(sim, {"preset": "glass_clear", "absorption_color": [1.0, 1.0, 1.0], "absorption_m": 0.01})
    assert np.abs(dyed - again).mean() == 0.0 and np.abs(dyed - clear).mean() == 0.0   # what the warning says


def test_a_preset_that_absorbs_warns_too_on_a_mesh(caplog):
    with caplog.at_level("WARNING", logger=LOGGER):
        material_tree("pane", "glass_clear")
        material_tree("pane", "glass_clear")
    msgs = _warnings(caplog)
    assert len(msgs) == 1 and "'pane'" in msgs[0], msgs


@pytest.mark.parametrize("make", [
    lambda: liquid_material("water"),                     # the liquid draws its absorption: no warning
    lambda: material_tree("pane", "glass_clear", volume=True),
    lambda: material_tree("white", {"preset": "glass_clear", "absorption_color": [1.0, 1.0, 1.0]}),
    lambda: material_tree("pad", "rubber_black"),
], ids=["sph_liquid", "caller_adds_volume", "untinted_colour", "no_absorption"])
def test_absorption_that_is_drawn_or_absent_stays_quiet(make, caplog):
    with caplog.at_level("WARNING", logger=LOGGER):
        make()
    assert _warnings(caplog) == []
