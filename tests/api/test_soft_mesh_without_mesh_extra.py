# SPDX-License-Identifier: Apache-2.0
"""A soft-material mesh on an install without the ``mesh`` extra: the error says ``rigid=True`` first.

Laws (CPU, a base install: pytetwild, trimesh and CoACD absent): a mesh of ``vinyl_soft`` (the material of YCB toy
fruit and balls) from ``add_mesh``, and the shipped ``sponge_kitchen`` from ``add_object``, still become soft
bodies by default, so their build raises ``FeatureUnavailableError`` naming the object; its fix names
``rigid=True`` before ``pip install "alloy-engine[mesh]"``. With ``rigid=True`` the same object builds on that
install as a rigid body and rests (``sim.check()`` ok). Before, the fix was the install line alone, though an object
for a robot to pick (a lemon beside an arm) is rigid in most scenes.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

import alloy.api as alloy

_T = (1 + 5 ** 0.5) / 2
#: an icosahedron, 3 cm in radius, faces counter-clockwise seen from outside
ICO_V = 0.03 / np.sqrt(1 + _T * _T) * np.array([
    (-1, _T, 0), (1, _T, 0), (-1, -_T, 0), (1, -_T, 0), (0, -1, _T), (0, 1, _T),
    (0, -1, -_T), (0, 1, -_T), (_T, 0, -1), (_T, 0, 1), (-_T, 0, -1), (-_T, 0, 1)])
ICO_F = np.array([(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
                  (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5),
                  (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)])
MESH_EXTRA = 'pip install "alloy-engine[mesh]"'


def _fruit(scene: alloy.Scene, **kw) -> None:
    scene.add_mesh("lemon", vertices=ICO_V, faces=ICO_F, pos=(0, 0, 0.04), material="vinyl_soft", **kw)


def _sponge(scene: alloy.Scene, **kw) -> None:
    scene.add_object("lemon", "sponge_kitchen", pos=(0, 0, 0.05), **kw)


@pytest.fixture
def base_install(monkeypatch):
    for module in ("pytetwild", "trimesh", "coacd"):       # import fails, as without the mesh and import extras
        monkeypatch.setitem(sys.modules, module, None)


def _scene(add, **kw) -> alloy.Scene:
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    add(scene, **kw)
    return scene


@pytest.mark.parametrize("add", [_fruit, _sponge], ids=["vinyl_soft mesh", "sponge_kitchen"])
def test_the_fix_names_rigid_first_and_the_mesh_extra_second(add, base_install):
    with pytest.raises(alloy.FeatureUnavailableError) as err:
        alloy.build(_scene(add), device="cpu")              # the default stays: a soft body
    fix = err.value.fix
    assert "lemon" in err.value.message and "rigid=True" in fix and MESH_EXTRA in fix
    assert fix.index("rigid=True") < fix.index(MESH_EXTRA)  # was the install line alone


@pytest.mark.compile_heavy
@pytest.mark.parametrize("add", [_fruit, _sponge], ids=["vinyl_soft mesh", "sponge_kitchen"])
def test_rigid_true_builds_on_that_install(add, base_install):
    sim = alloy.build(_scene(add, rigid=True), device="cpu")
    try:
        assert sim.routes()["lemon"]["route"] == "rigid"
        sim.step(20)
        assert sim.check()["ok"]
    finally:
        sim.close()
