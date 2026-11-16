# SPDX-License-Identifier: Apache-2.0
"""A kinematic part too small for MuJoCo's inertia limit builds at its own density beside a moving body.

Law (CPU): a 0.37 mm thread segment (a closed box mesh 0.37 x 0.37 x 0.56 mm), kinematic, at the default rigid
material's density, beside a free ball (so it steps in MuJoCo with the ball), builds with no inertia warning; its mass
is the one its density gives (only its inertia is raised), ``sim.set_pose`` moves it and the check is ok. Before,
MuJoCo refused the build with a bare ``ValueError`` ("inertia must have positive eigenvalues in fullinertia"): its
smallest principal moment, about 2e-15 kg m^2, is under the 1e-14 MuJoCo takes for a full inertia tensor, so a posed
part needed a raised density. The assembly rule itself (the kinematic floor, MuJoCo's limit, a tiny free body's
warning) has its laws in ``tests/scene/test_inertia.py``.
"""

from __future__ import annotations

import warnings

import numpy as np

import alloy.api as alloy
from alloy.api.scene import RIGID_MATERIAL
from alloy.cards.library import CardLibrary
from tests.api.test_posed_only_scene import POSED, SEG, _box_mesh


def _scene() -> alloy.Scene:
    v, f = _box_mesh(SEG)
    scene = alloy.Scene(dt=0.01)
    scene.add_box("worktop", size=(0.4, 0.4, 0.02), pos=(0, 0, 0.01), static=True)
    scene.add_mesh("thread", vertices=v, faces=f, pos=(0, 0, 0.05), kinematic=True, card_extra=POSED)
    scene.add_sphere("ball", 0.01, pos=(0.1, 0.0, 0.03), mass=0.01)
    return scene


def test_a_kinematic_part_under_mujocos_limit_builds_at_its_own_density(capfd):
    capfd.readouterr()
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        sim = alloy.build(_scene(), device="cpu")
    try:
        corrected = [w for w in seen if "Inertia validation corrected" in str(w.message)]
        ours = [ln for ln in capfd.readouterr().err.splitlines() if "under what MuJoCo accepts" in ln]
        assert not corrected and not ours, (corrected, ours)
        density = CardLibrary().material(RIGID_MATERIAL).density.value
        mass = float(sim.engine.model.body_mass.numpy()[sim.index.body("thread")])
        assert np.isclose(mass, density * float(np.prod(SEG)), rtol=1e-4), (mass, density)
        sim.set_pose("thread", pos=(0.01, 0.0, 0.05))
        sim.step(5)
        assert np.allclose(sim.state(world=0)["thread.pos"], (0.01, 0.0, 0.05), atol=1e-6)
        assert sim.check()["ok"], sim.check()["problems"]
    finally:
        sim.close()
