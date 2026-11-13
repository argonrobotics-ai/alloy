# SPDX-License-Identifier: Apache-2.0
"""A pour into an open box keeps all its particles (COUPLE-9, physics ruling 2026-11-11); CPU, or ALLOY_DEVICE.

Beside cloth with no free rigid body, the coupled clock ran the liquid at its bare stability step (clock safety 1).
A 6 ml and a 20 ml pour from 12 cm then splashed 20 % and 24 % over the rims of static open boxes, with health OK
(docs/known_gaps/liquid-pour-over-the-rim.md). ``DFSPHConfig.speed_margin`` 0.25 (a quarter of the CFL step in the
coupled estimate while the liquid moves fast) keeps every particle in.
Law: every poured particle ends inside its box (bound: exact counts) beside cloth, and beside cloth with a free ball.
Alone, the single-solver step still lets 3 of the 6 ml pour's 94 over the rim (0.1.1): a strict xfail until fixed.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import alloy

DEVICE = os.environ.get("ALLOY_DEVICE", "cpu")
BOXES = (("a", -0.12, 0.04, 6e-6), ("b", 0.12, 0.05, 20e-6))      # name, x, inner half width, asked volume


def _scene(cloth: bool, ball: bool) -> alloy.Scene:
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    for name, x, half, volume in BOXES:
        scene.add_open_box(f"{name}_box", inner=(2 * half, 2 * half, 0.06 if name == "a" else 0.08), wall=0.005,
                           pos=(x, 0, 0))
        scene.add_pour(name, volume, pos=(x, 0, 0.12), rate=1e-4)              # a stream landing from 12 cm
    if cloth:
        scene.add_cloth("sheet", size=(0.1, 0.1), pos=(0, 0.3, 0.01), resolution=6, mass=0.01)
    if ball:
        scene.add_sphere("ball", 0.02, pos=(0, -0.3, 0.02), mass=0.05)
    return scene


def _poured_and_outside(sim: alloy.Sim) -> dict[str, tuple[int, int]]:
    st = sim.state([b[0] for b in BOXES], world=0)
    out = {}
    for name, x, half, _ in BOXES:
        p, act = st[f"{name}.points"], st[f"{name}.active"]
        away = (np.abs(p[:, 0] - x) > half + 1e-3) | (np.abs(p[:, 1]) > half + 1e-3)
        out[name] = (int(act.sum()), int((act & away).sum()))
    return out


def _all_in(sim: alloy.Sim) -> dict[str, tuple[int, int]]:
    """Every particle of each pour poured, none outside its box."""
    return {name: (len(sim.index.particle_range(name)), 0) for name, _, _, _ in BOXES}


@pytest.mark.slow
@pytest.mark.timeout(2400)
@pytest.mark.parametrize("ball", [False, True], ids=["cloth", "cloth_and_ball"])
def test_a_pour_beside_cloth_keeps_every_particle_in_its_box(ball):
    sim = alloy.build(_scene(cloth=True, ball=ball), device=DEVICE)
    sim.run(0.6)
    assert _poured_and_outside(sim) == _all_in(sim)                  # was 19 of 94 and 74 of 312 out
    assert sim.check()["ok"]
    sim.close()


@pytest.mark.slow
@pytest.mark.timeout(1800)
@pytest.mark.xfail(strict=True, reason="alone, 3 of 94 go over the rim (liquid-pour-over-the-rim.md; 0.1.1)")
def test_a_pour_alone_keeps_every_particle_in_its_box():
    sim = alloy.build(_scene(cloth=False, ball=False), device=DEVICE)
    sim.run(0.6)
    assert _poured_and_outside(sim) == _all_in(sim)
    sim.close()
