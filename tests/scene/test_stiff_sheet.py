# SPDX-License-Identifier: Apache-2.0
"""Stiff CLOTH_VBD sheets start from Newton's inertial guess (alloy.scene.stiff_sheet, launch item 2): under the adaptive
start an unconverged stiff sheet gains energy (cardboard on two round supports still moved at 0.45 m/s after 3 s; PET
climbed 28 mm above them). Fabrics keep the adaptive start, which their drapes need."""

from __future__ import annotations

import math

import numpy as np
import pytest

import alloy
from alloy.scene import stiff_sheet as SS


def _sheet(scene: alloy.Scene, name: str, material: str, h: float, pos=(0.0, 0.0, 0.05), res=(10, 5)) -> None:
    scene.add_cloth(name, size=(0.2, 0.1), pos=pos, resolution=res, thickness=h, material=material,
                    self_contact=False)


def _start(sim) -> str:
    return [s for s in sim.engine.solvers if hasattr(s, "_accel")][0]._accel.cfg.initial_guess


@pytest.mark.parametrize("material, h, start", [("cotton_woven", 0.002, "adaptive"), ("vinyl_soft", 0.001, "adaptive"),
                                                ("pet", 0.0005, "inertial"), ("cardboard_corrugated", 0.003, "inertial")])
def test_the_start_follows_the_sheets_bending_length(material, h, start):
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    _sheet(scene, "c", material, h)
    sim = alloy.build(scene, device="cpu")
    reasons = sim.routes()["c"]["reasons"]
    assert sim.routes()["c"]["route"] == "cloth_vbd" and _start(sim) == start
    assert any(r.startswith("stiff sheet") for r in reasons) is (start == "inertial")
    sim.close()


def test_a_stiff_sheet_sets_the_shared_entrys_start():
    """The VBD entry is shared: a fabric beside a stiff sheet starts from the inertial guess too (module docstring)."""
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    _sheet(scene, "towel", "cotton_woven", 0.002, pos=(-0.3, 0.0, 0.05))
    _sheet(scene, "card", "cardboard_corrugated", 0.003, pos=(0.3, 0.0, 0.05))
    sim = alloy.build(scene, device="cpu")
    assert _start(sim) == "inertial"
    sim.close()


def test_entry_start_only_replaces_the_adaptive_start():
    kw = {"alloy_initial_guess": "adaptive", "alloy_chebyshev_rho": 0.9, "iterations": 40}
    assert SS.entry_start(kw, {SS.STIFF_EFFECT}) == {**kw, "alloy_initial_guess": "inertial"}
    assert SS.entry_start(kw, set()) == kw
    fast = {"alloy_initial_guess": "inertial", "alloy_chebyshev_rho": None, "iterations": 10}
    assert SS.entry_start(fast, {SS.STIFF_EFFECT}) == fast


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_a_cardboard_sheet_comes_to_rest_on_two_round_supports():
    """The rig that showed the pump: 3 mm corrugated cardboard dropped 5 mm onto two 1 cm round supports 10 cm apart.
    Under the adaptive start it still moved at 0.45 m/s after 3 s and walked; from the inertial guess it rests."""
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    for k, x in enumerate((-0.05, 0.05)):
        scene.add_cylinder(f"sup{k}", 0.01, 0.2, pos=(x, 0.0, 0.04), rpy=(math.pi / 2, 0.0, 0.0), static=True)
    _sheet(scene, "card", "cardboard_corrugated", 0.003, pos=(0.0, 0.0, 0.0565), res=(20, 10))
    sim = alloy.build(scene, device="cpu")
    tr = sim.run(2.0, record=["card"], every=5).world(0)
    q, t = tr["card.points"], tr["t"]
    v = np.linalg.norm(np.diff(q, axis=0), axis=2).max(axis=1) / np.diff(t)
    assert _start(sim) == "inertial" and sim.check()["ok"]
    assert v[t[1:] >= 1.5].max() < 0.01                          # at rest (adaptive: 0.45 m/s at 3 s)
    # on its supports, not walked off: centred, above the ground (it sags ~12 mm between them: 40 iterations are far
    # from converged on a stiff sheet, docs/known_gaps/cloth-stiff-sheet-sag.md)
    assert abs(q[-1, :, 0].mean()) < 1e-3 and q[-1, :, 2].min() > 0.02 and q[-1, :, 2].mean() > 0.04
    sim.close()
