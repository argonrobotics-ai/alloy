# SPDX-License-Identifier: Apache-2.0
"""A vbd entry of rods only runs ROD_ITERATIONS (4) VBD iterations; one with cloth keeps the cloth's own (S4,
2026-11-09), CPU.

The rod solve converges each chain directly, so the iterations only settle the contacts. Since F3's sticky contacts,
the R4 grip law (test_rods_grip.py) passes at 4: ramp end 1.0002, run-on +0.0002, torque 1.0195 against a 3 % bound.
Beside a robot every period runs 64 exchanges x 2 passes of this solve, so it is about 2.2x faster.
- A clamped cable alone in its vbd entry: 4 iterations.
- The same cable beside a cloth sheet (one vbd entry for both): the count the sheet alone gets, as before.
"""

from __future__ import annotations

import alloy
from alloy.scene.entries import ROD_ITERATIONS


def _vbd_iterations(sim: alloy.Sim) -> int:
    solvers = [s for s in sim.engine.solvers if type(s).__name__ != "SolverMuJoCo" and hasattr(s, "iterations")]
    assert len(solvers) == 1, [type(s).__name__ for s in sim.engine.solvers]
    return int(solvers[0].iterations)


def _scene(cloth: bool, rod: bool = True) -> alloy.Scene:
    sc = alloy.Scene(dt=0.01)
    sc.add_ground()
    if rod:
        sc.add_rod("cable", length=0.3, radius=0.003, pos=(0.0, 0.0, 0.3), segments=10, clamp="start")
    if cloth:
        sc.add_cloth("sheet", size=(0.1, 0.1), pos=(0.5, 0.0, 0.2), resolution=4, mass=0.01, self_contact=False)
    return sc


def test_a_rod_only_vbd_entry_runs_four_iterations_and_cloth_keeps_its_own():
    assert ROD_ITERATIONS == 4
    rods = alloy.build(_scene(cloth=False), device="cpu")
    assert rods.routes()["cable"]["route"] == "rod_vbd"
    assert _vbd_iterations(rods) == ROD_ITERATIONS
    mixed = alloy.build(_scene(cloth=True), device="cpu")
    assert mixed.routes()["sheet"]["route"] == "cloth_vbd"
    sheet = _vbd_iterations(alloy.build(_scene(cloth=True, rod=False), device="cpu"))
    assert _vbd_iterations(mixed) == sheet and sheet != ROD_ITERATIONS, sheet
