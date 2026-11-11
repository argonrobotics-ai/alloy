# SPDX-License-Identifier: Apache-2.0
"""PHYSA-4 laws (CPU): the shipped peg-in-hole precision fit inserts.

A 12 mm peg (``peg_d12``) dropped from 6 mm above a 12.2 mm bore in a 9 mm block (``block_bore_d12_2``), started
within the fit's 0.1 mm radial clearance, ends on the bore's bottom: peg centre at 25.0 mm (its half length), upright.

Before (audit PHYSA-4, rc8; the same on rc2-1109):
- under the names ``Scene.add_object``'s own example uses ("peg", beside a block called "blk") no precision pair was
  made, with no word: the block collided as its convex hull and the peg rested on top of it, centre at 33.999 mm,
  health OK. tests/scene/test_precision_pairs_by_card.py has the naming rules;
- ``demo_peg_d12`` spread the peg's start by +-0.5 mm against 0.1 mm of clearance on a bore without a chamfer, so a
  world in 16 inserted, the others wedged on the rim (31-32 mm, tilted 7-26 deg) or fell off the block. Named as its
  card, the peg inserted from 0, 0.02, 0.05 and 0.09 mm off the axis (24.999 mm) and did not from 0.15, 0.3 and 0.5 mm.

Law (pre-registered, coordinator 2026-11-10 01:16Z): every world ends at 25.0 mm within 0.1 mm, tilted under 1 deg,
health OK and ``sim.check()`` ok. Measured with the fix (CPU, MuJoCo Warp route, 1.5 s): 24.999 mm and 0.0 deg in both
worlds of the renamed scene and in all 16 worlds of the demo.
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy

BOTTOM_MM, DEPTH_TOL_MM, TILT_MAX_DEG = 25.0, 0.1, 1.0
SECONDS = 1.0                      # a peg 0.09 mm off the axis is down after 0.3 s


def _inserted(sim: alloy.Sim, peg: str) -> None:
    """Every world's peg is on the bore's bottom, upright, in a healthy world."""
    assert any(r.startswith("precision pair with") for r in sim.routes()[peg]["reasons"]), sim.routes()[peg]
    sim.run(SECONDS)
    state = sim.state([peg])
    z = 1e3 * state[f"{peg}.pos"][:, 2]
    q = state[f"{peg}.quat_xyzw"]
    tilt = np.degrees(2.0 * np.arcsin(np.clip(np.hypot(q[:, 0], q[:, 1]), 0.0, 1.0)))       # of the peg's axis (z)
    print(f"{peg}: end height {z.min():.3f}-{z.max():.3f} mm, tilt up to {tilt.max():.3f} deg, {len(z)} worlds")
    assert np.all(np.abs(z - BOTTOM_MM) <= DEPTH_TOL_MM), z
    assert np.all(tilt < TILT_MAX_DEG), tilt
    assert [w["severity"] for w in sim.health()["worlds"]] == ["OK"] * len(z), sim.health()
    assert sim.check()["ok"], sim.check()["problems"]


@pytest.mark.law
@pytest.mark.compile_heavy
@pytest.mark.timeout(1800)
def test_a_peg_and_a_block_under_other_names_insert():
    """The auditor's scene: the objects are not named after their cards; the peg starts on the axis and 0.05 mm off."""
    scene = alloy.Scene(worlds=2, dt=0.02)
    scene.add_object("table", "table_top_plywood", pos=(0.0, 0.0, -0.009), static=True)
    scene.add_object("blk", "block_bore_d12_2", pos=(0.0, 0.0, 0.0045))
    scene.add_object("peg", "peg_d12", pos=np.array([[0.0, 0.0, 0.04], [0.05e-3, 0.0, 0.04]]))
    sim = alloy.build(scene, profile="precision", device="cpu")
    assert "precision pair with blk: shaft / bore, clearance 0.2 mm" in sim.routes()["peg"]["reasons"]
    _inserted(sim, "peg")
    sim.close()


@pytest.mark.law
@pytest.mark.compile_heavy
@pytest.mark.timeout(2400)
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)])
def test_every_world_of_the_peg_demo_inserts(device):
    """``demo_peg_d12`` as shipped (AGENTS.md section 8 points users at it): its 16 worlds, their own start spread.
    On an L4 before 2026-11-10 one to three of the 16 worlds went INVALID (PAIR_PEN_FAIL at 0.062 s) while their pegs
    slid down the bore: the gap monitor read the straight segment between two poses of a sample touching the wall,
    0.1-1.2 um inside it, as a tunnel. With the march's contact-depth floor (alloy.contact.monitor.cross_floor) all 16
    insert there (2 runs: 24.999 mm, tilt 0.001 deg, health OK)."""
    sim = alloy.load("demo_peg_d12", worlds=16, profile="precision", device=device)
    start = 1e3 * sim.state(["peg_d12"])["peg_d12.pos"][:, :2]
    assert 0.0 < np.abs(start).max() <= 0.05 + 1e-6, start       # spread, and inside the 0.1 mm radial clearance
    _inserted(sim, "peg_d12")
    sim.close()
