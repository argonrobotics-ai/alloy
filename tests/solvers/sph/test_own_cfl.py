# SPDX-License-Identifier: Apache-2.0
"""Coupled SPH takes its substeps from its own CFL step (``ClockPolicy.own_cfl``, default on; S4 2026-11-09), CPU.

The coupled clock picks an entry's rung n so that dt_x / n <= ClockPolicy.safety (0.5) x the entry's stable step. SPH's
stable step is already its CFL step, cfl (0.2) x spacing / v_max, so the margin counted twice: coupled SPH ran at least
2 substeps per exchange even at rest, and 4 in the tilt below, where SPH alone runs 1 at the same speed. With
``own_cfl`` an estimator that declares ``alloy_clock_safety`` (SolverDFSPH: 1) is taken at that safety.
- The rule: SolverDFSPH's safety is 1 with own_cfl and the policy's without; an estimator with no declaration keeps
  the policy's. Beside MPM (an SPH entry paired with an MPM entry) SPH keeps the policy's safety: a slab floating on
  SPH water sat 2 points deeper at the CFL step itself (tests/integration/test_sph_mpm_laws.py L1, WP12).
- test_kinematic_tilt's fast law beside a free ball (a 30 deg tilt about the lip in 0.3 s): SPH runs 2 substeps per
  exchange by default, 4 with own_cfl off (main 7e09e7e8d, CPU: wall 51.8 -> 38.7 s), every particle stays in the jug,
  on or above its floor, and the water's height matches the static-ball reference within 1 mm (as the law).
"""

from __future__ import annotations

import dataclasses
import math
from collections import Counter

import numpy as np
import pytest

import alloy
from alloy.coupling.clocks import entry_safeties, entry_safety
from alloy.coupling.types import AlloyEntry, ClockPolicy, PairPolicy
from alloy.solvers.sph.solver import SolverDFSPH

IN, WALL, X, Z = 0.05, 0.004, 0.0, 0.10
LIP = (X + IN / 2 + WALL, 0.0, Z + WALL + IN)


def _own_cfl(sim: alloy.Sim, on: bool) -> None:
    """Set ClockPolicy.own_cfl on a built coupled sim (before its first step)."""
    st = sim.engine.stepper
    clock = dataclasses.replace(st._alloy_policy.clock, own_cfl=on)
    st._alloy_policy = dataclasses.replace(st._alloy_policy, clock=clock)
    st._clock.policy = clock
    st._clock.safety_for = entry_safeties(st._estimators, st._alloy_entries, st._alloy_pairs, clock)


def _tilt(free: bool, own: bool | None = None) -> tuple[np.ndarray, alloy.Sim, Counter]:
    """Water points in the jug floor's frame after the tilt, and the SPH substeps of every period."""
    sc = alloy.Scene(dt=0.01)
    sc.add_ground()
    sc.add_open_box("jug", inner=(IN, IN, IN), wall=WALL, pos=(X, 0, Z), kinematic=True)
    sc.add_liquid("water", size=(0.044, 0.044, 0.016), pos=(X, 0, Z + WALL + 0.009))
    sc.add_sphere("ball", 0.02, pos=(0.2, 0.0, 0.02), **({"mass": 0.1} if free else {"static": True}))
    sim = alloy.build(sc, device="cpu")
    if own is not None:
        _own_cfl(sim, own)
    subs: Counter = Counter()

    def periods(n: int) -> None:
        for _ in range(n):
            sim.step(1)
            now = getattr(sim.engine.stepper, "_substeps_now", {})
            subs[next((int(v) for k, v in now.items() if "sph" in k), 1)] += 1

    periods(5)
    sim.drive("jug", angular_velocity=(0, math.radians(100), 0), pivot=LIP, duration=0.3)
    periods(40)
    s = sim.state(["water", "jug_floor"], world=0)
    x, y, z, w = (float(c) for c in s["jug_floor.quat_xyzw"])
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return (s["water.points"] - s["jug_floor.pos"]) @ r, sim, subs


def test_sph_declares_its_own_safety_and_only_own_cfl_uses_it():
    pol = ClockPolicy()
    assert pol.own_cfl is True and SolverDFSPH.alloy_clock_safety == 1.0
    assert entry_safety(SolverDFSPH, pol) == 1.0
    assert entry_safety(object(), pol) == pol.safety
    off = dataclasses.replace(pol, own_cfl=False)
    assert entry_safety(SolverDFSPH, off) == off.safety


def test_beside_mpm_sph_keeps_the_policys_safety():
    pol = ClockPolicy()
    assert entry_safety(SolverDFSPH, pol, {"mpm"}) == pol.safety
    assert entry_safety(SolverDFSPH, pol, {"mujoco", "vbd"}) == 1.0
    entries = {n: AlloyEntry(entry=None, capability=c) for n, c in
               (("sph", "sph"), ("mpm", "mpm"), ("mujoco", "mujoco"), ("sph2", "sph"))}
    pairs = [PairPolicy("mujoco", "sph"), PairPolicy("sph", "mpm"), PairPolicy("mujoco", "sph2")]
    est = {"sph": SolverDFSPH, "mpm": None, "mujoco": object(), "sph2": SolverDFSPH}
    assert entry_safeties(est, entries, pairs, pol) == {"sph": pol.safety, "mujoco": pol.safety, "sph2": 1.0}


def _settle_rung(own: bool | None) -> int:
    """SPH's substeps per exchange in the jug scene's third period at rest beside the free ball."""
    sc = alloy.Scene(dt=0.01)
    sc.add_ground()
    sc.add_open_box("jug", inner=(IN, IN, IN), wall=WALL, pos=(X, 0, Z), kinematic=True)
    sc.add_liquid("water", size=(0.044, 0.044, 0.016), pos=(X, 0, Z + WALL + 0.009))
    sc.add_sphere("ball", 0.02, pos=(0.2, 0.0, 0.02), mass=0.1)
    sim = alloy.build(sc, device="cpu")
    if own is not None:
        _own_cfl(sim, own)
    sim.step(3)
    now = sim.engine.stepper._substeps_now
    rung = next((int(v) for k, v in now.items() if "sph" in k), None)
    assert rung is not None, f"no SPH entry in the coupled clock's substeps {now}"
    return rung


def test_the_default_clock_runs_coupled_sph_at_half_the_old_substeps():
    assert (_settle_rung(None), _settle_rung(False)) == (2, 4)


@pytest.mark.law
@pytest.mark.timeout(900)
def test_a_coupled_tilt_at_sphs_own_cfl_runs_half_the_substeps_and_keeps_its_water():
    loc, sim, subs = _tilt(free=True)
    print(f"own_cfl: SPH substeps per period {dict(sorted(subs.items()))}")
    assert max(subs) == 2, subs                         # own_cfl off runs 4 here
    inside = (np.abs(loc[:, 0]) < IN / 2 + 1e-3) & (np.abs(loc[:, 1]) < IN / 2 + 1e-3)
    assert inside.all(), int((~inside).sum())
    assert float(loc[:, 2].min()) > WALL / 2 - 1.5e-3, float(loc[:, 2].min())
    assert sim.check()["ok"], sim.check()["problems"]
    ref, _, _ = _tilt(free=False)
    assert float(np.median(loc[:, 2])) == pytest.approx(float(np.median(ref[:, 2])), abs=1.0e-3, rel=0)
