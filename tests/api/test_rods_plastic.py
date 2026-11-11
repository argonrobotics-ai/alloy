# SPDX-License-Identifier: Apache-2.0
"""SCR-WP9-ROD Phase 2: a copper wire that keeps a bend, through the public API, on CPU.

`Scene.add_rod(material="copper_c110")`: the card's `plastic.model: j2` (σ_y0 69 MPa, H 603 MPa) routes the rod to
the elastic rod, whose joints yield (alloy.solvers.rod.plastic). A clamped wire (0.2 m, R 1 mm, 20 segments, gravity
off) takes a held end moment of 1.5 M_y, ramped over 4 s (100 bending periods), and is let go. The moment is uniform along the span, so
the curvature is the end segment's rotation over the length. It matches the closed-form M(κ) of a round, linearly
hardening section loaded and released (alloy.solvers.plastic.closed_forms), and `"<rod>.plastic_strain"` reads the
yield.
- Loaded within 1 %, measured +0.66 %. The excess is the ramp's dynamic overshoot at the card's 2 % damping, and the
  overshoot yields: +1.0 % with a 2 s ramp. With critical damping the solver test reads +0.2 %
  (tests/solvers/rod/test_rod_plastic.py).
- Residual within 1 % of κ_max, measured 0.66 % (+2.6 % of itself: κ_res = κ_max - M / EI is a quarter of κ_max,
  so it carries the loaded error about 4x over).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import alloy
from alloy.solvers.plastic import closed_forms as C

L, R, N = 0.2, 1.0e-3, 20
E, SY, H = 115e9, 69e6, 603e6                  # copper_c110
EI = E * math.pi * R**4 / 4.0
M_Y = SY * math.pi * R**3 / 4.0


def _kappa_of(moment: float) -> float:
    lo, hi = 0.0, 50.0 * M_Y / EI
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if C.round_moment(mid, R, E, SY, H) < moment else (lo, mid)
    return 0.5 * (lo + hi)


def _about_z(q: np.ndarray, q0: np.ndarray) -> float:
    w = q[3] * q0[3] + q[0] * q0[0] + q[1] * q0[1] + q[2] * q0[2]
    z = -q[3] * q0[2] + q[2] * q0[3] - q[0] * q0[1] + q[1] * q0[0]
    return 2.0 * math.atan2(z, w)


def test_a_yielding_wire_routes_to_the_elastic_rod_and_says_why():
    sc = alloy.Scene(dt=0.01)
    sc.add_rod("wire", length=L, radius=R, pos=(0.0, 0.0, 0.3), material="copper_c110", segments=N)
    r = alloy.estimate(sc, 1, 0.1, device="cpu")["routes"]["wire"]
    assert r["route"] == "rod_vbd" and any("yields" in why for why in r["reasons"])


@pytest.mark.law
def test_a_copper_wire_bent_past_yield_keeps_its_set():
    sc = alloy.Scene(dt=0.01)
    sc.add_rod("wire", length=L, radius=R, pos=(0.0, 0.0, 0.3), material="copper_c110", segments=N, clamp="start")
    sim = alloy.build(sc, device="cpu")
    sim.engine.model.gravity.assign(np.zeros((1, 3), np.float32))
    end = f"wire/wire_edge_body_{N - 1}"
    q0 = sim.state([end], world=0)[f"{end}.quat_xyzw"].copy()

    def curvature() -> float:
        return _about_z(sim.state([end], world=0)[f"{end}.quat_xyzw"], q0) / L

    m = 1.5 * M_Y
    for k in range(400):
        sim.apply_wrench("wire", torque=(0.0, 0.0, m * (k + 1) / 400), at="end")
        sim.step(1)
    sim.step(60)
    loaded = curvature()
    for k in range(100):                                # unload as slowly
        sim.apply_wrench("wire", torque=(0.0, 0.0, m * (1.0 - (k + 1) / 100)), at="end")
        sim.step(1)
    sim.step(60)
    released = curvature()
    k_max = _kappa_of(m)
    assert loaded == pytest.approx(k_max, rel=0.01, abs=0)
    assert abs(released - C.springback_curvature(k_max, m, EI)) < 0.01 * k_max
    ps = sim.state(["wire"], world=0)["wire.plastic_strain"]
    assert ps.shape == (N - 1,) and ps.min() > 0.0
    assert sim.check()["ok"]
    sim.close()
