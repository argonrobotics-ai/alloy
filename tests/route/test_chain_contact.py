# SPDX-License-Identifier: Apache-2.0
"""ROD-5 CPU smoke: CHAIN self-contact (adjacent links excluded) and a cable on VBD cloth through the coupler.

- coil: a 30-link helical cable dropped on a table; at every sampled step the axis distance of every non-adjacent
  link pair stays ≥ r (a crossing would bring two axes closer than one radius); 0 crossings, finite state.
- cable on cloth: CHAIN (MuJoCo) + cloth (VBD) on a table through the stock proxy; no INVALID world (FakeHealth's
  non-finite and 20 m/s checks on bodies, plus particle checks here).
CPU runs 1 s (coil) and 0.4 s (cable on cloth); the 2 s GPU run is the `gpu` variant."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy.cards.types import CurveSource
from alloy.core.registry import isolated_registry
from alloy.precision.types import Severity
from alloy.route.router import Router
from alloy.route.types import RouterConfig
from alloy.scene.assembler import SceneAssembler
from alloy.sim import Simulation
from alloy.testing import route as T
from alloy.testing import scene as S
from alloy.testing.fakes import FakeHealth

PASS = {"sag_err": -6e-4, "drop_err": -0.03, "period_err": -0.003, "finite": 1.0, "passed": 1.0}
RADIUS = 0.003


@pytest.fixture(autouse=True)
def _registry():
    with isolated_registry(empty=True):
        yield


def helix(turns: float = 2.6, r: float = 0.03, pitch: float = 0.02, n: int = 120) -> tuple:
    t = np.linspace(0.0, 2 * math.pi * turns, n)
    return tuple((float(r * math.cos(a)), float(r * math.sin(a)), float(pitch * a / (2 * math.pi))) for a in t)


def seg_dist(p0, p1, q0, q1) -> np.ndarray:
    """Vectorised closest distance between segments p0-p1 and q0-q1 ([..., 3])."""
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a, e = np.sum(d1 * d1, -1), np.sum(d2 * d2, -1)
    b, c, f = np.sum(d1 * d2, -1), np.sum(d1 * r, -1), np.sum(d2 * r, -1)
    den = np.maximum(a * e - b * b, 1e-18)
    s = np.clip((b * f - c * e) / den, 0.0, 1.0)
    t = np.clip((b * s + f) / np.maximum(e, 1e-18), 0.0, 1.0)
    s = np.clip((b * t - c) / np.maximum(a, 1e-18), 0.0, 1.0)
    return np.linalg.norm(p0 + d1 * s[..., None] - q0 - d2 * t[..., None], axis=-1)


def link_segments(body_q: np.ndarray, lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Link axis endpoints from COM poses (x axis along each link)."""
    pos, q = body_q[:, :3], body_q[:, 3:]
    x = np.stack([1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2), 2 * (q[:, 0] * q[:, 1] + q[:, 3] * q[:, 2]),
                  2 * (q[:, 0] * q[:, 2] - q[:, 3] * q[:, 1])], axis=1)
    h = 0.5 * lengths[:, None] * x
    return pos - h, pos + h


class CrossingCounter:
    """Pass-through detector for non-adjacent link pairs: the sign of (d_i × d_j)·(c_i − c_j) (c = closest points)
    flips between consecutive samples while both closest points are interior and the axes are within 2r."""

    def __init__(self, lengths: np.ndarray, radius: float) -> None:
        self.lengths, self.radius = lengths, radius
        self.i, self.j = np.triu_indices(len(lengths), k=2)
        self.prev_sign: np.ndarray | None = None
        self.crossings, self.min_dist = 0, math.inf

    def update(self, body_q: np.ndarray) -> None:
        a, b = link_segments(body_q, self.lengths)
        p0, p1, q0, q1 = a[self.i], b[self.i], a[self.j], b[self.j]
        d = seg_dist(p0, p1, q0, q1)
        self.min_dist = min(self.min_dist, float(d.min()))
        sign = np.sign(np.einsum("ij,ij->i", np.cross(p1 - p0, q1 - q0), 0.5 * (p0 + p1) - 0.5 * (q0 + q1)))
        if self.prev_sign is not None:
            flip = (sign * self.prev_sign < 0) & (d < 2.0 * self.radius)
            self.crossings += int(np.sum(flip))
        self.prev_sign = sign


def test_rod5_coil_no_crossings():
    pvc = T.material("pvc", **T.CABLE_PVC)
    cab = T.obj("cable", CurveSource(points=helix(), radius=RADIUS), pvc)
    tab = S.table()
    card, lib = S.scene([tab, cab], [S.place("table"), S.place("cable", z=0.05)], worlds=1)
    pinned = RouterConfig(chain_links_min=30)       # ROD-5's coil at its proven 30 links (SCR-DC3-7 pin)
    built = SceneAssembler(lib, Router(pinned, rod_self_test=lambda rod, h: dict(PASS))).assemble(card, "cpu")
    h = built.handles["cable"]
    assert h.decision.route == "chain" and len(h.template.body) == 30
    sim = Simulation(built, health=FakeHealth(built.model))
    counter = CrossingCounter(_lengths(built, h), RADIUS)
    links = slice(h.template.body.start, h.template.body.stop)
    for _ in range(50 * sim.clock.steps_per_period):     # 1 s, sampled every exchange
        sim._exchange(sim.clock.dt)
        counter.update(sim.state.body_q.numpy()[links])
    rep = sim.report()["health"]
    n_cross = counter.crossings
    S.record("ROD-5.coil_crossings", n_cross, "pass" if n_cross == 0 else "fail", min_axis_dist_m=counter.min_dist,
             radius=RADIUS, sim_s=1.0, device="cpu", native=sim.native)
    assert n_cross == 0 and np.isfinite(sim.state.body_q.numpy()).all()
    assert not rep.invalid_worlds


def _lengths(built, h) -> np.ndarray:
    n = h.decision.numbers
    links = int(n["links"])
    l = n["rod_length"] / (links - 1)
    return np.asarray([0.5 * l] + [l] * (links - 2) + [0.5 * l])


@pytest.mark.timeout(300)  # the towel's coupled VBD entry runs the 40-iteration default: 78 s alone on host-1 (25 s at 10)
def test_rod5_cable_on_cloth_through_coupler():
    pvc = T.material("pvc", **T.CABLE_PVC)
    pts = tuple((0.01 * i - 0.08, 0.0, 0.0) for i in range(17))
    cab = T.obj("cable", CurveSource(points=pts, radius=RADIUS), pvc)
    tab, towel = S.table(), S.cloth()
    card, lib = S.scene([tab, towel, cab], [S.place("table"), S.place("towel", z=0.022),
                                            S.place("cable", z=0.03)], worlds=2)
    built = SceneAssembler(lib, Router(rod_self_test=lambda rod, h: dict(PASS))).assemble(card, "cpu")
    assert {e.name for e in built.entries} == {"mujoco", "vbd"}
    # a 17 g cable on a 15 g cloth is strong under D5 (m_a/m_s > 0.5): ADMM, reported strong-unrouted; until WP4's
    # SchedulerADMM lands Simulation runs it on the stock lagged proxy
    assert built.coupler.kind == "admm" and built.coupler.strong_unrouted == (("mujoco", "vbd"),)
    sim = Simulation(built, health=FakeHealth(built.model))
    for _ in range(20):                                  # 0.4 s
        sim.step_period()
    pq, pv = sim.state.particle_q.numpy(), sim.state.particle_qd.numpy()
    rep = sim.report()["health"]
    ok = np.isfinite(pq).all() and np.linalg.norm(pv, axis=1).max() < 20.0 and not rep.invalid_worlds
    S.record("ROD-5.cable_on_cloth", float(np.linalg.norm(pv, axis=1).max()), "pass" if ok else "fail",
             sim_s=0.4, device="cpu", invalid_worlds=list(rep.invalid_worlds), coupler=built.coupler.kind,
             stepper=type(sim.stepper).__name__)
    assert ok
    assert int(rep.severity.max()) < int(Severity.INVALID)
