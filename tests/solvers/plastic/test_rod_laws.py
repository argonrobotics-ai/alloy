# SPDX-License-Identifier: Apache-2.0
"""The plasticity gap page's two rod laws ("springs and wires that yield", docs/known_gaps/plasticity-metal.md) on
WP9's J2 rod joints (alloy.solvers.rod.plastic on SolverVBDRod), CPU, gravity off, elastic-perfectly-plastic copper
(E 117 GPa, ν 0.3, σ_y0 70 MPa, H 0; the closed forms in alloy.solvers.plastic.closed_forms):
- A: a clamped cantilever wire under a tip load past yield keeps a permanent tip set. Its closed form, on the rod's
  own joints: each joint at its lever arm carries M_i = F (x_F - x_i), bends h κ(M_i) and keeps h κ(M_i) - M_i h / EI;
  the set is the sum of the kept rotations times their lever arms (small deflection).
- B: a close-coiled spring compressed past the wire's torsional yield loses free length. Wire torsion theory: the
  load F on the spring's axis twists the wire by T = F Rc, so the free length lost is Rc L_w (φ'(T) - T / GJ) for a
  wire of length L_w.
Loads ramp over RAMP, are held for HOLD and released for HOLD; the residual is judged against the loaded value, as in
tests/solvers/rod/test_rod_plastic.py (most of a loaded deflection is elastic)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy._vendor import newton
from alloy.scene.analytic_rods import rod_card
from alloy.scene.analytic_rods_vbd import vbd_card
from alloy.scene.emitters.rod_vbd import build_rod_vbd
from alloy.solvers.plastic import closed_forms as C
from alloy.solvers.rod import SolverVBDRod
from alloy.solvers.rod.plastic import mark_joints

E, SY, RHO, R = 117e9, 70e6, 8960.0, 1.0e-3
G = E / 2.6                                             # rod_card: GJ = EI / 1.3
EI, GJ = E * math.pi * R**4 / 4.0, G * math.pi * R**4 / 2.0
M_Y, T_Y = SY * math.pi * R**3 / 4.0, math.pi * R**3 * (SY / math.sqrt(3.0)) / 2.0
DT, RAMP, HOLD = 1.25e-3, 0.2, 0.4


def _invert(f, target: float, hi: float) -> float:
    lo = 0.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
    return 0.5 * (lo + hi)


def _kappa(m: float) -> float:
    return _invert(lambda k: C.round_moment(k, R, E, SY), m, 50.0 * M_Y / EI)


def _phi(t: float) -> float:
    return _invert(lambda p: C.round_torque(p, R, G, SY), t, 50.0 * t / GJ)


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """xyzw quaternion q applied to v."""
    u, w = q[:3], q[3]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def _wire(points, links: int):
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    length = float(sum(np.linalg.norm(np.subtract(points[i + 1], points[i])) for i in range(len(points) - 1)))
    rod = vbd_card(rod_card(length, EI, links, radius=R, density=RHO, damping_ratio=1.0))
    h = build_rod_vbd(b, rod, points, clamp="start", label="wire")
    mark_joints(b, h.joints, SY, 0.0, R, length / (links - 1))      # the emitter's joint spacing (half-length ends)
    b.color()
    model = b.finalize(device="cpu")
    return model, SolverVBDRod(model, iterations=4), h, length


def _com(model, q: np.ndarray, body: int) -> np.ndarray:
    return q[body, :3] + _rotate(q[body, 3:], model.body_com.numpy()[body].astype(np.float64))


def _load(model, solver, end: int, wrench_of):
    """Ramp, hold and release ``wrench_of(t_scale, end COM) -> (6,)`` on body ``end``; the end COM at the start,
    loaded and released."""
    s0, s1, c = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    start = _com(model, s0.body_q.numpy().astype(np.float64), end)

    def run(seconds: float, scale) -> np.ndarray:
        nonlocal s0, s1
        for k in range(int(round(seconds / DT))):
            f = np.zeros((model.body_count, 6), np.float32)
            f[end] = wrench_of(scale((k + 1) * DT), _com(model, s0.body_q.numpy().astype(np.float64), end))
            s0.body_f.assign(f)
            solver.step(s0, s1, c, None, DT)
            s0, s1 = s1, s0
        return _com(model, s0.body_q.numpy().astype(np.float64), end)

    run(RAMP, lambda t: min(t / RAMP, 1.0))
    loaded = run(HOLD, lambda t: 1.0)
    return start, loaded, run(HOLD, lambda t: 0.0)


def _joint_x(model, h) -> np.ndarray:
    """World x of each ROD joint's anchor at the start (parent body pose times joint_X_p)."""
    from alloy.solvers.rod.topology import ROD

    q, xp, parent = model.body_q.numpy().astype(np.float64), model.joint_X_p.numpy(), model.joint_parent.numpy()
    rod = [int(j) for j in h.joints if int(model.joint_type.numpy()[int(j)]) == ROD]
    return np.asarray([q[parent[j], 0] + _rotate(q[parent[j], 3:], xp[j, :3].astype(np.float64))[0] for j in rod])


def _cantilever(root_moment: float, length: float = 0.2, links: int = 20):
    model, solver, h, _ = _wire([(0.0, 0.0, 0.3), (length, 0.0, 0.3)], links)
    end = int(h.bodies[-1])
    x_f = _com(model, model.body_q.numpy().astype(np.float64), end)[0]
    arm = x_f - _joint_x(model, h)
    assert arm.size == links - 1 and np.all(arm > 0.0)
    force = root_moment / float(arm.max())             # root_moment at the first ROD joint (the clamp does not yield)
    start, loaded, released = _load(model, solver, end, lambda s, _: np.array([0, 0, -force * s, 0, 0, 0]))
    m = force * arm
    hj = length / (links - 1)
    theta = np.array([hj * _kappa(mi) for mi in m])
    kept = theta - m * hj / EI
    closed = (float(theta @ arm), float(kept @ arm))
    return (start[2] - loaded[2], start[2] - released[2]), closed, solver


@pytest.mark.law
def test_a_cantilever_wire_past_yield_keeps_the_closed_form_tip_set():
    """Law A at a root moment of 1.5 M_y (0.88 M_p; a 0.2 m wire, tip deflection 6.5 % of its length): the loaded
    tip deflection within 1.5 % of the closed form on the rod's own joints (measured -0.66 %: the lever arms shorten
    as it bends) and the permanent set within 5 % of its closed form (measured -2.8 %; -1.6 % at 1.3 M_y). Near full
    yield it degrades as docs/known_gaps/rod-plastic-near-full-yield.md states: 1.6 M_y (0.94 M_p) sets -9.4 %."""
    (loaded, kept), (closed_loaded, closed_kept), solver = _cantilever(1.5 * M_Y)
    assert closed_kept > 0.05 * closed_loaded                    # the case really yields near the root
    assert loaded == pytest.approx(closed_loaded, rel=0.015, abs=0)
    assert kept == pytest.approx(closed_kept, rel=0.05, abs=0)
    assert float(solver.rod_plastic.ebar_max.numpy().max()) > 0.0


@pytest.mark.law
def test_a_cantilever_wire_below_yield_springs_back():
    (loaded, kept), (closed_loaded, closed_kept), solver = _cantilever(0.8 * M_Y)
    assert abs(closed_kept) < 1e-9 * closed_loaded and loaded == pytest.approx(closed_loaded, rel=0.02, abs=0)
    assert abs(kept) < 1e-3 * loaded and not solver.rod_plastic.ebar.numpy().any()


def _helix(rc: float, turns: float, pitch: float, z0: float = 0.3, k: int = 400) -> list[tuple[float, float, float]]:
    th = np.linspace(0.0, 2.0 * math.pi * turns, k)
    return [(rc * math.cos(t), rc * math.sin(t), z0 + pitch * t / (2.0 * math.pi)) for t in th]


def _spring(torque: float, rc: float = 0.01, turns: float = 3.0, pitch: float = 4e-3, links: int = 96):
    model, solver, h, wire = _wire(_helix(rc, turns, pitch), links)
    end = int(h.bodies[-1])
    force = torque / rc

    def on_axis(s: float, com: np.ndarray) -> np.ndarray:
        """F on the spring's axis (-z through x = y = 0), carried to the end segment's COM."""
        f = np.array([0.0, 0.0, -force * s])
        return np.concatenate([f, np.cross(np.array([-com[0], -com[1], 0.0]), f)])
    start, loaded, released = _load(model, solver, end, on_axis)
    closed = (rc * wire * _phi(torque), rc * wire * (_phi(torque) - torque / GJ))
    return (start[2] - loaded[2], start[2] - released[2]), closed, solver


@pytest.mark.law
def test_a_coil_spring_compressed_past_yield_loses_the_closed_form_free_length():
    """Law B at a wire torque of 1.25 T_y (0.94 T_p; Rc 10 mm, 3 turns, pitch 4 mm, 96 segments, 32 per turn): the
    loaded compression within 2 % of wire torsion theory (measured +0.87 %; +0.33 % elastic at 0.8 T_y) and the free
    length lost within 5 % of the theory's (measured 0.584 of 0.572 mm, +2.0 %).
    At 16 segments per turn the wire reads 1.9 % short and keeps 8 % less: each joint takes its twist about the
    child segment's axis, half a corner (11.25 deg) off the wire's tangent, so part of the torsion goes to the
    stiffer bend axis (WP9's review). Near T_p the flat torsion tangent amplifies that shortfall in the kept length."""
    (loaded, lost), (closed_loaded, closed_lost), solver = _spring(1.25 * T_Y)
    assert closed_lost > 0.1 * closed_loaded
    assert loaded == pytest.approx(closed_loaded, rel=0.02, abs=0)
    assert lost == pytest.approx(closed_lost, rel=0.05, abs=0)
    assert float(solver.rod_plastic.ebar_max.numpy().max()) > 0.0


@pytest.mark.law
def test_a_coil_spring_below_yield_returns_to_its_free_length():
    (loaded, lost), (closed_loaded, _), solver = _spring(0.8 * T_Y)
    assert loaded == pytest.approx(closed_loaded, rel=0.02, abs=0)
    assert abs(lost) < 2e-3 * loaded and not solver.rod_plastic.ebar.numpy().any()
