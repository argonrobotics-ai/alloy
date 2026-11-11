# SPDX-License-Identifier: Apache-2.0
"""SCR-WP2-10: the rod direct solve (alloy.solvers.rod) against D8's references, on CPU.

The cantilever is rodcheck's D7 layout: half-length end segments, cylinder masses, hinge ke = EI/h, root kinematic.
Measured at 20 segments ([internal note]):
- SolverVBDRod at 1 iteration: sag -0.19 %, drop -3.3 %, period -0.25 %;
- stock SolverVBD at 10 iterations: +88 % and +559 %.
The tests use 12 segments to stay short.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.route import rodref
from alloy.route.rodsim import period_from_signal
from alloy.solvers.rod import RodTopology, SolverVBDRod, hook_order_ok

G = 9.81
L, R, RHO = 0.5, 0.003, 1200.0
LIN = RHO * math.pi * R**2
DT = 1.25e-3


def _segments(n: int) -> tuple[np.ndarray, float]:
    h = L / (n - 1)
    ls = np.full(n, h)
    ls[0] = ls[-1] = 0.5 * h
    return ls, h


def _cantilever(n: int, EI: float, kd: float = 0.0, plug: float = 0.0, ground: bool = False):
    """(builder, rod bodies, rod joints, plug body or -1, plug weld or -1, plug half length)."""
    ls, h = _segments(n)
    x = np.concatenate([[0.0], np.cumsum(ls)])
    E = EI / (math.pi * R**4 / 4.0)
    area = math.pi * R**2
    b = newton.ModelBuilder()
    if ground:
        b.add_ground_plane()
    pts = np.stack([x, np.zeros_like(x), np.full_like(x, 1.0)], axis=1)
    bodies, joints = b.add_rod(rod=newton.Rod(pts, radius=R), cfg=newton.ModelBuilder.ShapeConfig(density=RHO, gap=0.002),
                               stretch_stiffness=E * area / h, shear_stiffness=0.35 * E * area / h,
                               bend_stiffness=EI / h, twist_stiffness=EI / 1.3 / h, bend_damping=kd, twist_damping=kd,
                               label="rod", wrap_in_articulation=plug == 0.0)
    for i, bi in enumerate(bodies):
        m = LIN * ls[i]
        It, Ia = m * (3 * R**2 + ls[i] ** 2) / 12.0, 0.5 * m * R**2
        b.body_mass[bi], b.body_inv_mass[bi] = m, 1.0 / m
        b.body_inertia[bi] = wp.mat33(It, 0, 0, 0, It, 0, 0, 0, Ia)
        b.body_inv_inertia[bi] = wp.mat33(1 / It, 0, 0, 0, 1 / It, 0, 0, 0, 1 / Ia)
    b.body_flags[bodies[0]] = int(newton.BodyFlags.KINEMATIC)
    box, weld, a = -1, -1, 0.01
    if plug > 0.0:
        box = b.add_link(xform=wp.transform((L + a, 0.0, 1.0), wp.quat_identity()), label="plug")
        b.add_shape_box(box, hx=a, hy=0.005, hz=0.005,
                        cfg=newton.ModelBuilder.ShapeConfig(density=plug / (8 * a * 0.005 * 0.005)))
        X_p = wp.transform_inverse(b.body_q[bodies[-1]]) * wp.transform((L, 0.0, 1.0), wp.quat_identity())
        weld = b.add_joint_fixed(bodies[-1], box, parent_xform=X_p,
                                 child_xform=wp.transform((-a, 0.0, 0.0), wp.quat_identity()))
        b.add_articulation([*joints, weld])
    return b, bodies, joints, box, weld, a


def _tf(row) -> wp.transform:
    return wp.transform(wp.vec3(*row[:3]), wp.quat(*row[3:7]))


def _tip_drop(q: np.ndarray, body: int, half: float) -> float:
    t = _tf(q[body]) * wp.transform((0.0, 0.0, half), wp.quat_identity())
    return 1.0 - float(t.p[2])


def _run(model, solver, steps: int, probe):
    s0, s1, c = model.state(), model.state(), model.control()
    out = []
    for _ in range(steps):
        s0.clear_forces()
        solver.step(s0, s1, c, None, DT)
        s0, s1 = s1, s0
        out.append(probe(s0.body_q.numpy()))
    return np.asarray(out), s0


def test_newton_hook_order_holds_on_the_pinned_newton():
    assert hook_order_ok()


def test_topology_finds_the_rod_and_its_welded_plug():
    b, bodies, joints, box, weld, _ = _cantilever(8, 0.01, plug=0.02)
    b.color()
    top = RodTopology.from_model(b.finalize(device="cpu"))
    assert top.rods == 1
    assert list(top.body) == [*bodies, box]
    assert list(top.joint) == [-1, *joints, weld]


def test_rod_groups_must_name_every_chain_once():
    b, *_ = _cantilever(6, 0.01)
    b.color()
    with pytest.raises(ValueError, match="every chain"):
        SolverVBDRod(b.finalize(device="cpu"), rod_groups=[[0], [0]], iterations=1)


def test_static_sag_meets_d8_at_one_iteration():
    n = 12
    EI = LIN * G * L**3 / 5.0
    w1 = rodref.CANTILEVER_BETA1**2 * math.sqrt(EI / (LIN * L**4))
    b, bodies, *_ = _cantilever(n, EI, kd=(1.4 / w1) * EI / _segments(n)[1])
    b.color()
    model = b.finalize(device="cpu")
    half = 0.5 * _segments(n)[0][-1]
    drop, _ = _run(model, SolverVBDRod(model, iterations=1), int(5 * 2 * math.pi / w1 / DT),
                   lambda q: _tip_drop(q, bodies[-1], half))
    ref = rodref.elastica_tip_drop(L, LIN, EI)
    disc = rodref.static_tip_drop(rodref.chain_layout(n, L, LIN, EI, R))
    assert abs(drop[-1] / ref - 1.0) < 0.01, (drop[-1], ref)          # D8: 5 %
    assert abs(drop[-1] / disc - 1.0) < 0.002, (drop[-1], disc)       # the discrete chain it solves


@pytest.mark.parametrize("split", [False, True])
def test_release_meets_d8_and_both_paths_agree(split):
    n = 12
    EI = LIN * G * L**3 / 0.5
    b, bodies, *_ = _cantilever(n, EI)
    b.color()
    model = b.finalize(device="cpu")
    lay = rodref.chain_layout(n, L, LIN, EI, R)
    period = rodref.release_reference(lay, L, LIN, EI, np.asarray([0.0])).period
    half = 0.5 * _segments(n)[0][-1]
    steps = int(1.6 * period / DT)
    drop, _ = _run(model, SolverVBDRod(model, iterations=1, rod_split=split), steps,
                   lambda q: _tip_drop(q, bodies[-1], half))
    t = DT * np.arange(1, steps + 1)
    ref = rodref.release_reference(lay, L, LIN, EI, t)
    peak = float(drop[t <= ref.period].max())
    assert abs(peak / ref.peak_drop - 1.0) < 0.05, (peak, ref.peak_drop)              # D8: drop 5 %
    assert abs(period_from_signal(t, drop - ref.static_drop) / ref.period - 1.0) < 0.02  # D8: period 2 %


def test_stock_vbd_is_limp_on_the_same_release():
    """Why the package exists: the same rod on stock SolverVBD at 10 iterations drops several times too far."""
    n = 12
    EI = LIN * G * L**3 / 0.5
    b, bodies, *_ = _cantilever(n, EI)
    b.color()
    model = b.finalize(device="cpu")
    lay = rodref.chain_layout(n, L, LIN, EI, R)
    ref = rodref.release_reference(lay, L, LIN, EI, np.asarray([0.0]))
    drop, _ = _run(model, newton.solvers.SolverVBD(model, iterations=10), int(ref.period / DT),
                   lambda q: _tip_drop(q, bodies[-1], 0.5 * _segments(n)[0][-1]))
    assert drop.max() > 2.0 * ref.peak_drop


def test_welded_plug_in_the_chain_matches_beam_theory():
    n, plug = 12, 0.020
    EI = LIN * G * L**3 / 0.2
    w1 = rodref.CANTILEVER_BETA1**2 * math.sqrt(EI / (LIN * L**4))
    b, bodies, joints, box, weld, a = _cantilever(n, EI, kd=(1.4 / w1) * EI / _segments(n)[1], plug=plug)
    b.color()
    model = b.finalize(device="cpu")
    solver = SolverVBDRod(model, iterations=1)
    assert list(solver.rods.body) == [*bodies, box]
    X_p = _tf(model.joint_X_p.numpy()[weld])
    _, s = _run(model, solver, int(6 * 2 * math.pi / w1 / DT), lambda q: 0.0)
    q = s.body_q.numpy()
    at_tip = _tf(q[bodies[-1]]) * X_p
    on_plug = _tf(q[box]) * wp.transform((-a, 0.0, 0.0), wp.quat_identity())
    P = float(model.body_mass.numpy()[box]) * G
    theory = LIN * G * L**4 / (8 * EI) + P * L**3 / (3 * EI) + P * a * L**2 / (2 * EI)
    assert abs((1.0 - float(at_tip.p[2])) / theory - 1.0) < 0.03
    assert np.linalg.norm(np.array(at_tip.p) - np.array(on_plug.p)) < 1e-5


def test_cable_dropped_on_the_ground_rests_on_it():
    n = 20
    b, bodies, *_ = _cantilever(n, 1e-4, kd=1e-6, ground=True)
    b.body_flags[bodies[0]] = int(newton.BodyFlags.DYNAMIC)   # a free cable, laid out 20 mm above the ground
    for bi in bodies:
        q = b.body_q[bi]
        b.body_q[bi] = wp.transform(wp.vec3(q.p[0], q.p[1], 0.02), q.q)
    b.color()
    model = b.finalize(device="cpu")
    pipe = newton.CollisionPipeline(model)
    contacts = pipe.contacts()
    solver = SolverVBDRod(model, iterations=2)
    s0, s1, c = model.state(), model.state(), model.control()
    for _ in range(800):
        s0.clear_forces()
        pipe.collide(s0, contacts)
        solver.step(s0, s1, c, contacts, DT)
        s0, s1 = s1, s0
    q, qd = s0.body_q.numpy(), s0.body_qd.numpy()
    assert np.isfinite(q).all()
    assert q[bodies, 2].min() - R > -5e-5                     # on the ground, not through it
    assert q[bodies, 2].max() - R < 1e-3                      # lying flat
    assert np.linalg.norm(qd[bodies, :3], axis=1).max() < 0.01


def test_default_groups_split_rods_of_one_world_and_share_across_worlds():
    """Rods of one world update one after the other (contacts between them are Gauss-Seidel); the k-th rod of every
    world shares group k, so one rod per world is one parallel group."""
    one = newton.ModelBuilder()
    for y in (0.0, 0.05):
        pts = np.stack([np.linspace(0.0, 0.2, 6), np.full(6, y), np.full(6, 1.0)], axis=1)
        one.add_rod(rod=newton.Rod(pts, radius=R), bend_stiffness=0.1, label=f"r{y}")
    b = newton.ModelBuilder()
    for _ in range(3):
        b.add_world(one)
    b.color()
    solver = SolverVBDRod(b.finalize(device="cpu"), iterations=1)
    groups = [g[0].numpy().tolist() for g in solver._groups]
    assert groups == [[0, 2, 4], [1, 3, 5]]


def test_rest_rotation_is_newtons_structural_rest():
    """A rod laid out straight with a pre-curved rest (a 90-degree arc over 8 segments) carries the arc's relative
    rotation in A0: about pi/16 per joint at the interior, so the rod solve relaxes it to the arc, not to straight."""
    from alloy.solvers.rod import rest_relative

    n = 8
    th = np.linspace(0.0, 0.5 * math.pi, n + 1)
    rest = np.stack([0.3 * np.sin(th), np.zeros_like(th), 1.0 - 0.3 * (1.0 - np.cos(th))], axis=1)
    seg = np.linalg.norm(np.diff(rest, axis=0), axis=1)
    x = np.concatenate([[0.0], np.cumsum(seg)])
    straight = np.stack([x, np.zeros_like(x), np.ones_like(x)], axis=1)
    b = newton.ModelBuilder()
    _, joints = b.add_rod(rod=newton.Rod(straight, rest_points=rest, radius=R), bend_stiffness=0.1, label="arc")
    b.color()
    model = b.finalize(device="cpu")
    A0 = rest_relative(model)[joints]
    angles = 2.0 * np.degrees(np.arccos(np.clip(np.abs(A0[:, 3]), 0.0, 1.0)))
    assert np.allclose(angles, 90.0 / n, atol=0.5), angles


def test_a_skipped_rod_pass_raises_instead_of_freezing_the_rod(monkeypatch):
    """Chain bodies are in no colour: if Newton ever stops calling the truncation hook, the step must fail loudly."""
    b, *_ = _cantilever(6, 0.01)
    b.color()
    model = b.finalize(device="cpu")
    solver = SolverVBDRod(model, iterations=1)
    monkeypatch.setattr(newton.solvers.SolverVBD, "_rigid_penetration_free_truncation", lambda self, state, contacts: None)
    monkeypatch.setattr(SolverVBDRod, "_rigid_penetration_free_truncation",
                        lambda self, state, contacts: None)
    s0, s1, c = model.state(), model.state(), model.control()
    with pytest.raises(RuntimeError, match="did not run"):
        solver.step(s0, s1, c, None, DT)


def test_a_chain_without_joint_stiffness_steps_like_stock_vbd():
    """With every ROD slot at ke = kd = 0 the chain's bodies are independent, so one rod-solve iteration is Newton's own
    per-body step: inertia, ground contact and the 6x6 block layout. It pins those conventions."""
    n = 6
    x = np.linspace(0.0, 0.3, n + 1)
    pts = np.stack([x, np.zeros_like(x), 0.004 + 0.05 * x], axis=1)        # tilted, low end 1 mm above the ground
    out = []
    for cls in (newton.solvers.SolverVBD, SolverVBDRod):
        b = newton.ModelBuilder()
        b.add_ground_plane()
        b.add_rod(rod=newton.Rod(pts, radius=R), cfg=newton.ModelBuilder.ShapeConfig(density=RHO, gap=0.002),
                  stretch_stiffness=0.0, shear_stiffness=0.0, bend_stiffness=0.0, twist_stiffness=0.0, label="loose")
        b.color()
        model = b.finalize(device="cpu")
        pipe = newton.CollisionPipeline(model)
        contacts = pipe.contacts()
        solver = cls(model, iterations=1)
        s0, s1, c = model.state(), model.state(), model.control()
        for _ in range(240):
            s0.clear_forces()
            pipe.collide(s0, contacts)
            solver.step(s0, s1, c, contacts, DT)
            s0, s1 = s1, s0
        out.append(s0.body_q.numpy())
    stock, rod = out
    assert np.isfinite(rod).all()
    assert stock[:, 2].min() < 0.004                                        # they reached the ground
    assert np.abs(rod[:, :3] - stock[:, :3]).max() < 2e-5, np.abs(rod[:, :3] - stock[:, :3]).max()
