# SPDX-License-Identifier: Apache-2.0
"""Exit-speed cap for cloth particles leaving a body contact (SCR-WP11-KICK, alloy/solvers/vbd/exit_speed.py). CPU.

VBD puts a particle that starts a step inside a body record back near the surface within the step and reads its
velocity off the displacement, so it leaves at the depth the step resolved / dt, whatever the stiffness (SC4's ClothFold
pinch: towel corner particles at 14-33 m/s from 19.5-39 us steps). Rigs, no gravity, make_vbd's defaults (log
barrier), soft_contact_ke 2e6 mixed with the shape's 2.5e3: one 60 mg particle (r 1 mm) over a static box; a 4 cm
cloth cell with a sphere body pressed into a face (one full-surface face record, no vertex record touching).
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.coupling.proxy_contacts import ProxyContactPipeline
from alloy.solvers.vbd import EXIT_KEY, make_vbd
from alloy.solvers.vbd import exit_speed as X

R, MASS, DT, KE, CAP = 1.0e-3, 6.0e-5, 2.0e-5, 2.5e3, 1.0


def _model(b: newton.ModelBuilder, mu: float = 0.5) -> newton.Model:
    b.color()
    m = b.finalize(device="cpu")
    m.soft_contact_ke, m.soft_contact_kd, m.soft_contact_mu = 2.0e6, 0.0, mu
    return m


class Box:
    """One particle started ``depth`` inside a static box's top face (z 0), velocity ``vel``."""

    def __init__(self, depth: float, cap: float | None, vel=(0.0, 0.0, 0.0), mu: float = 0.5) -> None:
        b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        b.add_particle(pos=wp.vec3(0.0, 0.0, R - depth), vel=wp.vec3(*vel), mass=MASS, radius=R)
        cfg = newton.ModelBuilder.ShapeConfig(ke=KE, kd=0.0, mu=mu, margin=0.0, gap=0.002)
        b.add_shape_box(-1, xform=wp.transform((0.0, 0.0, -0.01), wp.quat_identity()), hx=0.05, hy=0.05, hz=0.01,
                        cfg=cfg)
        self.model = m = _model(b, mu)
        self.solver = make_vbd(m, {"iterations": 10, EXIT_KEY: cap})
        self.pipe = newton.CollisionPipeline(m, soft_contact_gap=0.002)
        self.contacts = self.pipe.contacts()
        self.s0, self.s1 = m.state(), m.state()

    def step(self) -> tuple[np.ndarray, np.ndarray]:
        """One step: the particle's (position, velocity) after it."""
        self.pipe.collide(self.s0, self.contacts)
        self.solver.step(self.s0, self.s1, None, self.contacts, DT)
        self.s0, self.s1 = self.s1, self.s0
        return self.s0.particle_q.numpy()[0].copy(), self.s0.particle_qd.numpy()[0].copy()


@pytest.mark.law
def test_a_particle_started_inside_leaves_at_its_resolved_depth_over_dt_and_the_cap_holds_it():
    """Law (exit speed), 0.5 mm deep, dt 20 us: stock leaves at the depth the step resolved / dt (18.7 m/s, within
    0.1 %); with the cap at 1 m/s it leaves at 1 m/s, keeps that speed while the rest of the depth resolves, and is out
    of the box within 6 steps (the penetration is still resolved)."""
    depth = 5.0e-4
    q, v = Box(depth, None).step()
    assert v[2] > 15.0 and v[2] * DT == pytest.approx(depth - (R - q[2]), rel=1.0e-3), (q, v)
    rig = Box(depth, CAP)
    rows = [rig.step() for _ in range(6)]
    assert rows[0][1][2] == pytest.approx(CAP, abs=1.0e-4), rows[0]
    assert all(r[1][2] <= CAP + 1.0e-4 for r in rows), rows
    assert R - rows[-1][0][2] < 0.0, rows[-1]
    assert int(rig.solver.alloy_exit.capped.numpy()[0]) >= 1


@pytest.mark.parametrize("start", [(1.0e-5, 2.0), (-2.0e-4, -2.0)], ids=["leaving", "arriving"])
def test_a_particle_leaving_or_arriving_steps_as_stock(start):
    """A resting depth (10 um) lifted off at 2 m/s keeps its speed (the cut is bounded by the start depth: 0.5 m/s,
    under the cap), and an impact at 2 m/s still stops in one step: both bit for bit as stock, cap never cut."""
    depth, vz = start
    out = []
    for cap in (None, CAP):
        rig = Box(depth, cap, vel=(0.0, 0.0, vz))
        out.append(np.array([np.concatenate(rig.step()) for _ in range(10)]))
    assert np.array_equal(out[0], out[1])
    assert abs(out[0][-1, 5]) < 1.0 if vz < 0.0 else out[0][-1, 5] > 1.9


def test_only_the_normal_velocity_is_cut():
    """0.5 mm deep sliding at 0.3 m/s (frictionless): the step's position and tangential velocity are stock's, bit for
    bit."""
    (q0, v0), (q1, v1) = (Box(5.0e-4, cap, vel=(0.3, 0.0, 0.0), mu=0.0).step() for cap in (None, CAP))
    assert np.array_equal(q0, q1) and np.array_equal(v0[:2], v1[:2]) and v0[0] == pytest.approx(0.3, rel=1.0e-4)
    assert v0[2] > 15.0 and v1[2] == pytest.approx(CAP, abs=1.0e-4)


def _two_records(cap: float | None) -> float:
    """One step of a particle 0.3 mm into a soft box below and 0.1 mm into a 100x stiffer box above: its z velocity."""
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    zp = R - 3.0e-4
    b.add_particle(pos=wp.vec3(0.0, 0.0, zp), vel=wp.vec3(0.0), mass=MASS, radius=R)
    for ke, z in ((KE, -0.01), (2.0e8, zp + R - 1.0e-4 + 0.01)):
        cfg = newton.ModelBuilder.ShapeConfig(ke=ke, kd=0.0, mu=0.5, margin=0.0, gap=0.002)
        b.add_shape_box(-1, xform=wp.transform((0.0, 0.0, z), wp.quat_identity()), hx=0.05, hy=0.05, hz=0.01, cfg=cfg)
    m = _model(b)
    solver = make_vbd(m, {"iterations": 10, EXIT_KEY: cap})
    pipe = newton.CollisionPipeline(m, soft_contact_gap=0.002)
    c, s0, s1 = pipe.contacts(), m.state(), m.state()
    pipe.collide(s0, c)
    solver.step(s0, s1, None, c, DT)
    return float(s1.particle_qd.numpy()[0, 2])


@pytest.mark.law
def test_a_particle_thrown_out_of_its_shallower_record_is_capped():
    """Law (owner), AUDIT-STACK 2026-11-10: started deeper in a soft record and pushed out of a stiff one (a pinch
    between unequal fingers), stock leaves the stiff one at 3.5 m/s; the cap holds it to 1 m/s. A cut by the deepest
    record (the first slice) left the kick."""
    stock, capped = _two_records(None), _two_records(CAP)
    assert stock < -3.0 and capped == pytest.approx(-CAP, abs=1.0e-4), (stock, capped)


def test_of_two_records_the_one_that_cuts_most_caps():
    """A particle 0.3 mm into one box and 0.5 mm into a second, both tops facing up (the shallow box's record comes
    first): the step leaves it 0.13 mm inside the deep one at 18.7 m/s. The deep record's cut (17.7 m/s) wins over the
    shallow one's (14 m/s, bounded by its 0.3 mm), so it leaves at the cap, not at 4.7 m/s."""
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    b.add_particle(pos=wp.vec3(0.0, 0.0, R - 5.0e-4), vel=wp.vec3(0.0), mass=MASS, radius=R)
    for top in (-2.0e-4, 0.0):
        cfg = newton.ModelBuilder.ShapeConfig(ke=KE, kd=0.0, mu=0.5, margin=0.0, gap=0.002)
        b.add_shape_box(-1, xform=wp.transform((0.0, 0.0, top - 0.01), wp.quat_identity()), hx=0.05, hy=0.05,
                        hz=0.01, cfg=cfg)
    m = _model(b)
    solver = make_vbd(m, {"iterations": 10, EXIT_KEY: CAP})
    pipe = newton.CollisionPipeline(m, soft_contact_gap=0.002)
    c, s0, s1 = pipe.contacts(), m.state(), m.state()
    pipe.collide(s0, c)
    assert c.soft_contact_shape.numpy()[:2].tolist() == [0, 1]
    solver.step(s0, s1, None, c, DT)
    assert float(s1.particle_qd.numpy()[0, 2]) == pytest.approx(CAP, abs=1.0e-4)


def _pad_release(cap: float | None, press: float = 60.0, speed: float = 3.0) -> float:
    """A particle pressed onto a pad (a body moved by an outside solver) at ``press`` N until it rests, then let go as
    the pad pulls away at ``speed``: its z velocity after the first step."""
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    b.add_particle(pos=wp.vec3(0.0, 0.0, R), vel=wp.vec3(0.0), mass=MASS, radius=R)
    body = b.add_body(xform=wp.transform((0.0, 0.0, -0.01), wp.quat_identity()))
    cfg = newton.ModelBuilder.ShapeConfig(ke=KE, kd=0.0, mu=0.5, margin=0.0, gap=0.002, density=1.0e3)
    b.add_shape_box(body, hx=0.05, hy=0.05, hz=0.01, cfg=cfg)
    m = _model(b)
    solver = make_vbd(m, {"iterations": 10, EXIT_KEY: cap, "integrate_with_external_rigid_solver": True})
    pipe = newton.CollisionPipeline(m, soft_contact_gap=0.002)
    c, s0, s1 = pipe.contacts(), m.state(), m.state()
    z = -0.01
    for k in range(301):
        z1 = z - (speed * DT if k == 300 else 0.0)
        s0.particle_f.assign(np.array([[0.0, 0.0, 0.0 if k == 300 else -press]], np.float32))
        s0.body_q.assign(np.array([[0.0, 0.0, z, 0.0, 0.0, 0.0, 1.0]], np.float32))
        s1.body_q.assign(np.array([[0.0, 0.0, z1, 0.0, 0.0, 0.0, 1.0]], np.float32))
        pipe.collide(s0, c)
        solver.step(s0, s1, None, c, DT)
        s0, s1, z = s1, s0, z1
    return float(s0.particle_qd.numpy()[0, 2])


@pytest.mark.law
def test_a_particle_resting_on_a_pad_that_pulls_away_is_not_pulled_after_it():
    """Law (own outward speed), AUDIT-STACK 2026-11-10: a particle pressed 131 um into a pad at 60 N, let go as the
    pad pulls away at 3 m/s (60 um a step): stock springs off at +1.73 m/s; capped it does not follow the pad down
    (z velocity >= 0, at most stock's). A cut relative to the pad alone (the first slice) sent it down at 0.83 m/s."""
    stock, capped = _pad_release(None), _pad_release(CAP)
    assert stock > 1.0 and 0.0 <= capped <= stock, (stock, capped)


def _face_rig(cap: float | None, tri_ke: float) -> tuple[newton.State, np.ndarray, np.ndarray]:
    """A sphere body 0.3 mm into a 4 cm cell's face: (state after one step, the record's corners, its weights)."""
    z, cell, sr = 0.5, 0.04, 0.005
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    b.add_cloth_grid(pos=wp.vec3(0.0, 0.0, z), rot=wp.quat_identity(), vel=wp.vec3(0.0), dim_x=1, dim_y=1,
                     cell_x=cell, cell_y=cell, mass=MASS, particle_radius=R, tri_ke=tri_ke, tri_ka=tri_ke, tri_kd=0.0,
                     edge_ke=1.0e-3)
    cfg = newton.ModelBuilder.ShapeConfig(ke=KE, kd=0.0, mu=0.5, margin=0.0, gap=0.002, density=1.0e4)
    body = b.add_body(xform=wp.transform((0.012, 0.007, z + sr + R - 3.0e-4), wp.quat_identity()))
    b.add_shape_sphere(body, radius=sr, cfg=cfg)
    m = _model(b)
    solver = make_vbd(m, {"iterations": 10, EXIT_KEY: cap})
    pipe = ProxyContactPipeline(full_surface=True, full_surface_shapes=np.flatnonzero(m.shape_body.numpy() >= 0),
                                soft_contact_gap=0.002)(m)
    c = pipe.contacts()
    s0, s1 = m.state(), m.state()
    pipe.collide(s0, c)
    solver.step(s0, s1, None, c, DT)
    assert int(c.soft_contact_count.numpy()[0]) == 1
    return s1, c.soft_contact_indices.numpy()[0], c.soft_contact_barycentric.numpy()[0]


@pytest.mark.parametrize("tri_ke", [1.0e2, 1.0e6], ids=["soft", "stiff"])
def test_a_face_record_cuts_its_corners_by_their_share(tri_ke):
    """Full-surface face record (weights 0.525 / 0.3 / 0.175): stock throws its contact point out at over 5 m/s
    relative to the sphere; capped, the contact point leaves at about the cap (1.04 m/s on the soft cell: the corners'
    moves are not exactly the shares the cut assumes) and no corner leaves at more than 1.2 m/s. On a stiff cell the
    corners move together, so the heaviest-weighted corner's share of the cut is more than its own outward speed: the
    cut stops at the cap instead of sending it back into the sphere."""
    speeds = []
    for cap in (None, CAP):
        s, ids, w = _face_rig(cap, tri_ke)
        out = -(s.particle_qd.numpy()[ids, 2] - s.body_qd.numpy()[0, 2])     # outward = -z, relative to the sphere
        speeds.append((float(w @ out), out))
    assert speeds[0][0] > 5.0, speeds[0]
    assert CAP * 0.99 < speeds[1][0] < CAP * 1.1, speeds[1]
    assert CAP * 0.99 < speeds[1][1].min() and speeds[1][1].max() < 1.2 * CAP, speeds[1]


def test_the_option_reaches_make_vbd(monkeypatch):
    """Off by default; a speed or True (DEFAULT_SPEED) as the entry kwarg, or ALLOY_EXIT_SPEED; 0 is off; < 0 raises."""
    m = Box(0.0, None).model
    monkeypatch.delenv(X.EXIT_ENV, raising=False)
    assert not isinstance(make_vbd(m, {"iterations": 2}), X.ExitSpeedMixin)
    assert make_vbd(m, {"iterations": 2, EXIT_KEY: 0.5}).alloy_exit.v_cap == 0.5
    assert make_vbd(m, {"iterations": 2, EXIT_KEY: True}).alloy_exit.v_cap == X.DEFAULT_SPEED
    assert not isinstance(make_vbd(m, {"iterations": 2, EXIT_KEY: 0.0}), X.ExitSpeedMixin)
    monkeypatch.setenv(X.EXIT_ENV, "2.5")
    assert make_vbd(m, {"iterations": 2}).alloy_exit.v_cap == 2.5
    assert not isinstance(make_vbd(m, {"iterations": 2, EXIT_KEY: False}), X.ExitSpeedMixin)
    with pytest.raises(ValueError, match=EXIT_KEY):
        make_vbd(m, {"iterations": 2, EXIT_KEY: -1.0})
