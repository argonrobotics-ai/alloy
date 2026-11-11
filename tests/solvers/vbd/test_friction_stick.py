# SPDX-License-Identifier: Apache-2.0
"""Stick friction for cloth held by rigid contacts (SCR-WP11-STICK, alloy/solvers/vbd/stick.py). CPU.

Rig: a 4 x 12 cm cloth strip (1 cm cells, 1 g vertices, no gravity) hangs in the x-z plane. Two static pads overlap
the 1 mm particle radius by 0.1 mm on either side of its top three rows, so each of the 15 pinched vertices feels
ke pen = 1 N from each pad (linear penalty): a capacity of 2 mu ke pen = 0.6 N per vertex. A downward load of
``frac`` x capacity on each pinched vertex, ramped over 0.1 s. Newton's smoothed Coulomb friction lets the pinch creep
at friction_epsilon (1 - sqrt(1 - frac)) whatever the step; the stick holds it with no drift and slides past the
limit as before. Rows (10 and 40 iterations, eps 1e-2 and 1e-3, up to 0.9 of the limit): [internal note] notes/wp11-stick.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.solvers.vbd import STICK_KEY, SolverVBDAlloy, make_vbd
from alloy.solvers.vbd import stick as S

Z0, NX, NY, DX = 0.5, 4, 12, 0.01
KE, MU, PEN, R, MASS, DT, RAMP = 1.0e4, 0.3, 1.0e-4, 1.0e-3, 1.0e-3, 5.0e-4, 0.1


class Pinch:
    def __init__(self, stick: bool, eps: float = 1.0e-2, far: float = 0.0, iterations: int = 10) -> None:
        b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        upright = wp.quat_from_axis_angle(wp.vec3(1, 0, 0), math.pi / 2)
        b.add_cloth_grid(pos=wp.vec3(0.0, 0.0, Z0 - NY * DX), rot=upright, vel=wp.vec3(0.0), dim_x=NX, dim_y=NY,
                         cell_x=DX, cell_y=DX, mass=MASS, particle_radius=R, tri_ke=1.0e3, tri_ka=1.0e3, tri_kd=0.0,
                         edge_ke=1.0e-3)
        cfg = newton.ModelBuilder.ShapeConfig(ke=KE, kd=0.0, mu=MU, margin=0.0, gap=0.002)
        for side in (+1, -1):
            y = side * (R - PEN + 0.005 + far)
            b.add_shape_box(-1, xform=wp.transform((0.5 * NX * DX, y, Z0 - 0.01), wp.quat_identity()),
                            hx=0.5 * NX * DX + 0.01, hy=0.005, hz=0.015, cfg=cfg)
        b.color()
        self.model = m = b.finalize(device="cpu")
        m.soft_contact_ke, m.soft_contact_kd, m.soft_contact_mu = KE, 0.0, MU
        cls = S.with_stick(SolverVBDAlloy) if stick else SolverVBDAlloy
        self.solver = cls(m, iterations=iterations, friction_epsilon=eps, rigid_soft_contact_use_log_barrier=False)
        q = m.particle_q.numpy()
        self.top = np.flatnonzero(q[:, 2] > Z0 - 2.5 * DX)
        self.z0 = float(q[self.top, 2].mean())
        self.pipe = newton.CollisionPipeline(m, soft_contact_gap=0.002)
        self.contacts = self.pipe.contacts()
        self.s0, self.s1 = m.state(), m.state()
        self.k = 0

    def step(self, frac: float, collide: bool = True) -> None:
        f = np.zeros((self.model.particle_count, 3), np.float32)
        f[self.top, 2] = -min(1.0, (self.k + 1) * DT / RAMP) * frac * 2.0 * MU * KE * PEN
        self.s0.clear_forces()
        self.s0.particle_f.assign(f)
        if collide:
            self.pipe.collide(self.s0, self.contacts)
        self.solver.step(self.s0, self.s1, None, self.contacts, DT)
        self.s0, self.s1 = self.s1, self.s0
        self.k += 1

    def drop(self) -> float:
        """The pinched rows' drop from their start [m] (positive: out of the pinch)."""
        return self.z0 - float(self.s0.particle_q.numpy()[self.top, 2].mean())


def _hold(stick: bool, frac: float, seconds: float, **kw) -> tuple[float, float]:
    """(drop at the end, drop rate over the last half) of a pinch loaded at ``frac`` of its limit."""
    rig, drops = Pinch(stick, **kw), []
    for _ in range(int(round(seconds / DT))):
        rig.step(frac)
        drops.append(rig.drop())
    half = len(drops) // 2
    return drops[-1], (drops[-1] - drops[half]) / ((len(drops) - half) * DT)


@pytest.mark.law
def test_the_stock_pinch_creeps_at_the_smoothed_law_rate_and_the_stick_holds():
    """Law (hold), half the Coulomb limit, eps 1e-2, 0.6 s: stock creeps at eps (1 - sqrt(0.5)) = 2.93 mm/s (measured
    2.92); the stick's pinched rows stay within 10 um and do not drift (|rate| under 10 um/s)."""
    _, stock = _hold(False, 0.5, 0.6)
    drop, held = _hold(True, 0.5, 0.6)
    assert stock == pytest.approx(1.0e-2 * (1.0 - math.sqrt(0.5)), rel=0.03), stock
    assert abs(drop) < 1.0e-5 and abs(held) < 1.0e-5, (drop, held)


def test_past_the_limit_the_stick_slides_as_stock():
    """1.3 x the limit: both pinches slide out, with the same kinetic friction (drops within 1 %)."""
    stock, _ = _hold(False, 1.3, 0.3)
    stuck, _ = _hold(True, 1.3, 0.3)
    assert stock > 0.05 and stuck == pytest.approx(stock, rel=0.01), (stock, stuck)


def test_records_without_contact_step_bitwise_as_stock():
    """Pads 1 mm clear of the strip (records inside the 2 mm gap, none touching): the stick leaves every bit alone."""
    out = []
    for stick in (False, True):
        rig = Pinch(stick, far=0.001)
        for _ in range(40):
            rig.step(0.5)
        out.append(rig.s0.particle_q.numpy().copy())
    assert np.array_equal(out[0], out[1])


def test_a_record_not_refreshed_carries_one_shift_only():
    """Steps on the same records (no collide in between, as a coupler's re-step) take the last shift off before adding
    the next: the record's velocity is the pipeline's value plus the current shift, not a sum of shifts."""
    rig = Pinch(True)
    for _ in range(200):
        rig.step(0.5)
    rig.pipe.collide(rig.s0, rig.contacts)
    base = rig.contacts.soft_contact_body_vel.numpy().copy()
    for _ in range(3):
        rig.step(0.5, collide=False)
    n = int(rig.contacts.soft_contact_count.numpy()[0])
    shift = rig.solver.alloy_stick.added.numpy()[:n]
    assert np.abs(shift).max() > 0.0
    assert np.array_equal(rig.contacts.soft_contact_body_vel.numpy()[:n], (base[:n] + shift).astype(np.float32))


def test_reset_clears_the_anchors_and_the_snapshot_names_are_set():
    rig = Pinch(True)
    for _ in range(50):
        rig.step(0.5)
    st = rig.solver.alloy_stick
    assert (st.shape_of.numpy() != S.NO_SHAPE).sum() == 2 * len(rig.top)       # two pads per pinched vertex
    for key, arr in st.arrays().items():
        assert getattr(rig.solver, key) is arr and arr.shape[0] == rig.model.particle_count
    rig.solver.reset(rig.s0)
    assert (st.shape_of.numpy() == S.NO_SHAPE).all() and not st.slip.numpy().any()


def test_the_option_reaches_make_vbd(monkeypatch):
    """Off by default; the entry kwarg or ALLOY_FRICTION_STICK=1 mixes the stick in front of the entry's class."""
    m = Pinch(False).model
    monkeypatch.delenv(S.STICK_ENV, raising=False)
    assert not isinstance(make_vbd(m, {"iterations": 2}), S.StickMixin)
    assert isinstance(make_vbd(m, {"iterations": 2, STICK_KEY: True}), S.StickMixin)
    monkeypatch.setenv(S.STICK_ENV, "1")
    assert isinstance(make_vbd(m, {"iterations": 2}), S.StickMixin)
    assert not isinstance(make_vbd(m, {"iterations": 2, STICK_KEY: False}), S.StickMixin)


def test_two_records_of_one_particle_and_shape_share_one_anchor():
    """Full-surface contacts give a pinched vertex several records against one shape: they take one slot, written by
    the lowest record, and a second shape takes the other slot."""
    rig = Pinch(True)
    rig.step(0.5)
    st, cv = rig.solver.alloy_stick, rig.solver.alloy_stick._make_view(rig.contacts, rig.s1, rig.s0, DT)
    n = int(rig.contacts.soft_contact_count.numpy()[0])
    ids, shapes = rig.contacts.soft_contact_indices.numpy()[:n], rig.contacts.soft_contact_shape.numpy()[:n]
    p = int(ids[0, 0])
    mine = np.flatnonzero(ids[:, 0] == p)
    assert len(mine) == 2 and len(set(shapes[mine])) == 2              # one record per pad
    for key in ("soft_contact_indices", "soft_contact_shape", "soft_contact_body_pos", "soft_contact_body_vel",
                "soft_contact_normal"):
        arr = getattr(rig.contacts, key).numpy()
        arr[n] = arr[mine[0]]                                          # a duplicate of p's first record, appended
        getattr(rig.contacts, key).assign(arr)
    rig.contacts.soft_contact_count.assign(np.array([n + 1], np.int32))
    st.shape_of.fill_(S.NO_SHAPE)
    st._view, st._contacts = cv, rig.contacts
    st.end(rig.s0)
    got = st.shape_of.numpy()[p]
    used = sorted(int(g) for g in got if g != S.NO_SHAPE)
    assert used == sorted(shapes[mine].tolist()), got                   # no slot twice for one shape
    j = int(np.flatnonzero(got == shapes[mine[0]])[0])
    assert st.owner.numpy()[p, j] == min(mine[0], n)


def test_the_anchors_are_snapshot_state_per_world():
    """A coupled restart or a per-world restore takes the anchors with the world (alloy.coupling.adapters)."""
    from alloy.coupling.adapters import vbd_state_arrays, vbd_world_maps

    solver = Pinch(True).solver
    arrays, maps = vbd_state_arrays(solver), vbd_world_maps(solver)
    for key in ("vbd._stick_shape", "vbd._stick_slip"):
        assert key in arrays and key in maps, key


def _anchors(solver) -> tuple[np.ndarray, np.ndarray]:
    st = solver.alloy_stick
    return st.shape_of.numpy().copy(), st.slip.numpy().copy()


def test_a_restarted_coupled_pass_starts_from_the_exchanges_anchors():
    """A coupler steps the entry twice for one exchange (the lagged loop's two passes) from the same input: the
    second pass, restarted, must be the same map as the first (same anchors, same positions), not count the
    interval's slip twice (WP4 review). Substeps (no restart in between) chain."""
    from alloy._vendor.newton import StateFlags

    rig = Pinch(True)
    for _ in range(100):
        rig.step(0.5)
    flags = int(StateFlags.PARTICLE_Q | StateFlags.PARTICLE_QD)
    rig.pipe.collide(rig.s0, rig.contacts)
    s_in = rig.model.state()
    s_in.assign(rig.s0)
    runs = []
    for restart in (False, True):
        rig.s0.assign(s_in)
        rig.solver.coupling_notify_input_state_update(rig.s0, flags, iteration_restart=restart, dt=DT)
        rig.solver.step(rig.s0, rig.s1, None, rig.contacts, DT)
        runs.append((rig.s1.particle_q.numpy().copy(), *_anchors(rig.solver)))
    for a, b in zip(runs[0], runs[1]):
        assert np.array_equal(a, b)


def test_two_stick_solvers_on_one_record_buffer_raise():
    """A second stick on the same Contacts would read the first one's shift as a fresh value (AUDIT-STACK): refused."""
    a, b = Pinch(True), Pinch(True)
    a.step(0.5)
    b.contacts = a.contacts
    with pytest.raises(RuntimeError, match="share one Contacts"):
        b.solver.step(b.s0, b.s1, None, b.contacts, DT)


@pytest.mark.law
@pytest.mark.timeout(900)
def test_a_sheet_on_a_driven_plate_is_carried_along(monkeypatch):
    """Law (AUDIT-STACK 3b): a sheet resting on a level kinematic plate driven at 0.05 m/s goes with it. When VBD moves
    the plate itself, its finalization advances body_q_prev to the step's end; reading that after the step saw the
    plate still, and the anchors ratcheted the sheet back at friction_epsilon (10 mm/s, 5 mm over this drive). Fixed:
    the lag stays at stock's sub-millimetre start-up (0.35 mm in the audit) and does not grow."""
    import alloy

    monkeypatch.setenv(S.STICK_ENV, "1")
    sc = alloy.Scene(dt=0.01)
    sc.add_box("plate", size=(0.5, 0.5, 0.05), pos=(0, 0, 0.3), kinematic=True, friction=0.6)
    sc.add_cloth("sheet", size=(0.1, 0.1), pos=(0, 0, 0.3275), resolution=12, mass=0.02, friction=0.6)
    sim = alloy.build(sc, device="cpu")
    sim.run(0.3)
    sim.drive("plate", velocity=(0.05, 0, 0), duration=0.5)
    tr = sim.run(0.5, record=["sheet", "plate"])
    lag = np.asarray(tr["plate.pos"])[:, 0, 0] - np.asarray(tr["sheet.points"])[:, 0, :, 0].mean(-1)
    lag = lag - lag[0]
    assert abs(lag[-1]) < 1.0e-3 and abs(lag[-1] - lag[len(lag) // 2]) < 2.5e-4, lag[-5:]
