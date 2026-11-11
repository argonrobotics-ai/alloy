# SPDX-License-Identifier: Apache-2.0
"""SCR-WP11-F32: a coupled VBD cloth falls at g on every substep rung, at any height (coordinator 2026-11-08 21:47Z).

Float32 positions round each step's displacement to whole ulps; VBD took the velocity as (x - x_prev) / dt, so every
velocity change came out in quanta of ulp(x) / dt. A coupled cloth's fine substeps (a 2 kg ball beside it: 0.3125 ms
exchanges, rungs of 1-8 substeps) then fell at 9.77, 19.5 (2 g) and 0 m/s^2 at z = 2 m by rung (WP4's triage,
[internal note] notes/wp4-vbd-f32). VBD now takes the inertia target and the velocity as displacements in small numbers
and carries each position's sub-ulp remainder (particle_vbd_kernels.update_velocity).

Laws: (fast) a 5 x 5 sheet stepped by SolverVBDAlloy alone at each rung's substep, 312, 156, 78 and 39 us, at
z = 2, 0.75 and 0.05 m: its velocity gains g dt per step and its mean height accelerates at g, within 0.1 % (on the
CPU; ``gpu``: the same on CUDA, where Newton 1.8 compiles the VBD particle kernels with fast math by default); (slow,
through the API) the sheet beside a far 2 kg ball, its entry's rung forced to 1, 2, 4 and 8 in turn for 3 periods
each, at z = 2 m and 0.75 m: its mean height accelerates at g within 0.1 % on every rung.
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy

G = 9.81
SUBSTEPS = (3.125e-4, 1.5625e-4, 7.8125e-5, 3.90625e-5)   # s: rungs 1, 2, 4, 8 at a 0.3125 ms exchange
DT = 0.01
RUNGS = (1, 2, 4, 8)
PERIODS = 3


def _solo_fall(z0: float, dt: float, steps: int = 64, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """(mean height, mean vertical velocity) per step of a free 5 x 5 sheet on SolverVBDAlloy alone."""
    import warp as wp

    from alloy._vendor import newton
    from alloy.solvers.vbd import SolverVBDAlloy

    b = newton.ModelBuilder(gravity=(0.0, 0.0, -G))
    b.add_cloth_grid(pos=wp.vec3(-0.15, -0.15, z0), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0), dim_x=4,
                     dim_y=4, cell_x=0.075, cell_y=0.075, mass=0.002, tri_ke=1.0e3, tri_ka=1.0e3, tri_kd=0.0,
                     edge_ke=1.0e-4)
    b.color()
    model = b.finalize(device=device)
    solver = SolverVBDAlloy(model, iterations=10)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    z, v = [], []
    for _ in range(steps + 1):
        z.append(float(np.mean(s0.particle_q.numpy()[:, 2], dtype=np.float64)))
        v.append(float(np.mean(s0.particle_qd.numpy()[:, 2], dtype=np.float64)))
        solver.step(s0, s1, ctrl, None, dt)
        s0, s1 = s1, s0
    return np.asarray(z), np.asarray(v)


def _assert_falls_at_g(z0: float, dt: float, device: str) -> None:
    z, v = _solo_fall(z0, dt, device=device)
    t = dt * np.arange(len(z))
    assert (v[-1] - v[0]) / t[-1] == pytest.approx(-G, rel=1.0e-3), v[[0, 1, 2, -1]]
    assert 2.0 * np.polyfit(t, z, 2)[0] == pytest.approx(-G, rel=1.0e-3)


@pytest.mark.parametrize("z0", [2.0, 0.75, 0.05])
@pytest.mark.parametrize("dt", SUBSTEPS)
def test_vbd_alone_falls_at_g_at_every_substep(z0, dt):
    _assert_falls_at_g(z0, dt, "cpu")


@pytest.mark.gpu
@pytest.mark.parametrize("z0", [2.0, 0.05])
@pytest.mark.parametrize("dt", SUBSTEPS)
def test_vbd_alone_falls_at_g_at_every_substep_on_cuda(z0, dt):
    _assert_falls_at_g(z0, dt, "cuda:0")


def test_a_write_to_particle_q_that_skips_the_displacement_is_what_the_solve_sees():
    """WP0's review: the solve's inertia term reads x as particle_q plus its sub-ulp remainder, so a mid-step write
    to particle_q that bypasses the displacement (a plug-in, a coupling proxy) has no remainder (two-ulp guard) and
    the solve pulls from where particle_q is, as stock's (y - x) did, not from a stale reference + displacement."""
    import warp as wp

    from alloy._vendor import newton
    from alloy.solvers.vbd import SolverVBDAlloy

    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    b.add_cloth_grid(pos=wp.vec3(0.0, 0.0, 1.0), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0), dim_x=2,
                     dim_y=2, cell_x=0.05, cell_y=0.05, mass=0.001, tri_ke=1.0e-6, tri_ka=1.0e-6, tri_kd=0.0,
                     edge_ke=0.0)
    b.color()
    model = b.finalize(device="cpu")
    solver = SolverVBDAlloy(model, iterations=1)
    hit = []

    def bump(slv, colour, state, contacts):
        if not hit:                                    # once: the first colour's first particle moves 1 mm in x
            i = int(slv.model.particle_color_groups[colour].numpy()[0])
            q = state.particle_q.numpy()
            q[i, 0] += 1.0e-3
            state.particle_q.assign(q)
            hit.append(i)

    solver.add_colour_callback(bump, external=False, when="before_accumulate")
    s0, s1 = model.state(), model.state()
    x0 = s0.particle_q.numpy().copy()
    solver.step(s0, s1, model.control(), None, 1.0e-3)
    (i,) = hit
    assert s1.particle_q.numpy()[i, 0] - x0[i, 0] == pytest.approx(-1.0e-3, abs=1.0e-6)   # pulled back from pos


def _heights(z0: float) -> dict[int, list[float]]:
    """Mean sheet height at each period boundary, per forced rung (one sim, the rungs in turn)."""
    sc = alloy.Scene(dt=DT)
    sc.add_cloth("sheet", size=(0.3, 0.3), pos=(0.0, 0.0, z0), resolution=5, mass=0.05, self_contact=False)
    sc.add_sphere("ball", 0.04, pos=(1.0, 0.0, z0), mass=2.0, material="steel_carbon")
    sim = alloy.build(sc, device="cpu")
    st = sim.engine.stepper
    (name,) = [n for n in st._alloy_entries if n != "mujoco"]
    out = {}
    for rung in RUNGS:
        zs = []
        for _ in range(PERIODS + 1):
            zs.append(float(np.mean(sim.state(["sheet"])["sheet.points"][0, :, 2], dtype=np.float64)))
            st.set_substeps({name: rung})
            sim.step(1)
        out[rung] = zs
    assert sim.check()["ok"], sim.check()["problems"]
    return out


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("z0", [2.0, 0.75])
def test_a_coupled_cloth_falls_at_g_on_every_rung(z0):
    """~200 s per height on the CPU (the coupled exchange at 0.3125 ms)."""
    for rung, zs in _heights(z0).items():
        a = -np.diff(np.asarray(zs), 2) / DT**2       # per period, every step at this rung
        assert np.mean(a) == pytest.approx(G, rel=1.0e-3), (z0, rung, a.round(3).tolist())


def _sheet_solver(z0: float = 2.0):
    import warp as wp

    from alloy._vendor import newton
    from alloy.solvers.vbd import SolverVBDAlloy

    b = newton.ModelBuilder(gravity=(0.0, 0.0, -G))
    b.add_cloth_grid(pos=wp.vec3(-0.15, -0.15, z0), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0), dim_x=4,
                     dim_y=4, cell_x=0.075, cell_y=0.075, mass=0.002, tri_ke=1.0e3, tri_ka=1.0e3, tri_kd=0.0,
                     edge_ke=1.0e-4)
    b.color()
    model = b.finalize(device="cpu")
    return model, SolverVBDAlloy(model, iterations=10)


def test_a_state_snapshot_carries_the_remainder_and_reruns_bit_for_bit():
    """WP0's review: the remainder is State data (Model.state() allocates particle_q_lo), so a WorldSnapshot of the
    State alone (no solver arrays) restores it and a rerun repeats bit for bit; a reset zeroes it."""
    from alloy.core.snapshot import WorldSnapshot

    model, solver = _sheet_solver()
    s0, s1, ctrl = model.state(), model.state(), model.control()
    for _ in range(20):
        solver.step(s0, s1, ctrl, None, SUBSTEPS[-1])
        s0, s1 = s1, s0
    assert np.abs(s0.particle_q_lo.numpy()).max() > 0.0              # a falling sheet carries a remainder
    snap = WorldSnapshot(model, s0)
    assert "state:particle_q_lo" in {slot.name for slot in snap.slots}
    snap.save(s0)

    def run(a, b):
        for _ in range(10):
            solver.step(a, b, ctrl, None, SUBSTEPS[-1])
            a, b = b, a
        return a.particle_q.numpy().copy(), a.particle_qd.numpy().copy()

    first = run(s0, s1)
    snap.restore(s0)
    again = run(s0, s1)
    np.testing.assert_array_equal(first[0], again[0])
    np.testing.assert_array_equal(first[1], again[1])
    solver.reset(s0)
    assert not np.abs(s0.particle_q_lo.numpy()).any()


def test_the_coupled_exchange_carries_the_remainder_to_the_scene_state():
    """WP0's review: the coupler copies particle_q_lo with particle_q into the cloth entry and back into the scene
    state, so the scene state holds the entry's remainder after a period."""
    sc = alloy.Scene(dt=DT)
    sc.add_cloth("sheet", size=(0.3, 0.3), pos=(0.0, 0.0, 2.0), resolution=5, mass=0.05, self_contact=False)
    sc.add_sphere("ball", 0.04, pos=(1.0, 0.0, 2.0), mass=2.0, material="steel_carbon")
    sim = alloy.build(sc, device="cpu")
    sim.step(2)
    eng = sim.engine
    lo = eng.state.particle_q_lo.numpy()
    (name,) = [n for n in eng.stepper._alloy_entries if n != "mujoco"]
    entry = eng.stepper._entries[name]
    assert np.abs(lo).max() > 0.0
    assert np.abs(lo).max() <= 2.4e-7 * np.abs(eng.state.particle_q.numpy()).max()       # within two ulps
    local = getattr(entry, "state_1", None) or getattr(entry, "state_0", None)
    if local is not None and getattr(local, "particle_q_lo", None) is not None:
        assert np.abs(local.particle_q_lo.numpy()).max() > 0.0


def test_a_hand_built_state_without_the_remainder_steps_as_with_it_zeroed(caplog):
    """WP0's review: a State without particle_q_lo (hand-built, or from a copier that drops it) steps without
    crashing, warns once, and matches a run whose remainder is zeroed before every step."""
    import logging

    model, solver = _sheet_solver()
    a0, a1, ctrl = model.state(), model.state(), model.control()
    for s in (a0, a1):
        del s.particle_q_lo
    with caplog.at_level(logging.WARNING):
        for _ in range(8):
            solver.step(a0, a1, ctrl, None, SUBSTEPS[-1])
            a0, a1 = a1, a0
    assert sum("particle_q_lo" in r.getMessage() for r in caplog.records) == 1
    model2, solver2 = _sheet_solver()
    b0, b1 = model2.state(), model2.state()
    for _ in range(8):
        b0.particle_q_lo.zero_()
        solver2.step(b0, b1, ctrl, None, SUBSTEPS[-1])
        b0, b1 = b1, b0
    np.testing.assert_array_equal(a0.particle_q.numpy(), b0.particle_q.numpy())
    np.testing.assert_array_equal(a0.particle_qd.numpy(), b0.particle_qd.numpy())


def test_a_substepped_entry_reruns_from_a_reset_bit_for_bit():
    """The coupler steps a substepped entry from its state_1 after copying state_0 into it (_copy_same_view_state).
    The remainder rides that copy, so a rerun after sim.reset() repeats the first run bit for bit. Before, the first
    substep read the remainder state_1 kept from the last exchange (3.6e-7 m off here), and a rolled-back world's
    neighbours left their reference (tests/coupling/test_rollback_bitwise.py)."""
    sc = alloy.Scene(dt=DT)
    sc.add_cloth("sheet", size=(0.3, 0.3), pos=(0.0, 0.0, 2.0), resolution=5, mass=0.05, self_contact=False)
    sc.add_sphere("ball", 0.04, pos=(1.0, 0.0, 2.0), mass=2.0, material="steel_carbon")
    sim = alloy.build(sc, device="cpu")
    st = sim.engine.stepper
    (name,) = [n for n in st._alloy_entries if n != "mujoco"]

    def run():
        st.set_substeps({name: 2})
        sim.step(1)
        return sim.state(["sheet"])["sheet.points"].copy()

    first = run()
    sim.reset()
    np.testing.assert_array_equal(run(), first)
