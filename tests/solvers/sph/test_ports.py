# SPDX-License-Identifier: Apache-2.0
"""SPH ports (ports.py): absorb, emit and retire between steps, the held reservoir, ledgers, determinism, reset."""

from __future__ import annotations

from types import SimpleNamespace

from alloy._vendor import newton
import numpy as np
import pytest
import warp as wp

from alloy.solvers.sph.build import add_fluid_block, add_held_spares, add_spares
from alloy.solvers.sph.config import PHASE_HELD, PHASE_LIQUID, PHASE_RETIRED, DFSPHConfig, register_custom_attributes
from alloy.solvers.sph.ports import port_for, root_model
from alloy.solvers.sph.solver import SolverDFSPH

R = 2.0e-3
ACTIVE = int(newton.ParticleFlags.ACTIVE)


class _Health:
    def __init__(self, worlds: int):
        self.z = wp.zeros(worlds, dtype=wp.float64, device="cpu")
        self.w = np.zeros(worlds)

    def state_arrays(self):
        return {"health.energy_z_ref": self.z}

    def add_external_work(self, w_ext):
        self.w += w_ext.numpy()


def _scene(worlds: int = 2, held: int = 6, spares: int = 0):
    cfg = DFSPHConfig(particle_radius=R, world_bounds=((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)))
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    register_custom_attributes(b)
    b.add_ground_plane()
    for _ in range(worlds):
        b.begin_world()
        add_fluid_block(b, cfg, (0.0, 0.0, 0.0), (3, 3, 2))
        add_held_spares(b, cfg, held, park=(0.0, 0.0, -0.5))
        add_spares(b, cfg, spares, park=(0.0, 0.0, -0.6))
        b.end_world()
    m = b.finalize(device="cpu")
    s = SolverDFSPH(m, cfg)
    return m, s, m.state()


def _mask(m, ids):
    a = np.zeros(m.particle_count, np.int32)
    a[list(ids)] = 1
    return wp.array(a, dtype=wp.int32, device="cpu")


def _requests(worlds: int, slots: int, n: list[int]):
    pos = np.zeros((worlds, slots, 3), np.float32)
    pos[:, :, 2] = 0.05 + 0.004 * np.arange(slots)[None, :]
    vel = np.zeros((worlds, slots, 3), np.float32)
    vel[:, :, 0] = 0.3
    return (wp.array(pos, dtype=wp.vec3, device="cpu"), wp.array(vel, dtype=wp.vec3, device="cpu"),
            wp.array(np.asarray(n, np.int32), dtype=wp.int32, device="cpu"))


def test_absorb_then_emit_round_trips_through_the_reservoir():
    m, s, st = _scene()
    port = s.port
    assert port_for(m) is port and port.reservoir == 12 and port.counts()["held"].tolist() == [6, 6]
    assert np.isclose(port.particle_volume, float(m.particle_mass.numpy()[0]) / 1000.0)
    w1 = 24                                                    # world 1: 18 liquid + 6 held, ids 24..47
    port.absorb(st, _mask(m, [w1 + 2, w1 + 5]))
    ph, fl = s.layout.phase.numpy(), m.particle_flags.numpy()
    assert port.taken.numpy().tolist() == [0, 2]
    assert ph[w1 + 2] == PHASE_HELD and fl[w1 + 2] & ACTIVE == 0 and port.counts()["held"].tolist() == [6, 8]
    pos, vel, n = _requests(2, 4, [0, 3])
    port.emit(st, pos, vel, n)
    assert port.granted.numpy().tolist() == [0, 3]
    ph, fl, q = s.layout.phase.numpy(), m.particle_flags.numpy(), st.particle_q.numpy()
    out = [w1 + 5, w1 + 2, w1 + 18]                            # last absorbed first, then the build reservoir in order
    assert all(ph[i] == PHASE_LIQUID and fl[i] & ACTIVE for i in out)
    assert np.allclose(q[out][:, 2], [0.05, 0.054, 0.058]) and np.allclose(st.particle_qd.numpy()[out][:, 0], 0.3)
    led = s.mass_ledger()
    mp = float(m.particle_mass.numpy()[0])
    assert np.array_equal(led["active"] + led["out"], led["initial"] + led["in"])     # LIQ-1, exact
    assert np.allclose(led["out"] / led["scale"], [0.0, 2 * mp]) and np.allclose(led["in"] / led["scale"], [0.0, 3 * mp])
    assert port.counts()["absorbed"].tolist() == [0, 2] and port.counts()["emitted"].tolist() == [0, 3]


def test_emitters_never_take_held_particles_and_empty_reservoirs_grant_less():
    m, s, st = _scene(worlds=1, held=2, spares=3)
    pos, vel, n = _requests(1, 5, [5])
    s.port.emit(st, pos, vel, n)
    assert s.port.granted.numpy().tolist() == [2]               # 2 held: the 3 phase-0 spares are the emitters'
    ph = s.layout.phase.numpy()
    assert (ph[20:23] == 0).all()


def test_retire_is_final_and_energy_is_booked_both_ways():
    m, s, st = _scene(worlds=1)
    h = _Health(1)
    s.export_flux_energy(h)
    pos, vel, n = _requests(1, 2, [2])
    s.port.emit(st, pos, vel, n)
    s.export_flux_energy(h)
    e_in = h.w.copy()
    mp = float(m.particle_mass.numpy()[0])
    assert np.isclose(e_in[0], 2 * mp * 0.5 * 0.09 + mp * 9.81 * (0.05 + 0.054), rtol=1e-5)
    s.retire(st, _mask(m, [18, 19]))                           # the two just emitted, unchanged
    s.export_flux_energy(h)
    assert abs(h.w[0]) < 1e-9 * e_in[0] + 1e-15
    assert (s.layout.phase.numpy()[18:20] == PHASE_RETIRED).all() and s.port.counts()["retired"].tolist() == [2]
    s.port.emit(st, pos, vel, n)
    assert s.port.granted.numpy().tolist() == [2]               # the other held ones, never the retired


def test_one_world_does_not_see_another_worlds_calls():
    m, s, st = _scene()
    q0 = st.particle_q.numpy()[:24].copy()
    pos, vel, n = _requests(2, 3, [0, 3])
    s.port.emit(st, pos, vel, n)
    s.port.absorb(st, _mask(m, [24, 25]))
    assert np.array_equal(st.particle_q.numpy()[:24], q0)
    assert s.port.counts()["held"].tolist() == [6, 5]


def test_reset_restores_the_reservoir():
    m, s, st = _scene()
    pos, vel, n = _requests(2, 3, [3, 3])
    s.port.emit(st, pos, vel, n)
    s.reset(st, wp.array([False, True], dtype=wp.bool, device="cpu"))
    c = s.port.counts()
    assert c["held"].tolist() == [3, 6] and c["emitted"].tolist() == [3, 0]


def test_port_survives_steps_and_reaches_the_root_model():
    m, s, st = _scene()
    nxt = m.state()
    s.step(st, nxt, None, None, 1.0e-3)
    assert s.port.neighbor_count.numpy()[:18].max() > 0
    view = SimpleNamespace(parent=SimpleNamespace(parent=m))
    assert root_model(view) is m and port_for(view) is s.port


def test_port_calls_replay_in_a_cuda_graph(device):
    if not wp.get_device(device).is_cuda:
        pytest.skip("CUDA graphs")
    cfg = DFSPHConfig(particle_radius=R)
    b = newton.ModelBuilder()
    register_custom_attributes(b)
    b.begin_world()
    add_fluid_block(b, cfg, (0.0, 0.0, 0.0), (3, 3, 2))
    add_held_spares(b, cfg, 4)
    b.end_world()
    m = b.finalize(device=device)
    s, st = SolverDFSPH(m, cfg), m.state()
    pos = wp.zeros((1, 2), dtype=wp.vec3, device=device)
    vel = wp.zeros((1, 2), dtype=wp.vec3, device=device)
    n = wp.array([1], dtype=wp.int32, device=device)
    with wp.ScopedCapture(device=device) as cap:
        s.port.emit(st, pos, vel, n)
    wp.capture_launch(cap.graph)
    wp.capture_launch(cap.graph)
    assert s.port.counts()["emitted"].tolist() == [2] and s.port.counts()["held"].tolist() == [2]
