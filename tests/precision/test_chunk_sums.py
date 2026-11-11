# SPDX-License-Identifier: Apache-2.0
"""Parallel fixed-order energy sums (mixed-speed lever #6, CPU): health's ``world_energy``, the coupler's energy and
its external work.

Each world's bodies-then-particles sequence is summed in chunks of ``CHUNK`` items (one thread each), then over its
chunks in order. Laws: the chunk layout; every item counted once (against a float64 numpy sum); a world of at most
``CHUNK`` items gives exactly the serial loop's bits (one chunk per world = the old order); a larger world differs by
rounding only; a world's value is bitwise the same with or without other worlds beside it; the coupler's max speed
and non-finite flag come through the chunks; external work (forces, particle forces, then joint forces) the same.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.coupling import energy as en
from alloy.precision import chunk_sums as cs
from alloy.precision import health_physics as hp
from alloy.precision.health_setup import energy_model

DEV = "cpu"
SMALL, LARGE = (3, 40, 1), (2, 300, 2)     # (bodies, particles, seed): 43 items = 1 chunk; 302 = 3 chunks


def test_chunk_layout_counts_from_each_world_start():
    world, lo, hi, start = cs.chunk_layout(np.array([0, 1, 128, 129, 300]), 128)
    assert start.tolist() == [0, 0, 1, 2, 4, 7]
    assert world.tolist() == [1, 2, 3, 3, 4, 4, 4]
    assert lo.tolist() == [0, 0, 0, 128, 0, 128, 256]
    assert hi.tolist() == [1, 128, 128, 129, 128, 256, 300]
    with pytest.raises(ValueError):
        cs.chunk_layout(np.array([3]), 0)


def _world(n_bodies: int, n_particles: int, seed: int) -> newton.ModelBuilder:
    rng = np.random.default_rng(seed)
    b = newton.ModelBuilder()
    for _ in range(n_bodies):
        body = b.add_body(xform=wp.transform(tuple(rng.uniform(-1.0, 1.0, 3)), wp.quat_identity()),
                          mass=float(rng.uniform(0.1, 2.0)))
        b.add_shape_box(body, hx=0.05, hy=0.03, hz=0.02)
    for _ in range(n_particles):
        b.add_particle(pos=tuple(rng.uniform(-1.0, 1.0, 3)), vel=tuple(rng.uniform(-2.0, 2.0, 3)),
                       mass=float(rng.uniform(1e-3, 1e-2)), radius=0.01)
    return b


def _scene(*worlds: tuple[int, int, int]) -> tuple[newton.Model, newton.State]:
    """A model of the given worlds and a state with seeded body velocities (linear and angular)."""
    top = newton.ModelBuilder()
    for nb, npart, seed in worlds:
        top.add_world(_world(nb, npart, seed))
    model = top.finalize(device=DEV)
    state = model.state()
    qd = np.array([np.random.default_rng(100 * s + j).uniform(-1.0, 1.0, 6) for nb, _, s in worlds for j in range(nb)])
    if len(qd):
        state.body_qd.assign(qd.astype(np.float32))
    return model, state


def _health(model: newton.Model, state: newton.State) -> np.ndarray:
    em, W = energy_model(model), int(model.world_count)
    z_ref, out = wp.zeros(W, dtype=wp.float64, device=DEV), wp.zeros(W, dtype=wp.float64, device=DEV)
    hp.launch_world_energy(em, state.body_q, state.body_qd, state.particle_q, state.particle_qd, state.joint_q,
                           z_ref, out, DEV)
    return out.numpy()


def _coupler(model: newton.Model, state: newton.State) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inp, W = en.energy_inputs(model), int(model.world_count)
    z_ref, energy = wp.zeros(W, dtype=wp.float64, device=DEV), wp.zeros(W, dtype=wp.float64, device=DEV)
    vmax, bad = wp.zeros(W, dtype=float, device=DEV), wp.zeros(W, dtype=wp.int32, device=DEV)
    en.launch_energy(inp, state, z_ref, energy, vmax, bad, DEV)
    return energy.numpy(), vmax.numpy(), bad.numpy()


def _numpy_energy(model: newton.Model, state: newton.State) -> np.ndarray:
    """KE + m g z per world in float64 (gravity -z, z_ref 0), every item once, any order."""
    g = 9.81
    out = np.zeros(int(model.world_count))
    q, qd = state.body_q.numpy().astype(np.float64), state.body_qd.numpy().astype(np.float64)
    mass, inertia, com = (a.numpy().astype(np.float64) for a in (model.body_mass, model.body_inertia, model.body_com))
    for b, w in enumerate(model.body_world.numpy()):
        rot = _rotation(q[b, 3:])
        om_body = rot.T @ qd[b, 3:]
        z = (rot @ com[b] + q[b, :3])[2]
        out[w] += 0.5 * mass[b] * qd[b, :3] @ qd[b, :3] + 0.5 * om_body @ inertia[b] @ om_body + mass[b] * g * z
    px, pv = state.particle_q.numpy().astype(np.float64), state.particle_qd.numpy().astype(np.float64)
    pm = model.particle_mass.numpy().astype(np.float64)
    for i, w in enumerate(model.particle_world.numpy()):
        out[w] += 0.5 * pm[i] * pv[i] @ pv[i] + pm[i] * g * px[i, 2]
    return out


def _rotation(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = q_xyzw
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


@pytest.mark.parametrize("which", ["health", "coupler"])
def test_every_item_counted_once(which):
    model, state = _scene(SMALL, LARGE)
    got = _health(model, state) if which == "health" else _coupler(model, state)[0]
    np.testing.assert_allclose(got, _numpy_energy(model, state), rtol=1e-5, atol=0)   # float32 terms in the kernels


@pytest.mark.parametrize("which", ["health", "coupler"])
def test_one_chunk_world_keeps_the_serial_bits_and_large_ones_differ_by_rounding(monkeypatch, which):
    model, state = _scene(SMALL, LARGE)
    run = (lambda: _health(model, state)) if which == "health" else (lambda: _coupler(model, state)[0])
    chunked = run()
    monkeypatch.setattr(cs, "CHUNK", 10**9)                  # one chunk per world: the serial loop's order
    serial = run()
    assert chunked[0] == serial[0]
    assert chunked[1] == pytest.approx(serial[1], rel=1e-13, abs=0)
    np.testing.assert_array_equal(run(), serial)             # and again: deterministic


@pytest.mark.parametrize("which", ["health", "coupler"])
def test_a_world_does_not_depend_on_its_neighbours(which):
    def energy(*worlds):
        model, state = _scene(*worlds)
        return _health(model, state) if which == "health" else _coupler(model, state)[0]

    both = energy(SMALL, LARGE)
    assert both[0] == energy(SMALL)[0] and both[1] == energy(LARGE)[0]


def test_coupler_speed_and_nonfinite_flag_pass_through_the_chunks(monkeypatch):
    model, state = _scene(SMALL, LARGE)
    e, vmax, bad = _coupler(model, state)
    v = np.concatenate([state.body_qd.numpy()[:, :3], state.particle_qd.numpy()])
    world = np.concatenate([model.body_world.numpy(), model.particle_world.numpy()])
    np.testing.assert_allclose(vmax, [np.linalg.norm(v[world == w], axis=1).max() for w in (0, 1)], rtol=1e-6)
    assert bad.tolist() == [0, 0]
    pv = state.particle_qd.numpy()
    pv[40 + 280] = (np.inf, 0.0, 0.0)                       # world 1's particle 280: its third chunk
    state.particle_qd.assign(pv)
    e2, _, bad2 = _coupler(model, state)
    assert bad2.tolist() == [0, 1] and e2[0] == e[0] and not np.isfinite(e2[1])


DT = 0.004


def _seeded(worlds, salt: int, per_world, dim: int) -> np.ndarray:
    """Random rows seeded by each world's own seed (so a world's values do not depend on its neighbours)."""
    rows = [np.random.default_rng(1000 * s + salt).uniform(-1.0, 1.0, (per_world(nb, npart), dim))
            for nb, npart, s in worlds]
    return np.concatenate(rows).astype(np.float32)


def _work(*worlds: tuple[int, int, int]) -> np.ndarray:
    """W_ext per world over one exchange: seeded forces and displacements; also returns the numpy reference."""
    model, s_in = _scene(*worlds)
    s_out = model.state()
    q_out = s_in.body_q.numpy()
    q_out[:, :3] += 0.01 * _seeded(worlds, 1, lambda nb, n: nb, 3)
    s_out.body_q.assign(q_out)
    s_out.body_qd.assign(_seeded(worlds, 2, lambda nb, n: nb, 6))
    s_out.particle_q.assign(s_in.particle_q.numpy() + 0.01 * _seeded(worlds, 3, lambda nb, n: n, 3))
    s_in.body_f.assign(_seeded(worlds, 4, lambda nb, n: nb, 6))
    s_in.particle_f.assign(_seeded(worlds, 5, lambda nb, n: n, 3))
    ndof = lambda nb, n: 6 * nb                                    # noqa: E731  free bodies: 6 dofs each
    jf, qd_in, qd_out = (wp.array(_seeded(worlds, k, ndof, 1)[:, 0], dtype=float, device=DEV) for k in (6, 7, 8))
    inp, ext = en.work_inputs(model), wp.zeros(int(model.world_count), dtype=wp.float64, device=DEV)
    en.launch_external_work(inp, DT, (s_in.body_f, s_in.body_q, s_out.body_q, s_out.body_qd),
                            (s_in.particle_f, s_in.particle_q, s_out.particle_q), (jf, qd_in, qd_out), ext, DEV)
    return ext.numpy(), _numpy_work(model, s_in, s_out, (jf, qd_in, qd_out))


def _numpy_work(model, s_in, s_out, joint) -> np.ndarray:
    out = np.zeros(int(model.world_count))
    f64 = lambda a: a.numpy().astype(np.float64)                   # noqa: E731
    com, f, q_in, q_out, qd_out = f64(model.body_com), f64(s_in.body_f), f64(s_in.body_q), f64(s_out.body_q), \
        f64(s_out.body_qd)
    for b, w in enumerate(model.body_world.numpy()):
        dx = _rotation(q_out[b, 3:]) @ com[b] + q_out[b, :3] - _rotation(q_in[b, 3:]) @ com[b] - q_in[b, :3]
        out[w] += f[b, :3] @ dx + DT * f[b, 3:] @ qd_out[b, 3:]
    pf, dx = f64(s_in.particle_f), f64(s_out.particle_q) - f64(s_in.particle_q)
    for i, w in enumerate(model.particle_world.numpy()):
        out[w] += pf[i] @ dx[i]
    jf, qd_in, qd_out = (f64(a) for a in joint)
    dof_start = model.joint_dof_world_start.numpy()[: out.size + 1]
    dof_world = np.repeat(np.arange(out.size), np.diff(dof_start))
    np.add.at(out, dof_world, DT * jf * 0.5 * (qd_in + qd_out))
    return out


def test_external_work_counts_every_item_once_and_keeps_one_chunk_bits(monkeypatch):
    got, ref = _work(SMALL, LARGE)
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=0)
    monkeypatch.setattr(cs, "CHUNK", 10**9)
    serial, _ = _work(SMALL, LARGE)
    assert got[0] == serial[0] and got[1] == pytest.approx(serial[1], rel=1e-13, abs=0)


def test_external_work_does_not_depend_on_neighbours():
    both, _ = _work(SMALL, LARGE)
    assert both[0] == _work(SMALL)[0][0] and both[1] == _work(LARGE)[0][0]
