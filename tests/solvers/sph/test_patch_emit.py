# SPDX-License-Identifier: Apache-2.0
"""Triggered one-shot patch emitters (SCR-DC5-5 item 1): add_patch_emitter(world_mask, body, local_points) and
trigger(pid, world_flags); N per world, body-attached, one shot until reset, restart- and snapshot-safe."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.core.errors import CapacityError
from alloy.solvers.sph.build import add_spares, wall_box
from alloy.solvers.sph.config import register_custom_attributes
from alloy.solvers.sph.patch import film_offsets
from alloy.solvers.sph.solver import SolverDFSPH
from alloy.testing.sph import default_config

COUNTS = [5, 9, 0]
HALF = (2.5e-3, 2.5e-3)  # 2 x 2 lattice points per layer at 2r = 2 mm
FACE = wp.transform((0.0, 0.0, -2.0e-3), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi))  # bottom face
BODY_Q = (0.01, 0.0, 0.03, 0.0, 0.0, math.sin(0.2), math.cos(0.2))  # 0.4 rad about z
BODY_QD = (0.1, 0.0, -0.05, 0.0, 0.0, 2.0)  # v_com, omega (world)


def _scene(spares: int = 12):
    cfg = default_config()
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    register_custom_attributes(b)
    plugs = []
    for _ in COUNTS:
        b.begin_world()
        wall_box(b, (0.0, 0.0, -0.002), (0.02, 0.02, 0.002))
        plug = b.add_body(xform=wp.transform(BODY_Q[:3], BODY_Q[3:]), label="plug")
        wall_box(b, (0.0, 0.0, 0.0), (4.0e-3, 4.0e-3, 2.0e-3), body=plug)
        plugs.append(plug)
        add_spares(b, cfg, spares)
        b.end_world()
    m = b.finalize()
    s = SolverDFSPH(m, cfg)
    a, c = m.state(), m.state()
    for st in (a, c):
        st.body_q.assign(np.tile(np.array(BODY_Q, np.float32), (len(COUNTS), 1)))
        st.body_qd.assign(np.tile(np.array(BODY_QD, np.float32), (len(COUNTS), 1)))
    points = [film_offsets(HALF, n, cfg.spacing, FACE) for n in COUNTS]
    return m, s, plugs, points, a, c


def _register(s, plugs, points):
    return s.add_patch_emitter(np.array(COUNTS) > 0, plugs, points)


def _active(m, s, w: int) -> np.ndarray:
    act = (m.particle_flags.numpy() & int(newton.ParticleFlags.ACTIVE)) != 0
    return np.nonzero(act & (s.layout.phase.numpy() == 1) & (m.particle_world.numpy() == w))[0]


def _step(s, a, c):
    s.step(a, c, None, None, 1.0e-3)
    return c, a


def _flags(*on, dtype=wp.int32):
    return wp.array(np.array(on, dtype=bool if dtype == wp.bool else np.int32), dtype=dtype)


def test_film_offsets_fill_centre_out_then_layers():
    o = film_offsets(HALF, 5, 2.0e-3)
    assert o.shape == (5, 3)
    assert np.allclose(np.abs(o[:4, :2]), 1.0e-3)  # one layer of 4, then the second layer starts
    assert np.allclose(o[:4, 2], 1.0e-3) and np.isclose(o[4, 2], 3.0e-3)
    assert np.allclose(o[4, :2], o[0, :2])
    f = film_offsets(HALF, 5, 2.0e-3, FACE)  # mapped into the body frame: below the bottom face
    assert np.all(f[:, 2] < -2.0e-3)


def test_trigger_emits_once_per_world_with_the_body():
    m, s, plugs, points, a, c = _scene()
    pid = _register(s, plugs, points)
    assert pid == 0
    mass = float(m.particle_mass.numpy()[0])
    a, c = _step(s, a, c)
    assert all(len(_active(m, s, w)) == 0 for w in range(3))  # not triggered
    s.trigger(pid, _flags(1, 0, 1))  # world 2 has N = 0
    a, c = _step(s, a, c)
    ids = _active(m, s, 0)
    assert len(ids) == COUNTS[0] and len(_active(m, s, 1)) == 0 and len(_active(m, s, 2)) == 0
    bq, bqd = a.body_q.numpy()[plugs[0]], a.body_qd.numpy()[plugs[0]]  # the plug's pose at the end of the step
    X = wp.transform(*bq)
    want = np.array([wp.transform_point(X, wp.vec3(*o)) for o in points[0]])
    q, v = a.particle_q.numpy()[ids], a.particle_qd.numpy()[ids]
    assert np.allclose(np.sort(q, axis=0), np.sort(want, axis=0), atol=1e-7)
    com = np.array(wp.transform_point(X, wp.vec3(*m.body_com.numpy()[plugs[0]])))
    assert np.allclose(v, bqd[:3] + np.cross(bqd[3:], q - com), atol=1e-6)  # moves with the plug
    s.trigger(pid, _flags(1, 0, 0))  # already fired: ignored
    for _ in range(2):
        a, c = _step(s, a, c)
    assert len(_active(m, s, 0)) == COUNTS[0]
    s.trigger(pid, _flags(True, True, True, dtype=wp.bool))  # bool flags work too
    a, c = _step(s, a, c)
    assert len(_active(m, s, 1)) == COUNTS[1]
    led = s.mass_ledger()
    assert np.array_equal(led["active"] + led["out"], led["initial"] + led["in"])
    assert np.allclose(s.report().mass_in, np.array(COUNTS) * mass, rtol=1e-6)


def test_reset_rearms_restart_restores_and_snapshots_hold_the_state():
    m, s, plugs, points, a, c = _scene()
    pid = _register(s, plugs, points)
    s.trigger(pid, _flags(1, 1, 0))
    arrays, worlds = s.alloy_state_arrays(a), s.alloy_state_worlds()
    assert {"sph.patch_pending0", "sph.patch_fired0"} <= set(arrays) and "sph.patch_fired0" in worlds
    assert arrays["sph.patch_pending0"].numpy().tolist() == [1, 1]  # armed, not yet emitted
    a, c = _step(s, a, c)
    assert arrays["sph.patch_fired0"].numpy().tolist() == [1, 1]
    s.reset(a, wp.array([True, False, False], dtype=wp.bool))
    assert len(_active(m, s, 0)) == 0 and len(_active(m, s, 1)) == COUNTS[1]
    s.trigger(pid, _flags(1, 1, 0))  # world 0 re-armed, world 1 still fired
    a, c = _step(s, a, c)
    assert len(_active(m, s, 0)) == COUNTS[0] and len(_active(m, s, 1)) == COUNTS[1]
    s.reset(a)
    s.trigger(pid, _flags(1, 0, 0))
    s.coupling_notify_input_state_update(a, 0)  # exchange start: save (armed)
    a, c = _step(s, a, c)
    assert len(_active(m, s, 0)) == COUNTS[0]
    s.coupling_notify_input_state_update(a, 0, iteration_restart=True)  # restart pass: armed again, not emitted
    assert len(_active(m, s, 0)) == 0 and np.all(s.report().mass_in == 0.0)
    a, c = _step(s, a, c)
    assert len(_active(m, s, 0)) == COUNTS[0]


def test_patch_capacity_validation_and_box_emitters_share_spares():
    m, s, plugs, points, a, c = _scene(spares=8)
    with pytest.raises(CapacityError):
        _register(s, plugs, points)  # world 1 needs 9
    with pytest.raises(ValueError):
        s.add_patch_emitter(np.ones(2, bool), plugs, points)  # mask must cover every world
    m, s, plugs, points, a, c = _scene(spares=12)
    e = s.add_emitter(0, wp.transform((0.0, 0.0, 0.01), wp.quat_identity()), wp.vec3(1e-3, 1e-3, 1e-4), rate=1e-7,
                      velocity=wp.vec3(0.0, 0.0, -0.1))
    _register(s, plugs, points)
    box_pool = s.emitters.table.pool[e]
    assert len(box_pool) == 12 - COUNTS[0] and not set(box_pool) & set(s.emitters.patches.reserved(0))
    with pytest.raises(ValueError):
        s.trigger(5, _flags(1, 1, 1))
