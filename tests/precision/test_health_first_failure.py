# SPDX-License-Identifier: Apache-2.0
"""First-failure snapshot (coordinator ruling 2026-11-04 05:40Z): an INVALID world's record keeps the values of the
check that first made it INVALID beside the latest values (CPU).

T09's world 2 showed ``max_pair_pen_m`` 0 next to a real 256 um PAIR_PEN_FAIL: after the latch, the world is held at
its last good pose and every later check reads clean.

Rig: a 2 mm blade box and two particles of radius 0.5 mm per world, two worlds (as test_health_cut_front).
- fail then hold, contact: world 0's particle sits 0.8 mm deep (PEN_FAIL), then it is moved out. The latest values read
  clean; the snapshot keeps 0.8 mm, the particle, the blade shape, the flags, the check and its sim time.
- fail then hold, pair: a GapMonitor-style inbox reports pair 2 at 300 um with PAIR_PEN_FAIL, then a neutral inbox. The
  latest pair penetration reads 0; the snapshot keeps 300 um and pair 2.
- never fails: world 1's snapshot stays empty (flags 0, NaN / -1) and its latest values are as before.
"""

from __future__ import annotations

from alloy._vendor import newton
import numpy as np
import warp as wp

from alloy.precision.health import HealthMonitor
from alloy.precision.types import CheckPolicy, HealthFlags

R, HALF_T, INSIDE, DT = 5e-4, 1e-3, 3e-4, 1e-3
POLICY = CheckPolicy(level="cheap", every=1, full_every=1)
PEN_FAIL, PAIR_FAIL = int(HealthFlags.PEN_FAIL), int(HealthFlags.PAIR_PEN_FAIL)


def _rig():
    """World w: blade at the origin; particle 0 inside the +y face by INSIDE (world 0 only), particle 1 well outside."""
    b = newton.ModelBuilder()
    blade = b.add_body(xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()), mass=0.1)
    b.add_shape_box(blade, hx=0.02, hy=HALF_T, hz=0.02)
    b.add_particle(pos=(0.0, HALF_T - INSIDE, 0.1), vel=(0.0, 0.0, 0.0), mass=1e-3, radius=R)
    b.add_particle(pos=(0.0, 0.05, 0.1), vel=(0.0, 0.0, 0.0), mass=1e-3, radius=R)
    s = newton.ModelBuilder()
    s.replicate(b, 2)
    model = s.finalize(device="cpu")
    pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.01)
    return model, pipeline


def _first(model, w: int) -> int:
    return int(np.flatnonzero(model.particle_world.numpy() == w)[0])


def _state(model, inside_worlds: tuple[int, ...]):
    """A state with the first particle of each world in ``inside_worlds`` 0.8 mm deep in the blade, else 5 cm out."""
    state = model.state()
    q = state.particle_q.numpy()
    for w in range(2):
        if w not in inside_worlds:
            q[_first(model, w), 1] = 0.05
    state.particle_q.assign(q)
    return state


def _check(monitor, pipeline, step: int, state) -> None:
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    monitor.update(step, state, state, contacts, dt=DT)


def test_contact_failure_is_kept_after_the_world_reads_clean() -> None:
    """The particle sits 0.8 mm into the 2 mm blade, past half the thinner part (0.5 mm): since GAP-E1 (rulings
    2026-11-10 00:08Z, 05:27Z; WP5 reviews) that is PEN_FAIL once it has lasted the 0.1 s dwell, not at the first
    check. It is held there past the dwell, then at a clean pose: the first-failure snapshot keeps the failing check."""
    model, pipeline = _rig()
    monitor = HealthMonitor(model, POLICY)
    hold = int(round(0.1 / DT)) + 3                        # past the 0.1 s dwell, float32 sums included
    for k in range(hold):
        _check(monitor, pipeline, k, _state(model, (0,)))
    _check(monitor, pipeline, hold, _state(model, ()))     # held at a clean pose
    _check(monitor, pipeline, hold + 1, _state(model, ()))
    report = monitor.poll()
    assert report.invalid_worlds == (0,)
    assert np.isnan(report.max_pen_m[0]) or report.max_pen_m[0] < 1e-6, "latest value should read clean"
    f = report.first_failure.world(0)
    assert f is not None and "PEN_FAIL" in f["flags"]
    assert f["step"] == int(report.first_bad_step[0]) and hold - 5 <= f["step"] < hold
    np.testing.assert_allclose(f["pen_m"], INSIDE + R, rtol=1e-4)
    assert f["particle"] == _first(model, 0)
    blade_shape = int(np.flatnonzero(model.shape_world.numpy() == 0)[0])
    assert (f["shape_a"], f["shape_b"]) == (blade_shape, -1)
    np.testing.assert_allclose(f["time_s"], (f["step"] + 1) * DT, rtol=1e-6)    # the failing check's sim time
    text = report.failure_text(0)
    assert "PEN_FAIL" in text and f"particle {_first(model, 0)}" in text


def test_world_that_never_fails_keeps_an_empty_snapshot() -> None:
    model, pipeline = _rig()
    monitor = HealthMonitor(model, POLICY)
    _check(monitor, pipeline, 0, _state(model, (0,)))
    report = monitor.poll()
    assert 1 not in report.invalid_worlds and report.first_failure.world(1) is None
    ff = report.first_failure
    assert int(ff.flags[1]) == 0 and np.isnan(ff.pen_m[1]) and np.isnan(ff.time_s[1])
    assert (int(ff.shape_a[1]), int(ff.particle[1]), int(ff.pair_id[1]), int(ff.body[1])) == (-1, -1, -1, -1)
    assert int(report.first_bad_step[1]) == -1 and report.failure_text(1) == "never INVALID"
    assert report.max_pen_m[1] == 0.0 or np.isnan(report.max_pen_m[1])


def _inbox(monitor, w: int, pen: float, pair: int, flags: int) -> None:
    h = monitor.health.numpy()
    h["max_pair_pen_m"][w], h["max_pair_id"][w], h["flags"][w] = pen, pair, flags
    monitor.health.assign(h)


def test_pair_failure_is_kept_after_the_pair_reads_zero() -> None:
    """T09's world 2: PAIR_PEN_FAIL at 300 um from the GapMonitor, then the held pose reads 0."""
    model, _ = _rig()
    monitor = HealthMonitor(model, POLICY)
    state = _state(model, ())
    _inbox(monitor, 0, 3.0e-4, 2, PAIR_FAIL)
    monitor.update(0, state, state, None, dt=DT)
    for step in (1, 2):                                     # neutral inbox: the held pose
        monitor.update(step, state, state, None, dt=DT)
    report = monitor.poll()
    assert report.invalid_worlds == (0,)
    assert report.max_pair_pen_m[0] == 0.0
    f = report.first_failure.world(0)
    assert "PAIR_PEN_FAIL" in f["flags"] and f["pair_id"] == 2
    np.testing.assert_allclose(f["pair_pen_m"], 3.0e-4, rtol=1e-6)
    assert "pair 2 pen 0.0003 m" in report.failure_text(0)


def test_snapshot_without_dt_has_no_time() -> None:
    model, pipeline = _rig()
    monitor = HealthMonitor(model, POLICY)
    state = _state(model, (0,))
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    monitor.update(0, state, state, contacts)
    f = monitor.poll().first_failure.world(0)
    assert f is not None and np.isnan(f["time_s"])
