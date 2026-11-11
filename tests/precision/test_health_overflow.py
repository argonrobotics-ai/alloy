# SPDX-License-Identifier: Apache-2.0
"""SCR-SC0-1 addendum (DC6, 2026-11-03): MuJoCo Warp contact overflow must leave the world INVALID.

- MuJoCo Warp's per-world ``Data.overflow`` NARROWPHASE bit (contacts dropped at nconmax) gives CONTACT_OVERFLOW and
  INVALID in that world.
- INVALID is sticky in the record without ``latch``: a caller that polls once at the end, as DC6's rig does, reads
  INVALID after an overflow that cleared later (before the fix it read OK; only ``first_bad_step`` was kept).
- MuJoCo Warp's line-search limit (LS_ITERATIONS) reads as UNCONVERGED (DEGRADED), like its iteration limit.
"""

from __future__ import annotations

import sys
from pathlib import Path

from alloy._vendor import newton
import numpy as np
import warp as wp

from alloy.precision.health import HealthMonitor
from alloy.precision.types import HealthFlags, Severity

sys.path.insert(0, str(Path(__file__).parent))
from test_health_pen_measured import POLICY, _boxes  # noqa: E402


def test_mujoco_narrowphase_overflow_is_invalid() -> None:
    model, s0 = _boxes("cpu")
    s1, control = model.state(), model.control()
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=True, nconmax=1)  # a box on the ground needs 4
    monitor = HealthMonitor(model, POLICY, solvers=[solver])
    solver.step(s0, s1, control, None, 1.0e-3)
    monitor.update(0, s0, s1, None)
    assert (solver.mjw_data.overflow.numpy() & 8).all()  # NARROWPHASE in both worlds
    report = monitor.poll()
    assert report.invalid_worlds == (0, 1)
    assert all(int(f) & int(HealthFlags.CONTACT_OVERFLOW) for f in report.flags)


def test_a_solver_mujoco_subclass_gets_the_same_overflow_check() -> None:
    """WP8 2026-11-06: the type-name check missed subclasses (rk4's SyncedMuJoCo, the MuJoCo C route's
    MjcCoreSolver), so their Data.overflow was never read."""

    class Sub(newton.solvers.SolverMuJoCo):
        pass

    model, s0 = _boxes("cpu")
    s1, control = model.state(), model.control()
    solver = Sub(model, use_mujoco_contacts=True, nconmax=1)
    monitor = HealthMonitor(model, POLICY, solvers=[solver])
    solver.step(s0, s1, control, None, 1.0e-3)
    monitor.update(0, s0, s1, None)
    assert monitor.poll().invalid_worlds == (0, 1)


def test_invalid_stays_without_latch() -> None:
    model, s0 = _boxes("cpu")
    flag = wp.zeros(2, dtype=wp.int32)
    monitor = HealthMonitor(model, POLICY, capacity_flags=[flag])
    flag.assign(np.array([0, 1], dtype=np.int32))       # world 1 overflows at step 0 ...
    monitor.update(0, s0, s0, None)
    flag.zero_()                                         # ... and is clean from step 1 on
    for step in range(1, 4):
        monitor.update(step, s0, s0, None)
    report = monitor.poll()
    assert report.invalid_worlds == (1,)
    assert int(report.flags[1]) & int(HealthFlags.CONTACT_OVERFLOW)
    assert int(report.first_bad_step[1]) == 0 and int(report.severity[0]) == int(Severity.OK)


def test_linesearch_limit_is_unconverged() -> None:
    model, s0 = _boxes("cpu")
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=True)
    monitor = HealthMonitor(model, POLICY, solvers=[solver])
    solver.mjw_data.overflow.assign(np.array([0, 1024], dtype=np.int32))  # LS_ITERATIONS in world 1
    monitor.update(0, s0, s0, None)
    report = monitor.poll()
    assert int(report.flags[1]) & int(HealthFlags.UNCONVERGED) and int(report.severity[1]) == int(Severity.DEGRADED)
    assert not int(report.flags[0]) & int(HealthFlags.UNCONVERGED)
