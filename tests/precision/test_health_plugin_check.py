# SPDX-License-Identifier: Apache-2.0
"""Plug-in depth checks in the HealthMonitor (coordinator ruling 2026-11-04 18:25Z, SCR-SC11-1 item 4; CPU).

Rig: two worlds of one free box (no contacts: the plug-in depth is the only penetration signal).
- default off is bitwise: a monitor with a registered check whose depth stays 0 keeps the same record, field for field,
  as one with nothing registered; the plug-in fields read -1 / 0 and the snapshot stays empty;
- a registered check trips: world 0 at 2 mm against a 1 mm limit goes PEN_FAIL / INVALID, world 1 stays OK; the
  first-failure snapshot keeps the check and its depth after the depth goes back to 0; ``failure_text`` names it;
- two checks: the first registered one over its limit is the one named; a depth under its limit is the latest
  ``plugin_depth_m``;
- a solver that exposes ``health_depth_checks()`` registers its checks when the monitor is built;
- bad registrations raise.
"""

from __future__ import annotations

from alloy._vendor import newton
import numpy as np
import pytest
import warp as wp

from alloy.precision.health import HealthMonitor
from alloy.precision.types import CheckPolicy, DepthCheck, HealthFlags, Severity

POLICY = CheckPolicy(level="cheap", every=1, full_every=1)
PEN_FAIL = int(HealthFlags.PEN_FAIL)


def _model() -> newton.Model:
    b = newton.ModelBuilder()
    body = b.add_body(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()), mass=1.0)
    b.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    s = newton.ModelBuilder()
    s.replicate(b, 2)
    return s.finalize(device="cpu")


def _depth(values) -> wp.array:
    return wp.array(np.asarray(values, np.float32), dtype=wp.float32, device="cpu")


def _run(monitor: HealthMonitor, model: newton.Model, steps: int = 2, setter=None) -> np.ndarray:
    state = model.state()
    for step in range(steps):
        if setter is not None:
            setter(step)
        monitor.update(step, state, state, None, dt=1e-3)
    return monitor.record.numpy()


def test_default_off_is_bitwise() -> None:
    model = _model()
    plain = _run(HealthMonitor(model, POLICY), model)
    quiet = HealthMonitor(model, POLICY)
    quiet.register_depth_check("untracked_depth", _depth([0.0, np.nan]), 1e-3)
    with_check = _run(quiet, model)
    assert plain.tobytes() == with_check.tobytes()
    assert plain["plugin_check"].tolist() == [-1, -1] and plain["plugin_depth_m"].tolist() == [0.0, 0.0]
    ff = quiet.poll().first_failure
    assert ff.world(0) is None and ff.plugin_names == ("untracked_depth",)


def test_registered_check_trips_and_is_named() -> None:
    model = _model()
    monitor = HealthMonitor(model, POLICY)
    depth = _depth([2e-3, 0.0])
    assert monitor.register_depth_check("untracked_depth", depth, 1e-3) == 0
    _run(monitor, model, steps=3, setter=lambda step: depth.assign(np.float32([2e-3 if step == 0 else 0.0, 0.0])))
    report = monitor.poll()
    assert report.invalid_worlds == (0,) and int(report.severity[1]) == int(Severity.OK)
    assert int(report.flags[0]) & PEN_FAIL and not int(report.flags[1]) & PEN_FAIL
    rec = monitor.record.numpy()
    assert rec["plugin_check"].tolist() == [-1, -1]            # latest: depth back to 0
    f = report.first_failure.world(0)
    assert f["plugin_check"] == 0 and "PEN_FAIL" in f["flags"] and f["step"] == 0
    np.testing.assert_allclose(f["plugin_depth_m"], 2e-3, rtol=1e-6)
    np.testing.assert_allclose(f["time_s"], 1e-3)
    assert "plug-in untracked_depth depth 0.002 m" in report.failure_text(0)
    assert report.first_failure.world(1) is None


def test_first_registered_check_over_its_limit_is_named() -> None:
    model = _model()
    monitor = HealthMonitor(model, POLICY)
    monitor.register_depth_check("thread", _depth([0.5e-3, 0.5e-3]), 1e-3)        # under its limit
    monitor.register_depth_check("needle", _depth([3e-3, 0.0]), 2e-3)            # over it in world 0
    _run(monitor, model, steps=1)
    rec = monitor.record.numpy()
    assert rec["plugin_check"].tolist() == [1, -1]
    np.testing.assert_allclose(rec["plugin_depth_m"], [3e-3, 0.5e-3], rtol=1e-6)
    assert "plug-in needle depth 0.003 m" in monitor.poll().failure_text(0)


class _PluginSolver(newton.solvers.SolverSemiImplicit):
    def __init__(self, model: newton.Model, depth: wp.array) -> None:
        super().__init__(model)
        self.depth = depth

    def health_depth_checks(self):
        return [DepthCheck("untracked_depth", self.depth, 1e-3)]


def test_solver_checks_are_discovered() -> None:
    model = _model()
    solver = _PluginSolver(model, _depth([0.0, 5e-3]))
    monitor = HealthMonitor(model, POLICY, solvers=[solver])
    _run(monitor, model, steps=1)
    report = monitor.poll()
    assert report.invalid_worlds == (1,)
    assert "plug-in untracked_depth" in report.failure_text(1)


def test_bad_registrations_raise() -> None:
    model = _model()
    monitor = HealthMonitor(model, POLICY)
    with pytest.raises(ValueError, match="length 2"):
        monitor.register_depth_check("a", _depth([0.0, 0.0, 0.0]), 1e-3)
    with pytest.raises(ValueError, match="float32"):
        monitor.register_depth_check("a", wp.zeros(2, dtype=wp.float64, device="cpu"), 1e-3)
    with pytest.raises(ValueError, match="limit_m"):
        monitor.register_depth_check("a", _depth([0.0, 0.0]), 0.0)
    with pytest.raises(ValueError, match="non-empty"):
        monitor.register_depth_check("", _depth([0.0, 0.0]), 1e-3)
    monitor.register_depth_check("a", _depth([0.0, 0.0]), 1e-3)
    with pytest.raises(ValueError, match="already registered"):
        monitor.register_depth_check("a", _depth([0.0, 0.0]), 1e-3)
