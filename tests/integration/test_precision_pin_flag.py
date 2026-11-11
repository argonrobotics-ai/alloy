# SPDX-License-Identifier: Apache-2.0
"""SCR-JOB2-1 item 2, end to end: a pinned MuJoCo entry whose stability estimate asks for a smaller step than one
step per exchange raises ``HealthFlags.RUNG_PINNED`` (WP5, DEGRADED) through the health monitor (CPU).

dc5_like (MuJoCo plug on a jack with a precision pair, VBD pad; 2 worlds): the plug is pinned at one step per
exchange (target 1 ms, exchange 0.625 ms). WP4's MuJoCo estimate is ``min(0.4 tau_eff, 0.5 margin / |v|max)``: the
resting plug gets 8 ms, while world 1's plug, launched upward at 1 m/s, gets about 0.25 ms (margin = the 0.5 mm shape
gap), below ``dt_x / safety``. ``RungPinWatch`` flags world 1 after the first period and the health monitor raises
the bit on that world's checks in the next one; world 0 stays clear.
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy.coupling.scheduler  # noqa: F401 - imported before the empty registry, as in a full run (see run())
from alloy.core.registry import isolated_registry
from alloy.coupling.estimators import register_builtin_estimators
from alloy.integration import build_simulation
from alloy.precision.types import HealthFlags, Severity
from alloy.testing import integration as I

pytestmark = [pytest.mark.compile_heavy, pytest.mark.integration, pytest.mark.timeout(600)]

FAST, LAUNCH_MPS = 1, 1.0
CMD = "PYTHONPATH=. python -m pytest tests/integration/test_precision_pin_flag.py -q"


@pytest.fixture(scope="module")
def run():
    with isolated_registry(empty=True):
        from alloy.scene.builtin_plugins import register_builtin_solvers  # noqa: PLC0415

        register_builtin_solvers()
        register_builtin_estimators()   # alloy.coupling.register_builtins runs once per process, into the registry
                                        # current at the Scheduler's first import, not into this empty one
        built = I.dc5_like(2, "cpu")
        body = I.launch_body(built.model, "plug", FAST, LAUNCH_MPS)
        sim = build_simulation(built, "standard", graph="off")
        z0 = sim.state.body_q.numpy()[:, 2].copy()
        sim.step_period()
        first = sim.parts.rung_pin.flagged.copy(), sim.report()["health"]
        sim.step_period()
        return sim, body, z0, first


def test_rung_pinned_reaches_the_health_monitor(run):
    sim, body, z0, (flagged_1, health_1) = run
    rep = sim.report()
    pin, health, coupling = sim.parts.rung_pin, rep["health"], rep["coupling"]
    stable = np.asarray(coupling.notes["stable_dt"]["mujoco"], dtype=np.float64)
    dt_x, safety = float(coupling.dt_x), float(pin.clock.safety)
    bit = int(HealthFlags.RUNG_PINNED)
    flags = [int(f) for f in np.asarray(health.flags)]
    severity = [int(s) for s in np.asarray(health.severity)]
    rise = sim.state.body_q.numpy()[body, 2] - z0[body]
    passed = (pin.flagged.tolist() == [False, True] and bool(flags[FAST] & bit) and not flags[0] & bit
              and severity[FAST] == int(Severity.DEGRADED))
    I.record("WP12-JOB2-1 flag", float(passed), "pass" if passed else "fail", CMD, dt_x=dt_x, safety=safety,
             stable_dt_mujoco=stable.tolist(), flagged=pin.flagged.tolist(), flags=flags, severity=severity,
             flagged_after_period_1=flagged_1.tolist(), launch_mps=LAUNCH_MPS, plug_rise_m=float(rise))
    assert rise > 0.01, rise                                     # the launch reached MuJoCo
    assert stable[0] > dt_x / safety > stable[FAST], (stable, dt_x, safety)
    assert flagged_1.tolist() == [False, True] and pin.flagged.tolist() == [False, True]
    assert not int(np.asarray(health_1.flags)[FAST]) & bit       # raised on the checks after the watch's update
    assert flags[FAST] & bit and not flags[0] & bit, flags
    assert severity[FAST] == int(Severity.DEGRADED) and severity[0] < int(Severity.DEGRADED), severity
    assert pin.health_source() == (pin.flags, bit)
