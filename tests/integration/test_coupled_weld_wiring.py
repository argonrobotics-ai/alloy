# SPDX-License-Identifier: Apache-2.0
"""SCR-DC5-21 (WP12 part): ``wire`` hands the coupled stepper to WP10's ``TopologyController`` (``coupled=``), so a
weld on a MuJoCo entry under the Scheduler breaks at its force (CPU).

Under a coupled stepper the MuJoCo entry runs on its own ModelView, whose equality rows are copies; without the
stepper the controller's breaks never reached MuJoCo (DC5's weld held forever; WP10 now logs "coupled entry view").
Here dc5_like (1 world, MuJoCo plug + VBD pad under the Scheduler) welds the plug to the world where it
starts and launches it upward at 1 m/s, so the weld pulls it back. A weak weld breaks: the bond goes off, the MuJoCo
entry's own equality row goes off and the plug flies on (most of its free-flight rise). An unbreakable weld pulls
it back: MuJoCo's default soft weld (time constant 0.02 s) still lets it overshoot a few mm, a fraction of that rise.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import alloy.coupling.scheduler  # noqa: F401 - imported before the empty registry, as in a full run
from alloy.core.registry import isolated_registry
from alloy.coupling.estimators import register_builtin_estimators
from alloy.integration import build_simulation
from alloy.testing import integration as I

pytestmark = [pytest.mark.compile_heavy, pytest.mark.integration, pytest.mark.timeout(600)]

WEAK_N, LAUNCH_MPS, PERIODS = 0.05, 1.0, 3       # the weld's pull on the launched 4.2 g plug is well above 0.05 N
CONTROL_DT, G = 0.01, 9.81                        # dc5_like's control period [s]
T = PERIODS * CONTROL_DT
FREE_RISE_M = LAUNCH_MPS * T - 0.5 * G * T * T    # no weld: 25.6 mm
CMD = "PYTHONPATH=. python -m pytest tests/integration/test_coupled_weld_wiring.py -q"


def _run(weld_n: float) -> dict:
    with isolated_registry(empty=True):
        from alloy.scene.builtin_plugins import register_builtin_solvers  # noqa: PLC0415

        register_builtin_solvers()
        register_builtin_estimators()
        built = I.dc5_like(1, "cpu", weld_n=weld_n)
        body = I.launch_body(built.model, "plug", 0, LAUNCH_MPS)
        sim = build_simulation(built, "standard", graph="off")
        z0 = float(sim.state.body_q.numpy()[body, 2])
        for _ in range(PERIODS):
            sim.step_period()
    ctl = sim.parts.topology
    return {"stepper": type(sim.stepper).__name__, "intact": bool(ctl.bonds.intact.numpy()[0]),
            "entry_eq": bool(ctl.hooks.mujoco[0].model.mujoco.equality_constraint_enabled.numpy()[0]),
            "rise_m": float(sim.state.body_q.numpy()[body, 2]) - z0}


@pytest.fixture(scope="module")
def runs() -> tuple[dict, dict]:
    return _run(WEAK_N), _run(math.inf)


def test_weld_on_the_coupled_mujoco_entry_breaks_through_wiring(runs):
    weak, held = runs
    passed = (not weak["intact"] and not weak["entry_eq"] and held["intact"] and held["entry_eq"]
              and weak["rise_m"] > 0.8 * FREE_RISE_M and abs(held["rise_m"]) < 0.4 * FREE_RISE_M)
    I.record("WP12-DC5-21 weld", float(passed), "pass" if passed else "fail", CMD, weak_n=WEAK_N,
             launch_mps=LAUNCH_MPS, periods=PERIODS, free_rise_m=FREE_RISE_M, weak=weak, held=held)
    assert weak["stepper"] == held["stepper"] == "Scheduler", (weak, held)
    assert not weak["intact"] and not weak["entry_eq"], weak              # broke, and MuJoCo's own row went off
    assert held["intact"] and held["entry_eq"], held
    assert weak["rise_m"] > 0.8 * FREE_RISE_M, (weak, FREE_RISE_M)        # released early: the plug flies on
    assert abs(held["rise_m"]) < 0.4 * FREE_RISE_M, (held, FREE_RISE_M)   # pulled back (soft weld overshoot only)
    assert np.isfinite([weak["rise_m"], held["rise_m"]]).all()
