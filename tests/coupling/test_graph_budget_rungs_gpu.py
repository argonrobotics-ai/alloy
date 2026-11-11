# SPDX-License-Identifier: Apache-2.0
"""Graph-budget rungs on CUDA, end to end (speed-now S2, opt-in ``ClockPolicy.graph_budget_rungs``).

``IntegratedSimulation`` counts a new key's one-exchange graph before its first launch and sends a key over its node
budget to eager periods (``_fits``). With the flag on it also tells the Scheduler, whose clock caps the entry with the
most substeps one rung lower (``ClockController.cap_rung``). WP4's P3 chain (MuJoCo -> VBD -> SPH) with a swinging arm,
the contact gate on, and a one-node budget, so every key is over it: with the flag, each new cap makes the next period's
exchange finer with the capped entries within their caps; without it, no cap. (The speed itself is the VM rows'
business: [internal note] scr/WP10.md, speed-now S2.)"""

from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.integration, pytest.mark.timeout(1800)]

DEV = os.environ.get("ALLOY_DEVICE", "cuda:0")
PERIODS = 5
ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0.1 0 0">
  <joint name="shoulder" type="hinge" pos="-0.1 0 0" axis="0 1 0" range="-2 2"/>
  <geom type="capsule" fromto="-0.1 0 0 0.1 0 0" size="0.02" mass="1"/>
  </body></worldbody>
  <actuator><position joint="shoulder" kp="20" kv="0.2"/></actuator></mujoco>"""


def _run(flag: bool) -> list[tuple[float, dict[str, int], dict[str, int]]]:
    import warp as wp

    import alloy
    import alloy.coupling.compose as compose
    from alloy.coupling.contact_gate import gate_for

    if not wp.is_cuda_available():
        pytest.skip("needs CUDA")
    if not compose.proxy_dag_supported():
        pytest.skip("Newton without N-entry proxy coupling (pin < 11-07)")
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_open_box("tank", inner=(0.05, 0.05, 0.05), wall=0.006, pos=(0, 0, 0))
    scene.add_liquid("water", size=(0.05, 0.05, 0.016), pos=(0, 0, 0.008))
    scene.add_cloth("sheet", size=(0.04, 0.04), pos=(0, 0, 0.3), resolution=6, mass=0.003)
    scene.add_robot("arm", mjcf=ARM, pos=(0.6, 0, 0.3))
    sim = alloy.build(scene, device=DEV)
    sim.set_targets({"shoulder": 1.5})
    eng, st = sim.engine, sim.engine.stepper
    clock = dataclasses.replace(st._alloy_policy.clock, contact_gate=True, graph_budget_rungs=flag)
    st._alloy_policy = dataclasses.replace(st._alloy_policy, clock=clock)
    st._gate = gate_for(st)
    eng.graph_node_budget = 1                             # every key's one exchange is over it
    seen = []
    for _ in range(PERIODS):
        eng.step_period()
        rep = eng._last_coupling
        seen.append((float(rep.dt_x), dict(rep.substeps), dict(st._clock.rung_cap)))
    return seen


def test_over_budget_keys_cap_a_rung_with_the_flag_only() -> None:
    seen = _run(True)
    new = [p for p in range(PERIODS - 1) if seen[p][2] != (seen[p - 1][2] if p else {})]
    assert new, seen                                      # a cap was learned (every key is over the budget)
    for p in new:                                         # the next period: finer, each capped entry within its cap
        dt, sub, caps = seen[p + 1][0], seen[p + 1][1], seen[p][2]
        assert dt < seen[p][0], seen
        assert all(sub[name] <= cap for name, cap in caps.items()), seen
    off = _run(False)
    assert all(caps == {} for _, _, caps in off), off
    assert np.isfinite([dt for dt, _, _ in off]).all()
