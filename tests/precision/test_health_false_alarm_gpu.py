# SPDX-License-Identifier: Apache-2.0
"""PRC-10 partial (SPEC §5.2): 0 INVALID on clean scenes at >= 1e5 world-steps each. The speed-ladder rungs are WP8's
(not merged); these are WP5's GPU repeat scenes with the standard checks every step, full checks every 10."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.core.snapshot import WorldSnapshot
from alloy.precision.health import HealthMonitor
from alloy.precision.types import CheckPolicy, HealthFlags, Severity
from alloy.testing.precision import det_sim, write_row

pytestmark = [pytest.mark.gpu, pytest.mark.slow]
CMD = "scripts/gpu_run.sh WP5 'tests/precision/test_health_false_alarm_gpu.py' --vm <gpu-vm>"


@pytest.mark.parametrize("kind", ["xpbd", "steel", "mujoco"])
def test_prc10_partial(kind: str) -> None:
    worlds, boxes = (1024, 27) if kind != "mujoco" else (1024, 8)
    sim = det_sim(kind, worlds, det=kind != "mujoco", boxes=boxes)
    monitor = HealthMonitor(sim.model, CheckPolicy(level="full", every=1, full_every=10), solvers=[sim.solver])
    snap = WorldSnapshot(sim.model, sim.state)
    snap.save(sim.state)
    steps, worst, seen = 0, 0, 0
    for period in range(15):
        for _ in range(sim.clock.steps_per_period):
            snap.save(sim.state)
            sim.state.clear_forces()
            sim.pipeline.collide(sim.state, sim.contacts)
            sim.solver.step(sim.state, sim._next, sim.control, sim.contacts, sim.clock.dt)
            monitor.update(steps, sim.state, sim._next, sim.contacts)
            monitor.latch(sim._next, snap)
            sim.state, sim._next = sim._next, sim.state
            steps += 1
        rep = monitor.poll()
        worst = max(worst, int(rep.severity.max()))
        seen |= int(np.bitwise_or.reduce(rep.flags))
    world_steps = steps * worlds
    flags = "|".join(f.name for f in HealthFlags if seen & int(f)) or "none"
    write_row("PRC-10 partial", f"{kind} box_drop({boxes}) x {worlds} worlds: {world_steps:.3g} world-steps, "
              f"{len(rep.invalid_worlds)} INVALID, worst severity {Severity(worst).name}, flags seen {flags}",
              "pass" if not rep.invalid_worlds else "fail", CMD, solver=kind, world_steps=world_steps)
    assert world_steps >= 1e5 and rep.invalid_worlds == ()
