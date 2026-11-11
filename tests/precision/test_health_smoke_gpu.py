# SPDX-License-Identifier: Apache-2.0
"""GPU merge smoke (SPEC §6.3): HealthMonitor kernels compile and latch on CUDA (PRC-9 NaN case, 64 worlds)."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.precision.types import CheckPolicy, HealthFlags, Severity
from alloy.testing.precision import HealthLoop, set_body, world_bodies
from alloy.testing.scenes import box_on_plane

pytestmark = [pytest.mark.gpu]


@pytest.mark.smoke
def test_health_latch_cuda() -> None:
    loop = HealthLoop(box_on_plane(64, device="cuda:0"), checks=CheckPolicy(level="full", full_every=1),
                      solver_kwargs={"deterministic": wp.DeterministicMode.RUN_TO_RUN})
    loop.step(3)
    b = int(world_bodies(loop.scene.model, 9)[0])

    def nan(step: int, state) -> None:
        q = state.body_q.numpy()[b].copy()
        q[1] = np.nan
        set_body(state.body_q, b, q)

    loop.step(1, inject=nan)
    rep = loop.monitor.poll()
    assert rep.invalid_worlds == (9,) and rep.flags[9] & int(HealthFlags.LATCHED)
    assert rep.severity[9] == Severity.INVALID


def _monitored_period(sim, monitor, snap, step0: int) -> None:
    for k in range(sim.clock.steps_per_period):
        snap.save(sim.state)  # the caller keeps the last good state (FakeSimulation does this every step)
        sim.solver.step(sim.state, sim._next, sim.control, None, sim.clock.dt)
        monitor.update(step0 + k, sim.state, sim._next, None)
        monitor.latch(sim._next, snap)
        sim.state, sim._next = sim._next, sim.state


@pytest.mark.parametrize("blowup", [False, True])
def test_health_in_cuda_graph_matches_eager(blowup: bool) -> None:
    """update + latch are capture-safe: a captured period replays to the same records and state as eager runs."""
    from alloy.core.snapshot import WorldSnapshot
    from alloy.precision.health import HealthMonitor
    from alloy.testing.core import FakeSolver
    from alloy.testing.precision import DetSim, box_drop

    def build():
        model = box_drop(256, boxes=4)
        cfg = FakeSolver.Config(body_radius=0.05, blowup_speed=0.5 if blowup else None, blowup_gain=3.0)
        sim = DetSim(model, FakeSolver(model, cfg))
        mon = HealthMonitor(model, CheckPolicy(level="full", every=1, full_every=1), solvers=[sim.solver])
        snap = WorldSnapshot(model, sim.state)
        snap.save(sim.state)
        mon.reset_reference(sim.state)
        return sim, mon, snap

    eager = build()
    for p in range(6):
        _monitored_period(*eager, 0)
    graph = build()
    with wp.ScopedCapture(device="cuda:0") as cap:
        _monitored_period(*graph, 0)
    for _ in range(6):
        wp.capture_launch(cap.graph)
    he, hg = eager[1].record.numpy(), graph[1].record.numpy()
    assert np.array_equal(he.view(np.uint8), hg.view(np.uint8))
    assert np.array_equal(eager[0].state.body_q.numpy().view(np.uint32), graph[0].state.body_q.numpy().view(np.uint32))
    if blowup:  # boxes pass 0.5 m/s in the second period, then x3 per step: SPEED_FAIL and the worlds latch
        assert (hg["flags"] & int(HealthFlags.LATCHED)).any()
