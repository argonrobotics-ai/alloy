# SPDX-License-Identifier: Apache-2.0
"""``task_t12_graph`` (alloy.integration.t12_graph): T12 on IntegratedSimulation for WP8's "alloy_loop graph" row.

CPU: the target keeps task_t12's actions, and its loop (graph off) runs task_t12's step on its start state: after 3
periods of a moving command, bodies match the eager DirectSim target (bodies moved 11 mm; 4e-13 apart measured).
GPU (``-m gpu``, 64 worlds): periods replay graphs with no host sync, and the graph run stays on the eager one.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.bully.targets import get_target
from alloy.integration import t12_graph as G

pytestmark = [pytest.mark.integration, pytest.mark.timeout(1200)]

PERIODS = 3


def _command(target, k: int, worlds: int) -> np.ndarray:
    """Tick ``k``'s legal command: each target moves from the initial one toward the top of its box at full rate."""
    a = target.actions
    cmd = np.minimum(a.initial + (k + 1) * a.max_delta_per_tick, a.high)
    return np.tile(cmd, worlds).astype(np.float32)


def _drive(sim, target, worlds: int, periods: int = PERIODS) -> np.ndarray:
    for k in range(periods):
        sim.control.mujoco.ctrl.assign(_command(target, k, worlds))
        sim.step_period()
    return sim.state.body_q.numpy()


def test_target_keeps_task_t12s_actions() -> None:
    base, tg = get_target("task_t12"), get_target(G.NAME)
    assert tg.name == G.NAME and tg.make_sim is G.make_sim
    for field in ("low", "high", "initial", "max_delta_per_tick"):
        assert np.array_equal(getattr(base.actions, field), getattr(tg.actions, field)), field
    assert (tg.action_kind, tg.action_index, tg.robot_bodies) == (base.action_kind, base.action_index,
                                                                  base.robot_bodies)


def test_loop_runs_task_t12s_step_and_start() -> None:
    tg = get_target("task_t12")
    eager, loop = tg.make_sim(1, "cpu"), G.make_sim(1, "cpu", graph="off")
    assert loop.clock.steps_per_period == eager.substeps == 32
    assert np.array_equal(loop.state.joint_q.numpy(), eager.state.joint_q.numpy())
    assert np.array_equal(loop.control.mujoco.ctrl.numpy(), eager.control.mujoco.ctrl.numpy())
    q_eager, q_loop = _drive(eager, tg, 1), _drive(loop, tg, 1)
    assert np.abs(q_eager - q_loop).max() <= 1e-5


@pytest.mark.gpu
def test_graph_periods_replay_on_cuda() -> None:
    if not wp.is_cuda_available():
        pytest.skip("needs CUDA")
    tg, worlds = get_target(G.NAME), 64
    graph, eager = G.make_sim(worlds, "cuda:0"), G.make_sim(worlds, "cuda:0", graph="off")
    q_graph, q_eager = _drive(graph, tg, worlds, 4), _drive(eager, tg, worlds, 4)
    assert graph.graph_stats()["mode"] == "graph" and graph.host_syncs == 0
    assert np.isfinite(q_graph).all()
    assert np.abs(q_graph[:, :3] - q_eager[:, :3]).max() <= 1e-4     # float-atomic ordering only (PAR-14 rows)
