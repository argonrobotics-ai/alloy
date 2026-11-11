# SPDX-License-Identifier: Apache-2.0
"""DirectSim reports WP5 health (WP13, 2026-11-04): flags reach EpisodeRunner, a world a solver guard rescued reads
DEGRADED (FALLBACK_USED), and a reset starts the reset worlds' records and guard counters afresh."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton

from alloy.bully.direct import DirectSim
from alloy.bully.replay import EpisodeRunner
from alloy.bully.types import BullyLimits
from alloy.precision.types import HealthFlags, HealthReport, Severity
from alloy.testing.bully import seeded_fake_target

NONFINITE = int(HealthFlags.NONFINITE_Q | HealthFlags.NONFINITE_QD)


def _flags(sim: DirectSim) -> list[int]:
    rep = sim.report()["health"]
    assert isinstance(rep, HealthReport)
    return rep.flags.astype(np.int64).tolist()


def test_health_flags_reach_the_report_and_reset_per_world():
    sim = seeded_fake_target().make_sim(3, "cpu")
    sim.step_period()
    assert _flags(sim) == [0, 0, 0]
    qd = sim.state.body_qd.numpy()
    qd[1, 0] = np.nan
    sim.state.body_qd.assign(qd)
    sim.step_period()
    flags = _flags(sim)
    assert flags[1] & NONFINITE and flags[0] == flags[2] == 0
    assert int(sim.report()["health"].severity[1]) == int(Severity.INVALID)
    sim.reset(np.array([0, 1, 0], dtype=np.int32))
    assert _flags(sim) == [0, 0, 0]


def test_episode_runner_sees_direct_sim_health():
    target = seeded_fake_target()
    trace = np.zeros((2, 5, 2))
    res = EpisodeRunner(target, 2, "cpu", BullyLimits()).run(trace)
    assert res.health.severity.shape == (2,) and int(res.health.severity.max()) == int(Severity.OK)


def test_health_can_be_turned_off():
    sim = seeded_fake_target().make_sim(1, "cpu")
    off = DirectSim(sim.model, sim.solver, substeps=4, collide=False, health=False)
    off.step_period()
    assert off.report()["health"] is None


def _pile(worlds: int) -> newton.Model:
    b = newton.ModelBuilder()
    for i in range(2):
        body = b.add_body(xform=wp.transform(wp.vec3(0.03 * i, 0.0, 0.05 + 0.099 * i), wp.quat_identity()))
        b.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    s = newton.ModelBuilder()
    s.replicate(b, worlds)
    s.add_ground_plane()
    return s.finalize(device="cpu")


def test_guard_rescued_world_reads_degraded_until_reset():
    mjw_chol = pytest.importorskip("alloy.precision.mjw_chol")
    if not mjw_chol.install():
        pytest.skip("MuJoCo Warp guard not installed at this pin")
    model = _pile(2)
    solver = newton.solvers.SolverMuJoCo(model, njmax=200, nconmax=64, solver="newton")
    sim = DirectSim(model, solver, control_dt=0.004, substeps=2, collide=False)
    clamps = mjw_chol.clamp_counts(solver.mjw_data)
    assert clamps is not None
    clamps.assign(np.array([0, 2], dtype=np.int32))  # world 1 ran two clamped factorizations
    sim.step_period()
    rep = sim.report()["health"]
    fallback = rep.flags.astype(np.int64) & int(HealthFlags.FALLBACK_USED)
    assert fallback.tolist() == [0, int(HealthFlags.FALLBACK_USED)]
    assert int(rep.severity[1]) == int(Severity.OK)            # FALLBACK_USED is a notice (ruling 2026-11-10)
    sim.reset(np.array([0, 1], dtype=np.int32))
    assert clamps.numpy().tolist() == [0, 0]
    sim.step_period()
    assert int(sim.report()["health"].flags[1]) & int(HealthFlags.FALLBACK_USED) == 0


def test_sb7_lists_guard_rescued_episodes():
    from dataclasses import replace

    from alloy.bully.__main__ import _fallback_entries
    from alloy.bully.types import EpisodeResult

    health = HealthReport.ok(3)
    flags = np.array([0, int(HealthFlags.FALLBACK_USED), int(HealthFlags.FALLBACK_USED)], dtype=np.uint32)
    res = EpisodeResult(np.array([0, 0, 1], dtype=np.int32), np.full(3, -1), {}, replace(health, flags=flags),
                        np.zeros(3))
    assert _fallback_entries(res, 7, 3) == [{"seed": 7, "world": 1, "severity": 0},
                                            {"seed": 7, "world": 2, "severity": 1}]
    assert _fallback_entries(res, 7, 2) == [{"seed": 7, "world": 1, "severity": 0}]
