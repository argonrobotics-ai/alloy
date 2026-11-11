# SPDX-License-Identifier: Apache-2.0
"""The hand-built coupled rigs give health their solvers, so a dropped MuJoCo constraint row is ROW_OVERFLOW (CPU).

Before (GAP-10A r8, 2026-11-10): the sewing ply-pinch rig built ``HealthMonitor(m, policy,
capacity_flags=sched.capacity_flags())`` with no ``solvers=``. Its MuJoCo entry (njmax 50) dropped constraint rows
3,171 times in one run, the pads sank to the table, and health read clean: health reads a solver's own overflow word
only from the solvers it is given. ``alloy.build`` always passed them (law:
tests/tasks/test_build_constraint_rows.py); the seam rig and the film pinch law did not.

Law, per rig: the monitor the rig builds holds the Scheduler's MuJoCo solver; with that solver's overflow word set to
"nefc > njmax" (what MuJoCo Warp writes when it drops rows) the next check is ROW_OVERFLOW and INVALID; a monitor
built the old way reads the same state clean, and says in the log that it reads no solver. No physics is stepped: the
rigs' own slow laws do that.
"""

from __future__ import annotations

import logging

import pytest

from alloy._vendor.mujoco_warp import OverflowType
from alloy.integration.stepper import entry_solvers, mujoco_solver
from alloy.precision.health import HealthMonitor
from alloy.precision.types import CheckPolicy, HealthFlags, Severity

ROWS = int(HealthFlags.ROW_OVERFLOW)


def _seam():
    from alloy.tasks import sew_seam

    run = sew_seam._start(sew_seam.SeamSpec(), None)
    return run.m, run.sched, run.monitor, run.s0


def _film():
    from alloy._vendor import newton
    from alloy.coupling import film_pinch as F
    from alloy.coupling.pinch_rigs import _cloth_scheduler, gripper_model

    law = F.FilmPinch()
    cloth = F._law_cloth(law)
    m = gripper_model(cloth=cloth, closed=True, film_top=F._film_top(law), fix_top=False)
    sched = _cloth_scheduler(m, False, F.law_iterations(law, False), radius=cloth.radius)
    s0 = m.state()
    newton.eval_fk(m, m.joint_q, m.joint_qd, s0)
    return m, sched, F.law_monitor(m, sched), s0


def _check(monitor: HealthMonitor, state) -> tuple[int, int]:
    monitor.update(0, state, state, None, dt=1.0e-3)
    report = monitor.poll()
    return int(report.flags[0]), int(report.severity[0])


@pytest.mark.parametrize("rig", [_seam, _film])
def test_a_rigs_monitor_reads_its_mujoco_entrys_dropped_rows(rig, caplog) -> None:
    m, sched, monitor, state = rig()
    mj = mujoco_solver(entry_solvers(sched))
    assert mj is not None and any(s is mj for s in monitor._solvers)
    flags, severity = _check(monitor, state)
    assert not flags & ROWS and severity != int(Severity.INVALID)          # nothing dropped yet
    mj.mjw_data.overflow.fill_(int(OverflowType.NEFC))                     # MuJoCo Warp: nefc > njmax, rows dropped
    flags, severity = _check(monitor, state)
    assert flags & ROWS and severity == int(Severity.INVALID)
    with caplog.at_level(logging.WARNING, logger="alloy.precision"):
        blind = HealthMonitor(m, CheckPolicy(level="full", every=1, full_every=1),
                              capacity_flags=sched.capacity_flags())       # the rigs' monitor before the fix
    assert "no solvers given" in caplog.text
    flags, severity = _check(blind, state)
    assert not flags & ROWS and severity != int(Severity.INVALID)          # the gap: the same word, unread
