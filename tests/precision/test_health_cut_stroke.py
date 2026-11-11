# SPDX-License-Identifier: Apache-2.0
"""SCR-SC5-3 acceptance on WP10's cut-stroke rig (alloy.testing.topology.cut_stroke_rig, CPU): a blade descends
through a TET_CUT seam with real soft contacts, and ``HealthMonitor.update`` runs where the integrated loop runs it
(after the solve, before ``topology.apply``).
- With WP10's predicate (``set_cut_front``) the stroke reads 0 PEN_FAIL and reports ``cut_front_pen_m``, within the
  0.5 mm thin-part limit: since item 9 step 1 (coordinator 2026-11-06) BladeContact ramps a twin onto its face as the
  edge nears, so the cut front no longer crosses deep.
- The same stroke without the predicate also reads 0 PEN_FAIL.
- Twins whose bonds are already broken (phase 2) and plain particles, pinned inside the blade, still flag.
"""

from __future__ import annotations

import numpy as np

from alloy.precision.health import HealthMonitor
from alloy.precision.types import CheckPolicy, HealthFlags
from alloy.testing.topology import STROKE_SPEED, STROKE_TOP, clamp_particles, cut_stroke_rig, run_cut_stroke
from alloy.topology.types import BondKind

DT, POLICY = 1.0e-3, CheckPolicy(level="cheap", every=1, full_every=1)


def _descent(n: int) -> np.ndarray:
    return STROKE_TOP + 2.0e-3 - STROKE_SPEED * DT * np.arange(n)


def _monitored(rig, edge_z: np.ndarray, *, predicate: bool, x: float = 0.01):
    monitor = HealthMonitor(rig.sc.model, POLICY)
    if predicate:
        monitor.set_cut_front(rig.ctl.cut_front_blade, rig.ctl.blade_half_thickness)
    run_cut_stroke(rig, edge_z, x=x, dt=DT, observer=lambda k, s0, s1, c: monitor.update(k, s0, s1, c))
    return monitor.poll()


def _pen_fail(report) -> np.ndarray:
    return (report.flags.astype(np.int64) & int(HealthFlags.PEN_FAIL)) != 0


def test_cut_stroke_reads_no_pen_fail_and_reports_the_cut_front() -> None:
    rig = cut_stroke_rig("cpu", worlds=2)
    report = _monitored(rig, _descent(300), predicate=True)
    assert not _pen_fail(report).any(), report.flags
    assert int(np.count_nonzero(rig.ctl.cut_done.numpy())) > 0          # the stroke really cut bonds
    assert np.all(report.cut_front_pen_m > 0.0), report.cut_front_pen_m       # the cut front is still measured
    assert np.all(report.cut_front_pen_m <= 0.5e-3), report.cut_front_pen_m    # within the thin-part limit
    # no step needs the exemption now (cut_front_steps 0); the pinned phase-2 / plain-particle test below keeps the
    # predicate's own coverage


def test_the_same_stroke_without_the_predicate_reads_no_pen_fail_either() -> None:
    report = _monitored(cut_stroke_rig("cpu", worlds=2), _descent(300), predicate=False)
    assert not _pen_fail(report).any(), report.flags


def _pinned_in_the_blade(rig, x: float):
    """Pin the particles on the plane x and bring the blade over them in one step (WP10's _deep_contacts)."""
    m = rig.sc.model
    clamp_particles(m, np.flatnonzero(np.abs(m.particle_q.numpy()[:, 0] - x) < 1.0e-6))
    return _monitored(rig, np.array([2.0e-3]), predicate=True, x=x)


def test_phase_two_twins_and_plain_particles_in_the_blade_still_flag() -> None:
    rig = cut_stroke_rig("cpu")
    kind = rig.ctl.bonds.kind.numpy()
    rig.ctl.preset_broken(np.flatnonzero(kind == int(BondKind.TET_CUT)))   # seam twins, bonds broken (phase 2)
    assert _pen_fail(_pinned_in_the_blade(rig, 0.01)).all()
    assert _pen_fail(_pinned_in_the_blade(cut_stroke_rig("cpu"), 0.005)).all()   # x = 5 mm: no seam, plain particles
