# SPDX-License-Identifier: Apache-2.0
"""The coupler's report reads the R2-E3 band once per attempt (contact_band.read_band): sim.report(), sim.check() and
alloy.check run it again between periods, and a repeat must not clear the band (it used to: the next period then ran
with no band bound, its clock no longer the one an unwatched run takes)."""

from __future__ import annotations

import numpy as np
import pytest

import alloy


def _scene() -> alloy.Sim:
    """A 5 kg box dropped 5 cm onto a 4 mm sheet on a table: it lands at ~1 m/s in period 9, and the band binds the
    next periods' clock (test_fine_rerun's band law)."""
    side = (5.0 / 8000.0) ** (1.0 / 3.0)
    sc = alloy.Scene(worlds=1, dt=0.01)
    sc.add_ground()
    sc.add_box("table", size=(0.4, 0.4, 0.1), pos=(0, 0, 0.05), static=True, friction=0.5)
    sc.add_cloth("cloth", size=(0.2, 0.2), pos=(0, 0, 0.102), resolution=(9, 9), mass=0.03, thickness=0.004,
                 self_contact=False)
    sc.add_box("box", size=(side,) * 3, pos=(0, 0, 0.104 + 0.001 + side / 2 + 0.05), mass=5.0)
    return alloy.build(sc, device="cpu")


def _run(watch: bool, periods: int = 12) -> list[tuple]:
    sim = _scene()
    try:
        out = []
        for _ in range(periods):
            sim.step(1)
            if watch:
                sim.report()
                sim.report()
            st = sim.engine.stepper
            out.append((sim.engine._last_coupling.dt_x, float(st._band_bound),
                        np.asarray(sim.state(["box"])["box.pos"]).tobytes()))
        return out
    finally:
        sim.close()


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(1800)
def test_reports_between_periods_change_nothing() -> None:
    quiet, watched = _run(False), _run(True)
    assert any(b < 1e-3 for _, b, _ in quiet)                        # the band binds in this run
    assert quiet == watched


def test_read_band_reads_once_per_attempt() -> None:
    """Host only: a second read for the same attempt returns the first bound and leaves the maxima to nobody."""
    from types import SimpleNamespace

    from alloy.coupling import contact_band as CB

    reads = []
    band = SimpleNamespace(end_period=lambda travel: reads.append(travel) or 0.002)
    rerun = CB.BandRerun()
    sched = SimpleNamespace(_band=band, _band_rerun=rerun, _band_bound=float("inf"), _dt_x=1e-3, _control_dt=0.01,
                            _alloy_policy=SimpleNamespace(clock=SimpleNamespace(band_travel=1 / 6, j_max=10)))
    rerun.begin_attempt()
    sched._band_bound = CB.read_band(sched)
    assert sched._band_bound == 0.002 and CB.read_band(sched) == 0.002 and len(reads) == 1
    rerun.begin_attempt()
    CB.read_band(sched)
    assert len(reads) == 2
