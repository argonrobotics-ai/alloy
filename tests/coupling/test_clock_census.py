# SPDX-License-Identifier: Apache-2.0
"""Gap #10b P0 (alloy/coupling/clock_census.py): what a per-world exchange clock would save, counted per attempt and
over the run; nothing it reads changes the step (bitwise)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from alloy.coupling.clock_census import ClockCensusMixin


class _Base:
    def __init__(self, dt_x: float, ws: np.ndarray) -> None:
        self.dt_x, self.ws = dt_x, ws

    def report(self):
        return SimpleNamespace(dt_x=self.dt_x, notes={}, watch_state=self.ws)

    def on_rollback(self, world_mask=None) -> None:
        pass

    def choose_clock(self, control_dt, last):
        return self.dt_x, {}


class _Coupler(ClockCensusMixin, _Base):
    """Three worlds behind a contact gate: world 0 armed at a 1/64 bound, world 1 armed at 1/8, world 2 unarmed."""

    def __init__(self, dt_x: float = 0.01 / 64, j: int = 6, ws=(0, 0, 0)) -> None:
        super().__init__(dt_x, np.asarray(ws))
        self._watchdog = SimpleNamespace(world_count=3)
        self._alloy_policy = SimpleNamespace(clock=SimpleNamespace(dt_x_max=math.inf, j_max=10, safety=1.0,
                                                                   omega_dt_max=1.0, damping_bound=False))
        self._clock = SimpleNamespace(j=j, j_floor=0)
        self._band_bound, self._control_dt = math.inf, 0.01
        ends = SimpleNamespace(world=np.array([0, 1, 1, 2]), bound_s=np.array([0.01 / 64, 0.01 / 8, 0.01 / 4, 1e-6]),
                               fixed_bound_s=math.inf)
        self._gate = SimpleNamespace(ends=ends, armed_host=np.array([True, True, True, False]))
        self.shared = 0.01 / 64                                  # the clock's shared bound (min over armed)

    def _interface_bound(self) -> float:
        return self.shared

    def attempt(self):
        self.choose_clock(0.01, None)
        return self.report().notes["census"]


def test_each_world_is_counted_at_its_own_armed_bound() -> None:
    c = _Coupler()
    census = c.attempt()
    assert census["j_world"] == [6, 3, 0] and census["drift"] is False                      # world 2's 1 us proxy is not armed
    assert census["world_exchanges"] == 3 * 64 and census["per_world_exchanges"] == 64 + 8 + 1
    assert census["rerun_kept_exchanges"] == 0


def test_a_rerun_counts_the_kept_worlds_and_keeps_its_own_halvings() -> None:
    """A per-world rollback of world 0: the re-run (one onset halving: j 7 against the clock's 6) re-ran worlds 1 and 2
    for nothing; their own rungs carry the halving too (P0 counts no per-world re-run)."""
    c = _Coupler(dt_x=0.01 / 128)
    c.on_rollback(np.array([1, 0, 0], dtype=np.int32))
    census = c.attempt()
    assert census["j_world"] == [7, 4, 1] and census["rerun_kept_exchanges"] == 2 * 128
    c.report()                                                                # a second read: counted once
    assert c.attempt()["rerun_kept_exchanges"] == 0                          # the next attempt: no re-run
    whole = _Coupler()                                                        # a whole re-run for world 0
    whole._last_report = SimpleNamespace(watch_state=np.array([2, 0, 0]))     # the escalating attempt's
    whole.on_rollback(None)
    assert whole.attempt()["rerun_kept_exchanges"] == 2 * 64
    assert c.clock_census() == {"world_exchanges": 2 * 3 * 128, "per_world_exchanges": 2 * (128 + 16 + 2),
                                "rerun_kept_exchanges": 2 * 128, "attempts": 2}


def test_a_census_that_no_longer_reads_the_clocks_bound_says_so(caplog) -> None:
    """WP10's review: the minimum over the census's per-world bounds is the clock's shared bound; a clock change that
    breaks that (here the clock binds at 1/128 while no armed proxy does) is flagged, and logged once."""
    c = _Coupler()
    c.shared = 0.01 / 128
    with caplog.at_level("WARNING", logger="alloy.coupling"):
        assert c.attempt()["drift"] is True and c.attempt()["drift"] is True
    assert sum("clock census" in r.message for r in caplog.records) == 1


def test_without_a_gate_each_world_takes_its_pairs_bound_at_its_lightest_proxy() -> None:
    c = _Coupler(dt_x=0.01 / 128, j=7)
    c._gate = None
    c._alloy_pairs = [SimpleNamespace(source="a", destination="b")]
    c._pair_k_c = lambda pair: 1.0e4
    c._pair_c_d = lambda pair: None
    c._sets = []
    c._census_mass = {("a", "b"): np.array([1.0e-4, 1.0e-2, math.inf])}   # world 2: nothing touches
    c.shared = math.sqrt(1.0e-4 / 1.0e4)
    census = c.attempt()
    j_light = int(math.ceil(math.log2(0.01 / math.sqrt(1.0e-4 / 1.0e4))))
    assert census["j_world"] == [j_light, max(j_light - 3, 0), 0]



def _grip(worlds: int):
    """WP9's welded grip (tests/coupling/test_match_reset.py): 2 identical worlds."""
    from tests.coupling.test_match_reset import _grip as grip

    return grip(worlds)


@pytest.mark.timeout(900)
def test_identical_worlds_save_nothing_and_the_census_changes_no_step(monkeypatch) -> None:
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "mjc")
    runs = {}
    for on in (True, False):
        if not on:
            monkeypatch.setattr(ClockCensusMixin, "_census", lambda self, report: None)
        sim = _grip(2)
        try:
            sim.set_targets({"roll": 0.05})
            sim.step(3)
            runs[on] = sim.state(["cable"])["cable.points"].copy()
            report = sim.engine.stepper._last_report
            if on:
                census = report.notes["census"]
                total = sim.engine.stepper.clock_census()
                assert census["j_world"] == [sim.engine.clock.j] * 2 and census["drift"] is False
                assert total["per_world_exchanges"] == total["world_exchanges"] > 0
            else:
                assert "census" not in report.notes
        finally:
            sim.close()
    assert np.array_equal(runs[True], runs[False])
