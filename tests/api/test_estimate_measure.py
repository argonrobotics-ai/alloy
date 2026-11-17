# SPDX-License-Identifier: Apache-2.0
"""``alloy.estimate(..., measure=True)`` times past the start of the run (the late ease-cpu report, 0.1.7).

It timed the first 0.2 s and scaled them: three YCB scans dropped into a bin cost 1.7-1.9x more per step once they
rested than while they fell, so it read the run about 2x fast (0.78-0.87x real time where the run gave 0.38-0.48x, on
a spot n2 and n2d), and a pour 3-6x (its liquid grows as it leaves the nozzle). Laws (CPU):

- With periods twice as dear from 0.4 s on (a fake clock), it gives the run's own wall time; a run shorter than its
  1 s window is timed whole. The 0.2 s it timed before gave half.
- On a real scene it keeps the estimate's keys and says how much it timed (``window_s``).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import alloy
from alloy.api import _estimate_measure as EM


class _FakeSim:
    """A sim whose control periods cost 1 ms of a fake clock until t = 0.4 s and 2 ms after (parts landing)."""

    dt = 0.01

    def __init__(self, clock: list[float]) -> None:
        self.clock, self.periods, self.physics_steps = clock, 0, 0

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self.clock[0] += 1e-3 if self.periods * self.dt < 0.4 - 1e-9 else 2e-3
            self.periods += 1
            self.physics_steps += 8

    def routes(self) -> dict:
        return {}

    def close(self) -> None:
        return None


def test_measure_times_past_the_start_of_the_run(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(EM, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    monkeypatch.setattr(EM, "_sync", lambda device: None)
    monkeypatch.setattr("alloy.api._build.build", lambda scene, worlds, device: _FakeSim(clock))
    est = EM.measured(None, 1, 5.0, "cpu")
    run = 40 * 1e-3 + 460 * 2e-3                    # the 500 periods of a 5 s run from t = 0
    assert est["wall_time_s"] == pytest.approx(run, rel=0.01)
    assert est["window_s"] == pytest.approx(1.0) and est["step_time_s"] == pytest.approx(2e-3 / 8)
    assert est["physics_steps_per_world"] == 4000 and "first 100 control periods" in est["basis"]
    clock[0] = 0.0
    short = EM.measured(None, 1, 0.3, "cpu")        # a run shorter than the window is timed whole
    assert short["window_s"] == pytest.approx(0.3) and short["wall_time_s"] == pytest.approx(30e-3, rel=0.05)


def test_measure_on_a_real_scene_keeps_the_estimate_keys():
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_sphere("ball", 0.03, pos=(0.0, 0.0, 0.2), mass=0.1)
    est = alloy.estimate(s, worlds=1, seconds=2.0, device="cpu", measure=True)
    assert est["window_s"] == pytest.approx(1.0) and est["memory_bytes"] is None
    assert np.isfinite(est["wall_time_s"]) and est["wall_time_s"] > 0 and est["physics_steps_per_world"] > 0
    assert {"step_time_s", "world_steps_per_s", "real_time_factor", "compile_s_first_run", "routes"} <= set(est)
