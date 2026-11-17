# SPDX-License-Identifier: Apache-2.0
"""``alloy.estimate(..., measure=True)``: the wall time of a real run, timed rather than modelled.

It builds the scene, steps one control period (the compile, and a GPU's first graph capture), then times the first
``WINDOW_S`` seconds of the run period by period (all of a shorter run) and charges the rest of a longer run at the
rate of the window's last quarter. A scene is usually cheapest at its start: parts dropped into a bin cost a few
collision tests while they fall and many more once they rest on it (3 scanned objects of 32 convex parts each stepped
1.7-1.9x slower resting than falling, on the CPU), and a pour's liquid grows as it leaves the nozzle. Timing only the
first 0.2 s, as this did before, read such a scene's start as its whole run: the bin 2x fast, a pour 3-6x.

Example: ``alloy.estimate(scene, worlds=4, seconds=10.0, measure=True)["wall_time_s"]``.
"""

from __future__ import annotations

import time
from typing import Any

#: simulated seconds timed: a part dropped from 1 m lands in 0.45 s, so its resting cost is in the window's last quarter
WINDOW_S = 1.0


def measured(scene: Any, worlds: int, seconds: float, device: str) -> dict[str, Any]:
    """The run's wall time from a timed window (module docstring), in ``alloy.estimate``'s keys.

    Example: ``measured(scene, 4, 10.0, "cpu")["basis"]``."""
    from alloy.api._build import build  # noqa: PLC0415

    t0 = time.perf_counter()
    sim = build(scene, worlds=worlds, device=device)
    try:
        sim.step(1)
        _sync(device)
        compile_s = time.perf_counter() - t0
        periods = max(1, int(round(seconds / sim.dt)))
        window = max(2, min(periods, int(round(WINDOW_S / sim.dt))))
        tail = max(1, window // 4)
        steps0, t1 = sim.physics_steps, time.perf_counter()
        sim.step(window - tail)
        _sync(device)
        t2 = time.perf_counter()
        sim.step(tail)
        _sync(device)
        t3 = time.perf_counter()
        steps_per_period = max(1, (sim.physics_steps - steps0) // window)
        routes, dt = sim.routes(), sim.dt
    finally:
        sim.close()
    rate = (t3 - t2) / tail                              # per control period, at the window's end
    wall = (t3 - t1) + (periods - window) * rate
    return {"worlds": worlds, "seconds": seconds, "device": device,
            "physics_steps_per_world": periods * steps_per_period, "step_time_s": rate / steps_per_period,
            "wall_time_s": wall, "world_steps_per_s": worlds * periods * steps_per_period / wall,
            "real_time_factor": worlds * seconds / wall, "memory_bytes": None, "compile_s_first_run": compile_s,
            "routes": routes, "window_s": window * dt,
            "basis": f"measured: the first {window} control periods after a warm-up period, timed; the rest of the run "
                     f"at the rate of their last {tail}"}


def _sync(device: str) -> None:
    import warp as wp  # noqa: PLC0415

    wp.synchronize_device(device)
