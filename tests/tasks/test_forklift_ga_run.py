# SPDX-License-Identifier: Apache-2.0
"""GA's forklift tasks run on Alloy's physics with the scripted operator and scored by GA's ported criteria
(SCR-WP8-12 F5). Each task's scripted world passes every criterion; a world whose script makes one deliberate error
fails exactly the criterion that error breaks. One world each, on the suite's MuJoCo Warp CPU route: 1-4 min of wall
time per run (about 4x the MuJoCo C route's), the bales about 12 min."""

from __future__ import annotations

import dataclasses
import importlib

import pytest

from alloy.tasks.forklift_ga_run import Task, simulate
from alloy.tasks.forklift_path import segment
from alloy.tasks.forklift_plans import Plan

pytestmark = [pytest.mark.law, pytest.mark.slow]


def _failed(verdict) -> set[str]:
    return set(verdict.failed())


def _task(module: str, **kw) -> Task:
    return importlib.import_module(f"alloy.tasks.forklift_ga_{module}").task(**kw)


def _with_segment(task: Task, key: str, end_shift: tuple[float, float] = (0.0, 0.0), cap: float | None = None) -> Task:
    """The task with segment ``key`` ending ``end_shift`` (m, GA frame) from where it did, or capped at ``cap``."""
    seg = task.plan.segments[key]
    *head, last = seg.points
    moved = (last[0] + end_shift[0], last[1] + end_shift[1], *last[2:])
    plan = Plan(task.plan.kind, {**task.plan.segments, key: segment([*head, moved], cap or seg.cap_kmh)},
                task.plan.levels)
    return dataclasses.replace(task, plan=plan)


@pytest.mark.timeout(600)
def test_start_and_park():
    """Parked in the bay, forks on the floor; without the parking brake only ``parking_brake_set`` fails."""
    task = _task("park")
    assert simulate(task)[1].success
    off = simulate(dataclasses.replace(task, tamper=lambda c: dataclasses.replace(c, park=False)))[1]
    assert _failed(off) == {"parking_brake_set"}


@pytest.mark.timeout(1500)
def test_ramp_down_loaded():
    """Out of the yard trailer, down the ramp under 6 km/h, set down in the drop zone; driving the ramp at 9 km/h
    fails only ``slow_on_ramp``."""
    task = _task("ramp")
    verdict = simulate(task)[1]
    assert verdict.success, _failed(verdict)
    fast = simulate(_with_segment(task, "ramp", cap=9.0))[1]
    assert _failed(fast) == {"slow_on_ramp"}


@pytest.mark.timeout(1500)
def test_put_away_rack():
    """On both beams at 1.60 m, 51-101 mm past the front beam; creeping in 10 cm short fails only
    ``front_overhang``."""
    task = _task("rack")
    verdict = simulate(task)[1]
    assert verdict.success, _failed(verdict)
    short = simulate(_with_segment(task, "creep", end_shift=(0.10, 0.0)))[1]
    assert _failed(short) == {"front_overhang"}


@pytest.mark.timeout(1500)
def test_load_trailer():
    """Behind the nose pallet against the wall, the truck back on the dock; setting it down 10 cm short of the nose
    pallet fails only ``gap_to_pallets_ahead``."""
    task = _task("dock")
    verdict = simulate(task)[1]
    assert verdict.success, _failed(verdict)
    short = simulate(_with_segment(task, "in", end_shift=(0.0, -0.10)))[1]
    assert _failed(short) == {"gap_to_pallets_ahead"}


@pytest.mark.timeout(900)
def test_load_trailer_with_the_restraint_red_waits_off_the_leveler():
    """The restraint shows red (no hook, no chocks): the truck waits short of the leveler, parked, load on the forks
    (40 s of it here; GA's episode runs on to its 240 s limit, where the same criteria passed)."""
    verdict = simulate(dataclasses.replace(_task("dock", secured=False), time_limit=40.0))[1]
    assert verdict.success, _failed(verdict)


@pytest.mark.timeout(900)
def test_a_red_restraint_run_that_never_reaches_its_wait_fails_time_limit():
    """WP13's review of the rc1 red fix: a red run held at rest before it reaches the restraint check (here brakes on
    from the start, in its first drive phase) has not finished its plan, so it cannot pass time_limit by resting."""
    held = dataclasses.replace(_task("dock", secured=False), time_limit=15.0,
                               tamper=lambda c: dataclasses.replace(c, throttle=0.0, brake=1.0))
    ep, verdict = simulate(held)
    assert ep.plan_done_t is None
    assert "time_limit" in _failed(verdict)


@pytest.mark.timeout(3000)
def test_load_bales():
    """Two bales carried into the trailer, each on its strips at the nose, the truck back on the dock."""
    verdict = simulate(_task("bale_run"))[1]
    assert verdict.success, _failed(verdict)


@pytest.mark.gpu
@pytest.mark.timeout(900)
def test_start_and_park_on_cuda():
    """The L4 smoke: start-and-park on MuJoCo Warp on CUDA (contacts exported there too) passes every criterion;
    prints the wall time per simulated second."""
    import time  # noqa: PLC0415

    import warp as wp  # noqa: PLC0415

    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    t0 = time.perf_counter()
    episode, verdict = simulate(_task("park"), device="cuda:0")
    wall = time.perf_counter() - t0
    print(f"[ok] start-and-park on cuda: {wall / float(episode.t[-1]):.2f} s wall per simulated s, build included")
    assert verdict.success, _failed(verdict)
