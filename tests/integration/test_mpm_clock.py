# SPDX-License-Identifier: Apache-2.0
"""The MPM pair clock (``alloy.integration.mpm_clock``, speed-now-1109 S5): "fast" gives a scene whose two-way pairs
all end in MPM one lagged pass per exchange and a 2.5 ms exchange cap; "auto" picks it only for a servo-driven robot
with no other moving body starting near the material (``mpm_robots``), decided at build."""

from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.cards.library import default_root
from alloy.core.errors import SolverOptionError
from alloy.integration import mpm_clock

PANDA = default_root() / "robots" / "emb_e2_panda" / "scene.xml"


def test_the_coupling_is_auto_by_default_and_an_unknown_one_is_refused(monkeypatch) -> None:
    monkeypatch.delenv(mpm_clock.ENV, raising=False)
    assert mpm_clock.clock() == "auto"
    for how in ("fast", "auto"):
        monkeypatch.setenv(mpm_clock.ENV, how)
        assert mpm_clock.clock() == how
    monkeypatch.setenv(mpm_clock.ENV, "faster")
    with pytest.raises(SolverOptionError, match="use standard"):
        mpm_clock.clock()
    monkeypatch.delenv(mpm_clock.ENV)
    with mpm_clock.requested("standard"):               # alloy.build(mpm_coupling="standard")
        assert mpm_clock.clock() == "standard"
    assert mpm_clock.clock() == "auto"
    with pytest.raises(alloy.BuildError, match="unknown mpm_coupling"):
        alloy.build(alloy.Scene(dt=0.01), device="cpu", mpm_coupling="faster")


def _robot_and_sand(block_z: float | None) -> alloy.Scene:
    """A Panda (its build pose: arm up) beside a small dry-sand tray; a free steel block at ``block_z`` above it."""
    s = alloy.Scene(dt=0.01)
    s.add_robot("panda", PANDA)
    s.add_open_box("tray", inner=(0.06, 0.06, 0.03), wall=0.01, pos=(0.45, 0.25, 0.0))
    s.add_box("sand", size=(0.058, 0.058, 0.015), pos=(0.45, 0.25, 0.018), material="soil_sand_dry")
    if block_z is not None:
        s.add_box("block", size=(0.02, 0.02, 0.02), pos=(0.45, 0.25, block_z), material="steel_carbon")
    return s


def _clock_of(sim: alloy.Sim) -> tuple[int, float, str]:
    st = sim.engine.stepper
    loop = getattr(st, "iteration_loop", None) or st._loop
    return loop.cfg.k_max, st._clock.policy.dt_x_max, sim.routes()["sand"].get("mpm_coupling", "")


@pytest.mark.timeout(1800)
def test_auto_takes_one_pass_for_a_robot_in_sand_and_standard_beside_a_free_body_on_it(monkeypatch) -> None:
    monkeypatch.delenv(mpm_clock.ENV, raising=False)
    passes, cap, note = _clock_of(alloy.build(_robot_and_sand(None), device="cpu", mpm_coupling="auto"))
    assert (passes, cap) == (1, 0.002) and note.startswith("auto: 1 lagged pass") and "decided at build" in note, note
    passes, cap, note = _clock_of(alloy.build(_robot_and_sand(0.037), device="cpu", mpm_coupling="auto"))
    assert (passes, cap) == (2, 0.002) and note.startswith("standard") and "block (free)" in note, note
    passes, _cap, note = _clock_of(alloy.build(_robot_and_sand(None), device="cpu", mpm_coupling="standard"))
    assert passes == 2 and note.startswith("standard: today's clock; standard (alloy.build mpm_coupling)"), note


def _box_on_sand(periods: int) -> tuple[alloy.Sim, list[tuple[float, int, int]]]:
    """A 2 cm steel box dropped 5 mm onto a small dry-sand bed: per period (dt_x, passes, rollbacks)."""
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_box("sand", size=(0.04, 0.04, 0.012), pos=(0.0, 0.0, 0.0065), material="soil_sand_dry")
    s.add_box("box", size=(0.02, 0.02, 0.02), pos=(0.0, 0.0, 0.0275), material="steel_carbon")
    sim = alloy.build(s, device="cpu", seed=0)
    eng, rows = sim.engine, []
    for _ in range(periods):
        sim.step(1)
        rep = eng.stepper.report()
        rows.append((float(rep.dt_x), int(np.max(rep.iters_used)), int(eng.rollbacks)))
    return sim, rows


@pytest.mark.law
@pytest.mark.timeout(2400)
def test_a_box_on_sand_rests_alike_with_the_fast_pair_clock(monkeypatch) -> None:
    """Standard: 1.25 ms, two passes. Fast: one pass at 2.5 ms (the landing may refine a period or two: the force-jump
    re-run, SCR-WP11-4) and none after it; the box comes to rest on the sand at the same height within 0.5 mm, and the
    world stays healthy."""
    monkeypatch.delenv(mpm_clock.ENV, raising=False)
    std, rows = _box_on_sand(30)
    assert {r[0] for r in rows} == {0.00125} and rows[-1][1] == 2, rows
    z_std = float(std.state(["box"], world=0)["box.pos"][2])
    monkeypatch.setenv(mpm_clock.ENV, "fast")
    fast, rows = _box_on_sand(30)
    assert sum(r[0] < 0.0025 for r in rows) <= 2 and {r[0] for r in rows[-20:]} == {0.0025}, rows
    assert {r[1] for r in rows} == {1} and rows[-1][2] <= 3 and rows[-1][2] == rows[-20][2], rows    # landing only
    z_fast = float(fast.state(["box"], world=0)["box.pos"][2])
    v = np.linalg.norm(fast.state(["box"], world=0)["box.vel"][:3])
    assert abs(z_fast - z_std) < 5e-4 and v < 0.01, (z_std, z_fast, v)
    assert fast.check()["ok"], fast.check()["problems"]


def _drop(clock: str) -> tuple[float, float]:
    """The block dropped 4 cm onto the sand beside the robot: (its rest height [m] at 0.4 s, its largest speed [m/s]
    over 0.3-0.4 s)."""
    sim = alloy.build(_robot_and_sand(0.075), device="cpu", mpm_coupling=clock)
    speeds = []
    for _ in range(40):
        sim.step(1)
        st = sim.state(["block"], world=0)
        speeds.append(float(np.linalg.norm(st["block.vel"][:3])))
    return float(sim.state(["block"], world=0)["block.pos"][2]), max(speeds[30:])


@pytest.mark.law
@pytest.mark.timeout(2400)
def test_a_block_dropped_on_the_sand_later_rests_within_half_a_millimetre_on_the_auto_clock(monkeypatch) -> None:
    """auto is decided at build: a free block that starts 4 cm above the sand (outside the margin) and lands later runs
    on auto's clock (1 lagged pass). Ruled bound (coordinator 2026-11-09): its rest height within 0.5 mm of the
    standard clock's and its speed under 1 cm/s from 0.3 s. Measured (CPU): 28.48 against 28.02 mm, at most 6 mm/s."""
    monkeypatch.delenv(mpm_clock.ENV, raising=False)
    z_std, _ = _drop("standard")
    z_auto, v_auto = _drop("auto")
    assert abs(z_auto - z_std) < 5e-4 and v_auto < 0.01, (z_std, z_auto, v_auto)
