# SPDX-License-Identifier: Apache-2.0
"""Law: a flag one health check raised stays readable after later checks no longer raise it (GAP-E3; CPU).

Health checks run after every physics step (8 per 10 ms control period at the default rigid step) and each one
rewrites the record's flags, so a DEGRADED flag that was true for a few steps in the middle of a run was gone when
``sim.health()``, ``traj.meta["health"]`` or ``sim.check()`` read it at a period's end (measured on main: PEN_WARN in
the record on 40 of 800 checks, final health and check clean; a per-step ENERGY_JUMP on checks 0-4, in 0 of 100
period-end reads). Coordinator ruling 2026-11-10 00:30Z: the record keeps a mask of every flag seen since the reset.

Laws:
- ``seen_flags`` of a world is a superset of its ``flags`` at every read, and never loses a flag while stepping;
- a flag raised only in the middle of a run (a weld torn, then released; a box deep in a cube for a few steps; a
  start overlap popping out) is in ``seen_flags`` at the end although ``flags`` no longer has it, in its own world;
- severity and ``sim.check()["ok"]`` are unchanged by it; ``sim.check()`` lists each such flag as a warning;
- ``sim.reset()`` clears it (a per-world reset: its world only); ``sim.settle()`` starts it afresh.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import alloy

WELD_RIG = """<mujoco><compiler angle="radian"/><worldbody>
<body name="a" pos="0 0 0.6"><joint name="ja" type="slide" axis="1 0 0" range="-1 1"/>
  <geom type="box" size="0.03 0.03 0.03" mass="5"/></body>
<body name="b" pos="0.1 0 0.6"><freejoint/><geom type="box" size="0.05 0.05 0.05" mass="20"/></body>
</worldbody><equality><weld body1="a" body2="b"/></equality>
<actuator><position joint="ja" kp="5000" kv="200"/></actuator></mujoco>"""
PULL_N = (0.0, 3000.0)                 # world 0 quiet; world 1's weld opens 7 mm (test_health_joint_fail_law)


def _rows(sim) -> list[dict[str, Any]]:
    return sim.health()["worlds"]


def _assert_superset(rows: list[dict[str, Any]]) -> None:
    for w, row in enumerate(rows):
        assert set(row["flags"]) <= set(row["seen_flags"]), (w, row)


def _step_watching(sim, periods: int) -> list[set[str]]:
    """Step ``periods`` control periods; per world, the flags any period-end read listed. Checks the superset
    and never-loses laws at every read."""
    listed = [set() for _ in range(sim.worlds)]
    seen_before = [set() for _ in range(sim.worlds)]
    for _ in range(periods):
        sim.step(1)
        rows = _rows(sim)
        _assert_superset(rows)
        for w, row in enumerate(rows):
            assert seen_before[w] <= set(row["seen_flags"]), (w, seen_before[w], row)
            seen_before[w] = set(row["seen_flags"])
            listed[w] |= set(row["flags"])
    return listed


@pytest.fixture(scope="module")
def weld_sim():
    scene = alloy.Scene(worlds=len(PULL_N), dt=0.01)
    scene.add_ground()
    scene.add_robot("rig", mjcf=WELD_RIG)
    sim = alloy.build(scene, device="cpu", welds="mujoco")       # MuJoCo's soft row: torn on purpose
    sim.step(5)
    sim.apply_wrench("rig/b", force=np.array([[0.0, f, 0.0] for f in PULL_N]))
    listed_loaded = _step_watching(sim, 50)
    sim.apply_wrench("rig/b", force=np.zeros((len(PULL_N), 3)))
    _step_watching(sim, 100)
    return sim, listed_loaded


@pytest.mark.law
def test_a_flag_gone_from_flags_stays_in_seen_flags_in_its_world(weld_sim) -> None:
    sim, listed_loaded = weld_sim
    assert "JOINT_FAIL" in listed_loaded[1] and "JOINT_FAIL" not in listed_loaded[0]    # the scene does what it says
    rows = _rows(sim)
    assert [r["flags"] for r in rows] == [[], []] and [r["severity"] for r in rows] == ["OK", "OK"]
    assert "JOINT_FAIL" in rows[1]["seen_flags"]
    assert rows[0]["seen_flags"] == []                                 # each world keeps its own mask


@pytest.mark.law
def test_check_lists_it_as_a_warning_and_ok_is_unchanged(weld_sim) -> None:
    sim, _ = weld_sim
    result = sim.check()
    assert result["ok"] is True
    warned = [p for p in result["problems"] if p.get("flag") == "JOINT_FAIL"]
    assert [p["world"] for p in warned] == [1]
    assert warned[0]["level"] == "warn" and warned[0]["fix"] and "earlier in this run" in warned[0]["what"]
    meta = sim.run(0.02).meta["health"]                                # the trajectory's summary carries it too
    assert "JOINT_FAIL" in meta["worlds"][1]["seen_flags"] and meta["ok"] is True


@pytest.mark.law
def test_reset_clears_the_mask(weld_sim) -> None:
    sim, _ = weld_sim
    assert "JOINT_FAIL" in _rows(sim)[1]["seen_flags"]
    sim.reset()
    assert [r["seen_flags"] for r in _rows(sim)] == [[], []]
    sim.step(20)                                                       # no load after a reset: nothing to see
    assert [r["seen_flags"] for r in _rows(sim)] == [[], []]


@pytest.mark.law
def test_a_start_pop_energy_jump_is_seen_at_the_end() -> None:
    """A 2 kg box starting 20 mm in the ground pops out: a per-step energy gain on the first checks only (on main
    it was in 0 of 100 period-end reads). It must be in the mask after 1 s, with the world not INVALID from it."""
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_box("blk", size=(0.2, 0.2, 0.2), pos=(0, 0, 0.08), mass=2.0)
    sim = alloy.build(scene, device="cpu")
    _step_watching(sim, 100)
    row = _rows(sim)[0]
    assert "ENERGY_JUMP" in row["seen_flags"], row
    assert "ENERGY_JUMP" not in row["flags"]                           # the latest check alone no longer says so


@pytest.mark.law
def test_settle_starts_the_mask_afresh() -> None:
    """The same start pop, then ``sim.settle``: the settled state is the new start, so what was raised while
    settling is not reported as this run's (OSS-3's review: piles and drapes raise flags while they settle)."""
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_box("blk", size=(0.2, 0.2, 0.2), pos=(0, 0, 0.08), mass=2.0)
    sim = alloy.build(scene, device="cpu")
    sim.settle(1.0)
    row = _rows(sim)[0]
    assert "ENERGY_JUMP" not in row["seen_flags"] and set(row["seen_flags"]) == set(row["flags"]), row
    assert not [p for p in sim.check()["problems"] if p.get("flag") == "ENERGY_JUMP"]


@pytest.mark.law
def test_a_new_record_and_a_masked_reset_have_an_empty_mask(weld_sim) -> None:
    """Engine level (WP5's review): the neutral record's mask is 0, and a per-world reset clears only its world."""
    import warp as wp
    from alloy.precision.health_kernels import neutral_record
    from alloy.precision.types import HealthFlags

    assert int(neutral_record().seen_flags) == 0
    sim, _ = weld_sim
    monitor = sim.engine.health
    seen = monitor.record.numpy()["seen_flags"].copy()
    seen[:] = int(HealthFlags.JOINT_FAIL)
    rec = monitor.record.numpy()
    rec["seen_flags"] = seen
    monitor.record.assign(rec)
    monitor.reset_worlds(wp.array([0, 1], dtype=wp.int32, device=monitor.device))
    assert monitor.record.numpy()["seen_flags"].tolist() == [int(HealthFlags.JOINT_FAIL), 0]
