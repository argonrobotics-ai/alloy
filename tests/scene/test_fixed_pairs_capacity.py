# SPDX-License-Identifier: Apache-2.0
"""Contact capacity: ``alloy.build(..., contact_capacity=)`` and CONTACT_OVERFLOW's one console line, CPU.

The kinematic cup (tests/scene/test_fixed_pairs.py) overflowed MuJoCo Warp's contact buffer beside a robot, and the
user had no setting for it (8 contacts per collision shape, at least 64) while MuJoCo Warp printed its overflow
warning on every step (FIX-LIST item 6).
- ``contact_capacity=n`` sizes every MuJoCo entry at n contacts per world (MuJoCo Warp's pool: n x worlds) and 6
  constraint rows per contact, or the entry's own row budget when that is larger (a small n keeps it); bad values and
  worker processes raise a BuildError naming the field. None keeps the scene's sizing: a build at
  ``contact_capacity`` equal to that sizing steps bitwise like the default build.
- Ten boxes dropped onto the ground in each of 2 worlds at ``contact_capacity=8``, on MuJoCo Warp (the GPU's route)
  and on the CPU's MuJoCo C route (which keeps every contact in its physics but exports 8 per world to health): every
  world is CONTACT_OVERFLOW and INVALID, and over 40 periods the console shows one line, naming MuJoCo's buffer of 8
  per world, a ``contact_capacity`` of at least 16 (2 x 8, or the contacts seen per world when more) and
  ``sim.health()``, and none of MuJoCo Warp's own ("please increase nconmax"). At the default capacity the same scene
  has no overflow and prints no line.
"""

from __future__ import annotations

import ctypes
import re

import numpy as np
import pytest

import alloy
from alloy.scene.entries import ROWS_PER_CONTACT, default_capacities

BOXES = 10


def _drop(worlds: int = 2) -> alloy.Scene:
    """BOXES boxes in a row, 5 cm above the ground (no contact at the start, so MuJoCo keeps the capacity)."""
    sc = alloy.Scene(dt=0.01, worlds=worlds)
    sc.add_ground()
    for i in range(BOXES):
        sc.add_box(f"b{i}", size=(0.03, 0.03, 0.03), pos=(0.06 * i, 0.0, 0.065), mass=0.1)
    return sc


def _mujoco(sim: alloy.Sim):
    return next(s for s in sim.engine.solvers if hasattr(s, "mj_model"))


def _console(capfd) -> str:
    ctypes.CDLL(None).fflush(None)                    # Warp's CPU printf writes through C stdio
    out, err = capfd.readouterr()
    return out + err


@pytest.mark.parametrize("bad", [0, -4, 1.5, True, "64"])
def test_a_bad_contact_capacity_names_the_field(bad):
    with pytest.raises(alloy.BuildError) as e:
        alloy.build(_drop(), device="cpu", contact_capacity=bad)
    assert e.value.field == "contact_capacity"


def test_worker_processes_refuse_a_contact_capacity():
    with pytest.raises(alloy.BuildError) as e:
        alloy.build(_drop(), device="cpu", processes=2, contact_capacity=128)
    assert e.value.field == "contact_capacity"


def test_contact_capacity_sizes_the_mujoco_entry(monkeypatch):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "warp")
    sim = alloy.build(_drop(), device="cpu", contact_capacity=200)
    kw = next(e.solver_kwargs for e in sim.built.entries if e.capability == "mujoco")
    assert (kw["nconmax"], kw["njmax"]) == (200, ROWS_PER_CONTACT * 200)
    d = _mujoco(sim).mjw_data
    assert int(d.naconmax) == 200 * sim.worlds


def test_a_small_contact_capacity_keeps_the_entrys_row_budget(monkeypatch):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "warp")
    sim = alloy.build(_drop(), device="cpu", contact_capacity=8)
    kw = next(e.solver_kwargs for e in sim.built.entries if e.capability == "mujoco")
    assert (kw["nconmax"], kw["njmax"]) == (8, default_capacities(BOXES + 1)[1])      # 528 rows, not 48


def test_the_scenes_own_capacity_passed_by_hand_steps_bitwise(monkeypatch):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "warp")
    default = alloy.build(_drop(), device="cpu")
    kw = next(e.solver_kwargs for e in default.built.entries if e.capability == "mujoco")
    shapes = BOXES + 1
    assert (kw["nconmax"], kw["njmax"]) == default_capacities(shapes)
    same = alloy.build(_drop(), device="cpu", contact_capacity=kw["nconmax"])
    for sim in (default, same):
        sim.step(30)
    a, b = default.state(), same.state()
    assert all(np.array_equal(a[k], b[k]) for k in a), [k for k in a if not np.array_equal(a[k], b[k])]


@pytest.mark.law
@pytest.mark.parametrize("route", ["warp", "mjc"])
def test_an_overflow_prints_one_line_with_the_fix(route, monkeypatch, capfd):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", route)        # mjc: users' CPU route (the suite's default is MuJoCo Warp)
    sim = alloy.build(_drop(), device="cpu", contact_capacity=8)
    _console(capfd)
    sim.step(40)
    text = _console(capfd)
    lines = [ln for ln in text.splitlines() if "CONTACT_OVERFLOW" in ln]
    assert len(lines) == 1, text[-2000:]
    fix = re.search(r"contact_capacity=(\d+)", lines[0])
    assert fix and int(fix.group(1)) >= 16 and "MuJoCo's buffer (8 per world" in lines[0], lines[0]
    assert "first flagged: world" in lines[0] and "sim.health()" in lines[0], lines[0]
    print(lines[0])
    assert "increase nconmax" not in text, text[-2000:]
    worlds = sim.health()["worlds"]
    assert all(w["severity"] == "INVALID" and "CONTACT_OVERFLOW" in w["flags"] for w in worlds), worlds


def test_the_default_capacity_holds_the_drop_and_prints_nothing(monkeypatch, capfd):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "warp")
    sim = alloy.build(_drop(), device="cpu")
    _console(capfd)
    sim.step(40)
    text = _console(capfd)
    assert "CONTACT_OVERFLOW" not in text and "increase nconmax" not in text, text[-2000:]
    assert sim.check()["ok"], sim.check()["problems"]
