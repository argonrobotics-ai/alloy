# SPDX-License-Identifier: Apache-2.0
"""``alloy.build(..., constraint_rows=)``: the MuJoCo constraint row buffer sized by the user (CPU).

Alloy sizes a MuJoCo entry at 8 contacts per collision shape and 6 rows per contact. A shipped Panda scene is then
4,080 rows per world and uses 13, and MuJoCo Warp runs its constraint and solver kernels over the whole buffer: on
one L4 the shipped robots stepped 1.2-2.2x faster with the rows sized to the scene (WP13's per-step profile,
2026-11-09). ``contact_capacity`` alone cannot do it: a small one keeps the entry's row budget (WP9's law,
tests/scene/test_fixed_pairs_capacity.py). Laws here:
- ``constraint_rows=m`` sets njmax = m on every MuJoCo entry, with or without ``contact_capacity``; bad values and
  worker processes raise a BuildError naming the field; without it nothing changes;
- a Panda at ``contact_capacity=64, constraint_rows=384`` (the sizes of the measured row) steps bitwise like the
  default build, and uses under half of both buffers;
- ten boxes dropped at ``constraint_rows=8`` on MuJoCo Warp are flagged in every world (INVALID, frozen), never
  stepped on missing rows without a flag; on the CPU's MuJoCo C route, which sizes its own rows, the same build is
  bitwise the default's with no flag.
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.cards.library import default_root
from alloy.scene.entries import ROWS_PER_CONTACT, default_capacities

BOXES = 10
PANDA = default_root() / "robots" / "emb_e2_panda" / "scene.xml"
SIZED = {"contact_capacity": 64, "constraint_rows": 384}


def _drop(worlds: int = 2) -> alloy.Scene:
    """BOXES boxes in a row, 5 cm above the ground (tests/scene/test_fixed_pairs_capacity.py's scene)."""
    sc = alloy.Scene(dt=0.01, worlds=worlds)
    sc.add_ground()
    for i in range(BOXES):
        sc.add_box(f"b{i}", size=(0.03, 0.03, 0.03), pos=(0.06 * i, 0.0, 0.065), mass=0.1)
    return sc


def _panda() -> alloy.Scene:
    sc = alloy.Scene(dt=0.005, worlds=2)
    sc.add_robot("r", PANDA)
    return sc


def _kwargs(sim: alloy.Sim) -> dict:
    return next(e.solver_kwargs for e in sim.built.entries if e.capability == "mujoco")


def _data(sim: alloy.Sim):  # noqa: ANN202 - MuJoCo Warp's Data
    return next(s for s in sim.engine.solvers if hasattr(s, "mjw_data")).mjw_data


def test_constraint_rows_sets_the_row_buffer(monkeypatch):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "warp")
    ncon, rows = default_capacities(BOXES + 1)
    alone = alloy.build(_drop(), device="cpu", constraint_rows=96)
    assert (_kwargs(alone)["nconmax"], _kwargs(alone)["njmax"]) == (ncon, 96) and rows != 96
    both = alloy.build(_drop(), device="cpu", contact_capacity=32, constraint_rows=96)
    assert (_kwargs(both)["nconmax"], _kwargs(both)["njmax"]) == (32, 96)
    d = _data(both)
    assert (int(d.njmax), int(d.naconmax)) == (96, 32 * both.worlds)
    more = alloy.build(_drop(), device="cpu", contact_capacity=8, constraint_rows=4 * rows)
    assert _kwargs(more)["njmax"] == 4 * rows > ROWS_PER_CONTACT * 8


@pytest.mark.parametrize("bad", [0, -4, 2.5, True, "many"])
def test_a_bad_constraint_rows_names_the_field(bad):
    with pytest.raises(alloy.BuildError) as e:
        alloy.build(_drop(), device="cpu", constraint_rows=bad)
    assert e.value.field == "constraint_rows" and "rows per world" in e.value.message


def test_worker_processes_refuse_constraint_rows():
    with pytest.raises(alloy.BuildError) as e:
        alloy.build(_drop(), device="cpu", processes=2, constraint_rows=128)
    assert e.value.field == "constraint_rows"


@pytest.mark.law
@pytest.mark.timeout(600)      # two Panda builds and sim.check()'s collision pipeline: 6 s warm, minutes on cold kernels
def test_a_panda_at_the_measured_sizes_steps_bitwise_like_the_default(monkeypatch):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "warp")
    default, sized = alloy.build(_panda(), device="cpu"), alloy.build(_panda(), device="cpu", **SIZED)
    assert (_kwargs(sized)["nconmax"], _kwargs(sized)["njmax"]) == (64, 384)
    assert _kwargs(default)["njmax"] >= 10 * 384                      # the default: thousands of rows
    for sim in (default, sized):
        sim.step(20)
    a, b = default.state(), sized.state()
    assert all(np.array_equal(a[k], b[k]) for k in a), [k for k in a if not np.array_equal(a[k], b[k])]
    d = _data(sized)
    contacts, rows = int(d.nacon.numpy()[0]) / sized.worlds, int(d.nefc.numpy().max())
    print(f"panda, sized: {contacts:.0f} contacts of 64 and {rows} rows of 384 per world")
    assert 0 < contacts <= 32 and 0 < rows <= 192
    assert sized.check()["ok"] and not sized.health()["invalid_worlds"]


@pytest.mark.law
@pytest.mark.timeout(300)
def test_too_few_rows_are_flagged_in_every_world(monkeypatch):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "warp")       # MuJoCo Warp: the GPU's route
    sim = alloy.build(_drop(), device="cpu", constraint_rows=8)
    sim.step(40)
    worlds = sim.health()["worlds"]
    print(sorted({f for w in worlds for f in w["flags"]}))
    assert all(w["severity"] == "INVALID" for w in worlds), worlds
    assert all({"ROW_OVERFLOW", "CONTACT_OVERFLOW"} & set(w["flags"]) for w in worlds), worlds
    assert not sim.check()["ok"]


@pytest.mark.law
@pytest.mark.timeout(300)
def test_the_cpus_mujoco_c_route_sizes_its_own_rows(monkeypatch):
    """MuJoCo C grows its own constraint buffer, so on the CPU's default route nothing is missing at any
    ``constraint_rows``: the run is bitwise the default's and no world is flagged."""
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "mjc")        # users' CPU route (the suite's default is MuJoCo Warp)
    default, few = alloy.build(_drop(), device="cpu"), alloy.build(_drop(), device="cpu", constraint_rows=8)
    for sim in (default, few):
        sim.step(40)
    a, b = default.state(), few.state()
    assert all(np.array_equal(a[k], b[k]) for k in a), [k for k in a if not np.array_equal(a[k], b[k])]
    assert few.check()["ok"] and all(w["severity"] == "OK" for w in few.health()["worlds"]), few.health()["worlds"]
