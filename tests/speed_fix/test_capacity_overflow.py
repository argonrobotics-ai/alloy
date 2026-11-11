# SPDX-License-Identifier: Apache-2.0
"""Capacity overflows fail closed at ``contact_capacity=n`` (SPEED-FIX B1): rows, contacts and the CCD scratch.

- Rows: MuJoCo Warp's pool of n x worlds is shared, so one world can take more than n contacts while the pool holds
  them all; its rows (3 per contact here) then pass r x n + R. Ten boxes landing on the ground in world 0 (40
  contacts) and in the air in world 1, at n = 24: world 0 reads ROW_OVERFLOW (INVALID, frozen), not CONTACT_OVERFLOW
  (the pool of 48 holds the 40), and sim.check() names it with contact_capacity as the fix; world 1 stays OK. On the
  CPU's MuJoCo C route, which keeps every row, the same world reads CONTACT_OVERFLOW instead (its contact export
  is n per world).
- CCD scratch: a pool of 1 slot per world (WP5's internal ``set_ccd_slots``) under four cylinders resting on a
  table reads CONTACT_OVERFLOW in every world; at contact_capacity the scratch is the pool (naccdmax = naconmax) and
  the same scene reads nothing.
- The tight sizes of WP13's probe (64 contacts; G1 256) hold the Panda and G1 packs: 50 control periods, no
  overflow flag, sim.check() ok.
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.api._capacity import mujoco_solver

BOXES = 10


def _uneven(worlds: int = 2) -> alloy.Scene:
    """BOXES boxes 2 cm above the ground in world 0 (resting after ~0.07 s; no contact at the start, so Newton keeps
    the capacity), 1 m up in world 1 (they land after ~0.45 s)."""
    sc = alloy.Scene(dt=0.01, worlds=worlds)
    sc.add_ground()
    for i in range(BOXES):
        z = np.array([0.035] + [1.0] * (worlds - 1))
        pos = np.stack([np.full(worlds, 0.06 * i), np.zeros(worlds), z], axis=1)
        sc.add_box(f"b{i}", size=(0.03, 0.03, 0.03), pos=pos, mass=0.1)
    return sc


def _flags(sim: alloy.Sim) -> list[set[str]]:
    return [set(w["flags"]) for w in sim.health()["worlds"]]


@pytest.mark.law
def test_a_world_past_its_share_of_the_pool_reads_row_overflow(monkeypatch):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "warp")
    sim = alloy.build(_uneven(3), device="cpu", contact_capacity=16)   # rows for 2 x 16 contacts; world 0 holds 40
    assert (int(mujoco_solver(sim).mjw_data.njmax), int(mujoco_solver(sim).mjw_data.naconmax)) == (96, 48)
    sim.step(20)
    f = _flags(sim)
    assert "ROW_OVERFLOW" in f[0] and "CONTACT_OVERFLOW" not in f[0], f
    assert not (f[1] | f[2]) & {"ROW_OVERFLOW", "CONTACT_OVERFLOW"}, f
    chk = sim.check()
    bad = [p for p in chk["problems"] if p["world"] == 0]
    assert not chk["ok"] and bad and "contact_capacity" in bad[0]["fix"], chk["problems"]
    assert all(p["world"] == 0 for p in chk["problems"]), chk["problems"]


@pytest.mark.law
def test_the_c_route_reads_the_same_world_as_a_contact_overflow(monkeypatch):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "mjc")
    sim = alloy.build(_uneven(), device="cpu", contact_capacity=24)
    sim.step(20)
    f = _flags(sim)
    assert "CONTACT_OVERFLOW" in f[0] and not f[1] & {"ROW_OVERFLOW", "CONTACT_OVERFLOW"}, f


def _cylinders(worlds: int = 2) -> alloy.Scene:
    sc = alloy.Scene(dt=0.01, worlds=worlds)
    sc.add_ground()
    sc.add_box("table", size=(0.6, 0.6, 0.1), pos=(0.0, 0.0, 0.05), static=True)
    for i in range(4):
        sc.add_cylinder(f"c{i}", radius=0.02, height=0.06, pos=(0.08 * i - 0.12, 0.0, 0.131), mass=0.1)
    return sc


@pytest.mark.law
@pytest.mark.parametrize("slots", [1, None])
def test_a_ccd_pool_too_small_reads_contact_overflow(slots, monkeypatch):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "warp")
    from alloy.precision.mjw_ccd import set_ccd_slots

    sim = alloy.build(_cylinders(), device="cpu", contact_capacity=32)
    d = mujoco_solver(sim).mjw_data
    assert int(d.naccdmax) == int(d.naconmax) == 64
    if slots is not None:
        set_ccd_slots(mujoco_solver(sim), slots)            # before the first step: the scratch is sized per call
    sim.step(20)
    f = _flags(sim)
    if slots is None:
        assert sim.check()["ok"] and not any(w & {"CONTACT_OVERFLOW", "ROW_OVERFLOW"} for w in f), f
    else:
        assert all("CONTACT_OVERFLOW" in w for w in f), f


@pytest.mark.law
@pytest.mark.parametrize("pack, n", [("emb_e2_panda", 64), ("emb_e6_g1", 256)])
def test_the_tight_sizes_hold_the_packs(pack, n, monkeypatch):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "warp")
    from alloy.cards.library import default_root

    sc = alloy.Scene(dt=0.005, worlds=2)
    sc.add_robot("r", default_root() / "robots" / pack / "scene.xml")
    sim = alloy.build(sc, device="cpu", contact_capacity=n)
    sim.step(50)
    f = _flags(sim)
    assert not any(w & {"CONTACT_OVERFLOW", "ROW_OVERFLOW"} for w in f) and sim.check()["ok"], f
