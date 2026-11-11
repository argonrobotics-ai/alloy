# SPDX-License-Identifier: Apache-2.0
"""Tight capacities step bitwise like the default ones on the CPU (SPEED-FIX B1).

The Panda pack (arm holding, a box, cylinder and mug on its table: 29 contacts per world at peak) and the LEAP pack
with a 4 cm cube dropped into the hand (16 contacts), 2 worlds, 200 physics steps (50 periods of 5 ms at 1.25 ms),
at the default sizes (Panda 1,648 contacts and 9,888 rows per world; LEAP 592 / 3,552) and at contact_capacity=64
(rows 6 x 128 + 10 = 778 and 3 x 128 + 32 = 416): every state key is bitwise equal. On MuJoCo Warp (the suite's CPU
route) rows past nefc are skipped by every row kernel and the CPU runs each kernel's threads in order, so the
solver's sums see the same terms in the same order; on the CPU's MuJoCo C route njmax only sizes the health mirror.
Not shown here: a GPU, where tiled and atomic sums could order differently (the meter compares there).
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.api._capacity import mujoco_solver


def _pack(name: str) -> alloy.Scene:
    from alloy.cards.library import default_root

    sc = alloy.Scene(dt=0.005, worlds=2)
    sc.add_robot("r", default_root() / "robots" / name / "scene.xml")
    if name == "emb_e9_leap":
        sc.add_box("cube", size=(0.04, 0.04, 0.04), pos=(0.0, -0.03, 0.2), mass=0.05)
    return sc


def _run(name: str, capacity: int | None) -> tuple[dict, tuple[int, int]]:
    sim = alloy.build(_pack(name), device="cpu", contact_capacity=capacity)
    d = mujoco_solver(sim).mjw_data
    sim.step(50)
    assert sim.check()["ok"], sim.check()["problems"]
    return sim.state(), (int(d.naconmax) // 2, int(d.njmax))


@pytest.mark.law
@pytest.mark.parametrize("pack, route, sized", [("emb_e2_panda", "warp", (64, 778)), ("emb_e9_leap", "warp", (64, 416)),
                                                ("emb_e2_panda", "mjc", (64, 778))])
def test_tight_capacity_steps_bitwise_like_the_default(pack, route, sized, monkeypatch):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", route)
    ref, caps = _run(pack, None)
    got, tight = _run(pack, 64)
    assert tight == sized and caps[1] > 8 * sized[1], (caps, tight)
    diff = [k for k in ref if not np.array_equal(np.asarray(ref[k]), np.asarray(got[k]), equal_nan=True)]
    assert not diff, diff
