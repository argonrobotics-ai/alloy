# SPDX-License-Identifier: Apache-2.0
"""Bag a boxed candle on the GPU (alloy/tasks/gift_bag_candle.py; docs/notes/paper-pack.md). Each test runs the whole
11 s demo: on rc-1109 the 64-world demo took 2,124 s and 1,937 s on a pool L4 (seeds 0 and 1, every world bagged), and
both tests were still stepping at the 600 s GPU default (WP5's 100k pass, 2026-11-09), so they set TIMEOUT_S.

The scripted demo in 4 seeded worlds (cartons moved up to ±20 mm and turned up to ±30 deg): every world succeeds, and
a carton lowered 40 mm off the mouth's centre line catches the wall and fails (the bag responds, the checks see it).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

pytestmark = pytest.mark.gpu
TIMEOUT_S = 3600            # the 11 s demo: ~2,100 s wall at 64 worlds on a pool L4, with margin


@pytest.fixture(scope="module", autouse=True)
def _cuda():
    wp = pytest.importorskip("warp")
    if not wp.is_cuda_available():
        pytest.skip("needs CUDA")


@pytest.mark.timeout(TIMEOUT_S)
def test_the_scripted_demo_bags_the_candle_in_every_world():
    from alloy.tasks.gift_bag_candle import CandleSpec, run

    out = run(CandleSpec(), device="cuda:0", worlds=4, seed=0)
    assert out["check_ok"], out["problems"]
    assert out["success_rate"] == 1.0, out["worlds"]


@pytest.mark.timeout(TIMEOUT_S)
def test_a_carton_lowered_off_the_mouth_fails():
    from alloy.tasks.gift_bag_candle import CandleSpec, _layout, build, plan
    from alloy.tasks.gift_bag_checks import evaluate
    from alloy.tasks.gift_gantry import drive, schedule

    s = CandleSpec()
    sim, st = build(s, worlds=1, seed=0, device="cuda:0")
    drive(sim, schedule(dataclasses.replace(plan(s), to_y=np.array([0.04])), st))
    x = sim.state(["bag", "board", "candle"])
    v = evaluate(_layout(s), x["bag.points"][0], x["board.points"][0], x["candle.pos"][0], x["candle.quat_wxyz"][0])
    assert not v["candle_on_board"] and not v["success"], v
    assert sim.check()["ok"]
