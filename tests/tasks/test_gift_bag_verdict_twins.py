# SPDX-License-Identifier: Apache-2.0
"""The gift-bag verdict on the tearable bag. A tear line is a pair of duplicated vertices (twins) joined by a stiff
zero-length seam, so at the end of a run each twin sits within a few float32 steps of its partner. The verdict's
hulls and triangulations (qhull) can fail on such near-duplicates (QH6271, a wide merge): the tearable demo ran every
phase on one L4 and then raised in its verdict, at ``Delaunay(bag)``. The checks now take each place once.

Data (``test_gift_bag_verdict_twins.npz``, beside this file): the bag's points (float32), the board's points and the
candle's pose at the end of the tearable demo (one L4, seed 0, 11 s), with each twin set one or two float32 steps (up
to 0.15 um) off its partner, as a seam holds them. qhull (scipy 1.18) raises QH6271 on the raw bag of this layout.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from alloy.tasks.gift_bag_checks import evaluate

DATA = Path(__file__).with_suffix(".npz")


def test_the_verdict_judges_a_tearable_bag_whose_twins_sit_float32_steps_apart():
    """The same verdict as the demo's own run printed: all 11 criteria judged and met."""
    from alloy.tasks.gift_bag_candle import CandleSpec, _layout

    z = np.load(DATA)
    bag = z["bag"]
    L = _layout(CandleSpec(tearable=True))
    out = evaluate(L, bag, z["board"], z["candle_pos"], z["candle_quat_wxyz"], bottom_mark=bag[L.bottom].mean(0),
                   hands_off=True, torn_mm=0.0)
    assert out["success"] and out["ga_judged"] == "11 of 11", out
    assert out["candle_in_bag"] and out["candle_on_board"] and out["board_in_bag"] and out["mouth_open"]
    assert out["board_fraction"] == 1.0 and out["mouth_area_ratio"] == 1.0
    assert abs(out["least_gap_mm"] - 114.3) < 0.05 and abs(out["candle_lift_mm"] - 0.2) < 0.05
