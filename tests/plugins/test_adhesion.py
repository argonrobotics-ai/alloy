# SPDX-License-Identifier: Apache-2.0
"""Adhesion (alloy/plugins/library/adhesion.py): the harness laws (A-ADH-1, area, moment, energy cycle, null), the pad
points, the ctrl clamp and the config checks (CPU)."""

from __future__ import annotations

import numpy as np
import pytest

import alloy.plugins as ap
from alloy.plugins.harness.scene import build_simulation
from alloy.plugins.library.adhesion import Adhesion, pad_points, rest_scene

pytestmark = pytest.mark.law


def test_adhesion_quick_check_passes_every_law():
    report = ap.check("adhesion", quick=True)
    assert report.passed, report.summary()
    laws = {c.name for c in report.checks if c.group == "law"}
    assert {"A-ADH-1_holds_below", "A-ADH-1_lets_go_above", "A-ADH-1_partial_holds_below",
            "A-ADH-1_partial_lets_go_above", "area_holds_below", "area_lets_go_above", "A-ADH-1_partial_moment",
            "energy_cycle", "null_out_of_reach"} <= laws


@pytest.mark.slow
def test_adhesion_full_check():
    report = ap.check("adhesion")
    assert report.passed, report.summary()


def test_pad_points_lie_on_rings_in_the_face():
    pts = pad_points(0.02, 2, 8, (0.0, 0.0, -0.01), (0.0, 0.0, -1.0))
    assert pts.shape == (16, 3)
    assert np.allclose(pts[:, 2], -0.01)
    r = np.hypot(pts[:, 0], pts[:, 1])
    assert np.allclose(r[:8], 0.01) and np.allclose(r[8:], 0.02)


def test_pull_follows_the_spread_rule():
    assert Adhesion(gain=8.0).pull(0.5, share=0.25) == pytest.approx(4.0)          # MuJoCo: the whole ctrl·gain
    assert Adhesion(gain=8.0, spread="area").pull(0.5, share=0.25) == pytest.approx(1.0)
    assert Adhesion(gain=8.0).pull(-1.0) == 0.0


def test_negative_ctrl_pulls_nothing():
    plug = Adhesion()
    sim = build_simulation(rest_scene(plug, 5.0e-4, ctrl=-1.0, periods=1), [plug], "cpu")   # box 0.5 mm under
    sim.step_period()
    r = plug.read()
    assert np.all(r["active"] == 1.0) and np.all(r["target"] == 0) and np.all(r["force"] == 0.0)


def test_config_is_validated():
    with pytest.raises(ap.PluginError, match="ring_points"):
        Adhesion(ring_points=2)
    with pytest.raises(ap.PluginError, match="spread"):
        Adhesion(spread="mean")
    with pytest.raises(ap.PluginError, match="targets"):
        Adhesion(targets=())
