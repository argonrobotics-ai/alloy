# SPDX-License-Identifier: Apache-2.0
"""``HealthMonitor.register_share_check`` (GAP #6 slice 2b; WP5's ruling 2026-11-10): a per-world share over its bound
sets a HealthFlags bit (default QUASI_STATIC, DEGRADED) at that check, sticky in ``seen_flags``; a monitor without share
checks launches nothing. The physics law (a mass-scaled solid pressed too fast) is in
tests/solvers/plastic/test_quasi_static.py."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

import alloy
from alloy.precision.ladder import DEGRADED_MASK, row
from alloy.precision.types import HealthFlags, Severity


def _sim(worlds: int = 2):
    s = alloy.Scene(worlds=worlds, dt=0.01)
    s.add_ground()
    s.add_box("cube", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.025), mass=0.2)
    return alloy.build(s, device="cpu")


def test_quasi_static_is_a_degraded_flag_with_a_ladder_row():
    assert row("QUASI_STATIC").severity == Severity.DEGRADED
    assert DEGRADED_MASK & int(HealthFlags.QUASI_STATIC)


@pytest.mark.compile_heavy
def test_a_share_over_its_bound_flags_that_world_only_while_it_lasts():
    sim = _sim()
    health = sim.engine.health
    assert health._plugins.shares == []                     # nothing registered: nothing launches
    share = wp.zeros(2, dtype=wp.float32, device="cpu")
    assert health.register_share_check("probe", share, 0.02) == 0
    share.assign(np.array([0.01, 0.05], np.float32))        # world 1 over its bound
    sim.step(20)
    rows = sim.health()["worlds"]
    assert rows[0]["severity"] == "OK" and "QUASI_STATIC" not in rows[0]["flags"]
    assert rows[1]["severity"] == "DEGRADED" and "QUASI_STATIC" in rows[1]["flags"], rows[1]
    assert sim.check()["ok"]                                # DEGRADED: listed, not a failed run
    share.zero_()
    sim.step(20)
    rows = sim.health()["worlds"]
    assert "QUASI_STATIC" not in rows[1]["flags"] and rows[1]["severity"] == "OK", rows[1]
    assert "QUASI_STATIC" in rows[1].get("seen_flags", []), rows[1]     # sticky in seen_flags


@pytest.mark.compile_heavy
def test_bad_share_checks_are_refused():
    health = _sim().engine.health
    good = wp.zeros(2, dtype=wp.float32, device="cpu")
    for share, bound, what in ((wp.zeros(3, dtype=wp.float32, device="cpu"), 0.1, "shape"),
                               (wp.zeros(2, dtype=wp.float64, device="cpu"), 0.1, "float32"),
                               (good, 0.0, "positive"), (good, float("nan"), "positive")):
        with pytest.raises(ValueError, match=what):
            health.register_share_check("bad", share, bound)
    health.register_share_check("once", good, 0.1)
    with pytest.raises(ValueError, match="twice"):
        health.register_share_check("once", good, 0.1)
    for flag in (int(HealthFlags.QUASI_STATIC | HealthFlags.PEN_WARN), 1 << 30):      # two bits; no ladder row
        with pytest.raises(ValueError, match="one HealthFlags bit with a ladder row"):
            health.register_share_check(f"flag {flag}", good, 0.1, flag=flag)
