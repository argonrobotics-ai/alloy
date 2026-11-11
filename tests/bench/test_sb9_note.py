# SPDX-License-Identifier: Apache-2.0
"""SB-9: the CARD-1 coverage row on the pinned Newton decides (ruling scr/WP11.md 2026-11-04 23:10Z), and the board
states the known gap (coordinator 2026-11-04 16:25Z, SCR-WP11-3 (3)): the default hinge map's mesh dependence."""

from __future__ import annotations

import importlib.util

from alloy.bench import headline as H
from alloy.bench import sb9 as SB9
from alloy.core.newton_compat import NEWTON_PIN


def test_sb9_states_the_hinge_map_gap_and_the_profile_state(monkeypatch):
    gap = "default VBD hinge map is mesh-dependent (~1/120 of the card's D at 8.6 mm)"
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert "accurate-cloth profile parked (coordinator 2026-11-05 03:45Z" in SB9.hinge_gap()
    assert gap in SB9.hinge_gap()
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    assert SB9.hinge_gap().endswith("accurate-cloth profile available (alloy.solvers.vbd.cloth_profile)")
    assert gap in H.HEADLINES["SB-9"].note


def test_sb9_decides_on_the_installed_pin_only():
    on_pin, before = H.HEADLINES["SB-9"].feeds
    assert SB9.PIN == NEWTON_PIN[:8] and on_pin.marker == rf"\b{SB9.PIN}"
    assert on_pin.decides and not before.decides and before.not_marker == on_pin.marker
    assert f"pinned Newton {SB9.PIN} decides" in H.HEADLINES["SB-9"].note
