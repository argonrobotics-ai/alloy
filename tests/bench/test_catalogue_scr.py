# SPDX-License-Identifier: Apache-2.0
"""Catalogue lines added by accepted SCRs (SPEC §5 rows)."""

from __future__ import annotations

from alloy.bench.metric_catalogue import BY_ID


def test_liq10_resting_jitter_line_from_scr_wp9_8():
    m = BY_ID["LIQ-10"]   # WP0 accepted SCR-WP9-8 (main a092fc66): the line is WP6 / WP12's
    assert (m.unit, m.target, m.producer) == ("1", "<= 3 mm/s; no growth", "A13.resting_jitter (WP9)")
    assert BY_ID["LIQ-1"].target == "<= 1e-6"   # LIQ-10 does not shadow LIQ-1
