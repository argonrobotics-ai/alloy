# SPDX-License-Identifier: Apache-2.0
"""SB-12 membership rulings: A17 (SCR-WP11-2, coordinator 2026-11-04 05:53Z) joins the catalogue and SB-12, red."""

from __future__ import annotations

from alloy.bench import default_route as DR
from alloy.bench import law as L
from alloy.bench import report as R
from alloy.bench import scoreboard as SB
from alloy.bench.types import LawResult, MetricVerdict, Quantity


def _law(law: str, solver: str, verdict: str) -> LawResult:
    mv = MetricVerdict("m", "1", None, Quantity("m", "1", 0.4, 0.0, 1, 0.0), 0.4, 0.01, None, "mean",  # type: ignore[arg-type]
                       verdict)
    prov = {"suite": "analytic", "quick": "True", "alloy": "abc", "date": "2026-11-04"}
    return LawResult(law, solver, "cpu", (mv,), {}, prov, 0.1, True)


def test_a17_is_catalogued_for_wp11_and_gated_in_sb12_on_vbd():
    e = next(c for c in L.CATALOGUE if c.prefix == "A17")
    assert (e.suite, e.producer, e.module, e.title) == ("analytic", "WP11", "alloy.fit.momentum_law",
                                                        "VBD linear momentum")
    assert "A17" in SB.SB12_LAWS
    assert DR.on_default_route("A17.vbd_momentum", "vbd")


def test_sb12_counts_a17_and_prints_its_detail(tmp_path):
    R.write_report(_law("A5.slide", "mujoco", "pass"), tmp_path / "a")
    R.write_report(_law("A17.vbd_momentum", "vbd", "fail"), tmp_path / "b")
    by = {x.metric_id: x for x in SB.scoreboard([tmp_path])}
    assert (by["SB-12"].value, by["SB-12"].verdict) == (0.5, "fail")
    assert "A17: WP11's coarse step is the vbd default since main 8423239b" in by["SB-12"].reason


def test_a13_resting_jitter_is_counted_in_sb12_on_sph(tmp_path):
    """SCR-WP9-8 item 2 (WP0): SB-12 counts WP9's A13.resting_jitter (A13 prefix, sph is its default route)."""
    assert L.law_prefix("A13.resting_jitter") in SB.SB12_LAWS
    assert DR.on_default_route("A13.resting_jitter", "sph")
    R.write_report(_law("A5.slide", "mujoco", "pass"), tmp_path / "a")
    R.write_report(_law("A13.resting_jitter", "sph", "fail"), tmp_path / "b")
    by = {x.metric_id: x for x in SB.scoreboard([tmp_path])}
    assert (by["SB-12"].value, by["SB-12"].verdict) == (0.5, "fail")
    assert "A13.resting_jitter/sph/cpu fail" in by["SB-12"].reason
