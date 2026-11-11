# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from bench.render_gates import checks


def row(flip: float, lpips: float) -> dict:
    return {"raw": {"flip_mean": flip, "lpips": lpips}, "matched": {"flip_mean": flip, "lpips": lpips}}


S1 = {(128, "on"): row(0.02, 0.004), (16, "on"): row(0.04, 0.01)}


def test_g1_limit_is_twice_seed1_128spp_denoised() -> None:
    assert checks.g1_limit(S1) == {"flip_mean": 0.04, "lpips": 0.008}
    assert checks.g1_limit({(64, "on"): row(0.1, 0.1)}) is None


def test_parity_uses_highest_spp_and_best_denoise_state() -> None:
    eng = {(1024, "on"): row(0.03, 0.009), (1024, "off"): row(0.035, 0.007), (16, "on"): row(0.001, 0.0001)}
    rec = checks.check_parity(eng, S1)
    assert rec["best_image"] == "1024spp_off"
    assert rec["status"] == "PASS" and rec["ratio"] == pytest.approx(0.875)
    rec = checks.check_parity({(1024, "on"): row(0.05, 0.001)}, S1)
    assert rec["status"] == "FAIL" and rec["ratio"] == pytest.approx(1.25)


def test_g1_images_ratio_per_setting() -> None:
    g = checks.g1_images({(5, "on"): row(0.02, 0.009)}, S1)
    assert g["5spp_on"]["pass"] is False and g["5spp_on"]["ratio"] == pytest.approx(1.125)


def test_yardstick_is_worst_control() -> None:
    y = checks.yardstick({"cycles_s1": {(4, "on"): row(0.1, 0.01)}, "cycles_s2": {(4, "on"): row(0.08, 0.02)}})
    assert y == {(4, "on"): {"flip_mean": 0.1, "lpips": 0.02}}


def test_rows_exclude_the_engine_itself_when_it_is_a_control() -> None:
    controls = {"cycles_s1": {(4, "on"): row(0.1, 0.01)}, "cycles_s2": {(4, "on"): row(0.2, 0.02)}}
    rows = checks.rows_for("cycles_s2", controls["cycles_s2"], controls)
    assert len(rows) == 1 and rows[0]["ratio"] == pytest.approx(2.0) and not rows[0]["pass"]


def rows_with(passes: int, total: int, worst: float = 1.0) -> list[dict]:
    out = [{"pass": i < passes, "ratio": 1.0 if i < passes else 1.2} for i in range(total)]
    out[0]["ratio"] = worst
    return out


def test_statistical_verdict_compares_to_worst_control_rate() -> None:
    controls = {"cycles_s1": rows_with(9, 10), "cycles_s2": rows_with(8, 10)}
    assert checks.equal_spp_verdict(rows_with(8, 10), controls)["status"] == "PASS"
    assert checks.equal_spp_verdict(rows_with(7, 10), controls)["status"] == "FAIL"


def test_statistical_verdict_caps_any_row_at_1_25() -> None:
    v = checks.equal_spp_verdict(rows_with(10, 10, worst=1.3), {"cycles_s1": rows_with(5, 10)})
    assert v["status"] == "FAIL" and v["worst_ratio"] == pytest.approx(1.3)


def test_statistical_verdict_without_controls_needs_every_row() -> None:
    assert checks.equal_spp_verdict(rows_with(9, 10), {})["status"] == "FAIL"
    assert checks.equal_spp_verdict([], {})["status"] == "n/a"


def test_speed_bar_and_report_only() -> None:
    rows = [{"res": "128x128", "worlds": 1024, "images_per_s": 900}, {"res": "128x128", "worlds": 64, "images_per_s": 5000}]
    bar = {"match": {"res": "128x128", "worlds": 1024}, "min_images_per_s": 1000}
    assert checks.check_speed(rows, bar) == {"status": "FAIL", "value": 900, "threshold": 1000}
    assert checks.check_speed(rows, {"match": {"worlds": 64}})["status"] == "report"
    assert checks.check_speed([], bar)["status"] == "pending"
