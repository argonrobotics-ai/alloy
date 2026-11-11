# SPDX-License-Identifier: Apache-2.0
"""roll-strip with the stand-in roller (CPU, a small wall): a recommended coat passes every check, a thin coat fails to
hide, a 12-mil coat runs, and the stand-in lays exactly its coat on every cell of a lane."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.tasks.paint_roll_strip import pass_rates, run_standin, standin_passes
from alloy.tasks.paint_scene import Band, WallLayout

SMALL = WallLayout(width=0.4, height=0.352, cell=0.008, strip=Band(-0.1, 0.1, 0.076, 0.276))


def test_layout_needs_whole_cells_and_room_for_the_cut_in() -> None:
    with pytest.raises(ValueError, match="whole number"):
        WallLayout(width=0.4, height=0.5, cell=0.008)
    with pytest.raises(ValueError, match="cut-in"):
        WallLayout(width=0.4, height=0.352, cell=0.008, strip=Band(-0.1, 0.1, 0.0, 0.276))


def test_prepainted_bands_have_rebuilt_their_yield_stress() -> None:
    from alloy.tasks.paint_materials import aged_yield

    x, y = np.meshgrid(np.arange(-0.196, 0.2, 0.008), np.arange(0.004, 0.352, 0.008), indexing="ij")
    h, age = SMALL.prepaint(np.stack([x, y], -1))
    tau = aged_yield(SMALL.paint, age)
    strip, wet = SMALL.strip.mask(np.stack([x, y], -1)), h > 0.0
    assert not (strip & wet).any() and np.all(tau[~wet] == 0.25)                 # the new coat starts fresh
    assert wet.sum() > 0.5 * wet.size and np.all(tau[wet] > 1.9)                # 60-120 s: 3-6 recovery times
    assert np.all(tau[wet] / (1295.0 * 9.81) > h[wet])                          # h_c over the bands' depth: they stand


def test_standin_lays_exactly_its_coat() -> None:
    x, y = np.meshgrid(np.arange(-0.196, 0.2, 0.008), np.arange(0.004, 0.352, 0.008), indexing="ij")
    c = np.stack([x, y], -1)
    passes = standin_passes(SMALL, 100e-6)
    laid = sum(r for p in passes for r in pass_rates(SMALL, c, p)) * 0.05     # m: rate x control period
    strip = SMALL.strip.mask(c)
    assert np.allclose(laid[strip], 100e-6, rtol=1e-5) and np.all(laid[~strip] == 0.0)


@pytest.mark.compile_heavy
@pytest.mark.parametrize(("coat", "ok", "fails"), [(101.6e-6, True, set()), (30e-6, False, {"hide"}),
                                                   (300e-6, False, {"no_runs"})])
def test_roll_strip_standin(coat: float, ok: bool, fails: set[str]) -> None:
    v = run_standin(SMALL, coat=coat)
    assert v["dry_film"] == "stage"                       # the process stage's solids, not h * phi0
    assert v["dry_strip"]["p50_um"] == pytest.approx(coat * 0.42 * 1e6, rel=2e-3)   # dried h x phi_0 is 3 % short
    assert v["checks"]["solids_ledger"]["ok"]
    assert v["succeeded"] is ok
    assert {k for k, c in v["checks"].items() if c["ok"] is False} == fails
    assert v["checks"]["ledger"]["ok"]
    # AB-gym-6: what cannot fail here is not judged, and the verdict says how the scene differs (no tray: 6 criteria)
    assert v["judged"] == "4 of 6" and list(v["not_judged"]) == ["protected", "untouched"]
    assert v["checks"]["untouched"]["ok"] is None and v["checks"]["protected"]["ok"] is None
    assert "older rule" in v["checks"]["no_runs"]["note"] and "0.20 x 0.20 m strip" in v["scene"]
    assert "theirs: 0.6 x 2.23 m" in v["scene"] and v["summary"].startswith("4 of Gym-Anything's 6 criteria judged")


TWO_LANES = WallLayout(width=0.4, height=0.352, cell=0.008, strip=Band(-0.15, 0.15, 0.076, 0.276))


@pytest.mark.compile_heavy
@pytest.mark.parametrize(("pause", "lapped"), [(0.0, False), (600.0, True)])
def test_a_lane_rolled_back_over_a_dried_edge_leaves_a_lap(pause: float, lapped: bool) -> None:
    """The second lane rolls 32 mm back over the first: at once, into its wet edge, it merges (the doubled coat runs);
    600 s later the first lane is past its wet edge (phi_s > 0.57) and the overlap is a lap mark."""
    v = run_standin(TWO_LANES, coat=101.6e-6, lane_pause=pause, overlap=0.032)
    assert (not v["checks"]["no_laps"]["ok"]) is lapped
    assert v["checks"]["ledger"]["ok"]


def test_a_verdict_without_the_process_stage_cannot_pass_laps_or_solids() -> None:
    from alloy.tasks.paint_roll_strip import RunRecord, evaluate

    nx, ny = SMALL.cells
    x, y = np.meshgrid((np.arange(nx) + 0.5) * 0.008 - 0.2, (np.arange(ny) + 0.5) * 0.008, indexing="ij")
    h = np.full((nx, ny), 100e-6)
    run = RunRecord(0.0, 0.0, np.zeros((nx, ny)), np.zeros((nx, ny)))
    v = evaluate(SMALL, np.stack([x, y], -1), h, {"ledger": np.zeros(1)}, run)
    assert v["dry_film"] == "h*phi0"
    assert not v["checks"]["no_laps"]["ok"] and not v["checks"]["solids_ledger"]["ok"]


def test_aged_film_dries_at_the_stage_rate_and_stops_at_phi_m() -> None:
    from alloy.tasks.paint_process import aged_film, drying

    d = drying("latex_eggshell")
    h, solids = aged_film("latex_eggshell", np.array([110e-6, 110e-6, 50e-6, 0.0]), np.array([60.0, -1.0, 1e5, 60.0]))
    assert np.allclose(solids, [110e-6 * 0.42, 110e-6 * 0.42, 50e-6 * 0.42, 0.0], rtol=0, atol=1e-15)
    assert h[0] == pytest.approx(110e-6 - 60.0 * d.rate, rel=1e-12)          # 60 s at 5.2e-8 m/s: 3.1 um
    assert h[1] == 110e-6 and h[3] == 0.0                                    # not laid yet; no paint
    assert h[2] == pytest.approx(50e-6 * 0.42 / 0.64, rel=1e-12)              # long dry: solids / phi_m
