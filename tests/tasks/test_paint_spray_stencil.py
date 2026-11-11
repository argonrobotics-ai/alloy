# SPDX-License-Identifier: Apache-2.0
"""spray-stencil (CPU): the stencil's "3", the gauge and the spray checks on synthetic fields, the stand-in can's solids
books (a nozzle above the box's rim lets the mist out), and the stand-in on a crate cropped to the "3": two light
coats pass every check, one fails the coat."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy.ndimage import label

from alloy.tasks import paint_metrics as pm
from alloy.tasks.paint_spray_stencil import (Books, Can, SprayLayout, period_dust, period_rates, run_standin, schedule,
                                             stencil_holes)

SMALL = SprayLayout(crate=(0.12, 0.19), stencil=(0.12, 0.19))        # the "3" with a margin: 22,800 cells


def test_the_three_is_one_open_stroke() -> None:
    c = SprayLayout().centres()
    holes = stencil_holes(c)
    assert 6000 < holes.sum() < 7300                                  # mm²: two 22 mm bowls, open on the left
    assert label(holes)[1] == 1
    x, y = c[..., 0][holes], c[..., 1][holes]
    assert -0.052 < x.min() and x.max() < 0.052 and -0.089 < y.min() and y.max() < 0.089
    assert not holes[(np.abs(c[..., 0] + 0.04) < 1e-3) & (np.abs(np.abs(c[..., 1]) - 0.036) < 1e-3)].any()


def test_gauge_and_spray_checks_on_synthetic_coats() -> None:
    c = SMALL.centres()
    target = stencil_holes(c)
    even = np.where(target, 35e-6, 0.0)
    assert np.allclose(pm.gauge(even, target, 1e-3)[target], 35e-6)          # the gauge reads only the coated area
    coat, edges = pm.spray_coat(even, target, 1e-3)
    assert coat.ok and coat.value == 1.0 and edges.ok and edges.value == 0.0
    assert not pm.spray_coat(even * 2.0, target, 1e-3)[0].ok                # 70 um: over the can's 50 um
    from scipy.ndimage import binary_dilation

    bled = np.where(binary_dilation(target, iterations=3), 35e-6, 0.0)       # paint 3 mm past the aperture
    assert pm.spray_coat(bled, target, 1e-3)[0].ok and not pm.spray_coat(bled, target, 1e-3)[1].ok
    assert pm.overspray(0.04, 1.0).ok and not pm.overspray(0.06, 1.0).ok


@pytest.mark.parametrize(("standoff", "escapes"), [(0.30, False), (1.20, True)])
def test_spray_and_mist_leave_the_box_only_over_its_rim(standoff: float, escapes: bool) -> None:
    lay, can = SprayLayout(), dataclasses.replace(Can(), standoff=standoff, pause=0.0)
    c = lay.centres()
    sheet = lay.sheet(c)
    books = Books()
    laid = sum(period_rates(can, aims, c, ~sheet, books).astype(np.float64) for aims in schedule(can))
    assert books.emitted > 0 and books.closure() < 1e-12 and books.in_box > 0
    assert np.isclose(laid.sum() * 0.05 * 1e-6 * 0.25, books.on_crate, rtol=1e-5)    # the source is what the books say
    assert not laid[sheet].any()                                                  # nothing under the sheet
    assert (books.outside == 0.0) is (can.above_rim <= 0.0)                       # 0.3 m: the nozzle is in the box
    assert pm.overspray(books.outside, books.emitted).ok is not escapes          # 1.2 m: 13 % leaves


def test_sharp_edges_allow_one_cell_and_count_per_target_cell() -> None:
    from scipy.ndimage import binary_dilation

    target = stencil_holes(SMALL.centres())
    one, two = binary_dilation(target, iterations=1), binary_dilation(target, iterations=2)
    assert pm.spray_coat(np.where(one, 35e-6, 0.0), target, 1e-3)[1].value == 0.0        # the 1 mm margin is free
    ring = int((two & ~one).sum())
    assert pm.spray_coat(np.where(two, 35e-6, 0.0), target, 1e-3)[1].value == pytest.approx(ring / target.sum())


SHORT = dict(rows=(-0.08, -0.04, 0.0, 0.04, 0.08), span=0.10, pause=1.0)   # five rows over the cropped "3"


@pytest.mark.compile_heavy
@pytest.mark.parametrize(("coats", "fails", "dry_um"), [(2, set(), (40.0, 47.0)), (1, {"coat"}, (19.0, 24.0))])
def test_spray_stencil_standin(coats: int, fails: set[str], dry_um: tuple[float, float]) -> None:
    v = run_standin(SMALL, dataclasses.replace(Can(), coats=coats, **SHORT), settle=1.0)
    assert v["dry_film"] == "stage"                       # the solids the film carries, the mist's dust among them
    assert dry_um[0] < v["dry_target"]["p50_um"] < dry_um[1]     # 17 um a coat + 5-6 um of dust (h x phi_0: 31)
    assert {k for k, c in v["checks"].items() if c["ok"] is False} == fails
    assert v["succeeded"] is (not fails)
    # AB-gym-6: no marks are tracked, so untouched is not judged (it passed on an empty array before)
    assert v["judged"] == "4 of 5" and list(v["not_judged"]) == ["untouched"]
    assert v["checks"]["untouched"]["ok"] is None and "own books" in v["checks"]["overspray"]["note"]
    assert "fixed pose" in v["scene"]


def test_books_fail_without_the_films_solids() -> None:
    from alloy.tasks.paint_spray_stencil import _books_drift

    assert _books_drift({"supplied_volume": np.array([1e-6])}, Books(on_crate=2.5e-7)) == float("inf")


def test_mist_dust_lands_on_the_open_crate_as_dry_solids() -> None:
    """Two coats from 0.3 m: the mist's dust adds 9.8-12.2 um dry over the "3", none under the sheet, all of it inside
    the box's books."""
    lay, can = SprayLayout(), dataclasses.replace(Can(), pause=0.0)
    c = lay.centres()
    sheet = lay.sheet(c)
    books = Books()
    dust = np.zeros(c.shape[:2])
    for aims in schedule(can):
        period_rates(can, aims, c, ~sheet, books)
        dust += period_dust(can, aims, c, ~sheet, books)
    target = stencil_holes(c)
    assert 9.5e-6 < dust[target].min() and dust[target].max() < 12.5e-6 and not dust[sheet].any()
    assert 0.0 < books.dust < books.in_box
    # against the drift Gaussian's closed form: on the whole crate (nothing seen through) the dust is the mist's
    # solids x the Gaussian's share over the crate, to the 1 mm cells' quadrature
    from alloy.tasks.paint_spray_stencil import DRIFT, _gauss_rect

    whole, every, expect = Books(), np.ones_like(sheet), 0.0
    for aims in schedule(can):
        if aims is None:
            continue
        laid = period_dust(can, aims, c, every, whole)
        mist = can.wet_rate / len(aims) * (1.0 - can.efficiency) * 0.05 * 0.25
        expect += sum(mist * _gauss_rect(mx, my, DRIFT * can.standoff, 0.225, 0.15) for mx, my in aims)
        assert laid.min() >= 0.0
    assert whole.dust == pytest.approx(expect, rel=1e-4, abs=0.0)
    assert books.dust < whole.dust
