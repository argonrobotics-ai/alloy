# SPDX-License-Identifier: Apache-2.0
"""swatch-card with SOFT-3's round brush (CPU).

Fast: the crop sits on Gym-Anything's grid (cell centres on whole millimetres; a 51 x 51 square; outside from the
27 mm centres, on either axis); the strokes tile the square with 6 mm bands 5 mm apart whose outer edges and ends are
the edge cells' far side, under the tip's widest band; the verdict on synthetic fields passes and fails each check at
its limit (15.0 / 14.9 µm, the blue's share, the line's fixed cells, drips past the line, books, solids).

Slow (not in the gate's fast lane): the honest fill passes; Gym-Anything's corner episode (the last stroke 40 mm
short) fails only coverage; outer strokes 3 mm out fail only within_line; the brush loaded from the red well (the wrong
paint, beside the blue) fails only coverage; each pigment's books close."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.tasks.swatch_card import (BAND, CELL, CROP, SPECIES, WELLS, Plan, end_inset, evaluate, masks, max_band,
                                     pigment_books, run, strokes, well_at)

WELL = 9.0e-6
MM = 1e-3


def centres() -> np.ndarray:
    n = int(round(CROP[0] / CELL))
    x = (np.arange(n) + 0.5) * CELL - 0.5 * CROP[0]
    return np.stack(np.meshgrid(x, x, indexing="ij"), -1)


def at(x_mm: float, y_mm: float) -> tuple[int, int]:
    """The index of the cell centred at (x, y) [mm]."""
    c = centres()
    return int(np.argmin(np.abs(c[:, 0, 0] - x_mm * MM))), int(np.argmin(np.abs(c[0, :, 1] - y_mm * MM)))


def test_crop_on_gym_anythings_grid() -> None:
    """Cell centres on whole millimetres; the square is the 51 x 51 cells within 25 mm; a cell 26 mm out on one axis is
    inside the 1 mm margin, 27 mm out on either axis is outside, a corner cell at (26, 26) is not."""
    c = centres()
    assert np.allclose(c[..., 0] / MM, np.rint(c[..., 0] / MM), atol=1e-9)
    square, outside = masks(c)
    assert square.sum() == 51 * 51
    for (x, y), out in (((25, 0), False), ((26, 0), False), ((27, 0), True), ((0, -26), False), ((0, -27), True),
                        ((26, 26), False), ((27, -3), True), ((-40, 40), True)):  # fmt: skip
        assert bool(outside[at(x, y)]) is out, (x, y)
    assert square[at(-25, -25)] and square[at(25, 25)] and not square[at(26, 0)]


def test_strokes_tile_the_square() -> None:
    """Ten 6 mm bands 5 mm apart (1 mm overlap) from y = -22.5 to 22.5 mm, outer edges at the edge cells' far side
    (±25.5 mm); each stroke runs x = ∓25.5 mm, the band no wider than the tip lays; the corner plan stops the last one
    40 mm short."""
    s = strokes(Plan())
    ys = np.array([a[1] for a, _ in s]) / MM
    assert len(s) == 10 and ys[0] == pytest.approx(-22.5) and ys[-1] == pytest.approx(22.5)
    assert np.diff(ys).max() <= 5.0 + 1e-9
    assert BAND == pytest.approx(6 * MM) and BAND <= max_band()
    inset = end_inset()
    assert all(abs(a[0]) == pytest.approx(25.5 * MM - inset) and abs(b[0]) == pytest.approx(25.5 * MM - inset)
               for a, b in s)  # fmt: skip
    for y in range(-25, 26):                                   # every row of the square under a band
        assert any(abs(y - yk) <= 3.0 for yk in ys), y
    from alloy.tasks import paint_brush as pb  # noqa: PLC0415
    from alloy.tasks.swatch_card import PAINT  # noqa: PLC0415

    # SOFT-3's tip meters, laying under its whole contact past its centre: the strokes end its inset short (WP15 S4)
    assert pb.brush_store(PAINT).meter > 0.0 and inset == pytest.approx(pb.end_inset(BAND)) and inset > 0.0
    assert max_band() == pb.BAND_MAX
    a, b = strokes(Plan(corner=0.04))[-1]
    assert abs(a[0] - b[0]) == pytest.approx(51 * MM - 2 * inset - 0.04)


def synthetic(dry: np.ndarray) -> dict:
    """The film's read() of a world whose dry film is ``dry`` [nx, ny], laid by the brush, its books closed."""
    d = dry.reshape(1, -1)
    return {"dft": d, "species_supplied": (d * CELL * CELL)[..., None], "species_ledger": np.zeros((1, 1)),
            "ledger": np.zeros(1), "store_ledger": np.zeros(1), "tray_ledger": np.full(1, WELL),
            "tool_picked": np.full((1, 1), 2e-7)}


def failed(read: dict, drips: float = 0.0, share: np.ndarray | None = None) -> set[str]:
    v = evaluate(centres(), read, WELL, past_line_drips=drips, share=share)
    return {k for k, c in v["checks"].items() if c["ok"] is False}


def test_the_verdict_says_what_it_judges() -> None:
    """AB-gym-6: ``untouched`` is None (no marks are tracked), never a pass on an empty array; coverage and the line
    are judged by other measures and say so; with one paint coverage's share part is not judged either."""
    from alloy.tasks import paint_metrics as pm  # noqa: PLC0415

    c = centres()
    square, _ = masks(c)
    base = np.where(square, 30e-6, 0.0)
    v = evaluate(c, synthetic(base), WELL, share=np.ones(square.shape))
    assert v["succeeded"] and v["judged"] == "2 of 3" and v["not_judged"] == {"untouched": pm.NO_MARKS}
    assert v["checks"]["untouched"]["ok"] is None and v["checks"]["coverage"]["ok"] is True
    assert "looser by up to 5 %" in v["checks"]["coverage"]["note"]
    assert "81 mm crop" in v["checks"]["within_line"]["note"]
    assert "0.65" in v["scene"] and v["summary"].startswith("2 of Gym-Anything's 3 criteria judged; not judged")
    one = evaluate(c, synthetic(base), WELL)
    assert one["checks"]["coverage"]["ok"] is None and one["judged"] == "1 of 3" and one["succeeded"]


def test_verdict_passes_and_fails_each_check_at_its_limit() -> None:
    """Synthetic dry films on the square: 30 µm passes. Coverage: 6 % of the square at exactly 15.0 µm passes, at
    14.9 µm fails (4 % at 14.9 passes), and a blue share of 0.94 there fails. Within the line: a 51-cell column at
    26 mm passes, at 27 mm fails (0.51 cm²); a 50-cell row at -27 mm passes, 51 fail; 3.0 µm is not paint, 3.1 µm
    is; drips past the line by the 100 µm rule. Books at 1e-12; the solids ledger."""
    c = centres()
    square, _ = masks(c)
    base = np.where(square, 30e-6, 0.0)
    assert failed(synthetic(base)) == set()
    some = np.flatnonzero(square)
    for share, value, out in ((0.06, 15.0e-6, set()), (0.06, 14.9e-6, {"coverage"}), (0.04, 14.9e-6, set())):
        d = base.copy()
        d.ravel()[some[:int(share * square.sum())]] = value
        assert failed(synthetic(d)) == out, (share, value)
    blue = np.ones(square.shape)
    blue.ravel()[some[:int(0.06 * square.sum())]] = 0.94
    assert failed(synthetic(base), share=blue) == {"coverage"}
    assert failed(synthetic(base), share=np.ones(square.shape)) == set()
    for x, value, out in ((26, 4e-6, set()), (27, 4e-6, {"within_line"}), (27, 3.0e-6, set()),
                          (27, 3.1e-6, {"within_line"})):  # fmt: skip
        d = base.copy()
        i = at(x, 0)[0]
        d[i, at(0, -25)[1]:at(0, 25)[1] + 1] = value          # a 51-cell column
        assert failed(synthetic(d)) == out, (x, value)
    for count, out in ((50, set()), (51, {"within_line"})):
        d = base.copy()
        j = at(0, -27)[1]
        d[at(-25, 0)[0]:at(-25, 0)[0] + count, j] = 4e-6     # a row 27 mm below the square: outside on y alone
        assert failed(synthetic(d)) == out, count
    assert failed(synthetic(base), drips=0.49e-4 * 100e-6) == set()
    assert failed(synthetic(base), drips=0.51e-4 * 100e-6) == {"within_line"}
    for drift, out in ((0.5e-12, set()), (2e-12, {"books"})):
        r = synthetic(base)
        r["store_ledger"][0] = drift * 2e-7
        assert failed(r) == out
    r = synthetic(base)
    r["species_ledger"][0, 0] = 1e-9 * float(r["species_supplied"].sum())
    assert failed(r) == {"solids_ledger"}


def test_wells_and_pigment_books() -> None:
    """Gym-Anything's palette: the blue well (0) and the red (1) 70 mm apart, 0.5 m from the square; the film carries
    the solids and each pigment. The pigment books pass at 1e-12 of what was picked up and fail past it, in any of the
    film, the brush's bins or the wells."""
    blue, red = well_at("blue"), well_at("red")
    assert tuple(WELLS) == ("blue", "red") and SPECIES == ("solids", "blue", "red")
    assert math.dist(blue[:2], red[:2]) == pytest.approx(0.07) and math.dist(blue[:2], (0.0, 0.0)) > 0.45
    wells0 = np.array([2e-6, 1e-6, 1e-6])
    base = {"species_ledger": np.zeros((1, 3)), "store_species_ledger": np.zeros((1, 3)),
            "tray_species_ledger": wells0[None].copy(), "tool_species_picked": np.full((1, 1, 3), 1e-7)}
    assert pigment_books(base, wells0).ok
    for key in ("species_ledger", "store_species_ledger", "tray_species_ledger"):
        for drift, ok in ((0.5e-12, True), (2e-12, False)):
            r = {k: v.copy() for k, v in base.items()}
            r[key][0, 2] += drift * 1e-7
            assert pigment_books(r, wells0).ok is ok, (key, drift)


@pytest.mark.slow
@pytest.mark.timeout(1800)
@pytest.mark.parametrize(("plan", "out"), [(Plan(), set()), (Plan(corner=0.04), {"coverage"}),
                                           (Plan(overrun=0.003), {"within_line"}), (Plan(well="red"), {"coverage"})])
def test_swatch_card(plan: Plan, out: set[str]) -> None:
    v = run(plan, settle=5.0)
    assert {k for k, c in v["checks"].items() if c["ok"] is False} == out, v
    assert v["checks"]["books"]["ok"] and v["checks"]["solids_ledger"]["ok"] and v["checks"]["pigment_books"]["ok"]
