# SPDX-License-Identifier: Apache-2.0
"""behind-radiator's verdict on known fields: Gym-Anything's progress-v1 checks, term for term (CPU, no simulation).

Every limit is Gym-Anything's literal value, tested at its edge (``GA_*`` below), never read back from the module."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.tasks import radiator_metrics as rm
from alloy.tasks import radiator_scene as rs

AREA = rs.CELL * rs.CELL

# Gym-Anything's limits (PT/task.py:86, :149, :233-249, :443-447, :489-511; PT/params.py:238-252), in 4 mm cells
GA_HIDE = 0.98                  # the scored band's hidden share
GA_HOLIDAY = 125                # cells: 20 cm² (exactly, in floats too)
GA_RUN = 6                      # cells: 1 cm² is 6.25
GA_MARKS = GA_PROTECTED = 3     # cells: 0.5 cm² is 3.125
GA_SINK_FILM = 100e-6           # m: sink volume over this is area
GA_DAMAGE, GA_FAIL_SCALE = 10.0, 0.9
BOOKS = 1e-12                   # the ledgers' relative drift (WP15's rule; atol 0)


@pytest.fixture(scope="module")
def xz():
    nx, ny = rs.wall_cells()
    w, h = rs.wall_size()
    x = -0.5 * w + (np.arange(nx) + 0.5) * rs.CELL
    z = rs.BASE_H + (np.arange(ny) + 0.5) * rs.CELL
    return np.stack(np.meshgrid(x, z, indexing="ij"), axis=-1)


@pytest.fixture(scope="module")
def masks(xz):
    return rm.Masks(rs.band_mask(xz), rs.band_mask(xz) & ~rs.pocket_mask(xz), rs.edge_mask(xz, rm.RUN_EDGE))


def inside(masks, rows: int = 0) -> tuple[int, int]:
    """A cell 10 cells in from the scored band's low corner, ``rows`` above the run-edge row."""
    ii, jj = np.nonzero(masks.scored & ~masks.edge)
    return int(ii.min()) + 10, int(jj.min()) + 10 + rows


def test_band_and_pocket_are_gym_anythings(xz, masks):
    """The band is the wall over the radiator's length (160 columns) from the baseboard's top to 1 mm under the apron
    (156 rows); the pocket is Gym-Anything's x 0.525-0.580 m below z 0.350 m, inclusive (its right bound is a column
    of centres): 14 x 67 cells, 150.1 cm² (Gym-Anything's 145 in its 2 mm cells); the run edge is one 4 mm row. The
    wall's edges are closed, as Gym-Anything's (PT/system.py:150, PT/film.py:413-415)."""
    band = np.argwhere(masks.band)
    assert masks.band.sum() == 160 * 156 and np.ptp(band[:, 0]) == 159 and np.ptp(band[:, 1]) == 155
    pocket = np.argwhere(masks.band & ~masks.scored)
    assert len(pocket) == 14 * 67 and np.ptp(pocket[:, 0]) == 13 and np.ptp(pocket[:, 1]) == 66
    assert xz[pocket[:, 0], pocket[:, 1], 0].max() == pytest.approx(0.580 - 0.55, abs=1e-12)
    assert masks.edge.sum() == 160 and np.all(np.argwhere(masks.edge)[:, 1] == band[:, 1].min())
    assert rs.charts()[rs.WALL].edges == ("wall",) * 4


def test_hide_needs_98_percent_and_no_holiday_over_20_cm2(masks):
    """Gym-Anything's hide (PT/task.py:313, :332-337): single unhidden cells two apart (no 8-connected patch) on the
    whole 2 % allowance of the scored band pass, one more fails; one unhidden patch of 125 cells (20.00 cm²) passes,
    126 fail; the pocket left bare is not scored."""
    hid = masks.band.copy()
    ok, _ = rm.hide(hid, masks, AREA)
    assert ok.ok and ok.value == 1.0
    n = int(masks.scored.sum())
    allowance = int(np.floor((1.0 - GA_HIDE) * n))                    # 480 of 24,022
    i, j = np.indices(masks.band.shape)
    lattice = np.argwhere(masks.scored & (i % 2 == 0) & (j % 2 == 0))
    for k, ok in ((allowance, True), (allowance + 1, False)):
        spots = hid.copy()
        spots[lattice[:k, 0], lattice[:k, 1]] = False
        check, detail = rm.hide(spots, masks, AREA)
        assert check.ok is ok and detail["largest_unhidden_cm2"] == pytest.approx(0.16), k
        assert (check.value >= GA_HIDE) is ok
    i0, j0 = inside(masks)
    for cells, ok in ((GA_HOLIDAY, True), (GA_HOLIDAY + 1, False)):
        hole = hid.copy()
        hole[i0:i0 + 5, j0:j0 + 25] = False                           # 125 cells in one patch
        hole[i0 + 5, j0:j0 + cells - 125] = False
        check, detail = rm.hide(hole, masks, AREA)
        assert check.ok is ok and detail["largest_unhidden_cm2"] == pytest.approx(cells * 0.16), cells
    pocket = hid & masks.scored                                       # the pocket left bare is not scored
    assert rm.hide(pocket, masks, AREA)[0].ok


def test_remaining_is_zero_exactly_when_hide_passes(masks):
    hid = masks.band.copy()
    n = int(masks.scored.sum())
    allowance = int(np.floor(0.02 * n))
    ii, jj = np.nonzero(masks.scored)
    order = np.argsort(ii * 1000 + jj)[::7]                           # scattered single cells
    hid[ii[order[:allowance]], jj[order[:allowance]]] = False
    assert rm.remaining(hid, masks.scored, AREA)[0] == 0.0
    hid[ii[order[allowance + 1]], jj[order[allowance + 1]]] = False
    assert rm.remaining(hid, masks.scored, AREA)[0] > 0.0


def test_runs_read_flowed_in_dry_paint_outside_the_edge_rows(masks):
    """Gym-Anything's runs (PT/task.py:344-355): over 50 µm flowed in, on <= 1 cm²: 6 cells pass, 7 fail; 50 µm is
    not a run; the bottom 4 mm row is not read; a wet film over 250 µm fails only before the settle."""
    wet = np.full(masks.band.shape, 100e-6)
    flowed = np.zeros(masks.band.shape)
    assert rm.runs(flowed, wet, masks, AREA)[0].ok
    ii, jj = np.nonzero(masks.band & ~masks.edge)
    for cells, ok in ((GA_RUN, True), (GA_RUN + 1, False)):
        f = flowed.copy()
        f[ii[:cells], jj[:cells]] = 51e-6
        check, detail, run = rm.runs(f, wet, masks, AREA)
        assert check.ok is ok and run.sum() == cells and detail["flowed_in_max_um"] == pytest.approx(51.0), cells
    flowed[ii[:7], jj[:7]] = 50e-6                                    # at the limit: not a run
    assert rm.runs(flowed, wet, masks, AREA)[0].ok
    edge = np.zeros(masks.band.shape)
    edge[masks.edge] = 1e-3                                           # the bottom row is not read
    assert rm.runs(edge, wet, masks, AREA)[0].ok
    thick = wet.copy()
    thick[ii[0], jj[0]] = 300e-6                                      # over 250 µm wet: fails only before the settle
    assert rm.runs(np.zeros_like(wet), thick, masks, AREA)[0].ok
    assert not rm.runs(np.zeros_like(wet), thick, masks, AREA, settled=False)[0].ok


def test_flowed_in_is_the_dry_film_less_the_tools_net_lay():
    dft = np.array([30e-6, 80e-6, 10e-6])
    supplied = np.array([30e-6, 30e-6, 30e-6]) * AREA                 # each cell got 30 µm dry from the brush
    taken = np.array([0.0, 0.0, 20e-6]) * AREA                        # the brush took 20 µm back from the last
    got = rm.flowed_in(dft, 0.0, supplied, taken, AREA)
    np.testing.assert_allclose(got, [0.0, 50e-6, 0.0], rtol=0, atol=1e-18)


def test_protected_counts_painted_cells_and_sink_volume():
    """Cells over 1 µm dry (3 cells, 0.48 cm², pass; 4 fail), and sink volume over 100 µm: the row's 2.46 µL is
    0.246 cm²."""
    d = np.zeros(10)
    ca = np.full(10, AREA)
    assert rm.protected([(d, ca)], 0.0).ok
    d[:GA_PROTECTED] = 2e-6
    assert rm.protected([(d, ca)], 0.0).ok
    d[GA_PROTECTED] = 1.5e-6
    assert not rm.protected([(d, ca)], 0.0).ok
    assert rm.protected([(np.zeros(10), ca)], 2.46e-9).value == pytest.approx(0.246e-4, rel=1e-12)
    assert rm.protected([(np.zeros(10), ca)], 0.48e-4 * GA_SINK_FILM).ok
    assert not rm.protected([(np.zeros(10), ca)], 0.52e-4 * GA_SINK_FILM).ok


def test_no_marks_counts_marred_band_cells_to_half_a_square_centimetre(masks):
    """Gym-Anything's ``no_marks`` (PT/task.py:443-447): marred cells in the band, <= 0.5 cm²: three 4 mm cells
    (0.48 cm²) pass, four fail; marred wall outside the band does not count."""
    marred = np.zeros(masks.band.shape, bool)
    i, j = np.argwhere(masks.band)[0]
    marred[i:i + GA_MARKS, j] = True
    c = rm.marks(marred, masks, AREA)
    assert c.name == "no_marks" and c.ok and c.value == pytest.approx(3 * AREA, rel=1e-12)
    marred[i + GA_MARKS, j] = True
    assert not rm.marks(marred, masks, AREA).ok
    outside = np.zeros(masks.band.shape, bool)
    outside[~masks.band] = True
    assert rm.marks(outside, masks, AREA).value == 0.0
    none = rm.marks(None, masks, AREA)                     # AB-gym-6: no marks tracked is not judged, not a pass
    assert none.name == "no_marks" and none.ok is None and "not judged" in none.note


def test_progress_is_one_on_success_and_charges_damage_ten_fold(masks):
    """progress-v1 (PT/task.py:149, :489-511): the start is the bare band's excess over one 20 cm² patch; damage
    beyond 0.5 cm² costs ten-fold; marks and runs count beyond their tolerances over hidden cells only."""
    hid0 = np.zeros(masks.band.shape, bool)
    start = rm.start_area(hid0, masks, AREA)
    scored = masks.scored.sum() * AREA                                # one bare patch: its excess over 20 cm²
    assert start == pytest.approx(scored - 20e-4, rel=1e-12)
    none = np.zeros(masks.band.shape, bool)
    done = rm.progress(masks.band.copy(), masks, AREA, start, none, none, 0.0)
    assert done["progress"] == 1.0 and done["coverage"] == 1.0
    dmg = rm.progress(masks.band.copy(), masks, AREA, start, none, none, 0.5e-4 + 20e-4)
    assert dmg["defects_cm2"]["damage"] == pytest.approx(GA_DAMAGE * 20.0)
    assert dmg["progress"] == pytest.approx(1.0 - GA_DAMAGE * 20e-4 / start, abs=1e-6)
    half = rm.progress(np.zeros(masks.band.shape, bool), masks, AREA, start, none, none, 0.0)
    assert half["progress"] == 0.0
    i0, j0 = inside(masks, rows=2)
    run, marred = none.copy(), none.copy()
    run[i0, j0:j0 + 10] = True                                        # 1.6 cm² of runs: 0.6 beyond the 1 cm²
    marred[i0 + 2, j0:j0 + 4] = True                                  # 0.64 cm² of marks: 0.14 beyond the 0.5
    hid = masks.band.copy()
    v = rm.progress(hid, masks, AREA, start, marred, run, 0.0)
    assert v["defects_cm2"]["runs"] == pytest.approx(0.6) and v["defects_cm2"]["marks"] == pytest.approx(0.14)
    hid[i0, j0:j0 + 5] = hid[i0 + 2, j0:j0 + 2] = False               # half of each unhidden: not counted there
    v = rm.progress(hid, masks, AREA, start, marred, run, 0.0)
    assert v["defects_cm2"] == {"marks": 0.0, "runs": 0.0, "damage": 0.0} and v["progress"] == 1.0


def settled_read(dry_wall: np.ndarray) -> tuple[dict, np.ndarray]:
    """A set film's ``read()`` (one world, the three charts): the wall's dry film ``dry_wall`` [nx, ny] laid by the
    brush (no flow), the protected charts clean, nothing spilled, the books closed; and the cells' areas. Every
    entry is its own array."""
    from alloy.tasks import radiator_brush as rb  # noqa: PLC0415

    sizes = [int(np.prod(c.cells)) for c in rs.charts()]
    n = sum(sizes)
    areas = np.full(n, AREA)
    dft = np.zeros((1, n))
    dft[0, : sizes[0]] = dry_wall.ravel()
    sup = (dft * areas)[..., None]
    read = {"dft": dft, "species_supplied": sup, "species_out": np.zeros_like(sup), "phi_s": np.full((1, n), 0.64),
            "spilled": np.zeros((1, n)), "species_ledger": np.zeros((1, 1)), "to_tools_volume": np.full(1, -2e-5),
            "tray_ledger": np.full(1, rb.PAIL_PAINT), "tool_picked": np.full((1, 1), 4e-5),
            "tool_dripped": np.zeros((1, 1)), "tool_load": np.zeros((1, 1)),
            **{k: np.zeros(1) for k in ("ledger", "supplied_volume", "store_ledger")},
            **{f"supply_{k}": np.zeros(1) for k in ("staged", "sent", "to_tank", "caught", "lost")}}  # fmt: skip
    return read, areas


def judge(xz, read: dict, areas: np.ndarray, marred: np.ndarray | None = None) -> tuple[dict, set[str]]:
    from alloy.tasks import radiator_task as rt  # noqa: PLC0415

    v = rt.evaluate(read, np.zeros(xz.shape[:2]), xz, areas, marred)
    return v, {k for k, c in v["checks"].items() if c["ok"] is False}


def test_the_task_judges_the_charts_it_builds(xz, masks):
    """``radiator_task.evaluate`` on a set 40 µm coat over the band (the pocket bare): it passes, score 1. Paint on
    the radiator's top or the baseboard's ledge (cells over 1 µm dry), film spilled off any chart's edge (the
    protected charts' open ones; the wall's are closed), drips caught by the radiator's or the trim's shapes and
    drips the supply books lost (to GA's protected floor; volume / 100 µm) are protected's: 0.48 cm² passes,
    0.64 cm² fails that check alone, scoring 0.9 x progress."""
    sizes = [int(np.prod(c.cells)) for c in rs.charts()]
    coat = np.where(masks.scored, 40e-6, 0.0)
    v, out = judge(xz, *settled_read(coat))
    assert out == set() and v["score"] == 1.0
    # AB-gym-6: without tracked marks no_marks is not judged (3 of Gym-Anything's 4); with them it is
    assert v["checks"]["no_marks"]["ok"] is None and v["judged"] == "3 of 4" and list(v["not_judged"]) == ["no_marks"]
    assert "4 mm cells" in v["scene"] and "own measure" in v["checks"]["protected"]["note"]
    tracked, out = judge(xz, *settled_read(coat), marred=np.zeros(xz.shape[:2], bool))
    assert out == set() and tracked["judged"] == "4 of 4" and tracked["score"] == 1.0
    assert set(v["checks"]) == {"hide", "no_runs", "no_marks", "protected", "ledger", "solids_ledger"}
    for chart in rs.PROTECTED:
        lo = sum(sizes[:chart])
        for cells, dry, want in ((3, 2e-6, set()), (4, 1.0e-6, set()), (4, 1.1e-6, {"protected"})):
            read, areas = settled_read(coat)
            read["dft"][0, lo + 7:lo + 7 + cells] = dry
            assert judge(xz, read, areas)[1] == want, (chart, cells, dry)
    for where in (rs.WALL, *rs.PROTECTED, "caught", "lost"):
        for area, want in ((0.48e-4, set()), (0.64e-4, {"protected"})):
            read, areas = settled_read(coat)
            if isinstance(where, str):
                read[f"supply_{where}"][0] = area * GA_SINK_FILM
            else:
                read["spilled"][0, sum(sizes[:where])] = area * GA_SINK_FILM
            v, out = judge(xz, read, areas)
            assert out == want and v["checks"]["protected"]["value"] == pytest.approx(area, abs=1e-6), (where, area)
            assert v["score"] == (1.0 if not want else pytest.approx(GA_FAIL_SCALE * v["progress"], rel=1e-12))


def test_each_check_fails_alone_at_gym_anythings_limit(xz, masks):
    """``radiator_task.evaluate`` at each limit, one check at a time, on the set 40 µm coat: hide (a bare patch of
    125 cells passes, 126 fail), no_runs (6 cells flowed in 51 µm pass, 7 fail), no_marks (3 marred cells pass, 4
    fail), ledger and solids_ledger (drift 0.5e-12 of the paint put in passes, 2e-12 fails). A failed verdict scores
    0.9 x progress."""
    coat = np.where(masks.scored, 40e-6, 0.0)
    nx, ny = masks.band.shape
    i0, j0 = inside(masks, rows=2)

    def bare(cells: int) -> tuple:
        d = coat.copy()
        d[i0:i0 + 5, j0:j0 + 25] = 0.0
        d[i0 + 5, j0:j0 + cells - 125] = 0.0
        return settled_read(d) + (None,)

    def runs(cells: int) -> tuple:
        read, areas = settled_read(coat)
        read["dft"][0, i0 * ny + j0 + np.arange(cells)] += 51e-6        # flowed in: dry film the brush did not lay
        return read, areas, None

    def marks(cells: int) -> tuple:
        m = np.zeros((nx, ny), bool)
        m[i0, j0:j0 + cells] = True
        return settled_read(coat) + (m,)

    def drift(key: str, rel: float) -> tuple:
        read, areas = settled_read(coat)
        if key == "ledger":
            read["ledger"][0] = rel * -read["to_tools_volume"][0]
        else:
            read["species_ledger"][0, 0] = rel * read["species_supplied"].sum()
        return read, areas, None

    cases = [(bare(GA_HOLIDAY), set()), (bare(GA_HOLIDAY + 1), {"hide"}), (runs(GA_RUN), set()),
             (runs(GA_RUN + 1), {"no_runs"}), (marks(GA_MARKS), set()), (marks(GA_MARKS + 1), {"no_marks"})]
    for key in ("ledger", "solids_ledger"):
        cases += [(drift(key, 0.5 * BOOKS), set()), (drift(key, 2.0 * BOOKS), {key})]
    for (read, areas, marred), want in cases:
        v, out = judge(xz, read, areas, marred)
        assert out == want, (want, out)
        if want:
            assert 0.9 < v["progress"] <= 1.0 and v["score"] == pytest.approx(GA_FAIL_SCALE * v["progress"], rel=1e-12)


def test_the_first_coat_is_gym_anythings_patchy_coat(xz, masks):
    """Gym-Anything's first coat (``PT/system.py:383-392``, seed 0) on the wall: 43·(1 + 0.15 f), ``f`` read off its
    2 mm wall chart at its room's x (ours + 0.55 m); within ±15 %, patchy over the band, hiding nowhere alone."""
    from alloy.tasks import radiator_task as rt  # noqa: PLC0415
    from alloy.tasks.trim_layout import GA_CELL, ROOM, first_coat_field  # noqa: PLC0415

    coat = rt.first_coat(xz)
    f = first_coat_field(0)
    i = np.floor((xz[..., 0] + 0.55 + 0.5 * ROOM["x"]) / GA_CELL + 1e-6).astype(int)
    j = np.floor(xz[..., 1] / GA_CELL + 1e-6).astype(int)
    np.testing.assert_allclose(coat, 43e-6 * (1.0 + 0.15 * f[i, j]), rtol=1e-12, atol=0)
    band = coat[masks.band]
    assert 0.85 * 43e-6 <= band.min() and band.max() <= 1.15 * 43e-6 and np.ptp(band) > 0.1 * 43e-6
    r_g = rt.srgb8_to_linear(rt.COLOURS[rt.SUBSTRATE])
    assert not rt.optics(rt.PAINT, rt.COLOUR).hidden(1e6 * coat, np.broadcast_to(r_g, coat.shape + (3,))).any()
