# SPDX-License-Identifier: Apache-2.0
"""clear-coat (CPU): the can against Gym-Anything's aerosol, the passes, the escape rule against SOFT-3's stand-in
books, the mist's dust, the checks and Gym-Anything's score on synthetic prints (each threshold at its edge), the can's
books, and the scripted clear coat on a print crop: two light coats and their dust pass every check with the film's,
DC5's and the solids' books exact while the film dries."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from alloy.tasks import paint_clear_coat as cc
from alloy.tasks import paint_clear_coat_checks as cc_checks
from alloy.tasks import paint_metrics as pm
from alloy.tasks import paint_spray_stencil as st
from alloy.tasks import paint_supply as ps_


def test_the_can_is_gym_anythings_aerosol_on_an_expel_valve() -> None:
    can = cc.AerosolCan()
    assert can.rate == pytest.approx(2.0e-3 * 0.4 / 1000.0, rel=1e-12)          # 0.8 mL/s lands of 2.0 g/s
    assert can.cd_area * math.sqrt(2.0 * (can.pressure - cc.P_ATM) / cc.DENSITY) == pytest.approx(can.rate, rel=1e-12)
    assert can.rate / can.exit_area == pytest.approx(25.0, rel=1e-12)
    assert can.capacity == pytest.approx(0.34 * 0.4 / 1000.0, rel=1e-12)       # 136 mL of landed lacquer
    noz = can.nozzle()
    assert (noz.pattern, noz.body, noz.efficiency) == ("spray", cc.CAN, st.Can().efficiency)
    assert noz.angle == pytest.approx(2.0 * math.radians(12.0), rel=1e-12)
    edge = can.network()["edges"][0]
    assert (edge["kind"], edge["name"], edge["capacity"]) == ("expel", cc.EDGE, can.capacity)


def test_passes_lay_the_wet_film_rate_speed_and_pitch_give() -> None:
    lay, ps, can = cc.BoxLayout(print_size=(0.06, 0.05)), cc.Passes(), cc.AerosolCan()
    sched = cc.schedule(lay, ps)
    hx = 0.5 * lay.print_size[0] + ps.margin
    pause = int(round(ps.pause / cc.CONTROL_DT))
    rows = [k for k in range(1, len(sched)) if sched[k][2] and not sched[k - 1][2]]
    assert not sched[0][2] and len(rows) == 12 and sum(not s[2] for s in sched) == 12 + pause
    n = sum(s[2] for s in sched) // len(rows)
    ys = sorted({round(s[1], 9) for s in sched if s[2]})
    assert np.allclose(np.diff(ys), 0.5 * ps.pitch)                             # coat 2 sits half a pitch over
    assert sched[rows[0]][0] < 0 < sched[rows[1]][0]                            # back and forth
    wet = can.efficiency * can.rate * n * cc.CONTROL_DT / (2.0 * hx * ps.pitch)  # one coat, over its rows' band
    assert wet == pytest.approx(can.efficiency * can.rate / (ps.speed * ps.pitch), rel=0.03)
    assert 70e-6 < wet < 74e-6 and 14e-6 < wet * cc.PHI0 < 14.8e-6            # 14.4 um dry a coat, before the dust


@pytest.mark.parametrize("up", [0.05, 0.2, 0.75, 1.5])
@pytest.mark.parametrize("aim", [(0.0, 0.0), (0.2, 0.1), (0.29, 0.0)])
def test_escape_matches_soft3s_stand_in_books(up: float, aim: tuple[float, float]) -> None:
    """The nozzle's own rays (DC5's table) and SOFT-3's Gaussian quadrature book the same escape over the rim."""
    lay, can = cc.BoxLayout(), cc.AerosolCan()
    c = st.SprayLayout(crate=(0.12, 0.19), stencil=(0.12, 0.19)).centres()
    books = st.Books()
    st.period_rates(dataclasses.replace(st.Can(), standoff=up + st.BOX[2] - st.CRATE_Z), [aim], c,
                    np.ones(c.shape[:2], dtype=bool), books)
    mine = cc.escape_share(lay, can.efficiency, can.ray_table(), (*aim, lay.rim_z + up))
    assert abs(mine - books.outside / books.emitted) <= 0.01


def test_nothing_leaves_from_inside_the_box_and_all_from_beside_it() -> None:
    lay, can = cc.BoxLayout(), cc.AerosolCan()
    eff, rays = can.efficiency, can.ray_table()
    for off in (0.25, 0.40):
        assert cc.escape_share(lay, eff, rays, (0.1, -0.1, lay.print_z + off)) == 0.0
    assert cc.escape_share(lay, eff, rays, (0.5, 0.0, lay.print_z + 0.30)) == 1.0
    assert cc.escape_share(lay, eff, rays, (0.0, 0.0, lay.rim_z + 1.2)) > 0.05   # above the rim: it fails overspray
    up = 0.8
    assert cc.mist_escape(lay, (0.1, 0.0, lay.rim_z + up)) == pytest.approx(
        1.0 - cc_checks.rect_share(0.1, 0.0, 0.25 * up, 0.30, 0.225), rel=1e-12)
    assert cc.mist_escape(lay, (0.0, 0.0, lay.rim_z - 0.1)) == 0.0 and cc.mist_escape(lay, (0.4, 0.0, 1.0)) == 1.0


def test_all_of_it_leaves_from_beside_the_box_whatever_the_tables_rounding() -> None:
    """Beside the box every ray and the mist leave: exactly all of it, also from a table whose normalised fractions sum
    an ulp off 1 (seven sevenths: 0.9999999999999998; WP0's AVX-512 VM read 0.9999999999999999 for the can's)."""
    lay, up = cc.BoxLayout(), np.array([0.0, 0.0, 1.0])
    rays = (np.tile(-up, (7, 1)), np.full(7, 1.0 / 7.0))
    assert float(rays[1].sum()) != 1.0
    for eff in (0.7, 0.35, 1.0):
        assert cc.escape_share(lay, eff, rays, (0.5, 0.0, lay.print_z + 0.30)) == 1.0


def test_from_inside_the_box_only_rays_up_through_the_opening_leave() -> None:
    lay = cc.BoxLayout()
    up, wall = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.05]) / math.hypot(1.0, 0.05)
    rays = (np.stack([up, wall, -up]), np.array([0.5, 0.3, 0.2]))
    assert cc.escape_share(lay, 0.7, rays, (0.0, 0.0, lay.rim_z - 0.1)) == pytest.approx(0.7 * 0.5, rel=1e-12)


def _dust(lay: cc.BoxLayout, aims: np.ndarray, fall: np.ndarray, solids: np.ndarray) -> np.ndarray:
    xe, ye = lay.edges()
    return cc.mist_dust(xe, ye, np.full(lay.cells, lay.cell_area), aims, fall, solids)


def test_the_mist_falls_as_dust_drifting_a_quarter_of_its_fall() -> None:
    """Gym-Anything's dry fall (``params.py:200``, ``aerosol.py:298-302``): σ = 0.25 x the fall, in every world, each
    release point's own share where it was released, the chart's cells holding exactly their integral."""
    assert st.DRIFT == 0.25
    lay = cc.BoxLayout(print_size=(0.8, 0.8), bed=0.0)
    xs, _ = lay.axes()
    aims, fall = np.array([[0.0, 0.0], [0.04, 0.0]]), np.array([0.3, 0.4])
    solids = np.array([[1e-6, 3e-6], [2e-6, 2e-6]])
    d = _dust(lay, aims, fall, solids)
    vol = d.sum(axis=(1, 2)) * lay.cell_area
    for w in range(2):
        s = 0.25 * fall[w]
        inside = [cc_checks.rect_share(a[0], a[1], s, 0.4, 0.4) for a in aims]
        assert vol[w] == pytest.approx(float(solids[w] @ inside), rel=1e-12)      # exact over the chart
        col = d[w].sum(axis=1)
        mean = (col * xs).sum() / col.sum()
        assert mean == pytest.approx(float(solids[w] @ aims[:, 0]) / solids[w].sum(), abs=1e-4)   # 4 sigma: 3e-5 off
        var = (col * (xs - mean) ** 2).sum() / col.sum() - float(solids[w] @ (aims[:, 0] - mean) ** 2) / solids[w].sum()
        assert math.sqrt(var) == pytest.approx(s, rel=2e-3)                       # sigma = DRIFT x the fall
    assert not _dust(lay, aims, fall, np.zeros((2, 2))).any()


def test_the_dust_books_close_on_what_was_laid() -> None:
    """mist = on the chart + in the box + outside, the chart's share an analytic integral: dust laid twice, at one
    point or drifting a fifth as far breaks them."""
    lay = cc.BoxLayout(print_size=(0.06, 0.05))
    areas = np.full(lay.cells, lay.cell_area)
    aims, fall = np.array([[0.0, 0.0], [0.05, 0.02], [0.08, 0.04]]), np.array([0.25, 0.40])
    z = lay.print_z + fall
    solids = np.full((2, 3), 1e-7)

    def closure(d: np.ndarray) -> list[float]:
        books = cc.DustBooks()
        cc.dust_books(lay, aims, z, fall, solids, (d * areas).sum(axis=(1, 2)), books)
        return [books.closure(w, solids[w].sum()) for w in range(2)]

    assert max(closure(_dust(lay, aims, fall, solids))) <= 1e-13
    for bad in (2.0 * _dust(lay, aims, fall, solids), _dust(lay, aims[-1:], fall, solids[:, :1] * 3.0),
                _dust(lay, aims, 0.2 * fall, solids)):
        assert min(closure(bad)) > 1e-3


def test_coat_and_even_on_synthetic_films() -> None:
    lay = cc.BoxLayout()
    mask = lay.print_mask()
    assert mask.sum() == 254 * 203 and lay.cells == (314, 263)                   # the print and its 30 mm bed
    even = np.where(mask, 35e-6, 0.0)                                             # the bed's film is not read
    coat, ratio = cc.coat_checks(even, mask)
    assert (coat.value, coat.ok, ratio.ok) == (1.0, True, True) and ratio.value == pytest.approx(1.0, rel=1e-12)
    thin = even.copy()
    x0 = int(np.flatnonzero(mask.any(axis=1))[0])
    thin[x0:x0 + 26] = np.where(mask[x0:x0 + 26], 20e-6, 0.0)   # 26 of 254 columns at 20 um; the probe lifts the last
    coat, ratio = cc.coat_checks(thin, mask)
    assert not coat.ok and coat.value == pytest.approx(229 / 254, rel=1e-12)
    assert ratio.ok and ratio.value == pytest.approx(1.75, rel=1e-9)
    edge = even.copy()
    edge[x0:x0 + 15] = np.where(mask[x0:x0 + 15], 20e-6, 0.0)                    # 14 columns out: 0.945, under 0.95
    assert cc.coat_checks(edge, mask)[0].value == pytest.approx(240 / 254, rel=1e-12)
    assert not cc.coat_checks(edge, mask)[0].ok
    patch = np.where(mask, 40e-6, 0.0)
    patch[150:156, 120:126] = 16e-6                                               # a 6 mm patch: ratio 2.5, coat 0.96
    coat, ratio = cc.coat_checks(patch, mask)
    assert coat.ok and not ratio.ok and ratio.value == pytest.approx(2.5, rel=1e-9)
    bare = even.copy()
    bare[130:140, 130:140] = 0.0                                                  # a 10 mm bare patch
    assert cc.coat_checks(bare, mask)[1].value == 99.0 and not cc.coat_checks(bare, mask)[1].ok


def test_can_books_flash_what_did_not_land() -> None:
    read = {"supply_sent": np.array([1e-6]), "supply_edge_ledger": np.array([[-1.25e-6]])}
    b = cc.can_books(read)
    assert b["landed_kg"] == pytest.approx(1e-3, rel=1e-12) and b["flashed_kg"] == pytest.approx(1.5e-3, rel=1e-12)
    assert b["used_kg"] == pytest.approx(2.5e-3, rel=1e-12) and b["in_valve_m3"] == pytest.approx(0.25e-6, rel=1e-9)


TINY = cc.BoxLayout(print_size=(0.04, 0.04), bed=0.005)                         # 40 x 40 print cells in 50 x 50


def _read(lay: cc.BoxLayout, dft: np.ndarray, clip_cell: tuple[int, int] | None = None, solids_off: float = 0.0,
          sent: float = 1e-6, landed_frac: float = 0.5, laid: float = 1e-8) -> dict:
    """A one-world read with closed books: ``landed_frac`` of what the can sent landed on the chart, the rest is mist,
    and ``laid`` m³ of its solids were laid back on the chart as dust (:func:`_books`)."""
    n, landed = lay.cells[0] * lay.cells[1], landed_frac * sent
    clipped = np.zeros(lay.cells)
    if clip_cell is not None:
        clipped[clip_cell] = 1e-12
    one = np.array
    return {"dft": dft.reshape(1, n), "supplied": np.full((1, n), landed / n), "clipped": clipped.reshape(1, n),
            "supply_sent": one([sent]), "supplied_volume": one([landed]), "pending_volume": one([0.0]),
            "species_mass": one([[(landed * cc.PHI0 + laid) * (1.0 + solids_off)]]), "supply_swept": one([0.0]),
            "supply_withdrawn": one([0.0]), "supply_to_tank": one([0.0]), "supply_lost": one([0.0]),
            "supply_overspray": one([0.0]), "supply_mist": one([sent - landed]), "ledger": one([0.0]),
            "supply_caught": one([0.0]), "supply_lumped": one([0.0]),
            "evaporated_volume": one([0.0]), "supply_edge_ledger": one([[-sent]])}


def _books(read: dict, laid: float = 1e-8, in_box_off: float = 0.0) -> cc.DustBooks:
    """The dust's books for ``read``: all of DC5's mist stays in the box, ``laid`` of it on the chart."""
    mist = read["supply_mist"] * cc.PHI0
    books = cc.DustBooks()
    books.add(mist, np.array([laid]), mist - laid + in_box_off, np.zeros(1))
    return books


def _failed(v: dict) -> list[str]:
    return [k for k, c in v["checks"].items() if c["ok"] is False]


def test_untouched_is_judged_only_from_tracked_marks() -> None:
    """AB-gym-6: with no marks tracked ``untouched`` is None (not judged: 4 of Gym-Anything's 5), never a pass on an
    empty array; a tracked array is judged, passes clean and fails marred; the verdict names the scene's differences."""
    mask, area = TINY.print_mask(), TINY.cell_area
    good, wet = np.where(mask, 35e-6, 0.0), np.full(TINY.cells, 150e-6)
    r = _read(TINY, good)
    v = cc.evaluate(wet, r, 0.0, _books(r), mask, area)
    assert v["succeeded"] and v["score"] == 1.0 and v["checks"]["untouched"]["ok"] is None
    assert v["judged"] == "4 of 5" and v["not_judged"] == {"untouched": pm.NO_MARKS}
    assert "free-moving" in v["scene"] and v["summary"].startswith("4 of Gym-Anything's 5 criteria judged; not judged")
    assert "depth cap" in v["checks"]["no_runs"]["note"] and "own count" in v["checks"]["overspray"]["note"]
    marred = np.zeros(mask.shape, bool)
    clean = cc.evaluate(wet, r, 0.0, _books(r), mask, area, marred=marred)
    assert clean["checks"]["untouched"]["ok"] is True and clean["judged"] == "5 of 5" and clean["not_judged"] == {}
    marred[mask] = True
    bad = cc.evaluate(wet, r, 0.0, _books(r), mask, area, marred=marred)
    assert _failed(bad) == ["untouched"] and not bad["succeeded"] and bad["score"] < 0.9


def test_the_verdict_and_score_on_synthetic_prints() -> None:
    mask, area = TINY.print_mask(), TINY.cell_area
    good, wet = np.where(mask, 35e-6, 0.0), np.full(TINY.cells, 150e-6)
    r = _read(TINY, good)
    v = cc.evaluate(wet, r, 0.0, _books(r), mask, area)
    assert v["succeeded"] and v["score"] == 1.0 and v["raw"] == {"ledger": 0.0, "supply": 0.0, "solids": 0.0,
                                                                 "dust": 0.0}
    bed = int(np.flatnonzero(~mask.any(axis=1))[0])
    r = _read(TINY, good, clip_cell=(bed, 0))
    assert cc.evaluate(wet, r, 0.0, _books(r), mask, area)["succeeded"]
    spike = wet.copy()
    spike[bed, 0] = 2.0 * pm.RUN_H                                                       # a run on the bed: not read
    assert cc.evaluate(spike, r, 0.0, _books(r), mask, area)["succeeded"]
    r = _read(TINY, good, clip_cell=(25, 25))
    assert _failed(cc.evaluate(wet, r, 0.0, _books(r), mask, area)) == ["no_runs"]      # clipped on the print: a run
    r = _read(TINY, good, solids_off=1e-6)
    off = cc.evaluate(wet, r, 0.0, _books(r), mask, area)
    assert _failed(off) == ["solids"] and off["score"] == pytest.approx(0.9)
    r = _read(TINY, good)
    assert _failed(cc.evaluate(wet, r, 0.06e-6, _books(r), mask, area)) == ["overspray"]  # 6 % of the solids left
    assert _failed(cc.evaluate(wet, r, 0.0, _books(r, in_box_off=1e-15), mask, area)) == ["dust"]
    r = _read(TINY, good, landed_frac=0.99, laid=1e-10)                                  # 1 % mist, 2 % left the box
    assert _failed(cc.evaluate(wet, r, 0.02e-6, _books(r, laid=1e-10), mask, area)) == ["supply"]


SINKS = ("to_tank", "lost", "overspray", "mist", "caught", "lumped")      # host_api.SurfaceFlow.read's cross ledger


@pytest.mark.parametrize("sink", SINKS)
def test_the_supply_check_reads_every_sink(sink: str) -> None:
    """DC5's cross ledger: sent + swept + withdrawn = supplied + pending + each column water leaves by."""
    read = {k: np.zeros(1) for k in ("supply_swept", "supply_withdrawn", "pending_volume",
                                     *(f"supply_{c}" for c in SINKS))}
    read.update(supply_sent=np.array([1.0]), supplied_volume=np.array([0.5]), **{f"supply_{sink}": np.array([0.5])})
    assert ps_.supply_drift(read) == 0.0


def test_gym_anythings_spray_progress() -> None:
    mask, area = TINY.print_mask(), TINY.cell_area
    wet = np.zeros(TINY.cells)
    full = cc.spray_progress(np.where(mask, 35e-6, 0.0), mask, wet, 0.0, 0.0, area)
    assert full["progress"] == 1.0 and full["quality"] == 1.0
    assert cc.spray_progress(np.zeros(TINY.cells), mask, wet, 0.0, 0.0, area)["progress"] == 0.0
    half = cc.spray_progress(np.where(mask, 12.5e-6, 0.0), mask, wet, 0.0, 0.0, area)   # credit 0.5 everywhere
    assert half["progress"] == pytest.approx(1.0 - 0.45 / 0.95, rel=1e-12)
    out = cc.spray_progress(np.where(mask, 35e-6, 0.0), mask, wet, 0.15, 0.0, area)      # 15 % left the box
    assert out["quality"] == pytest.approx(0.8 + 0.2 * 0.5, rel=1e-12)
    patchy = np.where(mask, 60e-6, 0.0)
    patchy[mask.any(axis=1).nonzero()[0][:20]] = 20e-6                                  # ratio 3: finish 0.5
    assert cc.spray_progress(patchy, mask, wet, 0.0, 0.0, area)["quality"] == pytest.approx(0.8 + 0.2 * 0.5, rel=1e-12)
    pooled = cc.spray_progress(np.where(mask, 35e-6, 0.0), mask, np.where(mask, 2.0 * pm.RUN_H, 0.0), 0.0, 0.0, area)
    assert pooled["progress"] == 0.0                                                     # all of it pooled


@pytest.mark.compile_heavy
@pytest.mark.timeout(600)
def test_scripted_clear_coat_on_a_print_crop() -> None:
    res = cc.run(cc.SMALL, cc.Passes(), settle=2.0)
    w = res["worlds"][0]
    assert w["succeeded"] and w["score"] == 1.0, {k: c for k, c in w["checks"].items() if c["ok"] is False}
    assert all(abs(v) <= 1e-12 for v in w["raw"].values()), w["raw"]                    # the books, unrounded
    b = w["books_m3"]
    assert b["outside"] == 0.0 and b["dust_laid"] > 0.0 and abs(b["in_valve_m3"]) <= 1e-12 * b["sent"]
    assert b["mist_solids"] > b["dust_laid"]                                           # the rest lands on the box
    assert b["on_print"] + b["on_bed"] + b["off_chart_and_mist"] == pytest.approx(b["sent"], rel=1e-9)
    lo, mean, hi = w["film"]["gauge_um"]
    assert 30.0 < lo and hi < 45.0 and hi / lo < 1.1 and w["film"]["evaporated_ul"] > 0.0
