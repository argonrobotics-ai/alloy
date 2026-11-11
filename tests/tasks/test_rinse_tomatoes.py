# SPDX-License-Identifier: Apache-2.0
"""rinse_tomatoes (Gym-Anything rinse-tomatoes): the colander's holes and vessels, the start dust, the frustum test,
the verdict and its score, the plumbing, and one coarse run end to end."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.tasks.rinse_tomatoes import Tomatoes, start_dust
from alloy.tasks.rinse_tomatoes_scene import TOMATO_R, TOMATO_VOL_R, Colander, PrepSink, SteelBowl, tomato_starts
from alloy.tasks.rinse_tomatoes_verdict import Frustum, TomatoesEnd, verdict


def their_holes(c: Colander) -> list[tuple[float, int]]:
    """Gym-Anything's hole list (``ware.py:287-297``) as (z above the bowl's base, count) rings."""
    rings = {}
    g = np.arange(-c.r_base, c.r_base, c.pitch)
    for x in g:
        for y in g:
            if x * x + y * y < (c.r_base - 0.006) ** 2:
                rings[0.0006] = rings.get(0.0006, 0) + 1
    for z in np.arange(0.012, 0.6 * c.height, c.pitch):
        r = float(c.outer_r(z)) - c.wall / 2
        rings[float(z)] = int(2 * math.pi * r / c.pitch)
    return sorted(rings.items())


def test_the_colander_has_their_holes():
    """Same rings and counts as their builder: 308 in the floor, 6 rings up the wall; each ring's Cd·a is its count's,
    at the inner floor's height for the floor ring (theirs: 0.6 mm above the base, inside the 1.2 mm floor)."""
    c = Colander()
    ours, theirs = c.holes(), their_holes(c)
    a = c.hole_cd * math.pi * (0.5 * c.hole_d) ** 2
    assert [round(x[1] / a) for x in ours] == [n for _, n in theirs]
    assert theirs[0][1] == 308 and len(ours) == 7
    assert ours[0][0] == 0.0
    assert [x[0] for x in ours[1:]] == pytest.approx([z - c.bottom for z, _ in theirs[1:]], abs=1e-12)


def test_the_vessels_profiles():
    """Colander: outer r from 95 to 140 mm as s^0.8 over 100 mm, 0.8 mm wall; the bowl's sine profile from 80 to 150
    mm; both rise monotonically and their outer hulls' z rises strictly."""
    c, b = Colander(), SteelBowl()
    assert float(c.outer_r(0.0)) == 0.095 and float(c.outer_r(c.height)) == pytest.approx(0.140, abs=1e-15)
    assert float(b.outer_r(0.0)) == 0.080 and float(b.outer_r(b.height)) == pytest.approx(0.150, abs=1e-15)
    assert float(c.outer_r(0.05)) == pytest.approx(0.095 + 0.045 * 0.5 ** 0.8, abs=1e-15)          # s^0.8
    u = 1.0 - 0.5 ** (1.0 / 0.9)                                                                     # cos(πs/2)
    assert float(b.outer_r(0.055)) == pytest.approx(0.080 + 0.070 * math.sqrt(1.0 - u * u), abs=1e-15)
    for v in (c, b):
        r = np.asarray(v.radius(np.linspace(0.0, v.depth, 200)))
        assert np.all(np.diff(r) > 0.0)
        z = [n[1] for n in v.outer()]
        assert all(q > p for p, q in zip(z[:-1], z[1:]))


def test_tomatoes_start_in_one_layer():
    """Rings of 1 + 6 + 8 at 0, 32 and 64 mm, a tomato's radius plus 1 mm over the floor; none overlap."""
    p = tomato_starts(np.zeros(3))
    assert p.shape == (15, 3) and np.all(p[:, 2] == TOMATO_R + 0.001)
    assert sorted(np.round(np.hypot(p[:, 0], p[:, 1]), 6)) == [0.0] + [0.032] * 6 + [0.064] * 8
    d = np.linalg.norm(p[:, None] - p[None], axis=2) + np.eye(15)
    assert d.min() > 2.0 * TOMATO_R


def test_start_dust_matches_theirs():
    """60 % of 2 mm cells at 1e-3 x U(0.3, 1) kg/m²: a mean of 0.39 g/m², 14.4 mg on 15 tomatoes (their report); the
    worst 1 cm bin (25 cells) about 2.3x the mean."""
    area = 4.0 * math.pi * TOMATO_R ** 2
    load, worst = start_dust(Tomatoes(), 3, area)
    assert load.shape == worst.shape == (3, 15)
    assert load[0].mean() == pytest.approx(0.6 * 0.65e-3, rel=0.03)
    assert load[0].sum() * area * 1e6 == pytest.approx(14.4, rel=0.06)
    assert 1.8 < (worst[0] / load[0]).mean() < 2.6               # their worst 1 cm bin is about 2.3x the mean


def test_the_frustum_margins():
    """1 cm below the base, 2 cm above the rim, 1 cm outside r(z) (linear from base to rim) count as in."""
    f = Frustum(0.0, 0.1, 0.1, 0.14)
    pts = np.array([[0.0, 0.0, -0.0099], [0.0, 0.0, -0.0101], [0.0, 0.0, 0.1199], [0.0, 0.0, 0.1201],
                    [0.1299, 0.0, 0.05], [0.1301, 0.0, 0.05], [0.0, 0.1099, 0.0], [0.0, 0.1499, 0.2]])
    assert f.holds(pts).tolist() == [True, False, True, False, True, False, True, False]


def end(**kw) -> TomatoesEnd:
    base = dict(gauge0=np.full(15, 3.9e-4), gauge=np.full(15, 1.0e-7), kept=np.ones(15, bool), placed=True,
                free_water_ml=3.0, board_gauge=0.0, bowl_gauge=1.0e-7, floor_ml=0.0, lowest_z=0.85,
                faucet_open=False, held=False, books=True)
    base.update(kw)
    return TomatoesEnd(**base)


def test_a_clean_run_scores_one():
    v = verdict(end())
    assert v.success and v.score == 1.0 and v.failed() == []


def test_a_lost_dirty_tomato_and_a_wet_colander():
    """One tomato out of the colander: none_lost fails, its credit is 0 and it costs a unit; one left at 2 µg/cm²
    fails visibly_clean with its reduction; 12 mL fails drained and zeroes p."""
    g = np.full(15, 1.0e-7)
    g[3] = 2.0e-5
    kept = np.ones(15, bool)
    kept[7] = False
    v = verdict(end(gauge=g, kept=kept, free_water_ml=12.0))
    assert v.failed() == ["visibly_clean", "none_lost", "drained"]
    r = lambda x: min(1.0, math.log(39.0 / x) / math.log(39.0))     # noqa: E731 - start 39 µg/cm², limit 1
    c = (13 * r(0.01) + r(2.0)) / 15.0
    assert v.terms["p"] == 0.0 and v.terms["units"] == pytest.approx(1.0 / 15.0, abs=1e-15)
    assert v.score == pytest.approx(0.9 * (c / 2.0 - 1.0 / 15.0), abs=1e-12)


def test_floor_water_alone_breaks_containment():
    """60 mL on the floor fails contained and costs excess 0.2 over 15 tomatoes."""
    v = verdict(end(floor_ml=60.0))
    assert v.failed() == ["contained"] and v.terms["units"] == pytest.approx(0.2 / 15.0, abs=1e-15)


def test_a_running_faucet_zeroes_p():
    """The faucet left running fails water_off_and_hands_free, and their released (that gate, scoring.py:42, 73)
    zeroes p: clean tomatoes score 0.9 x (1 + 0)/2."""
    v = verdict(end(faucet_open=True))
    assert v.failed() == ["water_off_and_hands_free"] and v.terms["p"] == 0.0
    assert v.score == pytest.approx(0.45, abs=1e-12)


def test_a_dirty_bowl_alone_breaks_containment():
    """1.5 µg/cm² in the clean bowl fails contained and costs excess 0.5 over 15 tomatoes."""
    v = verdict(end(bowl_gauge=1.5e-5))
    assert v.failed() == ["contained"] and v.terms["units"] == pytest.approx(0.5 / 15.0, abs=1e-15)


def test_the_books_take_off_what_the_drop_stage_brought():
    """The film books the drop stage's arrivals (landed drops, lumps' wetting) as supplied, which the supply never
    sent: the cross ledger closes only once they are taken off."""
    from types import SimpleNamespace

    from alloy.tasks.rinse_books import Books

    def read():
        r = {"ledger": np.array([0.0]), "h": np.zeros((1, 2)), "supplied_volume": np.array([1.5e-3]),
             "pending_volume": np.array([0.0]), "supply_sent": np.array([3.0e-3]), "supply_swept": np.array([0.0]),
             "supply_withdrawn": np.array([0.0]), "supply_lumped": np.array([2.0e-3])}
        r.update({f"supply_{k}": np.array([0.0]) for k in ("to_tank", "lost", "overspray", "mist", "caught")})
        return r

    drops = SimpleNamespace(read=lambda: {"landed_volume": np.array([2e-4]), "lump_wet_volume": np.array([3e-4])})
    flow = SimpleNamespace(read=read, drops=drops)
    books = Books.start(flow)
    assert books.closed(flow)[0].tolist() == [True]
    flow.drops = None
    assert books.closed(flow)[0].tolist() == [False]


def test_holding_the_colander_zeroes_p():
    """Still held: water_off_and_hands_free fails and p = 0, so clean tomatoes score 0.9 x (1 + 0)/2."""
    v = verdict(end(held=True))
    assert v.failed() == ["water_off_and_hands_free"] and v.terms["p"] == 0.0
    assert v.score == pytest.approx(0.45, abs=1e-12)


def test_contamination_and_faults():
    v = verdict(end(bowl_gauge=1.5e-5, floor_ml=60.0, held=True))
    assert v.failed() == ["contained", "water_off_and_hands_free"]
    assert v.terms["units"] == pytest.approx((0.5 + 0.2) / 15.0, abs=1e-12)
    assert verdict(end(books=False)).score is None
    assert verdict(end(lowest_z=0.2)).failed() == ["intact_and_not_dropped"]


def test_the_plumbing():
    """Sink (drain open), colander and bowl cavities, the faucet's line; the regulator starts closed; the colander's
    pool drips through its holes; 15 floats."""
    st = PrepSink()
    net = st.network()
    assert [n.name for n in net["nodes"]] == ["sink", "colander_cavity", "bowl_cavity", "line"]
    assert st.closed() == ("regulator",) and "drain" in [e.name for e in net["edges"]]
    col = net["pools"][1]
    assert col.holes == st.colander.holes() and col.holes_to is None and col.spill == "colander_pour"
    assert len(net["floats"]) == 15 and net["floats"][0].sphere == TOMATO_VOL_R
    assert 0.0113 / (4.0 / 3.0 * math.pi * TOMATO_VOL_R ** 3) == pytest.approx(977.2, abs=0.1)     # they float


def test_the_soil_gauge_reads_the_film_per_area():
    """A film carrying 1e-5 kg/m² of soil on every cell of a chart reads 1 µg/cm² (the film's species are per area);
    one 1 cm bin at 3e-5 reads 3."""
    from types import SimpleNamespace

    from alloy.tasks.rinse_tomatoes import soil_gauge

    arr = lambda a: SimpleNamespace(numpy=lambda: np.asarray(a))                     # noqa: E731
    xy = np.array([(0.0025 + 0.005 * i, 0.0025 + 0.005 * j, 0.0) for i in range(4) for j in range(4)])
    mass = np.full((1, 16, 1), 1.0e-5)
    tables = SimpleNamespace(cells_of=lambda k: np.arange(16), cpos=arr(xy),
                             cnrm=arr(np.tile([0.0, 0.0, 1.0], (16, 1))), area_np=np.full(16, 2.5e-5))
    flow = SimpleNamespace(tables=tables, sp=SimpleNamespace(mass=arr(mass)))
    assert soil_gauge(flow, 0, 0) == pytest.approx(1.0e-5, rel=1e-12, abs=0)
    mass[0, [0, 1, 4, 5], 0] = 3.0e-5                               # the four 5 mm cells of the bin x, y < 1 cm
    assert soil_gauge(flow, 0, 0) == pytest.approx(3.0e-5, rel=1e-12, abs=0)


@pytest.mark.compile_heavy
@pytest.mark.timeout(900)                 # about 165 s on an idle CPU
def test_tomatoes_rinsed_drained_and_set_in_the_bowl():
    """Coarse charts, one world: the stream aimed at each tomato takes the dust off every one (14.3 mg to under
    1 µg/cm² each), none leaves the colander, the colander drains through its holes (no floor water) and sits in the
    clean bowl, the bowl and board stay clean, every book closes."""
    from alloy.tasks.rinse_tomatoes import run

    st = PrepSink(colander=Colander(ds=8e-3, n_theta=32), bowl=SteelBowl(ds=8e-3, n_theta=32), board_cell=0.02)
    res = run(Tomatoes(scene_=st), worlds=1)
    assert res.success == [True] and res.score == [1.0], res.failed
    assert res.floor_ml[0] < 1.0 and res.free_water_ml[0] <= 10.0 and res.ledger_max < 1e-9      # limit 50 mL


def test_two_micrograms_left_on_a_tomato_fail_it():
    """2 µg of dust left anywhere on a tomato reads 2 µg/cm² (all of it in one 1 cm² bin, at worst) and fails
    visibly_clean; 0.9 µg passes."""
    from alloy.tasks.rinse_tomatoes import tomato_gauge

    area = 4.0 * math.pi * TOMATO_R ** 2
    loads = np.full(15, 1e-12)
    loads[4] = 2.0e-9 / area
    g = tomato_gauge(loads, np.full(15, area))
    assert g[4] == pytest.approx(2.0e-5, rel=1e-12, abs=0)
    assert verdict(end(gauge=g)).failed() == ["visibly_clean"]
    loads[4] = 0.9e-9 / area
    assert verdict(end(gauge=tomato_gauge(loads, np.full(15, area)))).success


def test_the_board_reads_its_dust_twice():
    """Water carrying 0.6 µg/cm² of dust on the board reads 1.2 (their soil + suspended + tracer) and fails
    contained."""
    from types import SimpleNamespace

    from alloy.tasks.rinse_tomatoes import board_gauge

    arr = lambda a: SimpleNamespace(numpy=lambda: np.asarray(a))                     # noqa: E731
    xy = np.array([(0.0025 + 0.005 * i, 0.0025 + 0.005 * j, 0.0) for i in range(4) for j in range(4)])
    tables = SimpleNamespace(cells_of=lambda k: np.arange(16), cpos=arr(xy),
                             cnrm=arr(np.tile([0.0, 0.0, 1.0], (16, 1))), area_np=np.full(16, 2.5e-5))
    flow = SimpleNamespace(tables=tables, sp=SimpleNamespace(mass=arr(np.full((1, 16, 1), 0.6e-5))))
    b = board_gauge(flow, 0)
    assert b == pytest.approx(1.2e-5, rel=1e-12, abs=0)
    assert verdict(end(board_gauge=b)).failed() == ["contained"]


def test_not_judged_gates_are_none_and_a_judged_failure_still_fails():
    """Only the hands part is not judged; the stricter gauges and the floor definition are judged and say so."""
    v = verdict(end())
    assert v.criteria["water_off_and_hands_free"][0] is None
    assert v.success and v.score == 1.0 and v.judged == (7, 8)
    assert v.not_judged() == ["water_off_and_hands_free"]
    assert "stricter" in v.criteria["visibly_clean"][1]["scale"]
    contained = v.criteria["contained"][1]
    assert "stricter" in contained["board"] and "defined differently" in contained["floor"]
    left_open = verdict(end(faucet_open=True))
    assert left_open.failed() == ["water_off_and_hands_free"] and not left_open.success
