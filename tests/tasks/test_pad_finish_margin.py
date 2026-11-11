# SPDX-License-Identifier: Apache-2.0
"""pad-finish (SC9): the margin trowel's pass over the edger's bead (alloy.tasks.pad_finish_margin) on the host, without
a simulation: where its runs go, the levels it steps down, the plan's cut and ride need against their 20 N hold, and the
steel trowel's ride need on what it leaves."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.tasks import concrete_pad_start as PS
from alloy.tasks import concrete_tub as TUB
from alloy.tasks import pad_edge as PE
from alloy.tasks import pad_finish_margin as MG
from alloy.tasks import pad_finish_stages as SG
from alloy.tasks import pad_trowel as PT

REF = -1.5e-3                              # the darby's plane under the form tops
BEAD, PILE = 3.0e-3, 12.0e-3               # m over the plane: the bead's crest and the south-west corner's pile
EDGE_AGE = 7950.0                          # s: the edger's end on the L4 run (seed 0)


def _bead(f: dict, crest: float = BEAD, pile: float = PILE) -> dict:
    """``f`` with its paste a plane ``REF`` under the form tops, the edger's bead on it (a tent ``crest`` high from 80
    to 98 mm in from every form, its crest at 84 mm) and a cone ``pile`` high in the south-west corner's bead."""
    paste, d = np.asarray(f["paste_mask"], bool), np.asarray(f["d_edge"])
    up = np.where((d >= 0.080) & (d <= 0.098), crest * np.minimum((d - 0.079) / 0.005, (0.098 - d) / 0.014), 0.0)
    x0, _, y0, _ = MG.PR.paste_faces(f)
    X, Y = np.meshgrid(f["x"], f["y"], indexing="ij")
    up = np.maximum(up, pile * np.clip(1.0 - np.hypot(X - x0 - 0.088, Y - y0 - 0.088) / 0.015, 0.0, None))
    s = PS.FORM_H + REF + np.clip(up, 0.0, None)
    return dict(f, ts=np.where(paste, s - f["b"] - f["tb"], f["ts"]), age=EDGE_AGE)


@pytest.fixture(scope="module")
def bead():
    return _bead(PS.pad_fields("pad-finish", 0))


def test_the_runs_keep_the_blade_between_the_edgers_lip_and_the_core(bead):
    """Each side's two runs go out to either corner along the bead, the first from the middle, the second from
    ``2 HA + BEYOND`` past it on the first's way; the blade's band is ``IN`` to ``IN`` + its width from the form (the
    edger's lip ends 77.7 mm in, so its mark stays), and its ends stop ``IN`` short of the side forms (the corners'
    squares are cut by both sides' runs)."""
    x0, x1, y0, y1 = MG.PR.paste_faces(bead)
    assert MG.IN > 2.0 * PE.EDGER["half"][1] + PE.EDGER["lip_w"]
    for k, side in enumerate(MG.sides(bead)):
        for n, (a, b, yaw) in enumerate(side):
            along_x = abs(math.cos(yaw)) > 0.5
            across = a[1] if along_x else a[0]
            near = min(across - (y0 if along_x else x0), (y1 if along_x else x1) - across)
            assert near - MG.HC == pytest.approx(MG.IN, abs=1e-12), k
            e = b[0] if along_x else b[1]
            lo, hi = (x0, x1) if along_x else (y0, y1)
            assert min(e - MG.HA - lo, hi - e - MG.HA) == pytest.approx(MG.IN, abs=1e-12), k
            start = 0.5 * (lo + hi) + n * (2.0 * MG.HA + MG.BEYOND)
            assert (a[0] if along_x else a[1]) == pytest.approx(start, abs=1e-12)
        assert np.sign(side[0][1][0] - side[0][0][0] + side[0][1][1] - side[0][0][1]) == -np.sign(
            side[1][1][0] - side[1][0][0] + side[1][1][1] - side[1][0][1])


def test_each_side_runs_twice_at_the_plane(bead):
    """Every run is one pass at the plane: a side's first sets down ``CLEAR`` over the bead under it and comes down
    over ``RAMP``; its second sets down at the plane, unramped, on paste the first cut there."""
    p = MG.plan([bead], REF, t0=SG.START)
    soles = [s for _, s in p.runs]
    assert len(soles) == 8 and all(s.level == REF for s in soles)
    h = MG.surface(bead)
    for k, side in enumerate(MG.sides(bead)):
        first, second = soles[2 * k], soles[2 * k + 1]
        assert first.touch == pytest.approx(MG.under_start(side[0], h, bead) - REF + MG.CLEAR, abs=1e-12)
        assert first.touch > MG.CLEAR
        assert second.touch == 0.0 and first.ramp == MG.RAMP


def test_the_plan_cuts_the_bead_to_the_plane_within_their_hold(bead):
    """On the bead at the edger's end age: the plan's largest ride need is under their 20 N hold, no set-down is
    pressed into paste, the band is left at the plane, nothing outside the blade's band is cut, the cut is what stood
    over the plane there; then the steel trowel's ride need at its 165 min is under the hold too (it is not on the
    bead: well over it)."""
    p = MG.plan([bead], REF, t0=SG.START)
    h0, left, cell = MG.surface(bead), p.left[0], float(bead["cell"])
    band = MG.band(bead)
    assert 0.0 < p.need_n <= PT.HOLD
    assert p.pressed_m3 == 0.0
    assert float(left[band].max()) <= REF + 1e-12
    np.testing.assert_array_equal(left[~band], h0[~band])
    over = np.clip(h0 - REF, 0.0, None)[band].sum() * cell * cell
    assert p.cut_m3[0] == pytest.approx(over, rel=1e-9)
    trowel = SG.trowel_task(REF)
    after = dict(bead, ts=bead["ts"] + (left - h0), age=PS.AGES_21C["pad-trowel"] * 1.0)
    before = dict(bead, age=after["age"])
    assert PT.ride_need(trowel, after) <= PT.HOLD < PT.ride_need(trowel, before)


def test_the_program_runs_from_the_cart_and_empties_over_the_tub_after_each_side(bead):
    """Keys in time order from the cart and back; a release over the tub after each of the four sides; the timed runs
    (the ride watch's, one per control period at most) apart."""
    p = MG.plan([bead], REF, t0=SG.START)
    keys = p.program.keys
    times = [k.t for k in keys]
    assert all(b > a for a, b in zip(times, times[1:], strict=False))
    assert keys[0].pos == keys[-1].pos == TUB.home(bead) and keys[0].t == SG.START and p.end > times[-1]
    tx, ty, _ = TUB.tub(bead).center
    drops = [k for k in keys if k.release > 0.0]
    assert len(drops) == 4 and all(k.pos[:2] == (tx, ty) for k in drops)
    assert all(b[0] - a[0] >= 0.3 for a, b in zip(p.runs, p.runs[1:], strict=False))


def test_one_program_for_every_world_reads_the_highest_bead(bead):
    """Two worlds, the pile in one: one program, its set-downs from the highest (the pile's world: the same program
    as for that world alone), each world cut to the plane in the band, each world's cut its own."""
    flat = _bead(PS.pad_fields("pad-finish", 0), pile=0.0)
    p2 = MG.plan([flat, bead], REF, t0=SG.START)
    p1 = MG.plan([bead], REF, t0=SG.START)
    assert p2.program.keys == p1.program.keys
    band = MG.band(bead)
    assert float(p2.left[:, band].max()) <= REF + 1e-12
    assert p2.cut_m3[1] == pytest.approx(p1.cut_m3[0], rel=1e-12) and p2.cut_m3[0] < p2.cut_m3[1]


def test_the_blade_turns_in_the_air_not_over_the_paste(bead):
    """Every key low over the pad (under ``LIFT`` + a set-down over the plane's level) has its run's heading as the
    key before it: the blade turns only in the air."""
    p = MG.plan([bead], REF, t0=SG.START)
    keys = p.program.keys
    for a, b in zip(keys, keys[1:], strict=False):
        if b.pos[2] < PS.FORM_H + REF + MG.HZ + MG.LIFT and a.pos[2] < PS.FORM_H + REF + MG.HZ + 0.05:
            assert b.yaw == a.yaw, (a, b)


def test_what_the_knife_lays_is_cut_back_to_the_plane(bead):
    """The plan replayed with the most a loaded knife lays (every cell a run covers filled up to the sole's height
    when it last covers it): no set-down presses into paste, and a later run cuts everything laid over the plane back
    to it. Each side's second run sets down at the plane (on paste the first left there), unramped."""
    p = MG.plan([bead], REF, t0=SG.START)
    h, m, cell = MG.surface(bead), np.asarray(bead["paste_mask"], bool), float(bead["cell"])
    for _, sole in p.runs:
        r = MG.RD.run_need(sole, h, bead["x"], bead["y"], m, MG.tau_b(EDGE_AGE, 21.1), cell, h_out=h)
        assert r.pressed_m3 == 0.0
        MG.lay(sole, h, bead)
    assert float(h[MG.band(bead)].max()) <= REF + 1e-12
    np.testing.assert_array_equal(h, p.left[0])
    seconds = [sole for _, sole in p.runs if sole.level == REF and sole.touch == 0.0]
    assert len(seconds) == 4
    assert p.need_n <= PT.HOLD


def test_the_knife_lays_up_to_the_sole_it_last_passed_under(bead):
    """``last_under``: a cell the blade covers ends at least at the sole's level; on the ramp, where it leaves under
    the trailing end coming down, at the height it came down from; cells the blade never covers are not laid on."""
    x0, _, y0, _ = MG.PR.paste_faces(bead)
    sole = MG.RD.SoleRun((x0 + 0.3, y0 + 0.1), (x0 + 0.6, y0 + 0.1), 0.0, MG.HA, MG.HC, REF, 0.004, MG.RAMP)
    z = MG.last_under(sole, bead)
    X, Y = np.meshgrid(bead["x"] - x0, bead["y"] - y0, indexing="ij")
    near = 0.6 * float(bead["cell"])                                            # the cells nearest a point
    row = np.abs(Y - 0.1) < near
    behind = row & (np.abs(X - (0.3 - MG.HA + 0.5 * MG.RAMP)) < near)         # leaves halfway down the ramp
    assert z[behind].max() == pytest.approx(REF + 0.002, abs=2e-4)
    after = row & (np.abs(X - 0.45) < near)
    assert z[after].max() == pytest.approx(REF, abs=1e-12)
    assert np.isneginf(z[row & (X < 0.3 - MG.HA - 0.002)]).all() and np.isneginf(z[np.abs(Y - 0.1) > 0.03]).all()
    h = np.full(z.shape, REF - 0.001)
    laid = MG.lay(sole, h, bead)
    assert laid > 0.0 and float(h[after].max()) == pytest.approx(REF, abs=1e-12)
