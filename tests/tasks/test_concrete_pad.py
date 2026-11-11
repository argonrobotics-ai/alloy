# SPDX-License-Identifier: Apache-2.0
"""task concrete_pad (Gym-Anything's pad-darby): the split of the start into the fixed concrete below the darby's
sole and the film above it, the per-world programs (the tub fill, then the darby; CPU, fast), and the whole task
against their criteria (CPU, slow)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.tasks import concrete_materials as CM
from alloy.tasks import concrete_pad as CP
from alloy.tasks import concrete_tub as TUB


def test_the_paste_above_the_sole_level_is_the_film_and_nothing_is_lost():
    task = CP.PadDarby(factor=1)                                             # their 2 mm grid: forms exact
    f = CP.start(task, seed=0)
    base, film = CP.split(task, f)
    paste = np.asarray(f["paste_mask"]) > 0.5
    cut = CP.FORM_TOP - max(task.below_plane, task.late_below)              # the lowest sole level
    assert np.all(base[paste] <= cut + 1e-12) and np.all(film >= 0.0)
    assert np.all(base == base.astype(np.float32))                           # the film's float32 base input, and
    h = np.asarray(f["h"], np.float64)                                       # the film the rest of their surface
    assert np.array_equal(film, np.maximum(h - base, 0.0)) and np.all(h - base > -1e-8)     # (b + tb + ts) in f64
    forms = np.asarray(f["form_mask"]) > 0.5                                 # the form-top smear once, as film
    assert np.allclose(film[forms], f["smear"][forms], rtol=0, atol=1e-8) and f["smear"][forms].max() > 2e-3
    ground = ~paste & ~(np.asarray(f["form_mask"]) > 0.5)                    # beyond the end forms, at the subgrade
    n = round(CP.GROUND / float(f["cell"]))
    assert ground.sum() == 2 * n * ground.shape[1] and np.all(base[ground] == 0.0)
    assert ground[:n].all() and ground[-n:].all()                            # west and east
    assert np.all(base[forms] == np.float32(CP.FORM_TOP))                     # the forms: base at their top (f32)


def test_the_passes_go_back_and_forth_and_the_last_heads_to_each_worlds_low_spot():
    """Seed 0's low corner is south-west, seed 1's south-east (their layout.py:248). The passes alternate, so each end
    gets passes that arrive with a full roll; the last heads to the low, and every pass leaves the pad over the form
    beyond the end it heads to."""
    task = CP.PadDarby(factor=5)
    x_off = CP.INTERIOR_HALF[0] + 0.038 + CP.DARBY["half"][1] + 0.03
    for seed, want in ((0, 1.0), (1, -1.0)):
        f = CP.start(task, seed)
        cx = TUB.interior(f)[0]
        assert CP.heading(f) == want and (TUB.low_spot(f)[0] - cx) * want < 0.0
        keys = CP.program(task, f)[0].keys
        ends = [k for k in keys if abs(abs(k.pos[0] - cx) - x_off) < 1e-9]
        sides = [-1.0 if k.pos[0] > cx else 1.0 for k in ends[1::2]]           # +1: left over the west form
        assert len(ends) == 2 * task.passes and all(a == -b for a, b in zip(sides, sides[1:]))
        assert sides[-1] == want


def test_the_end_strokes_work_each_end_band_at_the_late_level_inside_the_forms():
    """After the passes, end_strokes there and back at each end, at the late level, the blade's edge END_CLEAR short
    of the end form's inner face: every cell 30-80 mm in from an end form lies under it for all but 5 mm of its
    distance in."""
    task = CP.PadDarby(factor=5)
    f = CP.start(task, 0)
    cx = TUB.interior(f)[0]
    z1 = CP.FORM_TOP - task.late_below + CP.DARBY["half"][2]
    keys = CP.program(task, f)[0].keys
    x_off = CP.INTERIOR_HALF[0] + 0.038 + CP.DARBY["half"][1] + 0.03
    i = max(k for k, key in enumerate(keys) if abs(abs(key.pos[0] - cx) - x_off) < 1e-9)       # the last pass's lift
    ends = [key for key in keys[i + 1:] if abs(key.pos[2] - z1) < 1e-9]
    xs = np.array([key.pos[0] - cx for key in ends])
    lim = CP.EDGE_IN - CP.END_CLEAR
    assert np.abs(xs).max() <= lim + 1e-12                                   # inside the forms at depth
    for e in (-1.0, 1.0):
        side = xs[xs * e > 0]
        outs = np.isclose(side, e * lim, atol=1e-12).sum()
        assert outs == task.end_strokes and np.isclose(side, e * (lim - CP.END_RUN), atol=1e-12).sum() >= outs
    assert CP.END_RUN >= 2 * CP.DARBY["half"][1]                             # a whole blade width in from the limit


def test_the_darby_works_the_whole_core_and_ends_on_the_cart():
    task = CP.PadDarby(factor=5)
    f = CP.start(task, seed=0)
    prog, duration = CP.program(task, f)
    x_off = CP.INTERIOR_HALF[0] + 0.038 + CP.DARBY["half"][1] + 0.03
    cx = TUB.interior(f)[0]                                                  # the pad's, not the chart's centre
    last = max(k for k, key in enumerate(prog.keys) if abs(abs(key.pos[0] - cx) - x_off) < 1e-9)
    keys = prog.keys[:last + 1]                                              # the passes (end strokes after them)
    levels = [CP.FORM_TOP - lv + CP.DARBY["half"][2] for lv in (task.below_plane, task.late_below)]
    work = [k for k in keys if min(abs(k.pos[2] - z) for z in levels) < 1e-9]
    xs = [k.pos[0] for k in work]
    trailing = max(xs) - cx - CP.DARBY["half"][1]
    assert trailing > CP.INTERIOR_HALF[0] - 0.03                           # every core cell crossed by the whole
    assert min(xs) - cx + CP.DARBY["half"][1] < -(CP.INTERIOR_HALF[0] - 0.03)    # width before each turn
    downs = [b for a, b in zip(keys, keys[1:]) if b in work and a.pos[2] > b.pos[2] + 0.05]
    assert len(downs) == task.passes                                       # set down once a pass: one way each
    late = [k for k in work if abs(k.pos[2] - levels[1]) < 1e-9]
    assert len(late) == 2 * task.late_passes                               # level along each late pass
    assert prog.keys[-1].pos == CP.darby_home(f) and duration > prog.keys[-1].t
    ix = cx                                                                 # below the plane: never over a form
    low = [k for k in keys if k.pos[2] < CP.FORM_TOP + CP.DARBY["half"][2] - 1e-9]
    assert low and all(abs(k.pos[0] - ix) <= CP.EDGE_IN - CP.RAMP + 1e-12 for k in low)


def test_each_world_fills_its_own_low_spot_before_the_darby_starts():
    task = CP.PadDarby(factor=5)
    fs = [CP.start(task, s) for s in (0, 1)]
    sets = CP.world_programs(task, fs)
    lows = [TUB.low_spot(f)[:2] for f in fs]
    for (darby, trowel), low, f in zip(sets, lows, fs, strict=True):
        lets = [k.pos[:2] for k in trowel.keys if k.release > 0.0]
        assert all(math.dist(p, low) < TUB.LOW_RADIUS for p in lets)          # this world's low, not world 0's
        n = math.ceil(task.tub_fill * TUB.low_spot(f)[2] / CP.gob_volume(task, f))
        assert len(lets) == CP.trips(task, f) == min(n, TUB.MAX_LOADS) == 9     # 1.5 x the low's ~0.22 L
        assert darby.keys[1].pos == CP.darby_home(f) and darby.keys[1].t > trowel.duration   # waits on the cart
    assert lows[0][0] < 0.0 < lows[1][0]                                    # seed 0 west, seed 1 east
    bare, _ = CP.program(task, fs[0])
    assert CP.world_programs(CP.PadDarby(factor=5, tub=False), fs[:1])[0][0].keys == bare.keys


def test_the_darby_and_trowel_are_theirs():
    """The tools as their task has them: the darby's roll capped with a float's 3 mm lift, the trowel a carrying
    knife store, 8 passes with the last 5 under the plane, both tools from their cart places."""
    task = CP.PadDarby(factor=5)
    fs = [CP.start(task, 0)]
    carriage, flow, layer, _ = CP.plugins(task, fs)
    darby, trowel = flow.config.tools
    assert darby.roll_lift == 0.003 and darby.store is None and trowel.store.scrape and trowel.store.fill == "knife"
    assert (task.passes, task.late_passes, task.late_below) == (8, 5, 0.0015)
    ix, iy = TUB.interior(fs[0])
    assert CP.darby_home(fs[0])[:2] == pytest.approx((ix + 1.15, iy - 1.07), abs=1e-12)
    assert carriage.flow is flow


@pytest.mark.slow
@pytest.mark.timeout(9000)
def test_the_scripted_darby_passes_their_criteria():
    """The whole task at 10 mm cells on the CPU, seed 1 (the south-east low corner, the start that failed first):
    success on their pad-darby criteria, the film's, the blade's and the tub's ledgers closed, every gob landed, and
    the carriage's push (bounding the tools' load) far under the paste's bearing capacity."""
    res = CP.run(CP.PadDarby(factor=5, seed=1), worlds=1)
    v = res.verdicts[0]
    assert v["success"], v["criteria"]
    assert res.ledger_rel < 1e-12 and res.cells_over_depth == 0 and res.landed_l == pytest.approx(res.released_l)
    assert res.store_ledger_m3 < 1e-15 and res.tub_ledger_m3 < 1e-15 and res.lost_l == [0.0]
    bearing = CM.NC * float(CM.Concrete().tau_bulk(float(CP.start(CP.PadDarby(factor=5), 1)["age"])))
    sole = 4 * CP.DARBY["half"][0] * CP.DARBY["half"][1]
    assert 0.0 < res.max_load_n / sole < 0.25 * bearing          # the carriage's push bounds it (not tool_pressure)
