# SPDX-License-Identifier: Apache-2.0
"""pad_trowel: their pad-trowel scripted on a surface-flow chart (alloy/tasks/pad_trowel.py, pad_ride.py,
pad_window.py).

Fast tests (numpy): their ride rule one row at a time, the plan under it and refused over it, the ride watch failing
a world on the sim's surface, the lanes keep the blade's ends off the arcs and the core and sweep it all, each lane
alternates, the cross lanes come last, the window scores every cell its runs reach as the run left it. The slow law runs a south-west corner on the CPU
(ALLOY_CPU_ROUTE=warp)."""

from __future__ import annotations

import math
import os
from types import SimpleNamespace

import numpy as np
import pytest

from alloy.plugins.errors import PluginError
from alloy.tasks import concrete_metrics as M
from alloy.tasks import pad_ride as RD
from alloy.tasks import pad_run as PR
from alloy.tasks import pad_trowel as T
from alloy.tasks import pad_window as PW
from alloy.tasks.concrete_pad_start import FORM_H, INTERIOR, pad_fields

ARC = 0.0095                       # m: their radius lines read the arc out to here at no shift (T:293-300)
SPILL = 0.020                      # m: their roll's end spill band (SPILL_BAND)
CORE = 0.030                       # m: their core starts this far from the forms
CORNER = (0.42, 0.42)              # m: the corner window of the slow law
TAU = 60.0e3                       # Pa: about their bulk yield at the pad-trowel start


@pytest.fixture(scope="module")
def whole() -> dict:
    return pad_fields("pad-trowel", 0)


def _flat(f: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, Y = np.meshgrid(f["x"], f["y"], indexing="ij")
    return np.zeros(X.shape), X, Y


def test_their_ride_rule_reads_the_row_entering_the_blade(whole) -> None:
    h, X, Y = _flat(whole)
    run = RD.SoleRun((-0.4, 0.0), (0.4, 0.0), 0.0, 0.0508, 0.178, 0.0)
    across = np.where(np.abs(X) < 0.01, 0.31e-3, h)                    # a 0.31 mm strip across the blade's way
    r = RD.run_need(run, across, whole["x"], whole["y"], whole["paste_mask"], TAU, 0.002)
    assert r.need_n == pytest.approx(3.0 * TAU * 0.31e-3 * 0.356, rel=0.01) and r.pressed_m3 == 0.0
    along = np.where(np.abs(Y) < 0.01, 1.0e-3, h)                      # 1 mm along it: a row meets 20 mm of it
    r = RD.run_need(run, along, whole["x"], whole["y"], whole["paste_mask"], TAU, 0.002)
    assert r.need_n == pytest.approx(3.0 * TAU * 1.0e-3 * 0.020, rel=0.11) and r.pressed_m3 > 0.0
    high = RD.SoleRun((-0.4, 0.0), (0.4, 0.0), 0.0, 0.0508, 0.178, 0.0, touch=2.0e-3, ramp=0.05)
    r = RD.run_need(high, along, whole["x"], whole["y"], whole["paste_mask"], TAU, 0.002)
    assert r.pressed_m3 == 0.0 and r.need_n < 3.0 * TAU * 1.0e-3 * 0.020   # set down above it, cut while sliding
    toe = np.where((X > 0.4508 - 0.006) & (X < 0.4508), 0.9e-3, h)    # a pile's toe under the stop's leading rows
    held = RD.run_need(run, toe, whole["x"], whole["y"], whole["paste_mask"], TAU, 0.002)
    assert held.need_n == pytest.approx(3.0 * TAU * 0.9e-3 * 0.356, rel=0.01) and 0.79 < held.at_m <= 0.8
    rising = RD.SoleRun((-0.4, 0.0), (0.4, 0.0), 0.0, 0.0508, 0.178, 0.0, touch=4.0e-3, rise=0.02, lift=4.0e-3)
    assert RD.run_need(rising, toe, whole["x"], whole["y"], whole["paste_mask"], TAU, 0.002).need_n == 0.0
    last = np.where((X > 0.4508 - 0.002) & (X < 0.4508), 0.7e-3, h)    # slumped into the last row alone
    held = RD.run_need(run, last, whole["x"], whole["y"], whole["paste_mask"], TAU, 0.002)
    assert held.need_n == pytest.approx(3.0 * TAU * 0.7e-3 * 0.356, rel=0.01)
    cross = RD.SoleRun((-0.4, 0.0), (0.4, 0.0), 0.0, 0.0508, 0.178, 0.0, rise=T.CROSS_RISE, lift=T.CROSS_LIFT)
    assert RD.run_need(cross, last, whole["x"], whole["y"], whole["paste_mask"], TAU, 0.002).need_n == 0.0


def test_a_pass_counts_only_the_travel_within_their_bearing_band() -> None:
    x, y = np.arange(-0.5, 0.5, 0.002) + 0.001, np.array([0.0])
    run = RD.SoleRun((-0.3, 0.0), (0.3, 0.0), 0.0, 0.0508, 0.178, 0.0, touch=4.0e-3, ramp=0.02, rise=0.012,
                     lift=1.2e-3)
    c = RD.pass_credit([run], x, y)[:, 0]
    down, up = 0.02 * (1.0 - RD.BEAR_BAND / 4.0e-3), 0.012 * (1.0 - RD.BEAR_BAND / 1.2e-3)   # travel off the band
    at = {p: c[np.argmin(np.abs(x - p))] for p in (-0.299, 0.0, 0.299)}
    assert at[0.0] == pytest.approx(1.0, abs=1e-12)
    for p, end in ((-0.299, -0.3 + down), (0.299, 0.3 - up)):           # the set-down and the rise, each its own
        xc = x[np.argmin(np.abs(x - p))]
        assert at[p] == pytest.approx((0.0508 - abs(xc - end)) / 0.1016, abs=1e-9)


def test_every_run_of_the_plan_cuts_less_than_the_push_holds_and_a_deep_plan_is_refused(whole, monkeypatch) -> None:
    task = T.PadTrowel()
    for seed in range(3):
        assert 5.0 < T.ride_need(task, pad_fields("pad-trowel", seed)) < T.HOLD
    steep = T.PadTrowel(level0=-1.0e-3, window=CORNER)                 # a first cut 1 mm under the plane: theirs rides
    assert T.ride_need(steep, T.start(steep, 0)) > T.HOLD

    def built(*_a, **_k):
        raise AssertionError("pad_trowel.run built a sim for a plan over the hold")

    monkeypatch.setattr("alloy.plugins.harness.scene.build_simulation", built)
    with pytest.raises(PluginError, match="plough"):
        T.run(steep)                                                    # refused before anything is built


def _flow(surface: np.ndarray) -> SimpleNamespace:
    """A stand-in flow as ``PR.surface`` reads it: film h and base [W, cells], the surface h + base - FORM_H."""
    film, base = np.zeros(surface.shape), surface + FORM_H
    return SimpleNamespace(h=SimpleNamespace(numpy=lambda: film), base=SimpleNamespace(numpy=lambda: base))


def test_the_ride_watch_fails_a_world_on_a_set_down_cut_or_a_run_over_the_hold(whole) -> None:
    """RideWatch before a run on three worlds' surfaces: flat at the level passes; a 5 mm bump under the set-down
    (4 mm above the level) is cut pressed and fails; a 0.5 mm strip across the blade's way needs more than the 20 N
    hold and fails. One hook per run: the plan's runs each get theirs, two starting in one period are refused."""
    task = T.PadTrowel()
    h, X, Y = _flat(whole)
    under = (np.abs(X + 0.4) < 0.0011) & (np.abs(Y) < 0.0011)
    worlds = np.stack([h, np.where(under, 5.0e-3, h), np.where(np.abs(X) < 0.01, 0.5e-3, h)]).reshape(3, -1)
    run = RD.SoleRun((-0.4, 0.0), (0.4, 0.0), 0.0, 0.0508, 0.178, 0.0, touch=T.TOUCH, ramp=T.RAMP, rise=T.RISE,
                     lift=T.TOUCH)
    t = 10.5 * PR.PERIOD
    watch = T.RideWatch(task, [whole] * 3, _flow(worlds), ((t, run),))
    hooks = watch.hooks()
    assert list(hooks) == [10]
    hooks[10]()
    tau = T.tau_b(task, whole["age"] + t)
    assert watch.pressed.tolist() == pytest.approx([0.0, under.sum() * 1.0e-3 * 0.002 ** 2, 0.0], rel=1e-6, abs=0)
    assert watch.need[0] == 0.0 and 0.0 < watch.need[1] < T.HOLD
    assert watch.need[2] == pytest.approx(3.0 * tau * 0.5e-3 * 0.356, rel=0.01) and watch.need[2] > T.HOLD
    assert watch.ok().tolist() == [True, False, False]
    assert len(watch.log) == 1 and watch.log[0]["pressed_m3"] == watch.pressed.max()
    p = T.plan(task, whole)
    assert len(T.RideWatch(task, [whole], _flow(worlds[:1]), p.runs).hooks()) == len(p.runs)
    with pytest.raises(PluginError, match="one control period"):
        T.RideWatch(task, [whole], _flow(worlds[:1]), ((t, run), (t + 0.25 * PR.PERIOD, run))).hooks()


def test_the_runs_keep_the_blade_and_its_spill_off_the_arcs_and_the_core_and_sweep_it_all(whole) -> None:
    task = T.PadTrowel()
    runs = T.lanes(task, whole)
    a, b = T.TROWEL["half"][:2]
    hx, hy = INTERIOR[0] / 2, INTERIOR[1] / 2
    assert [r[2] for r in runs] == [0.0] * 3 + [np.pi / 2] * 2
    for (xa, ya), (xb, yb), yaw in runs:
        ex, ey = (a, b) if yaw == 0.0 else (b, a)                      # the blade's half extents along x and y
        for gap in (xa - ex + hx, hx - (xa + ex), xb - ex + hx, hx - (xb + ex),
                    ya - ey + hy, hy - (ya + ey), yb - ey + hy, hy - (yb + ey)):
            assert gap - SPILL > ARC and gap <= CORE + 1e-9 or gap > CORE + SPILL
    core = M.pad_core(PR.start_grid(whole))
    assert T.covered(task, whole)[core].all()


def test_the_levels_descend_by_the_step_to_the_lowest_and_hold() -> None:
    lv = T.levels(T.PadTrowel())
    want = [1.0e-4, 5.0e-5, 0.0, -5.0e-5, -1.0e-4, -1.5e-4, -2.0e-4] + [-2.5e-4] * 7
    assert len(lv) == 14 and np.allclose(lv, want, rtol=0, atol=1e-15)


def test_each_lane_runs_the_other_way_every_pass_and_the_cross_lanes_come_last_out_from_their_middle(whole) -> None:
    for f, task in ((whole, T.PadTrowel()), (T.start(T.PadTrowel(window=CORNER), 0), T.PadTrowel(window=CORNER))):
        ps = T.passes(task, f)
        assert len(ps) == T.PASSES + T.CROSS_PASSES
        assert all(yaw == 0.0 for _, runs in ps[:T.PASSES] for *_, yaw in runs)
        assert all(yaw != 0.0 and lv == T.LEVEL_MIN for lv, runs in ps[T.PASSES:] for *_, yaw in runs)
        starts: dict = {}
        for _, runs in ps[:T.PASSES]:
            for p0, p1, yaw in runs:
                starts.setdefault((min(p0, p1), yaw), []).append(p0)
        for seq in starts.values():
            assert all(s != t for s, t in zip(seq, seq[1:]))
        cross = [r for r in T.lanes(task, f) if r[2] != 0.0]
        y_stop = INTERIOR[1] / 2 - T.RIM - T.TROWEL["half"][0]          # a cross lane's stops by the long forms
        for k, (_, runs) in enumerate(ps[T.PASSES:]):                   # never set down at a stop: out from the middle
            starts = np.array([a for a, _, _ in runs])
            ends = np.array([b for _, b, _ in runs])
            assert np.all(np.abs(starts[:, 1]) < y_stop - 0.1) and np.allclose(ends[:, 0], starts[:, 0], rtol=0, atol=0)
            if task.window is None:                                     # the whole pad: from y = 0 out to both stops
                assert len(runs) == 2 * len(cross) and np.all(starts[:, 1] == 0.0)
                assert sorted(np.abs(ends[:, 1]).round(12)) == [round(y_stop, 12)] * len(runs)
                assert np.sign(ends[0, 1]) == (-1.0 if k % 2 == 0 else 1.0)
            else:                                                       # a window: from its reach to the south stop
                assert len(runs) == len(cross) and np.all(ends[:, 1] == -y_stop) and np.all(starts[:, 1] < 0.0)


def test_the_program_sets_down_above_each_level_comes_up_at_each_stop_and_ends_on_the_cart(whole) -> None:
    task = T.PadTrowel(passes=3, cross_passes=1)
    p = T.plan(task, whole)
    hz = T.TROWEL["half"][2]
    n = sum(len(runs) for _, runs in T.passes(task, whole))
    assert len(p.runs) == n
    down = [k for k in p.program.keys if k.pos[2] < FORM_H + 0.01]
    lane = [r.yaw == 0.0 for _, r in p.runs]
    assert 0 < sum(lane) < n
    assert len(down) == 4 * n                                           # set-down, ramp end, rise start, stop
    want = sorted({*T.levels(task), task.level_min}, reverse=True)       # the lane passes, then the cross lanes
    assert np.allclose(sorted({r.level for _, r in p.runs}, reverse=True), want, rtol=0, atol=1e-15)
    for i, ((_, r), a) in enumerate(zip(p.runs, lane, strict=True)):
        ks, (rise, lift) = down[4 * i:4 * i + 4], ((T.RISE, T.TOUCH) if a else (T.CROSS_RISE, T.CROSS_LIFT))
        s = np.array([0.0, T.RAMP, r.length() - rise, r.length()])
        dz = np.array([T.TOUCH, 0.0, 0.0, lift])                       # a cross run comes up only CROSS_LIFT
        z = np.array([k.pos[2] - FORM_H - hz for k in ks])
        assert np.allclose([math.dist(r.start, k.pos[:2]) for k in ks], s, rtol=0, atol=1e-12)
        assert np.allclose(z, r.level + dz, rtol=0, atol=1e-12)
        assert np.allclose(r.height(s), z, rtol=0, atol=1e-12)
        assert np.allclose(ks[-1].pos[:2], r.end, rtol=0, atol=1e-12)
    assert p.program.keys[-1].pos == task.cart and p.end > p.program.keys[-1].t


def test_the_split_puts_film_only_from_gap_under_the_lowest_level(whole) -> None:
    task = T.PadTrowel()
    c = T.split(task)(whole)
    top = whole["b"] + whole["tb"] + whole["ts"]
    inside = whole["paste_mask"]
    assert np.allclose(c["base"] + c["film"], np.where(inside, top, whole["b"] + whole["smear"]), rtol=0, atol=1e-12)
    assert c["base"][inside].max() <= FORM_H + task.level_min - T.GAP + 1e-12


def test_a_window_set_into_the_whole_pad_never_fills_a_field_the_layer_lacks(whole) -> None:
    task = T.PadTrowel(window=CORNER)
    f = T.start(task, 0)
    g = PR.start_grid(f)
    g.dwc = None                                                       # a layer without WP6's dwc
    full = PW.embed(g, f, whole)
    assert full.dwc is None and not PW.has_water_states(full)
    g.D = np.full(f["h"].shape, 0.9, np.float32)
    full = PW.embed(g, f, whole)
    i0 = int(np.flatnonzero(np.isclose(whole["x"], f["x"][0]))[0])
    assert full.D[i0, 0] == np.float32(0.9) and full.D[-1, -1] == whole["D"][-1, -1]


def test_the_window_resets_only_what_its_runs_pass_less_than_the_whole_pads(whole) -> None:
    task = T.PadTrowel(window=CORNER)
    f = T.start(task, 0)
    reset = T.window_reset(task, f, whole)
    X, Y = np.meshgrid(f["x"], f["y"], indexing="ij")
    hx, hy = INTERIOR[0] / 2, INTERIOR[1] / 2
    d_real = np.minimum(X + hx, Y + hy)                                # from the window's real forms
    east = max(max(p0[0], p1[0]) for p0, p1, yaw in T.lanes(task, f) if yaw == 0.0) + T.TROWEL["half"][0]
    assert reset[(X > east) & (d_real > CORE + 0.002)].all()           # beyond the lane's reach (the rim: unworked)
    assert reset[(X > east - 0.03) & (X <= east) & (d_real > 0.1)].all()   # by its stop at the open edge
    away = (X < f["x"][-1] - 0.16) & (Y < f["y"][-1] - 0.16 - T.WINDOW_END)   # clear of its runs' stops (the cross
                                                                       # lane's another WINDOW_END in) at its open edges
    assert not reset[f["paste_mask"] & (d_real < CORE + 0.002) & away].any()   # arcs, rim and the core's edge
    north = max(p0[1] for p0, _, yaw in T.lanes(task, f) if yaw == 0.0) + T.TROWEL["half"][1]
    mid = (d_real > 0.16) & (X < east - 0.16) & (Y <= north)            # mid-lane: every pass a whole pass
    assert T.credit(task, f)[mid].min() >= T.PASSES - 1e-9


def test_the_window_scores_every_cell_it_keeps_as_left(whole) -> None:
    task = T.PadTrowel(window=CORNER)
    f = T.start(task, 0)
    reset = T.window_reset(task, f, whole)
    g = PR.start_grid(f)
    g.D, g.a = np.full(g.h.shape, 0.95), np.full(g.h.shape, 2.0e-5)
    X, Y = np.meshgrid(f["x"], f["y"], indexing="ij")
    hx, hy = INTERIOR[0] / 2, INTERIOR[1] / 2
    d_real = np.minimum(X + hx, Y + hy)
    edge_row = f["paste_mask"] & (d_real > CORE) & (d_real < CORE + 0.002) & ~reset
    g.h = np.where(edge_row, 4.0e-3, g.h)                              # a 4 mm pile on the core's outer row
    g.h = np.where(reset & f["paste_mask"], 9.0e-3, g.h)               # and 9 mm where the window is reset
    w = PW.window_score(g, f, whole, reset, M.PadScalars(tools_total=1))
    assert 4.0 <= w.flat_plane_mm < 9.0 and not PW.window_passed(w, 0)
    g.D = np.where(edge_row, 0.10, g.D)                                # the outer row left unworked
    w = PW.window_score(g, f, whole, reset, M.PadScalars(tools_total=1))
    core = M.pad_core(PW.embed(g, f, whole))[PW.window_slices(f, whole)] & ~reset
    assert (edge_row & core).sum() > 100
    assert w.smooth_hard == pytest.approx(1.0 - (edge_row & core).sum() / core.sum(), rel=0, abs=1e-12)


def test_the_window_radius_lines_pass_on_their_start_and_fail_with_the_arcs_filled(whole) -> None:
    task = T.PadTrowel(window=CORNER)
    f = T.start(task, 0)
    g = PR.start_grid(f)
    assert PW.window_radius(PW.embed(g, f, whole), f, whole) >= 0.98
    near = f["paste_mask"] & (f["d_edge"] < 0.008)
    g.h = np.where(near, 0.0, g.h)                                     # the arcs filled level with the form top
    assert PW.window_radius(PW.embed(g, f, whole), f, whole) < 0.5


def test_a_window_passes_on_its_own_scores_and_every_other_criterion_of_theirs() -> None:
    good = PW.WindowScore(smooth_hard=0.96, radius_profile=0.95, flat_straightedge_mm=2.0, flat_plane_mm=2.0)
    assert PW.window_passed(good, 0) and not PW.window_passed(good, 1)               # the trowel not back
    assert not PW.window_passed(PW.WindowScore(0.94, 0.95, 2.0, 2.0), 0)
    wet = {"no_blister": {"passed": True}, "surface_wc": {"passed": False}}
    assert not PW.window_passed(PW.WindowScore(0.96, 0.95, 2.0, 2.0, wet), 0)


@pytest.mark.slow
@pytest.mark.timeout(7200)
def test_troweling_the_south_west_corner_passes_pad_trowel() -> None:
    """The corner's runs at their 2 mm cells: smooth and hard on 95 % of their core in the window, the arcs kept as
    the run left them, flat, the trowel back, their other criteria (dwc, blisters, burn) passed, the film's volume
    exact, no run over their ride rule on the sim's surface and no cut at a set-down."""
    os.environ.setdefault("ALLOY_CPU_ROUTE", "warp")
    res = T.run(T.PadTrowel(window=CORNER))
    w = res.windows[0]
    assert res.success == 1, w
    assert w["criteria"] and "smooth_hard" not in w["criteria"]       # WP6's layer publishes dwc and bl
    assert w["smooth_hard"] >= 0.95 and w["radius_profile"] >= 0.95
    assert w["flat_straightedge_mm"] <= 3.0 and w["flat_plane_mm"] <= 3.0
    assert res.ledger_rel < 1e-12 and res.ride_need_n[0] < T.HOLD and res.pressed_m3[0] == 0.0
