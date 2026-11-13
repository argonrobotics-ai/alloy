# SPDX-License-Identifier: Apache-2.0
"""pad-trowel v4's staged margin lines predicted with the blade's own fill (WP15's ruling 2, 2026-11-09).

WP4's replay of the v4 L4 row ([internal note] notes/wp4-trowel/v4_fill_replay.md): every run that beat its prediction by
more than 2 N was a later run in its line, cutting paste the margin blade had laid itself. The blade's store fills
the lows under it while it bears on paste (``stores_fill``): a wedge behind a ramped set-down, and a layer at a stage's
level wherever it bore on a ridge. One line on a planted ridge (1.2 mm over the level, 1 cm wide, off the line's
centre), on paste as old as the v3 / v4 rows' 23.0 N run at t 2111 s, the blade carrying what a lift's earlier lines
gave it: read with ``pad_trowel_stages.read_line``, two stages, and each staged run's need measured on the sim's own
surface by their ride rule (``RideWatch``) is at most 2 N over its prediction. The cut-only predictor (v4 as
first pushed) runs it in one stage and predicts its second run, 26.1 N through the wedge, at 0 N.
"""

from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace

import numpy as np
import pytest

from alloy.tasks import concrete_materials as CM
from alloy.tasks import concrete_tub as TUB
from alloy.tasks import pad_finish_margin as MG
from alloy.tasks import pad_ride as RD
from alloy.tasks import pad_run as PR
from alloy.tasks import pad_trowel as T
from alloy.tasks import pad_trowel_lifts as LF
from alloy.tasks import pad_trowel_stages as ST
from alloy.tasks.concrete_pad_start import AGES_21C, CELL, FORM_H
from alloy.tasks.concrete_patch import CARRY_CAP
from alloy.tasks.concrete_tools import HEIGHT, Stroke, ToolProgram

OLD_S = 2111.27            # s: the v3 / v4 rows' 23.0 N run (1.0 mm into the blade's own set-down wedge)
RIDGE = (0.0012, 0.010, 0.012)   # m: the planted ridge's height over the level, its width, its offset from the line
CARRY = 0.15               # the blade's load at the line, of its cap (22.5 cm³; the v4 row's blade carried <= 39.7)
NX, NY, CELL = 170, 40, 0.002
MEASURED = (2.25, 13.48, 14.45, 14.45)   # N: the sim's needs on this line (law below, CPU, 2026-11-09)


def _ridge(X: np.ndarray, Y: np.ndarray, mid: float, hi: float) -> np.ndarray:
    return (np.abs(Y - RIDGE[2]) <= RIDGE[1] / 2) & (X >= mid - 0.02)


def _strip(X: np.ndarray, Y: np.ndarray, mid: float, hi: float) -> np.ndarray:
    """The blade's whole width from just past its first set-down's leading edge to just inside its first run's end."""
    return (X >= mid + MG.HA + 0.004) & (X <= hi + MG.HA - 0.004)


def _line(plant=_ridge, mid: float = -0.02, hi: float = 0.06, t0: float = 0.0,
          over: float = RIDGE[0]) -> SimpleNamespace:
    """The chart (paste everywhere, aged as at ``OLD_S``), ``over`` over the level where ``plant`` says (the ridge),
    and one line of the margin blade (pad-finish's two runs along x at y 0, the first from ``mid`` to ``hi``) planned
    from ``t0`` as the lifts plan it (``Knife.lift``)."""
    task = T.task()
    age = AGES_21C[T.TASK] * CM.masco_factor(task.temperature) + OLD_S
    x, y = -0.17 + CELL * (np.arange(NX) + 0.5), -0.04 + CELL * (np.arange(NY) + 0.5)
    f = {"x": x, "y": y, "cell": CELL, "paste_mask": np.ones((NX, NY)), "age": age, "h": np.zeros((NX, NY))}
    level = task.level_min
    runs = MG.two_runs(0.0, mid, -0.10, hi, 0.0)
    X, Y = np.meshgrid(x, y, indexing="ij")
    h0 = np.full((NX, NY), level)
    h0[plant(X, Y, mid, hi)] = level + over
    knife = ST.Knife(SimpleNamespace(touch=0.0), f, [Stroke(0.0, (*runs[0][0], FORM_H + 0.03), mode=HEIGHT)], 0.0)
    done = knife.lift(1, runs, level, t0)
    d = knife.downs[0]
    read = ST.read_line(d, h0, f, T.tau_b(task, age + d.t))
    return SimpleNamespace(task=task, f=f, h0=h0, knife=knife, d=d, read=read, done=done)


def _staged(read: ST.LineRead) -> list[float]:
    return [n for p in read.predicted if p is not None for n in p]


def test_the_fill_lays_a_wedge_and_a_layer_and_the_prediction_sees_them() -> None:
    """Host pin of the fill rule (``knife_fill``) on the planted ridge. The first run, set down over the ridge and
    ramped down, bears on it and lays the blade's whole width behind its trailing end up to the height it came down from
    (the wedge), and up to its level wherever else it bears (the layer); cut alone (``run_need``'s ``h_out``) leaves
    both at the level below. A blade held over everything lays nothing. The line reads two stages, each run predicted
    no more than 2 N under the sim's measured need (``MEASURED``, the law below); cut alone reads one stage and
    predicts its second run at 0 N."""
    ln = _line()
    pred = _staged(ln.read)
    assert ln.read.count == 2 and len(pred) == len(MEASURED), (ln.read.count, pred)
    assert all(m - p <= 2.0 for m, p in zip(MEASURED, pred, strict=True)), pred
    lv1, up1 = ln.read.stages[-2]
    assert lv1 > ln.d.level and up1 > 0.0
    sole = dataclasses.replace(ln.d.slots[-1][0].sole, level=lv1, touch=up1)
    laid, cut = ln.h0.copy(), ln.h0.copy()
    ST.knife_fill(sole, laid, ln.f)
    RD.run_need(sole, cut, ln.f["x"], ln.f["y"], ln.f["paste_mask"], 1e5, CELL, h_out=cut)
    X, Y = np.meshgrid(ln.f["x"], ln.f["y"], indexing="ij")
    under = np.abs(Y) <= MG.HC + 1e-9                                       # the blade's columns (run_need's)
    back = sole.start[0] - MG.HA                                            # the set-down's trailing edge
    behind = under & (X > back) & (X < back + MG.RAMP)                      # leave under it on the ramp
    ends = under & (np.abs(X - sole.end[0]) <= MG.HA + 1e-9)                # under it at its end (v6: its lift)
    after = under & (X > back + MG.RAMP) & (X < sole.end[0] - MG.HA - 1e-9)
    never = ~under | (X < back - 1e-9) | (X > sole.end[0] + MG.HA + 1e-9)
    assert laid[behind].max() >= ln.d.level + 0.5 * RIDGE[0] and cut[behind].max() == ln.d.level
    assert after.any() and np.all(laid[after] == lv1) and cut[after].min() == ln.d.level
    assert np.all(laid[ends] == ln.d.level + RIDGE[0])                      # up to the ridge its front meets last
    assert np.array_equal(laid[never], ln.h0[never])
    clear, held = ln.h0.copy(), dataclasses.replace(sole, level=ln.d.level + RIDGE[0] + 0.001, touch=0.0)
    ST.knife_fill(held, clear, ln.f)
    assert np.array_equal(clear, ln.h0)


def test_every_run_sets_down_over_what_the_runs_before_it_leave() -> None:
    """WP15's ruling 3 (v5): a strip planted past the line's set-down (the read has nothing over the level under it)
    and read in two stages. The stage above slides into the strip, bears, and lays its level under the set-down; the
    lower stage's first run is set down ``CLEAR`` over that (touch > 0, not on the read), and every run of the line,
    replayed as the predictor replays it, presses nothing at its set-down. On the read's touch (v4: 0 there) it would
    press about 4 cm³ (the v4b row's 9 lower-stage presses, 0.03-0.56 cm³ each)."""
    ln = _line(_strip)
    r, d, f = ln.read, ln.d, ln.f
    assert r.count == 2 and MG.touch(d.runs[0], d.level, ln.h0, f) == 0.0
    h = ln.h0.copy()
    for j, st in enumerate(r.stages):
        for i, sr in enumerate(d.slots[-1] if st is not None else ()):
            sole = dataclasses.replace(sr.sole, level=st[0], touch=r.touches(j)[i])
            assert RD.run_need(sole, h, f["x"], f["y"], f["paste_mask"], 1e5, CELL).pressed_m3 == 0.0, (j, i, sole)
            ST.knife_fill(sole, h, f)
    above = r.stages[-2][0]
    assert r.touches(ST.S_MAX - 1)[0] == pytest.approx(above - d.level + MG.CLEAR, abs=1e-12)


LAYER = 0.000106           # m over the level: the v5 row's probe at 2097.64 s, the stage above's layer on the last 4
                           # columns ahead of the first run's front edge


def test_the_end_lay_covers_every_row_a_lifting_blade_can_leave_uncut() -> None:
    """END (v6; WP15 ruling 5): the blade moves SPEED x EXCHANGE_DT per exchange (0.5 mm, a quarter cell), so only the
    row entering as it starts to lift can stand uncut under it; the end lay covers that row and the next."""
    assert ST.END >= math.ceil(ST.SPEED * PR.EXCHANGE_DT / CELL - 1e-9) + 1
    assert T.PadTrowel().exchange_dt == PR.EXCHANGE_DT


def test_a_lifting_blade_lays_its_end_footprint_and_the_next_run_sets_down_over_it() -> None:
    """v6 (WP15's ruling 4, the v5 row's probe): at 2097.64 s the first run of the -0.061 line's lowest stage had 4
    columns (6-8 mm) of the stage above's layer, 0.106 mm over its level, still ahead of its front edge; 40 ms later
    it stood at its end with its sole rising (+0.116 mm) and had laid 0.70 cm³ of its load over its whole footprint at
    0.105 mm, the set-down its second run pressed (0.696 cm³ at 2098.28 s). The same here on a line whose second run
    starts at its first's end: the end footprint is laid up to what the first run meets last, the second run is set
    down ``CLEAR`` over it and presses nothing; at the level (v5's rule there) it would press the probe's volume."""
    task = T.task()
    age = AGES_21C[T.TASK] * CM.masco_factor(task.temperature) + OLD_S
    x, y = -0.17 + CELL * (np.arange(NX) + 0.5), -0.04 + CELL * (np.arange(NY) + 0.5)
    f = {"x": x, "y": y, "cell": CELL, "paste_mask": np.ones((NX, NY)), "age": age, "h": np.zeros((NX, NY))}
    level, mid, hi = task.level_min, -0.06, 0.065                         # the probe's line: 0.125 m from mid to end
    runs = MG.two_runs(0.0, mid, -0.12, hi, 0.0)
    assert runs[1][0] == runs[0][1]                                         # the second run starts at the first's end
    X, Y = np.meshgrid(x, y, indexing="ij")
    h0 = np.full((NX, NY), level)
    h0[(X > hi + MG.HA - 4 * CELL) & (X <= hi + MG.HA + 1e-9)] = level + LAYER   # the last 4 columns it meets
    knife = ST.Knife(SimpleNamespace(touch=0.0), f, [Stroke(0.0, (*runs[0][0], FORM_H + 0.03), mode=HEIGHT)], 0.0)
    knife.lift(1, runs, level, 0.0)
    d = knife.downs[0]
    r = ST.read_line(d, h0, f, T.tau_b(task, age + d.t))
    assert r.count == 1 and r.stages[-1] == (level, ST.SET_DOWN_CLEAR)        # nothing over its level: ruling 7
    first = dataclasses.replace(d.slots[-1][0].sole, level=level, touch=ST.SET_DOWN_CLEAR)
    h = h0.copy()
    ST.knife_fill(first, h, f)
    ends = (np.abs(X - hi) <= MG.HA + 1e-9) & (np.abs(Y) <= MG.HC + 1e-9)
    assert np.all(h[ends] == level + LAYER) and np.array_equal(h[~ends], h0[~ends])
    laid = float((h - level)[ends].sum()) * CELL * CELL
    assert laid == pytest.approx(0.696e-6, rel=0.05), laid                 # the probe: 0.696 cm³ over the level there
    assert r.touches(ST.S_MAX - 1)[1] == pytest.approx(LAYER + MG.CLEAR, abs=1e-12)
    second = dataclasses.replace(d.slots[-1][1].sole, level=level, touch=r.touches(ST.S_MAX - 1)[1])
    assert RD.run_need(second, h, x, y, f["paste_mask"], 1e5, CELL).pressed_m3 == 0.0
    at_level = RD.run_need(dataclasses.replace(second, touch=0.0), h, x, y, f["paste_mask"], 1e5, CELL).pressed_m3
    assert at_level == pytest.approx(laid, rel=1e-9)


class _Watch(T.RideWatch):
    """Their ride rule before each staged run, on the run as each world's read set it down (``line_z``'s soles)."""

    def __init__(self, ln: SimpleNamespace, flow, *soles: dict[int, tuple[float, float]]) -> None:
        runs = tuple((t, s) for t, s in ln.knife.timed if soles[0][id(s)][0] < ln.d.level + 0.01)
        super().__init__(ln.task, [ln.f] * len(soles), flow, runs)
        self.soles = soles

    def run_for(self, run: RD.SoleRun, world: int) -> RD.SoleRun:
        level, touch = self.soles[world][id(run)]
        return dataclasses.replace(run, level=level, touch=touch)


def _sim(ln: SimpleNamespace, progs: list[ToolProgram], carry: float):
    """The line's rig on the CPU, one world per program: the real margin blade (its scraping store at their carry
    cap, loaded with ``carry`` of it) on the film at ``ln.h0``, the carriage on each world's program."""
    from alloy.plugins.harness.scene import TestScene, build_simulation
    from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow
    from alloy.tasks.concrete_tools import ToolCarriage

    blade = dataclasses.replace(MG.blade(), store=dataclasses.replace(TUB.store(), initial=carry))
    flow = SurfaceFlow(charts=(PlaneChart("pad", cells=(NX, NY)),), material=PR.surface_film(ln.f, ln.task.temperature),
                       inertia=False, max_depth=0.025, u_cap=PR.U_CAP, tools=(blade,), tool_speed=ST.SPEED)
    each = {} if len(progs) == 1 else {"world_programs": tuple((p,) for p in progs)}
    carriage = ToolCarriage(programs=(progs[0],), omega=PR.CARRIAGE_OMEGA, omega_rot=PR.CARRIAGE_OMEGA, flow=flow,
                            **each)
    body = PR.ToolBody(MG.BODY, TUB.TROWEL["half"], TUB.TROWEL["mass"], progs[0].keys[0].pos)
    scene = TestScene(build=PR.build_scene(ln.f, [body]), ground=False, worlds=len(progs), dt=PR.EXCHANGE_DT)
    sim = build_simulation(scene, [carriage, flow], "cpu")
    base = np.full((1, NX * NY), FORM_H + ln.task.level_min - T.GAP)
    flow.set_input("base", np.repeat(base, len(progs), axis=0).astype(np.float32))
    flow.set_depth(np.repeat(FORM_H + ln.h0.reshape(1, -1) - base, len(progs), axis=0))
    return sim, flow


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(900)
def test_every_staged_run_needs_at_most_2_n_over_its_prediction_on_the_blades_own_fill() -> None:
    """WP15's ruling 2, point 4 (CPU, 1 world, ~1 min): the line above with the real margin blade (its scraping store
    at their carry cap, loaded with ``CARRY``) on the film, the read planned up front (the same keys the lift writes,
    the bitwise law). Their ride rule on the sim's surface before each staged run: measured minus predicted <= 2 N for
    every run, among them a second run through a ramped set-down's wedge and a lower stage over the layer the stage
    above laid; the line reads two stages; the blade drops nothing and no set-down presses paste."""
    ln = _line()
    soles = ST.line_z(ln.d, ln.read)[1]
    sim, flow = _sim(ln, [ST.with_reads(ln.knife.keys, [ln.d], [ln.read])], CARRY)
    assert float(np.asarray(flow.read()["tool_load"]).sum()) == pytest.approx(CARRY * CARRY_CAP, rel=1e-12)
    watch = _Watch(ln, flow, soles)
    PR.step_watch(sim, flow, int(math.ceil((ln.done + 0.1) / PR.PERIOD)), hooks=watch.hooks())
    measured = [e["need_n"] for e in watch.log]
    pred = _staged(ln.read)
    assert len(measured) == len(pred) == 2 * ln.read.count, (measured, pred)
    worst = max(m - p for m, p in zip(measured, pred, strict=True))
    assert worst <= 2.0, list(zip(measured, pred, strict=True))
    assert ln.read.count == 2 and min(measured[1:]) > 10.0, measured      # the wedge and the layer were cut
    assert float(np.asarray(flow.read()["tool_dropped"]).sum()) == 0.0
    assert all(e["pressed_m3"] == 0.0 for e in watch.log), [e["pressed_m3"] for e in watch.log]   # ruling 3


END_FRONT = 0.129          # m: the first run's front edge at its end, on a cell centre (the row's east lines)
END_CARRY = 4.57e-6        # m3: the blade's load at the line's start (the probe's 40 ms before that lift)
LEAD = 2.0e-5              # s: the first run's end key this far before an exchange


def _last_columns(X: np.ndarray, Y: np.ndarray, mid: float, hi: float) -> np.ndarray:
    """The last 4 columns under the first run's end footprint (the probe at 2097.64 s: still ahead of its front)."""
    return (X > hi + MG.HA - 3.5 * CELL) & (X < hi + MG.HA + 0.5 * CELL)


def _end_line(t0: float = 0.0) -> tuple[SimpleNamespace, float]:
    """The fast end-lay law's line (0.125 m to the first run's end, where the second starts) moved 0.5 mm, its
    front edge onto ``END_FRONT``, with ``LAYER`` on ``_last_columns``, planned from ``t0``; and its first run's end
    key [s]."""
    ln = _line(_last_columns, END_FRONT - MG.HA - 0.125, END_FRONT - MG.HA, t0, LAYER)
    return ln, ln.knife.keys[ln.d.slots[-1][0].keys[-2]].t


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(1200)
def test_the_sims_lift_lays_the_end_footprint_and_only_v6s_set_down_clears_it() -> None:
    """WP15's ruling 5, should-fix 2 (CPU, 2 worlds, 3-4 min): the fast end-lay law's line in the sim, v5's rule as the
    mutant world. The rig is the v5 row's at 2097.68 s (its probe): the real margin blade loaded with 4.57 cm³, the
    stage above's layer (``LAYER``) on the last 4 columns under the first run's end footprint, and the two things that
    let a column stand uncut into the lift. The first run's front edge ends on a cell centre (the row's east lines),
    and its end key falls on an exchange (``LEAD`` before it; the row's key, 2097.67517 s, and its exchange,
    2097.675 s, are one float32 on the carriage's clock). The blade is then 0.02 mm short of its end as the lift
    starts and 0.04 mm past it an exchange later, its sole 0.146 mm up: the last column comes under a rising sole,
    which bears on it and lays its carry over the whole end footprint.

    What holds (measured 2026-11-09; world 0 on v6's read, world 1 on v5's: the read with ``END`` = 0; the first run
    is bitwise one in both):
    - the lift lays the end footprint: before the second run every cell of it stands 0.106 mm over the level,
      0.7055 cm³ (the row: 0.696); the blade's load falls 4.659 -> 3.949 cm³ over the lift, the 0.7099 cm³ the
      footprint gains; nothing moves outside it and nothing is dropped;
    - v6: the second run is set down 1.106 mm over the level (``LAYER`` + ``CLEAR``) and presses 0 on the ride
      watch; its runs need 2.58 and 2.58 N of the 20 N hold;
    - v5's rule: set down at the level, it presses 0.7055 cm³, the laid layer (the row's press: 0.696 cm³), and fails
      the watch.

    Not claimed: that a lift always lays. Per-exchange traces of this rig lay only with the front edge ending 0 to
    0.03 mm short of a cell centre (0.05 mm short or 0.03 mm past: the column is cut at the level before the lift)
    and the end key within about 0.12 ms before an exchange (phases 0.95, 0.99 and 1 of it lay; 0.04, 0.76 and 0.90
    do not). 0.09 and 0.106 mm layers are laid whole; under a 0.39 mm layer the blade lays to its sole, 0.138 mm.
    v6's model lays up to the top of whatever stands on a run's last ``END`` rows: a bound (on the row it
    over-predicts 4 of the 5 set-downs it changes, each 1.5 mm higher at the cost of a ramp)."""
    dt, j = PR.EXCHANGE_DT, ST.S_MAX - 1
    ln, k_end = _end_line((-_end_line()[1] - LEAD) % dt)
    assert 0.0 < math.ceil(k_end / dt) * dt - k_end < 2.0 * LEAD and ln.d.runs[1][0] == ln.d.runs[0][1]
    assert np.isclose(ln.f["x"], END_FRONT, rtol=0.0, atol=1e-12).any()
    v6 = ln.read
    with pytest.MonkeyPatch.context() as m:          # v5's rule: the next set-down read on a surface with no end lay
        m.setattr(ST, "END", 0)
        v5 = ST.read_line(ln.d, ln.h0, ln.f, T.tau_b(ln.task, ln.f["age"] + ln.d.t))
    assert v6.count == v5.count == 1 and v6.stages == v5.stages and v6.stages[j] == (ln.d.level, 0.0)
    assert v6.touches(j)[1] == pytest.approx(LAYER + MG.CLEAR, abs=1e-12) and v5.touches(j)[1] == 0.0
    sim, flow = _sim(ln, [ST.with_reads(ln.knife.keys, [ln.d], [r]) for r in (v6, v5)], END_CARRY / CARRY_CAP)
    watch = _Watch(ln, flow, *(ST.line_z(ln.d, r)[1] for r in (v6, v5)))
    hooks, snaps, checks = watch.hooks(), [], []

    def snap() -> None:
        snaps.append((PR.surface(flow, (NX, NY)), np.asarray(flow.read()["tool_load"]).sum(axis=1)))

    for t, _ in watch.runs:                           # each run's check, world by world; the surface at the second's
        check = hooks[int(t / PR.PERIOD)]
        hooks[int(t / PR.PERIOD)] = lambda check=check: (snap(), check(), checks.append(list(watch.last)))
    before = int(k_end / PR.PERIOD) - 1               # 20-40 ms before the lift: the blade 4-8 mm short of its end
    assert before not in hooks
    hooks[before] = snap
    PR.step_watch(sim, flow, int(math.ceil((ln.done + 0.1) / PR.PERIOD)), hooks=hooks)
    (_, _), (hb, lb), (ha, la) = snaps
    X, Y = np.meshgrid(ln.f["x"], ln.f["y"], indexing="ij")
    ends = (np.abs(X - ln.d.runs[0][1][0]) <= MG.HA + 1e-9) & (np.abs(Y) <= MG.HC + 1e-9)
    over = (ha - ln.d.level)[:, ends]
    laid = over.sum(axis=1) * CELL * CELL
    drop, gain = lb - la, (ha - hb)[:, ends].sum(axis=1) * CELL * CELL
    assert np.array_equal(ha[0], ha[1]) and la[0] == la[1]                 # one first run in both worlds
    moved = "no lay: the rig's phase window moved (docstring, Not claimed), not a v6 defect"
    assert np.abs(over - LAYER).max() < 1e-6, (moved, over.min(), over.max())   # the whole end footprint
    assert laid[0] == pytest.approx(0.696e-6, rel=0.05), laid              # the row's laid layer
    assert np.allclose(drop, gain, rtol=1e-6, atol=0.0), (lb, la, gain)    # the load it lost is what the footprint
    assert np.abs(ha - hb)[:, ~ends].max() < 1e-9                          # gained; nothing moved outside it
    assert drop[0] == pytest.approx(0.71e-6, rel=0.02) and lb[0] == pytest.approx(4.66e-6, rel=0.01), (lb, la)
    pressed = [[r.pressed_m3 for r in c] for c in checks]
    assert pressed[0] == [0.0, 0.0] and pressed[1][0] == 0.0, pressed      # v6: no set-down presses paste
    assert pressed[1][1] == pytest.approx(laid[1], rel=1e-6), pressed      # v5's rule presses the laid layer
    assert watch.ok().tolist() == [True, False] and watch.pressed[0] == 0.0, (watch.need, watch.pressed)
    needs = [[r.need_n for r in c] for c in checks]
    assert 2.0 < needs[1][0] and max(map(max, needs)) < 3.0 < T.HOLD, needs     # 2.58 N: v6's runs cut the layer
    assert float(np.asarray(flow.read()["tool_dropped"]).sum()) == 0.0


def test_the_row_books_the_blades_load_its_sole_and_every_run_beating_its_prediction() -> None:
    """WP15's ruling 2, points 2-3: the margin blade's load is its ``tool_load`` column; a predicted run's sole is read
    when the blade reaches the run's worst point (set down, then ``at_m`` at ``SPEED``), joining the hooks being
    stepped; the row lists every run that beats its prediction (worst first), with its level and that sole."""
    from alloy.plugins.surface_flow.stores_fill import ON_BASE

    assert ST.ON_BASE == pytest.approx(float(ON_BASE), rel=1e-6)
    w = LF.LiftWatch.__new__(LF.LiftWatch)
    w.flow = SimpleNamespace(read=lambda: {"tool_load": np.array([[1e-6, 2e-5], [0.0, 3e-5]])})
    assert w.margin_load().tolist() == [2e-5, 3e-5]
    w.sole, w.live, e = (lambda: np.array([-2.1e-4, -1.9e-4])), {}, {"at_m": 0.12}
    w._sample(e, 10.0)
    i = int((10.0 + 0.12 / ST.SPEED) / PR.PERIOD)
    assert list(w.live) == [i] and "sole_w" not in e
    w.live[i]()
    assert e["sole_w"] == [-2.1e-4, -1.9e-4]
    log = [dict(t=1.0, need_w=[5.0, 1.0], pred_w=[1.0, 1.0], level_w=[-2e-4, -2e-4], at_m=0.1, sole_w=[-3e-4, -2e-4]),
           dict(t=2.0, need_w=[9.0, 2.0], pred_w=[None, 4.0], level_w=[-2e-4, -2e-4], at_m=0.0),
           dict(t=3.0, need_w=[0.0, 0.0])]
    us = LF._unders(log)
    assert [(u["t"], u["world"], u["excess_n"]) for u in us] == [(1.0, 0, 4.0), (1.0, 1, 0.0), (2.0, 1, -2.0)]
    assert us[0]["sole"] == -3e-4 and us[2]["sole"] is None and [u for u in us if u["excess_n"] > LF.UNDER_N] == us[:1]
