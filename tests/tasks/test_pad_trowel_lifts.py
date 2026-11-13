# SPDX-License-Identifier: Apache-2.0
"""pad-trowel's margin lifts, v2 (alloy/tasks/pad_trowel_lifts.py, pad_trowel_strips.py; WP15's rulings 2026-11-08):
the lines, rule 1, the set-downs read at the lift (in place, replayed by the ride watch) and the books, host-side;
pad-finish's steel-trowel program pinned (it calls pad_trowel.plan, which the lifts hook into)."""

from __future__ import annotations

import hashlib
import math
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from alloy.tasks import concrete_tub as TUB
from alloy.tasks import pad_finish_margin as MG
from alloy.tasks import pad_run as PR
from alloy.tasks import pad_trowel as T
from alloy.tasks import pad_trowel_lifts as LF
from alloy.tasks import pad_trowel_stages as ST
from alloy.tasks import pad_trowel_strips as S
from alloy.tasks.concrete_pad_start import FORM_H, INTERIOR
from alloy.tasks.concrete_patch import CARRY_CAP
from alloy.tasks.concrete_tools import Key, Stroke, ToolCarriage, ToolProgram

T0 = (T.WAIT_AT + 5) * PR.PERIOD


@pytest.fixture(scope="module")
def whole():
    task = T.task()
    f = T.start(task, 0)
    return SimpleNamespace(task=task, f=f, plan=LF.plan(task, f, t0=T0), later=S.steel_later(task, f))


def _digest(p: T.Plan) -> str:
    rows = [f"{k.t:.9f} {k.pos[0]:.9f} {k.pos[1]:.9f} {k.pos[2]:.9f} {k.yaw:.9f} {k.pitch:.9f} {k.force:.6f} {k.mode} "
            f"{k.release:.6f}" for k in p.program.keys]
    runs = [f"{t:.9f} {r}" for t, r in p.runs]
    return hashlib.sha256("\n".join([*rows, *runs, f"{p.end:.9f}"]).encode()).hexdigest()[:16]


def test_pad_finishs_trowel_program_is_unchanged() -> None:
    """pad_finish_stages builds its trowel with pad_trowel.plan (no lifts): its keys and runs as on main before the
    lifts (pinned 2026-11-08 at 38cfac0ad), so SC9's 8-seed rows stand."""
    from alloy.tasks import pad_finish_stages as ST
    from alloy.tasks.concrete_pad_start import pad_fields

    f = pad_fields("pad-finish", 0, 21.1, "still")
    got = {ref: _digest(T.plan(ST.trowel_task(ref), f, t0=ST.START)) for ref in (0.0, -3.0e-4)}
    assert got == {0.0: "6dbe3f54640b2d44", -3.0e-4: "fab1e3d1743d7d2b"}


def test_pad_finishs_sides_are_two_runs_per_line() -> None:
    """pad_finish_margin.sides is built from two_runs (shared with the lifts): the first from the middle, the second
    from 2 HA + BEYOND past it back over the middle."""
    a, b = MG.two_runs(0.3, 0.0, -0.4, 0.4, math.pi / 2)
    assert a == ((0.3, 0.0), (0.3, 0.4), math.pi / 2)
    assert b == ((0.3, 2 * MG.HA + MG.BEYOND), (0.3, -0.4), math.pi / 2)
    assert MG.two_runs(0.1, 0.0, -0.5, 0.05, 0.0)[1][0] == (0.05, 0.1)        # a short line: from its end


def test_the_task_turns_the_lifts_on_and_plain_settings_keep_them_off(whole) -> None:
    assert whole.task.lifts and whole.task.level_min == T.TASK_LEVEL_MIN == -2.0e-4
    plain = T.PadTrowel()
    assert not plain.lifts and plain.level_min == T.LEVEL_MIN
    p = LF.plan(plain, whole.f, t0=T0)
    assert p.margin is None and not p.lifts and p.runs == T.plan(plain, whole.f, T0).runs


def _cells(run: tuple, f, half_along: float = MG.HA, half_across: float = MG.HC) -> np.ndarray:
    X, Y = np.meshgrid(f["x"], f["y"], indexing="ij")
    sole = SimpleNamespace(start=run[0], end=run[1], half_along=half_along, half_across=half_across,
                           along_x=lambda: abs(math.cos(run[2])) > 0.5)
    return S.footprint(sole, X, Y)


def test_the_lane_lines_cover_the_stops_and_the_roll_toes_off_the_forms(whole) -> None:
    """After lane pass 1: at each end form the lines cover |x| from the roll toe (the trowel's rise into the stop) to
    the core's edge, over the lanes' end bands; the inner two lines out to EDGE off the long forms (the corners), the
    outer one only as far as a later pass trowels at its level (rule 1); the blade stays EDGE off every form."""
    task, f = whole.task, whole.f
    runs = S.lane_lines(task, f, whole.later[1])
    X, Y = np.meshgrid(f["x"], f["y"], indexing="ij")
    cover = np.zeros(X.shape, bool)
    for r in runs:
        cover |= _cells(r, f)
    hx, hy = INTERIOR[0] / 2, INTERIOR[1] / 2
    a, b = T.TROWEL["half"][:2]
    toe = hx - task.rim - a - a - task.rise                       # where the lanes' runs rise into the stop
    want = (np.abs(X) >= toe) & (np.abs(X) <= hx - task.rim) & (np.abs(Y) <= 0.36 + MG.HA)
    corners = (np.abs(X) >= toe) & (np.abs(X) <= toe + 4 * MG.HC - 0.01) & (np.abs(Y) <= hy - S.EDGE)
    assert want.sum() > 0 and cover[want].all() and cover[corners & (np.asarray(f["paste_mask"]) > 0.5)].all()
    assert not cover[(np.abs(X) > 0.56) & (np.abs(Y) > 0.4235) & S.core(f)].any()      # rule 1's clip
    assert b > 0
    assert (np.abs(X[cover]) <= hx - S.EDGE + 1e-9).all() and (np.abs(Y[cover]) <= hy - S.EDGE + 1e-9).all()
    assert len(runs) == 12 and all(abs(r[2] - math.pi / 2) < 1e-12 for r in runs)


def test_the_bead_lines_sit_on_the_last_rounds_beads_inside_the_cross_reach(whole) -> None:
    """The last lane round leaves a bead at each blade end no later run of the round passes; inside the core those are
    y -0.061 and +0.188 on seed 0's lanes. The bead lines cover them where the cross lanes' blades reach (only there:
    no steel passes the middle again)."""
    task, f = whole.task, whole.f
    runs = S.bead_lines(task, f, whole.later[task.passes - 1])
    assert sorted({round(r[0][1], 3) for r in runs}) == [-0.061, 0.188]
    b = T.TROWEL["half"][1]
    X = np.meshgrid(f["x"], f["y"], indexing="ij")[0]
    for r in runs:
        cells = _cells(r, f) & S.core(f)                         # outside the core it may go out to EDGE
        assert r[2] == 0.0 and (np.abs(np.abs(X[cells]) - 0.4015) <= b + 1e-9).all()   # the cross blade's cells


def test_rule_1_the_steel_trowel_is_last_over_every_core_cell_the_knife_touches(whole) -> None:
    """WP15's rule 1 on the plan, and the plan refuses a lift that breaks it (v1's lift after the final pass: a knifed
    strip with no steel after it)."""
    p, f = whole.plan, whole.f
    assert not S.knife_last(f, p.runs, p.tools).any()
    t_end = max(t for t, _ in p.runs) + 5.0
    late = RD_sole(((0.4015, 0.38), (0.30, 0.38), 0.0))
    bad = S.knife_last(f, (*p.runs, (t_end, late)), {**p.tools, id(late): "margin"})
    assert bad.sum() > 100
    assert not [k for k, _, _ in p.lifts if k >= whole.task.passes]           # no lift after a cross pass


def RD_sole(run: tuple):                                                      # noqa: N802 - a SoleRun from a tuple
    from alloy.tasks import pad_ride as RD
    return RD.SoleRun(run[0], run[1], run[2], MG.HA, MG.HC, 0.0, 0.0, MG.RAMP)


def test_the_trowel_waits_while_the_margin_trowel_lifts_after_each_lane_pass(whole) -> None:
    """A lift after lane pass FROM_PASS and each later lane pass; no trowel run while one lifts; every margin run inside
    a lift; one ride hook per run of either tool."""
    p, task = whole.plan, whole.task
    assert [k for k, _, _ in p.lifts] == list(range(LF.FROM_PASS, task.passes))
    trowel = [t for t, r in p.runs if p.tools[id(r)] == "trowel"]
    margin = [t for t, r in p.runs if p.tools[id(r)] == "margin"]
    for _, s, e in p.lifts:
        assert not any(s <= t <= e for t in trowel)
    assert all(any(s - 1e-9 <= t <= e for _, s, e in p.lifts) for t in margin)
    assert len(T.RideWatch(task, [whole.f], None, p.runs).hooks()) == len(p.runs)


def test_each_line_is_s_max_stage_slots_of_its_two_runs_as_planned(whole) -> None:
    """Each line: S_MAX slots of (a first run at the nominal touch, a second at touch 0), all at its level as planned;
    each run's keys contiguous after the line's flight start."""
    p, task = whole.plan, whole.task
    keys = p.margin.keys
    margin = [r for _, r in p.runs if p.tools[id(r)] == "margin"]
    assert len(margin) == 2 * ST.S_MAX * len(p.downs)
    for d in p.downs:
        assert len(d.slots) == ST.S_MAX and keys[d.slots[0][0].keys[0] - 1].t == d.t
        for first, second in d.slots:
            assert (first.run, second.run) == d.runs and first.sole.touch == task.touch and second.sole.touch == 0.0
            for sr in (first, second):
                assert list(sr.keys) == list(range(sr.keys[0], sr.keys[-1] + 1))
                assert [keys[j].pos[2] for j in sr.keys] == ST.run_z(d.level, sr.sole.touch, len(sr.keys))


OLD_S = 2111.27          # s: the v3 row's 23.0 N run at 1.0 mm (paste ~35 min older than at the first lift)


def _tau(task: T.PadTrowel, f: dict, t: float) -> float:
    return T.tau_b(task, f["age"] + t)


def _read(d: ST.SetDown, f: dict, depth: float, tau: float) -> ST.LineRead:
    return ST.read_line(d, np.full(np.asarray(f["h"]).shape, d.level + depth), f, tau)


def test_a_line_takes_ceil_depth_over_1_6_mm_stages_down_to_its_level_and_the_rest_hang_clear(whole) -> None:
    """WP15's staging: at least S = ceil(d / 1.6 mm) up to S_MAX (v4 keeps v3's S as the least), the levels stepping
    down to the line's, the unused slots first and clear over the paste; a line deeper than S_MAX x 1.6 mm still ends at
    its level (the ride rule judges it)."""
    d, f = whole.plan.downs[0], whole.f
    for depth, n in ((0.0, 1), (0.001, 1), (0.002, 2), (0.00332, 3), (0.006, 3)):
        r = _read(d, f, depth, _tau(whole.task, f, d.t))
        assert r.depth == pytest.approx(depth, abs=1e-12) and r.count == n
        assert r.stages[:ST.S_MAX - n] == (None,) * (ST.S_MAX - n)
        levels = [st[0] for st in r.stages[ST.S_MAX - n:]]
        assert levels[-1] == d.level and np.all(np.diff(levels) < 0) if n > 1 else levels == [d.level]
        assert max(lv - d.level for lv in levels) <= depth and (n == ST.S_MAX or depth / n <= ST.STAGE + 1e-12)
        zs, soles = ST.line_z(d, r)
        for slot, st in zip(d.slots, r.stages, strict=True):
            if st is None:
                sole_z = min(zs[j] for sr in slot for j in sr.keys) - FORM_H - MG.HZ
                assert sole_z >= d.level + depth + MG.CLEAR and soles[id(slot[0].sole)][1] == 0.0


def test_the_same_read_takes_one_stage_on_young_paste_and_two_on_old(whole) -> None:
    """WP15's v4 ruling: S from the predicted need, not depth alone. A 1.0 mm read (the v3 row's 23.0 N run) is one
    stage on young paste and two on paste as old as t ~ 2111 s, each stage's runs predicted at most HOLD_PLAN; deeper
    reads on old paste still run at most S_MAX, and their prediction is booked for the ride rule to judge."""
    d, f, task = whole.plan.downs[0], whole.f, whole.task
    young, old = _read(d, f, 0.001, _tau(task, f, 0.0)), _read(d, f, 0.001, _tau(task, f, OLD_S))
    assert young.count == 1 and old.count == 2
    for r in (young, old):
        used = [p for p in r.predicted if p is not None]
        assert len(used) == r.count and max(max(p) for p in used) <= ST.HOLD_PLAN
        assert [p is None for p in r.predicted] == [s is None for s in r.stages]
    deep = _read(d, f, 0.006, _tau(task, f, OLD_S))
    assert deep.count == ST.S_MAX and max(max(p) for p in deep.predicted if p is not None) > ST.HOLD_PLAN


def test_every_planned_set_down_clears_the_paste_it_predicts_under_the_blade(whole, monkeypatch) -> None:
    """WP15 ruling 7: every run the plan sets down sits at least ``SET_DOWN_CLEAR`` (20 um: twice the 7.3 um the
    16-world line af3297 pressed, the ruling's floor above it) over the highest paste the plan predicts under the blade
    at its start, on each line's start read and on a read standing exactly at its level (``touch`` 0: af3297's
    presses). Without the clearance the second read sets down at 0 um over the paste."""
    assert ST.SET_DOWN_CLEAR >= max(2.0 * 2.93e-11 / 0.002**2, 2.0e-5)
    real, seen = MG.touch, []

    def spy(run, level, h, f):
        seen.append((level, MG.under_start(run, h, f)))
        return real(run, level, h, f)

    monkeypatch.setattr(MG, "touch", spy)
    f, at_level = whole.f, 0
    for d in whole.plan.downs:
        tau = _tau(whole.task, f, d.t)
        for h in (np.asarray(f["h"], dtype=float), np.full(np.asarray(f["h"]).shape, d.level)):
            stages = [(st[0],) for st in ST.read_line(d, h, f, tau).stages if st is not None]
            seen.clear()
            ups = [u for pair in ST.chain(d, stages, h, f, tau, read_touch=True)[1] for u in pair]
            assert len(ups) == len(seen)
            for (level, top), up in zip(seen, ups, strict=True):
                assert level + up >= top + ST.SET_DOWN_CLEAR - 1e-12, (d.t, level, up, top)
                at_level += int(top == level)
    assert at_level > 0                                     # the touch-0 case was planned and cleared


def test_a_stage_is_predicted_on_the_paste_the_one_above_left(whole) -> None:
    """Stage k's prediction cuts the read clipped at stage k-1's level: a ridge over a flat read needs less in its
    second stage than as one cut, and a stage that cuts nothing predicts nothing."""
    d, f, task = whole.plan.downs[0], whole.f, whole.task
    tau = _tau(task, f, OLD_S)
    h = np.full(np.asarray(f["h"]).shape, d.level + 0.001)
    one = ST.predict(d, [(d.level, 0.0)], h, f, tau)
    two = ST.predict(d, [(d.level + 0.0005, 0.0), (d.level, 0.0)], h, f, tau)
    assert max(two[1]) < max(one[0]) and max(two[0]) < max(one[0])
    assert ST.predict(d, [(d.level + 0.002, 0.0)], h, f, tau) == [(0.0, 0.0)]


def test_the_planned_tools_clear_the_form_tops_along_their_whole_paths(whole) -> None:
    """The forms proxy (pad_forms, WP15's ruling) on the whole pad's plan: the trowel's and the margin trowel's
    commanded paths (runs, set-downs, lifts, flights to the tub and back) stay over the form tops (19.8 / 42.1 mm at
    10 ms)."""
    from alloy.tasks import pad_forms as PF
    from alloy.tasks.concrete_tools import Key

    progs = [whole.plan.trowel.program, whole.plan.margin]
    K = max(len(q.keys) for q in progs)
    table = np.zeros((2, K), dtype=Key.numpy_dtype())
    for r, q in enumerate(progs):
        for j in range(K):
            s = q.keys[min(j, len(q.keys) - 1)]
            table[r, j] = (s.t, s.pos, s.yaw, s.pitch, s.force, s.mode, s.release)
    halves = [T.TROWEL["half"], (MG.HA, MG.HC, MG.HZ)]
    for k in range(2):
        m, i = PF.tool_clearance(PF.path(table[k], len(progs[k].keys), dt=0.05), halves[k], whole.f)
        assert i >= 0 and m > 0.01, (k, m)


def test_the_depth_is_read_along_both_runs_not_under_the_set_down(whole) -> None:
    """The v2 row's worst line: flat under its set-down, a 3.32 mm ridge mid-way along its second run. touch() reads
    nothing there; the footprint read stages it in three."""
    d, f = whole.plan.downs[0], whole.f
    h = np.full(np.asarray(f["h"]).shape, d.level)
    (a, b, _), mid = d.runs[1], None
    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    X, Y = np.meshgrid(f["x"], f["y"], indexing="ij")
    h[(np.abs(X - mid[0]) < 0.01) & (np.abs(Y - mid[1]) < 0.01)] = d.level + 0.00332
    assert MG.touch(d.runs[0], d.level, h, f) == 0.0
    tau = _tau(whole.task, f, d.t)
    assert _read(d, f, 0.0, tau).count == 1 and ST.read_line(d, h, f, tau).count == 3


def test_line_reads_planned_up_front_change_only_their_lines_keys_z(whole) -> None:
    p, task, f = whole.plan, whole.task, whole.f
    reads = [_read(d, f, 0.0015 * (j % 4), _tau(task, f, d.t)) for j, d in enumerate(p.downs)]
    q = LF.plan(task, f, t0=T0, touches=[reads, reads[::-1]])
    assert q.runs is not p.runs and [t for t, _ in q.runs] == [t for t, _ in p.runs]
    lines = {j for d in p.downs for slot in d.slots for sr in slot for j in sr.keys}
    for w, prog in enumerate(q.margin_worlds):
        changed = {j for j, (a, b) in enumerate(zip(p.margin.keys, prog.keys, strict=True)) if a != b}
        assert changed <= lines
        for d, r in zip(q.downs, (reads, reads[::-1])[w], strict=True):
            for j, z in ST.line_z(d, r)[0].items():
                a, b = p.margin.keys[j], prog.keys[j]
                assert b.pos[2] == z and (a.t, a.pos[:2], a.yaw) == (b.t, b.pos[:2], b.yaw)


def test_the_carriage_sets_heights_in_place_and_nothing_else() -> None:
    keys = tuple(Stroke(0.1 * i, (0.1 * i, 0.2, 0.3), yaw=0.5) for i in range(4))
    c = ToolCarriage(programs=(ToolProgram("a", keys), ToolProgram("b", keys)))
    table = np.zeros((2 * 2, 4), dtype=Key.numpy_dtype())
    for row in range(4):
        for j, s in enumerate(keys):
            table[row, j] = (s.t, s.pos, s.yaw, s.pitch, s.force, s.mode, s.release)
    c._table, c._tools, c.keys = table, 2, wp.array(table, dtype=Key, device="cpu")
    ptr = c.keys.ptr
    c.set_heights(1, 1, {2: 0.05})
    got = c.keys.numpy()
    assert c.keys.ptr == ptr and got["pos"][3, 2, 2] == np.float32(0.05)
    assert np.array_equal(np.delete(got["pos"].reshape(-1, 3), 3 * 4 + 2, axis=0),
                          np.delete(table["pos"].reshape(-1, 3), 3 * 4 + 2, axis=0) * 0 +
                          np.delete(np.array([[s.pos for s in keys]] * 4, np.float32).reshape(-1, 3), 3 * 4 + 2,
                                    axis=0))
    with pytest.raises(IndexError):
        c.set_heights(1, 0, {4: 0.0})


class _Flow:
    """PR.surface's inputs (h + base, flat per world) and no stores."""

    def __init__(self, h: np.ndarray) -> None:
        self.h = SimpleNamespace(numpy=lambda: (h + FORM_H).reshape(h.shape[0], -1))
        self.base = SimpleNamespace(numpy=lambda: np.zeros((h.shape[0], h[0].size)))
        self.stores = None


def test_the_ride_watch_replays_the_set_down_as_rewritten_for_each_world(whole) -> None:
    """A 3 mm pile under a line's set-down: world 0 read off it (two stages; v5, WP15's ruling 3: its level's first run
    set down ``CLEAR`` over what the stage above leaves there), world 1 given a flat read (its level's first run set
    down at the level). On the pile as the stage above cuts it, the watch replays each world's own run: world 1
    presses paste, world 0 not. A watch on the planned runs (the old keys) misses world 1's press."""
    p, f, task = whole.plan, whole.f, whole.task
    d = p.downs[0]
    h = np.full((2, *f["h"].shape), d.level)
    under = _cells(d.runs[0], f) & (np.asarray(f["paste_mask"]) > 0.5)
    h[:, under] = d.level + 0.003
    r0 = ST.read_line(d, h[0], f, _tau(task, f, d.t))
    above = r0.stages[-2][0]                                           # the stage above's level
    lw = LF.LiftWatch(task, [f, f], _Flow(np.minimum(h, above)), p)
    lw._record(d, [r0, ST.LineRead(0.0, (None,) * (ST.S_MAX - 1) + ((d.level, 0.0),))])
    sole = d.slots[-1][0].sole
    lw.check([t for t, r in p.runs if r is sole][0], sole)
    assert r0.count == 2 and lw.stage[id(sole)][0] == (d.level, pytest.approx(above - d.level + MG.CLEAR))
    assert lw.last[0].pressed_m3 == 0.0 and lw.last[1].pressed_m3 > 0.0
    assert lw.log[-1]["touch_w"] == [u for _, u in lw.stage[id(sole)]] and lw.log[-1]["tool"] == "margin"
    old = T.RideWatch(task, [f, f], _Flow(h), p.runs)                         # the mutant: planned keys
    old.check(0.0, sole)
    assert old.last[1].pressed_m3 == 0.0


def test_the_margin_trowel_empties_after_each_lift_and_ends_on_its_cart(whole) -> None:
    p, f = whole.plan, whole.f
    keys = p.margin.keys
    assert keys[0].pos == TUB.home(f) and keys[-1].pos == TUB.home(f)
    assert p.trowel.program.keys[-1].pos == whole.task.cart
    releases = [k.t for k in keys if k.release > 0.0]
    assert len(releases) == len(p.lifts)
    for (_, _, e), r, nxt in zip(p.lifts, releases, [s for _, s, _ in p.lifts[1:]] + [math.inf], strict=True):
        assert e < r < nxt                                       # let go over the tub before the next lift
    assert all(b.t >= a.t for a, b in zip(keys, keys[1:]))


def _flow(**over):
    w = 2
    r = {"to_tools_volume": np.full(w, 2e-4), "tool_released": np.full((w, 1), 2e-4), "tool_load": np.zeros((w, 1)),
         "tool_dropped": np.zeros((w, 1)), "ledger": np.full(w, 1.0e-2), "store_ledger": np.zeros(w)}
    r.update(over)
    return SimpleNamespace(stores=object(), read=lambda: r)


def test_the_books_close_only_with_nothing_dropped_or_left_and_no_carry_over_the_cap() -> None:
    v0 = np.full(2, 1.0e-2)
    books, ok = LF._books(_flow(), v0, np.full(2, 1e-4))
    assert ok.all() and books["taken_m3"] == [2e-4, 2e-4]
    assert not LF._books(_flow(tool_dropped=np.array([[1e-7], [0.0]])), v0, np.zeros(2))[1][0]
    assert not LF._books(_flow(tool_load=np.array([[0.0], [1e-6]])), v0, np.zeros(2))[1][1]
    assert not LF._books(_flow(), v0, np.array([CARRY_CAP * 1.01, 0.0]))[1][0]
    assert not LF._books(_flow(ledger=np.array([1.0e-2, 1.0e-2 + 1e-9])), v0, np.zeros(2))[1][1]


RIDGE = 0.0033        # m over a line's level: the v2 row's deepest read (38.2 N as one pair there)


def _plant(flow, f: dict, line: LF.SetDown, world: int) -> None:
    """``RIDGE`` over ``line``'s level across its first run's middle half in ``world``, as film (``set_depth``)."""
    n = f["h"].shape
    film, surf = flow.h.numpy().reshape(-1, *n).astype(np.float64), PR.surface(flow, n)[world]
    (a, b, yaw), m = line.runs[0], np.asarray(f["paste_mask"]) > 0.5
    X, Y = np.meshgrid(f["x"], f["y"], indexing="ij")
    mx, my, c, s = 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), math.cos(yaw), math.sin(yaw)
    bump = (m & (np.abs((X - mx) * c + (Y - my) * s) <= 0.25 * math.dist(a, b))
            & (np.abs((Y - my) * c - (X - mx) * s) <= MG.HC))
    film[world] += np.where(bump, np.maximum(line.level + RIDGE - surf, 0.0), 0.0)
    flow.set_depth(film.reshape(film.shape[0], -1))


def _ridge_before(monkeypatch, line: LF.SetDown, world: int = 0) -> None:
    """Every LiftWatch plants ``_plant``'s ridge in ``world`` at ``line``'s hook, before it is read (if it is): the
    same surface in a live run and in one planned up front."""
    hooks = LF.LiftWatch.hooks

    def planted(self):
        out, i = hooks(self), int(line.t / PR.PERIOD)
        then = out.get(i)

        def hook() -> None:
            _plant(self.flow, self.fs[world], line, world)
            if then is not None:
                then()
        out[i] = hook
        return out
    monkeypatch.setattr(LF.LiftWatch, "hooks", planted)


def _staged_runs_hold(a: LF.Sim, ridged: LF.SetDown) -> None:
    """WP15's staging law on a run's own surface: every line read deeper than one run's share is staged; each of its
    staged runs needs at most the 20 N hold, and the same line as one pair at its level (read then) needs more. The
    ridged line (world 0) is one: S_MAX stages, over the hold as one pair."""
    need = {round(e["t"], 2): e["need_w"] for e in a.watch.log if e.get("tool") == "margin"}
    when = {id(r): round(t, 2) for t, r in a.p.runs}
    staged = []
    for d in a.p.downs:
        for w, r in enumerate(a.watch.reads.get(id(d), [])):
            if r.count < 2:
                continue
            soles = [sr.sole for slot, st in zip(d.slots, r.stages, strict=True) if st is not None for sr in slot]
            worst = max(need[when[id(s)]][w] for s in soles)
            assert worst <= T.HOLD and a.watch.single[id(d)][w] > worst, (d.t, w, worst, a.watch.single[id(d)][w])
            staged.append((round(d.t, 2), w))
    line = next(d for d in a.p.downs if abs(d.t - ridged.t) < 1e-9)
    r = a.watch.reads[id(line)][0]
    assert r.count == ST.S_MAX and a.watch.single[id(line)][0] > T.HOLD, (r, a.watch.single[id(line)])
    assert (round(line.t, 2), 0) in staged, staged


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(7200)
def test_a_set_down_rewritten_at_the_lift_ends_bitwise_as_planned_up_front(monkeypatch) -> None:
    """WP15's condition 2 (Warp CPU route, 2 worlds): a corner window through its first lift, every line read and its
    stages rewritten at the lift, then again with those reads planned up front (per-world programs, no rewrite): the
    same film and tools bit for bit. A rewrite a step late, or one that misses a world, does not match. The first run
    also carries the staging law (``_staged_runs_hold``): world 0's first line gets a ridge ``RIDGE`` deep (the v2
    row's deepest) just before it is read, in both runs, since the window's own lines read under 1.6 mm (the deep
    ones came after pass 5 on the whole pad)."""
    task = T.task(window=(0.20, 0.42), passes=4)          # lift 1: the west end form's lane lines at x -0.478, -0.529
    probe = LF.plan(task, T.start(task, task.seed), t0=T0)
    assert probe.downs, "the window needs a lift"
    until = probe.lifts[0][2] + 1.0
    assert all(d.t > until for d in probe.downs if d.t > probe.lifts[0][2])   # lift 2's lines start after it
    _ridge_before(monkeypatch, probe.downs[0])
    a = LF.simulate(task, worlds=2, until=until)
    planned = [ST.LineRead(0.0, ((d.level, task.touch),) * ST.S_MAX) for d in a.p.downs]   # as built: never read
    reads = [[a.watch.reads.get(id(d), [p] * 2)[w] for d, p in zip(a.p.downs, planned, strict=True)] for w in range(2)]
    _staged_runs_hold(a, probe.downs[0])
    b = LF.simulate(task, worlds=2, until=until, touches=reads)
    for name in ("h", "base"):
        assert np.array_equal(getattr(a.flow, name).numpy(), getattr(b.flow, name).numpy()), name
    assert np.array_equal(a.sim.state.body_q.numpy(), b.sim.state.body_q.numpy())
