# SPDX-License-Identifier: Apache-2.0
"""The margin trowel's staged lines (WP15's ruling 2026-11-08 ~21:30Z, after the v2 L4 row).

Each late trowel pass leaves a ridge in the end lane strips: up to 3.1 mm high at lift 6's start, cut by its lane lines
mid-run (the v2 row: 14 of 164 margin runs over the 20 N ride hold, 38.2 N at 3.32 mm). A line is now planned as
``S_MAX`` stage slots, each a two-run pair as before (``pad_finish_margin.two_runs``), with every key's time fixed up
front. At the line's lift hook the surface is read (the same privileged sight as the set-down): ``d`` is the highest
paste under the blade along both runs' footprints over the level. The line takes S equal-depth stages stepping down to
its level: v4 (WP15's ruling 2026-11-09 ~03:20Z, after the v3 row's 3 of 566 margin runs at 20.7-23.0 N on 1.0-1.4 mm
single-stage lines: a fixed 1.6 mm per stage ignores the paste stiffening with age, 14.8 -> 23 N/mm) takes the
smallest S of 1..S_MAX whose every run's predicted need (``pad_ride.run_need``, the ride rule's own, at the line's
tau_b) is at most ``HOLD_PLAN``. A line's first run is predicted on the read; every later run, and every lower stage, on
the read clipped at the level above plus what the blade lays (WP15's ruling 2, 2026-11-09, after WP4's replay of the v4
row: all 40 runs that beat their prediction by > 2 N met paste the blade had laid itself). ``knife_fill`` replays a
run's cut and the store's knife fill (``stores_fill``: while it bears on paste it fills the lows under it up to the
level it rests on) from an unlimited carry, the safe bound. If even S_MAX predicts more, it runs S_MAX and the ride
rule judges it. v5 (WP15's ruling 3, after the v4b row's 10 set-down presses, 2.37 cm³: 9 a lower stage's first run
set down at its level over the layer the stage above had laid) sets every run down by ``touch`` on the surface the runs
before it leave, not on the read (``chain``). v6 (WP15's ruling 4, the v5 row's one press, 0.696 cm³ at 2098.28 s:
the first run's lift laid its end footprint up to what its front met last) adds that end lay to ``knife_fill``. The
slots it does not use hang clear over the paste (``CLEAR`` + ``LIFT`` over the footprint's highest cell). Only z
changes, in place (``ToolCarriage.set_heights``), so heights planned up front replay it bitwise.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alloy.tasks import pad_finish_margin as MG
from alloy.tasks import pad_ride as RD
from alloy.tasks import pad_trowel as T
from alloy.tasks.concrete_tools import HEIGHT, Stroke, ToolProgram
from alloy.tasks.concrete_pad_start import FORM_H

STAGE = 0.0016     # m of paste per run: v3's rule (the v2 row's hardest run, 38.2 N for 3.32 mm); v4 predicts the need
HOLD_PLAN = 18.0   # N: a staged run's predicted need may reach this, 2 N under the ride hold (WP15's v4 ruling)
S_MAX = 3          # stage slots per line: ridges to 4.8 mm (3.32 mm the deepest the v2 row read)
SPEED = 0.2        # m/s along a strip (the steel trowel's)
ON_BASE = 5.0e-5   # m: a blade bears on paste this close under its sole (stores_fill.ON_BASE, GA's on_base)
END = 2            # cells of travel: the rows a run meets last meet a rising blade (its lift starts at its end; v6).
#                    The blade moves SPEED x EXCHANGE_DT = 0.5 mm per exchange, a quarter of a 2 mm cell, so only the
#                    row entering at the lift can stand uncut; END covers it and the next (WP15 ruling 5)

Run = T.Run


@dataclass(frozen=True)
class StageRun:
    keys: tuple[int, ...]       # its key indices in the margin program: hover, set-down, ramp end, [run end], lift
    run: Run
    sole: RD.SoleRun            # as planned (its level and nominal touch)


@dataclass(frozen=True)
class SetDown:
    """A line: ``S_MAX`` stage slots of (first run, second run); the lift reads and rewrites all their keys' z."""

    t: float                                       # the line's first flight starts: read and rewrite
    level: float
    slots: tuple[tuple[StageRun, StageRun], ...]

    @property
    def runs(self) -> tuple[Run, Run]:
        return self.slots[-1][0].run, self.slots[-1][1].run


@dataclass(frozen=True)
class LineRead:
    """One world's read of a line: its depth over the level, each slot's (level, first run's set-down over it), its
    two runs' predicted needs [N] (``chain``) and its second run's set-down (v5; none given: at the level); None
    hangs. A read planned up front (not read) predicts nothing."""

    depth: float
    stages: tuple[tuple[float, float] | None, ...]
    predicted: tuple[tuple[float, float] | None, ...] = ()
    seconds: tuple[float | None, ...] = ()

    @property
    def count(self) -> int:
        return sum(s is not None for s in self.stages)

    def touches(self, j: int) -> tuple[float, float]:
        """Slot ``j``'s two runs' set-downs over its level [m]."""
        return self.stages[j][1], self.seconds[j] if self.seconds else 0.0


def under_run(run: Run, h: np.ndarray, f: dict[str, Any]) -> float:
    """The highest paste cell [m] under the blade along the whole run: ``under_start``'s rectangle (in the blade's yaw
    frame) swept from the start to the end, which may run against the yaw (a line's second run)."""
    (a, b, yaw), m = run, np.asarray(f["paste_mask"]) > 0.5
    X, Y = np.meshgrid(f["x"], f["y"], indexing="ij")
    c, s = math.cos(yaw), math.sin(yaw)
    along, across = (X - a[0]) * c + (Y - a[1]) * s, (Y - a[1]) * c - (X - a[0]) * s
    sb, cb = (b[0] - a[0]) * c + (b[1] - a[1]) * s, (b[1] - a[1]) * c - (b[0] - a[0]) * s
    under = (m & (along >= min(0.0, sb) - MG.HA - 1e-9) & (along <= max(0.0, sb) + MG.HA + 1e-9)
             & (across >= min(0.0, cb) - MG.HC - 1e-9) & (across <= max(0.0, cb) + MG.HC + 1e-9))
    return float(h[under].max()) if under.any() else -np.inf


def knife_fill(sole: RD.SoleRun, h: np.ndarray, f: dict[str, Any]) -> None:
    """What a run leaves on ``h`` (in place), one cell of travel per tick as ``pad_ride.run_need`` replays it: the
    footprint cut to the sole and, while the blade bears on paste (a cell within ``ON_BASE`` under its sole), every
    low under it filled up to the highest level it rests on (``stores_fill``'s knife fill) from an unlimited carry.
    At its end (v6, the v5 row's probe at 2097.68 s): the rows it meets in its last ``END`` cells of travel come under
    a blade already lifting, which bears on what stands there and lays its whole end footprint up to that height
    (0.70 cm³ at 0.105 mm in the row, the next run's set-down)."""
    paste = np.asarray(f["paste_mask"]) > 0.5
    hh, along, across, (s0, s1), c = RD._axes(sole, h, f["x"], f["y"], paste)
    cols = np.abs(across - c) <= sole.half_across + 1e-9
    cur, cell = hh[:, cols], float(f["cell"])
    sign = 1.0 if s1 >= s0 else -1.0
    ends = np.abs(along - s1) <= sole.half_along + 1e-9                    # the rows under it at its end
    last = cur[ends & (sign * (along - s1) > sole.half_along - END * cell + 1e-9)]
    top = float(last[np.isfinite(last)].max()) if np.isfinite(last).any() else -np.inf
    travel = np.minimum(np.arange(max(int(math.ceil(abs(s1 - s0) / cell - 1e-9)), 0) + 1) * cell, abs(s1 - s0))
    for s, z in zip(travel, sole.height(travel), strict=True):
        rows = np.flatnonzero(np.abs(along - (s0 + sign * s)) <= sole.half_along + 1e-9)
        if not len(rows):
            continue
        seg = cur[rows[0]:rows[-1] + 1]
        bear = seg >= z - ON_BASE                                 # (off the paste: -inf, never bears or fills)
        np.minimum(seg, z, out=seg)
        if bear.any():
            low = np.isfinite(seg)
            seg[low] = np.maximum(seg[low], seg[bear].max())
    if top > float(sole.height(travel[-1])):                              # it lifts off what it met last
        seg = cur[ends]
        seg[np.isfinite(seg)] = np.maximum(seg[np.isfinite(seg)], top)
        cur[ends] = seg
    left = hh.copy()
    left[:, cols] = cur
    h[...] = np.where(paste, left if sole.along_x() else left.T, h)


def chain(d: SetDown, stages: list[tuple[float, ...]], h: np.ndarray, f: dict[str, Any], tau: float,
          read_touch: bool = False) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Each stage's two runs' needs [N] (``pad_ride.run_need`` at ``tau``) and set-downs [m] for ``stages`` (level,
    first run's set-down), top down: the first run on ``h``, each later run on what the runs before it leave
    (``knife_fill``: cut and laid), each lower stage clipped at the level above. ``read_touch`` (v5): every run set
    down by ``touch`` on the surface before it (the first on ``h``), else the first as given and the second at its
    level."""
    paste, cell = np.asarray(f["paste_mask"]) > 0.5, float(f["cell"])
    out, ups, cur = [], [], np.array(h, dtype=float)
    for k, st in enumerate(stages):
        level = st[0]
        if k:
            cur = np.where(paste, np.minimum(cur, stages[k - 1][0]), cur)
        needs, downs = [], []
        for i, sr in enumerate(d.slots[-1]):
            up = MG.touch(sr.run, level, cur, f) if read_touch else (st[1] if i == 0 else 0.0)
            sole = dataclasses.replace(sr.sole, level=level, touch=up)
            needs.append(RD.run_need(sole, cur, f["x"], f["y"], f["paste_mask"], tau, cell).need_n)
            knife_fill(sole, cur, f)
            downs.append(up)
        out.append((needs[0], needs[1]))
        ups.append((downs[0], downs[1]))
    return out, ups


def predict(d: SetDown, stages: list[tuple[float, float]], h: np.ndarray, f: dict[str, Any],
            tau: float) -> list[tuple[float, float]]:
    """``chain``'s needs [N] for ``stages`` (level, first run's set-down) as given, each second run at its level."""
    return chain(d, stages, h, f, tau)[0]


def read_line(d: SetDown, h: np.ndarray, f: dict[str, Any], tau: float) -> LineRead:
    """The line's stages from one world's surface ``h`` at the paste's bulk yield ``tau`` [Pa]: the smallest S from
    v3's ceil(depth / STAGE) to S_MAX whose runs all predict at most ``HOLD_PLAN`` (else S_MAX; WP15's laws keep v3's
    S as the least: a flat 3.32 mm read on young paste predicts 15.6 N in two), equal-depth levels down to the line's,
    every run set down by ``touch`` at its stage's level on the surface the runs before it leave (``chain``, v5); the
    unused slots first."""
    depth = max(max(under_run(r, h, f) for r in d.runs) - d.level, 0.0)
    for n in range(min(max(math.ceil(depth / STAGE - 1e-9), 1), S_MAX), S_MAX + 1):   # v3's S the least (law 5)
        levels = [d.level + depth * (n - 1 - i) / n for i in range(n)]
        pred, ups = chain(d, [(lv,) for lv in levels], h, f, tau, read_touch=True)
        if max(max(p) for p in pred) <= HOLD_PLAN:
            break
    hang = (None,) * (S_MAX - n)
    return LineRead(depth, hang + tuple((lv, u[0]) for lv, u in zip(levels, ups, strict=True)), hang + tuple(pred),
                    hang + tuple(u[1] for u in ups))


def run_z(level: float, up: float, n: int) -> list[float]:
    """The z of a run's ``n`` keys at ``level`` set down ``up`` over it (``pad_finish_margin.run_keys``)."""
    z = FORM_H + level + MG.HZ
    return [z + up + MG.LIFT, z + up] + [z] * (n - 3) + [z + MG.LIFT]


def line_z(d: SetDown, read: LineRead) -> tuple[dict[int, float], dict[int, tuple[float, float]]]:
    """Key index -> z for every key of the line, and id(planned sole) -> (level, touch) for the ride rule. A hanging
    slot's keys sit ``CLEAR + LIFT`` over the footprint's highest cell (asserted), its soles at that height."""
    zs: dict[int, float] = {}
    soles: dict[int, tuple[float, float]] = {}
    hang = d.level + read.depth + MG.CLEAR + MG.LIFT
    for j, (slot, stage) in enumerate(zip(d.slots, read.stages, strict=True)):
        for i, sr in enumerate(slot):
            level, up = (hang, 0.0) if stage is None else (stage[0], read.touches(j)[i])
            zs.update(zip(sr.keys, run_z(level, up, len(sr.keys)), strict=True))
            soles[id(sr.sole)] = (level, up)
    clear = FORM_H + d.level + read.depth + MG.CLEAR + MG.HZ     # the hanging soles over the footprint's highest cell
    for slot, stage in zip(d.slots, read.stages, strict=True):
        if stage is None:
            low = min(zs[k] for sr in slot for k in sr.keys)
            assert low >= clear - 1e-12, (low, clear)
    return zs, soles


@dataclass
class Knife:
    """The margin trowel's program as it is built: keys, its timed runs, when it is free again (back from the tub)."""

    task: T.PadTrowel
    f: dict[str, Any]
    keys: list[Stroke]
    t: float
    timed: list[tuple[float, RD.SoleRun]] = field(default_factory=list)
    lifts: list[tuple[int, float, float]] = field(default_factory=list)   # (pass, first strip's hook, last lifted)
    downs: list[SetDown] = field(default_factory=list)

    def lift(self, k: int, runs: list[Run], level: float, t: float) -> float:
        """The lines ``runs`` (pairs: a first run, a second) at ``level`` from ``t`` (or once the blade is free), each
        in ``S_MAX`` stage slots, then over the tub to let go. Returns when its last run is lifted."""
        assert len(runs) % 2 == 0, f"lift {k}: {len(runs)} runs, not pairs"
        start = max(t, self.t)
        if start > self.keys[-1].t:                              # it waits where it is (over the tub) till then
            self.keys.append(Stroke(start, self.keys[-1].pos, yaw=self.keys[-1].yaw, mode=HEIGHT))
        self.t, first = start, None
        for j in range(0, len(runs), 2):
            t_line, slots = self.keys[-1].t, []
            for _ in range(S_MAX):
                slots.append(tuple(self._run(run, level, self.task.touch if i == 0 else 0.0)
                                   for i, run in enumerate(runs[j:j + 2])))
                first = self.timed[-2][0] if first is None else first
            self.downs.append(SetDown(t_line, level, tuple(slots)))
        done = self.t
        self.lifts.append((k, first, done))
        self.t = MG.empty(self.keys, self.t, self.f)
        return done

    def _run(self, run: Run, level: float, up: float) -> StageRun:
        """One run as planned (its hover ``up`` over the level), its key indices and timed sole."""
        hover = len(self.keys)
        self.t = MG.fly(self.keys, self.t, (*run[0], run_z(level, up, 5)[0]), yaw=run[2])
        sole = RD.SoleRun(run[0], run[1], run[2], MG.HA, MG.HC, level, up, min(MG.RAMP, math.dist(*run[:2])))
        self.timed.append((self.t, sole))
        ks, self.t = MG.run_keys(run, level, up, self.t, speed=SPEED)
        self.keys += ks[1:]
        return StageRun(tuple(range(hover, len(self.keys))), run, sole)


def with_reads(keys: list[Stroke], downs: list[SetDown], reads: list[LineRead]) -> ToolProgram:
    """The margin program with every line's keys at its read's z (``line_z``), all else as planned."""
    if len(reads) != len(downs):
        raise ValueError(f"touches: {len(reads)} reads for {len(downs)} lines")
    out = list(keys)
    for d, r in zip(downs, reads, strict=True):
        for j, z in line_z(d, r)[0].items():
            out[j] = dataclasses.replace(out[j], pos=(*out[j].pos[:2], z))
    return ToolProgram(MG.BODY, tuple(out))


__all__ = ["END", "HOLD_PLAN", "ON_BASE", "S_MAX", "STAGE", "Knife", "LineRead", "SetDown", "StageRun", "chain",
           "knife_fill", "line_z", "predict", "read_line", "run_z", "under_run", "with_reads"]
