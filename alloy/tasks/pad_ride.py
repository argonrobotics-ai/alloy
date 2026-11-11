# SPDX-License-Identifier: Apache-2.0
"""Gym-Anything's ride rule for a flat sole held at a level on a pad (host-side numpy).

Their ``_rides`` (``coupling.py:644-674``): a sliding sole ploughs the paste standing in its way only while the
plough's force, PLOUGH·tau_b over the frontal area it presents, stays within what holds the sole on the paste; past
that the paste lifts the sole and comes over it. The frontal area is the tick's cut over the rows of cells it spans
along the motion. Cells enter a moving footprint a row at a time, so on a sole already at its level only the entering
row is cut (one row); while the sole descends, every footprint cell above it is.

``run_need`` replays one run of a scripted, height-held sole over a surface the way their rule reads it: one cell of
travel per tick, the footprint's cells cut to the sole as it passes, the need of each tick, the largest returned. Its
set-down (not sliding) is outside the rule (a pressed sole cuts only what its bearing lets go, ``coupling.py:659``):
what the held sole would cut there is returned as a volume instead, for the caller to keep at zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

PLOUGH = 3.0                               # their plough factor (params.py:129)


@dataclass(frozen=True)
class SoleRun:
    """One run of a flat rectangular sole moving along its short side: its centre from ``start`` to ``end`` [m, pad
    frame], ``yaw`` (0: moving along x, pi/2: along y), half extents along the motion and across it, and its level
    [m, heights' frame]; set down ``touch`` above the level and brought down to it over the first ``ramp`` of travel,
    and up ``lift`` over the last ``rise``."""

    start: tuple[float, float]
    end: tuple[float, float]
    yaw: float
    half_along: float
    half_across: float
    level: float
    touch: float = 0.0
    ramp: float = 0.0
    rise: float = 0.0
    lift: float = 0.0

    def along_x(self) -> bool:
        return abs(math.cos(self.yaw)) > 0.5

    def length(self) -> float:
        return math.dist(self.start, self.end)

    def height(self, travelled: np.ndarray | float) -> np.ndarray:
        """The sole's level after ``travelled`` [m] of its run."""
        s = np.asarray(travelled, float)
        z = np.full(s.shape, self.level)
        if self.ramp > 0.0:
            z = z + self.touch * np.clip(1.0 - s / self.ramp, 0.0, 1.0)
        if self.rise > 0.0:
            z = z + self.lift * np.clip(1.0 - (self.length() - s) / self.rise, 0.0, 1.0)
        return z


@dataclass
class RunNeed:
    need_n: float                          # the largest force their plough needs over the run [N] (per tau_b given)
    pressed_m3: float                      # what the held sole cuts at its set-down, not sliding [m³]
    cut_m3: float                          # what it cuts over the whole run [m³]
    at_m: float = 0.0                      # travel from the start at the largest need [m]
    depth_m: float = 0.0                   # the deepest cut of that tick [m]
    rows: int = 0                          # the rows that tick's cut spans


def _axes(run: SoleRun, h: np.ndarray, x: np.ndarray, y: np.ndarray, paste: np.ndarray):
    """The surface with the motion along axis 0: (heights, positions along, positions across, the run's ends along,
    its line across). Cells off the paste never cut."""
    hh = np.where(paste, h, -np.inf)
    if run.along_x():
        return hh, x, y, (run.start[0], run.end[0]), run.start[1]
    return hh.T, y, x, (run.start[1], run.end[1]), run.start[0]


def run_need(run: SoleRun, h: np.ndarray, x: np.ndarray, y: np.ndarray, paste: np.ndarray, tau_b: float,
             cell: float, h_out: np.ndarray | None = None) -> RunNeed:
    """Their ride need over ``run`` on the surface ``h`` [m, nx x ny] at the paste's bulk yield ``tau_b`` [Pa].
    ``h_out``, when given, receives the surface the run leaves (cut to the sole where it passed; no roll, no spill)."""
    hh, along, across, (s0, s1), c = _axes(run, h, x, y, paste)
    cols = np.abs(across - c) <= run.half_across + 1e-9
    cur = hh[:, cols].copy()
    sign = 1.0 if s1 >= s0 else -1.0
    steps = max(int(math.ceil(abs(s1 - s0) / cell - 1e-9)), 0)
    travel = np.minimum(np.arange(steps + 1) * cell, abs(s1 - s0))
    zs = run.height(travel)

    def rows(pos: float) -> slice:
        idx = np.flatnonzero(np.abs(along - pos) <= run.half_along + 1e-9)
        return slice(idx[0], idx[-1] + 1) if len(idx) else slice(0, 0)

    first = rows(s0)
    pressed = float(np.clip(cur[first] - zs[0], 0.0, None).sum()) * cell * cell
    cur[first] = np.minimum(cur[first], zs[0])
    out = RunNeed(0.0, pressed, pressed)
    for k in range(1, steps + 1):
        fp = rows(s0 + sign * travel[k])
        cut = np.clip(cur[fp] - zs[k], 0.0, None)
        tot = float(cut.sum())
        if tot <= 0.0:
            continue
        hit = np.flatnonzero(cut.max(axis=1) > 0.0)
        n_rows = int(hit[-1] - hit[0] + 1)
        need = PLOUGH * tau_b * tot * cell / n_rows
        if need > out.need_n:
            out.need_n, out.at_m, out.depth_m, out.rows = need, float(travel[k]), float(cut.max()), n_rows
        out.cut_m3 += tot * cell * cell
        cur[fp] = np.minimum(cur[fp], zs[k])
    if h_out is not None:
        left = hh.copy()
        left[:, cols] = cur
        left = left if run.along_x() else left.T
        h_out[...] = np.where(paste, left, h)
    return out


BEAR_BAND = 3.0e-4                         # m: their _work's bearing band (a cell works within this under the sole)


def pass_credit(runs: list[SoleRun], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Their passes per cell [nx, ny] over ``runs`` by geometry (``coupling.py:1019-1027``: a pass per the bearing
    patch's extent of travel under it, ``passes = ds / b_along``): the travel over which a cell lies under the sole
    while it is within ``BEAR_BAND`` of its level (after its set-down ramp, before its rise), over the sole's length
    along the motion.
    A flat surface at the level is assumed: it says where a run can work, not what it does."""
    X, Y = np.meshgrid(x, y, indexing="ij")
    out = np.zeros(X.shape)
    for r in runs:
        along, across, (s0, s1), c = (X, Y, (r.start[0], r.end[0]), r.start[1]) if r.along_x() else \
            (Y, X, (r.start[1], r.end[1]), r.start[0])
        down, up = (max(1.0 - BEAR_BAND / dz, 0.0) if dz > 0.0 else 0.0 for dz in (r.touch, r.lift))  # off the band
        sign = 1.0 if s1 >= s0 else -1.0
        w0, w1 = sorted((s0 + sign * r.ramp * down, s1 - sign * r.rise * up))
        lo, hi = np.maximum(along - r.half_along, w0), np.minimum(along + r.half_along, w1)
        under = np.abs(across - c) <= r.half_across + 1e-9
        out += np.where(under, np.clip(hi - lo, 0.0, None), 0.0) / (2 * r.half_along)
    return out
