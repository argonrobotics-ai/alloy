# SPDX-License-Identifier: Apache-2.0
"""Where pad-trowel's margin lifts run (``pad_trowel_lifts``; v2 per WP15's rulings, 2026-11-08 ~18:10-18:30Z).

- **Lane lines**, after each lane pass from ``pad_trowel_lifts.FROM_PASS``. At each real end form, strips along y
  over the trowel's stop footprint and the roll toe its runs leave where they rise into the stop: |x| from
  ``stop - a - rise - INBOARD`` out to ``EDGE`` off the form. That is 3 strips per end (the v1 L4 check: 2.4-3.6 mm
  toes at |x| 0.458-0.478, beside the old strips).
- **Bead lines**, in the lift after the last lane pass only. Strips along x over the side beads the last lane round
  leaves where its runs' blade ends fall on paste an earlier run of the round finished. The first cross pass met
  those across its whole blade (57 N on the v1 check).
- Each line is pad-finish's two runs (``pad_finish_margin.two_runs``): the first from the line's middle, the second
  set down at the plane on the first's way and back through its set-down wedge.
- **Rule 1** (WP15): no strip's footprint meets a core cell unless a later trowel pass works that cell at its level
  (``pad_ride.pass_credit``), so the steel trowel is last over every core cell the knife touches. The margin trowel
  leaves its own texture: 92 % of the core cells it was last over were rough on the v1 check. Each line keeps the
  longest stretch where its blade meets no other core cell (``steel_later``). ``knife_last`` checks a whole plan.
  So no lift follows the final pass, and the strips' ends stay inside the cross passes' reach at the level.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np

from alloy.tasks import concrete_metrics as M
from alloy.tasks import pad_finish_margin as MG
from alloy.tasks import pad_ride as RD
from alloy.tasks import pad_trowel as T
from alloy.tasks.concrete_pad_start import INTERIOR

EDGE = 0.010                               # m: the blade stays this far from the forms (their arcs' lines 1.5-9.5 mm)
INBOARD = 0.005                            # m: the lane lines start this far inboard of the trowel's roll toe
BAND = 0.02                                # m: their end spill band beyond a blade's end

Run = T.Run


def core(f: dict[str, Any]) -> np.ndarray:
    """Their core: paste cells more than 30 mm from the forms (``concrete_metrics.pad_core``, their pad_d)."""
    return M.pad_core(SimpleNamespace(paste_mask=np.asarray(f["paste_mask"]) > 0.5, d_edge=f.get("d_edge"),
                                      cell=float(f["cell"])))


def steel_later(task: T.PadTrowel, f: dict[str, Any]) -> list[np.ndarray]:
    """Per pass k: the cells some pass after k works with the trowel at its level (``pad_ride.pass_credit`` > 0)."""
    works = [RD.pass_credit([T.sole_run(task, r, lv) for r in runs], f["x"], f["y"]) > 0.0
             for lv, runs in T.passes(task, f)]
    out: list[np.ndarray] = [np.zeros(0, bool)] * len(works)
    acc = np.zeros((len(f["x"]), len(f["y"])), bool)
    for k in range(len(works) - 1, -1, -1):
        out[k] = acc.copy()
        acc |= works[k]
    return out


def _strips(lo: float, hi: float) -> list[float]:
    """Centres of blade-width strips (2 HC) covering [lo, hi], the last one flush with ``hi``."""
    n = max(1, math.ceil((hi - lo) / (2.0 * MG.HC) - 1e-9))
    return [min(lo + MG.HC + 2.0 * MG.HC * k, hi - MG.HC) for k in range(n)]


def _line(f: dict[str, Any], allowed: np.ndarray, fixed: float, yaw: float, lim: tuple[float, float]) -> list[Run]:
    """pad-finish's two runs along one line (x = ``fixed`` along y for yaw pi/2, y = ``fixed`` along x for yaw 0),
    over the longest stretch of blade-centre positions within ``lim`` whose footprint meets no core cell outside
    ``allowed``; none when there is no such stretch."""
    along_y = yaw != 0.0
    s, c = (f["y"], f["x"]) if along_y else (f["x"], f["y"])
    band = np.abs(c - fixed) <= MG.HC + 1e-9                      # the cells across the blade
    bad2 = core(f) & ~allowed
    bad = (bad2[band, :] if along_y else bad2[:, band].T).any(axis=0)  # per position along the line
    cand = np.unique(np.clip(np.concatenate([s, list(lim)]), *lim))
    ok = np.array([not bad[np.abs(s - p) <= MG.HA + 1e-9].any() for p in cand])
    best, run_start = (0, -1), None
    for i, v in enumerate([*ok, False]):
        if v and run_start is None:
            run_start = i
        elif not v and run_start is not None:
            if i - 1 - run_start > best[1] - best[0]:
                best = (run_start, i - 1)
            run_start = None
    if best[1] < best[0] or cand[best[1]] - cand[best[0]] < 1e-9:
        return []
    lo, hi = float(cand[best[0]]), float(cand[best[1]])
    return MG.two_runs(fixed, 0.5 * (lo + hi), lo, hi, yaw)


def lane_lines(task: T.PadTrowel, f: dict[str, Any], allowed: np.ndarray) -> list[Run]:
    """A lane lift's runs: per real end form (the east first, the tub's side), its lines from the inboard one out,
    each along y over the lanes' end spill bands, ``EDGE`` off the long forms."""
    xl = [r for r in T.lanes(task, f) if r[2] == 0.0]
    if not xl:
        return []
    hx, hy = INTERIOR[0] / 2, INTERIOR[1] / 2
    a, b = T.TROWEL["half"][:2]
    lim = (max(min(r[0][1] for r in xl) - b - BAND, -hy + EDGE) + MG.HA,
           min(max(r[0][1] for r in xl) + b + BAND, hy - EDGE) - MG.HA)
    stops = sorted({round(p[0], 9) for r in xl for p in (r[0], r[1]) if abs(abs(p[0]) - (hx - task.rim - a)) < 1e-6},
                   reverse=True)
    out: list[Run] = []
    for xs in stops:
        side = 1.0 if xs > 0 else -1.0
        for x in _strips(abs(xs) - a - task.rise - INBOARD, hx - EDGE):
            out += _line(f, allowed, side * x, math.pi / 2, lim)
    return out


def bead_lines(task: T.PadTrowel, f: dict[str, Any], allowed: np.ndarray) -> list[Run]:
    """The last lane pass's lift, after its lane lines: along x over each side bead its round leaves (a blade end's
    spill band no later run of the round passes), on each side of the pad's middle (the steel's reach decides how far:
    ``allowed``), ``EDGE`` off the end forms."""
    level, runs = T.passes(task, f)[task.passes - 1]
    del level
    b, hx = T.TROWEL["half"][1], INTERIOR[0] / 2
    ys: list[float] = []
    for j, (p0, _, _) in enumerate(runs):
        for s in (1.0, -1.0):
            yc = p0[1] + s * (b + 0.5 * BAND)
            if not any(abs(yc - q[0][1]) <= b for q in runs[j + 1:]):
                ys.append(round(yc, 9))
    xs = f["x"]
    out: list[Run] = []
    for yc in sorted(set(ys)):
        for lim in ((max(float(xs[0]), -hx + EDGE) + MG.HA, -MG.HA), (MG.HA, min(float(xs[-1]), hx - EDGE) - MG.HA)):
            if lim[1] > lim[0]:
                out += _line(f, allowed, yc, 0.0, lim)
    return out


def footprint(run: RD.SoleRun, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Cells a sole sweeps over its run (half_along either side along the motion, half_across across it)."""
    (ax, ay), (bx, by) = run.start, run.end
    if run.along_x():
        return (np.abs(Y - ay) <= run.half_across + 1e-9) & (X >= min(ax, bx) - run.half_along - 1e-9) & (
            X <= max(ax, bx) + run.half_along + 1e-9)
    return (np.abs(X - ax) <= run.half_across + 1e-9) & (Y >= min(ay, by) - run.half_along - 1e-9) & (
        Y <= max(ay, by) + run.half_along + 1e-9)


def knife_last(f: dict[str, Any], runs: Any, tools: dict[int, str]) -> np.ndarray:
    """Rule 1's check of a plan: the core cells some margin run's footprint covers after the last trowel run that
    works them at its level (``pass_credit``; a run's time is when it hangs over its start)."""
    X, Y = np.meshgrid(f["x"], f["y"], indexing="ij")
    steel, knife = np.full(X.shape, -np.inf), np.full(X.shape, -np.inf)
    for t, r in runs:
        if tools.get(id(r), "trowel") == "trowel":
            m = RD.pass_credit([r], f["x"], f["y"]) > 0.0
            steel[m] = np.maximum(steel[m], t)
        else:
            m = footprint(r, X, Y)
            knife[m] = np.maximum(knife[m], t)
    return core(f) & (knife > steel)


__all__ = ["BAND", "EDGE", "INBOARD", "bead_lines", "core", "footprint", "knife_last", "lane_lines", "steel_later"]
