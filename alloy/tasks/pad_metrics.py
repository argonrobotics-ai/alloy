# SPDX-License-Identifier: Apache-2.0
"""Gym-Anything's pad-edge success criteria and progress for Alloy's surface-flow concrete proofs (host-side numpy).

Ports ``concrete_finishing/task.py`` (T:line) ``_perimeter`` (T:260-279), ``_edges`` (T:281-325), ``_pad_edges``
(T:327-348) and the pad-edge branch of ``_pad`` (T:361-362, 393-395), with ``task_set_progress.py`` (TSP:line) pad-edge
quantities and gates (TSP:100-110) and ``_edge_credit`` (TSP:224-249). Thresholds are their EVAL (``params.py``
P:179-184). The shared pad pieces (``PasteGrid``, ``PadScalars``, the verdict assembly, forms/tools/arms) are WP6's
``concrete_metrics``.

Geometry: the pad's own frame (x east, y north, origin at the interior's centre), cells given by their centre
coordinates ``x`` [nx] and ``y`` [ny]. Heights ``h`` are relative to the form-top plane (``concrete_metrics``'s
convention), so the edger's arc tops out at 0. Distances to each form are measured from these coordinates, as theirs
(``Y - PY0`` etc., T:270-277), so a window of the pad (a corner, for a light CPU run) is measured exactly like the
whole pad: its lines on a form it does not contain are empty and skipped, as theirs skip empty lines (T:288-289).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from alloy.tasks.concrete_metrics import (EDGER_RADIUS, LIP_W, RADIUS_TOL_M, Criterion, PadScalars, PasteGrid, Verdict,
                                          _c, _finish, _need, _pad_shared, _tools_gate)
from alloy.tasks.concrete_pad_start import INTERIOR
from alloy.tasks.concrete_progress import Quantity, clip, falloff

EDGE_SHARE, GOUGE_MM, FIELD_CHANGE_MM = 0.95, 2.0, 1.0      # P:180 edge_share, gouge_mm, field_change_mm
LINE_BAND = 0.10              # m: a perimeter line spans the cells within 0.10 m of its form (T:267, 272)
LIP_BAND = 0.0025             # m: the lip's mark is read on the cells within 2.5 mm of the form (T:303)
GOUGE_BAND = 0.080            # m: the edge band a gouge is looked for in, past the lip's slot (T:308)
FIELD_D = 0.150               # m: the field, past the edger's 76 mm track and its overflow (T:52)
EDGE_DRY_BAND = 0.080         # m: edge cells that must not be worked under bleed water (T:345; TSP:108)
COMPACT_MIN = 0.5             # half the normalized fabric: the lip's full credit (T:56; TSP:243)
SIDES = "snwe"


@dataclass(frozen=True)
class Line:
    """One perimeter profile (T:260-279): the cells of a column (s, n) or row (w, e) within 0.10 m of its form."""

    side: str
    h: np.ndarray                 # surface height relative to the form top [m]
    d: np.ndarray                 # distance in from the nearest form (the arc is about it: corners round both ways)
    lip: np.ndarray               # the lip's mark (their ``compact``)
    wet: np.ndarray               # worked under bleed water
    h0: np.ndarray                # the start surface


def form_distances(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Distance [m] in from each form's inner face, per cell [nx, ny] (T:270-277; L:299 for the nearest)."""
    hx, hy = INTERIOR[0] / 2, INTERIOR[1] / 2
    X, Y = np.broadcast_arrays(np.asarray(x, float)[:, None], np.asarray(y, float)[None, :])
    return {"s": Y + hy, "n": hy - Y, "w": X + hx, "e": hx - X}


def pad_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Distance in from the nearest form (their ``pad_d``, negative on the forms)."""
    return np.minimum.reduce(list(form_distances(x, y).values()))


def perimeter(g: PasteGrid, start: PasteGrid, x: np.ndarray, y: np.ndarray) -> list[Line]:
    """Every perimeter line, in their order: per interior column the s and n lines, then per interior row the w and
    e lines (T:267-278). Lines with no cells are kept (empty) so the side bookkeeping matches theirs."""
    _need(g, "C", "wet_worked")
    dist, dmin = form_distances(x, y), pad_distance(x, y)
    hx, hy = INTERIOR[0] / 2, INTERIOR[1] / 2
    on = g.paste_mask
    out: list[Line] = []
    for axis, sides, coords, half in ((0, "sn", np.asarray(x), hx), (1, "we", np.asarray(y), hy)):
        for k in np.flatnonzero((coords > -half) & (coords < half)):
            for side in sides:
                take = (lambda a: a[k]) if axis == 0 else (lambda a: a[:, k])
                d = take(dist[side])
                sel = take(on) & (d >= 0) & (d < LINE_BAND)
                out.append(Line(side, take(g.h)[sel], take(dmin)[sel], take(g.C)[sel], take(g.wet_worked)[sel],
                                take(start.h)[sel]))
    return out


def arc(d: np.ndarray, top: float = 0.0, rad: float = EDGER_RADIUS) -> np.ndarray:
    """The edger's 3/8 in arc at distance d in from the form, top outside the radius (T:291)."""
    return np.where(d < rad, top - (rad - np.sqrt(np.maximum(rad * rad - (rad - d) ** 2, 0.0))), top)


def arc_error(ln: Line, cell: float, top: float = 0.0) -> float:
    """Smallest max |h - arc| over the allowed shifts 0, 1/2, 1 cell, the lip's slot excluded (T:296-302;
    TSP:236-241). inf when no shift has a cell inside the radius."""
    error = math.inf
    for shift in (0.0, 0.5 * cell, cell):
        ds = ln.d - shift
        inr = (ds >= LIP_W) & (ds < EDGER_RADIUS)
        if inr.any():
            error = min(error, float(np.abs(ln.h[inr] - arc(ds[inr], top)).max()))
    return error


def gouge(ln: Line, cell: float) -> float:
    """Deepest cut [m] below both the arc (as cut, or begun one cell later) and the start surface, in the 80 mm band
    past the lip's slot (T:306-317)."""
    band = (ln.d >= LIP_W + cell) & (ln.d < GOUGE_BAND)
    if not band.any():
        return 0.0
    ds = ln.d - cell
    arc1 = np.where((ds >= 0) & (ds < EDGER_RADIUS), arc(ds), np.where(ds < 0, -EDGER_RADIUS, 0.0))
    ref = np.minimum(np.minimum(arc(ln.d), arc1), ln.h0)
    return float(np.clip(ref[band] - ln.h[band], 0.0, None).max())


@dataclass(frozen=True)
class Edges:
    """``_edges`` (T:281-325) over the non-empty lines: shares of lines on the arc and with the lip's mark, and the
    deepest gouge [m]."""

    radius: float
    lip: float
    gouge: float
    lines: int


def edges(lines: list[Line], cell: float) -> Edges:
    full = [ln for ln in lines if len(ln.h)]
    n = max(len(full), 1)
    prof = sum(arc_error(ln, cell) <= RADIUS_TOL_M for ln in full)
    lip = sum(bool(ln.lip[ln.d < LIP_BAND].any()) for ln in full)
    deepest = max((gouge(ln, cell) for ln in full), default=0.0)
    return Edges(prof / n, lip / n, deepest, len(full))


def edge_credit(lines: list[Line], cell: float) -> dict[str, float]:
    """Per-side graded arc and lip credit (TSP:224-249): radius = clip(1 - (error - tol)+ / r), lip =
    clip(max mark within 2.5 mm / 0.5), a line earns 0.5·radius + 0.5·radius·lip; a side its lines' mean."""
    out: dict[str, list[float]] = {s: [] for s in SIDES}
    for ln in lines:
        if not len(ln.h):
            continue
        radius = clip(1.0 - max(arc_error(ln, cell) - RADIUS_TOL_M, 0.0) / EDGER_RADIUS)
        mark = ln.lip[ln.d < LIP_BAND]
        lip = clip(float(mark.max()) / COMPACT_MIN) if len(mark) else 0.0
        out[ln.side].append(0.5 * radius + 0.5 * radius * lip)
    return {s: float(np.mean(v)) if v else 0.0 for s, v in out.items()}


def field_change(g: PasteGrid, start: PasteGrid, x: np.ndarray, y: np.ndarray) -> float:
    """Largest change [m] of the surface more than 150 mm from the forms (T:340-343). 0 when the window holds no
    such cell (their pad always does)."""
    far = g.paste_mask & (pad_distance(x, y) > FIELD_D)
    return float(np.abs(g.h[far] - start.h[far]).max()) if far.any() else 0.0


def edge_wet_cells(g: PasteGrid, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The edge band's cells (T:344; TSP:107)."""
    return g.paste_mask & (pad_distance(x, y) < EDGE_DRY_BAND)


def _measure_edge(g: PasteGrid, p: PadScalars, start: PasteGrid, x: np.ndarray, y: np.ndarray):
    """pad-edge: T:361-362, 327-348, 393-395; TSP:100-110."""
    lines = perimeter(g, start, x, y)
    e = edges(lines, g.cell)
    ch = field_change(g, start, x, y)
    band = edge_wet_cells(g, x, y)
    n_wet = int((g.wet_worked[band] > 0).sum())
    crits: list[Criterion] = [
        _c("radius_profile", e.radius, e.radius >= EDGE_SHARE, EDGE_SHARE),
        _c("lip_passed", e.lip, e.lip >= EDGE_SHARE, EDGE_SHARE),
        _c("no_gouge", e.gouge * 1e3, e.gouge * 1e3 <= GOUGE_MM, GOUGE_MM, "mm"),
        _c("field_untouched", ch * 1e3, ch * 1e3 < FIELD_CHANGE_MM, FIELD_CHANGE_MM, "mm"),
        _c("edge_not_worked_wet", n_wet, n_wet == 0, 0, "cells")] + _pad_shared(g, p)
    both = crits[0].passed and crits[1].passed                  # TSP:103-105: arc and lip both pass
    q = {f"edge_{s}": Quantity(v, 1.0, both, 1.0, "edge") for s, v in edge_credit(lines, g.cell).items()}
    gates = {"tools": _tools_gate(p), "gouge": falloff(e.gouge * 1e3, GOUGE_MM, 6.0),
             "field": falloff(ch * 1e3, FIELD_CHANGE_MM, 5.0),
             "edge_dry": 1.0 - (float((g.wet_worked[band] > 0).mean()) if band.any() else 0.0)}
    return crits, q, gates


def pad_edge(grid: PasteGrid, start: PasteGrid, x: np.ndarray, y: np.ndarray,
             pad: PadScalars | None = None) -> Verdict:
    """pad-edge verdict. ``start`` is the reset state (their ``h0``: the gouge and field criteria read it) and also
    the progress baseline. ``grid.C`` carries the lip's mark (their ``compact``)."""
    p = pad or PadScalars()
    return _finish("pad-edge", lambda g, r: _measure_edge(g, p, start, x, y), grid, start)
