# SPDX-License-Identifier: Apache-2.0
"""Gym-Anything concrete-finishing success criteria for Alloy's surface-flow proof tasks (host-side numpy).

Ports the four tasks pad-darby, pad-trowel, patch-flush and tool-joints from
gym-anything-robotics/benchmarks/roboworld/environments/concrete_finishing/ (read-only): task.py (T:line),
task_set_progress.py (TSP:line), params.py (P:line). One world at a time, on a regular 2-D grid.

Heights `h` are metres relative to the task's reference plane: the form tops for the pads (T:253 plane
z = FORM[1], so the plane criterion is max |h|), the slab top for patch-flush (T:603), the block face for
tool-joints (T:633, + out of the wall). Every function returns a `Verdict`; progress-v1 needs the start grid.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.ndimage import distance_transform_edt

from alloy.tasks import concrete_materials as CM
from alloy.tasks.concrete_progress import FAIL_FACTOR, Progress, Quantity, band, clip, falloff, progress_v1

log = logging.getLogger("alloy.tasks.concrete_metrics")

# thresholds: params.py:175-184 EVAL (authored), task.py:49-71
GAP_MM, STRAIGHTEDGE_M, PLANE_MM, BAND_M = 3.0, 1.2, 3.0, 0.03
AGG_MAX, AGG_SHARE, SEAL_MAX = 0.02, 0.98, 0.2
RADIUS_TOL_M, EDGE_SHARE_TROWEL = 1.5e-3, 0.90        # T:384-385 (share 0.90 for pad-trowel)
EDGER_RADIUS, LIP_W, EDGE_BAND_M = 0.0095, 0.0015, 0.10   # P:146; T:41; T:287 (profile within 0.10 m of the form)
DENS_MIN, TROWEL_TEX, TEX_SHARE = 0.8, 5.0e-5, 0.95
FLOAT_TEX = (1.5e-4, 6.0e-4)
DELAM_MAX, BURN_MAX, DELAM_Q, DWC_DUST, BLISTER_Q, BLISTER_LIFT = 0.01, 0.05, 1.0e-5, 0.05, 2.0e-4, 3.0e-4
LIP_MM, LIP_HARD_MM, PATCH_MEAN_MM, FILL_SHARE, FILL_LEVEL = 3.0, 6.4, 1.5, 0.98, -5e-4
SMEAR_T, OFF_TOL, FEATHER_DIST = 2.0e-4, 2.0e-8, 0.010        # T:51,54,605
TAU_TIMELY_PA, JOINT_W = 30e3, 0.0095                         # T:610 ; P:45
CONCAVE_MM, JOINT_SHARE, SMEAR_CM2_M = (1.4, 2.8), 0.95, 2.0
MORTAR_WINDOW, COMPACT_MIN, VOID_EXTRA_M = (5.0e3, 2.0e4), 0.5, 1.5e-3
FORMS_MAX_N = 150.0                                           # T:707-710, P:34
CONTRACT_V1, CONTRACT_LEGACY = "progress-v1", "legacy"


@dataclass
class PasteGrid:
    """Fields [nx, ny] on a regular grid of spacing `cell` [m]. Optional fields may be None when a task does not read
    them. Units: h, t, a, q, w, bl, bleed_left in m; D, E, S, C in [0, 1]; tau_w in Pa; dwc dimensionless."""

    h: np.ndarray                       # surface height above the reference plane (paste.py:108 b+tb+ts+bl)
    t: np.ndarray                       # paste thickness (smear tests, T:605)
    cell: float
    paste_mask: np.ndarray              # bool: pad interior / patch recess / joint cells
    D: np.ndarray | None = None         # densification
    a: np.ndarray | None = None         # texture amplitude
    E: np.ndarray | None = None         # coarse-aggregate exposure
    S: np.ndarray | None = None         # sealing
    C: np.ndarray | None = None         # mortar compaction (r.compact)
    q: np.ndarray | None = None         # trapped bleed water
    w: np.ndarray | None = None         # bleed film (informational: wet_worked is what the criteria read)
    bl: np.ndarray | None = None        # blister lift
    bleed_left: np.ndarray | None = None  # remaining_bleed(material, age) (T:75-85) for committed water
    tau_w: np.ndarray | None = None     # surface/bulk yield stress at last working (Pa)
    late: np.ndarray | None = None      # worked after initial set
    worked: np.ndarray | None = None    # distance worked (>0 means worked)
    wet_worked: np.ndarray | None = None  # worked while bleed film stood
    dwc: np.ndarray | None = None       # water-cement rise of the top 6 mm
    off_m3: float = 0.0                 # paste caught off-region (T:159-167)
    d_edge: np.ndarray | None = None    # m: each cell centre's distance in from the nearest form, as their pad_d
                                        # (T:225; concrete_pad_start's d_edge); None: from the mask's bounding box


@dataclass
class PadScalars:
    """Task-level inputs of the pad tasks (T:707-733, 839-844)."""

    forms_peak_n: float = 0.0           # peak horizontal force on forms/kickers
    tools_off: int = 0                  # tools not resting on a cart shelf
    tools_total: int = 1
    arms_over: int = 0                  # robot links/fingers/hands over the pad below z 0.40 m


@dataclass
class Criterion:
    name: str
    value: float | None
    threshold: float
    passed: bool
    unit: str = ""


@dataclass
class Verdict:
    task: str
    criteria: list[Criterion]
    success: bool
    score_legacy: float                  # share of criteria passed (T:872)
    progress: Progress | None = None     # None without a start grid
    score: float | None = None           # progress-v1: 1 on success else 0.95 x progress (TSP:275)
    contract: str = CONTRACT_V1

    def __getitem__(self, name: str) -> Criterion:
        return next(c for c in self.criteria if c.name == name)


def _need(g: PasteGrid, *names: str) -> None:
    missing = [n for n in names if getattr(g, n) is None]
    if missing:
        raise ValueError(f"PasteGrid is missing fields {missing} needed by this task")


def _share(mask: np.ndarray) -> float:
    """Mean of a bool array; 0 for an empty selection (nothing to pass)."""
    return float(mask.mean()) if mask.size else 0.0


def _upper_gap(seg: np.ndarray, step: float) -> float:
    """Gap under a straightedge resting on the upper convex hull of one window (T:178-193)."""
    x = np.arange(len(seg)) * step
    hull: list[tuple[float, float]] = []
    for k in range(len(seg)):
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (seg[k] - y1) - (y2 - y1) * (x[k] - x1) >= 0:
                hull.pop()
            else:
                break
        hull.append((x[k], seg[k]))
    hx, hy = np.array([p[0] for p in hull]), np.array([p[1] for p in hull])
    return float((np.interp(x, hx, hy) - seg).max())


def hull_gap(z: np.ndarray, step: float, window: float = STRAIGHTEDGE_M) -> float:
    """Largest gap (m) under a `window` straightedge on profile z sampled every `step` m (T:170-193): windows
    start every win//4 samples plus one flush with the end; lines under 3 samples give 0."""
    n = len(z)
    if n < 3:
        return 0.0
    win = max(3, int(round(window / step)))
    starts = [0] if n <= win else list(range(0, n - win + 1, max(1, win // 4))) + [n - win]
    return max(_upper_gap(np.asarray(z[s:s + win], float), step) for s in starts)


def straightedge(h: np.ndarray, core: np.ndarray, cell: float, window: float = STRAIGHTEDGE_M):
    """Max straightedge gap (m) and every placement's gap (T:237-251): along x and y every round(0.025/cell) cells
    over the core's bounding box, plus both diagonals (step cell*sqrt2, more than 10 cells)."""
    idx = np.argwhere(core)
    if not len(idx):
        raise ValueError("straightedge: empty core")
    i0, j0 = idx.min(axis=0)
    i1, j1 = idx.max(axis=0)
    every = max(1, int(round(0.025 / cell)))
    gaps = [hull_gap(h[i0:i1 + 1, j], cell, window) for j in range(j0, j1 + 1, every)]
    gaps += [hull_gap(h[i, j0:j1 + 1], cell, window) for i in range(i0, i1 + 1, every)]
    ii = np.arange(i0, i1 + 1)
    for k in range(-(i1 - i0), j1 - j0 + 1, every):
        for jj in (j0 + k + (ii - i0), j1 - k - (ii - i0)):
            ok = (jj >= j0) & (jj <= j1)
            if ok.sum() > 10:
                gaps.append(hull_gap(h[ii[ok], jj[ok]], cell * math.sqrt(2), window))
    return max(gaps), np.asarray(gaps)


def form_distance(mask: np.ndarray, cell: float) -> np.ndarray:
    """Distance (m) of each cell centre from the nearest of the pad's four form faces (T:225 pad_d), taking the
    pad as the bounding box of `mask` with its edge at the outer face of the edge cells."""
    idx = np.argwhere(mask)
    (i0, j0), (i1, j1) = idx.min(axis=0), idx.max(axis=0)
    i, j = np.arange(mask.shape[0])[:, None], np.arange(mask.shape[1])[None, :]
    dx, dy = np.minimum(i - i0 + 0.5, i1 - i + 0.5), np.minimum(j - j0 + 0.5, j1 - j + 0.5)
    return np.minimum(dx, dy) * cell


def edge_distance(g: PasteGrid) -> np.ndarray:
    """Each cell centre's distance (m) in from the nearest form: the grid's ``d_edge`` (their pad_d from the pad's true
    faces) when it has one, else ``form_distance`` of its mask. Their 1.219 m pad is 609.5 of their 2 mm cells, so
    the bounding box puts the east face 1 mm off and drops their core's east column (30.0 mm in, 427 cells)."""
    if g.d_edge is not None:
        return g.d_edge
    global _BOX_WARNED
    if not _BOX_WARNED:
        _BOX_WARNED = True
        log.warning("pad metrics without PasteGrid.d_edge: the forms' faces come from the mask's bounding box, up to "
                    "a cell off their pad_d (their core's east column dropped at 2 mm cells); pass d_edge=f['d_edge']")
    return form_distance(g.paste_mask, g.cell)


_BOX_WARNED = False


def pad_core(g: PasteGrid) -> np.ndarray:
    """Paste cells more than 30 mm from the forms (T:234-235)."""
    return g.paste_mask & (edge_distance(g) > BAND_M)


def _arc_ok(h: np.ndarray, d: np.ndarray, cell: float) -> bool:
    """One perimeter line within 1.5 mm of the 3/8 in arc, allowed shifts 0, 0.5, 1 cell, lip slot excluded
    (T:293-300). Heights relative to the form top (arc top = 0)."""
    for shift in (0.0, 0.5 * cell, cell):
        ds = d - shift
        inr = (ds >= LIP_W) & (ds < EDGER_RADIUS)
        if not inr.any():
            continue
        arc = -(EDGER_RADIUS - np.sqrt(np.maximum(EDGER_RADIUS**2 - (EDGER_RADIUS - ds[inr]) ** 2, 0.0)))
        if np.abs(h[inr] - arc).max() <= RADIUS_TOL_M:
            return True
    return False


def radius_profile(g: PasteGrid) -> float:
    """Share of perimeter lines on the edger arc (T:281-300,329-330): one line per cell column/row and per side,
    the cells within 0.10 m of that side's form, the arc about the nearest form (corners round both ways)."""
    mask, h, cell = g.paste_mask, g.h, g.cell
    dmin = edge_distance(g)
    idx = np.argwhere(mask)
    j0, j1 = idx[:, 1].min(), idx[:, 1].max()
    ok = n = 0
    for hh, mm, dd, lo, hi in ((h, mask, dmin, j0, j1), (h.T, mask.T, dmin.T, *(idx[:, 0].min(), idx[:, 0].max()))):
        pos = np.arange(hh.shape[1])
        for side in ((pos - lo + 0.5) * cell, (hi - pos + 0.5) * cell):
            near = side < EDGE_BAND_M
            for i in np.flatnonzero(mm.any(axis=1)):
                sel = mm[i] & near
                if sel.any():
                    n += 1
                    ok += _arc_ok(hh[i][sel], dd[i][sel], cell)
    return ok / max(n, 1)


def blister_lift(q: np.ndarray, bl: np.ndarray, cell: float) -> np.ndarray:
    """Blister raised by trapped water q (T:88-97), never below the standing blisters ``bl``: the materials' law."""
    return CM.blister_lift(q, cell, previous=bl)


def committed_water(q: np.ndarray, S: np.ndarray, bleed_left: np.ndarray) -> np.ndarray:
    """Water each cell will hold when bleeding ends (T:408-411): q + clip(S, 0, 1) x remaining bleed."""
    return q + np.clip(S, 0.0, 1.0) * bleed_left


def smear_residue(t: np.ndarray, mask: np.ndarray, cell: float, off_m3: float = 0.0, length: float = 1.0) -> float:
    """Progress-v1 residue (TSP:73-79): volume above 0.2 mm over `mask` (+ off-region catch above 2e-8 m3) as the
    area it would cover at 0.2 mm, cm2 per `length` m."""
    excess = float(np.maximum(t[mask] - SMEAR_T, 0).sum()) * cell**2 + max(off_m3 - OFF_TOL, 0.0)
    return excess / SMEAR_T * 1e4 / length


def _c(name: str, value, ok: bool, threshold: float, unit: str = "") -> Criterion:
    return Criterion(name, None if value is None else float(value), float(threshold), bool(ok), unit)


def _off_ok(g: PasteGrid) -> bool:
    return g.off_m3 <= OFF_TOL                          # T:159-167 _with_off


def _scalar_q(c: Criterion, zero: float, absolute: bool = False, group: str | None = None, weight: float = 1.0):
    """TSP:57-60 scalar(): smaller is better, keep = falloff(value, threshold, zero)."""
    v = abs(c.value) if absolute else c.value
    return Quantity(-v, -c.threshold, c.passed, falloff(v, c.threshold, zero), group or c.name, weight)


def _blend(share: float, graded: float, target: float, passed: bool, group: str) -> Quantity:
    """Half the criterion's share, half the graded per-cell credit (TSP:94,140,148,161-170)."""
    return Quantity(0.5 * share + 0.5 * graded, target, passed, 1.0, group)


_Measure = tuple[list[Criterion], dict[str, Quantity], dict[str, float]]


def _flat_criteria(g: PasteGrid):
    """flat_straightedge and flat_plane (T:237-257)."""
    core = pad_core(g)
    gap, gaps = straightedge(g.h, core, g.cell)
    plane = float(np.abs(g.h[core]).max())
    return core, gaps, [_c("flat_straightedge", gap * 1e3, gap * 1e3 <= GAP_MM, GAP_MM, "mm"),
                        _c("flat_plane", plane * 1e3, plane * 1e3 <= PLANE_MM, PLANE_MM, "mm")]


def _level_quantities(crits: list[Criterion], gaps: np.ndarray) -> dict[str, Quantity]:
    """Group 'level' (TSP:87-90): 0.25 plane + 0.25 straightedge + 0.5 graded placements."""
    edge, plane = crits
    graded = np.clip(1.0 - np.maximum(gaps * 1e3 - GAP_MM, 0) / GAP_MM, 0, 1)
    return {"flat_plane": _scalar_q(plane, 2 * PLANE_MM, group="level", weight=0.25),
            "flat_straightedge": _scalar_q(edge, 2 * GAP_MM, group="level", weight=0.25),
            "placements": Quantity(float(graded.mean()), 1.0, edge.passed, 1.0, "level", 0.5)}


def _pad_shared(g: PasteGrid, p: PadScalars) -> list[Criterion]:
    """forms_undisturbed, tools_back, arms_clear (T:393-395, 707-733, 839-844)."""
    return [_c("forms_undisturbed", p.forms_peak_n, p.forms_peak_n <= FORMS_MAX_N, FORMS_MAX_N, "N"),
            _c("tools_back", p.tools_off, p.tools_off == 0, 0, "tools"),
            _c("arms_clear", p.arms_over, p.arms_over == 0, 0, "links")]


def _tools_gate(p: PadScalars) -> float:
    return 0.9 + 0.1 * (p.tools_total - p.tools_off) / max(p.tools_total, 1)   # TSP:80-81


def _measure_darby(g: PasteGrid, p: PadScalars, ref: PasteGrid | None) -> _Measure:
    """pad-darby: T:354-360,386-395; TSP:87-99."""
    _need(g, "E", "S", "wet_worked")
    on = g.paste_mask
    core, gaps, crits = _flat_criteria(g)
    share = _share(g.E[core] < AGG_MAX)
    s_max = float(g.S[on].max())
    n_wet = int((g.wet_worked[on] > 0).sum())
    crits += [_c("stone_down", share, share >= AGG_SHARE, AGG_SHARE),
              _c("not_sealed", s_max, s_max < SEAL_MAX, SEAL_MAX),
              _c("no_water_worked", n_wet, n_wet == 0, 0, "cells")] + _pad_shared(g, p)
    q = _level_quantities(crits[:2], gaps)
    raw = np.clip((0.30 - g.E[core]) / (0.30 - AGG_MAX), 0, 1)       # TSP:94: exposure feedback before 2 %
    q["stone_down"] = _blend(share, float(raw.mean()), AGG_SHARE, crits[2].passed, "stone_down")
    gates = {"tools": _tools_gate(p), "water_worked": 1.0 - _share(g.wet_worked[on] > 0),
             "open_surface": falloff(s_max, SEAL_MAX, 0.6)}
    return crits, q, gates


def _trowel_water(g: PasteGrid, contract: str) -> tuple[np.ndarray, np.ndarray]:
    """Trapped water and blister mask: committed water under progress-v1 (TSP:114-121), as-is in legacy (T:375-378)."""
    if contract == CONTRACT_LEGACY:
        return g.q, g.bl > BLISTER_LIFT
    left = g.bleed_left if g.bleed_left is not None else np.zeros_like(g.q)
    qf = committed_water(g.q, g.S, left)
    return qf, blister_lift(qf, g.bl, g.cell) > BLISTER_LIFT


def _measure_trowel(g: PasteGrid, p: PadScalars, ref: PasteGrid | None, contract: str) -> _Measure:
    """pad-trowel: T:371-396; TSP:81-131."""
    _need(g, "D", "a", "S", "q", "bl", "late", "wet_worked", "dwc")
    on = g.paste_mask
    core, gaps, flat = _flat_criteria(g)
    smooth = _share((g.D[core] >= DENS_MIN) & (g.a[core] <= TROWEL_TEX))
    qw, blister = _trowel_water(g, contract)
    n_bl, delam = int(blister[on].sum()), _share(qw[on] > DELAM_Q)
    burn = _share((g.late[on] > 0) & (g.D[on] >= 0.98))
    radius, n_wet, dwc = radius_profile(g), int((g.wet_worked[on] > 0).sum()), float(g.dwc[on].max())
    crits = [_c("smooth_hard", smooth, smooth >= TEX_SHARE, TEX_SHARE),
             _c("no_blister", n_bl, n_bl == 0, 0, "cells"),
             _c("delamination_risk", delam, delam < DELAM_MAX, DELAM_MAX),
             _c("no_trowel_burn", burn, burn < BURN_MAX, BURN_MAX)] + flat
    crits += [_c("radius_profile", radius, radius >= EDGE_SHARE_TROWEL, EDGE_SHARE_TROWEL),
              _c("no_water_worked", n_wet, n_wet == 0, 0, "cells"),
              _c("surface_wc", dwc, dwc < DWC_DUST, DWC_DUST)] + _pad_shared(g, p)
    d0 = ref.D if ref is not None else np.zeros_like(g.D)
    tex = np.clip((FLOAT_TEX[0] - g.a) / (FLOAT_TEX[0] - TROWEL_TEX), 0, 1)
    dens = np.clip((g.D - d0) / np.maximum(DENS_MIN - d0, 1e-6), 0, 1)
    q = _level_quantities(flat, gaps)
    raw = float(np.minimum(tex, dens)[core].mean())
    q["smooth_hard"] = _blend(smooth, raw, TEX_SHARE, crits[0].passed, "smooth_hard")
    q["radius_profile"] = Quantity(radius, 0.90, radius >= EDGE_SHARE_TROWEL, clip((radius - 0.45) / 0.45),
                                   "radius_profile")                                      # TSP:130-131
    surface = (g.wet_worked > 0) | (g.dwc >= DWC_DUST) | ((g.late > 0) & (g.D >= 0.98))     # TSP:122-125
    gates = {"tools": _tools_gate(p), "water_free": 1.0 - _share(((qw > DELAM_Q) | blister)[on]),
             "surface_free": 1.0 - _share(surface[on])}
    return crits, q, gates


def _inner_ring(mask: np.ndarray) -> np.ndarray:
    """Cells of `mask` with a 4-neighbour outside it: the cut-edge ring (T:600)."""
    out = np.zeros_like(mask)
    for shift, ax in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
        out |= mask & ~np.roll(mask, shift, ax)
    return out


def _measure_patch(g: PasteGrid, ref: PasteGrid | None) -> _Measure:
    """patch-flush: T:597-625; TSP:133-152."""
    _need(g, "a", "tau_w", "worked")
    rec, h = g.paste_mask, g.h
    inner = _inner_ring(rec)
    lip, mean = float(np.abs(h[inner]).max()) * 1e3, float(h[rec].mean()) * 1e3
    filled = _share(h[inner] >= FILL_LEVEL)
    far = ~rec & (distance_transform_edt(~rec) * g.cell > FEATHER_DIST)                     # T:602
    n_smear = int(((g.t > SMEAR_T) & far).sum())
    worked = rec & (g.worked > 0)
    in_band = (g.a >= FLOAT_TEX[0]) & (g.a <= FLOAT_TEX[1])
    tex = _share((in_band & worked)[rec])
    late = float(g.tau_w[worked].max()) if worked.any() else None
    crits = [_c("no_lip", lip, lip <= LIP_MM, LIP_MM, "mm"),
             _c("no_hard_lip", lip, lip <= LIP_HARD_MM, LIP_HARD_MM, "mm"),
             _c("flush_mean", mean, abs(mean) <= PATCH_MEAN_MM, PATCH_MEAN_MM, "mm"),
             _c("against_the_cut", filled, filled >= FILL_SHARE, FILL_SHARE),
             _c("no_feather_edge", n_smear * g.cell**2 * 1e4, n_smear == 0 and _off_ok(g), 0.0, "cm2"),
             _c("float_texture", tex, tex >= TEX_SHARE, TEX_SHARE),
             _c("finished_in_time", None if late is None else late / 1e3, late is not None and late <= TAU_TIMELY_PA,
                30.0, "kPa")]
    crit = {c.name: c for c in crits}
    fill = np.clip(1.0 - np.maximum(FILL_LEVEL - h[inner], 0) / 0.01, 0, 1)                 # TSP:140
    timely = (g.worked > 0) & (g.tau_w <= TAU_TIMELY_PA)
    share = _share((in_band & timely)[rec])
    graded = float((band(g.a, *FLOAT_TEX, FLOAT_TEX[1]) * timely)[rec].mean())
    res = smear_residue(g.t, far, g.cell, g.off_m3)
    q = {"no_lip": _scalar_q(crit["no_lip"], LIP_HARD_MM),
         "flush_mean": _scalar_q(crit["flush_mean"], 2 * PATCH_MEAN_MM, absolute=True),
         "against_the_cut": _blend(filled, float(fill.mean()), FILL_SHARE, crit["against_the_cut"].passed,
                                   "against_the_cut"),
         "float_texture": _blend(share, graded, TEX_SHARE, crit["float_texture"].passed, "float_texture"),
         "no_feather_edge": Quantity(-res, 0.0, crit["no_feather_edge"].passed, falloff(res, 0.0, 20.0),
                                     "no_feather_edge")}
    gates = {"timely": 1.0 - _share(((g.worked > 0) & (g.tau_w > TAU_TIMELY_PA))[rec])}
    return crits, q, gates


def joint_length(n_centre: int, cell: float) -> float:
    """Joint length (m) behind the smear rule (T:639): N_centre x cell / max(round(0.75 JOINT_W / cell) - 1, 1)."""
    return max(n_centre * cell / max(int(round(0.75 * JOINT_W / cell)) - 1, 1), 1e-3)


def _measure_joints(g: PasteGrid, ref: PasteGrid | None) -> _Measure:
    """tool-joints: T:628-651; TSP:154-172."""
    _need(g, "C", "tau_w")
    on, h = g.paste_mask, g.h
    centre = on & (distance_transform_edt(on) * g.cell >= 0.75 * JOINT_W / 2)               # T:631
    depth, tw, comp = -h[centre], g.tau_w[centre], g.C[centre]
    lo, hi = CONCAVE_MM
    conc = _share((depth >= lo * 1e-3) & (depth <= hi * 1e-3))
    compacted = _share(comp >= COMPACT_MIN)
    ripe = _share((tw >= MORTAR_WINDOW[0]) & (tw <= MORTAR_WINDOW[1]))
    length = joint_length(int(centre.sum()), g.cell)
    face = ~on
    smear = int((face & (g.t > SMEAR_T)).sum()) * g.cell**2 * 1e4 / length
    n_void = int((on & (h < -(hi * 1e-3 + VOID_EXTRA_M))).sum())
    crits = [_c("concave", conc, conc >= JOINT_SHARE, JOINT_SHARE),
             _c("compacted", compacted, compacted >= JOINT_SHARE, JOINT_SHARE),
             _c("tooled_thumbprint_hard", ripe, ripe >= 0.90, 0.90),
             _c("smear", smear, smear < SMEAR_CM2_M and _off_ok(g), SMEAR_CM2_M, "cm2/m"),
             _c("no_void", n_void, n_void == 0, 0, "cells")]
    ok = {c.name: c.passed for c in crits}
    mean = lambda v: float(v.mean()) if v.size else 0.0                                       # noqa: E731
    w_lo, w_hi = MORTAR_WINDOW
    thumb = np.minimum(np.clip(tw / w_lo, 0, 1), np.clip((2 * w_hi - tw) / w_hi, 0, 1))      # TSP:166-168
    deficit = float(np.maximum(-h[on] - (hi * 1e-3 + VOID_EXTRA_M), 0).mean())
    res = smear_residue(g.t, face, g.cell, g.off_m3, length)
    q = {"concave": _blend(conc, mean(band(depth * 1e3, lo, hi, hi)), JOINT_SHARE, ok["concave"], "tooling"),
         "compacted": _blend(compacted, mean(np.clip(comp / COMPACT_MIN, 0, 1)), JOINT_SHARE,
                             ok["compacted"], "tooling"),
         "tooled_thumbprint_hard": _blend(ripe, mean(thumb), 0.90, ok["tooled_thumbprint_hard"], "tooling"),
         "no_void": Quantity(-deficit, 0.0, ok["no_void"], clip(1.0 - deficit / 0.003), "no_void"),
         "smear": Quantity(-res, 0.0, ok["smear"], falloff(res, 0.0, 6.0), "smear")}
    return crits, q, {}


def _finish(task: str, measure: Callable[[PasteGrid, PasteGrid | None], _Measure], g: PasteGrid,
            start: PasteGrid | None, contract: str = CONTRACT_V1) -> Verdict:
    crits, q, gates = measure(g, start)
    passed = [c.passed for c in crits]
    success = bool(passed) and all(passed)
    prog = None
    if start is not None:
        prog = progress_v1(q, gates, measure(start, start)[1])
    score = 1.0 if success else (FAIL_FACTOR * prog.value if prog is not None else None)
    return Verdict(task, crits, success, sum(passed) / max(len(passed), 1), prog, score, contract)


def pad_darby(grid: PasteGrid, pad: PadScalars | None = None, start: PasteGrid | None = None) -> Verdict:
    """pad-darby verdict (T:354-360,386-395)."""
    p = pad or PadScalars()
    return _finish("pad-darby", lambda g, r: _measure_darby(g, p, r), grid, start)


def pad_trowel(grid: PasteGrid, pad: PadScalars | None = None, start: PasteGrid | None = None,
               contract: str = CONTRACT_V1) -> Verdict:
    """pad-trowel verdict; contract 'legacy' reads q and bl as they are, 'progress-v1' the committed water."""
    p = pad or PadScalars()
    return _finish("pad-trowel", lambda g, r: _measure_trowel(g, p, r, contract), grid, start, contract)


def patch_flush(grid: PasteGrid, start: PasteGrid | None = None) -> Verdict:
    """patch-flush verdict (seven criteria, T:597-625)."""
    return _finish("patch-flush", _measure_patch, grid, start)


def tool_joints(grid: PasteGrid, start: PasteGrid | None = None) -> Verdict:
    """tool-joints verdict (five criteria, T:628-651)."""
    return _finish("tool-joints", _measure_joints, grid, start)
