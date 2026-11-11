# SPDX-License-Identifier: Apache-2.0
"""Gym-Anything's pad-finish success criteria and progress (host-side numpy).

Ports ``concrete_finishing/task.py`` (T:line): the authored tolerances (T:56-70), ``remaining_bleed`` (T:75-84),
the caps' helpers (T:100-145), ``_top``, ``_committed_water``, ``_finish_edges`` and ``_finish_criteria``
(T:398-536), the evaluator's start bookkeeping (T:210-216) and ``_finish``'s progress (T:537-596, reward version
``pad-finish/progress-v3``). Thresholds are their EVAL (P:179-184). Shared pieces are WP6's ``concrete_metrics``
(flatness, blister law, committed water) and the pad-edge port's perimeter lines (``pad_metrics``).

Heights relative to the form-top plane; geometry from cell coordinates, as ``pad_metrics``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import distance_transform_edt, label
from scipy.ndimage import maximum as nmax
from scipy.ndimage import sum as nsum

from alloy.tasks import concrete_materials as CM
from alloy.tasks import concrete_metrics as MX
from alloy.tasks import pad_metrics as PM
from alloy.tasks.concrete_metrics import Criterion, PadScalars, PasteGrid, _c, _need
from alloy.tasks.concrete_pad_start import AGES_21C
from alloy.tasks.concrete_progress import FAIL_FACTOR, FAIL_TERM

# T:56-70 (authored) and P:179-184 (EVAL)
TOP_D = PM.EDGER_RADIUS + 0.002            # the top outside the arris
EDGER_W = 0.0762 + 0.0015                  # the edger's flat and lip: its track counts as steel-finished by it
EDGER_TEX, TROWEL_TEX, FLOAT_TEX0 = 6.0e-5, 5.0e-5, 1.5e-4
DENS_MIN, TEX_SHARE, AGG_MAX, AGG_SHARE = 0.8, 0.95, 0.02, 0.98
SPOT_MM, PATCH_MM2, CLEAR_MISS, STONY_E = 75.0, 4400.0, 0.5, 0.05
SIDE_SHARE, SHORT_MISS_MM, LONG_MISS_MM, EDGE_SHARE = 0.80, 50.0, 100.0, 0.90
DELAM_Q, DELAM_MAX, BURN_MAX, DWC_DUST, BLISTER_LIFT = 1.0e-5, 0.01, 0.05, 0.05, 3.0e-4
FINISH_WORK = ("flat_plane", "flat_straightedge", "stone_down", "radius_profile", "lip_passed", "smooth_hard")


def remaining_bleed(age: np.ndarray, temp_c: float = 21.1, mat: CM.Concrete | None = None) -> np.ndarray:
    """m of bleed water still to rise at ``age`` [s] (T:75-84): the triangle rate's integral to its end."""
    mat = mat or CM.Concrete()
    te = float(mat.bleed_end_time(temp_c))
    peak = 2.0 * mat.bleed_mean / 3600.0
    a = np.clip(np.asarray(age, float), 0.0, te)
    return np.where(a < te / 3.0, 0.5 * peak * te - 1.5 * peak * a * a / te, 0.75 * peak * (te - a) ** 2 / te) / 1000.0


def spot_mm(fail: np.ndarray, cell: float) -> float:
    """Widest disc of failing cells [mm] (T:108-110)."""
    return float(2.0 * distance_transform_edt(fail).max() * cell * 1e3) if fail.any() else 0.0


def patch_mm2(fail: np.ndarray, cell: float) -> float:
    """Largest 8-connected failing patch [mm²] (T:113-119)."""
    if not fail.any():
        return 0.0
    lab, _ = label(fail, structure=np.ones((3, 3), bool))
    return float(np.bincount(lab.ravel())[1:].max() * (cell * 1e3) ** 2)


def rejected_deficit(miss: np.ndarray, deficit: np.ndarray, cell: float, n_top: int) -> float:
    """The work left in the components the caps reject, over the top's cell count (T:122-135)."""
    if not miss.any():
        return 0.0
    lab, n = label(miss, structure=np.ones((3, 3), bool))
    idx = np.arange(1, n + 1)
    area = np.bincount(lab.ravel())[1:] * (cell * 1e3) ** 2
    width = 2.0 * np.asarray(nmax(distance_transform_edt(miss), lab, idx)) * cell * 1e3
    rej = idx[(area > PATCH_MM2) | (width > SPOT_MM)]
    return float(np.asarray(nsum(deficit, lab, rej)).sum()) / max(n_top, 1) if len(rej) else 0.0


def miss_runs(ok: np.ndarray) -> list[int]:
    """Lengths, in lines, of the runs of consecutive failing lines (T:138-145)."""
    runs, cur = [], 0
    for v in ok:
        if v:
            if cur:
                runs.append(cur)
            cur = 0
        else:
            cur += 1
    if cur:
        runs.append(cur)
    return runs


def finish_edges(lines: list[PM.Line], cell: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per side, per line: on the arc within 1.5 mm (the arc's top the line's own median surface 11.5-21.5 mm in, a
    line without one takes the nearest line's), and the lip's mark (T:413-451)."""
    rad, out = PM.EDGER_RADIUS, {}
    for side in PM.SIDES:
        ls = [ln for ln in lines if ln.side == side and len(ln.h)]
        tops = np.array([float(np.median(ln.h[(ln.d >= rad + 0.002) & (ln.d < rad + 0.012)]))
                         if ((ln.d >= rad + 0.002) & (ln.d < rad + 0.012)).any() else np.nan for ln in ls])
        good = np.flatnonzero(np.isfinite(tops))
        if len(good):
            tops = tops[good[np.abs(np.arange(len(tops))[:, None] - good[None, :]).argmin(axis=1)]]
        else:
            tops = np.zeros(len(ls))
        prof = [PM.arc_error(ln, cell, top) <= MX.RADIUS_TOL_M for ln, top in zip(ls, tops, strict=True)]
        lip = [bool(ln.lip[ln.d < PM.LIP_BAND].any()) for ln in ls]
        out[side] = (np.array(prof, bool), np.array(lip, bool))
    return out


@dataclass
class FinishMeasure:
    """The criteria and the progress inputs (T:493-499, 530-533)."""

    criteria: list[Criterion]
    placements: float
    credit: float
    finish_left: float
    stone_left: float
    side_radius: dict[str, float]
    side_arc_lip: dict[str, float]
    water_free: float
    surface_free: float

    def value(self, name: str) -> float:
        return next(c.value for c in self.criteria if c.name == name)


def _caps(name: str, share: float, target: float, fail: np.ndarray, cell: float) -> Criterion:
    spot, patch = spot_mm(fail, cell), patch_mm2(fail, cell)
    return _c(name, share, share >= target and spot <= SPOT_MM and patch <= PATCH_MM2, target)


def _edge_criteria(edges: dict, cell: float) -> list[Criterion]:
    """radius_profile and lip_passed over the whole perimeter, each side >= 80 %, long misses <= 100 mm (T:474-491)."""
    out, mm = [], cell * 1e3
    for name, k in (("radius_profile", 0), ("lip_passed", 1)):
        flags = np.concatenate([edges[s][k] for s in PM.SIDES])
        share = float(flags.mean()) if len(flags) else 0.0
        side = [float(edges[s][k].mean()) for s in PM.SIDES if len(edges[s][k])]
        long = [(sum(n for n in miss_runs(edges[s][0]) if n * mm > SHORT_MISS_MM) if k == 0
                 else int((~edges[s][1]).sum())) * mm for s in PM.SIDES]
        ok = share >= EDGE_SHARE and min(side, default=0.0) >= SIDE_SHARE and max(long) <= LONG_MISS_MM
        out.append(_c(name, share, ok, EDGE_SHARE))
    return out


def measure(g: PasteGrid, x: np.ndarray, y: np.ndarray, start: PasteGrid, D0: np.ndarray, E0: np.ndarray,
            bleed_left: np.ndarray, p: PadScalars) -> FinishMeasure:
    """pad-finish's criteria from the pad alone (T:455-536) plus forms, tools and robot (T:542-545)."""
    _need(g, "D", "a", "E", "S", "q", "bl", "late", "wet_worked", "dwc", "C")
    on, pad_d = g.paste_mask, PM.pad_distance(x, y)
    top = on & (pad_d >= TOP_D)
    n_top = int(top.sum())
    tex = np.where(pad_d < EDGER_W, EDGER_TEX, TROWEL_TEX)
    ok = (g.D >= DENS_MIN) & (g.a <= tex)
    credit = np.minimum(np.clip((FLOAT_TEX0 - g.a) / (FLOAT_TEX0 - tex), 0.0, 1.0),
                        np.clip((g.D - D0) / np.maximum(DENS_MIN - D0, 1e-6), 0.0, 1.0))
    miss = top & ~ok & (credit < CLEAR_MISS)
    crits = [_caps("smooth_hard", float(ok[top].mean()), TEX_SHARE, miss, g.cell)]
    stony = top & (g.E > STONY_E)
    crits.append(_caps("stone_down", float((g.E < AGG_MAX)[top].mean()), AGG_SHARE, stony, g.cell))
    gap, gaps = MX.straightedge(g.h, top, g.cell)
    plane = float(np.abs(g.h[top]).max())
    crits += [_c("flat_straightedge", gap * 1e3, gap * 1e3 <= MX.GAP_MM, MX.GAP_MM, "mm"),
              _c("flat_plane", plane * 1e3, plane * 1e3 <= MX.PLANE_MM, MX.PLANE_MM, "mm")]
    edges = finish_edges(PM.perimeter(g, start, x, y), g.cell)
    crits += _edge_criteria(edges, g.cell)
    qf = MX.committed_water(g.q, g.S, bleed_left)
    blister = (MX.blister_lift(qf, g.bl, g.cell) > BLISTER_LIFT) & on
    dl = float((qf[on] > DELAM_Q).mean())
    burn_cells = (g.late > 0) & (g.D >= 0.98)
    burn = float(burn_cells[on].mean())
    n_wet = int(g.wet_worked[on].sum())
    dw = float(g.dwc[on].max())
    crits += [_c("no_blister", int(blister.sum()), not blister.any(), 0, "cells"),
              _c("delamination_risk", dl, dl < DELAM_MAX, DELAM_MAX),
              _c("no_trowel_burn", burn, burn < BURN_MAX, BURN_MAX),
              _c("no_water_worked", n_wet, n_wet == 0, 0, "cells"),
              _c("surface_wc", dw, dw < DWC_DUST, DWC_DUST)]
    crits += [c for c in MX._pad_shared(g, p) if c.name != "arms_clear"]
    crits.append(_c("robot_off_pad", p.arms_over, p.arms_over == 0, 0, "links"))
    stone_credit = np.clip((E0 - g.E) / np.maximum(E0 - AGG_MAX, 1e-6), 0.0, 1.0)
    return FinishMeasure(
        crits, placements=float((gaps * 1e3 <= MX.GAP_MM).mean()), credit=float(credit[top].mean()),
        finish_left=rejected_deficit(miss, 1.0 - credit, g.cell, n_top),
        stone_left=rejected_deficit(stony, 1.0 - stone_credit, g.cell, n_top),
        side_radius={s: float(edges[s][0].mean()) if len(edges[s][0]) else 0.0 for s in PM.SIDES},
        side_arc_lip={s: float((edges[s][0] & edges[s][1]).mean()) if len(edges[s][0]) else 0.0 for s in PM.SIDES},
        water_free=1.0 - float(((qf > DELAM_Q) | blister)[on].mean()),
        surface_free=1.0 - float(((g.wet_worked > 0) | (g.dwc >= DWC_DUST) | burn_cells)[on].mean()))


def lin(v: float, v0: float, thr: float, lower: bool = False) -> float:
    """Linear credit from the start value to the threshold (T:547-553): lower-is-better (mm) starts from 2x the
    threshold when the start already passes, higher-is-better from 0."""
    if lower:
        v0 = v0 if v0 > thr else 2.0 * thr
        return float(np.clip((v0 - v) / (v0 - thr), 0.0, 1.0))
    v0 = v0 if v0 < thr else 0.0
    return float(np.clip((v - v0) / (thr - v0), 0.0, 1.0))


@dataclass
class FinishVerdict:
    criteria: list[Criterion]
    success: bool
    score: float                         # 1 on success, else 0.95 x progress (T:851-853)
    progress: float
    terms: dict[str, float] = field(default_factory=dict)
    side_radius: dict[str, float] = field(default_factory=dict)     # per side: share of lines on the arc
    side_arc_lip: dict[str, float] = field(default_factory=dict)    # per side: on the arc and the lip passed

    def __getitem__(self, name: str) -> Criterion:
        return next(c for c in self.criteria if c.name == name)


def progress(m: FinishMeasure, m0: FinishMeasure, p: PadScalars) -> tuple[float, dict[str, float]]:
    """T:555-596: level, edge, stone and finish terms, scaled by the defect-free shares and the tools put back."""
    by = {c.name: c for c in m.criteria}

    def term(name: str) -> float:
        c = by[name]
        return 1.0 if c.passed else FAIL_TERM * lin(c.value, m0.value(name), c.threshold, c.unit == "mm")

    level = 0.25 * term("flat_plane") + 0.25 * term("flat_straightedge") + \
        0.5 * (1.0 if by["flat_straightedge"].passed else FAIL_TERM * lin(m.placements, m0.placements, 1.0))
    if by["radius_profile"].passed and by["lip_passed"].passed:
        edge = 1.0
    else:
        edge = FAIL_TERM * float(np.mean([0.5 * lin(m.side_radius[s], m0.side_radius[s], 0.90)
                                          + 0.5 * lin(m.side_arc_lip[s], m0.side_radius[s], 0.90) for s in PM.SIDES]))
    sh, sd = by["smooth_hard"], by["stone_down"]
    finish = 1.0 if sh.passed else FAIL_TERM * min(0.5 * lin(sh.value, m0.value("smooth_hard"), TEX_SHARE)
                                                   + 0.5 * lin(m.credit, m0.credit, TEX_SHARE), 1.0 - m.finish_left)
    stone = 1.0 if sd.passed else FAIL_TERM * min(lin(sd.value, m0.value("stone_down"), AGG_SHARE),
                                                  1.0 - m.stone_left)
    tidy = 0.9 + 0.1 * (p.tools_total - p.tools_off) / max(p.tools_total, 1)
    work = m.water_free * (level + edge + m.surface_free * (stone + finish)) / 4.0
    return work * tidy, {"level": level, "edge": edge, "stone": stone, "finish": finish, "work": work, "tools": tidy}


def pad_finish(grid: PasteGrid, start: PasteGrid, x: np.ndarray, y: np.ndarray, age: np.ndarray,
               temp_c: float = 21.1, pad: PadScalars | None = None, start_age: float | None = None) -> FinishVerdict:
    """pad-finish verdict. ``start`` is the reset state (its D and E are the credit baselines, T:212-216); ``age``
    the material clock per cell now [s] (the committed water's remaining bleed); ``start_age`` the reset's (their
    pad-finish start, 5 min MASCO-scaled, by default; progress reads none of the start's water terms)."""
    p = pad or PadScalars()
    a0 = AGES_21C["pad-finish"] * CM.masco_factor(temp_c) if start_age is None else start_age
    m0 = measure(start, x, y, start, start.D, start.E, remaining_bleed(np.full(start.h.shape, a0), temp_c),
                 PadScalars())
    m = measure(grid, x, y, start, start.D, start.E, remaining_bleed(age, temp_c), p)
    value, terms = progress(m, m0, p)
    success = all(c.passed for c in m.criteria)
    return FinishVerdict(m.criteria, success, 1.0 if success else FAIL_FACTOR * value, value, terms,
                         side_radius=m.side_radius, side_arc_lip=m.side_arc_lip)
