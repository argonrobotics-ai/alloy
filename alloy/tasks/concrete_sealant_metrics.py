# SPDX-License-Identifier: Apache-2.0
"""tool-sealant verdict: Gym-Anything's five criteria (``concrete_finishing/task.py:654-680``) and progress-v1
(``task_set_progress.py:174-197``) on the concrete tasks' metrics framework (``concrete_metrics``,
``concrete_progress``).

On the joint's segment between the tape marks (z 0.45-2.00 m): the centre line (|a| < 2 mm, a = y - CJ_Y) must lie
0.5-2.5 mm below the wall faces with at least 6 mm of sealant on 95 % of it; 98 % of the segment's rows must touch
both walls (a contact flag on an edge cell); no joint cell may hold under 1 mm; sealant beyond 1 mm of the joint's
edges (t > 0.2 mm) stays under 2 cm² per metre; and every tooled cell was worked before the skin formed, none torn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alloy.tasks import concrete_metrics as MX
from alloy.tasks.concrete_progress import Quantity, band, clip, falloff
from alloy.tasks.concrete_sealant_start import CJ_W, CJ_Y, SEGMENT, Joint

CENTRE_MM = (0.5, 2.5)                   # mm below the faces (params.py:183, sealant_centre_mm)
MIN_MM = 6.0                             # mm of sealant on the centre line (sealant_min_mm, Sikaflex-1A)
CONTACT_SHARE = 0.98                     # share of the rows wetted against both walls (params.py:184)
SMEAR_CM2_M = 2.0                        # cm² per metre (params.py:183)
CENTRE_HALF = 0.002                      # m: the centre line |a| < 2 mm (task.py:657)
VOID_T = 1.0e-3                          # m: a joint cell under this is a void (task.py:665)
EDGE_OFF = 0.001                         # m: smear counts beyond this past the joint's edges (task.py:666)


@dataclass
class SealantGrid:
    """The joint region's fields: the paste grid ``g`` (h, t, cell, paste_mask = the joint, worked, off_m3), each
    cell's ``across`` (y - CJ_Y) and ``z`` [m], the wall edge masks ``edge_l`` / ``edge_r`` and contact flags
    ``contact_l`` / ``contact_r``, the age at last working ``seal_age`` [s], ``skin_torn``, and the skin age [s]."""

    g: MX.PasteGrid
    across: np.ndarray
    z: np.ndarray
    edge_l: np.ndarray
    edge_r: np.ndarray
    contact_l: np.ndarray
    contact_r: np.ndarray
    seal_age: np.ndarray
    skin_torn: np.ndarray
    skin: float


def _mean(v: np.ndarray) -> float:
    return float(v.mean()) if v.size else 0.0


def _rows(s: SealantGrid, seg: np.ndarray, edge: np.ndarray, contact: np.ndarray) -> np.ndarray:
    """Per row of the segment (the grid's second axis): the best contact on that wall's edge cells."""
    return np.max(np.where(seg & edge, np.clip(contact, 0.0, 1.0), 0.0), axis=0)[seg.any(axis=0)]


def _measure(s: SealantGrid, ref: SealantGrid | None) -> MX._Measure:
    g = s.g
    inseg = (s.z >= SEGMENT[0]) & (s.z <= SEGMENT[1])
    seg = g.paste_mask & inseg
    centre = seg & (np.abs(s.across) < CENTRE_HALF)
    depth, t = -g.h[centre], g.t[centre]
    lo, hi = CENTRE_MM
    conc = MX._share((depth >= lo * 1e-3) & (depth <= hi * 1e-3) & (t >= MIN_MM * 1e-3))
    left, right = _rows(s, seg, s.edge_l, s.contact_l), _rows(s, seg, s.edge_r, s.contact_r)
    both = min(MX._share(left > 0), MX._share(right > 0))
    void = int((seg & (g.t < VOID_T)).sum())
    face = ~g.paste_mask & inseg & (np.abs(s.across) > CJ_W / 2 + EDGE_OFF)
    length = SEGMENT[1] - SEGMENT[0]
    smear = float((face & (g.t > MX.SMEAR_T)).sum()) * g.cell**2 * 1e4 / length
    worked = seg & (g.worked > 0)
    before = bool(worked.any()) and float(s.seal_age[worked].max()) < s.skin and not bool(s.skin_torn[seg].any())
    crits = [MX._c("concave_and_thick", conc, conc >= 0.95, 0.95),
             MX._c("wall_contact", both, both >= CONTACT_SHARE, CONTACT_SHARE),
             MX._c("no_void", void, void == 0, 0, "cells"),
             MX._c("smear", smear, smear < SMEAR_CM2_M and MX._off_ok(g), SMEAR_CM2_M, "cm2/m"),
             MX._c("tooled_before_skin", float(before), before, 1.0)]
    ok = {c.name: c.passed for c in crits}
    grade = np.minimum(band(depth * 1e3, lo, hi, 6.0), np.clip(t / (MIN_MM * 1e-3), 0.0, 1.0))      # TSP:178-181
    deficit = _mean(np.maximum(VOID_T - g.t[seg], 0.0))
    res = MX.smear_residue(g.t, face, g.cell, g.off_m3, length)
    q = {"concave_and_thick": MX._blend(conc, _mean(grade), 0.95, ok["concave_and_thick"], "concave_and_thick"),
         "wall_contact": Quantity(min(_mean(left), _mean(right)), CONTACT_SHARE, ok["wall_contact"], 1.0,
                                  "wall_contact"),
         "no_void": Quantity(-deficit, 0.0, ok["no_void"], clip(1.0 - deficit / 2.0e-5), "no_void"),
         "smear": Quantity(-res, 0.0, ok["smear"], falloff(res, 0.0, 6.0), "smear")}
    bad = (s.skin_torn > 0) | (worked & (s.seal_age >= s.skin))       # unworked cells are neutral (TSP:192-197)
    return crits, q, {"skin_sound": 1.0 - _mean(bad[seg])}


def start_grid(j: Joint, skin: float) -> SealantGrid:
    """The start as a grid: the bead's surface, its thickness over the backer, nothing worked."""
    z = np.zeros(j.joint.shape)
    t = np.where(j.joint, j.top - j.base, 0.0)
    return SealantGrid(g=MX.PasteGrid(h=j.top, t=t, cell=j.cell, paste_mask=j.joint, worked=z), across=j.y - CJ_Y,
                       z=j.z, edge_l=j.edge_l, edge_r=j.edge_r, contact_l=j.contact_l, contact_r=j.contact_r,
                       seal_age=z, skin_torn=z, skin=skin)


def tool_sealant(grid: SealantGrid, start: SealantGrid | None = None) -> MX.Verdict:
    """tool-sealant verdict (five criteria, T:654-680; progress-v1 with a start grid, TSP:174-197)."""
    return MX._finish("tool-sealant", _measure, grid, start)


__all__ = ["SealantGrid", "start_grid", "tool_sealant"]
