# SPDX-License-Identifier: Apache-2.0
"""Box-box line contacts on MuJoCo C data in numpy: the CPU route's fallback for BUL-0014 (SCR-WP8-6).

The rule and constants are :mod:`alloy.contact.mjw_boxbox_line` (``box_box_line``) and its C port
(``alloy/route/_mjc/boxbox_line.c``): when MuJoCo's ``mjc_BoxBox`` leaves exactly one contact on a box pair and the
two boxes' support features along its normal are an edge on a face or two parallel edges, the overlap segment's two
endpoints replace it. The C port does this inside the collision call. Here it runs after ``mj_step1``: a fired pair's
contact is rewritten and its second endpoint inserted right after it (``mj_addContact`` from that index on, so the
contact order is the C port's), then the constraint rows are rebuilt (``mj_makeConstraint``, ``mj_island`` unless
islands are disabled, ``mj_projectConstraint``, ``mj_referenceConstraint``). The new contacts' frames come from the
normal with a zero tangent (``mju_makeFrame``), as MuJoCo builds a primitive collider's frame. The band is
``margin + gap`` of the pair (explicit pair rows) or of the two geoms, what MuJoCo passes the collider. On T12 the rule
never fired in the 3,200 replay steps of the spike census but does in B0 episodes; the check itself is a vectorized
scan of ``d.contact``.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import mujoco
import numpy as np

FEATURE_SIN = 0.05
SEGMENT_MIN = 1.0e-4
_EDGE_PAR = math.sqrt(1.0 - FEATURE_SIN * FEATURE_SIN)


@dataclass(frozen=True)
class Line:
    dist: tuple[float, float]
    pos: tuple[np.ndarray, np.ndarray]
    normal: np.ndarray


def feature(rot: np.ndarray, d: np.ndarray) -> tuple[int, int]:
    """Support feature of a box along ``d``: (kind, axis), kind 0 vertex, 1 edge along axis, 2 face with normal axis."""
    perp, other, n_perp = -1, -1, 0
    for k in range(3):
        if abs(float(rot[:, k] @ d)) < FEATURE_SIN:
            n_perp, perp = n_perp + 1, k
        else:
            other = k
    return {1: (1, perp), 2: (2, other)}.get(n_perp, (0, -1))


def _support_edge(pos: np.ndarray, rot: np.ndarray, size: np.ndarray, d: np.ndarray, a: int) -> np.ndarray:
    base = pos.copy()
    for j in range(3):
        if j != a:
            r = rot[:, j]
            base = base + (1.0 if float(r @ d) >= 0.0 else -1.0) * size[j] * r
    return base


def _clip(span: tuple[float, float], alpha: float, beta: float, bound: float) -> tuple[float, float]:
    """Restrict span to alpha + beta t <= bound (empty: lo > hi)."""
    if abs(beta) < 1.0e-12:
        return (1.0, -1.0) if alpha > bound else span
    t = (bound - alpha) / beta
    return (span[0], min(span[1], t)) if beta > 0.0 else (max(span[0], t), span[1])


def _edge_on_face(face: tuple, fa: int, toward: np.ndarray, edge: tuple, ea: int, margin: float):
    fpos, frot, fsize = face
    r = frot[:, fa]
    f = r if float(r @ toward) >= 0.0 else -r
    cf = fpos + fsize[fa] * f
    u, v = frot[:, (fa + 1) % 3], frot[:, (fa + 2) % 3]
    hu, hv = fsize[(fa + 1) % 3], fsize[(fa + 2) % 3]
    base = _support_edge(*edge, -toward, ea)
    e = edge[1][:, ea]
    w = base - cf
    span = (-float(edge[2][ea]), float(edge[2][ea]))
    for a, b, bound in ((w @ u, e @ u, hu), (-(w @ u), -(e @ u), hu), (w @ v, e @ v, hv), (-(w @ v), -(e @ v), hv),
                        (w @ f, e @ f, margin)):
        span = _clip(span, float(a), float(b), float(bound))
    return span, base, e, f, cf


def box_box_line(b1: tuple, b2: tuple, normal: np.ndarray, margin: float) -> Line | None:
    """Two contacts spanning an edge-on-face or parallel-edge overlap along ``normal`` (box1 -> box2), or None."""
    f1, f2 = feature(b1[1], normal), feature(b2[1], -normal)
    seg_min = SEGMENT_MIN * float(np.sum(b1[2]) + np.sum(b2[2]))
    if {f1[0], f2[0]} == {1, 2}:
        if f1[0] == 2:
            span, base, e, f, cf = _edge_on_face(b1, f1[1], normal, b2, f2[1], margin)
            sign = 1.0
        else:
            span, base, e, f, cf = _edge_on_face(b2, f2[1], -normal, b1, f1[1], margin)
            sign = -1.0
        if span[1] - span[0] <= seg_min:
            return None
        q0, q1 = base + span[0] * e, base + span[1] * e
        d0, d1 = float((q0 - cf) @ f), float((q1 - cf) @ f)
        return Line((d0, d1), (q0 - 0.5 * d0 * f, q1 - 0.5 * d1 * f), sign * f)
    if f1[0] == 1 and f2[0] == 1:
        return _parallel_edges(b1, b2, f1[1], f2[1], normal, margin, seg_min)
    return None


def _parallel_edges(b1: tuple, b2: tuple, a1: int, a2: int, n: np.ndarray, margin: float,
                    seg_min: float) -> Line | None:
    e, e2 = b1[1][:, a1], b2[1][:, a2]
    if abs(float(e @ e2)) < _EDGE_PAR:
        return None   # crossing edges: MuJoCo's closest point is the contact
    e2 = e2 if float(e @ e2) >= 0.0 else -e2
    p1 = _support_edge(*b1, n, a1)
    p2 = _support_edge(*b2, -n, a2)
    c1, c2 = float(p1 @ e), float(p2 @ e)
    span = (max(c1 - b1[2][a1], c2 - b2[2][a2]), min(c1 + b1[2][a1], c2 + b2[2][a2]))
    span = _clip(span, float((p2 - c2 * e2 - p1 + c1 * e) @ n), float((e2 - e) @ n), margin)
    if span[1] - span[0] <= seg_min:
        return None
    pa = [p1 + (s - c1) * e for s in span]
    pb = [p2 + (s - c2) * e2 for s in span]
    return Line((float((pb[0] - pa[0]) @ n), float((pb[1] - pa[1]) @ n)),
                (0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1])), n.copy())


def make_frame(normal: np.ndarray) -> np.ndarray:
    """mju_makeFrame with a zero tangent: [normal, y, z] (9 values)."""
    x = normal / np.linalg.norm(normal)
    y = np.array([0.0, 1.0, 0.0]) if abs(x[1]) < 0.5 else np.array([0.0, 0.0, 1.0])
    y = y - x * float(x @ y)
    y = y / np.linalg.norm(y)
    return np.concatenate([x, y, np.cross(x, y)])


class LineRule:
    """The rule for one model (its box geoms and explicit pairs' bands)."""

    def __init__(self, m: mujoco.MjModel) -> None:
        self.isbox = np.asarray(m.geom_type) == int(mujoco.mjtGeom.mjGEOM_BOX)
        self.any = bool(self.isbox.sum() >= 2)
        self.ngeom = int(m.ngeom)
        self.pair_band = {tuple(sorted((int(a), int(b)))): float(mg) + float(gp) for a, b, mg, gp in
                          zip(m.pair_geom1, m.pair_geom2, m.pair_margin, m.pair_gap)}

    def singles(self, d: mujoco.MjData) -> np.ndarray:
        """Contact ids where a box-box collider call left exactly one contact: MuJoCo writes each call's contacts
        together, so a run of length 1 of the same geom pair (the C hook's ``n == 1``)."""
        if not self.any or d.ncon == 0:
            return np.zeros(0, dtype=np.int64)
        g = np.asarray(d.contact.geom[: d.ncon])
        bb = (g >= 0).all(axis=1) & self.isbox[np.maximum(g[:, 0], 0)] & self.isbox[np.maximum(g[:, 1], 0)]
        if not bb.any():
            return np.zeros(0, dtype=np.int64)
        key = g[:, 0].astype(np.int64) * self.ngeom + g[:, 1]
        alone = np.ones(key.size, dtype=bool)
        same = key[1:] == key[:-1]
        alone[1:] &= ~same
        alone[:-1] &= ~same
        return np.flatnonzero(bb & alone)

    def band(self, m: mujoco.MjModel, g1: int, g2: int) -> float:
        hit = self.pair_band.get((min(g1, g2), max(g1, g2)))
        if hit is not None:
            return hit
        return max(float(m.geom_margin[g1]), float(m.geom_margin[g2])) + max(float(m.geom_gap[g1]),
                                                                             float(m.geom_gap[g2]))

    def apply(self, m: mujoco.MjModel, d: mujoco.MjData) -> int:
        """Rewrite the fired pairs after ``mj_step1`` and rebuild the constraint rows; returns the pairs that fired."""
        fired: dict[int, Line] = {}
        for i in self.singles(d).tolist():
            g1, g2 = (int(x) for x in d.contact.geom[i])
            box = [(np.array(d.geom_xpos[g]), np.array(d.geom_xmat[g]).reshape(3, 3), np.array(m.geom_size[g]))
                   for g in (g1, g2)]
            line = box_box_line(box[0], box[1], np.array(d.contact.frame[i][:3]), self.band(m, g1, g2))
            if line is not None:
                fired[i] = line
        if fired:
            _rewrite(m, d, fired)
        return len(fired)


def _rewrite(m: mujoco.MjModel, d: mujoco.MjData, fired: dict[int, Line]) -> None:
    first = min(fired)
    tail = [copy.copy(d.contact[i]) for i in range(first, int(d.ncon))]
    d.ncon = first
    for k, con in enumerate(tail, start=first):
        line = fired.get(k)
        if line is None:
            mujoco.mj_addContact(m, d, con)
            continue
        frame = make_frame(line.normal)
        for dist, pos in zip(line.dist, line.pos):
            c = copy.copy(con)
            c.dist, c.pos, c.frame = dist, pos, frame
            mujoco.mj_addContact(m, d, c)
    mujoco.mj_makeConstraint(m, d)
    if not int(m.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_ISLAND):
        mujoco.mj_island(m, d)
    mujoco.mj_projectConstraint(m, d)
    mujoco.mj_referenceConstraint(m, d)


__all__ = ["FEATURE_SIN", "SEGMENT_MIN", "Line", "LineRule", "box_box_line", "feature", "make_frame"]
