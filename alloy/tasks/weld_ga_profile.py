# SPDX-License-Identifier: Apache-2.0
"""A slice's fillet read on its free surface, as GA reads a slice that has a pool profile (SCR-WP8-14; a port of GA's
``sections.profile_section``). Heat v3's surface is the cap model's (``grid_cap``): the wedge between the joint's
rays up to the toe chord plus the circular segment on that chord holding the slice's area, rebuilt here from the
slice's toes and area.

The cells are checked against the surface (weld metal iff a cell's centre lies under it): weld cells the surface does
not explain ("extra": frozen outside a cap that moved on) are kept whole, cells it claims that hold no metal
("missing") are removed, and beyond the toes the cells alone describe the section. The section is rasterised at
1/5 cell, and:
- legs: the toes, unless whole extra cells carry weld metal up a member past the toe (the cells' run is the leg) or
  the metal along the member stops short of the toe by more than a cell (the raster's run);
- size, throat, convexity: on the surface itself when the cells match it exactly; with extra cells, the larger of the
  surface's and the raster's throat and size and the raster's convexity; with missing cells, the raster's.
"""

from __future__ import annotations

import math

import numpy as np

SUBCELL = 5


def segment_angle(chord: float, area: float) -> float:
    """θ of the circular segment on ``chord`` holding ``area`` (bisection, as ``grid_cap.cap_angle``)."""
    lo, hi = 1e-9, 2 * math.pi - 1e-9
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if chord * chord * (mid - math.sin(mid)) / (8 * math.sin(0.5 * mid) ** 2) < area:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class Cap:
    """The cap on toes t1 (out along the plate, y) and t2 (up the bracket, z) holding ``area`` [m²], root at 0."""

    def __init__(self, t1: float, t2: float, area: float):
        self.p1, self.p2 = np.array([t1, 0.0]), np.array([0.0, t2])
        self.mid = 0.5 * (self.p1 + self.p2)
        self.chord = float(np.hypot(t1, t2))
        d = self.p2 - self.p1
        self.m = np.array([-d[1], d[0]]) / self.chord
        if self.m @ self.mid < 0:
            self.m = -self.m                           # outward, away from the root
        rest = area - 0.5 * t1 * t2
        self.convex = rest >= 0
        th = segment_angle(self.chord, rest) if self.convex else min(segment_angle(self.chord, -rest), math.pi)
        self.radius = self.chord / (2 * math.sin(0.5 * th))
        sign = -1.0 if self.convex else 1.0
        self.centre = self.mid + sign * self.radius * math.cos(0.5 * th) * self.m

    def under(self, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Points inside the bead (``grid_cap.in_cap``)."""
        side = (y - self.mid[0]) * self.m[0] + (z - self.mid[1]) * self.m[1]
        wedge = (side <= 0) & (y >= 0) & (z >= 0)
        inside = np.hypot(y - self.centre[0], z - self.centre[1]) <= self.radius
        return (wedge | ((side > 0) & inside)) if self.convex else (wedge & ~inside)

    def span(self, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Points the surface describes: between the toes along the chord."""
        u = ((y - self.p1[0]) * (self.p2 - self.p1)[0] + (z - self.p1[1]) * (self.p2 - self.p1)[1]) / self.chord
        return (u >= 0) & (u <= self.chord)

    def polyline(self, n: int = 400) -> np.ndarray:
        """The free surface from the plate's toe to the bracket's, densely."""
        a1 = math.atan2(*(self.p1 - self.centre)[::-1])
        a2 = math.atan2(*(self.p2 - self.centre)[::-1])
        for da in ((a2 - a1) % (2 * math.pi), (a2 - a1) % (2 * math.pi) - 2 * math.pi):
            a = a1 + da * np.linspace(0.0, 1.0, n)
            pts = self.centre + self.radius * np.stack([np.cos(a), np.sin(a)], axis=1)
            side = (pts[n // 2] - self.mid) @ self.m
            if (side > 0) == self.convex:
                return pts
        return pts


def _sub(edges: np.ndarray, lo: int, hi: int, sub: int) -> tuple[np.ndarray, np.ndarray]:
    """Cells lo..hi split ``sub`` ways: the fine edges and each fine cell's coarse index."""
    parts = [np.linspace(edges[i], edges[i + 1], sub + 1)[:-1] for i in range(lo, hi)]
    fine = np.concatenate([*parts, edges[hi:hi + 1]])
    return fine, np.repeat(np.arange(lo, hi), sub)


def _run(line: np.ndarray, edges: np.ndarray, k0: int) -> float:
    """The contiguous run of True along a member from the root, as an edge."""
    if not line[0]:
        return 0.0
    return float(edges[k0 + (int(np.argmin(line)) if not line.all() else len(line))])


def _leg(run: float, run_c: float, toe: float, extra_line: np.ndarray, edges_hi: np.ndarray, cell: float):
    if run <= 0 and run_c <= 0:
        return 0.0, "cells"
    beyond = extra_line & (edges_hi > toe + 1e-9)
    if beyond.any() and run_c >= float(edges_hi[beyond].min()) - 1e-9:
        return run_c, "cells"
    if run < toe - cell - 1e-9:
        return run, "cells"
    return toe, "profile"


def read(cap: Cap, weld: np.ndarray, orig: np.ndarray, ey: np.ndarray, ez: np.ndarray, sub: int = SUBCELL) -> dict:
    """GA's profile reading of one slice (``weld``, ``orig``: [ny, nz] weld / parent metal; ``ey``, ``ez``: edges)."""
    yc, zc = 0.5 * (ey[1:] + ey[:-1]), 0.5 * (ez[1:] + ez[:-1])
    jb, kb = int(np.searchsorted(yc, 0.0)), int(np.searchsorted(zc, 0.0))
    Y, Z = np.meshgrid(yc, zc, indexing="ij")
    q = (Y > 0) & (Z > 0)
    span = cap.span(Y, Z)
    pred, act = span & cap.under(Y, Z) & ~orig & q, weld & q
    extra, missing = act & ~pred, pred & ~act
    pts = cap.polyline()
    wy = max(float(pts[:, 0].max()), float(Y[act].max()) if act.any() else 0.0)
    wz = max(float(pts[:, 1].max()), float(Z[act].max()) if act.any() else 0.0)
    j1 = min(int(np.searchsorted(ey, wy)) + 2, len(yc))
    k1 = min(int(np.searchsorted(ez, wz)) + 2, len(zc))
    fy, pj = _sub(ey, max(jb - 1, 0), j1, sub)
    fz, pk = _sub(ez, max(kb - 1, 0), k1, sub)
    fyc, fzc = 0.5 * (fy[1:] + fy[:-1]), 0.5 * (fz[1:] + fz[:-1])
    Yf, Zf = np.meshgrid(fyc, fzc, indexing="ij")
    ix = np.ix_(pj, pk)
    orig_f, weld_c = orig[ix], weld[ix]
    span_f, qf = cap.span(Yf, Zf), (Yf > 0) & (Zf > 0)
    weld_f = ((span_f & cap.under(Yf, Zf) & ~orig_f & qf) | (~span_f & weld_c) | extra[ix]) & ~missing[ix]
    metal_f = weld_f | orig_f
    jbf, kbf = int(np.searchsorted(fyc, 0.0)), int(np.searchsorted(fzc, 0.0))
    cell_v, cell_h = float(ez[kb + 1] - ez[kb]), float(ey[jb + 1] - ey[jb])
    leg_v, src_v = _leg(_run(weld_f[jbf, kbf:], fz, kbf), _run(weld[jb, kb:], ez, kb), float(cap.p2[1]),
                        extra[jb, :], ez[1:], cell_v)
    leg_h, src_h = _leg(_run(weld_f[jbf:, kbf], fy, jbf), _run(weld[jb:, kb], ey, jb), float(cap.p1[0]),
                        extra[:, kb], ey[1:], cell_h)
    r = dict(leg_vertical_m=leg_v, leg_horizontal_m=leg_h, profile_extra_cells=int(extra.sum()),
             profile_missing_cells=int(missing.sum()), geometry="profile")
    air_f = ~metal_f & qf
    nb = np.zeros_like(weld_f)
    nb[1:, :] |= weld_f[:-1, :]
    nb[:-1, :] |= weld_f[1:, :]
    nb[:, 1:] |= weld_f[:, :-1]
    nb[:, :-1] |= weld_f[:, 1:]
    face = air_f & nb & (Yf <= leg_h + 2e-3) & (Zf <= leg_v + 2e-3)
    if not (leg_v > 0 and leg_h > 0 and face.any()):
        r.update(throat_m=0.0, size_m=0.0, convexity_m=0.0, concavity_m=0.0, face_width_m=0.0)
        return r
    half = 0.5 * float(min(np.diff(fy).min(), np.diff(fz).min()))
    p0, p1 = np.array([0.0, leg_v]), np.array([leg_h, 0.0])
    wl = float(np.linalg.norm(p1 - p0))
    tl = (p1 - p0) / wl
    nl = np.array([tl[1], -tl[0]])
    nl = -nl if nl @ np.array([1.0, 1.0]) < 0 else nl
    sd = (Yf - p0[0]) * nl[0] + (Zf - p0[1]) * nl[1]
    ud = (Yf - p0[0]) * tl[0] + (Zf - p0[1]) * tl[1]
    on = face & (ud > 0) & (ud < wl)
    s_air = float((Yf + Zf)[air_f].min()) - half if air_f.any() else np.inf
    ras = dict(throat_m=float(max(0.0, (np.sqrt(Yf**2 + Zf**2) - half)[face].min())),
               size_m=float(max(0.0, min(leg_v, leg_h, s_air))),
               convexity_m=float(max(0.0, (sd - half)[on].max())) if on.any() else 0.0,
               concavity_m=float(max(0.0, -(sd - half)[on].min())) if on.any() else 0.0)
    sp, up = (pts - p0) @ nl, (pts - p0) @ tl
    onp = (up > 0) & (up < wl)
    pol = dict(throat_m=float(np.hypot(pts[:, 0], pts[:, 1]).min()),
               size_m=float(max(0.0, min(leg_v, leg_h, float((pts[:, 0] + pts[:, 1]).min())))),
               convexity_m=float(max(0.0, sp[onp].max())) if onp.any() else 0.0,
               concavity_m=float(max(0.0, -sp[onp].min())) if onp.any() else 0.0)
    if not missing.any() and not extra.any() and src_v == src_h == "profile":
        r.update(pol)
    elif not missing.any():
        r.update(throat_m=max(pol["throat_m"], ras["throat_m"] - half), size_m=max(pol["size_m"], ras["size_m"] - half),
                 convexity_m=ras["convexity_m"], concavity_m=ras["concavity_m"])
    else:
        r.update(ras)
    r["face_width_m"] = wl
    return r


__all__ = ["Cap", "read", "segment_angle"]
