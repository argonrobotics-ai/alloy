# SPDX-License-Identifier: Apache-2.0
"""Heat v3 slice 11: a slice's free surface with the liquid's weight, in numpy (the reference for the laws and for
``grid_surface``; docs/notes/heat-v3-design.md §Slice 11, GA pool.py ``_solve_profiles`` and ``_caps``).

The surface is a height function h(u) over the toe chord (p1 on ray e1, p2 on ray e2), ``n`` intervals, h = 0 at
the toes, measured along the chord's normal m away from the root. It minimises, per unit length of weld,

    F[h] = gamma * length(h) + sum du (lin_i h_i + quad h_i^2 / 2),
    lin_i = -rho g . c_i - P + p_arc_i,    quad = -rho g . m,

(surface tension, the liquid's weight, the pool's pressure P and the arc's pressure), whose stationarity is
Young-Laplace gamma kappa = P + rho g . r - p_arc. F is convex where gravity holds the liquid toward the chord
(quad >= 0), so the minimiser is unique and the held area grows with P. Solved by projected Newton on the
tridiagonal Hessian with a backtracking line search; h is kept within [hmin, hmax] (solid metal below, frozen
stations above).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

N_NODES = 24                       # chord intervals (GA N_NODES)
TOE_LIMIT = math.radians(88.0)     # the steepest face at a toe before the slice overflows (GA POOL_TOE_SLOPE_MAX)


@dataclass(frozen=True)
class Chord:
    """A slice's toe chord: root, the two rays and the toes [m]; ``down``: gravity's unit direction in the plane."""

    root: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    t1: float
    t2: float

    @property
    def p1(self) -> np.ndarray:
        return self.root + self.t1 * self.e1

    @property
    def p2(self) -> np.ndarray:
        return self.root + self.t2 * self.e2

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.p2 - self.p1))

    @property
    def tangent(self) -> np.ndarray:
        return (self.p2 - self.p1) / self.length

    @property
    def normal(self) -> np.ndarray:
        """The chord's unit normal away from the root (up from a flat plate)."""
        d = self.tangent
        m = np.array([-d[1], d[0]])
        lean = float(m @ (0.5 * (self.p1 + self.p2) - self.root))
        if abs(lean) < 1e-12:
            m = np.array([-self.e1[1], self.e1[0]])
            m = m if m[1] >= 0 else -m
        elif lean < 0:
            m = -m
        return m

    @property
    def wedge(self) -> float:
        """The area between the rays and the chord [m^2]."""
        a, b = self.p1 - self.root, self.p2 - self.root
        return 0.5 * abs(float(a[0] * b[1] - a[1] * b[0]))


def nodes(ch: Chord, n: int = N_NODES) -> tuple[np.ndarray, float]:
    """The chord's node points (n + 1, 2) and the interval du."""
    u = np.linspace(0.0, ch.length, n + 1)
    return ch.p1[None, :] + u[:, None] * ch.tangent[None, :], ch.length / n


def solve(ch: Chord, p: float, gamma: float, rho: float, gravity: np.ndarray, p_arc: np.ndarray | None = None,
          hmin: np.ndarray | None = None, hmax: np.ndarray | None = None, n: int = N_NODES,
          h0: np.ndarray | None = None, iters: int = 60) -> np.ndarray:
    """The surface heights h (n + 1,) at pool pressure ``p`` [Pa]; ``gravity`` [m/s^2] in the slice plane."""
    c, du = nodes(ch, n)
    m = ch.normal
    pa = np.zeros(n + 1) if p_arc is None else np.asarray(p_arc, float)
    lin = (-rho * (c @ gravity) - p + pa)[1:-1]
    quad = -rho * float(m @ gravity)
    lo = np.full(n - 1, -np.inf) if hmin is None else np.asarray(hmin, float)[1:-1]
    hi = np.full(n - 1, np.inf) if hmax is None else np.asarray(hmax, float)[1:-1]
    h = np.zeros(n + 1) if h0 is None else np.array(h0, float)
    h[0] = h[-1] = 0.0
    h[1:-1] = np.clip(h[1:-1], lo, np.maximum(hi, lo))

    def energy(x: np.ndarray) -> float:
        s = np.diff(x) / du
        tension = gamma * du * float(np.sqrt(1.0 + s * s).sum())
        return tension + du * float((lin * x[1:-1] + 0.5 * quad * x[1:-1] ** 2).sum())

    f = energy(h)
    for _ in range(iters):
        s = np.diff(h) / du
        q = 1.0 / np.sqrt(1.0 + s * s)
        grad = gamma * (s[:-1] * q[:-1] - s[1:] * q[1:]) + du * (lin + quad * h[1:-1])
        cq = q ** 3
        diag = gamma / du * (cq[:-1] + cq[1:]) + du * quad
        off = -gamma / du * cq[1:-1]
        step = -thomas(off, diag, off, grad)
        t, done = 1.0, False
        while t > 1e-6:
            trial = h.copy()
            trial[1:-1] = np.clip(h[1:-1] + t * step, lo, np.maximum(hi, lo))
            ft = energy(trial)
            if ft <= f + 1e-4 * t * float(grad @ step) or ft <= f:
                done = abs(f - ft) <= 1e-14 * max(1.0, abs(f))
                h, f = trial, ft
                break
            t *= 0.5
        if t <= 1e-6 or done:
            break
    return h


def thomas(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Solve the tridiagonal system (sub-diagonal a, diagonal b, super-diagonal c) for right-hand side d."""
    n = len(b)
    cp, dp = np.zeros(n), np.zeros(n)
    cp[0], dp[0] = (c[0] / b[0] if n > 1 else 0.0), d[0] / b[0]
    for i in range(1, n):
        den = b[i] - a[i - 1] * cp[i - 1]
        cp[i] = c[i] / den if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / den
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def area(ch: Chord, h: np.ndarray) -> float:
    """The slice's liquid area [m^2]: the wedge plus the area between the chord and the surface (trapezoids)."""
    du = ch.length / (len(h) - 1)
    return ch.wedge + du * float(0.5 * (h[:-1] + h[1:]).sum())


def toe_angle(ch: Chord, h: np.ndarray) -> float:
    """The surface's steepest angle to the chord at either toe [rad], extrapolated from the first two intervals (GA
    ``_toe_angles``: a meniscus' angle varies about linearly near its edge)."""
    du = ch.length / (len(h) - 1)
    a1 = np.arctan(np.array([h[1], h[-2]]) / du)
    a2 = np.arctan(np.array([h[2] - h[1], h[-3] - h[-2]]) / du)
    return float(np.max(1.5 * a1 - 0.5 * a2))


def pressure_for(ch: Chord, target: float, gamma: float, rho: float, gravity: np.ndarray,
                 **kw) -> tuple[float, np.ndarray]:
    """The pool pressure at which the slice holds ``target`` [m^2], by bisection (area grows with P), bracketed by the
    cap pressure: a slice holding more than its cap overflows (returned at the cap)."""
    hi, h = cap_pressure(ch, gamma, rho, gravity, **kw)
    if area(ch, h) <= target:
        return hi, h
    lo = -hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        h = solve(ch, mid, gamma, rho, gravity, h0=h, **kw)
        lo, hi = (mid, hi) if area(ch, h) < target else (lo, mid)
    return hi, solve(ch, hi, gamma, rho, gravity, h0=h, **kw)


def cap_pressure(ch: Chord, gamma: float, rho: float, gravity: np.ndarray, limit: float = TOE_LIMIT,
                 **kw) -> tuple[float, np.ndarray]:
    """The pressure at which the surface meets a toe at ``limit`` (GA ``_caps``): the most the chord holds."""
    lo, hi = 0.0, 2.0 * gamma * math.sin(limit) / ch.length + rho * float(np.linalg.norm(gravity)) * ch.length
    h = None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        h = solve(ch, mid, gamma, rho, gravity, h0=h, **kw)
        lo, hi = (mid, hi) if toe_angle(ch, h) < limit else (lo, mid)
    return hi, solve(ch, hi, gamma, rho, gravity, h0=h, **kw)


__all__ = ["Chord", "N_NODES", "TOE_LIMIT", "area", "cap_pressure", "nodes", "pressure_for", "solve", "thomas",
           "toe_angle"]
