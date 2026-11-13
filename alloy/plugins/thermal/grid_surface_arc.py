# SPDX-License-Identifier: Apache-2.0
"""Heat v3 slice 11: a slice's free surface in polar form about the joint's root, in numpy (the reference for the
laws and the Warp kernel; docs/notes/heat-v3-design.md §Slice 11).

A slice's surface runs from toe p1 = root + t1 e1 to toe p2 = root + t2 e2, the liquid between it and the root, and
obeys Young-Laplace, gamma kappa = P + rho g . (x - root) - p_arc(x), with P the pool's pressure at the root and
p_arc GA's Gaussian arc pressure. Along the polar angle a about the root (0 on e1, ``sweep`` on e2), the surface's
radius r(a), tangent angle phi(a) and held area A(a) obey

    dr/da = r cot(psi),    dphi/da = kappa r / sin(psi),    dA/da = r^2 / 2,    psi = phi - a,

psi in (0, pi) being the angle from the radius to the tangent. So the curve stays between the joint's rays, its toe
angles are exact (no node slopes: a 24-node height function reads an 88 deg toe low and caps 21-45 % high, §6), and
its area is exact. For a pressure, the start angle psi0 at toe 1 is a bracketed root of the miss at the far ray,
r(sweep) - t2, which falls as psi0 grows (a start along the radius runs out past toe 2, one toward the root falls
short of it); the held area grows with P up to the cap, where the surface meets a toe at ``TOE_LIMIT`` to the chord
(GA's rule, pool.py ``_caps``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from alloy.plugins.thermal.grid_surface_ref import TOE_LIMIT, Chord

STEPS = 128                     # RK4 steps over the sweep
SCAN = 32                       # start angles tried between the cap's and the root's before refining
PSI_MIN = 1.0e-3                # a tangent this close to the radius ends a shot (the curve would fold back);
                                # the side it leaves (0: outward, pi: toward the root) says which way it missed


@dataclass(frozen=True)
class Frame:
    """The polar frame of a chord: unit e1 (a = 0) and its normal toward e2, the sweep to e2 and the toes [m]."""

    root: np.ndarray                 # the polar centre (the joint's root, or C for an overhang)
    u: np.ndarray
    v: np.ndarray
    sweep: float
    t1: float
    t2: float
    corner: float = 0.0              # the region's area outside the fan from the centre (0 about the root)
    a1: float = 0.0                  # the members beyond the toes: their directions in the frame
    a2: float | None = None
    guard: tuple | None = None       # (root, e1, m1, t1, e2, m2, t2) local: a point past a toe inside a member touches
    ref: tuple = (0.0, 0.0)          # the joint's root in the frame: the pool's pressure P is the root's

    def world(self, local: np.ndarray) -> np.ndarray:
        return self.root + local[..., :1] * self.u + local[..., 1:2] * self.v

    def local(self, x: np.ndarray) -> np.ndarray:
        d = np.asarray(x, float) - self.root
        return np.stack([d @ self.u, d @ self.v], axis=-1)

    @property
    def chord_angle(self) -> float:
        """The chord's direction from toe 1 to toe 2, in the frame."""
        return math.atan2(self.t2 * math.sin(self.sweep), self.t2 * math.cos(self.sweep) - self.t1)


@dataclass(frozen=True)
class Physics:
    """Surface tension [N/m], density [kg/m^3], gravity in the slice plane [m/s^2], the arc's pressure: its peak
    [Pa] at ``arc_at`` (a point in the plane) and its sigma [m] (0 peak: no arc)."""

    gamma: float
    rho: float
    gravity: np.ndarray
    arc_peak: float = 0.0
    arc_at: np.ndarray = field(default_factory=lambda: np.zeros(2))
    arc_sigma: float = 1.0


@dataclass(frozen=True)
class Shot:
    status: int                  # 0 reached the far ray; +1 ran out along a radius; -1 fell toward the root
    r_end: float
    phi_end: float
    area: float
    points: np.ndarray | None    # (STEPS + 1, 2) in the frame
    touch: float = math.nan      # where along e1 (from the root) the curve first entered the member past toe 1


@dataclass(frozen=True)
class Surface:
    """A solved surface: pressure, start angle, area [m^2], toe angles to the chord and to the solid [rad], its
    points in the plane, and how it was found ("held", "full": at the cap, "thin": the least it can hold).
    ``toes``: where it meets the rays (toe 2 short of the slice's own on a "wet", draining slice)."""

    pressure: float
    psi0: float
    area: float
    toe_chord: tuple[float, float]
    toe_solid: tuple[float, float]
    points: np.ndarray
    status: str = "held"
    toes: tuple[float, float] = (math.nan, math.nan)
    touch: float = math.nan          # where along e1 the face enters the member past toe 1 (a touchdown)


def frame(ch: Chord) -> Frame:
    """The polar frame turning from e1 to e2 through the liquid (the chord's side for a flat plate)."""
    e1 = ch.e1 / np.linalg.norm(ch.e1)
    e2 = ch.e2 / np.linalg.norm(ch.e2)
    perp = np.array([-e1[1], e1[0]])
    cr, dt = float(e1[0] * e2[1] - e1[1] * e2[0]), float(e1 @ e2)
    if abs(cr) < 1e-12 and dt < 0.0:
        side = 1.0 if float(perp @ ch.normal) > 0.0 else -1.0
        sweep = math.pi
    else:
        side = 1.0 if cr > 0.0 else -1.0
        sweep = math.atan2(abs(cr), dt)
    return Frame(np.asarray(ch.root, float), e1, side * perp, sweep, float(ch.t1), float(ch.t2))


def mirrored(ch: Chord) -> Chord:
    """The same chord with its toes swapped (to shoot from toe 2)."""
    return Chord(ch.root, ch.e2, ch.e1, ch.t2, ch.t1)


def local_physics(fr: Frame, ph: Physics) -> tuple:
    g = np.array([float(ph.gravity @ fr.u), float(ph.gravity @ fr.v)])
    return ph.gamma, ph.rho, g, ph.arc_peak, fr.local(ph.arc_at), ph.arc_sigma, np.asarray(fr.ref, float)


def _rhs(a: float, y: np.ndarray, p: float, lp: tuple) -> tuple[int, np.ndarray]:
    gamma, rho, g, peak, at, sig, ref = lp
    r, phi = y[0], y[1]
    psi = phi - a
    s = math.sin(psi)
    if r <= 0.0 or psi > math.pi - PSI_MIN:               # turned toward the root: falls short of toe 2
        return -1, y
    if psi < PSI_MIN:                                      # turned out along the radius: runs past toe 2
        return 1, y
    x = r * math.cos(a), r * math.sin(a)
    pa = 0.0
    if peak != 0.0:
        dx, dz = x[0] - at[0], x[1] - at[1]
        pa = peak * math.exp(-0.5 * (dx * dx + dz * dz) / (sig * sig))
    k = (p + rho * (g[0] * (x[0] - ref[0]) + g[1] * (x[1] - ref[1])) - pa) / gamma
    return 0, np.array([r * math.cos(psi) / s, k * r / s, 0.5 * r * r])


def shoot(fr: Frame, psi0: float, p: float, lp: tuple, steps: int = STEPS, keep: bool = False,
          r_max: float | None = None) -> Shot:
    """Integrate from toe 1 leaving at psi0 to the radius from the root to the far ray (or until it fails)."""
    h = fr.sweep / steps
    y = np.array([fr.t1, psi0, 0.0])
    touch = math.nan
    r_max = 10.0 * max(fr.t1, fr.t2) if r_max is None else r_max
    pts = [y.copy()] if keep else None
    for i in range(steps):
        a = i * h
        st, k1 = _rhs(a, y, p, lp)
        if st == 0:
            st, k2 = _rhs(a + 0.5 * h, y + 0.5 * h * k1, p, lp)
        if st == 0:
            st, k3 = _rhs(a + 0.5 * h, y + 0.5 * h * k2, p, lp)
        if st == 0:
            st, k4 = _rhs(a + h, y + h * k3, p, lp)
        if st != 0:
            return Shot(st, math.nan, math.nan, math.nan, None)
        y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        psi = y[1] - (i + 1) * h
        if y[0] > r_max or psi < PSI_MIN:
            return Shot(1, math.nan, math.nan, math.nan, None)
        if y[0] <= 0.0 or psi > math.pi - PSI_MIN:
            return Shot(-1, math.nan, math.nan, math.nan, None)
        if fr.guard is not None and math.isnan(touch):
            touch = _touches(fr.guard, y[0] * np.array([math.cos((i + 1) * h), math.sin((i + 1) * h)]))
        if pts is not None:
            pts.append(y.copy())
    out = None
    if pts is not None:
        arr = np.array(pts)
        ang = h * np.arange(steps + 1)
        out = np.stack([arr[:, 0] * np.cos(ang), arr[:, 0] * np.sin(ang)], axis=1)
    return Shot(0, float(y[0]), float(y[1]), float(y[2]) + fr.corner, out, touch)


def _touches(guard: tuple, x: np.ndarray) -> float:
    """Along e1 from the root, where x lies inside the member past toe 1 (nan if it does not)."""
    root, e1, m1, t1, _e2, _m2, _t2 = guard
    d = x - root
    along = float(d @ e1)
    return along if along > t1 and float(d @ m1) < 0.0 else math.nan


def root(f, lo: float, hi: float, flo: float, fhi: float, xtol: float, iters: int = 200) -> float:
    """A root of f in [lo, hi] (f(lo), f(hi) of opposite signs; +-inf marks a failed evaluation): Illinois regula
    falsi, bisecting while an end is infinite."""
    side = 0
    for _ in range(iters):
        if hi - lo <= xtol:
            break
        if math.isfinite(flo) and math.isfinite(fhi):
            x = (lo * fhi - hi * flo) / (fhi - flo)
            if not lo < x < hi:
                x = 0.5 * (lo + hi)
        else:
            x = 0.5 * (lo + hi)
        fx = f(x)
        if fx == 0.0:
            return x
        if (fx > 0.0) == (flo > 0.0):
            lo, flo = x, fx
            if side == -1 and math.isfinite(fhi):
                fhi *= 0.5
            side = -1
        else:
            hi, fhi = x, fx
            if side == 1 and math.isfinite(flo):
                flo *= 0.5
            side = 1
    return 0.5 * (lo + hi) if not (math.isfinite(flo) and math.isfinite(fhi)) else (
        lo if abs(flo) < abs(fhi) else hi)


def _miss(f_shot) -> float:
    return f_shot.r_end if f_shot.status == 0 else f_shot.status * math.inf


def _bracket(f, x0: float, scale: float, grow: int = 60) -> tuple[float, float, float, float] | None:
    """lo < hi around x0 with f(lo) > 0 > f(hi) (f falls), widening geometrically."""
    for k in range(grow):
        d = scale * 2.0 ** k
        lo, hi = x0 - d, x0 + d
        flo, fhi = f(lo), f(hi)
        if flo > 0.0 > fhi:
            return lo, hi, flo, fhi
    return None


def _scale(fr: Frame, lp: tuple) -> float:
    chord = math.hypot(fr.t2 * math.cos(fr.sweep) - fr.t1, fr.t2 * math.sin(fr.sweep))
    gamma, rho, g, peak = lp[:4]
    return gamma / max(chord, 1e-9) + rho * float(np.linalg.norm(g)) * chord + abs(peak)


def pressure_at_start(fr: Frame, psi0: float, lp: tuple, steps: int = STEPS) -> float | None:
    """The pressure whose curve leaving toe 1 at psi0 meets the far ray at toe 2. With the start fixed, more pressure
    curves the surface more everywhere, so it meets the far ray nearer the root (or curls back toward the root),
    and less runs it out past toe 2: the miss falls with P, failures included."""
    chord = math.hypot(fr.t2 * math.cos(fr.sweep) - fr.t1, fr.t2 * math.sin(fr.sweep))
    gamma, rho, g = lp[:3]
    mid = 0.5 * np.array([fr.t1 + fr.t2 * math.cos(fr.sweep), fr.t2 * math.sin(fr.sweep)]) - np.asarray(fr.ref)
    p0 = 2.0 * gamma * math.sin(fr.chord_angle - psi0) / chord - rho * float(g @ mid)

    def f(p: float) -> float:
        return _miss(shoot(fr, psi0, p, lp, steps)) - fr.t2

    scale = _scale(fr, lp)
    b = _bracket(f, p0, 1e-2 * scale)
    return None if b is None else root(f, *b, xtol=1e-12 * scale)


def _make(fr: Frame, psi0: float, p: float, lp: tuple, steps: int, status: str) -> Surface | None:
    s = shoot(fr, psi0, p, lp, steps, keep=True)
    if s.status != 0:
        return None
    beta = fr.chord_angle
    a2 = fr.sweep if fr.a2 is None else fr.a2
    return Surface(p, psi0, s.area, (beta - psi0, s.phi_end - beta), (fr.a1 + math.pi - psi0, s.phi_end - a2),
                   fr.world(s.points), status, (fr.t1, s.r_end), s.touch)


def surface_at_toe(ch: Chord, theta1: float, ph: Physics, steps: int = STEPS) -> Surface | None:
    """The surface leaving toe 1 at theta1 to the chord (positive: bulging out of the chord), or None."""
    fr = frame(ch)
    lp = local_physics(fr, ph)
    psi0 = fr.chord_angle - theta1
    if not 0.0 < psi0 < math.pi:
        return None
    p = pressure_at_start(fr, psi0, lp, steps)
    return None if p is None else _make(fr, psi0, p, lp, steps, "held")


def top(ch: Chord, ph: Physics, limit: float = TOE_LIMIT, steps: int = STEPS) -> Surface:
    """The fullest surface the toes hold (GA ``_caps``): a toe at ``limit`` to the chord, the other below it."""
    s1 = surface_at_toe(ch, limit, ph, steps)
    if s1 is not None and s1.toe_chord[1] <= limit + 1e-12:
        return Surface(s1.pressure, s1.psi0, s1.area, s1.toe_chord, s1.toe_solid, s1.points, "full", s1.toes)
    s2 = surface_at_toe(mirrored(ch), limit, ph, steps)
    if s2 is None:
        raise ValueError(f"no surface meets either toe at {math.degrees(limit):.1f} deg to the chord")
    s = surface_at_toe(ch, s2.toe_chord[1], ph, steps)          # the same surface, shot from toe 1
    if s is None:
        raise ValueError("the toe-2 cap does not reproduce from toe 1")
    return Surface(s.pressure, s.psi0, s.area, s.toe_chord, s.toe_solid, s.points, "full", s.toes)


def _edge(f, lo: float, hi: float) -> tuple[float, float]:
    """Between a start that fails (lo: f infinite) and one that does not (hi), the last that does not."""
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if math.isinf(f(mid)) else (lo, mid)
    return hi, f(hi)


def _crossing(fr: Frame, f, lo: float, hi: float, flo: float, fhi: float, p: float, lp: tuple,
              steps: int) -> Surface | None:
    """The surface between a start running past toe 2 (lo) and one meeting the far ray short of it (hi). If lo's
    shot failed (turned out along the radius), the root lies past the edge of the failing starts, or, when the shot
    just past that edge already falls short, the slice is draining: no surface of this pressure reaches toe 2 at a
    positive angle, and the one at the edge meets the wall tangentially short of it, GA's surface resting on the
    solid (pool.py hmin), its contact receded to there ("wet")."""
    if math.isinf(flo) and math.isinf(fhi):
        return None                               # overshoots, then curls back: no surface crosses toe 2 here
    if math.isinf(flo):
        lo, flo = _edge(f, lo, hi)
        if flo < 0.0:
            return _make(fr, lo, p, lp, steps, "wet")
    return _make(fr, root(f, lo, hi, flo, fhi, 1e-13), p, lp, steps, "held")


def surface(ch: Chord, p: float, ph: Physics, cap: Surface | None = None, steps: int = STEPS,
            scan: int = SCAN) -> Surface | None:
    """The surface at pool pressure p [Pa] on the held branch: the first root of the far-ray miss walking from the
    cap's start angle toward the root (positive there below the cap's pressure: the surface runs past toe 2). The
    first root is the one continuous with the cap as the pressure falls; the arc's pressure, a function of position,
    can add others nearer the root. At or above the cap's pressure, the cap itself ("full"); None if no surface
    between the toes has this pressure."""
    cap = top(ch, ph, steps=steps) if cap is None else cap
    if p >= cap.pressure:
        return cap
    fr = frame(ch)
    lp = local_physics(fr, ph)

    def f(psi0: float) -> float:
        return _miss(shoot(fr, psi0, p, lp, steps)) - fr.t2

    edges = np.linspace(cap.psi0, math.pi - 1e-6, scan + 1)
    lo, flo = float(edges[0]), f(float(edges[0]))
    if not flo > 0.0:
        return None
    for x in edges[1:]:
        fx = f(float(x))
        if fx < 0.0:
            return _crossing(fr, f, lo, float(x), flo, fx, p, lp, steps)
        lo, flo = float(x), fx
    return None


def pressure_for(ch: Chord, target: float, ph: Physics, limit: float = TOE_LIMIT, steps: int = STEPS) -> Surface:
    """The surface holding ``target`` [m^2]: the held area grows with the pressure (also on the draining branch),
    up to the cap ("full" past it); a slice holding less than any surface can is returned at the least ("thin")."""
    cap = top(ch, ph, limit, steps)
    if target >= cap.area:
        return cap

    def f(p: float) -> float:          # falls with p; no surface at all (too low a pressure) counts as too little
        s = surface(ch, p, ph, cap, steps)
        return math.inf if s is None else target - s.area

    scale = _pressure_scale(ch, ph)
    lo, flo = cap.pressure - scale, f(cap.pressure - scale)
    for k in range(60):
        if flo > 0.0:
            break
        lo = cap.pressure - scale * 2.0 ** (k + 1)
        flo = f(lo)
    if not flo > 0.0:
        raise ValueError("no pressure holds so little")
    p = root(f, lo, cap.pressure, flo, target - cap.area, 1e-12 * scale)
    s = surface(ch, p, ph, cap, steps)
    if s is not None and abs(s.area - target) <= 1e-7 * max(target, 1e-12):
        return s
    hi = p if s is not None else cap.pressure
    for _ in range(60):                # the least it holds: the lowest pressure with a surface
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if surface(ch, mid, ph, cap, steps) is None else (lo, mid)
    s = surface(ch, hi, ph, cap, steps)
    return Surface(s.pressure, s.psi0, s.area, s.toe_chord, s.toe_solid, s.points, "thin", s.toes)


def _pressure_scale(ch: Chord, ph: Physics) -> float:
    return ph.gamma / max(ch.length, 1e-9) + ph.rho * float(np.linalg.norm(ph.gravity)) * ch.length + ph.arc_peak

def overhang_frame(ch: Chord, depth: float = 2.5e-4) -> Frame:
    """The polar frame about C = M - depth n, below the chord's midpoint (GA pool.py:1146-1152): a bulge past a toe is
    star-shaped about it, not about the root. The fan from C covers the face; ``corner`` adds the joint's region
    between C and the rays (signed triangles C-p2-root and C-root-p1); ``guard`` flags a face entering the member
    beyond toe 1."""
    p1, p2, root = ch.p1, ch.p2, np.asarray(ch.root, float)
    n = ch.normal
    c = 0.5 * (p1 + p2) - depth * n
    u = (p1 - c) / np.linalg.norm(p1 - c)
    v = n - float(n @ u) * u
    v = v / np.linalg.norm(v)
    d2 = p2 - c
    sweep = math.atan2(float(d2 @ v), float(d2 @ u)) % (2.0 * math.pi)

    def cross(a: np.ndarray, b: np.ndarray) -> float:
        return float(a[0] * b[1] - a[1] * b[0])

    corner = 0.5 * (cross(p2 - c, root - c) + cross(root - c, p1 - c))
    e1 = ch.e1 / np.linalg.norm(ch.e1)
    e2 = ch.e2 / np.linalg.norm(ch.e2)
    m1 = e2 - float(e2 @ e1) * e1
    m1 = n if np.linalg.norm(m1) < 1e-9 else m1 / np.linalg.norm(m1)
    loc = lambda x: np.array([float(x @ u), float(x @ v)])                    # noqa: E731
    guard = (loc(root - c), loc(e1), loc(m1), float(ch.t1), loc(e2), None, float(ch.t2))
    a1 = math.atan2(float(e1 @ v), float(e1 @ u))
    a2 = math.atan2(float(e2 @ v), float(e2 @ u)) % (2.0 * math.pi)
    return Frame(c, u, v, sweep, float(np.linalg.norm(p1 - c)), float(np.linalg.norm(d2)), corner, a1, a2, guard,
                 tuple(loc(root - c)))


def pinned_for(ch: Chord, target: float, ph: Physics, steps: int = STEPS, step: float = 0.02) -> Surface:
    """A full slice whose toes cannot spread (the contact freezes) holding ``target``: past the cap its face turns
    beyond the 88 deg limit and overhangs (GA ``_overhang``: volume control, stable past the half circle). The family
    runs from the cap's start toward a face leaving toe 1 along the plate; "overhang" if one holds the target, else
    "touch": the face reaches the member past toe 1 first (``touch``: where along e1, GA's touchdown, the toe's new
    place; the metal between lies on unfused plate), or "touch2": it leans on the member past toe 2 first (its face
    there turned past 180 deg to it: GA touches down on either side, pool.py:1239-1253)."""
    cap = top(ch, ph, steps=steps)
    if target <= cap.area:
        return pressure_for(ch, target, ph, steps=steps)
    rf, fr = frame(ch), overhang_frame(ch)
    lp = local_physics(fr, ph)
    d = math.cos(cap.psi0) * rf.u + math.sin(cap.psi0) * rf.v                 # the cap's start, in the world
    hi = math.atan2(float(d @ fr.v), float(d @ fr.u))

    def at(psi0: float) -> Surface | None:
        p = pressure_at_start(fr, psi0, lp, steps)
        return None if p is None else _make(fr, psi0, p, lp, steps, "overhang")

    floor = max(fr.a1, 0.0) + 1e-4              # a face leaving toe 1 along the plate: the family's end
    s_hi = at(hi)
    if s_hi is not None and s_hi.toe_solid[1] >= math.pi:          # it already leans on the member past toe 2
        return Surface(s_hi.pressure, s_hi.psi0, s_hi.area, s_hi.toe_chord, s_hi.toe_solid, s_hi.points, "touch2",
                       s_hi.toes)
    lo, s_lo, broke = hi, s_hi, False
    while lo - step > floor:
        lo, prev = lo - step, s_lo
        s_lo = at(lo)
        if s_lo is None or not math.isnan(s_lo.touch) or s_lo.area >= target or s_lo.toe_solid[1] >= math.pi:
            hi, s_hi, broke = lo + step, prev, True
            break
    if broke and s_lo is not None and math.isnan(s_lo.touch) and s_lo.toe_solid[1] >= math.pi and \
            s_lo.area < target:                  # the face reaches the member past toe 2 first: it touches down there
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            sm = at(mid)
            lo, hi = (mid, hi) if sm is None or sm.toe_solid[1] >= math.pi else (lo, mid)
        s = at(hi)
        return Surface(s.pressure, s.psi0, s.area, s.toe_chord, s.toe_solid, s.points, "touch2", s.toes)
    if not broke:                                # the bulge lies on the plate at the toe: it touches down there
        lo, s_lo = floor, at(floor)
        if s_lo is not None and s_lo.area < target:
            return Surface(s_lo.pressure, s_lo.psi0, s_lo.area, s_lo.toe_chord, s_lo.toe_solid, s_lo.points, "touch",
                           s_lo.toes, ch.t1)
    if s_lo is not None and math.isnan(s_lo.touch) and s_lo.area >= target:
        x = root(lambda q: (lambda s: -math.inf if s is None else target - s.area)(at(q)), lo, hi,
                 target - s_lo.area, target - s_hi.area, 1e-12)
        s = at(x)
        if s.toe_solid[1] >= math.pi:            # holds it lying along the member past toe 2: touches down there
            return Surface(s.pressure, s.psi0, s.area, s.toe_chord, s.toe_solid, s.points, "touch2", s.toes)
        return s
    for _ in range(50):                                # the touchdown edge: the last start that does not touch
        mid = 0.5 * (lo + hi)
        sm = at(mid)
        lo, hi = (mid, hi) if sm is None or not math.isnan(sm.touch) else (lo, mid)
    s = at(hi)
    t = at(lo)
    where = t.touch if t is not None and not math.isnan(t.touch) else math.nan
    return Surface(s.pressure, s.psi0, s.area, s.toe_chord, s.toe_solid, s.points, "touch", s.toes, where)


__all__ = ["Frame", "Physics", "STEPS", "Shot", "Surface", "frame", "mirrored", "overhang_frame", "pinned_for",
           "pressure_at_start", "pressure_for", "root", "shoot", "surface", "surface_at_toe", "top"]
