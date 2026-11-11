# SPDX-License-Identifier: Apache-2.0
"""Wet-out of the ``Bonds`` adhesive, and the rubber roller that applies labels (SCR-WP8-11 Part B1).

**Wet-out.** Each film vertex carries w in [0, 1], the fraction of its area the adhesive has wetted. While it is
pressed at p >= p_form (its substrate card's):

    dw/dt = (1 - w)·(p/P_REF)^M / tau_w,   stepped exactly: w <- 1 - (1 - w)·exp(-(p/P_REF)^M·dt/tau_w)

(exact for a pressure held over the step). A free vertex the press leaves goes back to 0, as its dwell clock does; a
bonded one keeps its w; a released one starts again from 0.

**What w scales.** The cards' G rows (3M 7818, ASTM D3330) are for a standard application, so a bond's G is
G_card·w/W_STD. The front uses the width-mean of w over the bonds it releases next, for G and its tau cap. A pre-bonded
start is a standard application: w = W_STD.

**Calibration.** GA scores wet-out against a standard application. ``wet_tau`` sets tau_w so that one standard
application (``STD_PASSES`` passes of the ASTM D3330 roller at ``STD_SPEED``) gives W_STD at every point it covers.
GA's pass rule is then w_task / w_standard >= 0.90, with w_standard from a standard pass in the same model (WP13).

**Pressure under a roller (WP13 must-fix 1).** Under a rigid roller the contact reading k_c·pen/a_cell grows as 1/cell:
the line load rests on about one row of cells. So a vertex within a roller's reach takes the Hertz pressure of the
roller's rubber cover on the substrate instead. For a cylinder of radius R on a flat, or on a parallel tube of radius
R_s (1/R_e = 1/R + 1/R_s), under line load q:

    b = sqrt(4·q·R_e/(π·E*)),   p(s) = (2·q/(π·b))·sqrt(1 - s²/b²),   E* = E/(1 - ν²) (the substrate rigid),

where s is the vertex's distance from the contact line, across the roller. Nothing here depends on the mesh.

**The roller's load is measured on the film (WP13).** q is what the film carries: Σ k_c·pen over the roller's
contacts with the film in that world, over the roller's width, averaged over ``LOAD_TAU``. A roller that does not
touch the film carries 0 and presses nothing, whatever it hovers over; a robot pressing harder reads higher. So the
roller shape should not also collide with the substrate (the film then carries all of its load). The input
``roller_load`` [W, R] N/m, when > 0, replaces the measurement.

Sources and ESTIMATES:
- The standard roller is ASTM D3330's 4.5 lb (2.04 kg) roller: 3.75 in (95.25 mm) diameter, 1.75 in (44.45 mm) wide,
  with a Shore A 80 rubber cover (Thwing-Albert TA100 roller sheet, "meets ASTM D3330"). It is rolled twice, once
  each way.
- Cover modulus: Shore A 80 gives E = 9.35 MPa by Gent's relation (ESTIMATE, ν = 0.5).
- ``STD_SPEED`` = 10 mm/s is an ESTIMATE: D3330's rolling speed was not found. It only enters through tau_w.
- ``M`` = 1 is an ESTIMATE: wet-out as the PSA's stationary, linear creep into the substrate's roughness, a rate
  ∝ p. Laws W-1 and W-2 hold for any M > 0.
- p_form and the dwell are the card's (adhesives.py ESTIMATES).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

P_REF = 1.0e5                 # Pa
M_EST = 1.0                   # pressure exponent (ESTIMATE, module docstring)
W_STD = 0.90                  # w after a standard application (the calibration)
STD_SPEED = 0.01              # m/s, the standard roll speed (ESTIMATE)
STD_PASSES = 2                # D3330: once each way
LOAD_TAU = 0.01               # s, the averaging time of a roller's measured line load
FORCE_SCALE = 2.0 ** 24       # 1/N: a roller's film force is summed in fixed point (deterministic; 6e-8 N steps)
TA100 = "Thwing-Albert TA100 roller sheet (ASTM D3330): 4.5 lb, 3.75 in dia, 1.75 in wide, Shore A 80"


def gent_modulus(shore_a: float) -> float:
    """Young's modulus [Pa] of a rubber of Shore A hardness ``shore_a`` (Gent 1958; ESTIMATE, about ±20 %)."""
    s = float(shore_a)
    return 1.0e6 * 0.0981 * (56.0 + 7.62336 * s) / (0.137505 * (254.0 - 2.54 * s))


@dataclass(frozen=True)
class Roller:
    """A rubber-covered roller card: ``shape`` the roller's capsule or cylinder in the scene (its axis local z),
    ``radius`` and ``width`` its cover's (the shape must match), ``mass`` its own load, ``shore_a`` the cover."""

    shape: str = "roller"
    radius: float = 0.5 * 0.09525
    width: float = 0.04445
    mass: float = 2.04
    shore_a: float = 80.0
    poisson: float = 0.5
    source: str = TA100

    def __post_init__(self) -> None:
        if min(self.radius, self.width, self.mass) <= 0.0 or not 0.0 < self.shore_a < 100.0:
            raise ValueError(f"roller {self.shape!r}: radius, width, mass > 0 and 0 < shore_a < 100 required")

    @property
    def modulus(self) -> float:
        """E* [Pa] of the cover on a rigid substrate."""
        return gent_modulus(self.shore_a) / (1.0 - self.poisson ** 2)

    def line_load(self, gravity: float = 9.81) -> float:
        """N/m: the roller's own weight over its width."""
        return self.mass * gravity / self.width

    def half_width(self, q: float, r_sub: float = math.inf) -> float:
        r_e = 1.0 / (1.0 / self.radius + 1.0 / r_sub)
        return math.sqrt(4.0 * q * r_e / (math.pi * self.modulus))

    def pressure(self, s: np.ndarray, q: float, r_sub: float = math.inf) -> np.ndarray:
        """Hertz p(s) [Pa] at distance ``s`` from the contact line (module docstring)."""
        b = self.half_width(q, r_sub)
        u = np.clip(np.asarray(s, np.float64) / b, -1.0, 1.0)
        return 2.0 * q / (math.pi * b) * np.sqrt(1.0 - u * u)


STANDARD_ROLLER = Roller()


def exposure(roller: Roller, q: float, p_form: float, speed: float, m: float = M_EST, r_sub: float = math.inf) -> float:
    """One pass: ∫ (p/P_REF)^m dt [s] over the time the pressure is at least ``p_form``; ``r_sub`` the radius of a
    tube under the roller, parallel to it (1/R_e = 1/R + 1/r_sub)."""
    b = roller.half_width(q, r_sub)
    s = np.linspace(-b, b, 20001)
    p = roller.pressure(s, q, r_sub)
    f = np.where(p >= p_form, (p / P_REF) ** m, 0.0)
    return float(np.trapezoid(f, s) / speed)


def wet_tau(p_form: float, roller: Roller = STANDARD_ROLLER, speed: float = STD_SPEED, passes: int = STD_PASSES,
            m: float = M_EST, w_std: float = W_STD) -> float:
    """tau_w [s] that gives ``w_std`` after a standard application (module docstring, "Calibration")."""
    e = passes * exposure(roller, roller.line_load(), p_form, speed, m)
    if e <= 0.0:
        raise ValueError(f"the standard roller never reaches p_form = {p_form} Pa: its peak is "
                         f"{float(roller.pressure(0.0, roller.line_load())):.0f} Pa")
    return e / -math.log(1.0 - w_std)


def wet_after(w0: float, p: float, t: float, p_form: float, tau: float, m: float = M_EST) -> float:
    """The exact w after ``t`` s at a constant ``p`` (law W-1)."""
    return w0 if p < p_form else 1.0 - (1.0 - w0) * math.exp(-((p / P_REF) ** m) * t / tau)


def ga_wet_out(wet: np.ndarray, area: np.ndarray, band: np.ndarray, w_standard: float = W_STD,
               ratio: float = 0.90) -> dict[str, float | bool]:
    """GA's wet-out criterion (WP13's pass rule): the label's area-mean w and its edge band's (``band``: the vertices
    within 3 mm of the label's edge), each over ``w_standard``, at least ``ratio``. A standard pass in this model
    gives W_STD (the calibration; law W-3 holds the model's pass to the Hertz value within 2.5 %)."""
    w, a, m = np.asarray(wet, float), np.asarray(area, float), np.asarray(band, bool)
    label = float(np.sum(w * a) / np.sum(a)) / w_standard
    edge = float(np.sum(w[m] * a[m]) / np.sum(a[m])) / w_standard if m.any() else label
    return {"label": label, "edge_band": edge, "pass": bool(label >= ratio and edge >= ratio)}


# --- device side: the rollers' Hertz pressure at a film vertex -----------------------------------------------------

@wp.struct
class Rollers:
    """Per world w and roller r (``count`` per world): the model shape, cover E*, contact half-length and line load."""
    count: int
    shape: wp.array2d[wp.int32]        # [W, R] model shape ids
    of_shape: wp.array[wp.int32]       # [S] the roller index of a model shape (-1: not a roller)
    radius: wp.array[float]            # [R]
    half_len: wp.array[float]          # [R]
    estar: wp.array[float]             # [R]
    force: wp.array2d[wp.int64]        # [W, R] scratch: Σ k_c·pen of the step's film contacts, x FORCE_SCALE
    load: wp.array2d[float]            # [W, R] N/m, the line load in use (state: measured and averaged)
    override: wp.array2d[float]        # [W, R] N/m, input roller_load (> 0 replaces the measurement)
    reach: float                       # m beyond the radius a vertex counts as under the roller
    tau: float                         # s, LOAD_TAU
    sub_radius: wp.array[float]        # [S] a cylinder or capsule substrate's radius (0: flat), for R_e


PARALLEL = wp.constant(0.985)          # cos 10°: a tube within this of a roller's axis makes a line contact


@wp.func
def tube_radius(rl: Rollers, ax: wp.vec3, sub: int, body_q: wp.array[wp.transform], shape_body: wp.array[wp.int32],
                shape_xform: wp.array[wp.transform]) -> float:
    """R_s of substrate shape ``sub`` if it is a tube within 10° of parallel to a roller of axis ``ax``, else 0 (a flat
    face, or a tube across the roller: a point contact, not modelled; the line law with R_e = R is used)."""
    if sub < 0:
        return 0.0
    rs = rl.sub_radius[sub]
    if rs <= 0.0:
        return 0.0
    xf = shape_xform[sub]
    b = shape_body[sub]
    if b >= 0:
        xf = wp.transform_multiply(body_q[b], xf)
    axis = wp.quat_rotate(wp.transform_get_rotation(xf), wp.vec3(0.0, 0.0, 1.0))
    if wp.abs(wp.dot(axis, ax)) < PARALLEL:
        return 0.0
    return rs


@wp.func
def roller_pressure(rl: Rollers, w: int, x: wp.vec3, n: wp.vec3, sub: int, rp: float, body_q: wp.array[wp.transform],
                    shape_body: wp.array[wp.int32], shape_xform: wp.array[wp.transform]) -> float:
    """The Hertz pressure at a vertex at ``x`` on substrate shape ``sub`` (outward normal ``n``), if it lies within a
    roller's reach (largest over rollers); -1 if under none (the caller then reads the contact)."""
    out = float(-1.0)
    for r in range(rl.count):
        s = rl.shape[w, r]
        xf = shape_xform[s]
        b = shape_body[s]
        if b >= 0:
            xf = wp.transform_multiply(body_q[b], xf)
        c = wp.transform_get_translation(xf)
        ax = wp.quat_rotate(wp.transform_get_rotation(xf), wp.vec3(0.0, 0.0, 1.0))
        d = x - c
        along = wp.dot(d, ax)
        radial = d - along * ax
        if wp.abs(along) <= rl.half_len[r] and wp.length(radial) <= rl.radius[r] + rl.reach:
            q = rl.load[w, r]
            big = rl.radius[r]
            r_e = big
            across = wp.cross(ax, n)
            sl = wp.dot(radial, across) / wp.max(wp.length(across), 1.0e-12)
            rs = tube_radius(rl, ax, sub, body_q, shape_body, shape_xform)
            if rs > 0.0:                    # on a tube: R_e, and the offset along the tube's surface, not the roller's
                r_e = big * rs / (big + rs)
                sl = sl * (rs + rp) / (big + rs + 2.0 * rp)
            half = wp.sqrt(4.0 * q * r_e / (wp.pi * rl.estar[r]))
            p = float(0.0)
            if q > 0.0 and wp.abs(sl) < half:
                p = 2.0 * q / (wp.pi * half) * wp.sqrt(1.0 - (sl / half) * (sl / half))
            out = wp.max(out, p)
    return out


def rollers_struct(cards: tuple[Roller, ...], shapes: np.ndarray, shape_count: int, load: wp.array,
                   override: wp.array, reach: float, device: str, sub_radius: np.ndarray | None = None) -> Rollers:
    """``shapes`` [W, R] model shape ids of each card's roller; ``load`` its [W, R] state, ``override`` the input;
    ``sub_radius`` [S] the radius of each cylinder or capsule substrate shape (0 or None: flat)."""
    rl = Rollers()
    n = len(cards)
    arr = lambda v, t: wp.array(np.ascontiguousarray(v), dtype=t, device=device)  # noqa: E731
    of_shape = np.full(max(shape_count, 1), -1, np.int32)
    for r in range(n):
        of_shape[shapes[:, r]] = r
    rl.count = n
    rl.shape = arr(shapes.astype(np.int32), wp.int32)
    rl.of_shape = arr(of_shape, wp.int32)
    rl.radius = arr([c.radius for c in cards] or [0.0], float)
    rl.half_len = arr([0.5 * c.width for c in cards] or [0.0], float)
    rl.estar = arr([c.modulus for c in cards] or [1.0], float)
    rl.force = wp.zeros(shapes.shape, dtype=wp.int64, device=device)
    rl.load, rl.override, rl.reach, rl.tau = load, override, float(reach), LOAD_TAU
    sub = np.zeros(max(shape_count, 1)) if sub_radius is None else np.asarray(sub_radius, np.float64)
    rl.sub_radius = arr(sub, float)
    return rl


__all__ = ["FORCE_SCALE", "LOAD_TAU", "M_EST", "P_REF", "Roller", "Rollers", "STANDARD_ROLLER", "STD_PASSES",
           "STD_SPEED", "W_STD", "exposure", "ga_wet_out", "gent_modulus", "roller_pressure",
           "rollers_struct", "tube_radius",
           "wet_after", "wet_tau"]
