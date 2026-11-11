# SPDX-License-Identifier: Apache-2.0
"""Textbook answers for the J2 model of `j2.py` (monotonic loading, linear isotropic hardening). The laws compare the
fibre sections and the solver hooks with these.

- `stress_1d`: the uniaxial curve σ(ε), E_t = E H / (E + H) past ε_y = σ_y0 / E.
- `rect_moment`: a rectangular section's M(κ) per unit width, piecewise exact (elastic core |z| < σ_y0/(E κ)).
- `round_moment`, `round_torque`: a solid round section's M(κ) and T(φ'), integrated to 1e-12 with the elastic-plastic
  boundary as a breakpoint. In pure shear J2 gives τ_y = σ_y0/√3 and a shear hardening of H/3.
- `springback_curvature`: the curvature left after unloading from κ0 elastically, κ0 - M(κ0)/(E I).
  - For a perfectly plastic strip this is Gardiner's (1957) R0/Rf = 4a³ - 3a + 1, a = R0 σ_y / (E t).
  - The unloading is elastic as long as the residual stresses stay below yield. For a rectangle and a round section
    loaded to at most M_p they do: the surface residual is -0.5 σ_y and -0.70 σ_y.
"""

from __future__ import annotations

import math

import numpy as np


def stress_1d(eps: np.ndarray | float, E: float, sy0: float, hmod: float = 0.0) -> np.ndarray:
    eps = np.asarray(eps, dtype=float)
    et = E * hmod / (E + hmod)
    ey = sy0 / E
    return np.where(np.abs(eps) <= ey, E * eps, np.sign(eps) * (sy0 + et * (np.abs(eps) - ey)))


def rect_moment(kappa: float, t: float, E: float, sy0: float, hmod: float = 0.0) -> float:
    """M per unit width of a t-thick rectangle at curvature κ >= 0 (monotonic)."""
    h = 0.5 * t
    if kappa * h * E <= sy0:
        return E * kappa * t ** 3 / 12.0
    et = E * hmod / (E + hmod)
    zy = sy0 / (E * kappa)
    plastic = (sy0 - et * sy0 / E) * (h * h - zy * zy) / 2.0 + et * kappa * (h ** 3 - zy ** 3) / 3.0
    return 2.0 * (E * kappa * zy ** 3 / 3.0 + plastic)


def round_moment(kappa: float, R: float, E: float, sy0: float, hmod: float = 0.0) -> float:
    """M of a solid round section of radius R at curvature κ >= 0 (monotonic)."""
    from scipy.integrate import quad  # noqa: PLC0415

    ye = min(R, sy0 / (E * kappa)) if kappa > 0.0 else R
    pts = [-ye, ye] if ye < R else None
    return quad(lambda y: float(stress_1d(kappa * y, E, sy0, hmod)) * y * 2.0 * math.sqrt(max(R * R - y * y, 0.0)),
                -R, R, points=pts, epsabs=1e-14 * sy0 * R ** 3, epsrel=1e-12, limit=200)[0]


def round_torque(phi: float, R: float, G: float, sy0: float, hmod: float = 0.0) -> float:
    """T of a solid round section of radius R at twist rate φ' >= 0 (monotonic)."""
    from scipy.integrate import quad  # noqa: PLC0415

    ty = sy0 / math.sqrt(3.0)
    gt = G * (hmod / 3.0) / (G + hmod / 3.0)

    def tau(g: float) -> float:
        return G * g if G * g <= ty else ty + gt * (g - ty / G)

    re = ty / (G * phi) if phi > 0.0 else R
    pts = [re] if re < R else None
    return quad(lambda r: tau(phi * r) * r * r * 2.0 * math.pi, 0.0, R, points=pts,
                epsabs=1e-14 * sy0 * R ** 3, epsrel=1e-12, limit=200)[0]


def springback_curvature(kappa0: float, moment: float, EI: float) -> float:
    """κ left after elastic unloading from (κ0, M(κ0))."""
    return kappa0 - moment / EI


def gardiner_ratio(R0: float, t: float, E: float, sy: float) -> float:
    """R0 / Rf of a perfectly plastic strip bent to radius R0 and released (Gardiner 1957)."""
    a = R0 * sy / (E * t)
    return 4.0 * a ** 3 - 3.0 * a + 1.0
