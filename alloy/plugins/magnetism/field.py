# SPDX-License-Identifier: Apache-2.0
"""Point-dipole magnetostatics (host NumPy, float64): the reference the laws and tasks check the kernels against.

For a source dipole m_s at p_s and a target dipole m at p, r = p - p_s, d = |r|, r̂ = r/d, μ0/4π = 1e-7 T·m/A:

    B_s(p) = 1e-7 / d³ · (3 r̂ (m_s·r̂) - m_s)
    F      = 3e-7 / d⁴ · [(m_s·r̂) m + (m·r̂) m_s + (m_s·m) r̂ - 5 (m_s·r̂)(m·r̂) r̂]       (force on m)
    τ      = m × B_s(p)                                                                 (torque on m)
    U      = 1e-7 / d³ · [m_s·m - 3 (m_s·r̂)(m·r̂)]

Closed forms used by the laws: coaxial aligned dipoles attract with 3μ0 m_s m / (2π d⁴), side by side
(both ⊥ r̂, parallel) they repel with 3μ0 m_s m / (4π d⁴).

**Ferromagnetic plate** (a linear half-space of relative permeability μr behind a plane with outward normal n̂). A
dipole m at height h in front of it sees the plate's magnetisation as an image dipole at the mirror point (distance
2h), with
    m' = κ (2 (m·n̂) n̂ - m),   κ = (μr - 1) / (μr + 1):
the normal part stays, the tangential part flips. This is the magnetic analogue of a charge's image in a grounded
plane, for κ = 1 (soft iron). So a magnet normal to the plate is pulled in with 3μ0 κ m² / (32π h⁴), and one lying
flat with half that. The energy is U = -½ m·B'(p) (the plate's magnetisation is induced).
Sources: Jackson, *Classical Electrodynamics*, 3rd ed., sections 4.4 and 5.11 (image of a dipole in a permeable
half-space), from memory, UNVERIFIED section numbers.

**Uniform field** B: torque m × B, no force. A bar of inertia I about an axis ⊥ m, B oscillates about B at
ω0 = sqrt(m B / I).

**Rotating drive** (a magnetic stirrer). A dipole M turning at ω in a plane, at distance h along that plane's
normal from a bar of moment m on the same axis. The field at the bar is -1e-7 M / h³, antiparallel to the drive
moment. A bar on a pivot with viscous torque c·ω follows at a steady lag δ with
    m B sin δ = c ω,
up to the pull-out speed ω_max = m B / c. Faster, it slips.
"""

from __future__ import annotations

import numpy as np

MU0_4PI = 1.0e-7          # T·m/A
MU0 = 4.0e-7 * np.pi


def field(p_s: np.ndarray, m_s: np.ndarray, p: np.ndarray) -> np.ndarray:
    """B [T] of a dipole ``m_s`` at ``p_s``, at point ``p``."""
    r = np.asarray(p, float) - np.asarray(p_s, float)
    d = np.linalg.norm(r)
    rh = r / d
    return MU0_4PI / d ** 3 * (3.0 * rh * np.dot(m_s, rh) - np.asarray(m_s, float))


def force(p_s: np.ndarray, m_s: np.ndarray, p: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Force [N] on dipole ``m`` at ``p`` from ``m_s`` at ``p_s``."""
    r = np.asarray(p, float) - np.asarray(p_s, float)
    d = np.linalg.norm(r)
    rh = r / d
    ms, mm = np.asarray(m_s, float), np.asarray(m, float)
    a, b = np.dot(ms, rh), np.dot(mm, rh)
    return 3.0 * MU0_4PI / d ** 4 * (a * mm + b * ms + np.dot(ms, mm) * rh - 5.0 * a * b * rh)


def torque(p_s: np.ndarray, m_s: np.ndarray, p: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Torque [N·m] on dipole ``m`` at ``p`` from ``m_s`` at ``p_s``."""
    return np.cross(np.asarray(m, float), field(p_s, m_s, p))


def energy(p_s: np.ndarray, m_s: np.ndarray, p: np.ndarray, m: np.ndarray) -> float:
    """Interaction energy [J] of two permanent dipoles."""
    r = np.asarray(p, float) - np.asarray(p_s, float)
    d = np.linalg.norm(r)
    rh = r / d
    return float(MU0_4PI / d ** 3 * (np.dot(m_s, m) - 3.0 * np.dot(m_s, rh) * np.dot(m, rh)))


def kappa(mu_r: float) -> float:
    """Image strength (μr - 1)/(μr + 1) of a linear half-space."""
    return (mu_r - 1.0) / (mu_r + 1.0)


def image(p: np.ndarray, m: np.ndarray, plane_point: np.ndarray, normal: np.ndarray, k: float
          ) -> tuple[np.ndarray, np.ndarray]:
    """The image (position, moment) of dipole ``m`` at ``p`` in a plate through ``plane_point`` with unit outward
    ``normal`` and image strength ``k``."""
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    h = np.dot(np.asarray(p, float) - plane_point, n)
    return np.asarray(p, float) - 2.0 * h * n, k * (2.0 * np.dot(m, n) * n - np.asarray(m, float))


def plate_force_normal(m: float, h: float, k: float = 1.0) -> float:
    """Attraction [N] of a dipole normal to the plate at height h: 3μ0 κ m² / (32π h⁴)."""
    return 3.0 * MU0 * k * m * m / (32.0 * np.pi * h ** 4)


def stirrer_lag(m_bar: float, b: float, c: float, omega: float) -> float:
    """Steady lag angle [rad] of a viscous-damped bar behind a field rotating at ``omega``: asin(c ω / (m B)), or
    NaN beyond pull-out."""
    s = c * omega / (m_bar * b)
    return float(np.arcsin(s)) if s <= 1.0 else float("nan")


__all__ = ["MU0", "MU0_4PI", "energy", "field", "force", "image", "kappa", "plate_force_normal", "stirrer_lag",
           "torque"]
