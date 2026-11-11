# SPDX-License-Identifier: Apache-2.0
"""Hertz impact duration and the excitation spectrum of its force pulse (host reference; the kernels mirror it).

Hertz impact of two elastic bodies, approach speed v, effective mass m* = 1 / (1/m1 + 1/m2), effective radius
R* = 1 / (1/R1 + 1/R2) and effective modulus E* = 1 / ((1 - ν1²)/E1 + (1 - ν2²)/E2). The contact time is
    t_c = 2.87 (m*² / (R* E*² v))^(1/5).
Source: Johnson, *Contact Mechanics* (1985), section 11.4, τ = 2.94 δ*/V with δ* = (15 m V² / (16 E* R^½))^(2/5).
From memory; the equation number is UNVERIFIED. 2.94 · (15/16)^(2/5) = 2.87.

The force history is close to a half-sine of duration t_c, which is the classic approximation (Goldsmith,
*Impact*, 1960). A half-sine pulse carrying impulse J has the spectrum magnitude
    |F(ω)| = J · |cos(ωτ/2) / (1 - (ωτ/π)²)|,   which is J at ω = 0 and J·π/4 at ωτ = π.
A mode at ω therefore receives J·H(ωτ). A shorter contact (a harder or faster impact) excites higher modes, as
law S-5 checks. The kernels apply it as an exact modal velocity kick, so it does not depend on the sim step.
"""

from __future__ import annotations

import math

import numpy as np

HERTZ_C = 2.87


def effective_modulus(e1: float, nu1: float, e2: float, nu2: float) -> float:
    """E* [Pa] of two materials (an infinite E2 is a rigid partner)."""
    inv = (1.0 - nu1 ** 2) / e1 + (0.0 if math.isinf(e2) else (1.0 - nu2 ** 2) / e2)
    return 1.0 / inv


def effective(a: float, b: float) -> float:
    """1 / (1/a + 1/b), with infinities (a flat face, a static body) dropping out."""
    inv = (0.0 if math.isinf(a) else 1.0 / a) + (0.0 if math.isinf(b) else 1.0 / b)
    return math.inf if inv == 0.0 else 1.0 / inv


def contact_time(m_eff: float, r_eff: float, e_eff: float, v: float) -> float:
    """Hertz contact duration [s] at approach speed ``v`` [m/s] > 0."""
    return HERTZ_C * (m_eff ** 2 / (r_eff * e_eff ** 2 * v)) ** 0.2


def pulse_weight(omega: np.ndarray | float, tau: float) -> np.ndarray:
    """|F(ω)| / J of a half-sine pulse of duration ``tau`` (see the module docstring)."""
    x = np.asarray(omega, float) * tau
    den = 1.0 - (x / math.pi) ** 2
    near = np.abs(den) < 1e-6
    out = np.abs(np.cos(0.5 * x) / np.where(near, 1.0, den))
    return np.where(near, math.pi / 4.0, out)


__all__ = ["HERTZ_C", "contact_time", "effective", "effective_modulus", "pulse_weight"]
