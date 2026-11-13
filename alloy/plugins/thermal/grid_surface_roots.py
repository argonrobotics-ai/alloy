# SPDX-License-Identifier: Apache-2.0
"""Bracketed roots for the surface solver on the device (``grid_surface``): Illinois regula falsi, bisecting while
an end of the bracket is a failed evaluation (+-FAIL)."""

from __future__ import annotations

import warp as wp

FAIL = wp.constant(1.0e30)                  # a failed evaluation (signed)


@wp.struct
class Bracket:
    lo: float
    hi: float
    flo: float
    fhi: float
    side: int


@wp.func
def finite(f: float) -> bool:
    return wp.abs(f) < 0.5 * FAIL


@wp.func
def bracket(lo: float, hi: float, flo: float, fhi: float) -> Bracket:
    b = Bracket()
    b.lo = lo
    b.hi = hi
    b.flo = flo
    b.fhi = fhi
    b.side = 0
    return b


@wp.func
def guess(b: Bracket) -> float:
    """Illinois regula falsi inside the bracket; bisection while an end failed."""
    x = 0.5 * (b.lo + b.hi)
    if finite(b.flo) and finite(b.fhi) and b.fhi != b.flo:
        s = (b.lo * b.fhi - b.hi * b.flo) / (b.fhi - b.flo)
        if s > b.lo and s < b.hi:
            x = s
    return x


@wp.func
def narrow(b: Bracket, x: float, fx: float) -> Bracket:
    out = bracket(b.lo, b.hi, b.flo, b.fhi)
    if (fx > 0.0) == (b.flo > 0.0):
        out.lo = x
        out.flo = fx
        out.side = -1
        if b.side == -1 and finite(b.fhi):
            out.fhi = 0.5 * b.fhi
    else:
        out.hi = x
        out.fhi = fx
        out.side = 1
        if b.side == 1 and finite(b.flo):
            out.flo = 0.5 * b.flo
    return out


@wp.func
def settle(b: Bracket) -> float:
    if finite(b.flo) and finite(b.fhi):
        if wp.abs(b.flo) < wp.abs(b.fhi):
            return b.lo
        return b.hi
    return 0.5 * (b.lo + b.hi)


__all__ = ["FAIL", "Bracket", "bracket", "finite", "guess", "narrow", "settle"]
