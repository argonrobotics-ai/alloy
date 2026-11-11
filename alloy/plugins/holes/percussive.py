# SPDX-License-Identifier: Apache-2.0
"""Rotary-hammer drilling (drilling v2, phase C, D-6): a layer whose work row is percussive (``work.PERCUSSIVE``) is cut
by blows, not by Shaw and Oxford's lips.

A hammer of blow energy E at blow rate f puts power P = E f into the bit. A layer that takes e_v joules per removed
cubic metre at the bit's diameter d then deepens the hole at

    v = P / (e_v pi d^2 / 4)

while the bit is pressed onto it (the front force T > 0) and turns, once the full diameter is in. While only the
point is in the material (at entry and at breakthrough) or the bit enlarges a pilot, the blows remove that volume
over the section the point actually cuts, dV/db = f(b) - f(b - span), f(x) = pi max(min(R, x tan kappa)^2 - r_void^2,
0) (``section``): the energy per volume stays the table's. The model takes the rate as independent of the push once
pressed (the sources rate the tool, not a feed-force curve), and the bit's turning only indexes it: the feed per
revolution is v 2 pi / |omega|, with no flute limit. A percussive layer with the hammer off (P = 0) is not cut: the
front holds the bit like a face. Its thrust is what the press puts on it beyond the Shaw layers' thrust at the same
feed; its torque is not modelled.
"""

from __future__ import annotations

import warp as wp

from alloy.plugins.holes.front_types import Work

@wp.func
def energy_at(work: Work, d: wp.float32) -> wp.float32:
    """e_v [J/m^3] at diameter ``d``: linear between the table's diameters, held past its ends."""
    ev = work.ev[0]
    for k in range(1, work.ev_n):
        lo = work.ev_d[k - 1]
        if d > lo:
            t = wp.min((d - lo) / (work.ev_d[k] - lo), 1.0)
            ev = work.ev[k - 1] + t * (work.ev[k] - work.ev[k - 1])
    return ev


@wp.func
def _disk(x: wp.float32, radius: wp.float32, t: wp.float32, r_void: wp.float32) -> wp.float32:
    r = wp.min(radius, wp.max(x, 0.0) * t)
    return wp.max(r * r - r_void * r_void, 0.0)


@wp.func
def section(b: wp.float32, span: wp.float32, radius: wp.float32, t: wp.float32, r_void: wp.float32) -> wp.float32:
    """dV/db [m^2]: the area the drill point cuts per unit advance with its tip at ``b`` in a layer of ``span``
    (pi R^2 with the full diameter in and the tip short of the back face)."""
    return wp.pi * (_disk(b, radius, t, r_void) - _disk(b - span, radius, t, r_void))


@wp.func
def hammer_feed(power: wp.float32, work: Work, d: wp.float32, spin: wp.float32, area: wp.float32) -> wp.float32:
    """The feed per revolution [m] that deepens the hole at P / (e_v dV/db) at spin ``spin`` [rad/s]; 0 when the
    hammer is off or the bit does not turn; unbounded (the front's cap holds it) when nothing is cut yet."""
    turn = wp.abs(spin)
    ev = energy_at(work, d)
    if power <= 0.0 or turn <= 0.0 or ev <= 0.0:
        return wp.float32(0.0)
    if area <= 1.0e-12:
        return wp.float32(1.0e30)
    return (power / (ev * area)) * (2.0 * wp.pi) / turn


__all__ = ["energy_at", "hammer_feed", "section"]
