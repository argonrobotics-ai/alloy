# SPDX-License-Identifier: Apache-2.0
"""CONTACT_OVERFLOW's one console line per health monitor, with the fix (FIX-LIST item 6, 2026-11-09).

A world with more contacts than its buffer holds drops the rest; health flags it CONTACT_OVERFLOW and freezes it
(INVALID). MuJoCo Warp printed its own overflow warning on every step (an outside tester: 2,464 lines in 20 steps), so
that print is off for the bits health flags (``alloy.integration.builtin_plugins.quiet_flagged_overflow``). Instead
``classify`` prints this line the first time any world's record carries CONTACT_OVERFLOW, from the device (no host
read during stepping, so it also works inside CUDA graphs), once per monitor.

``OverflowNote`` holds the printed latch and the first MuJoCo source's buffer: its contacts per world on average
(``naconmax // nworld``; MuJoCo Warp's pool of naconmax is shared by the worlds, and on the CPU's MuJoCo C route it is
the contact export health reads; ``alloy.build(..., contact_capacity=)`` sizes both), its contact count (``nacon``,
which keeps counting past the pool) and its per-world overflow word. The line names ``contact_capacity`` only when
that buffer overflowed (a broad- or narrow-phase bit in the world's word, or nacon past the pool), suggesting
max(2 x capacity, nacon / worlds). An overflow from another source (Newton's rigid or soft contacts, a CCD or
height-field buffer) points to ``sim.health()`` instead, where ``contact_capacity`` is not the fix. A global overflow
flags every world, so the line names the first world flagged, not the busiest.
"""

from __future__ import annotations

from collections.abc import Sequence

import warp as wp


@wp.struct
class OverflowNote:
    printed: wp.array[wp.int32]          # [1]: 1 once the line printed
    cap: wp.int32                        # the MuJoCo source's contacts per world on average (0: no MuJoCo source)
    nworld: wp.int32                     # its worlds
    naconmax: wp.int32                   # its pool (the overflow test; cap is naconmax // nworld, rounded down)
    nacon: wp.array[wp.int32]            # [1] its contact count (past naconmax on an overflow; [0] if none)
    mjw: wp.array[wp.int32]              # its per-world overflow word (MuJoCo Warp bits; [0] if none)
    bits: wp.int32                       # capacity_bits()


def capacity_bits() -> int:
    """MuJoCo Warp's overflow bits that ``contact_capacity`` (nconmax) sizes: broad and narrow phase."""
    from alloy._vendor.mujoco_warp import OverflowType as O

    return int(O.BROADPHASE | O.NARROWPHASE)


def overflow_note(mjw_sources: Sequence[object], device: str) -> OverflowNote:
    """``classify``'s note for a monitor reading ``mjw_sources`` (``health_adapters.mjw_contact_sources``)."""
    note = OverflowNote()
    note.printed = wp.zeros(1, dtype=wp.int32, device=device)
    d = getattr(next(iter(mjw_sources), None), "data", None)
    empty = wp.zeros(0, dtype=wp.int32, device=device)
    note.nworld = max(int(d.nworld), 1) if d is not None else 1
    note.naconmax = int(d.naconmax) if d is not None else 0
    note.cap = note.naconmax // note.nworld
    note.nacon = getattr(d, "nacon", None) if d is not None and getattr(d, "nacon", None) is not None else empty
    note.mjw = d.overflow if d is not None else empty
    note.bits = capacity_bits()
    return note


@wp.func
def overflow_line(w: int, note: OverflowNote):
    """Print the first flagged world ``w`` and the fix: ``alloy.build(..., contact_capacity=n)`` when the MuJoCo
    source's buffer overflowed, else ``sim.health()``'s fix (module docstring)."""
    capacity = False
    want = 2 * note.cap
    if note.cap > 0 and w < note.mjw.shape[0]:
        capacity = (note.mjw[w] & note.bits) != 0
    if note.cap > 0 and note.nacon.shape[0] > 0:
        seen = note.nacon[0]
        capacity = capacity or seen > note.naconmax
        want = wp.max(want, (seen + note.nworld - 1) // note.nworld)
    if capacity:
        wp.printf("alloy: contacts overflowed MuJoCo's buffer (%d per world; first flagged: world %d): "
                  "CONTACT_OVERFLOW, results are not trustworthy and the flagged worlds are frozen. Fix: "
                  "alloy.build(..., contact_capacity=%d). sim.health() lists the worlds\n", note.cap, w, want)
    else:
        wp.printf("alloy: contacts overflowed a contact buffer (first flagged: world %d): CONTACT_OVERFLOW, results "
                  "are not trustworthy and the flagged worlds are frozen. sim.health() lists the worlds and the "
                  "fix\n", w)


__all__ = ["OverflowNote", "capacity_bits", "overflow_line", "overflow_note"]
