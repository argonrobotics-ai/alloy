# SPDX-License-Identifier: Apache-2.0
"""SCR-SC5-3 cut-front growth rule (re-ruled 2026-11-03 19:05Z), the part of the cut-front check that needs
per-particle state. ``health_kernels.check_soft_contacts`` holds cut-front contacts to 2 x blade half-thickness +
radius and writes each twin's deepest cut-front depth of the check into ``cut_now`` (and the limit a normal contact
would have into ``cut_floor``); this kernel then compares it with the twin's baseline.
"""

from __future__ import annotations

import warp as wp

from alloy.precision.health_kernels import PEN_PARTICLE, Accum, note_pen, pen_tag


@wp.kernel
def cut_front_growth(cut_blade: wp.array[wp.int32], particle_world: wp.array[wp.int32], growth: wp.float32,
                     cut_now: wp.array[wp.float32], cut_floor: wp.array[wp.float32], cut_first: wp.array[wp.float32],
                     acc: Accum):
    """SCR-SC5-3 growth rule, one thread per particle after ``check_soft_contacts``: the blade pushes cut-front twins
    out, never in. A twin's first check with a cut-front contact sets its baseline (its depth then, floored at 0: a
    twin first seen outside the blade starts at the surface). Later, deeper than max(normal-contact limit, baseline +
    ``growth``) is PEN_FAIL: an exempt twin is never held tighter than a normal contact (coordinator ruling
    2026-11-04 11:30Z; the normal limit is min(pen_fail, half the thinner of particle and blade shape), from
    ``check_soft_contacts``). The baseline resets when the twin leaves the cut front (``cut_blade`` back to -1)."""
    i = wp.tid()
    now = cut_now[i]
    floor = cut_floor[i]
    cut_now[i] = -wp.inf
    cut_floor[i] = -wp.inf
    if cut_blade[i] < 0:
        cut_first[i] = wp.nan
        return
    w = particle_world[i]
    if w < 0 or now == -wp.inf:
        return
    if wp.isnan(cut_first[i]):
        cut_first[i] = wp.max(now, 0.0)
        return
    note_pen(acc, w, now, wp.max(floor, cut_first[i] + growth), pen_tag(PEN_PARTICLE, i), 0.0)   # at once
