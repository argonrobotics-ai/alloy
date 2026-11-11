# SPDX-License-Identifier: Apache-2.0
"""Capacity-source flag kernels of the HealthMonitor: global and per-world overflow counters, bool flags and MuJoCo
Warp's overflow bitmask, mapped onto ``HealthFlags`` in the check's accumulators (split out of health_kernels)."""

from __future__ import annotations

import warp as wp

from alloy.precision.health_kernels import FL_CONTACT_OVERFLOW, FL_ROW_OVERFLOW, FL_UNCONVERGED, Accum

@wp.kernel
def flag_if_above(values: wp.array[wp.float32], limit: float, bit: wp.uint32, acc: Accum):
    """Per-world measure check: values[w] > limit flags world w (e.g. soft tet strain, ``set_soft_strain``)."""
    w = wp.tid()
    if values[w] > limit:
        acc.overflow[w] = acc.overflow[w] | bit


@wp.kernel
def flag_if_over(count: wp.array[wp.int32], index: wp.int32, cap: wp.int32, bit: wp.uint32, acc: Accum):
    """Global capacity check: count[index] > cap flags every world (the dropped items are not attributable)."""
    w = wp.tid()
    if count[index] > cap:
        acc.overflow[w] = acc.overflow[w] | bit


@wp.kernel
def flag_if_nonzero(src: wp.array[wp.int32], per_world: wp.int32, bit: wp.uint32, invalid: wp.int32, acc: Accum):
    """``invalid``: a nonzero value also makes the world INVALID (the path a blown cloth takes)."""
    w = wp.tid()
    v = src[0]
    if per_world != 0:
        v = src[w]
    if v != 0:
        acc.overflow[w] = acc.overflow[w] | bit
        if invalid != 0:
            acc.stretch_fail[w] = 1


@wp.kernel
def flag_if_true(src: wp.array[wp.bool], world_count: wp.int32, bit: wp.uint32, acc: Accum):
    """Per-world bool flags; a trailing global slot (Steel's ``constraint_overflow[world_count]``) flags every world."""
    w = wp.tid()
    hit = src[w]
    if src.shape[0] > world_count:
        hit = hit or src[world_count]
    if hit:
        acc.overflow[w] = acc.overflow[w] | bit


@wp.kernel
def flag_bitmask(src: wp.array[wp.int32], contact_bits: wp.int32, row_bits: wp.int32, iter_bits: wp.int32,
                 acc: Accum):
    """A per-world solver overflow bitmask (MJWarp ``Data.overflow``) mapped onto HealthFlags."""
    w = wp.tid()
    v = src[w]
    out = acc.overflow[w]
    if (v & contact_bits) != 0:
        out = out | FL_CONTACT_OVERFLOW
    if (v & row_bits) != 0:
        out = out | FL_ROW_OVERFLOW
    if (v & iter_bits) != 0:
        out = out | FL_UNCONVERGED
    acc.overflow[w] = out
