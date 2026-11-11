# SPDX-License-Identifier: Apache-2.0
"""Fixed64: order-independent int64 fixed-point scatter sums (SPEC §3.7, D9 rule D3).

Two passes per destination ``i`` (as Newton's hydroelastic reducer,
newton@ac7f7830:newton/_src/geometry/contact_reduction_hydroelastic.py:76-142):

1. ``fixed64_prepare``: ``exp[i] = max e`` over the terms, ``e`` the smallest integer with ``|x| < 2**e``, read
   exactly from the float32 bits (``atomic_max`` on int32 is order-independent).
2. ``fixed64_add``: ``acc[i] += round(x * 2**(mbits - exp[i]))`` (int64 ``atomic_add`` is associative).

``fixed64_read`` returns ``acc[i] * 2**(exp[i] - mbits)`` in float64. With ``mbits = 61 - bitlen(max_terms - 1)``
each term is ``< 2**mbits`` in magnitude and any ``max_terms`` of them sum below ``2**61``, so int64 never overflows.
The grid step is ``2**(exp - mbits)``; a float32 term is exact on the grid when its exponent is within
``mbits - 24`` binades of the largest term, so for such inputs the sum is the exact sum and the read differs from
``math.fsum`` only by the final float64 rounding. Terms below the grid are rounded half away from zero.

A non-finite term poisons ``exp[i]`` and the read returns NaN (fail closed). Every array is preallocated; the
kernels allocate nothing (capture-safe).
"""

from __future__ import annotations

import logging

import warp as wp

log = logging.getLogger("alloy.precision")


@wp.struct
class Fixed64:
    """Per-destination exponent and int64 accumulator (SPEC §3.7)."""

    exp: wp.array[wp.int32]
    acc: wp.array[wp.int64]


FIXED64_EMPTY = wp.constant(-1000)  # exp of a destination with no nonzero term
FIXED64_POISON = wp.constant(1000)  # exp at or above this: a non-finite term reached the destination
MAX_MANTISSA_BITS = 61


def fixed64_mantissa_bits(max_terms: int) -> int:
    """``61 - bitlen(max_terms - 1)``: 61 for one term, 60 for two, 51 for 1024, 41 for 1e6."""
    if max_terms < 1:
        raise ValueError(f"max_terms must be >= 1, got {max_terms}")
    return MAX_MANTISSA_BITS - int(max_terms - 1).bit_length()


@wp.func
def fixed64_pow2(k: wp.int32) -> wp.float64:
    """Exactly ``2**k`` for ``-1022 <= k <= 1023``, built from the IEEE-754 bits."""
    return wp.cast(wp.uint64(k + 1023) << wp.uint64(52), wp.float64)


@wp.func
def fixed64_exponent(x: wp.float32) -> wp.int32:
    """Smallest ``e`` with ``|x| < 2**e`` (exact; subnormals give -126); ``FIXED64_POISON`` for inf/NaN."""
    u = wp.cast(x, wp.uint32)
    biased = wp.int32((u >> wp.uint32(23)) & wp.uint32(0xFF))
    if biased == 255:
        return FIXED64_POISON
    return biased - 126


@wp.func
def fixed64_note(f: Fixed64, i: int, x: float):
    """Pass 1 for one term (use inside a caller's kernel): raise ``exp[i]`` to cover ``x``."""
    if x != 0.0:
        wp.atomic_max(f.exp, i, fixed64_exponent(x))


@wp.func
def fixed64_add(f: Fixed64, i: int, x: float, mbits: int):
    """Pass 2: add ``x`` into destination ``i`` (after pass 1 covered every term of ``i``)."""
    e = f.exp[i]
    if x == 0.0 or e >= FIXED64_POISON:
        return
    if fixed64_exponent(x) > e:  # pass 1 missed this term or it is non-finite: fail closed
        wp.atomic_max(f.exp, i, FIXED64_POISON)
        return
    q = wp.int64(wp.round(wp.float64(x) * fixed64_pow2(mbits - e)))
    wp.atomic_add(f.acc, i, q)


@wp.func
def fixed64_read(f: Fixed64, i: int, mbits: int) -> wp.float64:
    """The float64 value of destination ``i``: 0 if empty, NaN if poisoned."""
    e = f.exp[i]
    if e >= FIXED64_POISON:
        return wp.float64(wp.nan)
    if e <= FIXED64_EMPTY:
        return wp.float64(0.0)
    return wp.float64(f.acc[i]) * fixed64_pow2(e - mbits)


@wp.kernel
def _prepare_kernel(f: Fixed64, values: wp.array[wp.float32], dest: wp.array[wp.int32]):
    k = wp.tid()
    fixed64_note(f, dest[k], values[k])


@wp.kernel
def _add_kernel(f: Fixed64, values: wp.array[wp.float32], dest: wp.array[wp.int32], mbits: wp.int32):
    k = wp.tid()
    fixed64_add(f, dest[k], values[k], mbits)


@wp.kernel
def _read_kernel(f: Fixed64, mbits: wp.int32, out: wp.array[wp.float64]):
    i = wp.tid()
    out[i] = fixed64_read(f, i, mbits)


def fixed64_alloc(n: int, device: wp.context.Devicelike = None) -> Fixed64:
    """A ``Fixed64`` with ``n`` destinations, reset."""
    f = Fixed64()
    f.exp = wp.full(n, FIXED64_EMPTY, dtype=wp.int32, device=device)
    f.acc = wp.zeros(n, dtype=wp.int64, device=device)
    return f


def fixed64_reset(f: Fixed64) -> None:
    """Empty every destination (start of a new sum)."""
    f.exp.fill_(FIXED64_EMPTY)
    f.acc.zero_()


def fixed64_prepare(f: Fixed64, values: wp.array[wp.float32], dest: wp.array[wp.int32]) -> None:
    """Pass 1: ``values[k]`` goes to destination ``dest[k]``."""
    _check_pair(values, dest)
    wp.launch(_prepare_kernel, dim=values.shape[0], inputs=[f, values, dest], device=values.device)


def fixed64_accumulate(f: Fixed64, values: wp.array[wp.float32], dest: wp.array[wp.int32], mbits: int) -> None:
    """Pass 2 (after ``fixed64_prepare`` saw every term of the same sum)."""
    _check_pair(values, dest)
    wp.launch(_add_kernel, dim=values.shape[0], inputs=[f, values, dest, mbits], device=values.device)


def fixed64_values(f: Fixed64, mbits: int, out: wp.array[wp.float64]) -> None:
    """Read every destination into ``out`` (float64)."""
    wp.launch(_read_kernel, dim=out.shape[0], inputs=[f, mbits, out], device=out.device)


def fixed64_sum(values: wp.array[wp.float32], dest: wp.array[wp.int32], n_dest: int,
                max_terms: int | None = None) -> wp.array[wp.float64]:
    """Order-independent per-destination sums of ``values`` (allocates; for host code and tests).

    ``max_terms`` defaults to ``len(values)`` (the bound that is always safe)."""
    mbits = fixed64_mantissa_bits(max_terms or max(values.shape[0], 1))
    f = fixed64_alloc(n_dest, values.device)
    fixed64_prepare(f, values, dest)
    fixed64_accumulate(f, values, dest, mbits)
    out = wp.zeros(n_dest, dtype=wp.float64, device=values.device)
    fixed64_values(f, mbits, out)
    return out


def _check_pair(values: wp.array, dest: wp.array) -> None:
    if values.shape != dest.shape:
        raise ValueError(f"values {values.shape} and dest {dest.shape} must have the same shape")
