# SPDX-License-Identifier: Apache-2.0
"""Material sort of the wavefront queue: the per-path key (material class, then path id) and the class ranges of the
sorted queue (paths per material, for the material-specialised shade kernels).

Example::

    shift = key_shift(n_mats + 2)                        # bits left for the path id
    ranges = spec_launches(hist, n_mats, min_count=4096) # [(kernel, offset, count), ...]
"""
from __future__ import annotations

import warp as wp

from ..pbr.shade_data import ShadeData
from .state import F_TERM_NEXT, PathState


@wp.kernel(enable_backward=False)
def sort_key_kernel(ps: PathState, q: wp.array(dtype=wp.int32), sd: ShadeData, n_mats: wp.int32, shift: wp.int32,
                    keys: wp.array(dtype=wp.int32)):
    """Material of the hit (misses and terminated paths after all materials), then path id: the stable radix sort
    keeps each material's paths in pixel order, so the next bounce's rays stay coherent."""
    i = wp.tid()
    p = q[i]
    inst = ps.hit_inst[p]
    k = n_mats
    if inst >= 0:
        k = n_mats + 1
        if (ps.flags[p] & F_TERM_NEXT) == 0:
            m = sd.inst_src[inst]
            k = sd.tri_mat[sd.mesh_tri_off[m] + ps.hit_prim[p]]
    keys[i] = (k << shift) | p  # paths < 2^shift, key classes (n_mats + 2) <= 2^(31 - shift): key_shift()


@wp.func
def lower_bound(a: wp.array(dtype=wp.int32), n: int, x: int) -> int:
    lo = int(0)
    hi = n
    while lo < hi:
        mid = (lo + hi) >> 1
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@wp.kernel(enable_backward=False)
def key_bins(keys: wp.array(dtype=wp.int32), n: int, shift: int, hist: wp.array(dtype=wp.int32)):
    """Paths per key class (material, then miss, terminated) of the sorted queue: binary search of the class bounds
    (one thread per class; no atomics)."""
    k = wp.tid()
    hi = wp.where(k + 1 >= (1 << (31 - shift)), n, lower_bound(keys, n, (k + 1) << shift))
    hist[k] = hi - lower_bound(keys, n, k << shift)


def key_shift(n_classes: int) -> int:
    """Bits left for the path id in an int32 sort key below ``n_classes`` key classes."""
    return 31 - max(1, (n_classes - 1).bit_length())


def spec_launches(hist: list[int], n_mats: int, min_count: int) -> list[tuple[int, int, int]]:
    """(kernel, offset, count) launches over the material ranges of the sorted queue: kernel k >= 0 is material k's
    specialised kernel, -1 the escape kernel (bins n_mats, n_mats + 1: misses, terminated paths), -2 the generic
    kernel, which takes runs of neighbouring ranges smaller than ``min_count`` (one launch instead of many tiny ones).

    Example::

        spec_launches([5000, 10, 20, 0, 6000], 3, 4096)   # [(0, 0, 5000), (-2, 5000, 30), (-1, 5030, 6000)]
    """
    out, off, run0, run_n = [], 0, 0, 0
    for k, c in enumerate(hist):
        if c == 0:
            continue
        if c < min_count:
            if run_n == 0:
                run0 = off
            run_n += c
        else:
            if run_n:
                out.append((-2, run0, run_n))
                run_n = 0
            out.append((k if k < n_mats else -1, off, c))
        off += c
    if run_n:
        out.append((-2, run0, run_n))
    return out
