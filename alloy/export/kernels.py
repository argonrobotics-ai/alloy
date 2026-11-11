# SPDX-License-Identifier: Apache-2.0
"""Device gathers for the frame export (SCR-WP13-EXPORT): the picked worlds' rows into a compact staging buffer.

Every kernel only reads the simulation's arrays and writes the export's own, one thread per output item (no atomics),
so a frame never changes the simulation and the same state always gives the same frame, bit for bit.
"""

from __future__ import annotations

import warp as wp


@wp.kernel
def gather_transforms(src: wp.array[wp.transform], rows: wp.array2d[wp.int32], out: wp.array2d[wp.transform]):
    p, i = wp.tid()
    out[p, i] = src[rows[p, i]]


@wp.kernel
def gather_points(src: wp.array[wp.vec3], rows: wp.array2d[wp.int32], out: wp.array2d[wp.vec3]):
    p, i = wp.tid()
    out[p, i] = src[rows[p, i]]


@wp.kernel
def gather_flags(src: wp.array[wp.int32], rows: wp.array2d[wp.int32], col: int, out: wp.array2d[wp.int32]):
    """One particle set's live flags into its columns of ``out`` (``col`` is the set's first column)."""
    p, i = wp.tid()
    out[p, col + i] = src[rows[p, i]]


@wp.kernel
def gather_bytes(src: wp.array[wp.int8], rows: wp.array2d[wp.int32], out: wp.array2d[wp.int8]):
    p, i = wp.tid()
    out[p, i] = src[rows[p, i]]


@wp.kernel
def rod_points(body_q: wp.array[wp.transform], rows: wp.array2d[wp.int32], local: wp.array[wp.vec3],
               out: wp.array2d[wp.vec3]):
    """Each centre-line node: its segment body's pose applied to the node's body-frame point."""
    p, i = wp.tid()
    out[p, i] = wp.transform_point(body_q[rows[p, i]], local[i])


@wp.kernel
def topology_counts(bond_range: wp.array2d[wp.int32], order: wp.array[wp.int32], intact: wp.array[wp.int8],
                    body_rows: wp.array2d[wp.int32], body_active: wp.array[wp.int8], n_bodies: int,
                    out: wp.array2d[wp.int32]):
    """Per picked world: its broken bonds (column 0) and its retired bodies (column 1). The host combines them with
    the dormant tets into ``topology_version``."""
    p = wp.tid()
    n = int(0)
    for k in range(bond_range[p, 0], bond_range[p, 1]):
        if intact[order[k]] == wp.int8(0):
            n += 1
    r = int(0)
    for i in range(n_bodies):
        if body_active[body_rows[p, i]] == wp.int8(0):
            r += 1
    out[p, 0] = n
    out[p, 1] = r


@wp.kernel
def tet_live(materials: wp.array2d[wp.float32], rows: wp.array2d[wp.int32], out: wp.array2d[wp.int8]):
    """1 for a tet that carries material, 0 for a dormant virtual-node fragment or a whole tet a split retired
    (WP10: draw a tet only when its tet_materials row is non-zero)."""
    p, i = wp.tid()
    t = rows[p, i]
    if t < 0:                                           # padding (a world with fewer cuttable tets)
        out[p, i] = wp.int8(1)
        return
    live = materials[t, 0] != 0.0 or materials[t, 1] != 0.0 or materials[t, 2] != 0.0
    out[p, i] = wp.where(live, wp.int8(1), wp.int8(0))


__all__ = ["gather_bytes", "gather_flags", "gather_points", "gather_transforms", "rod_points", "tet_live",
           "topology_counts"]
