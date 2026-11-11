# SPDX-License-Identifier: Apache-2.0
"""Where the liquid field is evaluated: blocks of 2^BLOCK_SHIFT grid nodes a side that some particle's support reaches.

``mark_blocks`` sets ``g.blocks`` to 1 for every block touched by the box around a particle's support (radius R, plus
the half length of a stream segment); ``liquid_kernels.implicit`` writes "outside" (``-R``) to the nodes of the other
blocks without querying the hash grid. That is the value the sums give there anyway (no particle within reach), so
the field, and the surface, keep their bits; at 512 pours the field went from most of the liquid update to a fraction.

Example::

    g.bshift = BLOCK_SHIFT
    g.nbx, g.nby, g.nbz = block_counts(g.nx, g.ny, g.nz)
    wp.launch(mark_blocks, dim=total, inputs=[pts, world, flag, half, g, R], device=dev)
"""
from __future__ import annotations

import warp as wp

from .liquid_kernels import Grid

BLOCK_SHIFT = 2  # blocks of 4 nodes a side


def block_counts(nx: int, ny: int, nz: int, shift: int | None = None) -> tuple[int, int, int]:
    """Blocks per axis of an nx x ny x nz node grid (blocks of 2^shift nodes a side)."""
    b = 1 << (BLOCK_SHIFT if shift is None else shift)
    return -(-nx // b), -(-ny // b), -(-nz // b)


@wp.kernel(enable_backward=False)
def mark_blocks(pts: wp.array(dtype=wp.vec3), world: wp.array(dtype=wp.int32), flag: wp.array(dtype=wp.int32),
                half: wp.array(dtype=float), g: Grid, radius: float):
    """Mark the blocks within one support (plus a segment's half length) of each packed particle (any order: every
    writer stores 1)."""
    s = wp.tid()
    w = world[s]
    reach = radius + wp.where(flag[s] != 0, half[s], 0.0)
    rel = (pts[s] - g.tile[w]) / g.cell
    k = reach / g.cell + 1.0  # one node more each way: the field's own float rounding stays inside
    lo = wp.vec3i(wp.max(int(wp.floor(rel[0] - k)), 0), wp.max(int(wp.floor(rel[1] - k)), 0),
                  wp.max(int(wp.floor(rel[2] - k)), 0))
    hi = wp.vec3i(wp.min(int(wp.ceil(rel[0] + k)), g.nx - 1), wp.min(int(wp.ceil(rel[1] + k)), g.ny - 1),
                  wp.min(int(wp.ceil(rel[2] + k)), g.nz - 1))
    sh = g.bshift
    for bx in range(lo[0] >> sh, (hi[0] >> sh) + 1):
        for by in range(lo[1] >> sh, (hi[1] >> sh) + 1):
            for bz in range(lo[2] >> sh, (hi[2] >> sh) + 1):
                g.blocks[((w * g.nbx + bx) * g.nby + by) * g.nbz + bz] = 1
