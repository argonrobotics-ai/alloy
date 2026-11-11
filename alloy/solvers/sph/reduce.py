# SPDX-License-Identifier: Apache-2.0
"""Deterministic per-world reductions without float atomics (D9).

Particles of a world occupy one contiguous index range (``Model.particle_world_start``). ``WorldReducer`` sums a
per-particle value per world in two passes: fixed-size chunks aligned to each world's start are summed in float64,
then each world sums its chunks. The summation order of a world depends only on that world's own particles, so the
result is bitwise the same alone or batched, and run to run.

- On CPU each chunk and each world is summed serially, in index order.
- On CUDA (SCR-WP8-7 L1) one 256-thread block sums a chunk (``wp.tile_sum``, a fixed tree) and one block per world sums
  its chunks: the same chunk table, so the batch independence holds; the last bits differ from the CPU order.

``sum2_max`` / ``sum_max`` also return the per-world maximum of the (last) summed component (SCR-WP8-7 L2: the
solvers' error max, which had been one float atomic per particle). A maximum is exact in any order.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

BLOCK = wp.constant(256)  # threads per block on CUDA = the chunk size the tiled path needs
_NEG = wp.constant(-3.4e38)


# ----------------------------------------------------------------------------------------------- serial (CPU) path
@wp.kernel(enable_backward=False)
def _chunk_sum_kernel(values: wp.array[Any], chunk_begin: wp.array[wp.int32], chunk_end: wp.array[wp.int32],
                      partial: wp.array[wp.vec2d]):
    """partial[c] = (sum, max) of the chunk's values."""
    c = wp.tid()
    acc = wp.float64(0.0)
    mx = wp.float64(_NEG)
    for i in range(chunk_begin[c], chunk_end[c]):
        v = wp.float64(values[i])
        acc += v
        mx = wp.max(mx, v)
    partial[c] = wp.vec2d(acc, mx)


@wp.kernel(enable_backward=False)
def _chunk_sum2_kernel(values: wp.array[wp.vec2], chunk_begin: wp.array[wp.int32], chunk_end: wp.array[wp.int32],
                       partial: wp.array[wp.vec3d]):
    """partial[c] = (sum x, sum y, max y) of the chunk's values."""
    c = wp.tid()
    acc = wp.vec2d(0.0, 0.0)
    mx = wp.float64(_NEG)
    for i in range(chunk_begin[c], chunk_end[c]):
        v = values[i]
        acc += wp.vec2d(wp.float64(v[0]), wp.float64(v[1]))
        mx = wp.max(mx, wp.float64(v[1]))
    partial[c] = wp.vec3d(acc[0], acc[1], mx)


# ---------------------------------------------------------------------------------------------- tiled (CUDA) path
@wp.kernel(enable_backward=False)
def _chunk_tile_sum_kernel(values: wp.array[Any], chunk_begin: wp.array[wp.int32], chunk_end: wp.array[wp.int32],
                           partial: wp.array[wp.vec2d]):
    c, lane = wp.tid()
    i = chunk_begin[c] + lane
    v = wp.float64(0.0)
    m = wp.float64(_NEG)
    if i < chunk_end[c]:
        v = wp.float64(values[i])
        m = v
    s = wp.tile_extract(wp.tile_sum(wp.tile(v)), 0)
    mx = wp.tile_extract(wp.tile_max(wp.tile(m)), 0)
    if lane == 0:
        partial[c] = wp.vec2d(s, mx)


@wp.kernel(enable_backward=False)
def _chunk_tile_sum2_kernel(values: wp.array[wp.vec2], chunk_begin: wp.array[wp.int32],
                            chunk_end: wp.array[wp.int32], partial: wp.array[wp.vec3d]):
    c, lane = wp.tid()
    i = chunk_begin[c] + lane
    a = wp.float64(0.0)
    b = wp.float64(0.0)
    m = wp.float64(_NEG)
    if i < chunk_end[c]:
        v = values[i]
        a = wp.float64(v[0])
        b = wp.float64(v[1])
        m = b
    sa = wp.tile_extract(wp.tile_sum(wp.tile(a)), 0)
    sb = wp.tile_extract(wp.tile_sum(wp.tile(b)), 0)
    mx = wp.tile_extract(wp.tile_max(wp.tile(m)), 0)
    if lane == 0:
        partial[c] = wp.vec3d(sa, sb, mx)


for _dtype in (wp.float32, wp.float64):
    wp.overload(_chunk_sum_kernel, {"values": wp.array[_dtype]})
    wp.overload(_chunk_tile_sum_kernel, {"values": wp.array[_dtype]})


# -------------------------------------------------------------------------------------------- per-world epilogues
@wp.kernel(enable_backward=False)
def _world_sum_kernel(partial: wp.array[wp.vec2d], world_chunk_start: wp.array[wp.int32], out: wp.array[wp.float64],
                      out_max: wp.array[wp.float32], with_max: int):
    w = wp.tid()
    acc = wp.float64(0.0)
    mx = wp.float64(_NEG)
    for c in range(world_chunk_start[w], world_chunk_start[w + 1]):
        acc += partial[c][0]
        mx = wp.max(mx, partial[c][1])
    out[w] = acc
    if with_max != 0:
        out_max[w] = wp.float32(wp.max(mx, wp.float64(0.0)))


@wp.kernel(enable_backward=False)
def _world_sum2_kernel(partial: wp.array[wp.vec3d], world_chunk_start: wp.array[wp.int32],
                       out: wp.array[wp.vec2d], out_max: wp.array[wp.float32], with_max: int):
    w = wp.tid()
    acc = wp.vec2d(0.0, 0.0)
    mx = wp.float64(_NEG)
    for c in range(world_chunk_start[w], world_chunk_start[w + 1]):
        q = partial[c]
        acc += wp.vec2d(q[0], q[1])
        mx = wp.max(mx, q[2])
    out[w] = acc
    if with_max != 0:
        out_max[w] = wp.float32(wp.max(mx, wp.float64(0.0)))


@wp.kernel(enable_backward=False)
def _world_tile_sum_kernel(partial: wp.array[wp.vec2d], world_chunk_start: wp.array[wp.int32],
                           out: wp.array[wp.float64], out_max: wp.array[wp.float32], with_max: int):
    w, lane = wp.tid()
    acc = wp.float64(0.0)
    mx = wp.float64(_NEG)
    for c in range(world_chunk_start[w] + lane, world_chunk_start[w + 1], BLOCK):
        acc += partial[c][0]
        mx = wp.max(mx, partial[c][1])
    s = wp.tile_extract(wp.tile_sum(wp.tile(acc)), 0)
    m = wp.tile_extract(wp.tile_max(wp.tile(mx)), 0)
    if lane == 0:
        out[w] = s
        if with_max != 0:
            out_max[w] = wp.float32(wp.max(m, wp.float64(0.0)))


@wp.kernel(enable_backward=False)
def _world_tile_sum2_kernel(partial: wp.array[wp.vec3d], world_chunk_start: wp.array[wp.int32],
                            out: wp.array[wp.vec2d], out_max: wp.array[wp.float32], with_max: int):
    w, lane = wp.tid()
    a = wp.float64(0.0)
    b = wp.float64(0.0)
    mx = wp.float64(_NEG)
    for c in range(world_chunk_start[w] + lane, world_chunk_start[w + 1], BLOCK):
        q = partial[c]
        a += q[0]
        b += q[1]
        mx = wp.max(mx, q[2])
    sa = wp.tile_extract(wp.tile_sum(wp.tile(a)), 0)
    sb = wp.tile_extract(wp.tile_sum(wp.tile(b)), 0)
    m = wp.tile_extract(wp.tile_max(wp.tile(mx)), 0)
    if lane == 0:
        out[w] = wp.vec2d(sa, sb)
        if with_max != 0:
            out_max[w] = wp.float32(wp.max(m, wp.float64(0.0)))


def chunk_table(world_start: np.ndarray, chunk: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(chunk_begin, chunk_end, world_chunk_start) for contiguous world ranges ``world_start`` (length W + 1)."""
    begins: list[int] = []
    ends: list[int] = []
    starts = [0]
    for w in range(len(world_start) - 1):
        lo, hi = int(world_start[w]), int(world_start[w + 1])
        for b in range(lo, hi, chunk):
            begins.append(b)
            ends.append(min(b + chunk, hi))
        starts.append(len(begins))
    return (
        np.asarray(begins, dtype=np.int32),
        np.asarray(ends, dtype=np.int32),
        np.asarray(starts, dtype=np.int32),
    )


class WorldReducer:
    """Per-world float64 sums (and maxima) over contiguous particle ranges; capture-safe."""

    def __init__(self, world_start: np.ndarray, device: wp.context.Device, chunk: int = BLOCK):
        begin, end, wstart = chunk_table(np.asarray(world_start), chunk)
        self.world_count = len(world_start) - 1
        self.n_chunks = len(begin)
        self.device = wp.get_device(device)
        self.tiled = self.device.is_cuda and chunk == BLOCK
        self._begin = wp.array(begin if len(begin) else np.zeros(1, np.int32), dtype=wp.int32, device=device)
        self._end = wp.array(end if len(end) else np.zeros(1, np.int32), dtype=wp.int32, device=device)
        self._wstart = wp.array(wstart, dtype=wp.int32, device=device)
        self._partial = wp.zeros(max(self.n_chunks, 1), dtype=wp.vec2d, device=device)
        self._partial2 = wp.zeros(max(self.n_chunks, 1), dtype=wp.vec3d, device=device)
        self._no_max = wp.zeros(1, dtype=wp.float32, device=device)

    def _run(self, chunk_k, world_k, values, partial, out, out_max) -> None:
        with_max = int(out_max is not None)
        mx = out_max if out_max is not None else self._no_max
        if self.n_chunks > 0:
            ins = [values, self._begin, self._end, partial]
            if self.tiled:
                wp.launch_tiled(chunk_k[1], dim=[self.n_chunks], inputs=ins, block_dim=BLOCK, device=self.device)
            else:
                wp.launch(chunk_k[0], dim=self.n_chunks, inputs=ins, device=self.device)
        if self.world_count > 0:
            ins = [partial, self._wstart, out, mx, with_max]
            if self.tiled:
                wp.launch_tiled(world_k[1], dim=[self.world_count], inputs=ins, block_dim=BLOCK, device=self.device)
            else:
                wp.launch(world_k[0], dim=self.world_count, inputs=ins, device=self.device)

    def sum(self, values: wp.array, out: wp.array, out_max: wp.array | None = None) -> None:
        """out[w] = sum of values over world w's range (float64); ``out_max`` (float32, optional): their maximum,
        floored at 0 (the solvers' errors are >= 0)."""
        self._run((_chunk_sum_kernel, _chunk_tile_sum_kernel), (_world_sum_kernel, _world_tile_sum_kernel), values,
                  self._partial, out, out_max)

    def sum2(self, values: wp.array, out: wp.array, out_max: wp.array | None = None) -> None:
        """Two per-world float64 sums at once: ``values`` vec2 per particle, ``out`` vec2d per world; ``out_max``
        (float32, optional): the maximum of the second component, floored at 0."""
        self._run((_chunk_sum2_kernel, _chunk_tile_sum2_kernel), (_world_sum2_kernel, _world_tile_sum2_kernel),
                  values, self._partial2, out, out_max)
