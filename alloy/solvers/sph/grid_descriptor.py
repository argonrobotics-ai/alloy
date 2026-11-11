# SPDX-License-Identifier: Apache-2.0
"""Keep a warp HashGrid's own device descriptor across CUDA graph replays (OSS-4's find, 2026-11-08).

Warp 1.18's device ``HashGrid.build`` ends with a host-to-device copy of the grid descriptor (its buffers, dims, point
count and cell width) from one ``static HashGrid_t<Type>`` host struct, shared by every grid of that float type
(warp/native/hashgrid.cpp, ``hash_grid_update_device_impl``). A CUDA graph records that copy's source address, not
its value. On replay, every captured build therefore copies in the descriptor of whichever float32 grid the host built
last: another SPH solver in the same process, the heat plug-in's neighbour grid, a camera's or a user's grid. The
neighbour queries then read that grid's buffers: two pour sims stepped in turn on an L4 differed by up to 5.7e-5 m,
and with a camera grid in between the build kernel read out of bounds.

``GridDescriptor(grid)`` copies the grid's correct device descriptor into a buffer of its own after every eager
build. After every captured build it copies that buffer back over the grid. That is a device-to-device copy, which a
graph replays with fixed addresses. It assumes the descriptor does not change between builds (the same point count,
cell width and buffers), as for SPH's neighbour grids and MPM rheology's smoothing grid. On the CPU, or without a
graph, it does nothing. Use it for any HashGrid rebuilt inside a stepped (graph-captured) path, a plug-in's too:
``desc = GridDescriptor(grid)`` once, then ``desc.build(points, cell, groups=...)`` in place of ``grid.build``.

The layout is warp 1.18's ``HashGrid_t<Type>``: five pointers, seven ints, a context pointer, then two values of the
grid's float type (cell width and its inverse), padded to 8 bytes: 88 bytes for float16 and float32, 96 for float64.
The first eager build checks the dims, point count and cell width at their offsets. If they do not match (warp
changed the struct), the guard turns off and says so once: grids behave as they do in stock warp.
"""

from __future__ import annotations

import logging

import numpy as np
import warp as wp

log = logging.getLogger("alloy.sph")

DIMS_AT, NUM_AT, CELL_AT = 40, 52, 80    # byte offsets (warp 1.18 native/hashgrid.h): dim_x/y/z, num_points, cell_width
_FLOAT = {wp.float16: np.float16, wp.float32: np.float32, wp.float64: np.float64}


def descriptor_bytes(dtype) -> int:
    """sizeof(HashGrid_t<Type>) for a grid of float type ``dtype``: 80 bytes, two values, padded to 8."""
    size = np.dtype(_FLOAT[dtype]).itemsize
    return (CELL_AT + 2 * size + 7) // 8 * 8


class GridDescriptor:
    """The capture-safe descriptor of one ``wp.HashGrid`` (module docstring)."""

    def __init__(self, grid: wp.HashGrid, dim: int | None = None, device: wp.context.Device | None = None):
        """``dim`` and ``device`` are optional (05baba777's signature): read from the grid, checked when given."""
        dtype = getattr(grid, "dtype", wp.float32)
        self.dims = [int(getattr(grid, f"_dim_{a}")) for a in "xyz"]
        if dim is not None and self.dims != [int(dim)] * 3:
            raise ValueError(f"GridDescriptor: dim {dim} but the grid has dims {self.dims}")
        if device is not None and wp.get_device(device) != wp.get_device(grid.device):
            raise ValueError(f"GridDescriptor: device {device} but the grid is on {grid.device}")
        device = grid.device
        self.on = bool(getattr(wp.get_device(device), "is_cuda", False)) and dtype in _FLOAT
        self.device, self.ready, self.warned = device, False, False
        self.grid = grid                         # the view below points into this grid's memory: keep it alive
        self.np_float = _FLOAT.get(dtype, np.float32)
        if self.on:
            nbytes = descriptor_bytes(dtype)
            self.view = wp.array(ptr=int(grid.id), dtype=wp.uint8, shape=(nbytes,), device=device, copy=False)
            self.saved = wp.zeros(nbytes, dtype=wp.uint8, device=device)

    def build(self, points: wp.array, cell: float, groups: wp.array | None = None) -> None:
        """``grid.build(points, cell, groups=groups)``, kept right under a CUDA graph (module docstring)."""
        self.grid.build(points, cell, groups=groups)
        self.after_build(int(points.shape[0]), cell)

    def after_build(self, n: int, cell: float) -> None:
        """Call right after ``grid.build``: keep the descriptor (eager) or put it back (inside a capture)."""
        if not self.on:
            return
        if wp.get_device(self.device).is_capturing:
            if self.ready:
                wp.copy(self.view, self.saved)
            elif not self.warned:
                self.warned = True
                log.warning("GridDescriptor: a HashGrid's first build is inside a graph capture, so its descriptor "
                            "is unguarded in that graph (build once eagerly first)")
            return
        wp.copy(self.saved, self.view)
        if not self.ready:
            self.ready = self._layout_ok(n, cell)
            self.on = self.ready

    def _layout_ok(self, n: int, cell: float) -> bool:
        raw = self.saved.numpy()
        dims = np.frombuffer(raw[DIMS_AT:DIMS_AT + 12].tobytes(), np.int32)
        num = int(np.frombuffer(raw[NUM_AT:NUM_AT + 4].tobytes(), np.int32)[0])
        size = np.dtype(self.np_float).itemsize
        width = float(np.frombuffer(raw[CELL_AT:CELL_AT + size].tobytes(), self.np_float)[0])
        if list(dims) == self.dims and num == int(n) and abs(width - cell) <= 1e-3 * abs(cell):
            return True
        log.warning("GridDescriptor: warp's HashGrid layout is not the expected one (dims %s, points %d, cell %.4g "
                    "vs %s, %d, %.4g); graph replays keep stock warp's behaviour (another grid of the same float type "
                    "built between steps can overwrite this one)", list(dims), num, width, self.dims, n, cell)
        return False
