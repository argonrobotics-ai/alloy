# SPDX-License-Identifier: Apache-2.0
"""Out of GPU memory in Warp's sparse triplet build: a clear Alloy error instead of CUDA error 700.

Warp's native ``wp_bsr_matrix_from_triplets_device`` (warp/native/sparse.cu, Warp 1.18) takes its scratch through
``ScopedTemporary`` and ``wp_alloc_device`` and never checks it. When the GPU is full it prints "CUDA error 2: out of
memory", runs the radix sort on null buffers (sparse.cu:1315, "error 101"), and the next kernels fail with error 700
(illegal memory access). Error 700 leaves the CUDA context unusable. Newton's implicit MPM reaches it through
``warp.fem`` (``integrate`` / ``interpolate`` into a matrix) every step. Nov 8 pass, e97d31: soil_dig at 64 worlds on
an L4. The collider matrices hold 157 M triplets, and the plasticity system's 7.9 M-triplet build then ran out with
0.45 GB free.

``install()`` wraps that native entry point (Warp's Python side looks it up at each call, sparse.py:1092). Before
every build on CUDA (not while a graph is being captured), it reserves the native scratch through Warp's allocator:
2 x 8 B row/column keys and 2 x 4 B block offsets and indices per triplet (sparse.cu:1257-1264), plus ``MARGIN`` for
CUB's sort and encode scratch. The reservation comes from the same
stream-ordered pool, and is freed before the native call takes it. When the reservation fails, the build raises
``SparseMemoryError`` before any native work, and the context stays usable. Inside a capture, the graph's memory
nodes fail at instantiate instead (``IntegratedSimulation`` falls back to eager periods, and those meet this guard).
Small builds are reserved too (SCR-WP12-3 follow-up 1: no path to error 700); ``native_guard`` covers the grid
builds, sorts and run-length encodes of the same step.
"""

from __future__ import annotations

import logging
from typing import Any

import warp as wp

from alloy.core.errors import CapacityError

log = logging.getLogger("alloy.integration")

__all__ = ["BYTES_PER_TRIPLET", "MARGIN", "NATIVE", "SparseMemoryError", "install", "scratch_bytes"]

NATIVE = "wp_bsr_matrix_from_triplets_device"
BYTES_PER_TRIPLET = 24              # BsrRowCol (uint64) x 2 + int x 2, per triplet
MARGIN = 64 << 20                   # CUB radix sort / run-length encode / scan scratch
AUDITED = "1.18."                   # Warp versions whose native argument order was read (nnz is argument 5)
FIX = ("run fewer worlds (the scene's memory grows with the world count), a coarser MPM voxel or fewer particles, "
       "or a GPU with more memory")


class SparseMemoryError(CapacityError):
    """The GPU has no room for a sparse triplet build's scratch; raised before Warp's native code runs."""

    def __init__(self, message: str, *, nnz: int, need: int) -> None:
        super().__init__(message)
        self.nnz, self.need, self.fix = nnz, need, FIX


def scratch_bytes(nnz: int) -> int:
    """Bytes the native build allocates for ``nnz`` triplets (keys, offsets, indices, CUB scratch)."""
    return int(nnz) * BYTES_PER_TRIPLET + MARGIN


def _reserve(device: Any, nnz: int) -> None:
    from alloy.integration.native_guard import probe  # noqa: PLC0415

    need = scratch_bytes(nnz)
    if not probe(device, need):
        raise SparseMemoryError(
            f"out of GPU memory on {device}: a sparse matrix build of {nnz:,} triplets (an MPM step's warp.fem "
            f"system) needs {need / 2**30:.2f} GB of scratch, and the device has no room for it. Warp's native code "
            f"would go on without it and fail with CUDA error 700. Fix: {FIX}", nnz=nnz, need=need)


def _guarded(native: Any) -> Any:
    def call(*args: Any) -> Any:
        nnz = int(args[4])                              # (block_size, scalar_size, rows, cols, nnz, ...)
        # sparse.py calls the native function inside ScopedDevice(dest.device) (sparse.py:1094), so the current
        # device is the matrix's; if that scope ever goes, the reservation lands on the default device
        device = wp.get_device()
        if device.is_cuda and not device.is_capturing:
            _reserve(device, nnz)
        return native(*args)

    call.alloy_sparse_guard = True                      # type: ignore[attr-defined]
    call.native = native                                # type: ignore[attr-defined]
    return call


def _audited(native: Any) -> bool:
    """The Warp version read for this wrap, and the argument types it relies on (5 C ints, then a pointer)."""
    import ctypes  # noqa: PLC0415

    types = list(getattr(native, "argtypes", None) or [])
    return (wp.__version__.startswith(AUDITED) and len(types) > 5 and all(t is ctypes.c_int for t in types[:5])
            and hasattr(types[5], "contents"))


def install() -> bool:
    """Wrap Warp's native CUDA triplet build once per process; True when this call installed it. Also puts in
    warp.fem's integer packed-grid cell lookup (``nanogrid_lookup``), which every MPM entry's build needs too."""
    from alloy.integration import nanogrid_lookup  # noqa: PLC0415

    nanogrid_lookup.install()
    wp.init()
    from warp._src.context import runtime  # noqa: PLC0415

    core = runtime.core
    native = getattr(core, NATIVE, None)
    if native is None or getattr(native, "alloy_sparse_guard", False):
        return False
    if not _audited(native):
        log.warning("sparse guard not installed: Warp %s's %s is not the audited signature (Warp %sx, nnz the 5th "
                    "int); an out-of-memory in it can still end in CUDA error 700", wp.__version__, NATIVE, AUDITED)
        return False
    setattr(core, NATIVE, _guarded(native))
    log.debug("sparse guard: %s reserves its scratch first (CUDA out of memory -> SparseMemoryError)", NATIVE)
    return True
