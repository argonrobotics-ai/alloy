# SPDX-License-Identifier: Apache-2.0
"""Out of GPU memory in Warp's native builders on an MPM step: a clear Alloy error instead of CUDA error 700.

Warp's native code allocates scratch with ``wp_alloc_device`` and goes on when it gets null (Warp 1.18, read in
warp/native/): the kernels that follow write through null pointers, and the next CUDA call fails with error 700
(illegal memory access), which leaves the context unusable. ``sparse_guard`` covers the sparse triplet build. This
module covers the other natives an implicit-MPM step calls every step (SCR-WP12-3 follow-up 1; an L4 run with ~2.5 GB
free ended in 700 inside ``rebuild_count_points``):

- NanoVDB grid builds and in-place rebuilds (volume_builder.cu): per point, 7 x 8 B keys + 4 B run counts
  (``rebuild_alloc_key_scratch``), CUB's sort scratch (~8 B), and 12 B per leaf for the active-voxel prefix; then the
  grid itself (exact builds) or nothing (rebuilds into a reserved grid). Unchecked.
- ``wp_runlength_encode_int_device`` (runlength_encode.cu): CUB scratch, unchecked.
- ``wp_radix_sort_pairs_*_device`` (sort.cu): checks, but returns without raising, so the keys stay unsorted. Guarded
  so that a full GPU is an error, not wrong data.

``install()`` wraps each audited native (Warp's Python side looks them up at each call). Before a call on CUDA (not
while a graph is being captured: its memory nodes fail at instantiate, where ``IntegratedSimulation`` handles it), it
reserves the call's scratch bound through Warp's allocator and frees it; the native call then takes it from the same
stream-ordered pool. When the reservation fails, the call raises ``NativeMemoryError`` before any native work, and the
context stays usable. The per-point and per-node terms are upper bounds; the margins are heuristics for CUB scratch
and an exact grid's internal nodes (4 MiB is ~100 lower nodes, far above a per-world MPM grid, whose material spans
one or two; a single grid over many spread-out worlds could exceed it). Every byte of margin also lifts the mempool's
high-water mark, which the eager path pins as its release threshold (alloy/parallel/graphs.py).

- After any CUDA allocation failure in the process, the next native built on CUB fails once in its sizing call with
  "CUDA error 101: invalid device ordinal", whatever memory is free again by then; kernels, allocations and
  synchronizes in between do not clear it (Warp 1.18, driver 580, one T4, 2026-11-10: SCR-OSS22-9). A prefix sum
  then raises; a radix sort returns with its keys unsorted, an array sum and a run-length encode with their output
  unwritten. So a failure Alloy handles and goes on from (a step graph that did not fit, a refused native) must not
  leave that error behind: ``absorb_stale_error`` runs one tiny prefix sum to take it. The guard marks the device
  when its own reservation fails and absorbs before the next guarded native; ``graph_budget.release`` absorbs after
  a graph ran out of memory. Prefix sums, array sums and inner products are guarded for that (their scratch is small).

- BVH and mesh builds, rebuilds and mesh refits (bvh.cu, mesh.cu; GAPS-NOW #5a memory guards): every device
  allocation of a create is unchecked (a null pointer is written through), and so are an LBVH rebuild's. An MPM step's
  moving colliders refit their meshes. Bounds per item n (Warp 1.18, read in the sources): LBVH scratch 52n (sort
  keys, indices, deltas, ranges), the tree kept 84n (node bounds, parents, counts, indices), a mesh's bounds 24n and
  its winding-number data 288n; cuBQL's own builder scratch is not in Warp's tree, so its known terms are doubled. A
  rebuild or refit only names an id: its size comes from the create, which the guard records.
"""

from __future__ import annotations

import ctypes
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import warp as wp

from alloy.core.errors import CapacityError

log = logging.getLogger("alloy.integration")

__all__ = ["GUARDS", "SIZES", "NativeMemoryError", "absorb_stale_error", "install", "probe", "scratch_bytes"]

AUDITED = "1.18."                   # Warp versions whose native signatures and allocations were read
KEY_BYTES = 68                      # per point: 7 x 8 B keys + 4 B run counts + ~8 B CUB sort scratch
LEAF_PREFIX = 12                    # per leaf: active counts (4 B) + prefix (8 B), active-voxel grids
INDEX_LEAF = 128                    # an index leaf: 96 B in NanoVDB, rounded up
LOWER_NODE = 36 << 10               # 16^3 child table x 8 B + masks
UPPER_NODE = 288 << 10              # 32^3 child table x 8 B + masks
LEAVES_PER_POINT = 4                # a rebuild's prefix arrays follow the grid's leaf capacity: bounded per point
VOLUME_MARGIN = 4 << 20             # CUB scratch, an exact grid's internal nodes and header (heuristic)
SMALL_MARGIN = 1 << 20              # CUB scratch of a sort or run-length encode (heuristic)
BVH_SCRATCH = {2: 52, -1: 92}       # per item, by constructor (LBVH 2, cuBQL -1; SAH / median build on the host)
BVH_KEPT = 84                       # per item: node bounds 2 x 16 B x 2n, parents and counts 2 x 4 B x 2n, indices
MESH_KEPT = 24                      # per triangle: its bounds (lowers, uppers)
MESH_WINDING = 288                  # per triangle: solid-angle data 144 B x 2n (support_winding_number)
CUBQL_FACTOR = 2                    # cuBQL's builder scratch is not in Warp's tree: twice the known terms
BUILD_MARGIN = 4 << 20              # CUB radix sort / scan scratch (heuristic)
SIZES: dict[int, tuple[int, int]] = {}   # BVH / mesh id -> (items, constructor), recorded at create; never pruned: a
#   destroyed id keeps its entry until a later create reuses the address (one small tuple per BVH / mesh ever made)
_STALE: set[str] = set()            # CUDA devices with an allocation failure whose CUB error is not absorbed yet
FIX = ("run fewer worlds (the scene's memory grows with the world count), a coarser MPM voxel or fewer particles, "
       "or a GPU with more memory")


class NativeMemoryError(CapacityError):
    """The GPU has no room for a Warp native call's scratch; raised before the native code runs."""

    def __init__(self, message: str, *, native: str, count: int, need: int) -> None:
        super().__init__(message)
        self.native, self.count, self.need, self.fix = native, count, need, FIX


@dataclass(frozen=True)
class Guard:
    """One native: what it builds, its element count's argument, its scratch bound and the signature it was read at."""

    what: str
    count_arg: int
    need: Callable[[tuple], int]
    argtypes: tuple
    context_arg: bool               # argument 0 is the CUDA context (else the current device's)
    note: str = "an MPM step's grid or warp.fem system"
    by_id: bool = False             # argument 0 is a BVH / mesh id: the count is the create's (``SIZES``)
    records: tuple[int, int] | None = None   # create: (items arg, constructor arg) recorded under the id it returns


def _grid(leaves: int, lower: int, upper: int, leaf_bytes: int) -> int:
    return leaves * leaf_bytes + lower * LOWER_NODE + upper * UPPER_NODE


def _voxels(a: tuple) -> int:       # (ctx, points, n, mask, xform, translation, world, rebuildable, voxels, leaf, ...)
    n = int(a[2])
    if a[7]:
        return n * KEY_BYTES + int(a[9]) * LEAF_PREFIX + _grid(int(a[9]), int(a[10]), int(a[11]), INDEX_LEAF)
    return n * (KEY_BYTES + LEAF_PREFIX + INDEX_LEAF)


def _tiles(a: tuple) -> int:        # (ctx, points, n, mask, xform, transl., world, bg, bg_size, type, rebuildable..)
    n, leaf = int(a[2]), 96 + 512 * int(a[8])
    return n * KEY_BYTES + (_grid(int(a[11]), int(a[12]), int(a[13]), leaf) if a[10] else n * leaf)


def _index_tiles(a: tuple) -> int:  # (ctx, points, n, mask, xform, translation, world, rebuildable, tiles, lower, ...)
    n = int(a[2])
    return n * KEY_BYTES + (_grid(int(a[8]), int(a[9]), int(a[10]), INDEX_LEAF) if a[7] else n * INDEX_LEAF)


def _rebuild(a: tuple) -> int:      # (volume id, points, n, ...): the grid is already allocated
    return int(a[2]) * (KEY_BYTES + LEAVES_PER_POINT * LEAF_PREFIX)


def _bvh_bytes(n: int, constructor: int, rebuild: bool = False) -> int:
    known = BVH_SCRATCH.get(constructor, 0) * n + (0 if rebuild else BVH_KEPT * n)
    return (CUBQL_FACTOR * known if constructor == -1 else known) + BUILD_MARGIN


def _mesh_create(a: tuple) -> int:  # (ctx, points, velocities, indices, points n, tris n, winding, constructor, ...)
    n = int(a[5])
    return _bvh_bytes(n, int(a[7])) + n * (MESH_KEPT + (MESH_WINDING if a[6] else 0))


def _items(a: tuple) -> int:
    return SIZES.get(int(a[0]), (0, 0))[0]


def _rebuild_bvh(a: tuple) -> int:  # (id,): the create's items and constructor; 0 for an id made before install
    n, constructor = SIZES.get(int(a[0]), (0, 0))
    return _bvh_bytes(n, constructor, rebuild=True) if n else 0


_P, _I, _U32, _U64, _B, _V = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32, ctypes.c_uint64, ctypes.c_bool, None
try:
    from warp._src.types import array_t as _ARR
except ImportError:  # pragma: no cover - another Warp layout: the audit then refuses the mesh wraps
    _ARR = None
_BUILD = "a BVH / mesh build (cloth, mesh collision, an MPM step's colliders)"
_REDUCE = "a reduction over a device array"
_XF = (ctypes.c_float * 9, ctypes.c_float * 3)
_SORT = (_U64, _U64, _I, _I, _I, _I)
GUARDS: dict[str, Guard] = {
    "wp_volume_from_active_voxels_device": Guard(
        "a NanoVDB grid build", 2, lambda a: _voxels(a) + VOLUME_MARGIN,
        (_P, _P, _I, _P, *_XF, _B, _B, _U32, _U32, _U32, _U32, _P), True),
    "wp_volume_from_tiles_device": Guard(
        "a NanoVDB grid build", 2, lambda a: _tiles(a) + VOLUME_MARGIN,
        (_P, _P, _I, _P, *_XF, _B, _P, _U32, ctypes.c_char_p, _B, _U32, _U32, _U32, _P), True),
    "wp_volume_index_from_tiles_device": Guard(
        "a NanoVDB grid build", 2, lambda a: _index_tiles(a) + VOLUME_MARGIN,
        (_P, _P, _I, _P, *_XF, _B, _B, _U32, _U32, _U32, _P), True),
    "wp_volume_rebuild_from_active_voxels_device": Guard(
        "a NanoVDB grid rebuild", 2, lambda a: _rebuild(a) + VOLUME_MARGIN, (_U64, _P, _I, _P, *_XF, _B, _P), False),
    "wp_volume_rebuild_from_tiles_device": Guard(
        "a NanoVDB grid rebuild", 2, lambda a: _rebuild(a) + VOLUME_MARGIN,
        (_U64, _P, _I, _P, *_XF, _B, _P, _U32, ctypes.c_char_p, _P), False),
    "wp_volume_index_rebuild_from_tiles_device": Guard(
        "a NanoVDB grid rebuild", 2, lambda a: _rebuild(a) + VOLUME_MARGIN, (_U64, _P, _I, _P, *_XF, _B, _P), False),
    "wp_runlength_encode_int_device": Guard(
        "a run-length encode", 4, lambda a: int(a[4]) * 8 + SMALL_MARGIN, (_U64, _U64, _U64, _U64, _I), False),
    **{f"wp_radix_sort_pairs_{k}_device": Guard("a radix sort", 2, lambda a: int(a[2]) * 8 + SMALL_MARGIN, _SORT, False)
       for k in ("int", "uint", "float", "double", "int64", "uint64")},
    **{f"wp_array_scan_{k}_device": Guard("a prefix sum", 2, lambda a: int(a[2]) // 16 + SMALL_MARGIN,
                                           (_U64, _U64, _I, _I, _I, _I, _B), False) for k in ("int", "float")},
    **{f"wp_array_sum_{k}_device": Guard("an array sum", 2, lambda a: SMALL_MARGIN, (_U64, _U64, _I, _I, _I), False,
                                          _REDUCE) for k in ("float", "double")},
    **{f"wp_array_inner_{k}_device": Guard("an inner product", 3, lambda a: SMALL_MARGIN,
                                            (_U64, _U64, _U64, _I, _I, _I, _I), False, _REDUCE)
       for k in ("float", "double")},
    "wp_bvh_create_device": Guard("a BVH build", 3, lambda a: _bvh_bytes(int(a[3]), int(a[4])),
                                   (_P, _P, _P, _I, _I, _P, _I), True, _BUILD, records=(3, 4)),
    "wp_bvh_rebuild_device": Guard("a BVH rebuild", 0, _rebuild_bvh, (_U64,), False, _BUILD, by_id=True),
    "wp_mesh_create_device": Guard("a mesh build", 5, _mesh_create, (_P, _ARR, _ARR, _ARR, _I, _I, _I, _I, _P, _I),
                                    True, _BUILD, records=(5, 7)),
    "wp_mesh_refit_device": Guard("a mesh refit", 0, lambda a: 8 * _items(a) + SMALL_MARGIN, (_U64,), False, _BUILD,
                                   by_id=True),
    "wp_mesh_set_points_device": Guard("a mesh refit", 0, lambda a: 8 * _items(a) + SMALL_MARGIN, (_U64, _ARR), False,
                                        _BUILD, by_id=True),
}


def scratch_bytes(native: str, args: tuple) -> int:
    """The scratch bound ``native`` is reserved for when called with ``args``."""
    return GUARDS[native].need(args)


def probe(device: Any, need: int) -> bool:
    """True when ``need`` bytes can be allocated on ``device``: the driver's free memory covers it, or (the pool's
    unused reserve does not show there) Warp's allocator gives it, freed at once. At 256 eager worlds this runs ~2,500
    times per period: the free-memory read first (WP4) kept the guard's cost and the pool's high-water mark flat."""
    if _STALE and str(device) in _STALE:
        absorb_stale_error(device)
    if int(need) <= int(device.free_memory):
        return True
    alloc = device.get_allocator()
    try:
        ptr = alloc.allocate(int(need))
    except RuntimeError:
        _STALE.add(str(device))                         # the failed allocation leaves CUB's stale error (module doc)
        return False
    alloc.deallocate(ptr, int(need))
    return True


def absorb_stale_error(device: Any) -> bool:
    """Take the failure the next CUB native would get after a CUDA allocation failure on ``device`` (module
    docstring): tiny prefix sums until one runs clean, at most 3. False when the device has no room even for that;
    it then stays marked and the next guarded native tries again. Nothing to do on the CPU (True)."""
    device = wp.get_device(device)
    if not device.is_cuda:
        return True
    _STALE.discard(str(device))                         # first: the prefix sum below is a guarded native itself
    try:
        a = wp.zeros(2, dtype=wp.int32, device=device)
        for _ in range(3):
            try:
                wp.utils.array_scan(a, a, inclusive=True)
                return True
            except RuntimeError:
                log.debug("native guard: a prefix sum took the stale CUDA error on %s", device)
    except RuntimeError:                                # no room for 8 bytes: still stale
        pass
    _STALE.add(str(device))
    return False


def _current() -> Any:
    """The device whose context Warp's native code allocates on (``WP_CURRENT_CONTEXT``, sort.cu:413): the current
    CUDA device, not Warp's default device (its sort and encode call sites enter no ``ScopedDevice``)."""
    from warp._src.context import runtime  # noqa: PLC0415

    return runtime.get_current_cuda_device()


def _device(args: tuple, context_arg: bool) -> Any:
    """A build's device from its context argument; else the current CUDA device. A rebuild runs on its volume's own
    context (volume.cpp:1399), the current device on one GPU; on several, the probe can check another device."""
    if context_arg:
        ctx = getattr(args[0], "value", args[0])
        for d in wp.get_cuda_devices():
            if getattr(d.context, "value", d.context) == ctx:
                return d
    return _current()


def _guarded(name: str, native: Any, g: Guard) -> Any:
    def call(*args: Any) -> Any:
        device = _device(args, g.context_arg)
        if device.is_cuda and not device.is_capturing:
            need = g.need(args)
            if not probe(device, need):
                count = _items(args) if g.by_id else int(args[g.count_arg])
                raise NativeMemoryError(
                    f"out of GPU memory on {device}: {g.what} of {count:,} elements ({name}, {g.note}) needs up to "
                    f"{need / 2**30:.2f} GB of scratch, and the device has no room for it. Warp's native code would go "
                    f"on without it and fail with CUDA error 700. Fix: {FIX}",
                    native=name, count=count, need=need)
        out = native(*args)
        if g.records is not None and out:
            SIZES[int(out)] = (int(args[g.records[0]]), int(args[g.records[1]]))
        return out

    call.alloy_native_guard = True                      # type: ignore[attr-defined]
    call.native = native                                # type: ignore[attr-defined]
    return call


def _audited(native: Any, g: Guard) -> bool:
    """The Warp version read for this wrap, and the argument types it relies on."""
    types = tuple(getattr(native, "argtypes", None) or ())
    return wp.__version__.startswith(AUDITED) and len(types) == len(g.argtypes) and all(
        t is e or (getattr(t, "_type_", None), getattr(t, "_length_", None)) ==
        (getattr(e, "_type_", None), getattr(e, "_length_", None)) for t, e in zip(types, g.argtypes, strict=True))


def install() -> list[str]:
    """Wrap each audited native once per process; the names this call wrapped."""
    wp.init()
    from warp._src.context import runtime  # noqa: PLC0415

    core, done = runtime.core, []
    for name, g in GUARDS.items():
        native = getattr(core, name, None)
        if native is None or getattr(native, "alloy_native_guard", False):
            continue
        if not _audited(native, g):
            log.warning("native guard not installed on %s: Warp %s's signature is not the audited one (Warp %sx); "
                        "an out-of-memory in it can still end in CUDA error 700", name, wp.__version__, AUDITED)
            continue
        setattr(core, name, _guarded(name, native, g))
        done.append(name)
    if done:
        log.debug("native guard: %s reserve their scratch first (CUDA out of memory -> NativeMemoryError)", done)
    return done
