# SPDX-License-Identifier: Apache-2.0
"""Warp's native grid builds, sorts and run-length encodes go on with null scratch when the GPU is full, and the next
kernels hit CUDA error 700 (an L4 run with ~2.5 GB free: ``rebuild_count_points``). ``alloy.integration.native_guard``
reserves each call's scratch first and raises NativeMemoryError instead (SCR-WP12-3 follow-up 1). CPU tests with a
stand-in native function and device; the L4 laws are in tests/integration/test_native_guard_gpu.py."""

from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from alloy.core.errors import CapacityError
from alloy.integration import native_guard as G
from alloy.integration import native_guard as NG

SCAN = "wp_array_scan_int_device"

VOXELS = "wp_volume_from_active_voxels_device"
REBUILD = "wp_volume_rebuild_from_active_voxels_device"


def _voxel_args(n: int, rebuildable=False, leaf=0, lower=0, upper=0) -> tuple:
    return ("ctx", "points", n, None, "xform", "translation", False, rebuildable, n, leaf, lower, upper, None)


@pytest.fixture(autouse=True)
def _stale(monkeypatch):
    """The stand-in devices here are not Warp devices: a failed reservation marks them stale, and the absorb before
    the next native is recorded instead of run (its own tests: test_native_guard.py)."""
    absorbed: list[str] = []
    monkeypatch.setattr(NG, "_STALE", set())
    monkeypatch.setattr(NG, "absorb_stale_error", lambda d: (absorbed.append(str(d)), NG._STALE.discard(str(d)))[1])
    return absorbed


@pytest.fixture()
def native():
    calls: list[tuple] = []
    fn = lambda *a: calls.append(a) or 7  # noqa: E731
    return fn, calls


class _Alloc:
    """Warp's allocator stand-in: records each request, fails above ``room`` bytes, and frees what it gave."""

    def __init__(self, room: int) -> None:
        self.room, self.asked, self.freed = room, [], []

    def allocate(self, size: int) -> int:
        self.asked.append(size)
        if size > self.room:
            raise RuntimeError(f"Failed to allocate {size} bytes on device 'cuda:0'")
        return 1234

    def deallocate(self, ptr: int, size: int) -> None:
        self.freed.append((ptr, size))


def _device(monkeypatch, *, cuda=True, capturing=False, room=1 << 40, context="ctx"):
    alloc = _Alloc(room)
    dev = SimpleNamespace(is_cuda=cuda, is_capturing=capturing, context=context, get_allocator=lambda: alloc,
                          free_memory=0, __str__=lambda self: "cuda:0")      # the pool decides
    monkeypatch.setattr(G, "_current", lambda: dev)
    monkeypatch.setattr(G.wp, "get_cuda_devices", lambda: [dev])
    return dev, alloc


@pytest.mark.parametrize("name", [VOXELS, REBUILD, "wp_runlength_encode_int_device", "wp_radix_sort_pairs_int_device"])
def test_a_full_gpu_raises_before_the_native_call(monkeypatch, native, name) -> None:
    fn, calls = native
    _device(monkeypatch, room=0)
    g = G.GUARDS[name]
    args = list(_voxel_args(200_000))
    if name != VOXELS:
        args[g.count_arg] = 200_000
    with pytest.raises(G.NativeMemoryError) as e:
        G._guarded(name, fn, g)(*args)
    assert not calls                                            # Warp's native code never ran
    err = e.value
    assert isinstance(err, CapacityError) and err.native == name and err.count == 200_000
    assert err.need == g.need(tuple(args)) and "fewer worlds" in err.fix
    assert "200,000 elements" in str(err) and "CUDA error 700" in str(err)


def test_with_room_the_scratch_is_reserved_freed_then_the_native_runs(monkeypatch, native) -> None:
    fn, calls = native
    _, alloc = _device(monkeypatch)
    args = _voxel_args(100_000)
    assert G._guarded(VOXELS, fn, G.GUARDS[VOXELS])(*args) == 7
    need = G.scratch_bytes(VOXELS, args)
    assert calls == [args] and alloc.asked == [need] and alloc.freed == [(1234, need)]


def test_the_bounds_cover_what_volume_builder_allocates() -> None:
    """volume_builder.cu (Warp 1.18): 7 x 8 B keys + 4 B run counts per point, CUB's sort scratch, 12 B per leaf; an
    exact grid adds its leaves, a rebuildable one the grid at its capacities."""
    n = 100_000
    exact = G.scratch_bytes(VOXELS, _voxel_args(n))
    assert exact >= n * (60 + 8 + 12 + 96) + G.VOLUME_MARGIN
    reserved = G.scratch_bytes(VOXELS, _voxel_args(n, True, leaf=n, lower=40, upper=2))
    assert reserved >= n * (60 + 8 + 12 + 96) + 40 * 33_000 + 2 * 270_000
    assert G.scratch_bytes(REBUILD, (1, "p", n, None, "x", "t", False, None)) >= n * (60 + 8) + G.VOLUME_MARGIN


@pytest.mark.parametrize("case", ["cpu", "capturing"])
def test_no_reservation_on_the_cpu_or_inside_a_capture(monkeypatch, native, case) -> None:
    fn, calls = native
    _, alloc = _device(monkeypatch, cuda=case != "cpu", capturing=case == "capturing")
    G._guarded(VOXELS, fn, G.GUARDS[VOXELS])(*_voxel_args(10))
    assert calls == [_voxel_args(10)] and alloc.asked == []


def test_a_build_reserves_on_the_device_its_context_names(monkeypatch, native) -> None:
    fn, _ = native
    current, cur_alloc = _device(monkeypatch, context=ctypes.c_void_p(1))
    other_alloc = _Alloc(1 << 40)
    other = SimpleNamespace(is_cuda=True, is_capturing=False, context=ctypes.c_void_p(99),
                            get_allocator=lambda: other_alloc, free_memory=0)
    monkeypatch.setattr(G.wp, "get_cuda_devices", lambda: [current, other])
    G._guarded(VOXELS, fn, G.GUARDS[VOXELS])(ctypes.c_void_p(99), *_voxel_args(10)[1:])
    assert len(other_alloc.asked) == 1 and cur_alloc.asked == []


def test_every_guard_matches_warps_native_signature() -> None:
    """The bounds read arguments by position (the count, a rebuild's capacities): only the audited signature."""
    from warp._src.context import runtime

    wp.init()
    for name, g in G.GUARDS.items():
        native = getattr(runtime.core, name)
        assert G._audited(getattr(native, "native", native), g), name


def test_install_skips_a_changed_signature(monkeypatch) -> None:
    from warp._src.context import runtime

    wp.init()
    good = [getattr(runtime.core, n) for n in G.GUARDS]
    fakes = {n: SimpleNamespace(argtypes=list(getattr(f, "native", f).argtypes)) for n, f in zip(G.GUARDS, good)}
    fakes[VOXELS].argtypes = fakes[VOXELS].argtypes[:-1]                      # one argument fewer
    monkeypatch.setattr(runtime, "core", SimpleNamespace(**fakes))
    done = G.install()
    assert VOXELS not in done and REBUILD in done and len(done) == len(G.GUARDS) - 1
    assert G.install() == []                                                  # idempotent
    monkeypatch.setattr(G.wp, "__version__", "1.19.0")
    monkeypatch.setattr(runtime, "core", SimpleNamespace(**{n: SimpleNamespace(argtypes=f.argtypes)
                                                            for n, f in fakes.items()}))
    assert G.install() == []                                                  # another Warp: nothing wrapped


def test_warp_still_sorts_encodes_and_builds_on_the_cpu_once_installed() -> None:
    G.install()
    keys = wp.array([3, 1, 2, 0, 0, 0, 0, 0], dtype=int, device="cpu")
    vals = wp.array([30, 10, 20, 0, 0, 0, 0, 0], dtype=int, device="cpu")
    wp.utils.radix_sort_pairs(keys, vals, 4)
    assert keys.numpy()[:4].tolist() == [0, 1, 2, 3] and vals.numpy()[:4].tolist() == [0, 10, 20, 30]
    pts = wp.array(np.array([[0, 0, 0], [1, 0, 0], [40, 0, 0]], np.int32), dtype=wp.vec3i, device="cpu")
    vol = wp.Volume.allocate_by_voxels(pts, voxel_size=0.01, device="cpu")
    assert vol.get_voxel_count() == 3


def test_enough_driver_free_memory_skips_the_allocation(monkeypatch, native) -> None:
    """WP4's lever: with room in the driver's free memory, nothing is allocated (the probe ran ~2,500 times per period
    at 256 eager worlds)."""
    fn, calls = native
    dev, alloc = _device(monkeypatch)
    dev.free_memory = 1 << 40
    G._guarded(VOXELS, fn, G.GUARDS[VOXELS])(*_voxel_args(100_000))
    assert calls and alloc.asked == []


def test_a_failed_reservation_marks_the_device_and_the_next_native_absorbs_first(monkeypatch, native, _stale) -> None:
    """After a CUDA allocation failure the next CUB native fails once with error 101 (module docstring): the guard
    marks the device when its own reservation fails and absorbs before the next guarded native (SCR-OSS22-9)."""
    fn, calls = native
    dev, alloc = _device(monkeypatch, room=0)
    with pytest.raises(G.NativeMemoryError):
        G._guarded(SCAN, fn, G.GUARDS[SCAN])(1, 2, 4096, 4, 4, 1, False)
    assert G._STALE == {str(dev)} and not calls and not _stale
    alloc.room = 1 << 40
    assert G._guarded(SCAN, fn, G.GUARDS[SCAN])(1, 2, 4096, 4, 4, 1, False) == 7
    assert _stale == [str(dev)] and not G._STALE and len(calls) == 1
    G._guarded(SCAN, fn, G.GUARDS[SCAN])(1, 2, 4096, 4, 4, 1, False)
    assert _stale == [str(dev)]                                    # absorbed once


class _Cuda:
    is_cuda = True

    def __str__(self) -> str:
        return "cuda:0"


@pytest.mark.parametrize("fails, clean", [(0, True), (1, True), (2, True), (3, False)])
def test_absorb_runs_tiny_prefix_sums_until_one_is_clean(monkeypatch, fails, clean) -> None:
    """The first prefix sum after an allocation failure raises error 101 and the second is clean (measured on a T4);
    the absorb tries 3 and leaves the device marked when none is clean."""
    monkeypatch.undo()                                             # the real absorb_stale_error
    monkeypatch.setattr(G, "_STALE", {"cuda:0"})
    seen: list[tuple] = []
    monkeypatch.setattr(G.wp, "get_device", lambda d: _Cuda())
    monkeypatch.setattr(G.wp, "zeros", lambda n, dtype, device: "two ints")

    def scan(a, b, inclusive):
        seen.append((a, b, inclusive, set(G._STALE)))
        if len(seen) <= fails:
            raise RuntimeError("Warp CUDA error 101: invalid device ordinal (in function scan_device)")

    monkeypatch.setattr(G.wp.utils, "array_scan", scan)
    assert G.absorb_stale_error("cuda:0") is clean
    assert len(seen) == min(fails + 1, 3) and seen[0] == ("two ints", "two ints", True, set())   # unmarked first
    assert G._STALE == (set() if clean else {"cuda:0"})


def test_absorb_without_room_leaves_the_device_marked_and_asks_nothing_of_the_cpu(monkeypatch) -> None:
    monkeypatch.undo()
    monkeypatch.setattr(G, "_STALE", set())

    def zeros(n, dtype, device):
        raise RuntimeError("Failed to allocate 8 bytes on device 'cuda:0'")

    monkeypatch.setattr(G.wp, "get_device", lambda d: _Cuda() if d == "cuda:0" else SimpleNamespace(is_cuda=False))
    monkeypatch.setattr(G.wp, "zeros", zeros)
    assert G.absorb_stale_error("cpu") is True and not G._STALE
    assert G.absorb_stale_error("cuda:0") is False and G._STALE == {"cuda:0"}
