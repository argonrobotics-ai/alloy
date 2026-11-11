# SPDX-License-Identifier: Apache-2.0
"""Warp's native triplet build runs out of GPU memory without raising, and the next kernels hit CUDA error 700
(Nov 8 pass, e97d31). ``alloy.integration.sparse_guard`` reserves the build's scratch first and raises
SparseMemoryError instead. These are CPU tests with a stand-in native function and device; the L4 check
(soil_dig at 64 worlds: SparseMemoryError, and the context still works) is in docs/known_gaps/mpm-gpu-memory.md."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from alloy.core.errors import CapacityError
from alloy.integration import native_guard as NG
from alloy.integration import sparse_guard as G

BIG = (1 << 20) + 1


def _args(nnz: int) -> tuple:
    return (1, 4, 10, 10, nnz, None, "rows", "cols", None, 0, False, None, None, "offsets", None, "columns")


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


def _device(monkeypatch, *, cuda=True, capturing=False, room=1 << 40):
    alloc = _Alloc(room)
    dev = SimpleNamespace(is_cuda=cuda, is_capturing=capturing, get_allocator=lambda: alloc,
                          free_memory=0, __str__=lambda self: "cuda:0")      # the pool decides
    monkeypatch.setattr(G.wp, "get_device", lambda *a: dev)
    return alloc


def test_a_full_gpu_raises_before_the_native_build(monkeypatch, native) -> None:
    fn, calls = native
    _device(monkeypatch, room=0)
    with pytest.raises(G.SparseMemoryError) as e:
        G._guarded(fn)(*_args(157_286_400))
    assert not calls                                            # Warp's native code never ran
    assert isinstance(e.value, CapacityError) and e.value.nnz == 157_286_400
    assert e.value.need == G.scratch_bytes(157_286_400) and "fewer worlds" in e.value.fix
    assert "157,286,400 triplets" in str(e.value) and "CUDA error 700" in str(e.value)


def test_with_room_the_scratch_is_reserved_freed_then_the_native_build_runs(monkeypatch, native) -> None:
    fn, calls = native
    alloc = _device(monkeypatch)
    assert G._guarded(fn)(*_args(BIG)) == 7
    need = BIG * G.BYTES_PER_TRIPLET + G.MARGIN
    assert calls == [_args(BIG)] and alloc.asked == [need] and alloc.freed == [(1234, need)]


def test_small_builds_are_reserved_too(monkeypatch, native) -> None:
    """SCR-WP12-3 follow-up 1: no path to error 700, so a full GPU fails a small build cleanly as well."""
    fn, calls = native
    _device(monkeypatch, room=0)
    with pytest.raises(G.SparseMemoryError):
        G._guarded(fn)(*_args(1000))
    assert not calls


@pytest.mark.parametrize("case", ["cpu", "capturing"])
def test_no_reservation_on_the_cpu_or_inside_a_capture(monkeypatch, native, case) -> None:
    fn, calls = native
    alloc = _device(monkeypatch, cuda=case != "cpu", capturing=case == "capturing")
    G._guarded(fn)(*_args(BIG))
    assert calls == [_args(BIG)] and alloc.asked == []


def _fake_native(argtypes):
    fn = lambda *a: None  # noqa: E731
    fn.argtypes = argtypes
    return fn


def test_install_only_on_the_audited_signature(monkeypatch) -> None:
    """WP4's review: nnz is argument 5 on Warp 1.18; another order would read a pointer as nnz."""
    import ctypes

    from warp._src.context import runtime

    wp.init()
    good = [ctypes.c_int] * 5 + [ctypes.POINTER(ctypes.c_int)]
    for argtypes, version, ok in ((good, "1.18.0", True), (good, "1.19.0", False),
                                  ([ctypes.c_int] * 4 + [ctypes.c_void_p] * 2, "1.18.0", False)):
        monkeypatch.setattr(runtime, "core", SimpleNamespace(**{G.NATIVE: _fake_native(argtypes)}))
        monkeypatch.setattr(G.wp, "__version__", version)
        assert G.install() is ok, (version, argtypes)
        assert bool(getattr(getattr(runtime.core, G.NATIVE), "alloy_sparse_guard", False)) is ok


def test_install_wraps_once_and_warp_sparse_still_builds_on_the_cpu(monkeypatch) -> None:
    import ctypes

    from warp._src.context import runtime

    wp.init()
    good = [ctypes.c_int] * 5 + [ctypes.POINTER(ctypes.c_int)]
    monkeypatch.setattr(runtime, "core", SimpleNamespace(**{G.NATIVE: _fake_native(good)}))
    assert G.install() and not G.install()                     # idempotent
    assert getattr(runtime.core, G.NATIVE).alloy_sparse_guard
    monkeypatch.undo()
    G.install()                                                 # the real library: CPU builds take the host path
    import warp.sparse as ws

    m = ws.bsr_from_triplets(3, 3, wp.array([0, 1, 1], dtype=int, device="cpu"),
                             wp.array([0, 1, 1], dtype=int, device="cpu"),
                             wp.array([1.0, 2.0, 3.0], dtype=float, device="cpu"))
    y = ws.bsr_mv(m, wp.array(np.ones(3), dtype=float, device="cpu"))
    assert np.allclose(y.numpy(), [1.0, 5.0, 0.0])                    # duplicates summed
