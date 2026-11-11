# SPDX-License-Identifier: Apache-2.0
"""Warp's BVH and mesh builders (bvh.cu, mesh.cu, Warp 1.18) write through unchecked device allocations when the GPU
is full, and the next kernels hit CUDA error 700. ``alloy.integration.native_guard`` reserves each build's scratch and
the tree it keeps first, and bounds a rebuild or refit by the size its create recorded (GAPS-NOW #5a memory guards).
CPU tests with a stand-in native and device, as in tests/integration/test_native_guard.py."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from alloy.integration import native_guard as G
from alloy.integration import native_guard as NG

N = 300_000
BVH, MESH = "wp_bvh_create_device", "wp_mesh_create_device"


def _bvh_args(n: int = N, constructor: int = 2) -> tuple:     # (ctx, lowers, uppers, n, constructor, groups, leaf)
    return ("ctx", "lowers", "uppers", n, constructor, None, 4)


def _mesh_args(n: int = N, winding: int = 0, constructor: int = 2) -> tuple:
    return ("ctx", "points", "velocities", "indices", n, n, winding, constructor, None, 4)


class _Alloc:
    def __init__(self, room: int) -> None:
        self.room, self.asked = room, []

    def allocate(self, size: int) -> int:
        self.asked.append(size)
        if size > self.room:
            raise RuntimeError(f"Failed to allocate {size} bytes on device 'cuda:0'")
        return 99

    def deallocate(self, ptr: int, size: int) -> None:
        pass


def _device(monkeypatch, room: int):
    alloc = _Alloc(room)
    dev = SimpleNamespace(is_cuda=True, is_capturing=False, context="ctx", get_allocator=lambda: alloc, free_memory=0)
    monkeypatch.setattr(G, "_current", lambda: dev)
    monkeypatch.setattr(G.wp, "get_cuda_devices", lambda: [dev])
    return alloc


@pytest.fixture(autouse=True)
def _stale(monkeypatch):
    """The stand-in devices here are not Warp devices: a failed reservation marks them stale, and the absorb before
    the next native is recorded instead of run (its own tests: test_native_guard.py)."""
    absorbed: list[str] = []
    monkeypatch.setattr(NG, "_STALE", set())
    monkeypatch.setattr(NG, "absorb_stale_error", lambda d: (absorbed.append(str(d)), NG._STALE.discard(str(d)))[1])
    return absorbed


@pytest.fixture(autouse=True)
def _sizes(monkeypatch):
    monkeypatch.setattr(G, "SIZES", {})


def _native(result: int = 7):
    calls: list[tuple] = []
    return (lambda *a: calls.append(a) or result), calls


@pytest.mark.parametrize("name, args", [(BVH, _bvh_args()), (MESH, _mesh_args())])
def test_a_full_gpu_refuses_a_bvh_or_mesh_build_before_the_native_call(monkeypatch, name, args) -> None:
    fn, calls = _native()
    _device(monkeypatch, room=0)
    with pytest.raises(G.NativeMemoryError) as e:
        G._guarded(name, fn, G.GUARDS[name])(*args)
    assert not calls and e.value.count == N and e.value.need == G.scratch_bytes(name, args)
    assert f"{N:,} elements" in str(e.value) and "BVH / mesh build" in str(e.value)


def test_a_rebuild_or_refit_is_bounded_by_the_size_its_create_recorded(monkeypatch) -> None:
    fn, _ = _native(result=41)
    _device(monkeypatch, room=1 << 40)
    assert G._guarded(BVH, fn, G.GUARDS[BVH])(*_bvh_args()) == 41
    assert G.SIZES[41] == (N, 2)
    _device(monkeypatch, room=0)
    for name in ("wp_bvh_rebuild_device", "wp_mesh_refit_device"):
        rebuild, calls = _native()
        with pytest.raises(G.NativeMemoryError) as e:
            G._guarded(name, rebuild, G.GUARDS[name])(41)
        assert not calls and e.value.count == N
    rebuild, calls = _native()
    G._guarded("wp_bvh_rebuild_device", rebuild, G.GUARDS["wp_bvh_rebuild_device"])(12345)   # made before install
    assert calls == [(12345,)]


def test_the_bounds_cover_what_the_builders_allocate() -> None:
    """LBVH: 52n scratch + 84n kept; a mesh adds 24n (+288n with winding numbers); cuBQL doubles the known terms (its
    builder's own scratch is not in Warp's tree); a rebuild reuses the tree (scratch only)."""
    m = G.BUILD_MARGIN
    assert G.scratch_bytes(BVH, _bvh_args()) == 136 * N + m
    assert G.scratch_bytes(BVH, _bvh_args(constructor=-1)) == 2 * 176 * N + m
    assert G.scratch_bytes(BVH, _bvh_args(constructor=0)) == 84 * N + m              # SAH: built on the host
    assert G.scratch_bytes(MESH, _mesh_args()) == 136 * N + m + 24 * N
    assert G.scratch_bytes(MESH, _mesh_args(winding=1)) == 136 * N + m + 312 * N
    G.SIZES[5] = (N, 2)
    assert G.scratch_bytes("wp_bvh_rebuild_device", (5,)) == 52 * N + m


def test_warp_still_builds_bvhs_and_meshes_on_the_cpu_once_installed() -> None:
    G.install()
    pts = wp.array(np.random.default_rng(0).random((64, 3)), dtype=wp.vec3, device="cpu")
    bvh = wp.Bvh(pts, pts)
    mesh = wp.Mesh(points=pts, indices=wp.array(np.arange(60, dtype=np.int32), dtype=int, device="cpu"))
    mesh.refit()
    assert bvh.id and mesh.id
