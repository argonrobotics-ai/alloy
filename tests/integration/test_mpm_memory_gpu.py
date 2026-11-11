# SPDX-License-Identifier: Apache-2.0
"""Laws (L4, ``-m gpu``): an MPM scene runs in the mode that fits, and only a scene that fits in neither is refused,
never with CUDA error 700 (Nov 8 pass, e97d31; SCR-WP12-3).

- 256 worlds of dough under a press do not fit as CUDA graphs (~135 GB predicted) but do eagerly (10.4 GB measured):
  they build, run without graphs, and step with a clean health check;
- a scene that fits as graphs stays on the graph path, bit for bit as with the check switched off;
- a scene that fits in neither mode is refused at build, before the first step: MPMMemoryError (a CapacityError)
  with the predicted and free bytes, the worlds that fit, and the fix; the CUDA context still works afterwards;
- with the check off (``ALLOY_MPM_MEMORY_CHECK=off``), soil_dig at 64 worlds as graphs on a GPU under 40 GB fails
  its first step with a Python exception, not error 700, and the context still works (``sparse_guard``, the backstop);
- with the check off and a ballast leaving 2.5 GB, the same scene fails with a Python exception too (before
  ``native_guard`` its NanoVDB grid rebuild ran out and ended in error 700: SCR-WP12-3 follow-up 1);
- a NanoVDB grid build and an in-place rebuild that have no room raise NativeMemoryError before Warp's native code
  runs, and the context still works.
"Fits in neither" is made real on any GPU: a ballast array holds all but a set amount of the free memory first.
"""

from __future__ import annotations

import contextlib
import dataclasses
import gc

import numpy as np
import pytest
import warp as wp

pytestmark = [pytest.mark.gpu, pytest.mark.integration]

ROOM = int(2.5 * 2**30)             # free memory left beside the ballast: soil_dig at 64 worlds needs ~3 GB eager
GB = 2**30
DOUGH = 17_256                      # MPM particles per dough_ball world
MODEL_256 = int(2.7 * GB)           # 256 dough worlds' model on the device (an L4: 21.4 GB free before it, 18.7 after)


def _dig(worlds: int):
    from alloy.tasks import soil_dig as D

    return D.build(dataclasses.replace(D.DigSpec(), worlds=worlds, device="cuda:0"))


def _dough(worlds: int):
    import alloy

    s = alloy.Scene(worlds=worlds, dt=0.01)
    s.add_ground()
    s.add_object("dough", "dough_ball", pos=(0, 0, 0.040))                 # r 3.9 cm, resting on the ground
    s.add_box("press", size=(0.12, 0.12, 0.01), pos=(0, 0, 0.079), kinematic=True, material="steel_carbon")
    return s


def _press(sim, periods: int) -> None:
    for _ in range(periods):
        sim.drive("press", velocity=(0, 0, -0.05), duration=sim.dt)
        sim.step(1)


@contextlib.contextmanager
def _ballast(room: int = ROOM):
    """Hold all but ``room`` bytes of the device's free memory."""
    from alloy.integration.mpm_memory import free_bytes

    dev = wp.get_device("cuda:0")
    rows = max((free_bytes(dev) - room) >> 20, 0)
    held = wp.empty((rows, 1 << 20), dtype=wp.uint8, device=dev) if rows else None
    try:
        yield
    finally:
        del held
        gc.collect()


def _context_alive() -> None:
    gc.collect()
    x = wp.zeros(16, dtype=float, device="cuda:0")
    wp.launch(_add_one, dim=16, inputs=[x], device="cuda:0")
    assert np.allclose(x.numpy(), 1.0)


@wp.kernel
def _add_one(x: wp.array(dtype=float)):
    x[wp.tid()] += 1.0


def _room_for_256_dough() -> None:
    """Skip when, after a full collection, this process holds so much of the GPU that 256 dough worlds cannot fit even
    eagerly: the law is about an otherwise idle L4, and the memory check then refuses them by design."""
    from alloy.integration import mpm_memory as M

    gc.collect()
    dev = wp.get_device("cuda:0")
    free = M.free_bytes(dev)
    need = M.predict(256 * DOUGH, graph=False) / M.HEADROOM + MODEL_256
    if free < need:
        pytest.skip(f"{free / GB:.1f} of {dev.total_memory / GB:.1f} GB free after earlier GPU work in this process; "
                    f"256 dough worlds need about {need / GB:.1f} GB eager: run this file in its own process")


@pytest.fixture(autouse=True)
def _cuda():
    if not wp.is_cuda_available():
        pytest.skip("needs CUDA")


@pytest.mark.timeout(1200)
def test_256_dough_worlds_build_and_step_without_graphs() -> None:
    import alloy

    dev = wp.get_device("cuda:0")
    if dev.total_memory < 20 * 2**30 or dev.total_memory > 140 * 2**30:
        pytest.skip("the law is for a GPU on which 256 dough worlds fit only eagerly (an L4: 23 GB)")
    _room_for_256_dough()
    sim = alloy.build(_dough(256), device="cuda:0")
    assert any(n.startswith("graphs off:") for n in sim.engine.parts.notes)
    _press(sim, 3)
    assert sim.engine.graph_stats()["mode"] == "eager" and sim.check()["ok"]
    q = sim.engine.state.particle_q.numpy()
    assert np.isfinite(q).all()
    del sim
    _context_alive()


@pytest.mark.timeout(900)
def test_a_scene_that_fits_as_graphs_stays_bitwise_on_the_graph_path(monkeypatch) -> None:
    import alloy

    runs = []
    for check in ("on", "off"):
        monkeypatch.setenv("ALLOY_MPM_MEMORY_CHECK", check)
        sim = alloy.build(_dough(4), device="cuda:0", seed=0)
        _press(sim, 4)
        stats = sim.engine.graph_stats()
        assert stats["mode"] == "graph" and stats["launches"] > 0, stats
        assert not any(n.startswith("graphs off:") for n in sim.engine.parts.notes)
        runs.append(sim.engine.state.particle_q.numpy().copy())
        del sim
        gc.collect()
    assert np.array_equal(runs[0], runs[1])


@pytest.mark.timeout(900)
def test_a_scene_that_fits_in_neither_mode_is_refused_before_the_first_step() -> None:
    import alloy

    with _ballast(), pytest.raises(alloy.BuildError) as e:
        _dig(64)
    text = e.value.message
    assert "MPMMemoryError" in text and "even without CUDA graphs" in text and "GB is free" in text
    assert "world(s) of this scene fit" in text and "fewer worlds" in (e.value.fix or "")
    _context_alive()


@pytest.mark.timeout(900)
def test_without_the_check_the_first_step_fails_cleanly_not_700(monkeypatch) -> None:
    """soil_dig at 64 worlds as graphs (~30 GB) on a GPU under 40 GB: the first step runs out inside Warp's sparse
    build, which ``sparse_guard`` turns into an error (or, with less free, the build runs out allocating a snapshot).
    """
    if wp.get_device("cuda:0").total_memory >= 40 * 2**30:
        pytest.skip("this GPU holds 64 dig worlds as graphs")
    monkeypatch.setenv("ALLOY_MPM_MEMORY_CHECK", "off")
    monkeypatch.setenv("ALLOY_MPM_CELLS", "particles")      # one cell per particle: the ~30 GB graph of the Nov 8 pass
    from alloy.integration.sparse_guard import SparseMemoryError

    sim = None
    with pytest.raises(Exception) as e:                       # noqa: PT011 - the guard's error, or Warp's own OOM
        sim = _dig(64)                                        # after other tests in this process: may fail here
        sim.step(2)
    chain = [e.value, e.value.__cause__, e.value.__context__]
    assert any(isinstance(x, SparseMemoryError) or "Failed to allocate" in str(x) for x in chain), repr(e.value)
    assert "illegal memory access" not in str(e.value)        # Warp's CUDA 700 text
    e.value.__traceback__ = None                             # its frames hold the 64-world scene's arrays
    del sim, e
    _context_alive()


@pytest.mark.timeout(900)
def test_without_the_check_a_nearly_full_gpu_fails_cleanly_not_700(monkeypatch) -> None:
    """The case that ended in error 700 inside the NanoVDB rebuild (``rebuild_count_points``) before ``native_guard``:
    soil_dig at 64 worlds, the check off, a ballast leaving 2.5 GB."""
    monkeypatch.setenv("ALLOY_MPM_MEMORY_CHECK", "off")
    sim = None
    with _ballast(), pytest.raises(Exception) as e:          # noqa: PT011 - a guard's error, or Warp's own OOM
        sim = _dig(64)
        sim.step(2)
    _no_700(e)
    del sim, e
    _context_alive()


def _no_700(e) -> None:
    from alloy.integration.native_guard import NativeMemoryError
    from alloy.integration.sparse_guard import SparseMemoryError

    chain = [e.value, e.value.__cause__, e.value.__context__]
    ok = (NativeMemoryError, SparseMemoryError)
    assert any(isinstance(x, ok) or "Failed to allocate" in str(x) for x in chain), repr(e.value)
    assert "illegal memory access" not in str(e.value)        # Warp's CUDA 700 text
    e.value.__traceback__ = None                             # its frames hold the scene's arrays


def _points(n: int):
    rng = np.random.default_rng(0)
    return wp.array(rng.integers(0, 4096, size=(n, 3)).astype(np.int32), dtype=wp.vec3i, device="cuda:0")


@pytest.mark.timeout(600)
@pytest.mark.parametrize("kind", ["build", "rebuild"])
def test_a_grid_build_with_no_room_raises_before_warps_native_code(kind) -> None:
    """Law for the grid-rebuild OOM: 4 M scattered voxels need ~0.9 GB of builder scratch; 64 MB is left."""
    from alloy.integration import native_guard

    native_guard.install()
    big = _points(4_000_000)
    vol = wp.Volume.allocate_by_voxels(_points(1000), voxel_size=0.005, device="cuda:0", max_active_voxels=10**6,
                                       max_leaf_nodes=10**5, max_lower_nodes=64, max_upper_nodes=8) \
        if kind == "rebuild" else None                       # ~14 MB grid (Warp's default node capacities: ~TB)
    with _ballast(64 << 20), pytest.raises(native_guard.NativeMemoryError) as e:
        if vol is None:
            wp.Volume.allocate_by_voxels(big, voxel_size=0.005, device="cuda:0")
        else:
            vol.rebuild(big)
    assert e.value.count == 4_000_000 and ("rebuild" in str(e.value)) == (kind == "rebuild")
    e.value.__traceback__ = None
    del big, vol, e
    _context_alive()


@pytest.mark.timeout(900)
def test_a_small_scene_builds_and_steps_on_graphs() -> None:
    sim = _dig(4)
    sim.step(2)
    assert sim.check()["ok"] and sim.engine.graph_stats()["mode"] == "graph"
    _context_alive()
