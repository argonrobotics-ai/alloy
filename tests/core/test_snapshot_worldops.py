# SPDX-License-Identifier: Apache-2.0
"""WorldSnapshot masked restore (bitwise, incl. extra arrays) and per-world reductions (SPEC §3.1)."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.core.errors import SnapshotError
from alloy.core.snapshot import WorldSnapshot, mask_from_worlds
from alloy.core.worldops import joint_coord_world, joint_dof_world, per_world_count, per_world_max, per_world_min
from alloy.testing.scenes import box_on_cloth, box_on_plane, pendulum


def _perturb(state, rng: np.random.Generator) -> None:
    for name in ("body_q", "body_qd", "joint_q", "joint_qd", "particle_q", "particle_qd"):
        arr = getattr(state, name, None)
        if arr is not None:
            data = arr.numpy()
            arr.assign(data + rng.standard_normal(data.shape).astype(data.dtype))


def _host(state) -> dict[str, np.ndarray]:
    return {k: v.numpy().copy() for k, v in vars(state).items() if isinstance(v, wp.array)}


def test_masked_restore_is_bitwise_and_only_masked_worlds(rng: np.random.Generator) -> None:
    scene = box_on_plane(4)
    model, state = scene.model, scene.model.state()
    extra = {"warm": wp.array(rng.standard_normal((4, 3)).astype(np.float32), dtype=wp.float32),
             "tiny": wp.array(np.arange(4, dtype=np.int8), dtype=wp.int8)}
    snap = WorldSnapshot(model, state, extra)
    _perturb(state, rng)
    q = state.body_q.numpy()
    q[0, 0] = np.nan                                      # NaN payloads survive bitwise
    q[1, 1] = -0.0
    state.body_q.assign(q)
    snap.save(state)
    saved, saved_extra = _host(state), {k: v.numpy().copy() for k, v in extra.items()}
    _perturb(state, rng)
    for arr in extra.values():
        arr.assign(np.full(arr.shape, 7, dtype=arr.numpy().dtype))
    after = _host(state)
    snap.restore(state, mask_from_worlds([1, 3], 4, model.device))
    body_world = model.body_world.numpy()
    for name, now in _host(state).items():
        world = {"body_q": body_world, "body_qd": body_world, "body_f": body_world,
                 "joint_q": joint_coord_world(model).numpy(), "joint_qd": joint_dof_world(model).numpy()}[name]
        sel = np.isin(world, [1, 3])
        assert now[sel].tobytes() == saved[name][sel].tobytes(), name
        assert now[~sel].tobytes() == after[name][~sel].tobytes(), name
    warm = extra["warm"].numpy()
    assert warm[[1, 3]].tobytes() == saved_extra["warm"][[1, 3]].tobytes()
    assert np.all(warm[[0, 2]] == 7)
    assert list(extra["tiny"].numpy()) == [7, 1, 7, 3]


def test_full_restore_without_mask() -> None:
    scene = pendulum(2, 0.3)
    state = scene.model.state()
    snap = WorldSnapshot(scene.model, state)
    snap.save(state)
    before = _host(state)
    _perturb(state, np.random.default_rng(1))
    snap.restore(state)
    for name, arr in _host(state).items():
        assert arr.tobytes() == before[name].tobytes()


def test_whole_array_blocks_per_world_restore() -> None:
    scene = box_on_plane(2)
    state = scene.model.state()
    snap = WorldSnapshot(scene.model, state, {"global": wp.zeros(5, dtype=wp.float32)})
    assert snap.whole_arrays == ("extra:global",)
    snap.save(state)
    snap.restore(state)                                   # whole restore works
    with pytest.raises(SnapshotError):
        snap.restore(state, mask_from_worlds([0], 2, scene.model.device))


def test_particles_are_per_world() -> None:
    scene = box_on_cloth(2, 1.0)
    state = scene.model.state()
    snap = WorldSnapshot(scene.model, state)
    snap.save(state)
    before = state.particle_q.numpy().copy()
    state.particle_q.assign(before + 1.0)
    snap.restore(state, mask_from_worlds([0], 2, scene.model.device))
    pw = scene.model.particle_world.numpy()
    now = state.particle_q.numpy()
    assert np.array_equal(now[pw == 0], before[pw == 0]) and np.array_equal(now[pw == 1], before[pw == 1] + 1.0)


def test_mask_validation() -> None:
    scene = box_on_plane(3)
    snap = WorldSnapshot(scene.model, scene.model.state())
    with pytest.raises(SnapshotError):
        snap.restore(scene.model.state(), wp.zeros(2, dtype=wp.int32))
    with pytest.raises(SnapshotError):
        snap.restore(scene.model.state(), wp.zeros(3, dtype=wp.float32))


def test_per_world_reductions_match_numpy(rng: np.random.Generator) -> None:
    n, worlds = 1000, 7
    item_world = rng.integers(-1, worlds, n).astype(np.int32)
    values = rng.standard_normal(n).astype(np.float32)
    values[5] = np.nan
    mask = rng.integers(0, 2, n).astype(np.int32)
    out_max, out_min = wp.zeros(worlds, dtype=wp.float32), wp.zeros(worlds, dtype=wp.float32)
    out_cnt = wp.zeros(worlds, dtype=wp.int32)
    wv, ww, wm = wp.array(values), wp.array(item_world), wp.array(mask)
    per_world_max(wv, ww, out_max)
    per_world_min(wv, ww, out_min)
    per_world_count(wm, ww, out_cnt)
    for w in range(worlds):
        sel = (item_world == w) & ~np.isnan(values)
        assert out_max.numpy()[w] == values[sel].max()
        assert out_min.numpy()[w] == values[sel].min()
        assert out_cnt.numpy()[w] == int(mask[item_world == w].sum())


def test_reductions_reset_out_and_handle_empty_worlds() -> None:
    out = wp.full(3, 99.0, dtype=wp.float32)
    per_world_max(wp.array([1.0], dtype=wp.float32), wp.array([0], dtype=wp.int32), out)
    assert out.numpy()[0] == 1.0 and np.all(np.isneginf(out.numpy()[1:]))
    with pytest.raises(ValueError):
        per_world_max(wp.zeros(2, dtype=wp.float32), wp.zeros(3, dtype=wp.int32), out)
