# SPDX-License-Identifier: Apache-2.0
"""World digests and the run digest (SPEC §3.7; research precision §4.6): equal worlds give equal digests, a 1-ulp
change moves only its world's digest (R8 negative control), a world's digest does not depend on the world count
(R4), extras are covered, and the run digest binds the config and the environment."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.precision.config import preset
from alloy.precision.digest import env_tuple, run_digest, world_digests
from alloy.precision.types import GpuTier
from alloy.testing.scenes import box_on_plane, pendulum


def _nudge(arr: wp.array, index: tuple[int, ...]) -> None:
    host = arr.numpy()
    flat = host.reshape(host.shape[0], -1)  # a view: edits land in host
    row, col = index
    flat[row, col] = np.nextafter(flat[row, col], np.float32(np.inf))
    arr.assign(wp.array(host, dtype=arr.dtype, device=arr.device))


def test_identical_worlds_equal_and_one_ulp_moves_one_world(device: str) -> None:
    scene = box_on_plane(8, device=device)
    state = scene.model.state()
    d0 = world_digests(scene.model, state)
    assert d0.shape == (8, 2) and d0.dtype == np.uint64
    assert (d0 == d0[0]).all()
    _nudge(state.body_q, (5, 2))  # world 5's box z by one ulp
    d1 = world_digests(scene.model, state)
    changed = np.flatnonzero((d1 != d0).any(axis=1))
    assert changed.tolist() == [5]
    assert (d1[5] != d0[5]).all()  # both lanes move


def test_joint_coordinates_are_digested(device: str) -> None:
    scene = pendulum(3, 0.3, device=device)
    state = scene.model.state()
    d0 = world_digests(scene.model, state)
    _nudge(state.joint_q, (1, 0))
    d1 = world_digests(scene.model, state)
    assert np.flatnonzero((d1 != d0).any(axis=1)).tolist() == [1]


def test_world_digest_independent_of_world_count(device: str) -> None:
    small, big = box_on_plane(1, device=device), box_on_plane(16, device=device)
    d_small = world_digests(small.model, small.model.state())
    d_big = world_digests(big.model, big.model.state())
    assert (d_big == d_small[0]).all()


def test_extras_per_world_and_whole(device: str) -> None:
    scene = box_on_plane(4, device=device)
    state = scene.model.state()
    per_world = wp.zeros((4, 3), dtype=wp.float32, device=device)
    whole = wp.zeros(7, dtype=wp.float64, device=device)
    base = world_digests(scene.model, state, {"warm": per_world, "global": whole})
    host = per_world.numpy()
    host[2, 1] = 1.0
    per_world.assign(host)
    d = world_digests(scene.model, state, {"warm": per_world, "global": whole})
    assert np.flatnonzero((d != base).any(axis=1)).tolist() == [2]
    whole.assign(np.arange(7, dtype=np.float64))
    d2 = world_digests(scene.model, state, {"warm": per_world, "global": whole})
    assert ((d2 != d).any(axis=1)).all()  # a whole array reaches every world


def test_run_digest_binds_config_and_env(device: str) -> None:
    scene = box_on_plane(2, device=device)
    wd = world_digests(scene.model, scene.model.state())
    env = env_tuple(device, 2)
    std, fast = preset("standard", GpuTier.CPU), preset("fast", GpuTier.CPU)
    a = run_digest(wd, std, env)
    assert a == run_digest(wd.copy(), std, dict(env))
    assert a != run_digest(wd, fast, env)
    assert a != run_digest(wd, std, {**env, "world_count": "3"})
    assert len(a) == 64 and set(env) >= {"device", "warp", "newton", "alloy", "world_count"}


def test_world_digest_is_stable_across_calls(device: str) -> None:
    scene = box_on_plane(3, device=device)
    state = scene.model.state()
    assert np.array_equal(world_digests(scene.model, state), world_digests(scene.model, state))


@pytest.mark.parametrize("dtype", [wp.int8, wp.float16, wp.bool])
def test_odd_sized_extras(device: str, dtype) -> None:
    scene = box_on_plane(2, device=device)
    arr = wp.zeros((2, 5), dtype=dtype, device=device)
    d = world_digests(scene.model, scene.model.state(), {"odd": arr})
    assert d.shape == (2, 2)


def test_extra_world_maps(device: str) -> None:
    """An extra array with an item->world map is hashed per item world (like WorldSnapshot extra_world)."""
    scene = box_on_plane(3, device=device)
    state = scene.model.state()
    items = wp.zeros(6, dtype=wp.float32, device=device)  # two items per world, world-major
    item_world = wp.array(np.array([0, 0, 1, 1, 2, 2], np.int32), device=device)
    maps = {"items": item_world}
    base = world_digests(scene.model, state, {"items": items}, extra_world=maps)
    assert (base == base[0]).all()
    items.assign(np.array([0, 0, 0, 1.0, 0, 0], np.float32))
    d = world_digests(scene.model, state, {"items": items}, extra_world=maps)
    assert np.flatnonzero((d != base).any(axis=1)).tolist() == [1]


def test_whole_extras_can_be_left_out(device: str) -> None:
    """R4 compares across world counts, so arrays without a per-world axis are left out there."""
    one, four = box_on_plane(1, device=device), box_on_plane(4, device=device)
    d1 = world_digests(one.model, one.model.state(), {"impulse": wp.zeros(5, dtype=wp.float32, device=device)},
                       whole_extras=False)
    d4 = world_digests(four.model, four.model.state(), {"impulse": wp.zeros(17, dtype=wp.float32, device=device)},
                       whole_extras=False)
    assert (d4 == d1[0]).all()
