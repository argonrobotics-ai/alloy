# SPDX-License-Identifier: Apache-2.0
"""Frames (SPEC §7 WP5 acceptance: replicate with spacing raises under worlds_at_origin; F4 rebasing)."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.core.errors import PrecisionError
from alloy.precision.frames import REBASE_GRID_M, check_worlds_at_origin, plan_rebase, rebase_worlds
from alloy.precision.types import FramePolicy
from alloy.testing.scenes import box_on_plane, pendulum


def _boxes(n: int, *, spacing=(0.0, 0.0, 0.0), jitter: float = 0.0, offset=(0.0, 0.0, 0.0), ground_per_world=False,
           device: str = "cpu", seed: int = 0) -> newton.Model:
    template = newton.ModelBuilder()
    for k in range(2):
        b = template.add_body(xform=wp.transform(wp.vec3(0.3 * k + offset[0], offset[1], 0.1 + offset[2]),
                                                 wp.quat_identity()))
        template.add_shape_box(b, hx=0.05, hy=0.05, hz=0.05)
    if ground_per_world:
        template.add_ground_plane()
    root = newton.ModelBuilder()
    root.replicate(template, n, spacing=spacing)
    model = root.finalize(device=device)
    if jitter:
        q = model.body_q.numpy()
        q[:, :3] += np.random.default_rng(seed).uniform(-jitter, jitter, (q.shape[0], 3))
        model.body_q.assign(q)
    return model


def test_zero_spacing_and_randomized_worlds_pass(device: str) -> None:
    check_worlds_at_origin(box_on_plane(8, device=device).model, FramePolicy())
    check_worlds_at_origin(pendulum(4, 0.2, device=device).model, FramePolicy())
    check_worlds_at_origin(_boxes(6, jitter=0.02, device=device), FramePolicy())


@pytest.mark.parametrize("spacing", [(2.0, 0.0, 0.0), (0.0, 0.05, 0.0), (1.0, 1.0, 0.0)])
def test_replicate_with_spacing_raises(device: str, spacing) -> None:
    model = _boxes(4, spacing=spacing, device=device)
    with pytest.raises(PrecisionError, match="spaced"):
        check_worlds_at_origin(model, FramePolicy())
    check_worlds_at_origin(model, FramePolicy(worlds_at_origin=False))  # opt-out


def test_single_body_world_spacing(device: str) -> None:
    template = newton.ModelBuilder()
    b = template.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.2), wp.quat_identity()))
    template.add_shape_sphere(b, radius=0.05)
    root = newton.ModelBuilder()
    root.replicate(template, 3, spacing=(3.0, 0.0, 0.0))
    with pytest.raises(PrecisionError):
        check_worlds_at_origin(root.finalize(device=device), FramePolicy())


def test_world_far_from_origin_raises(device: str) -> None:
    model = _boxes(1, offset=(40.0, 0.0, 0.0), device=device)
    with pytest.raises(PrecisionError, match="from the origin"):
        check_worlds_at_origin(model, FramePolicy())


def test_rebase_moves_only_selected_worlds(device: str) -> None:
    model = _boxes(4, ground_per_world=True, device=device)
    state = model.state()
    before_q = state.body_q.numpy().copy()
    before_shapes = model.shape_transform.numpy().copy()
    origin = wp.zeros(4, dtype=wp.vec3d, device=device)
    shift = np.array([[1.0, -0.5, 0.25]])
    rebase_worlds(model, state, wp.array([2], dtype=wp.int32, device=device),
                  wp.array(shift, dtype=wp.vec3d, device=device), origin)
    after = state.body_q.numpy()
    world = model.body_world.numpy()
    assert np.array_equal(after[world != 2], before_q[world != 2])
    np.testing.assert_allclose(after[world == 2, :3], before_q[world == 2, :3] - shift, atol=1e-7)
    assert np.array_equal(after[world == 2, 3:], before_q[world == 2, 3:])
    sw, sb = model.shape_world.numpy(), model.shape_body.numpy()
    moved = (sw == 2) & (sb < 0)
    assert moved.any()
    np.testing.assert_allclose(model.shape_transform.numpy()[moved, :3], before_shapes[moved, :3] - shift, atol=1e-7)
    assert np.array_equal(model.shape_transform.numpy()[~moved], before_shapes[~moved])
    np.testing.assert_array_equal(origin.numpy()[2], shift[0])
    assert not origin.numpy()[[0, 1, 3]].any()


def test_rebase_free_joint_coordinates(device: str) -> None:
    model = _boxes(2, ground_per_world=True, device=device)
    state = model.state()
    q0 = state.joint_q.numpy().copy()
    rebase_worlds(model, state, wp.array([1], dtype=wp.int32, device=device),
                  wp.array([[0.5, 0.0, 0.0]], dtype=wp.vec3d, device=device), wp.zeros(2, dtype=wp.vec3d, device=device))
    q1 = state.joint_q.numpy()
    starts, jw = model.joint_q_start.numpy(), model.joint_world.numpy()
    for j in range(model.joint_count):
        s = starts[j]
        expect = q0[s] - (0.5 if jw[j] == 1 else 0.0)
        assert q1[s] == pytest.approx(expect, abs=1e-7)


def test_rebase_refuses_shared_statics(device: str) -> None:
    scene = box_on_plane(2, device=device)  # ground plane in world -1
    with pytest.raises(PrecisionError, match="per-world statics"):
        rebase_worlds(scene.model, scene.model.state(), wp.array([0], dtype=wp.int32, device=device),
                      wp.zeros(1, dtype=wp.vec3d, device=device), wp.zeros(2, dtype=wp.vec3d, device=device))


def test_plan_rebase_snaps_to_grid(device: str) -> None:
    model = _boxes(3, ground_per_world=True, device=device)
    state = model.state()
    q = state.body_q.numpy()
    q[model.body_world.numpy() == 1, 0] += 3.0
    state.body_q.assign(q)
    plan = plan_rebase(model, state, FramePolicy(rebase="auto", rebase_radius_m=2.0))
    assert plan is not None
    ids, shifts = plan
    assert ids.tolist() == [1]
    assert np.allclose(shifts / REBASE_GRID_M, np.round(shifts / REBASE_GRID_M))
    assert plan_rebase(model, state, FramePolicy(rebase="off")) is None
