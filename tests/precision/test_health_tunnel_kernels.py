# SPDX-License-Identifier: Apache-2.0
"""L5: kernel unit tests of the tunnel check (``health_tunnel_kernels``), on the CPU with hand-made positions.

``static_sdf`` against closed forms per primitive; ``TunnelWatch.after_step`` on a built model whose bodies are moved
by hand between two ``body_q`` arrays: a ball through a 2 mm plate in one step, the exit-side rule, the exclusion
list, reset(mask)."""

from __future__ import annotations

import numpy as np  # noqa: F401
import pytest
import warp as wp

import alloy
from alloy.precision import health_tunnel as HT
from alloy.precision import health_tunnel_kernels as K
from alloy.precision import health_tunnel_setup as S


@wp.kernel
def _probe(kind: wp.array[wp.int32], half: wp.array[wp.vec3], p: wp.array[wp.vec3], out: wp.array[wp.vec4]):
    i = wp.tid()
    out[i] = K.static_sdf(kind[i], half[i], p[i])


def _sdf(kind: int, half, p) -> np.ndarray:
    out = wp.zeros(1, dtype=wp.vec4, device="cpu")
    wp.launch(_probe, dim=1, inputs=[wp.array([kind], dtype=wp.int32, device="cpu"),
                                     wp.array([half], dtype=wp.vec3, device="cpu"),
                                     wp.array([p], dtype=wp.vec3, device="cpu"), out], device="cpu")
    return out.numpy()[0]


@pytest.mark.parametrize("kind,half,p,dist,normal", [
    (K.T_PLANE, (0, 0, 0), (3, 4, 0.5), 0.5, (0, 0, 1)),
    (K.T_PLANE, (0, 0, 0), (3, 4, -0.2), -0.2, (0, 0, 1)),
    (K.T_SPHERE, (0.5, 0, 0), (2, 0, 0), 1.5, (1, 0, 0)),
    (K.T_SPHERE, (0.5, 0, 0), (0.25, 0, 0), -0.25, (1, 0, 0)),
    (K.T_BOX, (1, 2, 3), (4, 0, 0), 3.0, (1, 0, 0)),
    (K.T_BOX, (1, 2, 3), (-1, -2, 4), 1.0, (0, 0, 1)),
    (K.T_BOX, (1, 2, 3), (-4, 6, 3), 5.0, (-0.6, 0.8, 0)),
    (K.T_BOX, (1, 2, 3), (0.9, 0, 0), -0.1, (1, 0, 0)),
    (K.T_CAPSULE, (0.5, 1, 0), (2, 0, 0.5), 1.5, (1, 0, 0)),
    (K.T_CAPSULE, (0.5, 1, 0), (0, 0, 3.0), 1.5, (0, 0, 1)),
    (K.T_CYLINDER, (0.5, 1, 0), (2, 0, 0.5), 1.5, (1, 0, 0)),
    (K.T_CYLINDER, (0.5, 1, 0), (0, 0, -2.0), 1.0, (0, 0, -1)),
    (K.T_CYLINDER, (0.5, 1, 0), (0.4, 0, 0.0), -0.1, (1, 0, 0)),
])
def test_static_sdf_matches_the_closed_form(kind, half, p, dist, normal) -> None:
    got = _sdf(kind, half, p)
    assert got[0] == pytest.approx(dist, abs=1e-5)
    assert got[1:] == pytest.approx(normal, abs=1e-5)


@pytest.fixture(scope="module")
def world():
    """A built scene: a 1 mm static plate, a static bar, a ball and a box; the watch of its model, fresh."""
    scene = alloy.Scene(worlds=2, dt=0.01)
    scene.add_ground()
    scene.add_box("plate", size=(1.0, 1.0, 0.002), pos=(0, 0, 0.3), static=True)
    scene.add_sphere("ball", 0.01, pos=(0, 0, 1.0), density=1000.0)
    sim = alloy.build(scene, device="cpu")
    return sim.engine.model


def _watch(model):
    return HT.TunnelWatch(model, S.judged_bodies(model), S.static_shapes(model))


def _move(model, tw, z0: float, z1: float, x1: float = 0.0, world: int = 0) -> None:
    """One step of the ball of ``world`` from (0, 0, z0) to (x1, 0, z1); the other world holds still."""
    names = list(model.body_label)
    ball = [i for i, n in enumerate(names) if n.endswith("ball") and int(model.body_world.numpy()[i]) == world][0]
    s0, s1 = model.state(), model.state()
    q = s0.body_q.numpy().copy()
    q0, q1 = q.copy(), q.copy()
    q0[ball, :3], q1[ball, :3] = (0, 0, z0), (x1, 0, z1)
    s0.body_q.assign(q0)
    s1.body_q.assign(q1)
    tw.after_step(s0, s1, 0.00125)


def _state(tw) -> tuple[list[int], list[int]]:
    return tw.fast.numpy().tolist(), tw.tunnel.numpy().tolist()


def test_a_ball_crossing_a_1_mm_plate_in_one_step_is_tunnel_in_its_world_only(world) -> None:
    tw = _watch(world)
    _move(world, tw, 0.32, 0.28)                                    # 4 cm in one step: both ends outside the plate
    fast, tunnel = _state(tw)
    assert tunnel == [1, 0] and fast == [0, 0]                      # world 1's ball never moved; 0.28 is far from the plate
    assert tw.notes(0.00125)[0]["tunnel"] == ("ball", "plate")


def test_a_ball_that_enters_the_plate_and_leaves_by_the_same_side_is_not_tunnel(world) -> None:
    tw = _watch(world)
    _move(world, tw, 0.32, 0.3003)                                  # centre inside the 2 mm plate (top 0.301)
    assert _state(tw)[1] == [0, 0] and tw.tr.in_shape.numpy().max() >= 0
    _move(world, tw, 0.3003, 0.33)                                  # and back up
    assert _state(tw)[1] == [0, 0] and tw.tr.in_shape.numpy().max() == -1


def test_a_ball_that_enters_and_leaves_by_the_far_side_is_tunnel(world) -> None:
    tw = _watch(world)
    _move(world, tw, 0.3015, 0.3003)
    assert _state(tw)[1] == [0, 0]
    _move(world, tw, 0.3003, 0.2985)
    assert _state(tw)[1] == [1, 0]


def test_a_body_wholly_under_the_ground_plane_is_tunnel_and_one_resting_on_it_is_not(world) -> None:
    tw = _watch(world)
    _move(world, tw, 0.05, 0.0)                                     # centre on the plane: inside by 0, r_out 1 cm
    assert _state(tw)[1] == [0, 0]
    _move(world, tw, 0.0, -0.011)                                   # the centre enters the plane's solid
    assert _state(tw)[1] == [0, 0]
    _move(world, tw, -0.011, -0.012)                                # deeper than the ball's radius: wholly under
    assert _state(tw)[1][0] == 1


def test_a_sliding_ball_closing_by_less_than_the_gap_carries_no_fast_contact(world) -> None:
    tw = _watch(world)
    _move(world, tw, 0.32, 0.317, x1=0.5)                           # 3 mm closer, 0.5 m sideways
    assert _state(tw) == ([0, 0], [0, 0])
    _move(world, tw, 0.320, 0.312, x1=0.0)                          # 8 mm closer to the plate, ball surface at 2 mm
    assert _state(tw)[0][0] == K.HOLD_STEPS
    assert tw.notes(0.00125)[0]["fast"][0] == pytest.approx(0.008, abs=1e-4)


def test_a_filtered_pair_is_skipped_and_reset_forgets_the_flags(world) -> None:
    tw = _watch(world)
    plate = [i for i, s in enumerate(tw.statics) if "plate" in str(world.shape_label[int(s)])][0]
    tw.bt.ex_idx = wp.array([plate], dtype=wp.int32, device="cpu")                 # the ball may pass the plate
    assert len(tw.bodies) == 2
    tw.bt.ex_start = wp.array([0, 1, 1], dtype=wp.int32, device="cpu")
    _move(world, tw, 0.32, 0.28)
    assert _state(tw)[1] == [0, 0]
    tw2 = _watch(world)
    _move(world, tw2, 0.32, 0.28)
    _move(world, tw2, 0.32, 0.28, world=1)
    assert _state(tw2)[1] == [1, 1]
    tw2.reset(wp.array([0, 1], dtype=wp.int32, device="cpu"))
    assert _state(tw2) == ([0, 0], [1, 0])                          # only world 1 forgot
    tw2.reset(None)
    assert _state(tw2) == ([0, 0], [0, 0])


@wp.kernel
def _passable(kind: wp.array[wp.int32], half: wp.array[wp.vec3], travel: float, out: wp.array[wp.int32]):
    i = wp.tid()
    out[i] = int(K.can_be_passed(kind[i], half[i], wp.transform_identity(), wp.vec3(0.0, 0.0, 1.0), travel))


@pytest.mark.parametrize("kind,half,travel,expected", [
    (K.T_PLANE, (0, 0, 0), 1.0, 0),                       # a half-space cannot be passed, at any travel
    (K.T_BOX, (0.25, 0.25, 0.005), 0.0111, 1),            # 1 cm plate, 11.1 mm per step: 2 x travel > 10 mm
    (K.T_BOX, (0.25, 0.25, 0.005), 0.0078, 1),            # 15.7 mm > 10 mm
    (K.T_BOX, (0.25, 0.25, 0.005), 0.004, 0),             # 8 mm < 10 mm
    (K.T_BOX, (0.3, 0.3, 0.2), 0.0096, 0),                # 40 cm table
    (K.T_SPHERE, (0.05, 0, 0), 0.06, 1),                  # 10 cm ball, 12 cm travel
    (K.T_SPHERE, (0.05, 0, 0), 0.04, 0),
    (K.T_CAPSULE, (0.01, 0.5, 0), 0.03, 0),               # along z the capsule is 1.02 m thick
    (K.T_CYLINDER, (0.5, 0.005, 0), 0.0111, 1),           # a thin disc seen along its axis
])
def test_can_be_passed_follows_the_thickness_along_the_approach(kind, half, travel, expected) -> None:
    out = wp.zeros(1, dtype=wp.int32, device="cpu")
    wp.launch(_passable, dim=1, inputs=[wp.array([kind], dtype=wp.int32, device="cpu"),
                                        wp.array([half], dtype=wp.vec3, device="cpu"), travel, out], device="cpu")
    assert int(out.numpy()[0]) == expected
