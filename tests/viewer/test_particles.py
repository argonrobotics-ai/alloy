# SPDX-License-Identifier: Apache-2.0
"""Per-world particle placement for the viewer: shown worlds follow the viewer's world offsets, hidden worlds drop out
of view, and the simulation's own particle array is never written."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.viewer import _particles as P


class _Model:
    """The fields ParticlePlacer reads from a Newton model: 2 worlds of 3 particles and one particle of no world."""

    def __init__(self, wp, worlds: int = 2):
        self.device = "cpu"
        self.world_count = worlds
        self.up_axis = 2
        world = np.array([0, 0, 0, 1, 1, 1, -1], np.int32)
        self.particle_count = world.size
        self.particle_world = wp.array(world, dtype=wp.int32, device="cpu")
        self.tri_indices = wp.array(np.array([[0, 1, 2], [3, 4, 5]], np.int32), dtype=wp.int32, device="cpu")
        self.tet_indices = None


class _State:
    def __init__(self, wp, q: np.ndarray):
        self.particle_q = wp.array(q, dtype=wp.vec3, device="cpu")
        self.body_q = "bodies"


class _Backend:
    def __init__(self, wp, offsets: np.ndarray | None):
        self.world_offsets = None if offsets is None else wp.array(offsets, dtype=wp.vec3, device="cpu")


@pytest.fixture(scope="module")
def wp():
    import warp

    return warp


def test_world_shifts_shown_and_hidden():
    offsets = np.array([[1.0, 0, 0], [0, 2.0, 0], [3.0, 3.0, 0]], np.float32)
    shift = P.world_shifts(3, offsets, shown=[0, 2], up_axis=2)
    assert np.array_equal(shift[0], [1, 0, 0]) and np.array_equal(shift[2], [3, 3, 0])
    assert np.array_equal(shift[1], [0, 0, -P.HIDDEN_DROP_M])
    assert np.array_equal(P.world_shifts(2, None, shown=[0, 1], up_axis=2), np.zeros((2, 3), np.float32))


def test_placed_state_follows_offsets_and_leaves_the_sim_alone(wp):
    q = np.arange(21, dtype=np.float32).reshape(7, 3)
    model, state = _Model(wp), _State(wp, q)
    backend = _Backend(wp, np.array([[-0.5, 0, 0], [0.5, 0, 0]], np.float32))
    placer = P.ParticlePlacer(model)
    view = placer.state(backend, state, shown=[0, 1])
    out = view.particle_q.numpy()
    assert np.array_equal(out[:3], q[:3] + np.float32([-0.5, 0, 0]))
    assert np.array_equal(out[3:6], q[3:6] + np.float32([0.5, 0, 0]))
    assert np.array_equal(out[6], q[6])                       # a particle of no world stays put
    assert view.body_q == "bodies"                            # everything else reads through
    assert np.array_equal(state.particle_q.numpy(), q)        # the sim's array is untouched
    hidden = placer.state(backend, state, shown=[1]).particle_q.numpy()
    assert np.all(hidden[:3, 2] < -0.5 * P.HIDDEN_DROP_M) and np.array_equal(hidden[3:6], out[3:6])


def test_one_world_or_no_particles_pass_the_state_through(wp):
    state = _State(wp, np.zeros((7, 3), np.float32))
    one = _Model(wp, worlds=1)
    assert P.ParticlePlacer(one).state(_Backend(wp, None), state, shown=[0]) is state


def test_loose_particles_detected(wp):
    model = _Model(wp)
    assert P.has_loose_particles(model)                       # particle 6 is in no triangle
    model.tri_indices = wp.array(np.array([[0, 1, 2], [3, 4, 5], [4, 5, 6]], np.int32), dtype=wp.int32, device="cpu")
    assert not P.has_loose_particles(model)
