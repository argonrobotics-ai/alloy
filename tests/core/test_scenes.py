# SPDX-License-Identifier: Apache-2.0
"""Reference scenes step on CPU (SPEC §7 WP0 acceptance: box_on_plane(16), box_on_cloth(1, 2.0))."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.testing import scenes as S


def test_box_on_plane_16_rests() -> None:
    sc = S.box_on_plane(16)
    assert sc.world_count == 16 and sc.model.body_count == 16
    r = sc.runner("xpbd")
    r.step(200)
    z = r.state.body_q.numpy()[:, 2]
    assert np.all(np.abs(z - 0.1) < 2e-3), z
    assert len({row.tobytes() for row in r.state.body_q.numpy()}) == 1   # identical worlds stay identical


@pytest.mark.compile_heavy
def test_box_on_plane_mujoco() -> None:
    r = S.box_on_plane(4).runner("mujoco")
    r.step(50)
    assert np.all(np.abs(r.state.body_q.numpy()[:, 2] - 0.1) < 2e-3)


@pytest.mark.compile_heavy
def test_box_on_cloth_steps_coupled() -> None:
    sc = S.box_on_cloth(1, 2.0)
    assert sc.default_solver == "coupled" and sc.model.particle_count == len(sc.particles)
    r = sc.runner()
    z0 = r.state.body_q.numpy()[0, 2]
    r.step(40)
    q, x = r.state.body_q.numpy(), r.state.particle_q.numpy()
    assert np.all(np.isfinite(q)) and np.all(np.isfinite(x))
    assert q[0, 2] < z0                       # the box falls onto the cloth
    assert x[:, 2].min() > 0.3                # the pinned cloth holds


def test_pendulum_period() -> None:
    sc = S.pendulum(2, 0.1)
    r = sc.runner("featherstone")
    q = []
    for _ in range(1200):
        r.step(1)
        q.append(r.state.joint_q.numpy()[0])
    q = np.array(q)
    crossings = np.flatnonzero(np.diff(np.sign(q)) != 0)   # every zero crossing = half a period
    period = 2.0 * (crossings[1] - crossings[0]) * sc.dt
    length = sc.params["length"]
    radius = sc.params["radius"]
    inertia = 0.4 * radius**2 + length**2                  # sphere about the pivot, per unit mass
    expected = 2 * math.pi * math.sqrt(inertia / (9.81 * length))
    assert period == pytest.approx(expected, rel=0.02)


@pytest.mark.parametrize(("mu", "sticks"), [(0.8, True), (0.1, False)])
def test_incline_coulomb(mu: float, sticks: bool) -> None:
    sc = S.incline(1, mu, 0.3)
    r = sc.runner("xpbd")
    x0 = r.state.body_q.numpy()[0, :3].copy()
    r.step(200)
    moved = float(np.linalg.norm(r.state.body_q.numpy()[0, :3] - x0))
    assert (moved < 5e-3) if sticks else (moved > 1e-2), moved


def test_peg_in_ring_geometry() -> None:
    sc = S.peg_in_ring(2, 2e-4)
    assert sc.params["inner_radius"] == pytest.approx(0.0101)
    r = sc.runner("xpbd")
    r.step(100)
    assert np.all(np.isfinite(r.state.body_q.numpy()))
    with pytest.raises(ValueError):
        S.peg_in_ring(1, 0.0)
