# SPDX-License-Identifier: Apache-2.0
"""SCR-OSS3-2 (coordinator 2026-11-06, ruling (a)): scene-built SolverDFSPH gets a default escape box
(``escape_box.auto_world_bounds``) so ejecta retire, counted per world, never silently."""

from __future__ import annotations

from alloy._vendor import newton
import numpy as np
import pytest
import warp as wp
from alloy._vendor.newton import ParticleFlags

from alloy.solvers.sph import escape_box as eb
from alloy.solvers.sph import plugin
from alloy.solvers.sph.build import add_fluid_block, add_fluid_points
from alloy.solvers.sph.config import DFSPHConfig, register_custom_attributes

R = 2.0e-3


def _scene(shot: bool):
    """Ground plane, a 10 cm static plate at z 0.1 m, a 4^3 block resting on it, optionally one particle shot up
    at 8 m/s (ballistic top 3.3 m)."""
    cfg = DFSPHConfig(particle_radius=R)
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    register_custom_attributes(b)
    b.add_ground_plane()
    b.begin_world()
    b.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.095), wp.quat_identity()), hx=0.05, hy=0.05, hz=0.005)
    add_fluid_block(b, cfg, (-0.008, -0.008, 0.1), (4, 4, 4))
    if shot:
        add_fluid_points(b, cfg, np.array([[0.03, 0.0, 0.11]]), velocity=(0.0, 0.0, 8.0))
    b.end_world()
    return b.finalize(device="cpu")


def _run(sol, m, steps: int, dt: float = 1.0e-3):
    a, b = m.state(), m.state()
    for _ in range(steps):
        sol.step(a, b, None, None, dt)
        a, b = b, a
    return a


def test_auto_box_covers_the_content_with_its_pad():
    m = _scene(shot=False)
    lo, hi = eb.auto_world_bounds(m)
    # content: the plate's bounding sphere (radius |(0.05, 0.05, 0.005)|) around (0, 0, 0.095); the ground is left out
    rp = float(np.linalg.norm([0.05, 0.05, 0.005]))
    assert np.allclose(lo, np.array([-rp, -rp, 0.095 - rp]) - 0.25, atol=1e-6)
    assert np.allclose(hi, np.array([rp, rp, 0.095 + rp]) + 0.25, atol=1e-6)
    assert eb.auto_world_bounds(newton.ModelBuilder().finalize(device="cpu")) is None


def test_scene_factory_retires_ejecta_and_counts_them():
    m = _scene(shot=True)
    sol = plugin._factory(m, {"particle_radius": R})
    top = sol.config.world_bounds[1][2]
    assert 0.3 < top < 1.0
    st = _run(sol, m, 120)  # the shot particle passes the top at ~0.05 s
    shot = m.particle_count - 1
    assert (m.particle_flags.numpy()[shot] & int(ParticleFlags.ACTIVE)) == 0
    rep = sol.report()
    assert rep.escaped.tolist() == [1]
    assert np.isclose(rep.mass_out[0], float(m.particle_mass.numpy()[shot]), rtol=1e-6)
    assert np.all(st.particle_q.numpy()[:shot, 2] < 0.2)  # the block stays on its plate


@pytest.mark.parametrize("kw", [{"world_bounds": None}, {"config": DFSPHConfig(particle_radius=R)}])
def test_factory_keeps_an_explicit_choice(kw):
    m = _scene(shot=True)
    assert plugin._factory(m, kw).config.world_bounds is None
    box = ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    assert plugin._factory(m, {"particle_radius": R, "world_bounds": box}).config.world_bounds == box


def test_escape_box_leaves_a_resting_scene_bitwise():
    m = _scene(shot=False)
    q = [_run(plugin._factory(m, kw), m, 40).particle_q.numpy().copy()
         for kw in ({"particle_radius": R}, {"particle_radius": R, "world_bounds": None})]
    assert np.array_equal(q[0], q[1])
