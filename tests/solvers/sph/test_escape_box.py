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


def test_drive_reach_bounds_every_place_a_drive_takes_a_shape():
    c, r = np.array([[0.1, 0.0, 0.2]]), np.array([0.05])
    zero = np.zeros((1, 3))
    lo, hi = eb.drive_reach(c, r, c, np.array([[5.0, 0.0, 0.0]]), zero, 0.15)       # a 0.75 m slide in +x
    assert np.allclose(lo, [0.05, -0.05, 0.15], rtol=0.0, atol=1e-12)
    assert np.allclose(hi, [0.9, 0.05, 0.25], rtol=0.0, atol=1e-12)
    piv = np.array([[0.0, 0.0, 0.2]])                                                # a turn about a point 0.1 m off
    lo, hi = eb.drive_reach(c, r, piv, zero, np.array([[0.0, 0.0, 1.0]]), 10.0)
    assert np.allclose(lo, [-0.15, -0.15, 0.05], rtol=0.0, atol=1e-12)
    assert np.allclose(hi, [0.15, 0.15, 0.35], rtol=0.0, atol=1e-12)
    lo, hi = eb.drive_reach(c, r, c, np.array([[0.0, -1.0, 0.0]]), zero, float("inf"))   # no end: to -inf in y
    assert lo[1] == -np.inf and np.isfinite(hi).all() and np.isfinite(lo[[0, 2]]).all()


def test_widen_grows_only_an_automatic_box():
    m = _scene(shot=False)
    auto = plugin._factory(m, {"particle_radius": R})
    lo0, hi0 = (np.array(v) for v in auto.config.world_bounds)
    assert not eb.widen(auto, lo0 + 0.3, hi0 - 0.3)                    # inside the box: unchanged
    assert eb.widen(auto, np.array([0.0, 0.0, 0.1]), np.array([2.0, 0.0, 0.1]))   # 2 m along +x, inside elsewhere
    lo, hi = (np.array(v) for v in auto.config.world_bounds)
    assert hi[0] == 2.0 + auto.escape_box_pad and np.array_equal(lo, lo0) and np.array_equal(hi[1:], hi0[1:])
    assert np.allclose(np.array(auto.emitters.bounds[1]), hi, rtol=0.0, atol=1e-6)   # the retire kernel's copy
    box = ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    given = plugin._factory(m, {"particle_radius": R, "world_bounds": box})
    assert not eb.widen(given, np.zeros(3), np.array([5.0, 0.0, 0.0])) and given.config.world_bounds == box


@pytest.mark.compile_heavy
def test_liquid_in_a_driven_sealed_box_is_carried_not_retired():
    """S1-015: a sealed 20 mm box with 700 particles driven 0.75 m at 5 m/s left all 700 behind, retired and frozen
    in mid-air where they crossed the build-time escape box (health ok). The drive widens the box: none retires and
    every particle is inside the box at the end. The lid sits flush on the walls (the repro's 0.5 mm gap let 1 %
    through under the 5 m/s start: a real gap, by design)."""
    import alloy.api as alloy  # noqa: PLC0415

    sc = alloy.Scene(dt=0.01)
    sc.add_ground()
    sc.add_open_box("vessel", inner=(0.08, 0.08, 0.08), wall=0.02, pos=(0.0, 0.0, 0.20), kinematic=True)
    sc.add_box("lid", size=(0.12, 0.12, 0.02), pos=(0.0, 0.0, 0.31), kinematic=True)
    sc.add_liquid("water", size=(0.04, 0.04, 0.03), pos=(0.0, 0.0, 0.24))
    sim = alloy.build(sc, device="cpu")
    try:
        for name in ("vessel", "lid"):
            sim.drive(name, velocity=(5.0, 0.0, 0.0), duration=0.15)
        sim.step(16)
        st, h = sim.state(world=0), sim.health()
        rel = st["water.points"] - np.array([st["vessel.pos"][0], st["vessel.pos"][1], 0.0])
        inside = (np.abs(rel[:, 0]) <= 0.06) & (np.abs(rel[:, 1]) <= 0.06) & (rel[:, 2] >= 0.2) & (rel[:, 2] <= 0.32)
        assert float(st["vessel.pos"][0]) == pytest.approx(0.75, rel=0.0, abs=1e-5)
        assert h["retired_particles"] == [0], h["retired_particles"]
        assert int(inside.sum()) == len(rel), f"{len(rel) - int(inside.sum())} of {len(rel)} outside the vessel"
    finally:
        sim.close()
