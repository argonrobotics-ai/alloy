# SPDX-License-Identifier: Apache-2.0
"""Liquid against walls thinner than the kernel support (liquid fix 3: 1 mm glass at h = 3-8 mm).

The slot pass keeps a particle on the side of a wall it started on, but only when the step ends inside the wall: a
step longer than the wall's thickness ends on the far side, outside it, and nothing tests the path unless the step
is longer than SWEEP h. Steps longer than half the thinnest wall are now swept (``integrate.sweep_length``)."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

import alloy

from alloy._vendor import newton
from alloy.solvers.sph.build import add_fluid_points, wall_box
from alloy.solvers.sph.config import BOUNDARY_SDF, register_custom_attributes
from alloy.solvers.sph.integrate import SWEEP, sweep_length
from alloy.solvers.sph.layout import thinnest_wall
from alloy.testing.sph import SPHScene, default_config, one_particle

R = 1.0e-3                      # super's radius: h = 4 mm, SWEEP h = 2 mm
T = 1.0e-3                      # the wall
DT = 0.02 / 64


def _shot(step: float):
    """One particle at x = 0 moving +x by ``step`` per DT, a 1 mm wall from x = 0.4 mm (within h: in its slots)."""
    return one_particle([((0.4e-3 + 0.5 * T, 0.0, 0.0), (0.5 * T, 0.02, 0.02))], (0.0, 0.0, 0.0),
                        (step / DT, 0.0, 0.0), r=R)


def _one_step(scene):
    s = scene.make_solver()
    mon = s.enable_leak_monitor()
    a, b = scene.model.state(), scene.model.state()
    s.step(a, b, None, None, DT)
    return s, mon, b.particle_q.numpy()[0]


def test_the_sweep_length_follows_the_thinnest_wall():
    assert sweep_length(4 * R, T) == pytest.approx(0.5 * T)
    assert sweep_length(4 * R, 1.0) == pytest.approx(SWEEP * 4 * R)
    scene = _shot(1.5e-3)
    assert thinnest_wall(scene.model, np.arange(scene.model.shape_count)) == pytest.approx(T, rel=1e-6)


def _thin_shape(b: newton.ModelBuilder, kind: str, z: float) -> None:
    """A 1 mm wall across the z axis, its near face at ``z``: a box, a cylinder disc (r = 20 mm) or a cone as flat."""
    attrs = {"sph:shape_boundary": BOUNDARY_SDF}
    xf = wp.transform((0.0, 0.0, z + 0.5 * T), wp.quat_identity())
    if kind == "box":
        wall_box(b, (0.0, 0.0, z + 0.5 * T), (0.02, 0.02, 0.5 * T))
    elif kind == "disc":
        b.add_shape_cylinder(-1, xform=xf, radius=0.02, half_height=0.5 * T, custom_attributes=attrs)
    else:
        b.add_shape_cone(-1, xform=xf, radius=0.02, half_height=0.5 * T, custom_attributes=attrs)


def test_a_disc_or_cone_is_as_thin_as_its_height():
    """A cylinder or cone counts by the smaller of its diameter and height: a 1 mm disc of 40 mm diameter is 1 mm
    thin (it read 40 mm, so a step across it was never swept); a 1 mm rod is 1 mm thin too."""
    for kind in ("disc", "cone"):
        b = newton.ModelBuilder()
        register_custom_attributes(b)
        _thin_shape(b, kind, 0.0)
        m = b.finalize(device="cpu")
        assert thinnest_wall(m, np.arange(m.shape_count)) == pytest.approx(T, rel=1e-6), kind
    b = newton.ModelBuilder()
    b.add_shape_cylinder(-1, radius=0.5 * T, half_height=0.02)
    m = b.finalize(device="cpu")
    assert thinnest_wall(m, np.arange(m.shape_count)) == pytest.approx(T, rel=1e-6)


def _channel(kind: str, step: float) -> SPHScene:
    """One particle (r = 2 mm, h = 8 mm, SWEEP h = 4 mm) at the origin moving +z by ``step`` per DT, in a channel of
    four walls 10 mm thick, 2 mm off, that fill its 4 boundary slots, so a 1 mm wall 2.5 mm ahead is not in them: only
    the path sweep can stop it, and only if the step is measured against the 1 mm wall, not SWEEP h."""
    cfg = default_config(2.0e-3)
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    register_custom_attributes(b)
    b.begin_world()
    for x, y in ((1, 0), (-1, 0), (0, 1), (0, -1)):          # 10 mm thick: none of them sets the sweep
        wall_box(b, (x * 7.0e-3, y * 7.0e-3, 0.0), (5.0e-3 if x else 7.0e-3, 5.0e-3 if y else 7.0e-3, 5.0e-3))
    _thin_shape(b, kind, 2.5e-3)
    add_fluid_points(b, cfg, np.zeros((1, 3)), 1000.0, 1.0e-3, (0.0, 0.0, step / DT))
    b.end_world()
    return SPHScene(b.finalize(device="cpu"), cfg, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), [(1, 1, 1)])


@pytest.mark.parametrize("kind", ["box", "disc", "cone"])
@pytest.mark.parametrize("step", [3.6e-3, 3.9e-3])
def test_a_step_across_a_1mm_wall_in_a_channel_is_stopped(kind, step):
    """Law: steps of 3.6 and 3.9 mm (under SWEEP h = 4 mm, past the wall's far face at 3.5 mm) end before the wall,
    whatever primitive it is; nothing crosses wall solid. A solver that sweeps only past SWEEP h, or reads the disc
    or cone by its diameter, lets the particle through."""
    _, mon, q = _one_step(_channel(kind, step))
    assert q[2] < 2.5e-3, f"{kind}: the particle ended at z = {q[2] * 1e3:.3f} mm, past the wall's near face (2.5 mm)"
    assert int(mon.events.numpy().sum()) == 0


def _thin_tub(accuracy: str) -> alloy.Scene:
    """A tub with 1 mm glass walls from boxes (``features.min_wall`` declared: the API refuses solids under 1.25 mm
    otherwise), inner 48 x 48 mm, water 24 mm deep."""
    extra = {"features": {"min_wall": {"value": T, "source": "model", "note": "1 mm glass"}}}
    sc = alloy.Scene(dt=0.01, liquid_accuracy=accuracy)
    sc.add_ground()
    inner, depth = 0.048, 0.04
    sc.add_box("floor", size=(inner + 2 * T, inner + 2 * T, T), pos=(0, 0, 0.5 * T), static=True,
               material="glass_soda_lime", card_extra=extra)
    for k, (x, y) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
        size = (T, inner + 2 * T, depth) if x else (inner, T, depth)
        sc.add_box(f"wall{k}", size=size, pos=(x * 0.5 * (inner + T), y * 0.5 * (inner + T), T + 0.5 * depth),
                   static=True, material="glass_soda_lime", card_extra=extra)
    sc.add_liquid("water", size=(inner, inner, 0.024), pos=(0, 0, T + 0.012))
    return sc


@pytest.mark.compile_heavy
def test_still_water_stays_in_a_tub_of_1mm_glass():
    """1 s at default (h = 8 mm, 8x the wall): no particle ends outside the tub or inside a wall, health ok. 5 s at
    every spacing in notes (zero leaks)."""
    sim = alloy.build(_thin_tub("default"), device="cpu")
    try:
        sim.run(1.0)
        p = sim.state(["water"])["water.points"][0]
        assert sim.check()["ok"], sim.check()["problems"][:2]
        assert np.all(np.abs(p[:, :2]) < 0.024) and np.all(p[:, 2] > T), "a particle left through 1 mm glass"
    finally:
        sim.close()
