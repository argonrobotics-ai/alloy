# SPDX-License-Identifier: Apache-2.0
"""The concrete tasks' scripted carriage (alloy.tasks.concrete_tools): a tool follows its keyframes in x, y and
attitude, presses with the set total force (FORCE) or holds a height (HEIGHT), and publishes its load (CPU)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy._vendor import newton

from alloy.plugins.errors import PluginError
from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.tasks.concrete_tools import FORCE, HEIGHT, Stroke, ToolCarriage, ToolProgram

HALF = (0.1, 0.04, 0.005)                    # a float-like block: 200 x 80 x 10 mm
DENSITY = 2500.0                             # 0.4 kg


def _scene(z0: float):
    def build(b: newton.ModelBuilder) -> None:
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, -0.01), hx=0.4, hy=0.3, hz=0.01, label="slab")
        tool = b.add_body(xform=body_xform(-0.1, 0.0, z0), label="float")
        b.add_shape_box(tool, hx=HALF[0], hy=HALF[1], hz=HALF[2], cfg=newton.ModelBuilder.ShapeConfig(density=DENSITY))
    return build


def _run(program: ToolProgram, periods: int, z0: float = HALF[2] + 1e-4,
         gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)):
    carriage = ToolCarriage(programs=(program,))
    sim = build_simulation(TestScene(build=_scene(z0), ground=False, worlds=2, dt=2.0e-3, gravity=gravity),
                           [carriage], "cpu")
    for _ in range(periods):
        sim.step_period()
    return sim, carriage


def test_a_pressing_tool_follows_its_stroke_and_publishes_its_load():
    """FORCE: 20 N in total on the slab while it slides 0.2 m in x in 2 s at a 2° pitch; it keeps its yaw."""
    z = HALF[2]
    prog = ToolProgram("float", (Stroke(0.0, (-0.1, 0.0, z), pitch=math.radians(2.0), force=20.0, mode=FORCE),
                                 Stroke(2.0, (0.1, 0.0, z), pitch=math.radians(2.0), force=20.0, mode=FORCE)))
    sim, carriage = _run(prog, periods=125)                              # 2.5 s: the stroke and 0.5 s of hold
    q = sim.state.body_q.numpy()
    assert np.allclose(q[:, 0], 0.1, atol=2e-3), q[:, :3]                 # it got there, both worlds
    assert np.allclose(q[0], q[1])                                        # identical worlds
    assert np.allclose(carriage.read()["load"], 20.0, rtol=1e-3)
    qx, qy, qz, qw = q[0, 3:]
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    assert abs(yaw) < 0.01


def test_a_height_segment_holds_the_tool_off_the_slab():
    """HEIGHT: the body origin is held 20 mm up (no contact), so the load is the push that cancels gravity: 0."""
    prog = ToolProgram("float", (Stroke(0.0, (-0.1, 0.0, 0.02), mode=HEIGHT),))
    sim, carriage = _run(prog, periods=50, z0=0.02)
    q = sim.state.body_q.numpy()
    assert np.allclose(q[:, 2], 0.02, atol=5e-4), q[:, 2]
    assert np.abs(carriage.read()["load"]).max() < 0.5                   # held up: no load on the work


def test_a_work_face_laid_flat_with_gravity_along_it():
    """Gravity along -y (a wall's joint laid flat): the carriage carries the weight in plane, so a held tool stays put
    and a pressing one presses with exactly its force (the weight no longer adds to it)."""
    g = (0.0, -9.81, 0.0)
    held = ToolProgram("float", (Stroke(0.0, (-0.1, 0.0, 0.02), mode=HEIGHT),))
    sim, carriage = _run(held, periods=50, z0=0.02, gravity=g)
    q = sim.state.body_q.numpy()
    assert np.allclose(q[:, :3], (-0.1, 0.0, 0.02), atol=5e-4), q[:, :3]
    assert np.abs(carriage.read()["load"]).max() < 0.5
    z = HALF[2]
    press = ToolProgram("float", (Stroke(0.0, (-0.1, 0.0, z), force=20.0, mode=FORCE),))
    sim, carriage = _run(press, periods=50, gravity=g)
    q = sim.state.body_q.numpy()
    assert np.allclose(q[:, 1], 0.0, atol=2e-3), q[:, :3]                # it does not slide down the face
    assert np.allclose(carriage.read()["load"], 20.0, rtol=1e-3)


def test_programs_refuse_bad_keyframes_and_unstable_gains():
    with pytest.raises(PluginError, match="increase"):
        ToolProgram("float", (Stroke(1.0, (0, 0, 0)), Stroke(1.0, (0, 0, 0))))
    with pytest.raises(PluginError, match="mode"):
        ToolProgram("float", (Stroke(0.0, (0, 0, 0), mode=7),))
    with pytest.raises(PluginError, match=r"release must be in \[0, 1\]"):
        ToolProgram("float", (Stroke(0.0, (0, 0, 0), release=1.5),))
    carriage = ToolCarriage(programs=(ToolProgram("float", (Stroke(0.0, (0, 0, 0.02), mode=HEIGHT),)),), omega=2e3)
    with pytest.raises(PluginError, match="unstable"):
        build_simulation(TestScene(build=_scene(0.02), ground=False, worlds=1, dt=2.0e-3), [carriage],
                         "cpu").step_period()


def test_each_world_can_follow_its_own_program():
    """world_programs: world 0 holds the tool 20 mm up, world 1 30 mm (a policy aiming at each world's start)."""
    progs = tuple(ToolProgram("float", (Stroke(0.0, (-0.1, 0.0, z), mode=HEIGHT),)) for z in (0.02, 0.03))
    carriage = ToolCarriage(programs=progs[:1], world_programs=tuple((p,) for p in progs))
    sim = build_simulation(TestScene(build=_scene(0.025), ground=False, worlds=2, dt=2.0e-3), [carriage], "cpu")
    for _ in range(50):
        sim.step_period()
    assert np.allclose(sim.state.body_q.numpy()[:, 2], (0.02, 0.03), atol=5e-4)
    with pytest.raises(PluginError, match="in that order"):
        ToolCarriage(programs=progs[:1], world_programs=((ToolProgram("other", progs[0].keys),),))
    one = ToolCarriage(programs=progs[:1], world_programs=((progs[0],),))
    with pytest.raises(PluginError, match="1 sets for 2 worlds"):
        build_simulation(TestScene(build=_scene(0.02), ground=False, worlds=2, dt=2.0e-3), [one], "cpu").step_period()


def test_a_keys_release_drives_the_films_store_tool_during_its_segment_only():
    """Stroke.release: the share of a store tool's free load let go each exchange of the segment its key starts,
    written into the film's per-tool input (flow.tool_release, matched by body); 0 outside that segment."""
    from alloy.plugins.surface_flow import FilmMaterial, PlaneChart, SurfaceFlow
    from alloy.plugins.surface_flow.applicators import Applicator
    from alloy.plugins.surface_flow.stores import Store

    z = 0.02
    keys = (Stroke(0.0, (-0.1, 0.0, z), mode=HEIGHT), Stroke(0.5, (-0.1, 0.0, z), mode=HEIGHT, release=0.5),
            Stroke(1.0, (-0.1, 0.0, z), mode=HEIGHT))
    flow = SurfaceFlow(charts=(PlaneChart("slab", cells=(16, 12)),), inertia=False, max_depth=2e-3,
                       material=FilmMaterial("paste", density=2000.0, viscosity=20.0, yield_stress=500.0,
                                             surface_tension=0.0),
                       tools=(Applicator(body="float", shape="rect", size=HALF[:2], center=(0.0, 0.0, -HALF[2]),
                                         stiffness=5.0e5, damping=5.0e-4,
                                         store=Store(capacity=1e-4, retained=0.0, held=1.0, split=1.0, drip=0.0)),))
    carriage = ToolCarriage(programs=(ToolProgram("float", keys),), flow=flow)
    sim = build_simulation(TestScene(build=_scene(z), ground=False, worlds=2, dt=2.0e-3), [carriage, flow], "cpu")
    seen = []
    for i in range(75):
        sim.step_period()
        seen.append(((i + 1) * 0.02, flow.tool_release.numpy()[:, 0].copy()))
    inside = [r for t, r in seen if 0.54 <= t <= 0.98]
    outside = [r for t, r in seen if t <= 0.46 or t >= 1.04]
    assert inside and outside
    assert all(np.all(r == 0.5) for r in inside) and all(np.all(r == 0.0) for r in outside)

