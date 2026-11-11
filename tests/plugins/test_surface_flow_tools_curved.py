# SPDX-License-Identifier: Apache-2.0
"""Surface flow, applicators on a revolution chart (WP8's sponge on a pan): an implicit flat pad's underside is measured
along each cell's own normal, so it rests in a curved trough at the closed-form give and slides round a ring,
across the chart's seam at theta = 0, with an exact ledger (CPU)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy._vendor import newton

from alloy.plugins.errors import PluginError
from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.surface_flow import FilmMaterial, RevolutionChart, SurfaceFlow
from alloy.plugins.surface_flow.applicators import Applicator

G = 9.81
OIL = FilmMaterial("oil_1000cSt", density=970.0, viscosity=0.97, surface_tension=0.0)
R0, RC, PSI = 0.1, 0.05, 0.5          # a ring trough: its bottom circle's radius, its section's radius, half its arc
A, B, HZ, DROP = 0.008, 0.015, 0.005, 0.002     # the pad's half extents (across, along the ring); the block above it
TROUGH = tuple((R0 + RC * math.sin(p), RC * (1.0 - math.cos(p))) for p in np.linspace(-PSI, PSI, 101))


def trough(n_theta: int = 256) -> RevolutionChart:
    """Wetted on its inside (the profile runs outward; the material is on its right)."""
    return RevolutionChart(profile=TROUGH, n_theta=n_theta, ds=2.0e-3)


def pad(x: float = 0.0, **kw) -> Applicator:
    """The block's underside, DROP below it; the block's centre is ``x`` along the body's x axis."""
    return Applicator(body="tool", shape="rect", size=(A, B), center=(x, 0.0, -HZ - DROP), implicit=True, **kw)


def free_tool(z_underside: float, density: float, dx: float = 0.0):
    """A free block centred over x = R0 + ``dx``, its pad (``pad()``) at ``z_underside``."""
    def build(b: newton.ModelBuilder) -> None:
        tool = b.add_body(xform=body_xform(R0 + dx, 0.0, z_underside + DROP + HZ), label="tool")
        b.add_shape_box(tool, hx=A, hy=B, hz=HZ, cfg=newton.ModelBuilder.ShapeConfig(density=density))
    return build


def ring_tool(z_underside: float, theta0: float):
    """The block on an arm turning about the trough's axis, starting at angle ``theta0``: it keeps its height."""
    def build(b: newton.ModelBuilder) -> None:
        q = (0.0, 0.0, math.sin(0.5 * theta0), math.cos(0.5 * theta0))
        xf = wp_transform((0.0, 0.0, z_underside + DROP + HZ), q)
        tool = b.add_link(xform=xf, label="tool")
        b.add_articulation([b.add_joint_revolute(-1, tool, parent_xform=xf, axis=newton.Axis.Z)])
        b.add_shape_box(tool, xform=body_xform(R0, 0.0, 0.0), hx=A, hy=B, hz=HZ,
                        cfg=newton.ModelBuilder.ShapeConfig(density=2.0e6))
    return build


def wp_transform(p, q):
    import warp as wp  # noqa: PLC0415
    return wp.transform(wp.vec3(*p), wp.quat(*q))


def _sim(flow: SurfaceFlow, build, init=None, gravity=(0.0, 0.0, -G), dt: float = 2.0e-3):
    return build_simulation(TestScene(build=build, init=init, ground=False, gravity=gravity, worlds=1, dt=dt),
                            [flow], "cpu")


def test_a_pad_rests_in_a_curved_trough_at_the_closed_form_give():
    """A flat pad (2a across the trough) rests on the sides of a trough of section radius R_c: it presses
    k·(d + x²/(2R_c)) along the normals, so its weight balances at d = m·g/(k·A) - a²/(6R_c) below the bottom
    (0.28 mm here, against 0.49 mm on a plane), to 5 %. The normals all pass through the section's centre, so a
    frictionless pad swings there like a pendulum (period 2π·sqrt(R_c/g), 0.45 s; the damper acts on approach,
    not on that swing) about its rest: the ring's bottom line under it, b²/(6R_0) inward of the pad's centre (the
    ring curves in under its ends). Started there, over one swing it stays at it on average, within 0.1° of level,
    across the chart's seam at theta = 0."""
    k = 1.0e6
    flow = SurfaceFlow(charts=(trough(),), material=OIL, max_depth=2.0e-3,
                       tools=(pad(stiffness=k, damping=0.01, friction=0.0),))
    x_rest = -B * B / (6.0 * R0)
    sim = _sim(flow, free_tool(-0.28e-3, 5000.0, x_rest), dt=4.0e-3)
    for _ in range(40):
        sim.step_period()
    swing = []
    for _ in range(112):                                                 # one swing at 4 ms
        sim.step_period()
        swing.append(sim.state.body_q.numpy()[0].copy())
    q = np.mean(swing, axis=0)
    m = float(sim.model.body_mass.numpy()[0])
    give = -(float(q[2]) - DROP - HZ)                                   # the pad's centre below the trough's bottom
    want = m * G / (k * 4.0 * A * B) - A * A / (6.0 * RC)
    assert give == pytest.approx(want, rel=0.05, abs=0.0), (give, want)
    assert abs(float(q[0]) - R0 - x_rest) < 0.05e-3 and abs(float(q[1])) < 1.0e-5, q[:3]
    assert np.abs(np.asarray(swing)[:, 3:6]).max() < 1.0e-3                       # within 0.1° of level
    cover = flow.read()["tool_cover"][0]
    theta = np.arctan2(flow.tables.cpos.numpy()[:, 1], flow.tables.cpos.numpy()[:, 0])
    assert (cover[theta > 0.0] > 0.0).sum() > 10 and (cover[theta < 0.0] > 0.0).sum() > 10   # both sides of the seam


def test_a_pad_slides_round_the_ring_and_ploughs_with_an_exact_ledger():
    """A pad held 0.3 mm over the trough's bottom on an arm sliding at U = 0.05 m/s round it through theta = 0, over
    0.5 mm of oil: the covered cells slide along the ring at ω·r (to 0.5 %), the film it ploughs piles ahead of it,
    and the volume ledger closes to 1e-12."""
    h0, gap, U, theta0 = 0.5e-3, 0.3e-3, 0.05, -0.05
    flow = SurfaceFlow(charts=(trough(),), material=OIL, max_depth=3.0e-3, tools=(pad(R0, stiffness=1.0e3),))

    def init(sim) -> None:
        flow.set_depth(h0)
        sim.state.joint_qd.assign(np.full(1, U / R0, np.float32))
        newton.eval_fk(sim.model, sim.state.joint_q, sim.state.joint_qd, sim.state)
    sim = _sim(flow, ring_tool(gap, theta0), init, gravity=(0.0, 0.0, 0.0))
    start = None
    for n in range(60):
        sim.step_period()
        if n == 0:
            start = float(flow.read()["ledger"][0])
    r = flow.read()
    assert abs(float(r["ledger"][0]) - start) < 1.0e-12 * start
    omega = float(sim.state.joint_qd.numpy()[0])
    pos = flow.tables.cpos.numpy()
    covered = r["tool_cover"][0] >= 1.0
    v = r["tool_vel"][0][covered]
    hoop = np.stack([-pos[covered, 1], pos[covered, 0], np.zeros(covered.sum())], 1)    # ω × r about z
    assert covered.sum() > 20 and np.abs(v - omega * hoop).max() < 0.005 * U, np.abs(v - omega * hoop).max()
    h = flow.depth(0)[0].reshape(-1)
    theta = np.arctan2(pos[:, 1], pos[:, 0])
    lead = theta0 + float(sim.state.joint_q.numpy()[0]) + B / R0       # the pad's leading edge (angle)
    ring = np.abs(np.hypot(pos[:, 0], pos[:, 1]) - R0) < 0.5 * A
    ahead = ring & (theta > lead) & (theta < lead + 0.03)
    assert h[ahead].max() > h0 * 1.05                                   # the roll ahead of the pad
    assert h[covered].max() < h0                                        # less under it than it met


def test_a_pad_on_a_two_sided_dish_rests_on_the_floor_it_faces():
    """WP8's pan: one chart wraps a dish's inner floor, wall, rim, outer wall and outer bottom, 2 mm under the floor. A
    pad sunk 1.5 mm into the floor is nearer the outer bottom, which faces away: the window starts from the nearest
    segment the underside faces, so it rests on the inner floor at m·g/(k·A) on its covered area (to 2 %; the area
    to 5 %, the hoop cells covered by their own widths), on upward cells only (fails starting from the nearest
    segment: it covers nothing and falls through)."""
    T, k = 2.0e-3, 1.0e5
    dish = RevolutionChart(profile=((0.025, T), (0.09, T), (0.09, 0.08), (0.093, 0.08), (0.093, 0.0), (0.025, 0.0)),
                           n_theta=64, ds=2.0e-3, ends=("open", "open"))
    ha, hb = 0.03, 0.02
    tool = Applicator(body="tool", shape="rect", size=(ha, hb), center=(0.0, 0.0, -HZ - DROP), implicit=True,
                      stiffness=k, damping=0.01, friction=0.4)
    flow = SurfaceFlow(charts=(dish,), material=OIL, max_depth=2.0e-3, tools=(tool,))

    def build(b: newton.ModelBuilder) -> None:
        body = b.add_body(xform=body_xform(0.057, 0.0, T - 1.4e-3 + DROP + HZ), label="tool")   # on the floor
        b.add_shape_box(body, hx=ha, hy=hb, hz=HZ, cfg=newton.ModelBuilder.ShapeConfig(density=1500.0))
    sim = _sim(flow, build, dt=4.0e-3)
    for _ in range(60):
        sim.step_period()
    m = float(sim.model.body_mass.numpy()[0])
    q = sim.state.body_q.numpy()[0]
    cover = flow.read()["tool_cover"][0]
    area = float((cover * flow.tables.area_np).sum())
    give = T - (float(q[2]) - DROP - HZ)
    assert area == pytest.approx(4.0 * ha * hb, rel=0.05, abs=0.0)
    assert give == pytest.approx(m * G / (k * area), rel=0.02, abs=0.0), (give, m * G / (k * area))
    assert give > 0.5 * T and (flow.tables.cnrm.numpy()[cover > 0.0][:, 2] > 0.5).all()   # none underneath


def test_a_pad_far_behind_a_curved_floor_does_not_act_on_it():
    """SCR-WP8-9: a pad 0.26 m under a dish's floor, centred inside its radius and facing it, read a gap of -0.26 m
    with full cover (the plane rule in the cell's frame has no far side), and the implicit pad threw an 8 g sponge at
    13.6 m/s through the pan. A tool acts on a curved cell only down to max(1 cm, its reach) behind it: it covers
    nothing there and stays at rest. Pressed 1.5 mm in, it still rests (the dish test above)."""
    T = 2.0e-3
    dish = RevolutionChart(profile=((0.025, T), (0.09, T), (0.09, 0.08), (0.093, 0.08), (0.093, 0.0), (0.025, 0.0)),
                           n_theta=64, ds=2.0e-3, ends=("open", "open"))
    ha, hb = 0.03, 0.02
    tool = Applicator(body="tool", shape="rect", size=(ha, hb), center=(0.0, 0.0, -HZ - DROP), implicit=True,
                      stiffness=1.0e5, damping=0.01, friction=0.4)
    flow = SurfaceFlow(charts=(dish,), material=OIL, max_depth=2.0e-3, tools=(tool,))

    def build(b: newton.ModelBuilder) -> None:
        body = b.add_body(xform=body_xform(0.057, 0.0, T - 0.26 + DROP + HZ), label="tool")   # 0.26 m under
        b.add_shape_box(body, hx=ha, hy=hb, hz=HZ, cfg=newton.ModelBuilder.ShapeConfig(density=1500.0))
    sim = _sim(flow, build, gravity=(0.0, 0.0, 0.0), dt=4.0e-3)
    for _ in range(5):
        sim.step_period()
    assert float(flow.read()["tool_cover"][0].sum()) == 0.0
    assert float(np.abs(sim.state.body_qd.numpy()[0]).max()) == 0.0


@pytest.mark.parametrize("kw, match", [({"implicit": False}, "implicit"), ({"shape": "roller"}, "implicit")])
def test_a_revolution_chart_refuses_tools_it_does_not_take(kw, match):
    spec = {"body": "tool", "shape": "rect", "size": (A, B), "center": (0.0, 0.0, -HZ - DROP), "implicit": True}
    spec.update(kw)
    flow = SurfaceFlow(charts=(trough(),), material=OIL, max_depth=2.0e-3, tools=(Applicator(**spec),))
    with pytest.raises(PluginError, match=match):
        _sim(flow, free_tool(0.0, 5000.0))
