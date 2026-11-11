# SPDX-License-Identifier: Apache-2.0
"""Surface flow, the tools' passes (tool_passes.py, tool_passes_force.py) against tool_kernels' one thread per world:
on the same state, the ceilings are the serial kernel's bit for bit, and the pads' and drag's sums agree with it to
float32 rounding and repeat bit for bit. Plane chart: three tools in two worlds (a sliding blade, an implicit pad
across it, a roller) and a store's brush; revolution chart: an implicit pad round a ring across its seam (CPU)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton

from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.surface_flow import FilmMaterial, PlaneChart, RevolutionChart, SurfaceFlow
from alloy.plugins.surface_flow import tool_passes as TP
from alloy.plugins.surface_flow import tool_passes_force as TPF
from alloy.plugins.surface_flow import tool_steps as TS
from alloy.plugins.surface_flow.applicators import Applicator
from alloy.plugins.surface_flow.host_api import H_WET
from alloy.plugins.surface_flow.stores import Store

OIL = FilmMaterial("oil_1000cSt", density=970.0, viscosity=0.97, surface_tension=0.0)
LENGTH, WIDTH, Z = 0.2, 0.08, 0.05
# carriages: (label, x0, y, x speed [m/s] per world) on slides along x at height Z. The pad and the blade come at each
# other, so the pad's roll's zone reaches the cells the blade covers (and pushes the other way)
CARTS = (("blade", 0.0, -0.015, (-0.02, -0.01)), ("pad", -0.022, 0.005, (0.08, 0.07)),
         ("roller", 0.0, 0.025, (0.04, 0.06)), ("brush", 0.0, -0.03, (0.03, 0.02)))


def carts_scene():
    """The plate on a body welded to the world (the chart rides a body: the pads' reaction goes to it), and the
    carriages."""
    def build(b: newton.ModelBuilder) -> None:
        pxf = body_xform(0.0, 0.0, -0.01)
        plate = b.add_link(xform=pxf, label="slab")
        b.add_shape_box(plate, hx=0.5 * LENGTH, hy=0.5 * WIDTH, hz=0.01, label="plate")
        b.add_articulation([b.add_joint_fixed(-1, plate, parent_xform=pxf)])
        heavy = newton.ModelBuilder.ShapeConfig(density=1.0e6)
        for label, x0, y, _ in CARTS:
            xf = body_xform(x0, y, Z)
            cart = b.add_link(xform=xf, label=label)
            b.add_shape_sphere(cart, radius=0.004, cfg=heavy)
            b.add_articulation([b.add_joint_prismatic(-1, cart, parent_xform=xf, axis=newton.Axis.X)])
    return build


def plane_tools() -> tuple[Applicator, ...]:
    """A ceiling blade 0.4 mm over the plate; an implicit pad 0.1 mm into it whose footprint overlaps the blade's; a
    roller, axis along y, its bottom 0.3 mm over the plate; a store's brush 0.2 mm into it."""
    return (Applicator(body="blade", shape="rect", size=(0.008, 0.012), center=(0.0, 0.0, 4.0e-4 - Z)),
            Applicator(body="pad", shape="rect", size=(0.01, 0.014), center=(0.0, -0.012, -1.0e-4 - Z),
                       implicit=True, stiffness=2.0e5),
            Applicator(body="roller", shape="roller", size=(0.006, 0.01), axis=(0.0, 1.0, 0.0),
                       center=(0.0, 0.0, 0.006 + 3.0e-4 - Z), stiffness=1.0e5),
            Applicator(body="brush", shape="rect", size=(0.004, 0.006), center=(0.0, 0.0, -2.0e-4 - Z),
                       stiffness=1.0e6, friction=0.0,
                       store=Store(capacity=2.0e-6, retained=0.0, held=1.0, initial=1.0, drip=0.0, pickup=0.0)))


def plane_sim(periods: int):
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(100, 40)),), material=OIL, max_depth=3.0e-3,
                       tools=plane_tools())

    def init(sim) -> None:
        c = flow.cell_centres(0).reshape(-1, 3)
        depth = 1.0e-3 * (1.0 + 0.3 * np.sin(60.0 * c[:, 0]) * np.cos(40.0 * c[:, 1]))
        flow.set_depth(np.tile(depth, (2, 1)))
        qd = np.array([[v[w] for *_, v in CARTS] for w in range(2)], np.float32).reshape(-1)
        sim.state.joint_qd.assign(qd)
        newton.eval_fk(sim.model, sim.state.joint_q, sim.state.joint_qd, sim.state)
    sim = build_simulation(TestScene(build=carts_scene(), init=init, ground=False, gravity=(0.0, 0.0, -9.81),
                                     worlds=2, dt=2.0e-3), [flow], "cpu")
    for _ in range(periods):
        sim.step_period()
    return flow, sim


R0, RC, PSI = 0.1, 0.05, 0.5
TROUGH = tuple((R0 + RC * math.sin(p), RC * (1.0 - math.cos(p))) for p in np.linspace(-PSI, PSI, 101))


def ring_sim(periods: int):
    """An implicit pad on an arm turning round a ring trough, starting 0.1 rad before its seam (WP8's sponge)."""
    def build(b: newton.ModelBuilder) -> None:
        th = -0.1
        q = wp.quat(0.0, 0.0, math.sin(0.5 * th), math.cos(0.5 * th))
        xf = wp.transform(wp.vec3(0.0, 0.0, 2.0e-3 + 0.005), q)
        tool = b.add_link(xform=xf, label="tool")
        b.add_articulation([b.add_joint_revolute(-1, tool, parent_xform=xf, axis=newton.Axis.Z)])
        b.add_shape_box(tool, xform=body_xform(R0, 0.0, 0.0), hx=0.008, hy=0.015, hz=0.005,
                        cfg=newton.ModelBuilder.ShapeConfig(density=2.0e6))
    pad = Applicator(body="tool", shape="rect", size=(0.008, 0.015), center=(R0, 0.0, -0.005 - 2.0e-3 - 4.0e-4),
                     implicit=True)
    flow = SurfaceFlow(charts=(RevolutionChart(profile=TROUGH, n_theta=128, ds=2.0e-3),), material=OIL,
                       max_depth=3.0e-3, tools=(pad,))

    def init(sim) -> None:
        flow.set_depth(1.0e-3)
        sim.state.joint_qd.assign(np.array([0.8], np.float32))
        newton.eval_fk(sim.model, sim.state.joint_q, sim.state.joint_qd, sim.state)
    sim = build_simulation(TestScene(build=build, init=init, ground=False, gravity=(0.0, 0.0, -9.81), worlds=1,
                                     dt=2.0e-3), [flow], "cpu")
    for _ in range(periods):
        sim.step_period()
    return flow, sim


def ceiling_arrays(flow) -> dict[str, wp.array]:
    tf = flow.tf
    return dict(ceil=tf.ceil, cover=tf.cover, vel=tf.vel, push=tf.push, press=tf.press, wceil=tf.wceil,
                wcover=tf.wcover, wvel=tf.wvel, pore=tf.pore, cwet=flow.cwet, wins=flow.tools.wins)


def pad_arrays(flow, forces) -> dict[str, wp.array]:
    return dict(body_f=forces.body_f, drag_scale=flow.tool_drag_scale, pad_ratio=flow.tool_pad_ratio,
                normal=flow.tool_normal)


def from_same(arrays: dict[str, wp.array], *runs) -> list[dict]:
    """Run each of ``runs`` from the same starting values of ``arrays``; each one's results."""
    keep = {k: a.numpy().copy() for k, a in arrays.items()}
    out = []
    for run in runs:
        for k, v in arrays.items():
            v.assign(keep[k])
        run()
        out.append({k: v.numpy().copy() for k, v in arrays.items()})
    return out


def forces_of(sim) -> SimpleNamespace:
    return SimpleNamespace(body_f=wp.zeros(sim.model.body_count, dtype=wp.spatial_vector, device="cpu"))


@pytest.mark.parametrize("make", [plane_sim, ring_sim], ids=["plane", "ring"])
def test_the_ceilings_are_the_serial_kernels_bit_for_bit(make):
    """Mid-run (tools in the film, the blade's roll zone ahead, the pad over the blade's footprint, the ring's pad
    across its seam): every field the ceilings write equals tools_kernel's exactly."""
    for periods in (3, 9):
        flow, sim = make(periods)
        ser, par = from_same(ceiling_arrays(flow), lambda: TS.serial_ceilings(flow, sim.state),
                             lambda: TP.ceilings(flow, sim.state))
        flow.tool_passes.check()
        assert int((par["cover"] > 0).sum()) > 20 and int((np.abs(par["push"]).sum(axis=-1) > 0).sum()) > 0
        for k in ser:
            assert np.array_equal(ser[k], par[k]), (periods, k, np.abs(ser[k].astype(float)
                                                                       - par[k].astype(float)).max())


@pytest.mark.parametrize("make", [plane_sim, ring_sim], ids=["plane", "ring"])
def test_the_pads_agree_with_the_serial_kernel_to_rounding_and_repeat_exactly(make):
    """The pads' loads, ratios, drag caps and wrenches on the same state: within float32 rounding of
    tool_force_kernel's (the sums' order differs), and bit for bit the same run twice."""
    flow, sim = make(6)
    forces = forces_of(sim)
    arrays = pad_arrays(flow, forces)
    pads = lambda: TPF.pads(flow, sim.state, forces, 2.0e-3, H_WET)              # noqa: E731
    ser, par, par2 = from_same(arrays, lambda: TS.serial_pads(flow, sim.state, forces, 2.0e-3), pads, pads)
    flow.tool_passes.check()
    assert float(np.abs(ser["normal"]).max()) > 0.0 and float(np.abs(ser["body_f"]).max()) > 0.0
    if make is plane_sim:                                          # the implicit pad's reaction reaches the slab
        assert float(np.abs(ser["body_f"][flow.tables.cbody.numpy()[0, 0]]).max()) > 0.0
    for k in ser:
        scale = max(float(np.abs(ser[k]).max()), 1e-30)
        assert np.allclose(par[k], ser[k], rtol=2e-5, atol=2e-5 * scale), (k, np.abs(par[k] - ser[k]).max(), scale)
        assert np.array_equal(par[k], par2[k]), k


def test_runs_with_the_passes_track_the_serial_kernels():
    """Twelve periods with each: the film and the carts end within float32 drift of each other."""
    ends = []
    for parallel in (False, True):
        TS.PARALLEL = parallel
        try:
            flow, sim = plane_sim(12)
        finally:
            TS.PARALLEL = True
        ends.append((flow.h.numpy().copy(), sim.state.body_q.numpy().copy()))
    (h0, q0), (h1, q1) = ends
    assert np.allclose(h1, h0, rtol=0.0, atol=1e-6 * float(h0.max()))
    assert np.allclose(q1, q0, rtol=0.0, atol=1e-6)


def test_a_window_never_outgrows_its_launch():
    """The launch sizes hold every window the tools reach (the overflow flag stays down), and a revolution chart's
    tool may take its whole grid."""
    flow, _ = plane_sim(12)
    flow.tool_passes.check()
    assert all(m > 0 for m in flow.tool_passes.maxw)
    flow, _ = ring_sim(12)
    flow.tool_passes.check()
    nx, ny = flow.tables.shapes[0]
    assert flow.tool_passes.maxw == [nx * ny]
