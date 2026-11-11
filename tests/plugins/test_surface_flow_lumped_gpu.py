# SPDX-License-Identifier: Apache-2.0
"""Surface-flow lumped films on the GPU (design §4.9; WP9; light-test rule: one smoke): the harness check with a lumped
body and the drop stage, then two exact laws on cuda:0 with every control period captured in a CUDA graph and
replayed: drops that carry their bodies' soap (two worlds, landed and floored), and sponges over a revolution chart's
θ seam. Each prints what it measured."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton

import alloy.plugins as ap
from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.surface_flow import PlaneChart, RevolutionChart, SurfaceFlow
from alloy.plugins.surface_flow.applicators import Applicator
from alloy.plugins.surface_flow.lumped_specs import Lumped, Pores

DEV, DT, R, RD = "cuda:0", 2.0e-3, 0.014, 1.5e-3
DROP = 4.0 / 3.0 * math.pi * RD ** 3
HOLD = 5.0e-5 * 0.01


def _fixed(b, label: str, xf) -> int:
    body = b.add_link(xform=xf, label=label)
    b.add_articulation([b.add_joint_fixed(-1, body, parent_xform=xf)])
    return body


def _dummy(b) -> None:
    d = b.add_body(xform=body_xform(0.5, 0.0, 1.0), label="dummy")
    b.add_shape_sphere(d, radius=0.01)


@pytest.mark.gpu
def test_harness_check_of_a_lumped_body_with_drops_on_gpu():
    flow = SurfaceFlow(species=("soap",), drops={"floor_z": -0.5}, lumped=(Lumped("dummy", area=0.01),))
    report = ap.check(flow, device=DEV, laws=False)
    print(report.summary())
    assert report.passed, report.summary()


def _shelf(b) -> None:
    b.add_shape_box(-1, xform=body_xform(0.0, 0.0, -0.01), hx=0.2, hy=0.2, hz=0.01, label="slab")
    b.add_shape_box(-1, xform=body_xform(0.0, 0.0, 0.02 + 2.0 * R + 0.01), hx=0.1, hy=0.1, hz=0.002, label="shelf")
    b.add_shape_sphere(_fixed(b, "tomato", body_xform(0.0, 0.0, 0.02 + R)), radius=R)
    b.add_shape_sphere(_fixed(b, "plum", body_xform(0.35, 0.0, 0.02 + R)), radius=R)
    _dummy(b)


@pytest.mark.gpu
def test_drops_carry_their_bodies_soap_in_a_cuda_graph():
    """The CPU shelf law (test_surface_flow_lumped_under) on cuda:0, each period a replayed graph: per world the
    tomato's 30 drops land on the slab and the plum's 10 floor, every body's soap closes (wet + waiting + dropped +
    tanks = shed), the landed soap is the film's at the tomato's concentration in every wet cell, the floored soap the
    plum's dropped (1e-12), and the shelf stays dry."""
    flow = SurfaceFlow(charts=(PlaneChart("slab", cells=(10, 10)), PlaneChart("shelf", cells=(10, 10))),
                       material="water", max_depth=5.0e-3, species=("soap",),
                       drops={"drop_radius": RD, "floor_z": -0.05},
                       lumped=(Lumped("tomato", area=0.01), Lumped("plum", area=0.01)))
    c = np.array([[6.0, 4.0], [2.0, 1.0]])
    water = np.array([HOLD + 30.2 * DROP, HOLD + 10.2 * DROP])

    def init(sim) -> None:
        flow.lumped.set_water(np.broadcast_to(water, (2, 2)))
        flow.lumped.set_species(c * water, "soap")
    sim = build_simulation(TestScene(build=_shelf, init=init, ground=False, gravity=(0.0, 0.0, -9.81), worlds=2, dt=DT,
                                     control_dt=DT), [flow], DEV, graph=True)
    for _ in range(80):
        sim.step_period()
    r, d, f = flow.lumped.read(), flow.drops.read(), flow.read()
    h, mass, A = flow.h.numpy(), flow.sp.mass.numpy()[:, :, 0], flow.tables.area_np
    for w in range(2):
        sp = {k: r[f"lump_species_{k}"][w, :, 0] for k in ("shed", "wet", "drip_pending", "dropped", "tanks")}
        film = f["species_supplied"][w].sum(axis=0)[0] + float((flow.sp.src.numpy()[w, :, 0] * A).sum())
        wet = h[w] > 0.0
        print(f"world {w}: landed {d['landed_volume'][w] / DROP:.6f} floored {d['floored_volume'][w] / DROP:.6f} "
              f"drops; soap film/landed - 1 {film / d['species_landed'][w, 0] - 1:+.1e}, cell c/{c[w, 0]} - 1 "
              f"{np.abs(mass[w][wet] / h[w][wet] / c[w, 0] - 1).max():.1e}")
        assert d["landed_volume"][w] == pytest.approx(30 * DROP, rel=1e-12, abs=0.0)
        assert d["floored_volume"][w] == pytest.approx(10 * DROP, rel=1e-12, abs=0.0)
        assert h[w][flow.tables.cells_of(1)].max() == 0.0
        np.testing.assert_allclose(sp["dropped"], c[w] * np.array([30.0, 10.0]) * DROP, rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(sp["shed"], sp["wet"] + sp["drip_pending"] + sp["dropped"] + sp["tanks"],
                                   rtol=1e-12, atol=0.0)
        assert film == pytest.approx(d["species_landed"][w, 0], rel=1e-12, abs=0.0)
        assert d["species_floored"][w, 0] == pytest.approx(sp["dropped"][1], rel=1e-12, abs=0.0)
        np.testing.assert_allclose(mass[w][wet] / h[w][wet], c[w, 0], rtol=1e-12, atol=0.0)


HZ, HX = 0.01, 0.01
SPOTS = (("seam", 0.115, 0.0), ("pi", 0.115, math.pi), ("inner", 0.02, 0.5 * math.pi))


def _sponges(b) -> None:
    for name, rc, th in SPOTS:
        xf = wp.transform(wp.vec3(rc * math.cos(th), rc * math.sin(th), -1.0e-3 + 0.002 + HZ),
                          wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), th))
        s = b.add_link(xform=xf, label=name)
        b.add_articulation([b.add_joint_fixed(-1, s, parent_xform=xf)])
        b.add_shape_box(s, hx=HX, hy=HX, hz=HZ, cfg=newton.ModelBuilder.ShapeConfig(density=2.0e6))
    _dummy(b)                                                   # a free body: the graphed state has no empty arrays


@pytest.mark.gpu
def test_sponges_at_the_seam_take_alike_in_a_cuda_graph():
    """The CPU seam law (test_surface_flow_lumped_seam) on cuda:0 with graphs: sponges pressing porous pads into a
    0.5 mm film at the rim over the θ seam and opposite it take alike, the inner whole-turn window closes, film +
    sponges = the start (1e-12)."""
    pores = Pores(area=4 * HX * HX)
    pads = tuple(Applicator(body=n, shape="rect", size=(HX, HX), center=(0.0, 0.0, -HZ - 0.002),
                            stiffness=pores.modulus / pores.height, porous=True, implicit=True) for n, _, _ in SPOTS)
    flow = SurfaceFlow(charts=(RevolutionChart(profile=((0.0125, 0.0), (0.12, 0.0)), n_theta=64, ds=2.0e-3),),
                       material="water", max_stride=1, tools=pads,
                       lumped=tuple(Lumped(n, pores=Pores(area=4 * HX * HX, tool=n)) for n, _, _ in SPOTS))
    sim = build_simulation(TestScene(build=_sponges, init=lambda sim: flow.set_depth(5.0e-4), ground=False,
                                     gravity=(0.0, 0.0, 0.0), worlds=2, dt=DT, control_dt=DT), [flow], DEV, graph=True)
    A = flow.tables.area_np
    start = (flow.h.numpy() * A).sum(axis=1)
    for _ in range(3):
        sim.step_period()
    r, f = flow.lumped.read(), flow.read()
    up = r["lump_uptake"]
    film = (flow.h.numpy() * A).sum(axis=1) + f["pending_volume"]
    print(f"uptake seam {up[0, 0]:.9e} pi {up[0, 1]:.9e} inner {up[0, 2]:.9e}; seam/pi - 1 "
          f"{np.abs(up[:, 0] / up[:, 1] - 1).max():.1e}; system - start "
          f"{(film + r['lump_water'].sum(axis=1) - start).tolist()}")
    assert up.min() > 0.0
    np.testing.assert_allclose(up[:, 0], up[:, 1], rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(up[0], up[1], rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(film + r["lump_water"].sum(axis=1), start, rtol=1e-12, atol=0.0)
