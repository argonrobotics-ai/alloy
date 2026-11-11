# SPDX-License-Identifier: Apache-2.0
"""Surface-flow sponges on a revolution chart's θ seam (WP15's review of round 6, must-fix M1 and should-fix S2): a
pad's window runs past column 0 or ny there (tool_geometry.rev_window), so the sponge kernels walk it wrapped (cell_at).
CPU, harness, after WP15's probe p4."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton

from alloy.plugins.harness.scene import TestScene, build_simulation
from alloy.plugins.surface_flow import RevolutionChart, SurfaceFlow
from alloy.plugins.surface_flow.applicators import Applicator
from alloy.plugins.surface_flow.lumped_specs import Lumped, Pores

DT, HZ, HX, NT = 2.0e-3, 0.01, 0.01, 64
SPOTS = (("seam", 0.115, 0.0), ("pi", 0.115, math.pi), ("inner", 0.02, 0.5 * math.pi))    # (name, radius, θ)
PORES = Pores(area=4 * HX * HX)


def _build(spots):
    def build(b) -> None:
        for name, rc, th in spots:
            xf = wp.transform(wp.vec3(rc * math.cos(th), rc * math.sin(th), -1.0e-3 + 0.002 + HZ),
                              wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), th))       # its pad 1 mm into the disc
            s = b.add_link(xform=xf, label=name)
            b.add_articulation([b.add_joint_fixed(-1, s, parent_xform=xf)])
            b.add_shape_box(s, hx=HX, hy=HX, hz=HZ, cfg=newton.ModelBuilder.ShapeConfig(density=2.0e6))
    return build


def _flow(spots, depth: float, water: float) -> tuple[SurfaceFlow, object]:
    pads = tuple(Applicator(body=n, shape="rect", size=(HX, HX), center=(0.0, 0.0, -HZ - 0.002),
                            stiffness=PORES.modulus / PORES.height, porous=True, implicit=True) for n, _, _ in spots)
    lumps = tuple(Lumped(n, pores=Pores(area=4 * HX * HX, tool=n)) for n, _, _ in spots)
    flow = SurfaceFlow(charts=(RevolutionChart(profile=((0.0125, 0.0), (0.12, 0.0)), n_theta=NT, ds=2.0e-3),),
                       material="water", max_stride=1, tools=pads, lumped=lumps)

    def init(sim) -> None:
        flow.set_depth(depth)
        flow.lumped.set_water(water)
    sim = build_simulation(TestScene(build=_build(spots), init=init, ground=False, gravity=(0.0, 0.0, 0.0), worlds=2,
                                     dt=DT, control_dt=DT), [flow], "cpu")
    return flow, sim


@pytest.mark.compile_heavy
def test_sponges_take_up_alike_at_the_seam_and_opposite_it():
    """WP15's M1: three empty sponges press their porous pads into a 0.5 mm film on a disc, one at the rim over the θ
    seam, one at the rim opposite it, one at the inner edge (its window spans the whole turn), in two worlds. The seam
    sponge takes what the opposite one does (1e-12; 6.6 % went missing at the seam before), every world alike, and the
    film lost = the sponges' uptake = to_lumps, film + sponges = the start (1e-12; the inner window read the cells
    before the world's row, and the other world's, before)."""
    flow, sim = _flow(SPOTS, 5.0e-4, 0.0)
    A = flow.tables.area_np
    start = (flow.h.numpy() * A).sum(axis=1)
    for _ in range(3):
        sim.step_period()
    r, f = flow.lumped.read(), flow.read()
    up = r["lump_uptake"]
    assert up.min() > 0.0
    np.testing.assert_allclose(up[:, 0], up[:, 1], rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(up[0], up[1], rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(r["to_lumps_volume"], up.sum(axis=1), rtol=1e-12, atol=0.0)
    film = (flow.h.numpy() * A).sum(axis=1) + f["pending_volume"]
    np.testing.assert_allclose(film + r["lump_water"].sum(axis=1), start, rtol=1e-12, atol=0.0)
    assert (flow.lumped.film_out.absorb.numpy() == 0.0).all()                       # nothing absorbed is left behind


@pytest.mark.compile_heavy
def test_a_full_sponge_at_the_seam_squeezes_onto_its_ring_across_it():
    """WP15's S2: a full sponge pressed into a dry disc over the θ seam squeezes onto the ring of cells just outside its
    pad: the cells written are the host's ring (uncovered, a 4-neighbour covered, columns wrapped: on both sides of
    the seam), all at one depth (pro rata by area: the cells' areas differ along the radius; sharing by count spread
    the depth 46 %, without the wrap half the ring went unwritten), and they hold what it squeezed (1e-12)."""
    flow, sim = _flow(SPOTS[:1], 0.0, PORES.free)
    sim.step_period()
    nx, ny = flow.tables.shapes[0]
    A = flow.tables.area_np
    out = flow.lumped.read()["lump_squeezed"][:, 0]
    assert out.min() > 0.0
    for w in range(2):
        cov = flow.tf.cover.numpy()[w].reshape(nx, ny) > 0.0
        nb = np.zeros_like(cov)
        nb[1:] |= cov[:-1]
        nb[:-1] |= cov[1:]
        nb |= np.roll(cov, 1, axis=1) | np.roll(cov, -1, axis=1)               # θ wraps
        ring = (nb & ~cov).ravel()
        src = flow.src_acc.numpy()[w]
        assert cov[:, 0].any() and cov[:, -1].any()                               # the pad straddles the seam
        assert np.array_equal(src > 0.0, ring)
        np.testing.assert_allclose(src[ring], out[w] / A[ring].sum(), rtol=1e-12, atol=0.0)
        assert (src * A).sum() == pytest.approx(out[w], rel=1e-12, abs=0.0)
