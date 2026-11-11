# SPDX-License-Identifier: Apache-2.0
"""Surface-flow lumped bodies: drips from their shapes, species books, pressing, a full drop pool, checks (WP15's review
of 94ea57120; design note §4.9; WP9). CPU, harness."""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton

from alloy.plugins.errors import PluginError
from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow
from alloy.plugins.surface_flow.applicators import Applicator
from alloy.plugins.surface_flow.lumped_specs import Lumped, Pores

R, DT, G = 0.014, 2.0e-3, (0.0, 0.0, -9.81)
TOOL_HALF = (0.02, 0.04, 0.01)
DROPS = {"drop_radius": 1.5e-3}


def _sim(flow: SurfaceFlow, build, init=None, gravity=G, worlds: int = 1):
    return build_simulation(TestScene(build=build, init=init, ground=False, gravity=gravity, worlds=worlds, dt=DT,
                                      control_dt=DT), [flow], "cpu")


def _fixed(b, label: str, pos: tuple) -> int:
    xf = body_xform(*pos)
    body = b.add_link(xform=xf, label=label)
    b.add_articulation([b.add_joint_fixed(-1, body, parent_xform=xf)])
    return body


def _rest_and_handle(b) -> None:
    """A 0.4 m slab (top at z = 0), a tomato resting 0.5 mm above it, and a pan handle: a capsule 0.12 m off its
    body's origin along x (WP15's probe p2: x 0.06-0.18 m, lowest point at z = 0.09 m)."""
    b.add_shape_box(-1, xform=body_xform(0.0, 0.0, -0.01), hx=0.2, hy=0.2, hz=0.01, label="slab")
    b.add_shape_sphere(_fixed(b, "tomato", (-0.1, 0.0, R + 5.0e-4)), radius=R)
    q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi / 2.0)
    b.add_shape_capsule(_fixed(b, "pan", (0.0, 0.0, 0.10)), xform=wp.transform(wp.vec3(0.12, 0.0, 0.0), q),
                        radius=0.01, half_height=0.05, label="handle")
    d = b.add_body(xform=body_xform(0.5, 0.0, 1.0), label="dummy")                 # free; the harness needs one
    b.add_shape_sphere(d, radius=0.01)


@pytest.mark.compile_heavy
def test_a_resting_body_wets_the_board_and_an_offset_handle_drips_from_itself():
    """WP15's must-fix 1: drips start at a body's lowest point over its shapes at their world pose. A tomato resting
    0.5 mm over the slab's chart gives its shed water to the cell under it (lump_wet), none floored; a handle 0.12 m
    off its body's origin drips from under the handle (x 0.06-0.18 m) and its drops land on the slab, none floored
    (they started under the slab before). Books: shed = lump_wet + lump_pending + lump_drops; the film took lump_wet
    plus the landed drops."""
    flow = SurfaceFlow(charts=(PlaneChart("slab", cells=(20, 20)),), material="water", max_depth=3.0e-3,
                       drops=DROPS, lumped=(Lumped("tomato"), Lumped("pan", area=0.002)))
    q = 5.0e-9

    def init(sim) -> None:
        flow.lumped.inputs.vol.fill_(q)
    sim = _sim(flow, _rest_and_handle, init)
    xs = []
    for _ in range(150):
        sim.step_period()
        alive = flow.drops.pool.alive.numpy()[0] == 1
        xs += flow.drops.pool.pos.numpy()[0][alive][:, 0].tolist()
    r, d, f = flow.lumped.read(), flow.drops.read(), flow.read()
    shed = r["lump_shed"][0]
    assert shed[0] > 0.0 and shed[1] > 0.0 and d["floored_volume"][0] == 0.0
    assert d["lump_wet_volume"][0] > 0.0 and d["landed_volume"][0] > 0.0
    assert xs and 0.06 <= min(xs) and max(xs) <= 0.18                     # every drop under the handle
    assert shed.sum() == pytest.approx(d["lump_wet_volume"][0] + d["lump_pending_volume"][0]
                                       + d["lump_drops_volume"][0], rel=1e-12, abs=0.0)
    assert d["lump_drops_volume"][0] == pytest.approx(d["landed_volume"][0] + d["in_flight_volume"][0], rel=1e-12,
                                                      abs=0.0)
    assert f["supplied_volume"][0] + f["pending_volume"][0] == pytest.approx(
        d["lump_wet_volume"][0] + d["landed_volume"][0], rel=1e-12, abs=0.0)


def _sponge_scene(underside: float):
    def build(b) -> None:
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, -0.01), hx=0.1, hy=0.05, hz=0.01, label="plate")
        xf = body_xform(0.0, 0.0, underside + 0.002 + TOOL_HALF[2])
        sponge = b.add_link(xform=xf, label="sponge")
        b.add_articulation([b.add_joint_prismatic(-1, sponge, parent_xform=xf, axis=newton.Axis.X)])
        b.add_shape_box(sponge, hx=TOOL_HALF[0], hy=TOOL_HALF[1], hz=TOOL_HALF[2],
                        cfg=newton.ModelBuilder.ShapeConfig(density=2.0e6))
    return build


def _sponge_flow(**kw) -> tuple[SurfaceFlow, Pores]:
    pores = Pores(area=0.04 * 0.07, tool="sponge")
    pad = Applicator(body="sponge", shape="rect", size=(TOOL_HALF[0], 0.035), center=(0.0, 0.0, -TOOL_HALF[2] - 0.002),
                     stiffness=pores.modulus / pores.height, porous=True)
    return SurfaceFlow(charts=(PlaneChart("plate", cells=(40, 10)),), material="water", max_stride=1, tools=(pad,),
                       max_depth=3.0e-3, species=("soap",), lumped=(Lumped("sponge", pores=pores),), **kw), pores


@pytest.mark.compile_heavy
def test_a_sponge_takes_up_soap_at_the_films_concentration():
    """Species books, uptake (WP15's should-fix 3): an empty sponge whose porous pad rests 0.2 mm over a 0.5 mm film of
    3 kg/m³ soap absorbs it with its water (WP15's porous pads carry the cells' species), so the soap it took over the
    water it took is 3 kg/m³ and the film keeps 3 kg/m³ (1e-12; a half-share passed before). Its species books close
    (species = in + uptake + soil - shed - squeezed, abs=0), and the film's (with species_to_lumps) do too."""
    flow, _ = _sponge_flow()
    c, h0 = 3.0, 5.0e-4

    def init(sim) -> None:
        flow.set_depth(h0)
        flow.set_species(c * h0, "soap")
    sim = _sim(flow, _sponge_scene(2.0e-4), init)
    m0 = c * h0 * flow.tables.area_np.sum()
    for _ in range(10):
        sim.step_period()
    r, f = flow.lumped.read(), flow.read()
    up, sp_up = r["lump_uptake"][0, 0], r["lump_species_uptake"][0, 0, 0]
    h, m = flow.h.numpy()[0], flow.sp.mass.numpy()[0, :, 0]
    assert up > 0.0 and sp_up / up == pytest.approx(c, rel=1e-12, abs=0.0)
    wet = h > 1.0e-6
    assert np.allclose(m[wet] / h[wet], c, rtol=1e-12, atol=0.0)
    assert r["lump_species"][0, 0, 0] == pytest.approx(sp_up, rel=1e-12, abs=0.0)
    assert np.abs(r["lump_species_ledger"]).max() <= 1e-15 * m0
    assert f["species_ledger"][0, 0] == pytest.approx(m0, rel=1e-12, abs=0.0)    # with species_to_lumps


@pytest.mark.compile_heavy
def test_a_pressed_soapy_sponge_squeezes_its_soap_and_the_system_closes():
    """Species books, squeeze (WP15's should-fix 3): soapy water (5 kg/m³) streams into a pressed full sponge for 10
    exchanges; what is over its pores leaves onto the cells under its pad with the sponge's share of its soap, booked
    squeezed. The sponge's books close (abs=0), and the whole system does: film soap + its outflow + sponge soap =
    the stream's soap (the film's supplied soap is the squeezed soap; 1e-12)."""
    flow, pores = _sponge_flow()
    q, c = 2.0e-7, 5.0

    def init(sim) -> None:
        flow.lumped.set_water(pores.free)
        flow.lumped.inputs.vol.fill_(q)
        sp = flow.lumped.inputs.sp.numpy().copy()
        sp[0, 0, 0] = c * q
        flow.lumped.inputs.sp.assign(sp)
    sim = _sim(flow, _sponge_scene(-1.0e-3), init, gravity=(0.0, 0.0, 0.0))
    for _ in range(10):
        sim.step_period()
    flow.lumped.inputs.vol.zero_()
    flow.lumped.inputs.sp.zero_()
    for _ in range(3):                                                   # the last squeeze reaches the film
        sim.step_period()
    r, f = flow.lumped.read(), flow.read()
    sq, sp_in = r["lump_species_squeezed"][0, 0, 0], r["lump_species_in"][0, 0, 0]
    assert r["lump_squeezed"][0, 0] > 0.0 and sq > 0.0 and r["lump_shed"][0, 0] == 0.0
    assert np.abs(r["lump_species_ledger"]).max() <= 1e-15 * sp_in
    assert f["species_supplied"][0].sum(axis=0)[0] == pytest.approx(sq, rel=1e-12, abs=0.0)
    total = f["species_mass"][0, 0] + f["species_out"][0].sum(axis=0)[0] + r["lump_species"][0, 0, 0]
    assert total == pytest.approx(sp_in, rel=1e-12, abs=0.0)


@pytest.mark.compile_heavy
def test_a_hovering_sponge_sheds_and_does_not_squeeze():
    """WP15's should-fix 7: squeezing needs pressure, not cover. A full sponge whose pad hovers 2 mm over the plate
    (covering it, pressing nothing) sheds what a stream brings over its pores; nothing is squeezed onto the plate."""
    flow, pores = _sponge_flow()
    q = 2.0e-7

    def init(sim) -> None:
        flow.lumped.set_water(pores.free)
        flow.lumped.inputs.vol.fill_(q)
    sim = _sim(flow, _sponge_scene(2.0e-3), init, gravity=(0.0, 0.0, 0.0))
    for _ in range(5):
        sim.step_period()
    r, f = flow.lumped.read(), flow.read()
    assert flow.tf.cover.numpy().max() > 0.0 and flow.tf.press.numpy().max() == 0.0
    assert r["lump_squeezed"][0, 0] == 0.0 and r["lump_shed"][0, 0] == pytest.approx(5 * q, rel=1e-12, abs=0.0)
    assert f["supplied_volume"][0] == 0.0 and f["pending_volume"][0] == 0.0


@pytest.mark.compile_heavy
def test_a_full_drop_pool_keeps_the_drips_waiting_and_says_so(caplog):
    """WP15's should-fix 6: DC5's faucet on a 30 mm tomato sheds faster than a 16-drop pool can fly them. The drips
    wait (lump_pending, booked: shed = lump_wet + lump_pending + lump_drops exactly), none is lost, and read() warns
    once, naming the pool it needed."""
    from alloy.plugins.surface_flow.supply_stub import CHART, FAUCET, faucet_network, slab_scene  # noqa: PLC0415

    def scene(b) -> None:
        slab_scene()(b)
        b.add_shape_sphere(_fixed(b, "tomato", (0.0, 0.0, 0.12)), radius=0.03)
    flow = SurfaceFlow(charts=(CHART,), material="water", nozzles=(FAUCET,), network=faucet_network(),
                       max_depth=3.0e-3, drops={"drop_radius": 1.5e-3, "pool": 16}, lumped=(Lumped("tomato"),))
    sim = build_simulation(TestScene(build=scene, ground=True, worlds=1, dt=DT, control_dt=DT), [flow], "cpu")
    for _ in range(60):
        sim.step_period()
    caplog.set_level(logging.WARNING, logger="alloy.plugins.surface_flow")
    r, d = flow.lumped.read(), flow.drops.read()
    d2 = flow.drops.read()
    assert d["pool_full"][0] > 0 and d["lump_pending_volume"][0] > 0.0 and d["pool_need"][0] > 16
    assert r["lump_shed"][0, 0] == pytest.approx(d["lump_wet_volume"][0] + d["lump_pending_volume"][0]
                                                 + d["lump_drops_volume"][0], rel=1e-12, abs=0.0)
    warned = [rec for rec in caplog.records if "drop pool" in rec.getMessage()]
    assert len(warned) == 1 and f"pool={int(d['pool_need'][0])}" in warned[0].getMessage()
    assert d2["pool_full"][0] == d["pool_full"][0]


def test_pores_strain_refuses_a_nan_and_clamps_at_zero():
    """WP15's nits: the host strain is 0 for no or negative pressure and refuses a non-finite one."""
    p = Pores()
    assert p.strain(-5.0) == 0.0 and p.strain(0.0) == 0.0
    assert p.strain(1.0e9) == pytest.approx(p.q_stop / p.height)
    with pytest.raises(PluginError, match="finite"):
        p.strain(float("nan"))


@pytest.mark.parametrize("case", ["store", "shapes"])
def test_bad_lumps_are_refused(case):
    """WP15's nits: a sponge whose pad has a store (it would never set cover), and labelled shapes off the body."""
    from alloy.plugins.surface_flow.stores import Store  # noqa: PLC0415
    if case == "store":
        pores = Pores(area=0.04 * 0.07, tool="sponge")
        pad = Applicator(body="sponge", shape="rect", size=(TOOL_HALF[0], 0.035),
                         center=(0.0, 0.0, -TOOL_HALF[2] - 0.002), stiffness=1.0e3, store=Store(capacity=1.0e-5),
                         porous=True)
        flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(8, 4)),), material="water", tools=(pad,),
                           lumped=(Lumped("sponge", pores=pores),))
        build, words = _sponge_scene(2.0e-4), "has a store"
    else:
        flow = SurfaceFlow(charts=(PlaneChart("slab", cells=(8, 8)),), material="water",
                           lumped=(Lumped("tomato", shapes=("slab",)),))
        build, words = _rest_and_handle, "not all on its body"
    with pytest.raises(PluginError, match=words):
        _sim(flow, build)
