# SPDX-License-Identifier: Apache-2.0
"""Surface-flow lumped object films (design note §4.9; WP9): one well-mixed film per small body. CPU, harness.

The supply's LUMP hits (DC5) are hand-set in most laws: without a supply nothing clears ``lump_in``, so a value set once
enters every exchange (a steady stream). The last law runs DC5's real faucet on a tomato."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy._vendor import newton

from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow
from alloy.plugins.surface_flow.lumped_specs import Lumped, Pores
from alloy.plugins.surface_flow.process import Soil

R = 0.014                                                              # a cherry tomato, r 14 mm (GA's)
DT = 2.0e-3                                                            # one exchange per period


def _tomato_scene(b) -> None:
    b.add_shape_box(-1, xform=body_xform(0.0, 0.0, -0.01), hx=0.1, hy=0.1, hz=0.01, label="slab")
    tomato = b.add_body(xform=body_xform(0.0, 0.0, 0.1), label="tomato")
    b.add_shape_sphere(tomato, radius=R)


def _sim(flow: SurfaceFlow, build, init, gravity=(0.0, 0.0, 0.0)):
    return build_simulation(TestScene(build=build, init=init, ground=False, gravity=gravity, worlds=1, dt=DT,
                                      control_dt=DT), [flow], "cpu")


def _stream(flow: SurfaceFlow, vol: float, p: float = 0.0, share: float = 0.0, sp: tuple = ()):
    def init(sim) -> None:
        inp = flow.lumped.inputs
        inp.vol.fill_(vol)
        inp.p.fill_(p)
        inp.share.fill_(share)
        if sp:
            a = inp.sp.numpy().copy()
            a[0, 0, :len(sp)] = sp
            inp.sp.assign(a)
    return init


def test_a_tomato_holds_its_clinging_film_and_sheds_the_rest():
    """Water into a tomato (a sphere: area 4πr²) fills it to hold·area, then everything more is shed; the ledger
    (water = in - shed) is exact; the species leave with the water at its mixed concentration."""
    flow = SurfaceFlow(charts=(PlaneChart("slab", cells=(8, 8)),), material="water", species=("soap",),
                       lumped=(Lumped("tomato"),))
    q = 5.0e-9                                                         # m³ per exchange
    sim = _sim(flow, _tomato_scene, _stream(flow, q, sp=(q * 3.0,)))   # 3 kg/m³ of soap
    n = 100
    for _ in range(n):
        sim.step_period()
    r = flow.lumped.read()
    cap = 5.0e-5 * flow.lumped.lump_area.numpy()[0]                    # 4πr² of the model's (float32) radius
    assert cap == pytest.approx(5.0e-5 * 4.0 * math.pi * R ** 2, rel=1e-6, abs=0.0)
    assert n * q > 2.0 * cap
    assert r["lump_water"][0, 0] == pytest.approx(cap, rel=1e-12, abs=0.0)
    assert r["lump_shed"][0, 0] == pytest.approx(n * q - cap, rel=1e-12, abs=0.0)
    assert abs(r["lump_ledger"][0, 0]) <= 1e-12 * n * q
    assert r["lump_species"][0, 0, 0] == pytest.approx(3.0 * cap, rel=1e-12, abs=0.0)
    assert r["lump_species"][0, 0, 0] + r["lump_species_shed"][0, 0, 0] == pytest.approx(3.0 * n * q, rel=1e-12,
                                                                                          abs=0.0)


def test_dry_paste_on_a_tomato_erodes_from_the_first_hit():
    """A jet at p_stag 390 kPa (WP8's plate) on a quarter of a tomato coated with dried tomato paste (σ 200 kPa dry,
    150 Pa wet; t_soak 5.9 s): from the first exchange the soil falls at k_i·share·(p/σ_c - 1) with σ_c at the
    exchange's start hydration (soaking while wet), exactly; impact only; the removed soil joins the water."""
    soil = Soil(initial=0.2)
    flow = SurfaceFlow(charts=(PlaneChart("slab", cells=(8, 8)),), material="water", species=("soil",),
                       lumped=(Lumped("tomato", soil=soil),))
    sim = _sim(flow, _tomato_scene, _stream(flow, 1.0e-9, p=390.0e3, share=0.25))
    n = 30
    for _ in range(n):
        sim.step_period()
    hy = 1.0 - np.exp(-np.arange(n) * DT / soil.t_soak_at)
    sig = soil.sigma_dry * (1.0 - hy) ** 2 + soil.sigma_wet
    rate = soil.k_i * 0.25 * np.maximum(390.0e3 / sig - 1.0, 0.0) * DT
    removed = np.minimum(np.cumsum(rate), soil.initial)[-1]
    r = flow.lumped.read()
    area = flow.lumped.lump_area.numpy()[0]
    assert rate[0] > 0.0 and removed < soil.initial
    assert r["lump_removed_by_mass"][0, 0, 0] == pytest.approx(removed * area, rel=1e-12, abs=0.0)
    assert np.all(r["lump_removed_by_mass"][0, 0, 1:] == 0.0)
    assert r["lump_soil_mass"][0, 0] + removed * area == pytest.approx(soil.initial * area, rel=1e-12, abs=0.0)
    assert r["lump_species"][0, 0, 0] + r["lump_species_shed"][0, 0, 0] == pytest.approx(removed * area, rel=1e-12,
                                                                                          abs=0.0)


TOOL_HALF = (0.02, 0.04, 0.01)


def _sponge_scene(b, underside: float = -1.0e-3) -> None:
    """A plate (top at z = 0) and a sponge block on a slide along x, its pad underside at ``underside`` (1 mm into
    the plate by default)."""
    b.add_shape_box(-1, xform=body_xform(0.0, 0.0, -0.01), hx=0.1, hy=0.05, hz=0.01, label="plate")
    xf = body_xform(0.0, 0.0, underside + 0.002 + TOOL_HALF[2])
    sponge = b.add_link(xform=xf, label="sponge")
    b.add_articulation([b.add_joint_prismatic(-1, sponge, parent_xform=xf, axis=newton.Axis.X)])
    b.add_shape_box(sponge, hx=TOOL_HALF[0], hy=TOOL_HALF[1], hz=TOOL_HALF[2],
                    cfg=newton.ModelBuilder.ShapeConfig(density=2.0e6))


def test_a_pressed_sponge_squeezes_its_excess_beside_its_pad():
    """A full sponge (Pores: 0.9 of its 12 mm cellulose over the pad) pressed on a plate: its pad (stiffness E/H, 1 mm
    deep) carries p = 4.2 kPa, the strain solves p/E = ε + 3ε³, and the pores keep (1 - ε) of their volume; the rest
    leaves onto the ring of cells just outside the pad (uncovered, a neighbour covered), pro rata by area, none under
    it (WP15's ruling on JOB3's wring: a porous pad bears on the substrate; fails squeezing under the pad), booked
    squeezed by the sponge and supplied by the film."""
    from alloy.plugins.surface_flow.applicators import Applicator  # noqa: PLC0415
    pores = Pores(area=0.04 * 0.07, tool="sponge")
    pad = Applicator(body="sponge", shape="rect", size=(TOOL_HALF[0], 0.035), center=(0.0, 0.0, -TOOL_HALF[2] - 0.002),
                     stiffness=pores.modulus / pores.height, porous=True)
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(40, 10)),), material="water", max_stride=1, tools=(pad,),
                       lumped=(Lumped("sponge", pores=pores),))

    def init(sim) -> None:
        flow.lumped.set_water(pores.free)
    sim = _sim(flow, _sponge_scene, init)
    sim.step_period()                                                  # squeezed: pending until the next film step
    cov = flow.tf.cover.numpy()[0].reshape(40, 10) > 0.0
    nb = np.zeros_like(cov)
    nb[1:] |= cov[:-1]
    nb[:-1] |= cov[1:]
    nb[:, 1:] |= cov[:, :-1]
    nb[:, :-1] |= cov[:, 1:]
    ring = (nb & ~cov).ravel()
    src, A = flow.src_acc.numpy()[0], flow.tables.area_np
    out1 = flow.lumped.read()["lump_squeezed"][0, 0]
    assert cov.any() and ring.any() and out1 > 0.0 and src[cov.ravel()].max() == 0.0 and src[~ring].max() == 0.0
    np.testing.assert_allclose(src[ring], out1 / A[ring].sum(), rtol=1e-12, atol=0.0)
    sim.step_period()
    r, f = flow.lumped.read(), flow.read()
    press = flow.tf.press.numpy()[0]
    cover = flow.tf.cover.numpy()[0].astype(np.float64)
    A = flow.tables.area_np
    p = float((press * cover * A).sum() / (cover * A).sum())
    eps = float(pores.strain(p))
    assert p == pytest.approx(pores.modulus / pores.height * 1.0e-3, rel=1e-5) and 0.05 < eps < 0.15
    assert r["lump_eps"][0, 0] == pytest.approx(eps, rel=1e-12, abs=0.0)
    assert r["lump_water"][0, 0] == pytest.approx(pores.free * (1.0 - r["lump_eps"][0, 0]), rel=1e-12, abs=0.0)
    out = pores.free - r["lump_water"][0, 0]
    assert r["lump_squeezed"][0, 0] == pytest.approx(out, rel=1e-12, abs=0.0) and r["lump_shed"][0, 0] == 0.0
    assert f["supplied_volume"][0] + f["pending_volume"][0] == pytest.approx(out, rel=1e-12, abs=0.0)
    assert f["film_volume"][0] > 0.0 and abs(f["ledger"][0]) <= 1e-12 * out


def _small_plate(b) -> None:
    """A 30 mm plate (top at z = 0) under the sponge's 40 x 70 mm pad, 1 mm into it."""
    b.add_shape_box(-1, xform=body_xform(0.0, 0.0, -0.01), hx=0.015, hy=0.015, hz=0.01, label="plate")
    xf = body_xform(0.0, 0.0, -1.0e-3 + 0.002 + TOOL_HALF[2])
    sponge = b.add_link(xform=xf, label="sponge")
    b.add_articulation([b.add_joint_prismatic(-1, sponge, parent_xform=xf, axis=newton.Axis.X)])
    b.add_shape_box(sponge, hx=TOOL_HALF[0], hy=TOOL_HALF[1], hz=TOOL_HALF[2],
                    cfg=newton.ModelBuilder.ShapeConfig(density=2.0e6))


def test_a_pad_over_the_whole_chart_squeezes_under_itself():
    """WP15's round-6 S1: a full sponge pressed onto a plate its pad covers whole has no ring of cells outside the pad
    on that chart, so the squeeze falls back onto the cells under it, by cover·area: they hold what it squeezed
    (1e-12; it was lost when the fallback was dropped)."""
    from alloy.plugins.surface_flow.applicators import Applicator  # noqa: PLC0415
    pores = Pores(area=0.04 * 0.07, tool="sponge")
    pad = Applicator(body="sponge", shape="rect", size=(TOOL_HALF[0], 0.035), center=(0.0, 0.0, -TOOL_HALF[2] - 0.002),
                     stiffness=pores.modulus / pores.height, porous=True)
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(6, 6)),), material="water", max_stride=1, tools=(pad,),
                       lumped=(Lumped("sponge", pores=pores),))
    sim = _sim(flow, _small_plate, lambda sim: flow.lumped.set_water(pores.free))
    sim.step_period()
    cover = flow.tf.cover.numpy()[0].astype(np.float64)
    src, A = flow.src_acc.numpy()[0], flow.tables.area_np
    out = flow.lumped.read()["lump_squeezed"][0, 0]
    assert (cover > 0.0).all() and out > 0.0
    np.testing.assert_allclose(src, out * cover / (cover * A).sum(), rtol=1e-12, atol=0.0)


def _two_tomatoes(b) -> None:
    b.add_shape_box(-1, xform=body_xform(0.0, 0.0, -0.01), hx=0.1, hy=0.1, hz=0.01, label="slab")
    for name, x in (("strong", -0.05), ("weak", 0.05)):
        t = b.add_body(xform=body_xform(x, 0.0, 0.1), label=name)
        b.add_shape_sphere(t, radius=R)


def test_the_sanitizer_clock_on_a_lump_counts_only_at_c_min():
    """The process stage's Sanitizer (c_min 50 ppm) on lumps, wet from the first exchange (water over H_WET·area, as a
    chart cell's h > H_WET): a tomato rinsed at 1.01 c_min counts every exchange (expo = n·dt, exactly); one at
    0.99 c_min never starts (WP15: a clock at 1.5 c_min passed the earlier 100 vs 10 ppm law)."""
    from alloy.plugins.surface_flow.process import Sanitizer  # noqa: PLC0415
    flow = SurfaceFlow(charts=(PlaneChart("slab", cells=(8, 8)),), material="water", species=("sanitizer",),
                       process={"sanitizer": Sanitizer(c_min=0.05)},
                       lumped=(Lumped("strong"), Lumped("weak")))
    q = 1.0e-8                                                          # > H_WET·4πr² = 4.9e-9 m³

    def init(sim) -> None:
        inp = flow.lumped.inputs
        inp.vol.fill_(q)
        sp = inp.sp.numpy().copy()
        sp[0, :, 0] = (1.01 * 0.05 * q, 0.99 * 0.05 * q)
        inp.sp.assign(sp)
    sim = _sim(flow, _two_tomatoes, init)
    n = 25
    for _ in range(n):
        sim.step_period()
    r = flow.lumped.read()
    assert q > flow.lumped.spec.wet.numpy()[0]
    assert r["lump_expo"][0].tolist() == [pytest.approx(n * DT, rel=1e-12, abs=0.0), 0.0]
    assert r["lump_expo_best"][0, 0] == r["lump_expo"][0, 0]


def _fixed_tomato_scene(b) -> None:
    b.add_shape_box(-1, xform=body_xform(0.0, 0.0, -0.01), hx=0.1, hy=0.1, hz=0.01, label="slab")
    xf = body_xform(0.0, 0.0, 0.06)
    tomato = b.add_link(xform=xf, label="tomato")
    b.add_articulation([b.add_joint_fixed(-1, tomato, parent_xform=xf)])
    b.add_shape_sphere(tomato, radius=R)
    dummy = b.add_body(xform=body_xform(0.5, 0.0, 1.0), label="dummy")            # free; the harness needs one
    b.add_shape_sphere(dummy, radius=0.01)


def test_a_full_tomato_drips_through_the_drop_stage():
    """WP15's note: a lump's shed water leaves through the drop stage when it is configured. Drops of the stage's
    volume fall from below the tomato (2r apart), land on the slab's chart, and the books close exactly: the lump's
    shed = its pending drips + the drops made; the drops made = landed + in flight."""
    flow = SurfaceFlow(charts=(PlaneChart("slab", cells=(20, 20)),), material="water", max_depth=3.0e-3,
                       drops={"drop_radius": 1.5e-3}, lumped=(Lumped("tomato"),))
    q = 5.0e-9
    sim = _sim(flow, _fixed_tomato_scene, _stream(flow, q), gravity=(0.0, 0.0, -9.81))
    n = 200
    for _ in range(n):
        sim.step_period()
    r, d = flow.lumped.read(), flow.drops.read()
    made = d["to_pool_volume"][0]
    assert made > 0.0 and d["landed_volume"][0] > 0.0
    assert r["lump_shed"][0, 0] == pytest.approx(r["lump_drip_pending"][0, 0] + made, rel=1e-12, abs=0.0)
    assert r["lump_drip_pending"][0, 0] < 4.0 / 3.0 * math.pi * 1.5e-3 ** 3 + q    # + this exchange's (drips next)
    assert made == pytest.approx(d["landed_volume"][0] + d["in_flight_volume"][0], rel=1e-12, abs=0.0)
    assert r["lump_water"][0, 0] + r["lump_shed"][0, 0] == pytest.approx(n * q, rel=1e-12, abs=0.0)


def test_a_sponge_on_a_film_takes_it_up():
    """G5's capillary uptake: an empty sponge resting in a 0.5 mm film (its porous pad 0.2 mm above the plate) absorbs
    the film its pad would plough (WP15's porous pads): most in its first exchange (the film above its underside), then
    what flows back under it; the film's
    loss is the sponge's gain, both books exact (the film's volume plus ``to_lumps``), and it never holds more than
    its pores."""
    from alloy.plugins.surface_flow.applicators import Applicator  # noqa: PLC0415
    pores = Pores(area=0.04 * 0.07, tool="sponge")
    pad = Applicator(body="sponge", shape="rect", size=(TOOL_HALF[0], 0.035), center=(0.0, 0.0, -TOOL_HALF[2] - 0.002),
                     stiffness=pores.modulus / pores.height, porous=True)
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(40, 10)),), material="water", max_stride=1, tools=(pad,),
                       max_depth=2.0e-3, lumped=(Lumped("sponge", pores=pores),))
    sim = _sim(flow, lambda b: _sponge_scene(b, 2.0e-4), lambda sim: flow.set_depth(5.0e-4), gravity=(0.0, 0.0, -9.81))
    vol0 = 5.0e-4 * flow.tables.area_np.sum()
    sim.step_period()
    first = flow.lumped.read()["lump_uptake"][0, 0]
    for _ in range(49):
        sim.step_period()
    r, f = flow.lumped.read(), flow.read()
    assert 0.0 < first < r["lump_uptake"][0, 0]                        # it keeps taking as the film flows back
    assert r["lump_water"][0, 0] <= pores.free * (1.0 - r["lump_eps"][0, 0]) and r["lump_squeezed"][0, 0] == 0.0
    assert r["to_lumps_volume"][0] == pytest.approx(r["lump_uptake"][0, 0], rel=1e-12, abs=0.0)
    assert r["lump_water"][0, 0] == pytest.approx(r["lump_uptake"][0, 0], rel=1e-12, abs=0.0)
    assert f["film_volume"][0] + f["pending_volume"][0] + r["to_lumps_volume"][0] == pytest.approx(vol0, rel=1e-12,
                                                                                                     abs=0.0)
    assert f["ledger"][0] == pytest.approx(vol0, rel=1e-12, abs=0.0)                   # host_api adds to_lumps
    assert f["lump_uptake"][0, 0] == r["lump_uptake"][0, 0]



def test_uptake_on_a_strided_film_matches_every_exchange():
    """WP15's condition: on a film stepping every k exchanges the sponge takes up once per film step at its dt, and
    1 - exp(-k·dt/τ) is exactly k exchange takes compounded. On a frozen film under a pad pressing 0.5 mm into the plate
    (the stage driven by hand), 8 takes at stride 1 and one at stride 8 leave the same film and fill the sponge alike
    (1e-12); between film steps nothing is taken."""
    from alloy.plugins.surface_flow.applicators import Applicator  # noqa: PLC0415
    pores = Pores(area=0.04 * 0.07, tool="sponge")
    pad = Applicator(body="sponge", shape="rect", size=(TOOL_HALF[0], 0.035), center=(0.0, 0.0, -TOOL_HALF[2] - 0.002),
                     stiffness=pores.modulus / pores.height, porous=True)
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(40, 10)),), material="water", max_stride=1, tools=(pad,),
                       max_depth=2.0e-3, lumped=(Lumped("sponge", pores=pores),))
    sim = _sim(flow, lambda b: _sponge_scene(b, -5.0e-4), lambda sim: flow.set_depth(5.0e-4))
    sim.step_period()                                                  # the pad's field over the cells
    flow.tf.absorb.zero_()                                             # what its pores took in that step
    flow.tf.absorb_sp.zero_()
    out = {}
    for k in (1, 8):
        flow.set_depth(5.0e-4)
        flow.lumped.set_water(0.0)
        flow.lumped.state.ledger.zero_()
        flow.stride = k
        taken = []
        for t in range(1, 9):
            flow.tick.fill_(t)
            flow.lumped.after_film(sim.state, DT)
            taken.append(flow.lumped.read()["lump_uptake"][0, 0])
        out[k] = (flow.h.numpy()[0].copy(), taken)
    (h1, t1), (h8, t8) = out[1], out[8]
    assert t8[:7] == [0.0] * 7 and t1[0] > 0.0                          # strided: only on the film step
    assert t8[-1] == pytest.approx(t1[-1], rel=1e-12, abs=0.0)
    assert np.allclose(h8, h1, rtol=1e-12, atol=0.0)


def _faucet_tomato(b) -> None:
    from alloy.plugins.surface_flow.supply_stub import slab_scene  # noqa: PLC0415
    slab_scene()(b)
    xf = body_xform(0.0, 0.0, 0.12)
    t = b.add_link(xform=xf, label="tomato")
    b.add_articulation([b.add_joint_fixed(-1, t, parent_xform=xf)])
    b.add_shape_sphere(t, radius=0.03)


@pytest.mark.compile_heavy
def test_a_real_faucet_on_a_tomato_erodes_from_its_first_hit():
    """DC5's faucet (supply_stub) 0.18 m over a fixed 30 mm tomato coated with soil that is hard dry (σ 1 kPa) and soft
    wet (10 Pa): the supply's LUMP hits bring the jet's water, stagnation pressure and share, and from the first
    exchange that hits it the soil falls by k_i·share·(p/σ_c - 1)·dt at that exchange's starting strength (soaking
    while wet), exactly, every exchange; the supply's lumped ledger equals the water the tomato took in, and the
    tomato's own ledger closes."""
    from alloy.plugins.surface_flow.supply_stub import CHART, FAUCET, faucet_network  # noqa: PLC0415
    soil = Soil(initial=0.2, sigma_dry=1000.0, sigma_wet=10.0, t_soak=5.9, k_e=0.0)
    flow = SurfaceFlow(charts=(CHART,), material="water", nozzles=(FAUCET,), network=faucet_network(),
                       max_depth=3.0e-3, lumped=(Lumped("tomato", soil=soil),))
    sim = build_simulation(TestScene(build=_faucet_tomato, ground=True, worlds=1, dt=DT, control_dt=DT), [flow],
                           "cpu")
    area = flow.lumped.lump_area.numpy()[0]
    hits = 0
    for _ in range(40):
        before = flow.lumped.read()
        sim.step_period()
        r = flow.lumped.read()
        p, share = float(flow.lump_p.numpy()[0, 0]), float(flow.lump_share.numpy()[0, 0])
        if p <= 0.0:
            continue
        sigma = soil.sigma_dry * (1.0 - before["lump_hyd"][0, 0]) ** 2 + soil.sigma_wet
        expect = soil.k_i * share * max(p / sigma - 1.0, 0.0) * DT
        got = (r["lump_removed_by_mass"][0, 0, 0] - before["lump_removed_by_mass"][0, 0, 0]) / area
        assert got == pytest.approx(expect, rel=1e-12, abs=0.0)
        assert expect > 0.0 or hits > 0                                   # the first hit already erodes
        hits += 1
    r = flow.lumped.read()
    assert hits >= 30 and r["lump_in_volume"][0, 0] > 0.0
    assert flow.supply.read()["lumped"][0] == pytest.approx(r["lump_in_volume"][0, 0], rel=1e-12, abs=0.0)
    assert abs(r["lump_ledger"][0, 0]) <= 1e-12 * r["lump_in_volume"][0, 0]
