# SPDX-License-Identifier: Apache-2.0
"""Surface flow, a scraping store's overflow and lifted gob (stores_lift.py; GA's carrying knife, coupling.py:922-938):
a full blade's cut beyond capacity goes to ``tool_overflow`` for WP13's end bands, and a lifted blade keeps only what
its paste's yield stress holds (CPU)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.plugins.errors import PluginError
from alloy.plugins.surface_flow import FilmMaterial
from alloy.plugins.surface_flow.stores import Store
from alloy.plugins.surface_flow.stores_lift import DROP_STICK
from tests.plugins.test_surface_flow_stores import VISCOUS, roll, rolled_over

G = 9.81
PASTE = FilmMaterial("paste_200Pa", density=2000.0, viscosity=10.0, yield_stress=200.0, surface_tension=0.0)
CARRY = 1.0e-5                                                  # m³, the blade's start load


def _blade(**kw) -> Store:
    return Store(capacity=CARRY, retained=0.0, held=1.0, split=1.0, pickup=0.0, drip=0.0, initial=1.0,
                 scrape=True, **kw)


# the underside's outward normal (body frame) and its stick share max(0.02, 0.5·(1 + up)), in float32 as the kernel
SHARES = {"level": ((0.0, 0.0, -1.0), 1.0), "tilted": ((0.6, 0.0, -0.8), float(np.float32(0.5) * np.float32(1.8))),
          "overturned": ((0.0, 0.0, 1.0), float(np.float32(0.02)))}


@pytest.mark.parametrize("pose", list(SHARES))
@pytest.mark.parametrize("material", [PASTE, VISCOUS], ids=["paste", "newtonian"])
def test_a_lifted_blade_keeps_what_its_yield_stress_holds(pose, material):
    """5 cm over a dry plate (touching nothing, no load on its pad) a blade carrying 10 ml keeps
    0.5·A·tau_y/(rho·g·6)·max(0.02, 0.5·(1 + up)) and drops the rest at once (``tool_dropped``), with any pending
    overflow. A Newtonian film (tau_y = 0) all falls. The books close exactly."""
    normal, share = SHARES[pose]
    pending = 2.0e-6

    def setup(flow) -> None:
        flow.tool_overflow.fill_(pending)
    flow, sim = roll(_blade(), 0.0, 0.05, periods=3, blade_gap=0.05, speed=0.0, material=material,
                     gravity=(0.0, 0.0, -G), tool_kw={"normal": normal}, setup=setup)
    area = flow.config.tools[0].footprint
    g = float(np.float32(G))                                     # the model's gravity is float32
    stick = 0.5 * area * material.yield_stress / (material.density * g * DROP_STICK) * share
    r = flow.read()
    load, dropped = float(r["tool_load"][0].sum()), float(r["tool_dropped"][0].sum())
    rel = 1.0e-12 if pose != "tilted" else 1.0e-6                # float32 normal: up = 0.8 to an ulp
    assert load == pytest.approx(stick, rel=rel, abs=0.0)
    assert dropped == pytest.approx(CARRY - stick + pending, rel=1.0e-12, abs=0.0)
    assert float(r["tool_overflow"][0].sum()) == 0.0
    assert abs(float(r["store_ledger"][0]) - (CARRY + pending)) <= 1.0e-12 * CARRY
    if material is PASTE and pose == "level":
        assert stick > 1.0e-7                                    # the paste holds a real share: 0.51 ml


def test_a_blades_stick_area_replaces_its_footprint():
    """SC9's putty knife bears on a 10 mm band but holds paste on its whole 51 x 70 mm face (GA: the sole's
    4·hx·hy, coupling.py:933): ``Store(stick_area=)`` sets the area its lifted stick limit uses."""
    face = 6.0e-3                                                # m², 10x the 0.6 cm² footprint
    flow, sim = roll(_blade(stick_area=face), 0.0, 0.05, periods=2, blade_gap=0.05, speed=0.0, material=PASTE,
                     gravity=(0.0, 0.0, -G))
    g = float(np.float32(G))
    stick = 0.5 * face * PASTE.yield_stress / (PASTE.density * g * DROP_STICK)
    assert float(flow.read()["tool_load"][0].sum()) == pytest.approx(stick, rel=1.0e-12, abs=0.0)
    with pytest.raises(PluginError, match="stick_area"):
        Store(capacity=1.0e-6, scrape=True, split=1.0, stick_area=-1.0)


def test_a_blade_on_the_film_drops_nothing():
    """A blade riding 0.3 mm over 0.4 mm of film with no load on its pad is not lifted: its roll rests on the film."""
    flow, sim = roll(_blade(), 4.0e-4, 0.05, periods=3, blade_gap=3.0e-4, speed=0.0, material=PASTE,
                     gravity=(0.0, 0.0, -G))
    assert float(flow.read()["tool_dropped"][0].sum()) == 0.0


@pytest.mark.parametrize("over_on", [0, 1])
def test_a_full_blade_sends_its_cut_beyond_capacity_to_the_overflow(over_on):
    """A full blade (WP13's carrying-knife end spill) 0.3 mm over 0.4 mm of soapy film. With ``overflow_on`` it
    still cuts everything above its underside and the load stays at capacity; the cut beyond capacity goes to
    ``tool_overflow`` (pending), WP13's end bands (laid, ``tool_roll_spill``) or off the chart (``tool_overflow_off``:
    this 4-cell strip has no room for the bands), its species at the film's concentration. Off, it cuts only what
    fits: nothing."""
    g, h0, cap, m0 = 3.0e-4, 4.0e-4, 1.0e-6, 2.0e-4                 # m, m, m³, kg/m² of soap

    def setup(flow) -> None:
        flow.overflow_on = over_on
        flow.set_species(np.full(flow.h.shape, m0), "soap")
    store = Store(capacity=cap, retained=0.0, held=1.0, split=1.0, pickup=0.0, drip=0.0, initial=1.0, scrape=True)
    flow, sim = roll(store, h0, 0.05, periods=100, blade_gap=g, flow_kw={"species": ("soap",)}, setup=setup)
    r = flow.read()
    area = flow.tables.area_np
    start = h0 * float(area.sum())
    over = float(r["tool_overflow"][0].sum())
    got = rolled_over(flow, -0.1, sim)
    assert abs(float(r["ledger"][0]) - start) <= 1.0e-12 * start
    assert abs(float(r["store_ledger"][0]) - cap) <= 1.0e-12 * cap
    assert abs(float(r["species_ledger"][0, 0]) - m0 * float(area.sum())) <= 1.0e-12 * m0 * float(area.sum())
    if not over_on:
        assert over == 0.0 and np.array_equal(flow.h.numpy(), np.full(flow.h.shape, h0))
        return
    assert got.size >= 40                                                     # 1 cm swept
    assert np.allclose(got, g, rtol=0.0, atol=2.0e-8), np.abs(got - g).max()     # the float32 pose's jitter
    assert float(r["tool_load"][0].sum()) == pytest.approx(cap, rel=1.0e-12, abs=0.0)
    laid, off = float(r["tool_roll_spill"][0].sum()), float(r["tool_overflow_off"][0].sum())
    cut = float(r["to_tools_volume"][0]) + laid                              # the lay takes laid back from to_tools
    assert cut > 0.25 * cap
    assert over + laid + off == pytest.approx(cut, rel=1.0e-12, abs=0.0)
    soap = float(flow.tool_overflow_sp.numpy()[0, 0, 0] + r["tool_overflow_off_sp"][0, 0, 0])
    soap += float(r["species_supplied"][0, :, 0].sum())                     # laid, booked as supplied
    assert soap == pytest.approx(cut * m0 / h0, rel=1.0e-12, abs=0.0)        # the cut's concentration


def test_a_lifted_gob_leaves_through_the_drip_channel():
    """With a supply the gob (and the pending overflow) leaves through DC5's stage-drips channel, as releases do."""
    from alloy.plugins.surface_flow.supply import Drain  # noqa: PLC0415

    kw = {"drains": (Drain("grate", chart=0, center=(0.12, 0.0, 0.0), radius=0.01, to="sink"),),
          "network": {"nodes": [{"name": "sink", "medium": "tank", "volume": 0.0, "area": 0.01}], "edges": []}}

    def setup(flow) -> None:
        flow.tool_overflow.fill_(2.0e-6)
    flow, sim = roll(_blade(), 0.0, 0.05, periods=1, blade_gap=0.05, speed=0.0, material=PASTE,
                     gravity=(0.0, 0.0, -G), flow_kw=kw, setup=setup)
    r = flow.read()
    dropped = float(r["tool_dropped"][0].sum())
    staged = r.get("supply_staged")
    sent = float(staged[0]) if staged is not None else float(flow.src_drip_v.numpy().sum())
    assert dropped > 1.0e-5 and sent == pytest.approx(dropped, rel=1.0e-12, abs=0.0)


@pytest.mark.parametrize("gap, fills", [(0.0, True), (3.0e-5, True), (8.0e-5, False)])
def test_a_blade_bears_within_gas_on_base_tolerance(gap, fills):
    """SC9's knives: at exactly zero gap over a dry board, float32 decided whether a blade touched, so hole fills ran
    only in some exchanges. A blade within 0.05 mm of the base bears on it (GA's on_base, coupling.py:478) and fills
    a groove beside dry faces to their level every exchange; 0.08 mm over, it touches nothing."""
    rows = np.arange(120 * 4) % 4                                    # across the plate: face, face, groove, face

    def base(flow) -> np.ndarray:
        return np.where(rows == 2, -2.0e-3, 0.0).reshape(1, -1)
    flow, sim = roll(_blade(), 0.0, 0.05, periods=40, blade_gap=gap, base=base, material=PASTE)
    x = flow.cell_centres(0)[:, 0, 0]
    x1 = float(sim.state.body_q.numpy()[0][0])
    done = np.repeat((x > -0.1 + 0.005) & (x < x1 - 0.005), 4)
    h = flow.h.numpy()[0]
    assert done.sum() >= 32
    assert (h[done & (rows != 2)] == 0.0).all()                       # the faces stay bare
    if fills:
        assert np.allclose(h[done & (rows == 2)], 2.0e-3, rtol=1e-6, atol=0.0)
    else:
        assert (h == 0.0).all()
    assert abs(float(flow.read()["store_ledger"][0]) - CARRY) <= 1.0e-12 * CARRY


def lift_scene(x0: float, z0: float):
    """A static plate (top at z = 0) and a blade on an x-then-z slide."""
    from alloy._vendor import newton  # noqa: PLC0415

    from alloy.plugins.harness.scene import body_xform  # noqa: PLC0415

    def build(b) -> None:
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, -0.01), hx=0.15, hy=0.025, hz=0.01, label="plate")
        xf = body_xform(x0, 0.0, z0)
        cart = b.add_link(xform=xf, label="cart")
        b.add_shape_sphere(cart, radius=0.005, cfg=newton.ModelBuilder.ShapeConfig(density=1.0e5))
        blade = b.add_link(xform=xf, label="carriage")
        b.add_shape_sphere(blade, radius=0.005, cfg=newton.ModelBuilder.ShapeConfig(density=1.0e5))
        sx = b.add_joint_prismatic(-1, cart, parent_xform=xf, axis=newton.Axis.X)
        sz = b.add_joint_prismatic(cart, blade, axis=newton.Axis.Z)
        b.add_articulation([sx, sz])
    return build


@pytest.mark.parametrize("lift", [0.1, 0.01])
def test_a_rising_blade_keeps_its_carry(lift):
    """WP6's margin trowel: a blade that cut a strip and lifts away laid its carry back under itself while any part of
    it still met the film: a soft-edged leading cell stood above its underside (fast lift), and inside the 0.05 mm
    bearing tolerance it refilled the cut cells to its rising underside (slow lift, under 0.05 mm per exchange).
    Lifting, it lays nothing: the load it cut stays on it (no gravity here, so no gob)."""
    from alloy._vendor import newton  # noqa: PLC0415

    from alloy.plugins.harness.scene import TestScene, build_simulation  # noqa: PLC0415
    from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow  # noqa: PLC0415
    from alloy.plugins.surface_flow.applicators import Applicator  # noqa: PLC0415

    g, h0, z0 = 2.0e-3, 1.0e-2, 0.05                           # a 10 mm paste: the blade cuts 8 mm
    tool = Applicator(body="carriage", shape="rect", size=(0.005, 0.03), center=(0.0, 0.0, g - z0), stiffness=1.0e3,
                      store=Store(capacity=1.0e-3, retained=0.0, held=1.0, split=1.0, pickup=0.0, drip=0.0,
                                  scrape=True))
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(120, 4)),), material=PASTE, inertia=False,
                       max_depth=2.0e-2, tools=(tool,))

    def drive(sim, vx: float, vz: float) -> None:
        sim.state.joint_qd.assign(np.asarray([vx, vz], np.float32))

    def init(sim) -> None:
        flow.set_depth(h0)
        drive(sim, 0.05, 0.0)
        newton.eval_fk(sim.model, sim.state.joint_q, sim.state.joint_qd, sim.state)
    sim = build_simulation(TestScene(build=lift_scene(-0.1, z0), init=init, ground=False, gravity=(0.0, 0.0, 0.0),
                                     worlds=1, dt=2.0e-3), [flow], "cpu")
    for _ in range(40):
        sim.step_period()
    cut = float(flow.read()["tool_load"][0].sum())
    drive(sim, 0.0, lift)
    for _ in range(60):
        sim.step_period()
    r = flow.read()
    assert lift * flow._exchange_dt < 5.0e-5 or lift > 0.05               # the slow lift: inside the tolerance
    assert float(sim.state.body_q.numpy()[1][2]) > z0 + 0.1 * lift          # lifted
    assert cut > 1.0e-8
    assert float(r["tool_load"][0].sum()) == pytest.approx(cut, rel=1.0e-12, abs=0.0)


def test_a_blade_in_a_trays_heap_is_not_lifted():
    """WP6's trowel in the tub: a scraping blade below a tray's level rests in its heap (GA: touching), so it loads
    toward capacity and drops nothing, though it touches no film and its stick limit is under its load."""
    from alloy.plugins.harness.scene import TestScene, build_simulation  # noqa: PLC0415
    from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow  # noqa: PLC0415
    from alloy.plugins.surface_flow.applicators import Applicator  # noqa: PLC0415
    from alloy.plugins.surface_flow.stores import Tray  # noqa: PLC0415
    from tests.plugins.test_surface_flow_stores import NX, R, roller_scene  # noqa: PLC0415

    cap, vol = 2.0e-6, 2.0e-4
    blade = Applicator(body="cover", shape="disc", size=(0.02, 0.02), center=(0.0, 0.0, -R), stiffness=1.0e3,
                       store=Store(capacity=cap, retained=0.0, held=1.0, split=1.0, pickup=2.0, drip=0.0,
                                   scrape=True))
    tray = Tray(center=(0.3, 0.0, -0.05), half=(0.05, 0.05), volume=vol)      # beside the plate: no film under it
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(NX, 4)),), material=PASTE, inertia=False,
                       max_depth=1.0e-3, tools=(blade,), trays=(tray,))
    sim = build_simulation(TestScene(build=roller_scene(-0.05 + R + 0.01, 0.3), ground=False,
                                     gravity=(0.0, 0.0, -G), worlds=1, dt=2.0e-3), [flow], "cpu")
    for _ in range(25):
        sim.step_period()
    r = flow.read()
    assert flow.stores is not None and float(r["tool_dropped"][0].sum()) == 0.0
    assert float(r["tool_load"][0].sum()) > 0.5 * cap > 0.5 * float(flow.stores.stores.stick.numpy()[0])


@pytest.mark.parametrize("case", ["bare_mesh_cells", "on_the_support"])
def test_a_knife_fills_up_to_the_support_it_rides(case):
    """DC5's review of SC9's drywall: a putty knife riding a mesh's fibre tops at 0.5 mm over a 0.15 mm mesh base.
    ``bare_mesh_cells``: amid compound at 0.5 mm, a few bare mesh cells capped the whole fill at their 0.15 mm base and
    never filled; they fill to the knife. ``on_the_support``: over a bare mesh with ``support`` = the fibre tops (GA's
    support_height), the knife bears on it and fills the mesh and a groove up to it."""
    gap, mesh = 5.0e-4, 1.5e-4
    rows = np.arange(120 * 4) % 4
    cols = np.repeat(np.arange(120), 4)

    def base(flow) -> np.ndarray:
        return np.where(rows == 2, -2.0e-3, mesh).reshape(1, -1)

    def depth(flow) -> np.ndarray:
        if case == "on_the_support":
            return np.zeros((1, 480))
        h = np.where(rows == 2, 2.0e-3 + gap, gap - mesh)
        return np.where((cols % 7 == 3) & (rows != 2), 0.0, h).reshape(1, -1)     # some bare mesh cells

    def setup(flow) -> None:
        if case == "on_the_support":
            flow.set_input("support", np.where(rows == 2, -2.0e-3, gap).astype(np.float32).reshape(1, -1))
    flow, sim = roll(_blade(), depth, 0.05, periods=40, blade_gap=gap, base=base, material=PASTE, setup=setup)
    x = flow.cell_centres(0)[:, 0, 0]
    x1 = float(sim.state.body_q.numpy()[0][0])
    done = np.repeat((x > -0.1 + 0.005) & (x < x1 - 0.005), 4)
    surface = flow.h.numpy()[0] + base(flow)[0]
    assert done.sum() >= 32
    assert np.allclose(surface[done], gap, rtol=0.0, atol=2.0e-8), np.abs(surface[done] - gap).max()
    r = flow.read()
    assert abs(float(r["store_ledger"][0]) - CARRY) <= 1.0e-12 * CARRY
