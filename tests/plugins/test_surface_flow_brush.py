# SPDX-License-Identifier: Apache-2.0
"""Surface flow, a metering brush (stores_meter.py) and a pointed round tip: GA's brush law h_m = L_c·sqrt(C·eta·U/w),
laid and picked up toward it while sliding; a pointed tip's band grows with its press (CPU)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy._vendor import newton

from alloy.plugins.errors import PluginError
from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.surface_flow import FilmMaterial, PlaneChart, SurfaceFlow
from alloy.plugins.surface_flow.applicators import Applicator
from alloy.plugins.surface_flow.stores import Store

LATEX = FilmMaterial("latex_1Pas", density=1300.0, viscosity=1.0, surface_tension=0.0)
LENGTH, WIDTH, Z, U, K = 0.1, 0.04, 0.05, 0.05, 1.0e6      # plate [m]; the carriage's height, speed; pad [Pa/m]


def carriage_scene(x0: float):
    """A static plate (top face at z = 0, the chart) and a carriage at height Z sliding along x."""
    def build(b: newton.ModelBuilder) -> None:
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, -0.01), hx=0.5 * LENGTH, hy=0.5 * WIDTH, hz=0.01,
                        label="plate")
        xf = body_xform(x0, 0.0, Z)
        cart = b.add_link(xform=xf, label="brush")
        b.add_shape_sphere(cart, radius=0.005, cfg=newton.ModelBuilder.ShapeConfig(density=1.0e5))
        b.add_articulation([b.add_joint_prismatic(-1, cart, parent_xform=xf, axis=newton.Axis.X)])
    return build


def stroke(tool: Applicator, cells: tuple[int, int], periods: int, depth=0.0, speed: float = U, x0: float = -0.03,
           base=None, mask=None, max_depth: float = 1.0e-3):
    """Slide ``tool`` along x; ``depth``, ``base`` and ``mask`` (paste_mask) are values or functions of the flow."""
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=cells),), material=LATEX, inertia=False, max_depth=max_depth,
                       tools=(tool,))

    def init(sim) -> None:
        if base is not None:
            flow.set_input("base", base(flow).astype(np.float32))
        if mask is not None:
            flow.set_input("paste_mask", mask(flow).astype(np.float32))
        flow.set_depth(depth(flow) if callable(depth) else depth)
        sim.state.joint_qd.assign(np.full(sim.state.joint_qd.shape, speed, np.float32))
        newton.eval_fk(sim.model, sim.state.joint_q, sim.state.joint_qd, sim.state)
    sim = build_simulation(TestScene(build=carriage_scene(x0), init=init, ground=False, gravity=(0.0, 0.0, 0.0),
                                     worlds=1, dt=2.0e-3), [flow], "cpu")
    for _ in range(periods):
        sim.step_period()
    return flow, sim


def brush(store: Store, press: float, shape: str = "rect", size=(0.004, 0.006), slope: float = 0.0) -> Applicator:
    """A brush on the carriage, its underside (or tip) ``press`` into the plate; frictionless, so it keeps its speed."""
    return Applicator(body="brush", shape=shape, size=size, center=(0.0, 0.0, -Z - press), stiffness=K,
                      friction=0.0, store=store, tip_slope=slope)


def loaded(meter: float = 0.012, meter_max: float = 150e-6) -> Store:
    return Store(capacity=2.0e-6, retained=0.0, held=1.0, initial=1.0, drip=0.0, pickup=0.0, meter=meter,
                 meter_max=meter_max)


def swept(flow, sim, x0: float = -0.03) -> np.ndarray:
    """The cells the brush slid fully over, [columns, rows]."""
    x = flow.cell_centres(0)[:, 0, 0]
    x1 = float(sim.state.body_q.numpy()[0][0])
    return flow.depth(0)[0][(x > x0 + 0.006) & (x < x1 - 0.006)]


def test_a_sliding_brush_lays_gas_metered_film():
    """GA's law: a flat brush 0.5 mm into a dry plate, sliding at U, lays h_m = L_c·sqrt(C·eta·U/w) on what it
    slid over (L_c its contact area over its width, w its normal force over its width): to 5e-5, as U comes from
    float32 poses 6e-5 m apart. 4x faster lays 2x thicker, and the books close."""
    cells, out = (80, 32), {}
    for v in (U, 4.0 * U):
        flow, sim = stroke(brush(loaded(meter=0.003), 5.0e-4), cells, periods=round(20 * U / v), speed=v)  # 2 cm
        dy = WIDTH / cells[1]
        y = flow.cell_centres(0)[0, :, 1]
        on = np.abs(y) < 0.006 + 0.5 * dy                         # rows the pad's 12 mm width touches (cover > 0)
        wc = float(y[on].max() - y[on].min()) + dy
        area = 4.0 * 0.004 * 0.006                                # the pad's soft-edged area sums to its footprint
        force = float(flow.tool_normal.numpy()[0, 0])
        want = (area / wc) * math.sqrt(0.003 * LATEX.viscosity * v * wc / force)    # under the 150 um cap
        hm = float(flow.read()["tool_hm"][0, 0])
        assert force == pytest.approx(K * 5.0e-4 * area, rel=1e-5)
        assert hm == pytest.approx(want, rel=5e-5), (hm, want)
        laid = swept(flow, sim)[:, on]
        assert laid.size >= 20 and np.allclose(laid, hm, rtol=5e-5, atol=0.0), (laid.min(), laid.max(), hm)
        r = flow.read()
        assert abs(float(r["store_ledger"][0]) - 2.0e-6) <= 1e-12 * 2.0e-6
        assert abs(float(r["ledger"][0])) <= 1e-12 * 2.0e-6         # film + what the tools took (laid: < 0)
        out[v] = hm
    assert out[4.0 * U] / out[U] == pytest.approx(2.0, rel=5e-5)


def test_the_metered_film_is_capped_and_a_still_brush_lays_nothing():
    """h_m is capped at meter_max (GA's 150 um), and a brush that doesn't slide meters nothing (GA: no tip slid)."""
    flow, sim = stroke(brush(loaded(meter=50.0), 5.0e-4), (80, 32), periods=20)
    assert float(flow.read()["tool_hm"][0, 0]) == 150e-6
    flow, sim = stroke(brush(loaded(), 5.0e-4), (80, 32), periods=5, speed=0.0)
    assert float(flow.h.numpy().max()) == 0.0


def test_a_brush_over_a_thick_film_picks_it_back_to_h_m():
    """Over 0.4 mm of wet film, more than h_m, an empty brush takes the film back down to h_m where it slides, into
    the room it has; the store gains exactly what the film lost."""
    store = Store(capacity=2.0e-6, retained=0.0, held=1.0, initial=0.0, drip=0.0, pickup=0.0, meter=0.012)
    flow, sim = stroke(brush(store, 5.0e-4), (80, 32), periods=20, depth=4.0e-4)
    hm = float(flow.read()["tool_hm"][0, 0])
    assert 0.0 < hm < 4.0e-4
    y = flow.cell_centres(0)[0, :, 1]
    on = np.abs(y) < 0.006 + 0.5 * WIDTH / 32
    assert np.allclose(swept(flow, sim)[:, on], hm, rtol=5e-5, atol=0.0)
    r = flow.read()
    assert float(r["tool_load"][0].sum()) == pytest.approx(float(r["to_tools_volume"][0]), rel=1e-12, abs=0.0)
    assert float(r["tool_load"][0].sum()) > 0.0


@pytest.mark.parametrize("press", [1.0e-3, 2.0e-3])
def test_a_pointed_tip_lays_a_band_as_wide_as_its_press(press):
    """A round brush's tip rising 1·rho from its point (tip_slope 1), level and pressed d into the plate, touches a
    disc of radius d: its stroke lays a band 2d wide (GA: a round's line grows with its press), to a cell."""
    cells = (100, 80)                                             # 1 mm x 0.5 mm cells
    flow, sim = stroke(brush(loaded(), press, shape="disc", size=(0.006, 0.006), slope=1.0), cells, periods=20)
    band = swept(flow, sim)
    rows = (band > 0.0).sum(axis=1)                               # wet rows per swept column
    dy = WIDTH / cells[1]
    assert rows.size >= 6
    assert np.all(np.abs(rows * dy - 2.0 * press) <= 2.0 * dy), (rows, 2.0 * press / dy)


def test_tip_slope_and_meter_are_checked():
    with pytest.raises(PluginError, match="tip_slope"):
        Applicator(body="brush", shape="rect", tip_slope=1.0)
    with pytest.raises(PluginError, match="roller"):
        Applicator(body="brush", shape="roller", size=(0.02, 0.03), store=loaded())
    with pytest.raises(PluginError, match="meter"):
        Store(capacity=1.0e-6, meter=-1.0)


def test_meter_viscosity_replaces_the_films():
    """GA meters at the paint's high-shear viscosity (brush.py:386), not a card's low-shear one: with
    meter_viscosity = 0.25 Pa·s on a 1 Pa·s film, h_m is half what the film's viscosity gives."""
    hm = []
    for eta in (0.0, 0.25):
        store = Store(capacity=2.0e-6, retained=0.0, held=1.0, initial=1.0, drip=0.0, pickup=0.0, meter=0.003,
                      meter_viscosity=eta)
        flow, sim = stroke(brush(store, 5.0e-4), (80, 32), periods=10)
        hm.append(float(flow.read()["tool_hm"][0, 0]))
    assert hm[1] / hm[0] == pytest.approx(0.5, rel=5e-5), hm


def test_a_sealant_nose_fills_the_joint_up_to_its_own_profile():
    """GA's tooling nose (coupling.py:561-567, 580-584): a scraping roller, r 11 mm, sliding along a 12 mm joint with
    its bottom 2.1 mm below the faces, fills every joint cell under it up to its own profile, up to both walls, and
    cuts what stands above it. A 4 mm-low stretch is refilled to the nose's arc, and the faces (paste_mask 0) stay
    bare. SOFT-2's tool-sealant. The nose starts half full (15 mL, as much load as before, and as much room): a full
    nose would spill its cut beyond capacity onto the faces at its ends (GA: "a loaded sealant tool overflows over the
    joint's edges until it is wiped", WP13's end bands), and this test is about the profile fill, not the overflow."""
    r, half_joint, bottom, cells = 0.011, 0.006, -2.1e-3, (100, 60)
    store = Store(capacity=3.0e-5, retained=0.0, held=1.0, split=1.0, bins=1, initial=0.5, drip=0.0, pickup=0.0,
                  scrape=True, fill="profile")
    nose = Applicator(body="brush", shape="roller", size=(r, 0.009), axis=(1.0, 0.0, 0.0),
                      center=(0.0, 0.0, bottom + r - Z), stiffness=1.0e3, friction=0.0, store=store)

    def yx(flow):
        c = flow.cell_centres(0).reshape(-1, 3)
        return c[:, 1], c[:, 0]

    def base(flow):
        y, _ = yx(flow)
        return np.where(np.abs(y) < half_joint, -8.5e-3, 0.0).reshape(1, -1)

    def mask(flow):
        y, _ = yx(flow)
        return (np.abs(y) < half_joint).astype(np.float64).reshape(1, -1)

    def depth(flow):                                            # full (top 1 mm low), then sagged (top 4 mm low)
        y, x = yx(flow)
        return np.where(np.abs(y) < half_joint, np.where(x < -0.015, 7.5e-3, 4.5e-3), 0.0).reshape(1, -1)
    flow, sim = stroke(nose, cells, periods=60, depth=depth, base=base, mask=mask, max_depth=1.0e-2)
    y, x = yx(flow)
    x1 = float(sim.state.body_q.numpy()[0][0])
    h, b = flow.h.numpy()[0], base(flow)[0]
    done = (x > -0.03 + 0.010) & (x < x1 - 0.010)                # passed by the 18 mm nose
    joint = done & (np.abs(y) < half_joint)
    arc = bottom + r - np.sqrt(np.maximum(r * r - y * y, 0.0))  # the nose's underside across the joint
    assert joint.sum() >= 40 and (done & (x > -0.015) & (np.abs(y) < half_joint)).sum() >= 20
    assert np.allclose((b + h)[joint], arc[joint], rtol=0.0, atol=2.0e-8), np.abs((b + h - arc)[joint]).max()
    assert (h[done & (np.abs(y) >= half_joint)] == 0.0).all()             # the faces stay bare
    r_ = flow.read()
    area = flow.tables.area_np
    start = float((depth(flow)[0] * area).sum())
    assert abs(float(r_["ledger"][0]) - start) <= 1e-12 * start
    assert abs(float(r_["store_ledger"][0]) - 1.5e-5) <= 1e-12 * 1.5e-5



def test_a_hovering_metering_brush_drips():
    """A metering brush touches only where its underside meets the film (no free-load reach: DC6 review): held
    0.1 mm over a dry plate with a 20 mm 'nap' of free load, it drips above held like any lifted store."""
    store = Store(capacity=2.0e-6, retained=0.0, held=0.5, initial=1.0, drip=1.0, pickup=0.0, meter=0.012)
    flow, sim = stroke(brush(store, -1.0e-4), (80, 32), periods=5, speed=0.0)
    steps, h = int(flow.tick.numpy()[0]), flow._exchange_dt
    want = 0.5 * 2.0e-6 * (1.0 - math.exp(-1.0 * h) ** steps)
    assert steps > 5
    assert float(flow.read()["tool_dripped"][0].sum()) == pytest.approx(want, rel=1.0e-12, abs=0.0)
