# SPDX-License-Identifier: Apache-2.0
"""Surface-flow film core (phase 1): harness check and laws, config round trips, step bound, moving frames,
revolution charts (CPU)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton

import alloy.plugins as ap
from alloy.plugins.base import make_config
from alloy.plugins.errors import PluginError
from alloy.plugins.harness.check import fresh
from alloy.plugins.harness.run import run_scene
from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.surface_flow import FilmMaterial, PlaneChart, RevolutionChart, SurfaceFlow
from alloy.plugins.surface_flow import laws as sl

G = 9.81
WATER = FilmMaterial("water_flat", surface_tension=0.0)


LAWS = ("drainage", "falling_film", "spreading", "bingham_slump", "dam_break", "levelling", "contact_angle", "runup",
        "drop_2d", "ledger")


@pytest.mark.compile_heavy
def test_harness_quick_check_passes():
    """The harness's generic checks; the laws run one per test below (together they take minutes on a CPU)."""
    report = ap.check(SurfaceFlow(), quick=True, laws=False)
    assert report.passed, report.summary()
    assert tuple(law.name for law in SurfaceFlow().laws()) == LAWS              # a new law gets its own test


@pytest.mark.compile_heavy
@pytest.mark.parametrize("name", LAWS)
def test_law(name):
    flow = SurfaceFlow()
    law = next(law for law in flow.laws() if law.name == name)
    run = run_scene(law.scene or flow.test_scene(), [fresh(flow, dict(law.config or {}))], "cpu")
    assert run.error is None, run.error
    res = law.check(run)
    assert res.passed, f"{name}: {res.value:.4g} vs bound {res.bound:.4g}: {res.detail} {res.fix}"


def test_config_survives_dicts_and_copies():
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(8, 4)),
                               RevolutionChart(profile=((0.04, 0.1), (0.04, 0.0)), n_theta=8)),
                       material=FilmMaterial("paint", density=1300.0, viscosity=0.2))
    copy = fresh(flow, {"u_cap": 1.0})                                    # the harness copies via asdict
    assert copy.config.charts == flow.config.charts and copy.config.material == flow.config.material
    cfg = make_config(SurfaceFlow.Config, {"charts": [{"shape": "plate", "cells": (4, 4)}], "material": "latex_paint"},
                      {})
    assert isinstance(cfg.charts[0], PlaneChart)
    with pytest.raises(PluginError, match="chart"):
        make_config(SurfaceFlow.Config, {"charts": [{"nothing": 1}]}, {})
    assert not flow.has_supply and not flow.applies_forces                  # no supply: no apply hook
    assert SurfaceFlow(nozzles=({"name": "tap", "edge": "spout"},)).applies_forces


def _plate(on_body: bool, rot: tuple[float, ...] = (0.0, 0.0, 0.0, 1.0)):
    """A 0.2 x 0.05 plate box (top face = the chart), static or on a free body, plus a free dummy."""
    def build(b: newton.ModelBuilder) -> None:
        xf = wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat(*rot))
        if on_body:
            body = b.add_body(xform=xf, label="carrier")
            b.add_shape_box(body, hx=0.1, hy=0.025, hz=0.01, label="plate")
        else:
            b.add_shape_box(-1, xform=xf, hx=0.1, hy=0.025, hz=0.01, label="plate")
        dummy = b.add_body(xform=body_xform(1.0, 0.0, 1.0), label="dummy")
        b.add_shape_sphere(dummy, radius=0.01)
    return build


def _run(build, gravity, periods: int = 10, **config):
    cfg = {"charts": (PlaneChart("plate", cells=(40, 1), edges=("open", "wall", "wall", "wall")),),
           "material": WATER, "max_depth": 2.0e-3}
    cfg.update(config)
    flow = SurfaceFlow(**cfg)

    def init(sim):
        x = flow.cell_centres(0)[:, 0, 0]
        flow.set_depth((1.0e-3 + 5.0e-4 * np.cos(math.pi * x / 0.1))[:, None], chart=0)
    scene = TestScene(build=build, init=init, ground=False, gravity=gravity, periods=periods, worlds=1)
    run = run_scene(scene, [flow], "cpu")
    assert run.error is None, run.error
    return flow, run


def test_free_falling_plate_carries_its_film_unchanged():
    gravity = (-3.0, 0.0, -G)
    static, _ = _run(_plate(False), gravity)
    falling, run = _run(_plate(True), gravity)
    moved = float(np.abs(static.depth(0)[0] - static.depth(0)[0][::-1]).max())    # the static film flowed
    assert moved > 1.0e-5
    start = 1.0e-3 + 5.0e-4 * np.cos(math.pi * falling.cell_centres(0)[:, 0, 0] / 0.1)
    drift = float(np.abs(falling.depth(0)[0][:, 0] - start).max())
    assert drift < 1.0e-3 * moved              # weightless: nothing flows (float32 g - a leaves ~1e-6 g)
    assert float(run.trace.body_qd[-1, 0, 2]) < -0.5                            # the carrier did fall


def test_turned_plate_drains_like_tilted_gravity():
    """A plate turned 90° about y (its x axis pointing down) under normal gravity = a flat plate under gravity
    along +x."""
    s = math.sin(math.pi / 4.0)
    turned, _ = _run(_plate(False, (0.0, s, 0.0, s)), (0.0, 0.0, -G))
    tilted, _ = _run(_plate(False), (G, 0.0, 0.0))
    a, b = turned.depth(0)[0], tilted.depth(0)[0]
    assert np.abs(a - b).max() < 1.0e-6 * float(b.max())
    assert turned.read()["spilled_volume"][0] == 0.0                            # it drains away from the open x- edge


def test_check_dt_refuses_too_many_substeps():
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(400, 1)),), material=WATER, max_depth=0.05,
                       max_substeps=4)
    scene = TestScene(build=_plate(False), ground=False, periods=1, worlds=1)
    run = run_scene(scene, [flow], "cpu")
    assert run.error is not None and "film substeps" in run.error


def test_revolution_chart_drains_down_a_cylinder_symmetrically():
    """Water on the outside of a vertical cylinder (profile run downward, open at the bottom) drains down, stays
    symmetric in θ, and closes its ledger."""
    chart = RevolutionChart(profile=((0.03, 0.1), (0.03, 0.0)), n_theta=12, ds=2.5e-3, ends=("wall", "open"))
    flow = SurfaceFlow(charts=(chart,), material=FilmMaterial("oil", density=970.0, viscosity=0.1,
                                                              surface_tension=0.0), max_depth=1.5e-3)

    def init(sim):
        flow.set_depth(1.0e-3)
    scene = TestScene(build=sl.plate_scene(0.02, 0.02), init=init, ground=False, periods=25, worlds=2)
    run = run_scene(scene, [flow], "cpu")
    assert run.error is None, run.error
    h = flow.depth(0)
    assert np.abs(h - h[:, :, :1]).max() < 1.0e-12                              # every θ column alike
    assert h[0, 0].mean() < h[0, -1].mean() < 1.0e-3                            # thinner on top, draining
    r = flow.read()
    assert (r["spilled_volume"] > 0.0).all()
    assert np.abs(r["ledger"] - r["ledger"][0]).max() < 1e-15 and abs(r["ledger"][0] - 1.0e-3 * 2 * math.pi * 0.03
                                                                         * 0.1) < 1e-12


def test_set_depth_and_source_validate():
    flow, _ = _run(_plate(False), (0.0, 0.0, -G), periods=1)
    with pytest.raises(PluginError, match="depth"):
        flow.set_depth(-1.0)
    with pytest.raises(PluginError, match="sinks are drains"):
        flow.set_source(-1.0e-4)                                               # WP2 review M1
    with pytest.raises(PluginError, match="chart"):
        flow.set_depth(0.0, chart=3)
    with pytest.raises(PluginError, match="does not fit"):
        flow.set_depth(np.zeros((40, 1)))                                      # a chart grid without chart=
    flow.set_depth(np.full((1, 40, 1), 2e-4), chart=0)
    assert np.allclose(flow.depth(0), 2e-4, rtol=0.0, atol=0.0)
    flow.set_source(np.ones((40, 1)) * 1e-4, chart=0)
    assert np.allclose(flow.src_fixed.numpy(), 1e-4, rtol=0.0, atol=0.0)


def test_film_arriving_at_rest_slows_the_flow_by_momentum():
    """A uniform source S into a uniform film moving at u0 (no friction, no slope): momentum h·u is kept, so
    u = u0·h0/(h0 + S·t)."""
    h0, u0, S = 1.0e-3, 0.4, 2.0e-3
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(40, 1)),), material=FilmMaterial("inviscid", viscosity=1e-9,
                                                                                           surface_tension=0.0),
                       max_depth=3.0e-3)

    def init(sim):
        flow.set_depth(h0)
        flow.set_source(S)
        u = flow.u.numpy()
        u[:, 1:40] = u0                                     # the interior x-faces
        flow.u.assign(u)
    scene = TestScene(build=_plate(False), init=init, ground=False, periods=4, worlds=1)    # before the walls' waves
    run = run_scene(scene, [flow], "cpu")                                                       # reach the middle
    assert run.error is None, run.error
    t = float(run.trace.t[-1])
    want = u0 * h0 / (h0 + S * t)
    got = float(flow.u.numpy()[0, 20])
    assert abs(got / want - 1.0) < 1.0e-3, (got, want)


def test_supply_wired_faucet_fills_the_plate_and_drains_into_the_sink():
    """DC5's supply inside SurfaceFlow: a faucet (network edge) jets onto a slab whose grate drains into a sink tank.
    Both ledgers close and the drained volume is what the tank received."""
    from tests.plugins.test_surface_flow_supply import CHART, FAUCET, faucet_network, slab_scene  # noqa: PLC0415
    from alloy.plugins.surface_flow.supply import Drain  # noqa: PLC0415

    grate = Drain("grate", chart=0, center=(0.12, 0.0, 0.0), radius=0.03, to="sink")
    flow = SurfaceFlow(charts=(CHART,), material="water", nozzles=(FAUCET,), drains=(grate,),
                       network=faucet_network(), max_depth=3.0e-3)
    scene = TestScene(build=slab_scene(), ground=True, worlds=2, periods=25)
    run = run_scene(scene, [flow], "cpu")
    assert run.error is None, run.error
    r = flow.read()
    sent, swept = r["supply_sent"], r["supply_swept"]
    assert (sent > 1.0e-5).all()                                             # about 0.1 L/s for 0.5 s
    cross = sent + swept - (r["supplied_volume"] + r["pending_volume"] + r["supply_to_tank"] + r["supply_lost"]
                            + r["supply_overspray"] + r["supply_mist"] + r["supply_caught"] + r["supply_lumped"])
    assert np.abs(cross).max() < 1.0e-12 * sent.max()
    assert np.abs(r["ledger"] - r["ledger"][0]).max() < 1.0e-12 * sent.max() and abs(r["ledger"][0]) < 1e-12
    assert np.allclose(r["drained_total_volume"], r["supply_drained"], rtol=1e-12, atol=1e-15)
    assert (r["film_volume"] > 0.0).all() and (r["drained_total_volume"] > 0.0).all()
    assert np.abs(r["h"][0] - r["h"][1]).max() == 0.0                        # identical worlds


def test_slow_film_strides_and_matches_every_exchange():
    """Latex paint on a vertical plate with a steady source: the film steps every k exchanges (k > 1) with the
    summed inflow, matches stepping every exchange to 1e-3, and the ledger stays exact."""
    def run(max_stride: int):
        flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(40, 1), edges=("open", "wall", "wall", "wall")),),
                           material="latex_paint", inertia=False, max_depth=3.0e-4, max_stride=max_stride)

        def init(sim):
            flow.set_depth(1.5e-4)
            rate = np.zeros((40, 1))
            rate[-3:] = 2.0e-5                                 # 20 µm/s into the top cells
            flow.set_source(rate, chart=0)
        scene = TestScene(build=_plate(False), init=init, ground=False, gravity=(-G, 0.0, 0.0), periods=8, worlds=1,
                          dt=1.25e-3)
        run = run_scene(scene, [flow], "cpu")
        assert run.error is None, run.error
        return flow
    fast, slow = run(1), run(16)
    assert fast.stride == 1 and slow.stride == 16 and slow.n_sub == 1
    a, b = fast.depth(0)[0], slow.depth(0)[0]
    assert np.abs(a - b).max() < 1.0e-3 * a.max()
    r = slow.read()
    start = 1.5e-4 * slow.tables.area_np.sum()                                  # film volume at the start
    assert abs(r["ledger"][0] - start) < 1e-12 * start



def test_a_strided_film_takes_a_source_changed_every_period():
    """A stand-in roller: the host moves and scales the source every period, and the film steps once every 4
    periods (16 exchanges). Each exchange gets the rate set for it (SOFT-3's report: the step read only the last)."""
    def run(max_stride: int):
        flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(40, 1), edges=("wall",) * 4),),
                           material="latex_paint", inertia=False, max_depth=3.0e-4, max_stride=max_stride)
        sim = build_simulation(TestScene(build=_plate(False), init=lambda s: flow.set_depth(1.0e-4), ground=False,
                                         gravity=(-G, 0.0, 0.0), worlds=1, dt=5.0e-3), [flow], "cpu")
        want = 0.0
        for p in range(24):
            rate = np.zeros((40, 1))
            rate[5 + p // 2: 8 + p // 2] = 2.0e-5 * (1 + p % 3)
            flow.set_source(rate, chart=0)
            want += (rate.reshape(-1) * flow.tables.area_np).sum() * 0.02           # a 20 ms period
            sim.step_period()
        return flow, want
    (fast, want), (slow, _) = run(1), run(16)
    assert slow.stride == 16 and int(slow.tick.numpy()[0]) % 16 == 0
    for flow in (fast, slow):
        assert abs(flow.read()["supplied_volume"][0] - want) < 1e-12 * want
    a, b = fast.depth(0)[0], slow.depth(0)[0]
    assert np.abs(a - b).max() < 1.0e-6 * a.max()               # below its yield depth the paint stays where it lands

def test_spin_up_adds_the_euler_force():
    """A chart on a body spun up about its normal at α (no gravity): the film feels -α×r (WP2 review F3), plus the
    centrifugal -ω×(ω×r)."""
    from alloy.plugins.surface_flow import kernels as K  # noqa: PLC0415

    @wp.kernel
    def probe(fr: K.Frame, p: wp.vec3, out: wp.array[wp.vec3]):
        out[0] = K.body_force(fr, 0, 0, p)

    alpha, omega, dt = 15.0, 2.0, 1.25e-3
    body_q = wp.array([wp.transform_identity()], dtype=wp.transform, device="cpu")
    qd_in = wp.array([wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, omega - alpha * dt)], dtype=wp.spatial_vector,
                     device="cpu")
    qd_out = wp.array([wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, omega)], dtype=wp.spatial_vector, device="cpu")
    fr = K.Frame()
    fr.g, fr.omega, fr.alpha = (wp.zeros((1, 1), dtype=wp.vec3, device="cpu") for _ in range(3))
    fr.com = wp.zeros((1, 1), dtype=wp.vec3, device="cpu")
    sums = [wp.zeros((1, 1), dtype=wp.vec3, device="cpu") for _ in range(2)]
    wp.launch(K.frames_kernel, dim=(1, 1),
              inputs=[body_q, qd_in, qd_out, wp.array([[0]], dtype=wp.int32, device="cpu"),
                      wp.array([[wp.transform_identity()]], dtype=wp.transform, device="cpu"), wp.vec3(0.0, 0.0, 0.0),
                      dt, wp.ones(1, dtype=wp.int32, device="cpu"), 1],
              outputs=[*sums, fr.g, fr.omega, fr.alpha], device="cpu")
    out = wp.zeros(1, dtype=wp.vec3, device="cpu")
    wp.launch(probe, dim=1, inputs=[fr, wp.vec3(0.2, 0.0, 0.0)], outputs=[out], device="cpu")
    a = out.numpy()[0]
    assert a[1] == pytest.approx(-alpha * 0.2, rel=1e-4, abs=0.0)                       # Euler: -α×r, tangential
    assert a[0] == pytest.approx(omega**2 * 0.2, rel=1e-4, abs=0.0)                     # centrifugal, outward


@pytest.mark.parametrize("shape", ["cell", "strip"])
def test_beads_thinner_than_the_angle_stay(shape):
    """A wet cell or a one-cell-wide strip whose edge slope h/dx is under tan θ stays put with no gravity: opposite
    dry faces do not cancel the contact angle (WP2 review F2b)."""
    theta = 20.0
    mat = FilmMaterial("water_20deg", surface_tension=0.0728, contact_angle=theta)
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(40, 10)),), material=mat, max_depth=5.0e-4)
    dx = 0.2 / 40
    h0 = 0.6 * dx * math.tan(math.radians(theta))

    def init(sim):
        h = np.zeros((40, 10))
        if shape == "cell":
            h[20, 5] = h0
        else:
            h[8:32, 5] = h0
        flow.set_depth(h, chart=0)
    scene = TestScene(build=_plate(False), init=init, ground=False, gravity=(0.0, 0.0, -1.0e-3), periods=10, worlds=1)
    run = run_scene(scene, [flow], "cpu")
    assert run.error is None, run.error
    h = flow.depth(0)[0]
    assert int((h > 1.0e-6).sum()) == (1 if shape == "cell" else 24)           # no wider (a2d40a2f5 held them too)
    assert (h[:, :5] == 0.0).all() and (h[:, 6:] == 0.0).all()                  # nothing beside the row
    if shape == "cell":
        assert abs(float(h.max()) / h0 - 1.0) < 1.0e-6                         # the strip's ends pull along it


def test_stride_averages_the_frame_over_its_window():
    """A strided film sees the mean of the frame over its window: one exchange's acceleration spike counts once,
    not k times (WP2 review). Here the carrier's speed jumps once (a kick) inside a 16-exchange window."""
    from alloy.plugins.surface_flow import cell_kernels as CK  # noqa: PLC0415
    from alloy.plugins.surface_flow import kernels as K  # noqa: PLC0415

    W, k, dt = 1, 16, 1.25e-3
    dev = "cpu"
    tick = wp.zeros(W, dtype=wp.int32, device=dev)
    sums = [wp.zeros((1, 1), dtype=wp.vec3, device=dev) for _ in range(2)]
    outs = [wp.zeros((1, 1), dtype=wp.vec3, device=dev) for _ in range(3)]
    body_q = wp.array([wp.transform_identity()], dtype=wp.transform, device=dev)
    cbody = wp.array([[0]], dtype=wp.int32, device=dev)
    clocal = wp.array([[wp.transform_identity()]], dtype=wp.transform, device=dev)
    rest = wp.array([wp.spatial_vector()], dtype=wp.spatial_vector, device=dev)
    kicked = wp.array([wp.spatial_vector(0.1, 0.0, 0.0, 0.0, 0.0, 0.0)], dtype=wp.spatial_vector, device=dev)
    for n in range(k):
        q_in, q_out = (rest, kicked) if n == 3 else ((kicked, kicked) if n > 3 else (rest, rest))
        wp.launch(CK.tick_kernel, dim=W, inputs=[], outputs=[tick], device=dev)
        wp.launch(K.frames_kernel, dim=(1, 1), inputs=[body_q, q_in, q_out, cbody, clocal, wp.vec3(0.0, 0.0, -G), dt,
                                                       tick, k], outputs=[*sums, *outs], device=dev)
    g = outs[0].numpy()[0, 0]
    assert g[0] == pytest.approx(-0.1 / dt / k, rel=1e-5, abs=0.0)                       # the kick's mean, not the kick
    assert g[2] == pytest.approx(-G, rel=1e-6, abs=0.0) and np.allclose(sums[0].numpy(), 0.0)


def test_open_edge_outflux_is_queued_for_the_drop_stage():
    """Design §4.7: every m³ that leaves an open edge without a drain is queued in shed (with its momentum) for WP9's
    drop stage, as well as booked in spilled; dripped joins the ledger."""
    flow, run = _run(_plate(False), (-G, 0.0, 0.0), periods=10)
    r = flow.read()
    shed, spilled = r["shed"][0].sum(), r["spilled"][0].sum()
    assert spilled > 0.0 and shed == pytest.approx(spilled, rel=1e-12, abs=0.0)
    out = r["shed_mom"][0].sum(axis=0) / shed                               # mean outflow velocity, chart frame
    assert out[0] < 0.0 and abs(out[1]) < 1e-6 * abs(out[0])                # out through the x- edge
    assert r["dripped_volume"][0] == 0.0
