# SPDX-License-Identifier: Apache-2.0
"""Surface-flow drops without SPH (design §4.4, §4.7; WP9): the ballistic pool, landing, drips, the ledger (CPU)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton

import alloy.plugins as ap
from alloy.plugins.harness.run import run_scene
from alloy.plugins.harness.scene import TestScene, body_xform
from alloy.plugins.surface_flow import FilmMaterial, PlaneChart, SurfaceFlow

G = 9.81
WATER = FilmMaterial("water_flat", surface_tension=0.0)


def _box(z: float, hx: float, hy: float, rot=(0.0, 0.0, 0.0, 1.0)):
    def build(b: newton.ModelBuilder) -> None:
        b.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.0, z), wp.quat(*rot)), hx=hx, hy=hy, hz=0.01,
                        label="plate")
        dummy = b.add_body(xform=body_xform(2.0, 0.0, 1.0), label="dummy")   # the harness needs a body
        b.add_shape_sphere(dummy, radius=0.01)
    return build


def _run(build, flow, gravity, periods, init=None):
    run = run_scene(TestScene(build=build, init=init, ground=False, gravity=gravity, periods=periods, worlds=1),
                    [flow], "cpu")
    assert run.error is None, run.error
    return flow.read(), flow.drops.read()


def _closes(r, d) -> float:
    """D-1, the drop stage's two books (the film's own ledger is WP15's law F-11): everything shed (open edges and
    drips) went to drops, to SPH or is pending; every drop made is in flight, floored or landed. Relative residual."""
    shed = r["spilled_volume"][0] + r["dripped_volume"][0]
    a = shed - (d["to_pool_volume"][0] + d["to_sph_volume"][0] + d["pending_volume"][0])
    b = d["to_pool_volume"][0] - (d["in_flight_volume"][0] + d["floored_volume"][0] + d["landed_volume"][0])
    return max(abs(a), abs(b)) / max(shed, 1e-30)


@pytest.mark.compile_heavy
def test_film_off_an_open_edge_falls_to_the_floor_and_the_ledger_closes():
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(40, 1), edges=("open", "wall", "wall", "wall")),),
                       material=WATER, max_depth=2.0e-3, drops={"floor_z": 0.0, "drop_radius": 1.0e-3})

    def init(sim):
        flow.set_depth(1.0e-3, chart=0)
    r, d = _run(_box(0.5, 0.1, 0.025), flow, (-3.0, 0.0, -G), 50, init)
    assert d["to_pool_volume"][0] > 0.0 and d["floored_volume"][0] > 0.0
    assert r["spilled_volume"][0] + r["dripped_volume"][0] == pytest.approx(
        d["to_pool_volume"][0] + d["pending_volume"][0], rel=1e-12, abs=0.0)
    assert _closes(r, d) < 1e-10       # float64 rounding over thousands of small adds (one drop: 4e-4)


@pytest.mark.compile_heavy
def test_a_drop_lands_at_the_projectile_range():
    """D-4: a drop launched at (x0, z0) with speed u flies the exact parabola (1e-5 m after 0.1 s: a symplectic Euler
    step would be 1 mm off), lands on the plate top at x0 + u·sqrt(2·Δz/g) within a cell, and joins the film there
    (the landing cell is set wet, so the dry-tile skip steps it)."""
    x0, z0, u, top = -0.2, 0.5, 0.6, 0.21
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(120, 40)),), material=WATER, max_depth=5.0e-3,
                       drops={"drop_radius": 2.0e-3})

    def init(sim):
        flow.drops.pool.pos.assign(np.array([[[x0, 0.0, z0]] + [[0.0, 0.0, -9.0]] * 255], np.float32))
        flow.drops.pool.vel.assign(np.array([[[u, 0.0, 0.0]] + [[0.0, 0.0, 0.0]] * 255], np.float32))
        flow.drops.pool.alive.assign(np.array([[1] + [0] * 255], np.int32))
    r, d = _run(_box(0.2, 0.3, 0.1), flow, (0.0, 0.0, -G), 15, init)
    assert d["landed_volume"][0] == pytest.approx(flow.drops.drop_volume, rel=1e-12, abs=0.0)
    assert r["film_volume"][0] == pytest.approx(flow.drops.drop_volume, rel=1e-6, abs=0.0)
    h = flow.depth(0)[0]
    i = np.unravel_index(np.argmax(h), h.shape)[0]
    x_cell = flow.cell_centres(0)[i, 0, 0]
    x_pred = x0 + u * math.sqrt(2.0 * (z0 - top) / G)
    assert abs(x_cell - x_pred) <= 0.6 / 120 * 1.01
    flight = SurfaceFlow(charts=(PlaneChart("plate", cells=(120, 40)),), material=WATER, max_depth=5.0e-3,
                         drops={"drop_radius": 2.0e-3})

    def launch(sim):
        flight.drops.pool.pos.assign(np.array([[[x0, 0.0, z0]] + [[0.0, 0.0, -9.0]] * 255], np.float32))
        flight.drops.pool.vel.assign(np.array([[[u, 0.0, 0.0]] + [[0.0, 0.0, 0.0]] * 255], np.float32))
        flight.drops.pool.alive.assign(np.array([[1] + [0] * 255], np.int32))
    _run(_box(0.2, 0.3, 0.1), flight, (0.0, 0.0, -G), 5, launch)      # 0.1 s, still in the air
    p = flight.drops.pool.pos.numpy()[0, 0]
    assert np.allclose(p, [x0 + u * 0.1, 0.0, z0 - 0.5 * G * 0.01], atol=1e-5)


@pytest.mark.compile_heavy
def test_a_hanging_film_drips_to_its_retention_depth():
    """D-3: a film under a plate (chart facing down) drains to h_ret = 2 l_c sin(θ_r/2) (θ_r 90°) and stops."""
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(8, 8)),), material="water", max_depth=8.0e-3,
                       drops={"floor_z": 0.0, "tau_drip": 0.02})

    def init(sim):
        flow.set_depth(6.0e-3, chart=0)
    r, d = _run(_box(0.5, 0.02, 0.02, rot=(1.0, 0.0, 0.0, 0.0)), flow, (0.0, 0.0, -G), 25, init)
    l_c = math.sqrt(0.0728 / (1000.0 * G))
    h_ret = 2.0 * l_c * math.sin(math.radians(45.0))
    h = flow.depth(0)[0]
    assert np.allclose(h, h_ret, rtol=1e-3, atol=0.0)
    assert r["dripped_volume"][0] == pytest.approx((6.0e-3 - h.mean()) * 0.04 * 0.04, rel=1e-6, abs=0.0)
    assert _closes(r, d) < 1e-10       # float64 rounding over thousands of small adds (one drop: 4e-4)


def _incline_run(periods: int, theta: float, q_src: float):
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(40, 1), edges=("open", "wall", "wall", "wall")),),
                       material=WATER, max_depth=2.0e-3, drops={"floor_z": 0.0, "drop_radius": 1.0e-3})

    def init(sim):
        src = np.zeros((40, 1))
        src[36:] = q_src
        flow.set_source(src, chart=0)
    gravity = (-G * math.sin(theta), 0.0, -G * math.cos(theta))
    return _run(_box(0.5, 0.05, 0.025), flow, gravity, periods, init)


@pytest.mark.compile_heavy
def test_a_steady_film_sheds_at_its_supply_rate():
    """D-2: a Nusselt film fed at Q on a 30° incline: at steady state the open edge sheds Q (2 %), every m³ into
    drops or pending; and the ledger closes."""
    theta, s = math.radians(30.0), 1.0e-3                       # 4 cells x 1.25e-4 m² x 1e-3 m/s = 5e-7 m³/s
    q = s * 4 * (0.1 / 40) * 0.05
    (r1, d1), (r2, d2) = _incline_run(150, theta, s), _incline_run(200, theta, s)
    shed = lambda r: r["spilled_volume"][0]                     # noqa: E731
    rate = (shed(r2) - shed(r1)) / (50 * 0.02)
    assert rate == pytest.approx(q, rel=0.02, abs=0.0)
    made = (d2["to_pool_volume"][0] - d1["to_pool_volume"][0]) / (50 * 0.02)    # the drop stage's own output
    assert made == pytest.approx(q, rel=0.03, abs=0.0)                   # 2 % plus one drop of ~120 per second
    assert shed(r2) == pytest.approx(d2["to_pool_volume"][0] + d2["pending_volume"][0], rel=1e-12, abs=0.0)
    assert _closes(r2, d2) < 1e-10


@pytest.mark.compile_heavy
def test_drops_are_deterministic_and_worlds_identical():
    """Harness rules on a scene that sheds: two worlds give the same drops and film bit for bit, and so do two runs."""
    def once():
        flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(40, 1), edges=("open", "wall", "wall", "wall")),),
                           material=WATER, max_depth=2.0e-3, drops={"floor_z": 0.0, "drop_radius": 1.0e-3})

        def init(sim):
            flow.set_depth(1.0e-3, chart=0)
        scene = TestScene(build=_box(0.5, 0.1, 0.025), init=init, ground=False, gravity=(-3.0, 0.0, -G), periods=30,
                          worlds=2)
        run = run_scene(scene, [flow], "cpu")
        assert run.error is None, run.error
        return flow.read(), flow.drops.read(), flow.drops.pool.pos.numpy()
    (r1, d1, p1), (r2, d2, p2) = once(), once()
    assert d1["to_pool_volume"][0] > 0.0
    for k in d1:
        assert d1[k][0] == d1[k][1] and np.array_equal(d1[k], d2[k]), k
    assert np.array_equal(r1["h"][0], r1["h"][1]) and np.array_equal(r1["h"], r2["h"])
    assert np.array_equal(p1[0], p1[1]) and np.array_equal(p1, p2)


@pytest.mark.compile_heavy
def test_harness_quick_check_of_a_shedding_film():
    """The harness's quick checks (contract, float atomics, crash, hook-level determinism, worlds and graph capture) on
    its tilted plate with the x- edge open: the film sheds 0.5 mm drops that fall to z = -0.5 m (the GPU smoke runs the
    full check)."""
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(20, 20), edges=("open", "wall", "wall", "wall")),),
                       material="water", drops={"floor_z": -0.5, "drop_radius": 5.0e-4})
    report = ap.check(flow, quick=True, laws=False)
    assert report.passed, report.summary()
    run = run_scene(flow.test_scene(), [flow], "cpu")                      # the scene really sheds
    assert run.error is None, run.error
    assert flow.drops.read()["to_pool_volume"][0] > 0.0
