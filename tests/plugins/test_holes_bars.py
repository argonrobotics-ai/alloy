# SPDX-License-Identifier: Apache-2.0
"""Bars in a drilled layer (drilling v2, phase C): the clearance between the drill-point solid and a bar, the bottom
past which a hole has cut through a bar, the bar and hammer configs, a bar in a stack's second layer, and the jammed
front's force (CPU, light)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton

from alloy.plugins.drilling import Drill
from alloy.plugins.drilling.kernels import jam_force
from alloy.plugins.errors import PluginError
from alloy.plugins.holes.bars import Bar, bar_clearance, bar_through, clearance_np, through_np
from alloy.plugins.holes.front_types import Bit, LayerFront
from alloy.plugins.holes.work import PERCUSSIVE, energy_per_volume


@wp.kernel
def _clearances(tips: wp.array[wp.vec3], bars: wp.array[wp.vec3], dirs: wp.array[wp.vec3], out: wp.array[wp.float32]):
    k = wp.tid()
    out[k] = bar_clearance(tips[k], wp.vec3(0.0, 0.0, -1.0), 0.00635, 1.6642795, bars[k], dirs[k], 0.00635)


def _dense_clearance(tip: np.ndarray, bar: Bar, radius: float, t: float) -> float:
    """Brute force in 3D: the bar axis's points against the point solid's profile (distance in the radial-axial
    half-plane of each point), less the bar's radius."""
    xs = np.linspace(-0.06, 0.0, 3001)
    rs = np.minimum(radius, -xs * t)
    lam = np.linspace(-0.15, 0.15, 6001)[:, None]
    q = np.asarray(bar.point) + lam * np.asarray(bar.direction) - tip
    x = -q[:, 2]                                                                 # along the drill axis (0, 0, -1)
    rho = np.hypot(q[:, 0], q[:, 1])
    inside = (x <= 0.0) & (rho <= np.minimum(radius, -x * t))
    d = np.sqrt((rho[:, None] - rs[None, :]) ** 2 + (x[:, None] - xs[None, :]) ** 2).min(axis=1)
    return float(np.where(inside, 0.0, d).min()) - bar.radius


def test_the_bar_clearance_matches_its_host_twin_and_a_dense_3d_search():
    """The device clearance equals the host twin and a brute-force 3D search, for bars across the drill axis and
    tilted up to 10 deg out of the face plane, anywhere along them."""
    rng = np.random.default_rng(3)
    n = 24
    tips = np.column_stack([rng.uniform(-0.02, 0.02, n), rng.uniform(-0.01, 0.01, n), rng.uniform(-0.04, -0.01, n)])
    tilt = np.radians(rng.uniform(-10.0, 10.0, n)) * (np.arange(n) % 2)          # half across, half tilted
    yaw = rng.uniform(0.0, 2.0 * np.pi, n)
    dirs = np.column_stack([np.cos(yaw) * np.cos(tilt), np.sin(yaw) * np.cos(tilt), np.sin(tilt)])
    pts = np.column_stack([rng.uniform(-0.08, 0.08, n), rng.uniform(-0.08, 0.08, n), np.full(n, -0.03)])
    out = wp.zeros(n, dtype=wp.float32, device="cpu")
    wp.launch(_clearances, dim=n, inputs=[wp.array(tips, dtype=wp.vec3, device="cpu"),
                                          wp.array(pts, dtype=wp.vec3, device="cpu"),
                                          wp.array(dirs, dtype=wp.vec3, device="cpu")], outputs=[out], device="cpu")
    t = 1.6642795
    for k in range(n):
        bar = Bar(point=tuple(pts[k]), direction=tuple(dirs[k]), radius=0.00635)
        host = clearance_np(tips[k], (0.0, 0.0, -1.0), 0.00635, t, bar)
        dense = _dense_clearance(tips[k], bar, 0.00635, t)
        assert abs(float(out.numpy()[k]) - host) < 2e-6 and abs(host - dense) < 5e-6, (k, host, dense)
    with pytest.raises(ValueError):
        Bar(point=(0.0, 0.0, -0.03), direction=(0.0, 0.97, 0.26))                    # 15 deg: refused


@wp.kernel
def _throughs(bars: wp.array[wp.vec3], dirs: wp.array[wp.vec3], out: wp.array[wp.float32]):
    k = wp.tid()
    out[k] = bar_through(wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, -1.0), 0.00635, 1.6642795, bars[k], dirs[k],
                         0.00635)


def _dense_through(bar: Bar, radius: float, t: float) -> float:
    """The largest z + rho / t over the bar's part inside the drill's cylinder (entry at the origin, axis -z): each
    line of bar points parallel to its axis (a 12001 x 41 grid over its cross-section) is inside the cylinder between
    the two roots of |p_perp| = R, where the convex z + rho / t is largest; -inf if no line reaches it."""
    ub = np.asarray(bar.direction)
    e1 = np.cross(ub, (1.0, 0.0, 0.0) if abs(ub[0]) < 0.9 else (0.0, 1.0, 0.0))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ub, e1)
    th, r = np.meshgrid(np.linspace(0.0, 2.0 * np.pi, 12001), np.linspace(0.0, bar.radius, 41))
    p0 = np.asarray(bar.point) + r.reshape(-1, 1) * (np.cos(th).reshape(-1, 1) * e1 + np.sin(th).reshape(-1, 1) * e2)
    a = np.array([0.0, 0.0, -1.0])
    u_perp, p_perp = ub - (ub @ a) * a, p0 - (p0 @ a)[:, None] * a
    qa, qb, qc = u_perp @ u_perp, 2.0 * (p_perp @ u_perp), (p_perp * p_perp).sum(axis=1) - radius**2
    disc = qb * qb - 4.0 * qa * qc
    ok = disc >= 0.0
    if not ok.any():
        return -math.inf
    roots = (-qb[ok, None] + np.array([-1.0, 1.0]) * np.sqrt(disc[ok, None])) / (2.0 * qa)
    z = (p0[ok] @ a)[:, None] + roots * (ub @ a)
    return float(z.max()) + radius / t


def test_the_through_bottom_matches_its_host_twin_and_a_dense_search():
    """The bottom past which a hole has cut through a bar: device = host twin (float32), and = a dense search over the
    bar's part inside the drill's cylinder for bars across the axis, over it, beside it and out of reach (2 um); for a
    bar tilted out of the face plane the bound lies above the search, by at most r_bar plus twice the tilt term."""
    t, radius, rb = 1.6642795, 0.00635, 0.00635
    offsets = (0.0, 0.004, 0.008, 0.0115, 0.0126, 0.013, 0.0, 0.006, 0.010)
    tilts = (0.0,) * 6 + (4.0, 8.0, 10.0)
    yaw = np.radians([0.0, 30.0, 70.0, 110.0, 150.0, 200.0, 20.0, 240.0, 300.0])
    dirs = np.column_stack([-np.sin(yaw) * np.cos(np.radians(tilts)), np.cos(yaw) * np.cos(np.radians(tilts)),
                            np.sin(np.radians(tilts))])
    pts = np.column_stack([np.asarray(offsets) * np.cos(yaw), np.asarray(offsets) * np.sin(yaw), np.full(9, -0.03)])
    out = wp.zeros(len(offsets), dtype=wp.float32, device="cpu")
    wp.launch(_throughs, dim=len(offsets), inputs=[wp.array(pts, dtype=wp.vec3, device="cpu"),
                                                   wp.array(dirs, dtype=wp.vec3, device="cpu")], outputs=[out],
              device="cpu")
    for k in range(len(offsets)):
        bar = Bar(point=tuple(pts[k]), direction=tuple(dirs[k]), radius=rb)
        host = through_np((0.0, 0.0, 0.0), (0.0, 0.0, -1.0), radius, t, bar)
        dense = _dense_through(bar, radius, t)
        if not math.isfinite(dense):
            assert host == -math.inf and float(out.numpy()[k]) < -1e29, k                # out of reach
            continue
        assert abs(float(out.numpy()[k]) - host) < 2e-6, (k, float(out.numpy()[k]), host)
        if tilts[k] == 0.0:
            assert abs(host - dense) < 2e-6, (k, host, dense)
        else:
            c = math.sin(math.radians(tilts[k]))
            assert -2e-6 < host - dense < rb + 2.0 * c * (radius + rb) / math.sqrt(1.0 - c * c), (k, host, dense)


def test_hammer_and_bar_configs_are_checked():
    from alloy.plugins.harness.check import fresh

    d = Drill(layers=({"workpiece": "wall", "work": "concrete_medium_hard", "density": 2400.0,
                       "bars": ({"point": (0.0, 0.0, -0.03)},)},), blow_energy=1.8, blow_rate=85.0)
    assert isinstance(d.config.layers[0].bars[0], Bar) and fresh(d).config.layers == d.config.layers
    assert energy_per_volume(PERCUSSIVE["concrete_medium_hard"], 0.010) == pytest.approx(0.5 * (3.28e8 + 2.04e8),
                                                                                          rel=2e-3)
    for bad in ({"blow_energy": -1.0}, {"blow_rate": math.nan},
                {"layers": ({"workpiece": "wall", "bars": ({"point": (0, 0, 0), "radius": 0.0},)},)},
                {"layers": ({"workpiece": "wall", "work": "concrete_c99"},)}):
        with pytest.raises(PluginError):
            Drill(**bad)


@pytest.mark.compile_heavy
def test_a_bar_in_the_second_layer_stops_the_bit_there():
    """A 5 mm pine ledger on the concrete block, a #4 bar 25 mm under the concrete's face: the 12.7 mm bit drills the
    ledger (Shaw), then the concrete (blows), and stops where its point first touches the bar in layer 1."""
    from alloy.plugins.drilling import laws as dl
    from alloy.plugins.harness.check import fresh
    from alloy.plugins.harness.run import run_scene
    from alloy.plugins.harness.scene import TestScene, body_xform
    from alloy.plugins.holes import concrete_laws as cl

    def build(b: newton.ModelBuilder) -> None:
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, 0.03), hx=0.04, hy=0.04, hz=0.03, label="wall")
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, 0.0625), hx=0.04, hy=0.04, hz=0.0025, label="ledger")
        z = 0.065 + dl.GAP + dl.HEAD_HALF + dl.BIT_LEN
        head = b.add_body(xform=body_xform(0.0, 0.0, z), label="spindle")
        b.add_shape_box(head, hx=dl.HEAD_HALF, hy=dl.HEAD_HALF, hz=dl.HEAD_HALF,
                        cfg=newton.ModelBuilder.ShapeConfig(density=dl.HEAD_MASS / (2.0 * dl.HEAD_HALF) ** 3))

    stop = cl.bar_stop_depth(0.0)
    init = cl._init((cl.BIT,), (4.0 * cl.BLOW_E,), (cl.BLOW_F,), travel=(dl.GAP + 0.005 + stop + 0.0015,))
    scene = TestScene(build=build, init=init, worlds=1, periods=80, dt=dl.STEP, ground=False,
                      companions=dl._head(Drill().config))
    config = {"layers": ({"workpiece": "ledger", "work": "pine_southern", "density": 538.0},
                         {**cl.LAYER, "workpiece": "wall", "bars": (cl.BAR,)}), "tip": (0.0, 0.0, -0.07)}
    run = run_scene(scene, [fresh(Drill(), config)], "cpu")
    assert run.error is None, run.error
    r = run.plugin.read()
    wall = run.plugin.holes(0, layer=1)
    assert int(r["jammed"][0]) == 1 and len(wall) == 1
    assert abs(wall[0].stages[0].depth - stop) < 5e-6 and float(r["bar_cut"][0]) < 1e-6


@wp.kernel
def _jam(e: wp.float32, v: wp.array[wp.vec3], bit: Bit, out: wp.array[wp.float32]):
    k = wp.tid()
    lf = LayerFront()
    lf.e = e
    lf.v = v[k]
    lf.a = wp.vec3(0.0, 0.0, -1.0)
    out[k] = jam_force(lf, bit)


def test_the_jammed_front_damps_like_a_face():
    """A bit jammed on a bar is held by the front's spring k e plus its Kelvin-Voigt damping, k c v along the hole
    axis: harder while the bit closes on the bar, softer while it backs off, never pulling (the face's convention)."""
    bit = Bit()
    bit.stiffness, bit.damping = 2.0e6, 1.0e-3
    v = wp.array([(0.0, 0.0, -0.05), (0.0, 0.0, 0.0), (0.0, 0.0, 0.05), (0.0, 0.0, 0.5)], dtype=wp.vec3,
                 device="cpu")
    out = wp.zeros(4, dtype=wp.float32, device="cpu")
    wp.launch(_jam, dim=4, inputs=[1.0e-4, v, bit], outputs=[out], device="cpu")
    want = [2.0e6 * (1.0e-4 + 1.0e-3 * s) for s in (0.05, 0.0, -0.05)] + [0.0]
    assert np.allclose(out.numpy(), want, rtol=1e-5, atol=1e-3), out.numpy()


@pytest.mark.compile_heavy
def test_a_jam_on_an_exposed_bar_is_reported():
    """A #4 bar whose top is the concrete's face: the 12.7 mm bit lands on it before its hole has a stage. It is held
    there, and reported jammed with the front's thrust; no hole is cut and nothing into the bar."""
    from alloy.plugins.drilling import laws as dl
    from alloy.plugins.harness.check import fresh
    from alloy.plugins.harness.run import run_scene
    from alloy.plugins.harness.scene import TestScene, body_xform
    from alloy.plugins.holes import concrete_laws as cl

    def build(b: newton.ModelBuilder) -> None:
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, 0.03), hx=0.04, hy=0.04, hz=0.03, label="wall")
        head = b.add_body(xform=body_xform(0.0, 0.0, 0.06 + dl.GAP + 0.07), label="spindle")
        b.add_shape_box(head, hx=dl.HEAD_HALF, hy=dl.HEAD_HALF, hz=dl.HEAD_HALF,
                        cfg=newton.ModelBuilder.ShapeConfig(density=dl.HEAD_MASS / (2.0 * dl.HEAD_HALF) ** 3))

    init = cl._init((cl.BIT,), (cl.BLOW_E,), (cl.BLOW_F,), travel=(dl.GAP + 0.0005,))
    scene = TestScene(build=build, init=init, worlds=1, periods=40, dt=dl.STEP, ground=False,
                      companions=dl._head(Drill().config))
    bar = Bar(point=(0.0, 0.0, -cl.BAR_R), direction=(0.0, 1.0, 0.0), radius=cl.BAR_R)
    config = {"layers": ({**cl.LAYER, "workpiece": "wall", "bars": (bar,)},), "tip": (0.0, 0.0, -0.07)}
    run = run_scene(scene, [fresh(Drill(), config)], "cpu")
    assert run.error is None, run.error
    r = run.plugin.read()
    assert int(r["jammed"][0]) == 1 and float(r["thrust"][0]) > 100.0, (r["jammed"], r["thrust"])
    assert float(r["bar_cut"][0]) == 0.0 and float(r["removed"][0]) == 0.0
