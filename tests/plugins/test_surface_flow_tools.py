# SPDX-License-Identifier: Apache-2.0
"""Surface flow, applicators (phase 2): a tool sliding on a film shears it toward the Couette mean U/2, a blade
ploughs a film ahead of it with an exact ledger, and a tool rests on the film's top under its weight (CPU)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy._vendor import newton

from alloy.plugins.errors import PluginError
from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.surface_flow import FilmMaterial, PlaneChart, SurfaceFlow
from alloy.plugins.surface_flow.applicators import Applicator

G = 9.81
OIL = FilmMaterial("oil_1000cSt", density=970.0, viscosity=0.97, surface_tension=0.0)
LENGTH, WIDTH = 0.2, 0.05
TOOL_HALF = (0.02, 0.04, 0.01)        # the tool block: its underside face is 2 mm below the block
DROP = 0.002


def scene(z_underside: float, x0: float, gravity: tuple[float, float, float], density: float = 2000.0,
          guided: bool = False):
    """A static plate (top face at z = 0, the chart) and a tool block whose underside sits at ``z_underside`` at
    ``x0``: free, or ``guided`` on a slide along x (it keeps its height and attitude: a blade coater)."""
    def build(b: newton.ModelBuilder) -> None:
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, -0.01), hx=0.5 * LENGTH, hy=0.5 * WIDTH, hz=0.01,
                        label="plate")
        xf = body_xform(x0, 0.0, z_underside + DROP + TOOL_HALF[2])
        if guided:
            tool = b.add_link(xform=xf, label="tool")
            b.add_articulation([b.add_joint_prismatic(-1, tool, parent_xform=xf, axis=newton.Axis.X)])
        else:
            tool = b.add_body(xform=xf, label="tool")
        b.add_shape_box(tool, hx=TOOL_HALF[0], hy=TOOL_HALF[1], hz=TOOL_HALF[2],
                        cfg=newton.ModelBuilder.ShapeConfig(density=density))
    return build


def blade(**kw) -> Applicator:
    return Applicator(body="tool", shape="rect", size=(TOOL_HALF[0], 0.035),
                      center=(0.0, 0.0, -TOOL_HALF[2] - DROP), **kw)


def run(flow: SurfaceFlow, build, depth: float, speed: float, gravity, periods: int, worlds: int = 1):
    def init(sim) -> None:
        flow.set_depth(depth)
        qd = sim.state.joint_qd.numpy()
        if qd.size == worlds:                                           # a slide: one coordinate per world
            qd[:] = speed
        else:                                                           # a free joint: (vx, vy, vz, wx, wy, wz)
            qd = qd.reshape(-1, 6)
            qd[:, :3] = (speed, 0.0, 0.0)
        sim.state.joint_qd.assign(qd.reshape(-1))
        newton.eval_fk(sim.model, sim.state.joint_q, sim.state.joint_qd, sim.state)
    sim = build_simulation(TestScene(build=build, init=init, ground=False, gravity=gravity, worlds=worlds,
                                     dt=2.0e-3), [flow], "cpu")
    for _ in range(periods):
        sim.step_period()
    return sim


def test_a_sliding_tool_shears_the_film_toward_half_its_speed():
    """Couette: under a plate sliding at U on a film it touches, the depth-averaged velocity relaxes to U/2."""
    h0, U = 1.0e-3, 0.05
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(80, 4)),), material=OIL, max_depth=2.0e-3,
                       tools=(blade(stiffness=1.0e3),))                     # a light touch: the tool keeps its height
    sim = run(flow, scene(h0 - 2.0e-6, 0.0, (0.0, 0.0, 0.0), density=2.0e6, guided=True), h0, U, (0.0, 0.0, 0.0), periods=1)
    U_now = float(sim.state.joint_qd.numpy()[0])                       # the film's drag slows the tool a little
    a, b = flow.tables.fa.numpy(), flow.tables.fb.numpy()
    touch = (flow.tf.cover.numpy()[0] >= 1.0) & (flow.h.numpy()[0] >= flow.tf.ceil.numpy()[0])
    i_faces = np.arange(81 * 4)                                         # i-faces first: their direction is +x
    under = i_faces[(a[i_faces] >= 0) & (b[i_faces] >= 0)]
    under = under[touch[a[under]] & touch[b[under]]]
    u = flow.u.numpy()[0, under]
    assert len(under) >= 8 and 0.9 * U < U_now < U
    assert np.allclose(u, 0.5 * U_now, rtol=0.01), (u, U_now)
    x, tool_x = flow.tables.fpos.numpy()[:, 0], float(sim.state.body_q.numpy()[0][0])
    far = i_faces[np.abs(x[i_faces] - tool_x) > TOOL_HALF[0] + 0.02]
    assert np.abs(flow.u.numpy()[0, far]).max() < 0.05 * U             # the free film far from the tool stays put


def test_a_blade_leaves_half_its_gap_and_ploughs_the_rest_with_an_exact_ledger():
    """A heavy blade gliding at a fixed gap over a thicker film: under it the film is sheared at U/2, so it leaves
    gap/2 behind (blade coating's Couette limit) and pushes the rest ahead; no film is made or lost."""
    h0, gap, U, x0 = 1.0e-3, 0.4e-3, 0.1, -0.06
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(80, 4)),), material=OIL, max_depth=3.0e-3,
                       tools=(blade(stiffness=1.0e3),))
    sim = run(flow, scene(gap, x0, (0.0, 0.0, 0.0), density=2.0e6, guided=True), h0, U, (0.0, 0.0, 0.0), periods=40)
    h = flow.depth(0)[0].mean(axis=1)
    x = flow.cell_centres(0)[:, 0, 0]
    x_end, dx = float(sim.state.body_q.numpy()[0][0]), LENGTH / 80
    assert x_end - x0 > 0.95 * U * 0.8                                  # it kept its speed
    behind = (x > x0 + TOOL_HALF[0] + 2 * dx) & (x < x_end - TOOL_HALF[0] - 2 * dx)
    assert behind.sum() >= 6
    assert abs(h[behind].mean() / (0.5 * gap) - 1.0) < 0.03, h[behind]   # a stride/cell beat: +12 % bumps
    assert np.abs(h[behind] / (0.5 * gap) - 1.0).max() < 0.15, h[behind]
    assert h[x > x_end + TOOL_HALF[0]].max() > 1.5 * h0                 # the bow wave ahead
    r = flow.read()
    start = h0 * flow.tables.area_np.sum()
    assert abs(r["ledger"][0] - start) < 1e-12 * start


def test_a_pressing_tool_squeezes_the_film_out_and_rests_on_the_substrate():
    """A blade set down on a film presses it out sideways (volume exact) and rests on the plate's pad."""
    h0, k = 1.0e-3, 2.0e7
    viscous = FilmMaterial("viscous_10Pas", density=1000.0, viscosity=10.0, surface_tension=0.0)
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(40, 10)),), material=viscous, max_depth=3.0e-3,
                       tools=(blade(stiffness=k, damping=0.002),))
    sim = run(flow, scene(h0 + 1.0e-4, 0.0, (0.0, 0.0, -G), density=20000.0), h0, 0.0, (0.0, 0.0, -G),
              periods=30)
    mass = float(sim.model.body_mass.numpy()[0])
    area = 2.0 * TOOL_HALF[0] * WIDTH                                   # the blade is wider than the plate
    z = float(sim.state.body_q.numpy()[0][2]) - DROP - TOOL_HALF[2]   # the underside
    assert z == pytest.approx(-mass * G / (k * area), rel=0.05, abs=0.0)        # on the pad
    x = flow.cell_centres(0)[:, 0, 0]
    h = flow.depth(0)[0]
    assert h[np.abs(x) < TOOL_HALF[0] - 0.005].max() < 1.0e-6        # pressed out
    r = flow.read()
    start = h0 * flow.tables.area_np.sum()
    assert abs(r["ledger"][0] - start) < 1e-12 * start


def test_tools_refuse_bad_specs_and_unstable_pads():
    with pytest.raises(PluginError, match="shape"):
        Applicator(body="tool", shape="sphere")
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(8, 4)),), material=OIL, tools=(blade(stiffness=1.0e12),))
    with pytest.raises(PluginError, match="unstable"):
        build_simulation(TestScene(build=scene(1e-3, 0.0, (0.0, 0.0, -G)), ground=False, worlds=1, dt=2.0e-3),
                         [flow], "cpu").step_period()


def punch_scene(density: float, face: float, drop: float = 0.02):
    """A static slab (top face at z = 0, the chart) and a free block carrying a flat disc punch ``drop`` below it,
    its face at height ``face``."""
    def build(b: newton.ModelBuilder) -> None:
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, -0.01), hx=0.06, hy=0.06, hz=0.01, label="slab")
        tool = b.add_body(xform=body_xform(0.0, 0.0, face + drop + 0.01), label="tool")
        b.add_shape_box(tool, hx=0.025, hy=0.025, hz=0.01, cfg=newton.ModelBuilder.ShapeConfig(density=density))
    return build


@pytest.mark.parametrize("implicit", [False, True])
@pytest.mark.parametrize("load", [0.9, 1.2])
def test_a_punch_sinks_into_paste_only_above_its_bearing_capacity(load, implicit):
    """F-18 (Prandtl 1920): a flat punch on a plastic bulk holds below p_bc = (2 + π)·τ_y and sinks above it at
    (p - p_bc)/C_Y; the paste it displaces heaves beside it, volume exact."""
    from alloy.plugins.surface_flow.forming import Bulk  # noqa: PLC0415

    bulk, k, radius, drop = Bulk(yield_stress=200.0, resistance=2.5e4), 5.0e6, 0.02, 0.02
    area = np.pi * radius**2
    density = load * bulk.bearing * area / G / (0.05 * 0.05 * 0.02)       # the block's weight on the punch
    punch = Applicator(body="tool", shape="disc", size=(radius, radius), center=(0.0, 0.0, -0.01 - drop),
                       stiffness=k, damping=0.004, implicit=implicit)      # ζ ≈ 0.4: a ringing pad pumps the paste
    flow = SurfaceFlow(charts=(PlaneChart("slab", cells=(48, 48)),), material="concrete_paste", inertia=False,
                       max_depth=1.0e-3, tools=(punch,), bulk=bulk)
    rest = -load * bulk.bearing / k                                         # set down at its elastic rest
    sim = build_simulation(TestScene(build=punch_scene(density, rest, drop), ground=False, worlds=1, dt=2.0e-3), [flow],
                           "cpu")
    z = []
    for _ in range(20):
        sim.step_period()
        z.append(float(sim.state.body_q.numpy()[0][2]) - 0.01 - drop)      # the punch face
    b = flow.read()
    moved = np.abs(b["bulk"][0] * flow.tables.area_np).sum()
    if load < 1.0:
        assert moved == 0.0 and abs(z[-1] - z[-5]) < 1.0e-7                  # it holds
        return
    rate = (z[5] - z[-1]) / (0.02 * (len(z) - 6))
    assert rate == pytest.approx((load - 1.0) * bulk.bearing / bulk.resistance, rel=0.05, abs=0.0)
    assert abs(b["bulk_volume"][0]) < 1.0e-12 * moved                       # displaced, not made or lost
    x = flow.cell_centres(0)[..., :2]
    rim = np.abs(np.hypot(x[..., 0], x[..., 1]) - radius - 1.25e-3) < 1.5e-3
    assert flow.base.numpy()[0].reshape(48, 48)[rim].mean() > 0.0          # the heave beside the punch


def test_the_drag_cap_is_reported_when_it_binds():
    """WP2 review: the explicit viscous drag is capped for stability; read() says by how much (tool_drag_scale) and
    how often (tool_drag_capped). A light free tool on a thin viscous film binds the cap; a heavy one does not."""
    def capped(density: float) -> tuple[float, int]:
        flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(80, 4)),), material=OIL, max_depth=3.0e-3,
                           tools=(blade(stiffness=1.0e3),))
        run(flow, scene(5.0e-5, -0.06, (0.0, 0.0, 0.0), density=density, guided=True), 1.0e-4, 0.1,
            (0.0, 0.0, 0.0), periods=5)
        r = flow.read()
        return float(r["tool_drag_scale"][0, 0]), int(r["tool_drag_capped"][0, 0])
    light, heavy = capped(20.0), capped(2.0e6)
    assert light[0] < 1.0 and light[1] > 0
    assert heavy == (1.0, 0)


def _jointer(implicit: bool, stiffness: float = 5.0e7):
    """WP6's jointer: a 0.12 kg rod, its 137 x 12.7 mm sole set down on a plate at its elastic rest, dt 1.25 ms."""
    half = (0.0685, 0.00635, 0.004)
    density = 0.12 / (8.0 * half[0] * half[1] * half[2])
    sole = Applicator(body="tool", shape="rect", size=half[:2], center=(0.0, 0.0, -half[2] - DROP), stiffness=stiffness,
                      damping=1.0e-4, implicit=implicit)
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(80, 4)),), material=OIL, max_depth=3.0e-3, tools=(sole,))

    def build(b: newton.ModelBuilder) -> None:
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, -0.01), hx=0.5 * LENGTH, hy=0.5 * WIDTH, hz=0.01, label="plate")
        tool = b.add_body(xform=body_xform(0.0, 0.0, DROP + half[2]), label="tool")
        b.add_shape_box(tool, hx=half[0], hy=half[1], hz=half[2], cfg=newton.ModelBuilder.ShapeConfig(density=density))
    sim = build_simulation(TestScene(build=build, ground=False, gravity=(0.0, 0.0, -G), worlds=1, dt=1.25e-3), [flow],
                           "cpu")
    return flow, sim, 4.0 * half[0] * half[1]


def test_an_implicit_pad_holds_a_light_stiff_tool_the_explicit_one_cannot():
    """A 0.12 kg jointer on a 5e7 Pa/m pad at 1.25 ms: 100x the explicit bound (refused, naming the fix). Implicit,
    it settles where its weight balances the pad, m·g/(k·A) (~13 µm on its covered area), still and level."""
    with pytest.raises(PluginError, match="unstable"):
        _jointer(False)[1].step_period()                                     # check_dt runs at the first step
    flow, sim, area = _jointer(True)
    for _ in range(10):
        sim.step_period()
    q, qd = sim.state.body_q.numpy()[0], sim.state.body_qd.numpy()[0]
    give = -(float(q[2]) - 0.004 - DROP)                                     # the sole below the plate's top
    r = flow.read()
    covered = float((r["tool_cover"][0] * flow.tables.area_np).sum())         # the pad's area on the cells
    assert covered == pytest.approx(area, rel=0.1, abs=0.0)
    assert give == pytest.approx(0.12 * G / (5.0e7 * covered), rel=0.01, abs=0.0)
    assert np.abs(qd).max() < 1.0e-4 and abs(float(q[3])) + abs(float(q[4])) < 1.0e-6     # at rest, not tipped
    assert float(r["tool_pad_ratio"][0, 0]) == pytest.approx(1.0, rel=0.05, abs=0.0)


def _roll(nx: int) -> tuple[float, float]:
    """A blade ploughing 2 mm of a standing paste (h_c 4.2 mm) at a 0.5 mm gap for 0.4 s: the roll's apex above the
    layer ahead of it (the top cell's mean plus half a cell at 45°) and its cross-section per width."""
    from alloy.plugins.surface_flow import laws as sl  # noqa: PLC0415

    h0, gap, U, x0 = 2.0e-3, 0.5e-3, 0.1, -0.06
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(nx, 4)),), material=sl.PASTE, inertia=False,
                       max_depth=0.03, tools=(blade(stiffness=1.0e3),))
    sim = run(flow, scene(gap, x0, (0.0, 0.0, 0.0), density=2.0e6, guided=True), h0, U, (0.0, 0.0, -G), periods=20)
    x = flow.cell_centres(0)[:, 0, 0]
    h = flow.depth(0)[0].mean(axis=1)
    front = float(sim.state.body_q.numpy()[0][0]) + TOOL_HALF[0]
    ahead = x > front
    dx = LENGTH / nx
    apex = float(h[ahead].max() - h0) + 0.5 * dx                       # the top cell's mean is half a cell below
    return apex, float(((h[ahead] - h0).clip(min=0.0)).sum() * dx)


def test_a_tools_roll_stands_as_a_wedge_whatever_the_cell_size():
    """WP6: the film a tool ploughs piled into the one row ahead of it, so the roll's height grew as the cells shrank.
    Now it stands at 45°: height sqrt(2A) for a cross-section A, at 2.5 mm and 1.25 mm cells alike."""
    coarse, fine = _roll(80), _roll(160)
    for apex, section in (coarse, fine):
        assert section > 5.0e-5                                         # the blade pushed a roll
        assert apex == pytest.approx(math.sqrt(2.0 * section), rel=0.1, abs=0.0), (apex, section)
    assert fine[0] == pytest.approx(coarse[0], rel=0.1, abs=0.0)        # piled in one row it would double


def test_a_rods_roll_zone_is_as_wide_as_the_rod():
    """SOFT-2's jointer: a rod (a roller, radius 3 mm, 60 mm long) sliding along its axis marks its roll zone only
    across its own width (the footprint's extent across the slide), not its reach (60 mm): the zone ahead of it
    spans |y| <= radius + a cell."""
    R, HW = 0.003, 0.06
    rod = Applicator(body="tool", shape="roller", size=(R, HW), axis=(1.0, 0.0, 0.0),
                     center=(0.0, 0.0, -TOOL_HALF[2] - DROP + R), stiffness=1.0e3)
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(80, 20)),), material=OIL, max_depth=3.0e-3, tools=(rod,))
    run(flow, scene(0.5e-3, -0.03, (0.0, 0.0, 0.0), density=2.0e6, guided=True), 1.0e-3, 0.05, (0.0, 0.0, -G),
        periods=1)
    y = flow.cell_centres(0)[..., 1].reshape(-1)
    push = np.abs(flow.tf.push.numpy()[0]).sum(axis=1)
    zone = (flow.tf.cover.numpy()[0] <= 0.0) & (push > 0.0)
    assert zone.sum() > 0
    assert np.abs(y[zone]).max() <= R + WIDTH / 20, np.abs(y[zone]).max()


def test_a_ploughed_film_keeps_its_concentration():
    """WP9's review: the plough and the roll moved film without its species, so a pad left soap behind (10 kg/m³ became
    4e4 in places). A blade ploughing soapy oil at 2 kg/m³ now moves the soap with it: wherever film stands the
    concentration stays 2 (to 1e-9), and the soap's books close (1e-14)."""
    C0 = 2.0
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(80, 4)),), material=OIL, max_depth=3.0e-3,
                       species=("soap",), tools=(blade(stiffness=1.0e3),))

    def init(sim) -> None:
        flow.set_depth(1.0e-3)
        flow.set_species(C0 * 1.0e-3, "soap")
        qd = sim.state.joint_qd.numpy()
        qd[:] = 0.05
        sim.state.joint_qd.assign(qd)
        newton.eval_fk(sim.model, sim.state.joint_q, sim.state.joint_qd, sim.state)
    sim = build_simulation(TestScene(build=scene(0.4e-3, -0.06, (0.0, 0.0, 0.0), density=2.0e6, guided=True),
                                     init=init, ground=False, gravity=(0.0, 0.0, -G), worlds=1, dt=2.0e-3),
                           [flow], "cpu")
    start = None
    for n in range(40):
        sim.step_period()
        if n == 0:
            start = float(flow.read()["species_ledger"][0, 0])
    r = flow.read()
    h, m = r["h"][0], r["species"][0, :, 0]
    wet = h > 1.0e-6
    assert r["h"][0].max() > 1.05e-3                                   # it ploughed a roll
    assert np.abs(m[wet] / h[wet] / C0 - 1.0).max() < 1.0e-9
    assert abs(float(r["species_ledger"][0, 0]) - start) <= 1.0e-14 * start


def test_a_tool_over_a_deep_base_works_its_surface():
    """WP6's margin trowel: lift-off was measured from the chart's plane, so a small tool working a paste base 85 mm
    above the plane (its lowest point 75 mm up) got no window: no ceiling, no plough, no nip. Lift-off is per cell
    from the surface (base + h): a blade 0.4 mm into 1 mm of oil on an 85 mm base ploughs it as on a bare plate."""
    B = 0.085
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(80, 4)),), material=OIL, max_depth=3.0e-3,
                       tools=(blade(stiffness=1.0e3),))

    def init(sim) -> None:
        flow.set_depth(1.0e-3)
        flow.base.assign(np.full(flow.base.shape, B, np.float32))
        sim.state.joint_qd.assign(np.full(1, 0.05, np.float32))
        newton.eval_fk(sim.model, sim.state.joint_q, sim.state.joint_qd, sim.state)
    sim = build_simulation(TestScene(build=scene(B + 0.4e-3, -0.06, (0.0, 0.0, 0.0), density=2.0e6, guided=True),
                                     init=init, ground=False, gravity=(0.0, 0.0, -G), worlds=1, dt=2.0e-3),
                           [flow], "cpu")
    for _ in range(5):
        sim.step_period()
    r = flow.read()
    assert (r["tool_cover"][0] > 0.0).sum() >= 8
    assert float(r["h"][0].max()) > 1.0e-3                              # the plough's roll ahead of it


@pytest.mark.parametrize("stride, dt", [(1, 2.0e-3), (8, 2.0e-3), (1, 1.0e-3)])
def test_a_porous_pad_absorbs_the_film_above_its_underside(stride, dt):
    """WP9's sponge: a porous pad with room absorbs what it would plough. Held 0.4 mm over 1 mm of oil, its first film
    step takes Σ cover·(h - gap)·area into tf.absorb plus the little that flows in under it during that step (0.08 %),
    to 0.2 % whatever the stride or dt (the plough made WP9's take 5x step-dependent), its species with it; the
    film's books close with film_absorbed. Without room it ploughs and absorbs nothing."""
    h0, gap, C0 = 1.0e-3, 0.4e-3, 3.0
    out = {}
    for room in (1.0e-3, 0.0):
        sponge = Applicator(body="tool", shape="rect", size=(TOOL_HALF[0], 0.035),
                            center=(0.0, 0.0, -TOOL_HALF[2] - DROP), stiffness=1.0e3, porous=True)
        flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(80, 4)),), material=OIL, max_depth=3.0e-3,
                           species=("soap",), tools=(sponge,), max_stride=stride)

        def init(sim, flow=flow, room=room) -> None:
            flow.set_depth(h0)
            flow.set_species(C0 * h0, "soap")
            flow.tool_room.assign(np.full(flow.tool_room.shape, room))
        sim = build_simulation(TestScene(build=scene(gap, 0.0, (0.0, 0.0, 0.0), density=2.0e6, guided=True),
                                         init=init, ground=False, gravity=(0.0, 0.0, -G), worlds=1, dt=dt),
                               [flow], "cpu")
        start = None
        while int(flow.tick.numpy()[0]) < flow.stride:                  # through the first film step
            sim.step_period()
            if start is None:
                start = flow.read()
        out[room] = (flow.read(), start)
    r, _ = out[1.0e-3]
    cover, area = r["tool_cover"][0], flow.tables.area_np
    want = float((cover * (h0 - gap) * area).sum())
    got = float(r["film_absorbed"][0])
    assert want > 0.0 and want <= got <= want * 1.002, (got, want)
    assert abs(float(r["ledger"][0]) - h0 * area.sum()) <= 1e-12 * h0 * area.sum()
    assert float(r["species_ledger"][0, 0]) == pytest.approx(C0 * h0 * area.sum(), rel=1e-12, abs=0.0)
    assert float(out[0.0][0]["film_absorbed"][0]) == 0.0


def test_the_ceiling_readout_is_the_underside_over_its_cells():
    """read()["tool_ceiling"] (WP6's touch test): a blade held 0.4 mm over the plate reads its underside's height
    above the chart where it covers a cell, and +inf everywhere else."""
    gap = 0.4e-3
    blade = Applicator(body="tool", shape="rect", size=(TOOL_HALF[0], 0.035), center=(0.0, 0.0, -TOOL_HALF[2] - DROP),
                       stiffness=1.0e3)
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(80, 4)),), material=OIL, max_depth=3.0e-3, tools=(blade,))
    sim = build_simulation(TestScene(build=scene(gap, 0.0, (0.0, 0.0, 0.0), density=2.0e6, guided=True),
                                     init=lambda sim: flow.set_depth(1.0e-3), ground=False, gravity=(0.0, 0.0, -G),
                                     worlds=1, dt=2.0e-3), [flow], "cpu")
    sim.step_period()
    r = flow.read()
    cover, ceil = r["tool_cover"][0], r["tool_ceiling"][0]
    assert (cover > 0.0).sum() >= 8
    assert np.allclose(ceil[cover > 0.0], gap, rtol=1.0e-3, atol=0.0), ceil[cover > 0.0]
    assert np.isinf(ceil[cover == 0.0]).all()


def test_the_work_fields_read_every_tool_a_store_too():
    """SOFT-2's sealant nose for WP6's paste layer: a store tool (here a blade with a store, held 0.4 mm over the
    plate) writes the work fields (``tool_work_ceiling``/``_cover``/``_vel``) as a ceiling tool writes the film's, but
    not the film's own ceiling, cover or velocity, so the film never feels it as a ceiling."""
    from alloy.plugins.surface_flow.stores import Store  # noqa: PLC0415

    gap = 0.4e-3
    out = {}
    for store in (None, Store(capacity=1.0e-6, scrape=True, split=1.0)):
        blade = Applicator(body="tool", shape="rect", size=(TOOL_HALF[0], 0.035),
                           center=(0.0, 0.0, -TOOL_HALF[2] - DROP), stiffness=1.0e3, store=store)
        flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(80, 4)),), material=OIL, max_depth=3.0e-3,
                           tools=(blade,))
        sim = build_simulation(TestScene(build=scene(gap, 0.0, (0.0, 0.0, 0.0), density=2.0e6, guided=True),
                                         init=lambda sim, f=flow: f.set_depth(1.0e-4), ground=False,
                                         gravity=(0.0, 0.0, -G), worlds=1, dt=2.0e-3), [flow], "cpu")
        sim.step_period()
        out[store is None] = flow.read()
    ceiling, stored = out[True], out[False]
    wcover, wceil = stored["tool_work_cover"][0], stored["tool_work_ceiling"][0]
    assert (wcover > 0.0).sum() >= 8
    assert np.array_equal(wcover, ceiling["tool_cover"][0])                  # where a ceiling tool would read
    assert np.array_equal(wceil, ceiling["tool_ceiling"][0])
    assert np.array_equal(stored["tool_work_vel"][0], ceiling["tool_vel"][0])
    assert np.allclose(wceil[wcover > 0.0], gap, rtol=1.0e-3, atol=0.0)
    assert (stored["tool_cover"][0] == 0.0).all() and np.isinf(stored["tool_ceiling"][0]).all()   # not the film's
    assert np.array_equal(ceiling["tool_work_ceiling"][0], ceiling["tool_ceiling"][0])          # a ceiling's: both


def test_an_edger_sole_follows_gas_profile():
    """shape="edger" (WP4's pad-edge; GA paste.py:557-566): held level 5 mm over a dry plate whose -y edge is the form,
    its ceiling is flat away from the lip side and follows the quarter round down toward the lip, cell by cell to the
    pose's float32. The lip itself runs down the form face, past the plate."""
    g, hy, r, lip_w = 5.0e-3, 0.02, 9.5e-3, 1.5e-3
    cy = -0.5 * WIDTH - 2.0e-3 + hy                                 # the lip's face 2 mm past the plate's edge
    edger = Applicator(body="tool", shape="edger", size=(0.015, hy), center=(0.0, cy, -TOOL_HALF[2] - DROP),
                       stiffness=1.0e3, edge_radius=r, edge_lip=12.7e-3, edge_lip_w=lip_w, edge_side=1)
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(16, 40)),), material=OIL, max_depth=3.0e-3, tools=(edger,))
    sim = build_simulation(TestScene(build=scene(g, 0.0, (0.0, 0.0, 0.0), density=2.0e6, guided=True), ground=False,
                                     gravity=(0.0, 0.0, -G), worlds=1, dt=2.0e-3), [flow], "cpu")
    sim.step_period()
    r_ = flow.read()
    cover, ceil = r_["tool_cover"][0], r_["tool_ceiling"][0]
    y = flow.cell_centres(0)[..., 1].reshape(-1)
    d = hy + (y - cy)                                               # in from the lip's face (+e2 is -y here)
    z = np.where((d >= 0.0) & (d < r), -(r - np.sqrt(np.maximum(r * r - (r - d) ** 2, 0.0))), 0.0)
    want = g + np.where(d < lip_w, -12.7e-3, z)
    on = cover > 0.0
    assert on.sum() >= 40 and (on & (want < g - 1.0e-4)).sum() >= 8   # the round is resolved
    assert np.allclose(ceil[on], want[on], rtol=0.0, atol=1.0e-8), np.abs(ceil[on] - want[on]).max()
    for bad in ({"edge_radius": 0.0}, {"edge_radius": r, "edge_lip_w": r}, {"edge_radius": r, "edge_side": 0}):
        with pytest.raises(PluginError, match="edge"):
            Applicator(body="tool", shape="edger", size=(0.015, hy), **bad)
