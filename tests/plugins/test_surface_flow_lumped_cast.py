# SPDX-License-Identifier: Apache-2.0
"""Surface-flow lumped bodies: the drip cast's corner cases (lumped_drip.py; WP15's fourth review of the lumped films:
probes s1, s5, s7 and the M3b-M3d, M9-M12 mutants). CPU, harness."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton

from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow
from alloy.plugins.surface_flow.lumped_specs import Lumped
from alloy.plugins.surface_flow.supply import Pool

R, DT, G, RD = 0.014, 2.0e-3, (0.0, 0.0, -9.81), 1.5e-3
DROP = 4.0 / 3.0 * math.pi * RD ** 3
HOLD = 5.0e-5 * 0.01                                     # Lumped(area=0.01): its clinging film


def _fixed(b, label: str, xf) -> int:
    body = b.add_link(xform=xf, label=label)
    b.add_articulation([b.add_joint_fixed(-1, body, parent_xform=xf)])
    return body


def _dummy(b) -> None:
    d = b.add_body(xform=body_xform(0.5, 0.0, 1.0), label="dummy")
    b.add_shape_sphere(d, radius=0.01)


def _sim(flow: SurfaceFlow, build, init):
    return build_simulation(TestScene(build=build, init=init, ground=False, gravity=G, worlds=1, dt=DT, control_dt=DT),
                            [flow], "cpu")


def _stream(flow: SurfaceFlow, q: float, c: float = 0.0):
    """q m³ per exchange into every lumped body, at c kg/m³ of its first species."""
    def init(sim) -> None:
        inp = flow.lumped.inputs
        inp.vol.fill_(q)
        if c:
            sp = inp.sp.numpy().copy()
            sp[..., 0] = c * q
            inp.sp.assign(sp)
    return init


# --- s1, s7, M3b: a thin-walled beaker does not drip into itself; a sunk body rests ----------------------------------

T, RB = 2.0e-3, 2.7e-3                                     # the beaker's base and floor thickness, the drop radius


def _beaker_and_sunk(b) -> None:
    b.add_shape_box(-1, xform=body_xform(0.0, 0.0, -0.01), hx=0.2, hy=0.1, hz=0.01, label="slab")
    b.add_shape_sphere(_fixed(b, "tomato", body_xform(-0.1, 0.0, R - 2.5e-3)), radius=R)    # sunk 2.5 mm < r
    k = _fixed(b, "beaker", body_xform(0.1, 0.0, 0.05))
    b.add_shape_box(k, xform=body_xform(0.0, 0.0, -T / 4), hx=0.03, hy=0.03, hz=T / 4, label="base")
    b.add_shape_box(k, xform=body_xform(0.0, 0.0, T / 4), hx=0.03, hy=0.03, hz=T / 4, label="floor")
    _dummy(b)


@pytest.mark.compile_heavy
def test_a_thin_beaker_drips_outside_and_a_sunk_tomato_rests():
    """WP15's should-fix 1 (probes s1, s7; mutant M3b). A beaker 50 mm over a slab whose lumped outside is its base,
    1 mm thick, under its own inner floor chart (1 mm), with 2.7 mm drops: the cast starts a drop radius above its
    lowest point, above its own floor, but a chart on the body itself above that point is not under it, so its soapy
    water drips onto the slab and its floor stays dry (it filled its own floor through the base before). A tomato
    sunk 2.5 mm (< r) into the slab rests on it and wets it. Books: shed = wet + waiting + drops; drops = landed + in
    flight, none floored; each body's soap = wet + waiting + dropped (1e-12)."""
    flow = SurfaceFlow(charts=(PlaneChart("slab", cells=(20, 10)), PlaneChart("floor", cells=(6, 6))),
                       material="water", max_depth=5.0e-3, species=("soap",),
                       drops={"drop_radius": RB, "floor_z": -0.5},
                       lumped=(Lumped("tomato", area=0.01), Lumped("beaker", shapes=("base",), area=0.01)))
    sim = _sim(flow, _beaker_and_sunk, _stream(flow, 1.0e-7, 4.0))
    for _ in range(80):
        sim.step_period()
    r, d = flow.lumped.read(), flow.drops.read()
    floor = flow.tables.cells_of(1)
    assert flow.h.numpy()[0][floor].max() == 0.0 and flow.src_acc.numpy()[0][floor].max() == 0.0
    sp = {k: r[f"lump_species_{k}"][0, :, 0] for k in ("shed", "wet", "drip_pending", "dropped")}
    assert sp["wet"][0] > 0.0 and sp["wet"][1] == 0.0 and sp["dropped"][0] == 0.0 and sp["dropped"][1] > 0.0
    np.testing.assert_allclose(sp["shed"], sp["wet"] + sp["drip_pending"] + sp["dropped"], rtol=1e-12, atol=0.0)
    assert r["lump_shed"][0].sum() == pytest.approx(d["lump_wet_volume"][0] + d["lump_pending_volume"][0]
                                                     + d["lump_drops_volume"][0], rel=1e-12, abs=0.0)
    assert d["landed_volume"][0] > 0.0 and d["floored_volume"][0] == 0.0
    assert d["lump_drops_volume"][0] == pytest.approx(d["landed_volume"][0] + d["in_flight_volume"][0], rel=1e-12,
                                                      abs=0.0)


# --- M3c, M3d: the nearest of two charts under a body; a sloped board's substrate along gravity ----------------------

TH, BASE = math.radians(60.0), 2.5e-3
TILT = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), TH)
ROT = np.array(wp.quat_to_matrix(TILT), np.float64).reshape(3, 3)
BOARD = np.array([0.3, 0.0, 0.0])


def _tray_and_slope(b) -> None:
    b.add_shape_box(-1, xform=body_xform(-0.2, 0.0, -0.01), hx=0.15, hy=0.15, hz=0.01, label="slab")
    b.add_shape_box(-1, xform=body_xform(-0.2, 0.0, 0.025), hx=0.05, hy=0.05, hz=0.005, label="tray")   # top 30 mm
    b.add_shape_sphere(_fixed(b, "tomato", body_xform(-0.2, 0.0, 0.05 + R)), radius=R)    # 20 mm over the tray
    b.add_shape_box(-1, xform=wp.transform(wp.vec3(*BOARD), TILT), hx=0.1, hy=0.1, hz=0.005, label="board")
    foot = BOARD + ROT @ np.array([0.0, 0.0, 0.005])        # on the board's top, at its centre
    b.add_shape_sphere(_fixed(b, "egg", body_xform(*(foot + np.array([0.0, 0.0, 7.0e-3 + R])))), radius=R)
    _dummy(b)


@pytest.mark.compile_heavy
def test_drips_take_the_nearest_chart_and_a_sloped_substrate_along_gravity():
    """WP15's mutants M3c, M3d. A tomato 20 mm over a tray that stands 30 mm over a slab sheds 30 drops' worth: the
    nearest chart under it (the tray) limits the column (6 drops a radius over it), so all 30 land on the tray and the
    slab stays dry. An egg 7 mm straight over a 60° board under a 2.5 mm substrate (base) is 2 mm over its top along
    gravity (2.5 mm / cos 60°), within 2r: it rests and gives its 10 drops' worth to the board, no drops (4.5 mm, and
    drips, when the substrate is measured along the board's normal)."""
    flow = SurfaceFlow(charts=(PlaneChart("slab", cells=(10, 10)), PlaneChart("tray", cells=(10, 10)),
                               PlaneChart("board", cells=(20, 20))), material="water", max_depth=5.0e-3,
                       drops={"drop_radius": RD, "floor_z": -0.5},
                       lumped=(Lumped("tomato", area=0.01), Lumped("egg", area=0.01)))

    def init(sim) -> None:
        flow.lumped.set_water(np.array([[HOLD + 30.2 * DROP, HOLD + 10.0 * DROP]]))
        base = np.zeros(flow.base.shape, np.float32)
        base[0, flow.tables.cells_of(2)] = BASE
        flow.base.assign(base)
    sim = _sim(flow, _tray_and_slope, init)
    sim.step_period()
    sim.step_period()
    alive = flow.drops.pool.alive.numpy()[0] == 1
    assert alive.sum() == 6
    for _ in range(78):
        sim.step_period()
    r, d = flow.lumped.read(), flow.drops.read()
    assert d["lump_drops_volume"][0] == pytest.approx(30 * DROP, rel=1e-12, abs=0.0)
    assert d["landed_volume"][0] == pytest.approx(30 * DROP, rel=1e-12, abs=0.0) and d["floored_volume"][0] == 0.0
    assert flow.h.numpy()[0][flow.tables.cells_of(0)].max() == 0.0                   # the slab stays dry
    assert d["lump_wet_volume"][0] == pytest.approx(r["lump_shed"][0, 1], rel=1e-12, abs=0.0)


# --- M9, M10: a drip starts in a pool's water a drop radius under the lowest point, and over its floor ---------------

LEVEL = -0.25                                              # the 2 L bowl's level: floor -0.3 + 2 L / 0.04 m²


def _by_the_level(b) -> None:
    b.add_shape_box(-1, xform=body_xform(0.0, 0.0, 0.5), hx=0.1, hy=0.1, hz=0.01, label="plate")
    b.add_shape_sphere(_fixed(b, "low", body_xform(-0.06, 0.0, LEVEL + 0.75e-3 + R)), radius=R)
    b.add_shape_sphere(_fixed(b, "high", body_xform(0.0, 0.0, LEVEL + 2.25e-3 + R)), radius=R)
    b.add_shape_sphere(_fixed(b, "under", body_xform(0.06, 0.0, -0.35)), radius=R)
    _dummy(b)


@pytest.mark.compile_heavy
def test_where_a_drip_starts_decides_if_it_joins_the_pool():
    """WP15's mutants M9, M10: three tomatoes over a 2 L bowl under a stream, their lowest points 0.75 mm and 2.25 mm
    over its level and one 36 mm under its floor (r 1.5 mm). The first's drips start 0.75 mm under the water: they
    join the tank (lump_tanks: shed = tanks + waiting, 1e-12). The second's start 0.75 mm over it: it drips, and its
    drops join the tank as they cross the water (to_tanks, not lump_tanks). The third's start under the box pool's
    floor: they fall on, below it (drops made = to_tanks + in flight + floored, 1e-9)."""
    net = {"nodes": [{"name": "bowl", "medium": "tank", "volume": 2.0e-3, "area": 0.04}], "edges": [],
           "pools": [Pool("bowl", floor=(0.0, 0.0, -0.3), half=(0.1, 0.1))]}
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(4, 4)),), material="water", network=net,
                       drops={"drop_radius": RD, "floor_z": -1.0},
                       lumped=(Lumped("low", area=0.01), Lumped("high", area=0.01), Lumped("under", area=0.01)))
    sim = _sim(flow, _by_the_level, _stream(flow, 1.0e-7))
    for _ in range(60):
        sim.step_period()
    r, d, s = flow.lumped.read(), flow.drops.read(), flow.supply.read()
    assert LEVEL - 1.0e-6 < float(flow.supply.level.numpy()[0, 0]) < LEVEL + 0.5e-3     # risen less than 0.75 mm
    tanks, shed = r["lump_drip_tanks"][0], r["lump_shed"][0]
    assert tanks[0] > 0.0 and tanks[1] == 0.0 and tanks[2] == 0.0
    assert shed[0] == pytest.approx(tanks[0] + r["lump_drip_pending"][0, 0], rel=1e-12, abs=0.0)
    assert d["to_tanks_volume"][0] > 0.0 and shed[1] > 0.0 and shed[2] > 0.0
    assert d["lump_drops_volume"][0] == pytest.approx(d["to_tanks_volume"][0] + d["in_flight_volume"][0]
                                                      + d["floored_volume"][0], rel=1e-12, abs=0.0)
    assert s["lump_pool"][0] == pytest.approx(tanks.sum() + d["to_tanks_volume"][0], rel=1e-12, abs=0.0)


# --- M11, M12: each world drips from its own body's pose onto its own charts ----------------------------------------

def _world_scene(x: float, gap: float, tilt: float):
    def build(b) -> None:
        q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), tilt)
        b.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.0, -0.01), q), hx=0.15, hy=0.1, hz=0.01, label="slab")
        rot = np.array(wp.quat_to_matrix(q), np.float64).reshape(3, 3)
        n = rot @ np.array([0.0, 0.0, 1.0])
        top = (n @ (rot @ np.array([0.0, 0.0, 0.0]))) - n[0] * x                 # the slab's top over x: n·p = 0
        b.add_shape_sphere(_fixed(b, "tomato", body_xform(x, 0.0, top / n[2] + gap + R)), radius=R)
        _dummy(b)
    return build


def _per_world(flow: SurfaceFlow, builds, init):
    """One template per world (same structure, other poses: ModelBuilder.add_world; WP15's probe s5)."""
    from alloy.plugins.host import simulation  # noqa: PLC0415
    from alloy.scene.builtin_plugins import register_builtin_solvers  # noqa: PLC0415
    from alloy.testing.integration import built_from_model, entry_spec  # noqa: PLC0415

    def make(plugins, device, worlds, graph):
        register_builtin_solvers()
        root = newton.ModelBuilder(gravity=G)
        for build in builds:
            t = newton.ModelBuilder(gravity=G)
            build(t)
            root.add_world(t)
        model = root.finalize(device=device)
        return simulation(built_from_model(model, [entry_spec(model, "mujoco", DT)], control_dt=DT), list(plugins))
    return build_simulation(TestScene(build=builds[0], make=make, init=init, ground=False, gravity=G,
                                      worlds=len(builds), dt=DT, control_dt=DT), [flow], "cpu")


@pytest.mark.compile_heavy
def test_each_world_drips_from_its_own_pose_onto_its_own_charts():
    """WP15's mutants M11, M12 (probe s5): world 0's tomato rests 0.5 mm over a flat slab, world 1's hovers 20 mm over
    its slab tilted 30° at x = 50 mm (the chart's frame differs per world), each with 10 drops' worth: world 0 wets its
    slab, no drops; world 1 drips 10 drops that all land on its tilted slab, none floored."""
    flow = SurfaceFlow(charts=(PlaneChart("slab", cells=(30, 20)),), material="water", max_depth=5.0e-3,
                       drops={"drop_radius": RD, "floor_z": -0.5}, lumped=(Lumped("tomato", area=0.01),))
    sim = _per_world(flow, [_world_scene(0.0, 5.0e-4, 0.0), _world_scene(0.05, 0.02, math.radians(30.0))],
                     lambda sim: flow.lumped.set_water(HOLD + 10.2 * DROP))
    for _ in range(100):
        sim.step_period()
    d = flow.drops.read()
    assert d["lump_wet_volume"][0] == pytest.approx(10.2 * DROP, rel=1e-12, abs=0.0)
    assert d["lump_drops_volume"][0] == 0.0
    assert d["lump_wet_volume"][1] == 0.0 and d["lump_drops_volume"][1] == pytest.approx(10 * DROP, rel=1e-12, abs=0.0)
    assert d["landed_volume"][1] == pytest.approx(10 * DROP, rel=1e-12, abs=0.0) and d["floored_volume"].max() == 0.0


# --- WP15's round-5 nit: a body drips onto a chart it carries below its lowest point ---------------------------------

def _pan_with_handle(b) -> None:
    b.add_shape_box(-1, xform=body_xform(0.0, 0.0, -0.01), hx=0.15, hy=0.15, hz=0.01, label="slab")
    k = _fixed(b, "pan", body_xform(0.0, 0.0, 0.0))
    b.add_shape_box(k, xform=body_xform(0.0, 0.0, 0.0275), hx=0.05, hy=0.05, hz=0.0025, label="panfloor")  # top 30 mm
    b.add_shape_box(k, xform=body_xform(0.0, 0.0, 0.066), hx=0.01, hy=0.01, hz=0.005, label="handle")      # bottom 61
    _dummy(b)


@pytest.mark.compile_heavy
def test_a_handle_drips_onto_its_own_pans_floor():
    """WP15's round-5 nit (the own-body skip only drops hits above the lowest point): a pan's lumped handle 31 mm
    over the pan's own floor chart sheds 30 drops' worth: its floor, on the same body but below the handle, is the
    chart under it, so the first column is the 10 drops a radius over it and all 30 land on it; the slab 30 mm below
    the floor stays dry (fails when the body's own charts are skipped)."""
    flow = SurfaceFlow(charts=(PlaneChart("slab", cells=(10, 10)), PlaneChart("panfloor", cells=(10, 10))),
                       material="water", max_depth=5.0e-3, drops={"drop_radius": RD, "floor_z": -0.5},
                       lumped=(Lumped("pan", shapes=("handle",), area=0.01),))
    sim = _sim(flow, _pan_with_handle, lambda sim: flow.lumped.set_water(HOLD + 30.2 * DROP))
    sim.step_period()
    sim.step_period()
    assert (flow.drops.pool.alive.numpy()[0] == 1).sum() == 10
    for _ in range(78):
        sim.step_period()
    d = flow.drops.read()
    assert d["landed_volume"][0] == pytest.approx(30 * DROP, rel=1e-12, abs=0.0) and d["floored_volume"][0] == 0.0
    assert flow.h.numpy()[0][flow.tables.cells_of(0)].max() == 0.0
