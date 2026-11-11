# SPDX-License-Identifier: Apache-2.0
"""Cavity pools (supply_cavity.py) on the stub film, CPU: a vessel's water lies normal to gravity at any tilt; a tipped
glass pours over its rim's lowest point down to its capacity; a glass dunked in a sink fills over its rim (and floods
its inside from its own water), and an inverted one traps its air and stays dry inside."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy.plugins.errors import PluginError
from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.surface_flow.charts import PlaneChart, RevolutionChart
from alloy.plugins.surface_flow.supply import Nozzle, Pool
from alloy.plugins.surface_flow.supply_stub import StubFilm, ledger_sides

G = 9.81
R, HT = 0.03, 0.1                                       # a straight glass: 30 mm inner radius, 100 mm deep
A = math.pi * R * R
CYL = ((R, 0.0), (R, HT))
BRIM_RTOL = 5e-8         # the brim is float32 geometry: its rim height (+1.5e-8) and the vessel's up (~3e-9)
FAR = PlaneChart("far", cells=(2, 2))                   # the stub film needs a chart: a slab out of the way


def _far(b) -> None:
    b.add_shape_box(-1, xform=body_xform(2.0, 0.0, 0.01), hx=0.05, hy=0.05, hz=0.01, label="far")
    dummy = b.add_body(xform=body_xform(-2.0, 0.0, 0.5), label="dummy")
    b.add_shape_sphere(dummy, radius=0.01)


def _glass(volume: float, pour: bool = False) -> dict:
    edges = [{"name": "pour", "kind": "source", "a": "glass", "flow": -1.0e-3}] if pour else []
    return {"nodes": [{"name": "glass", "medium": "tank", "volume": volume, "area": A}], "edges": edges}


def _tilted(deg: float) -> tuple[float, float, float]:
    th = math.radians(deg)
    return (G * math.sin(th), 0.0, -G * math.cos(th))   # tilting gravity tilts the glass relative to it


def _sim(p: StubFilm, build, gravity=(0.0, 0.0, -G)):
    return build_simulation(TestScene(build=build, ground=False, gravity=gravity, worlds=2), [p], "cpu")


@pytest.mark.parametrize("tilt", [0.0, 20.0])
def test_a_tilted_glass_holds_its_surface_normal_to_gravity(tilt):
    """Half full: the plane meets the axis at half height whatever the tilt (a cylinder's volume below a plane that
    cuts its wall all round is πR² times the plane's height on the axis). It spills above the rim's lowest point
    (πR²·(H - R·tan θ)); air is trapped only above the highest (all of it)."""
    p = StubFilm(charts=(FAR,), network={**_glass(0.5 * A * HT), "pools": [Pool("glass", half=(R, R), cavity=CYL)]})
    _sim(p, _far, _tilted(tilt)).step_period()
    th = math.radians(tilt)
    surf, cap = p.supply.cav_surf.numpy()[0, 0], p.supply.cav_cap.numpy()[0, 0]
    assert surf[3] == pytest.approx(0.5 * HT * math.cos(th), rel=1e-5)
    assert cap[0] == pytest.approx(A * (HT - R * math.tan(th)), rel=1e-6)          # up from float32 poses
    assert cap[1] == pytest.approx(A * HT, rel=1e-9)


def test_a_conical_glass_level_matches_the_frustum():
    """Upright cone, 20 mm at the floor to 40 mm at the rim: the volume of the frustum to 50 mm sits at 50 mm (the
    slices are cylinders at their mid radius: second order in the slice height)."""
    cone = ((0.02, 0.0), (0.04, HT))
    v = math.pi / 3.0 * 0.05 * (0.02 ** 2 + 0.02 * 0.03 + 0.03 ** 2)
    p = StubFilm(charts=(FAR,), network={**_glass(v), "pools": [Pool("glass", cavity=cone)]})
    _sim(p, _far).step_period()
    assert p.supply.cav_surf.numpy()[0, 0][3] == pytest.approx(0.05, abs=1e-5)


def test_a_tipped_glass_holding_a_wedge_has_its_surface_through_the_floor_centre():
    """Tipped 20° holding (2/3)·R³·tan 20°: the water is the cylindrical wedge under a plane through the floor's
    centre. The plane cuts the floor, so the chord integral's closed form is checked across the floor too."""
    th = math.radians(20.0)
    p = StubFilm(charts=(FAR,), network={**_glass(2.0 / 3.0 * R ** 3 * math.tan(th)),
                                         "pools": [Pool("glass", cavity=CYL)]})
    _sim(p, _far, _tilted(20.0)).step_period()
    assert np.abs(p.supply.cav_surf.numpy()[:, 0, 3]).max() < 1e-7


def test_a_tipped_glass_spills_at_the_weir_rate_along_its_rim():
    """Filled to 95 mm and tipped 30°: the spill is the sharp-crested weir with its head integrated along the wetted
    rim, Q = (2/3)·Cd·sqrt(2g)·∫ H(φ)^(3/2)·R dφ, here against a fine quadrature at the surface the exchange used (the
    deepest head times the wetted length is 1.7-2x this)."""
    pool = Pool("glass", cavity=CYL, spill="pour")
    lip = Nozzle("lip", edge="pour", rim_of="glass", exit_area=1e-4)
    p = StubFilm(charts=(FAR,), nozzles=(lip,), network={**_glass(0.095 * A, pour=True), "pools": [pool]})
    _sim(p, _far, _tilted(30.0)).step_period()
    sv = p.supply.cav_prev.numpy()[:, 0].astype(np.float64)
    q = p.supply.fluid.opening.numpy()[:, 0] * 1.0e-3                    # the pour edge's rate at full opening
    n = 200_000
    phi = (np.arange(n) + 0.5) * (2.0 * math.pi / n)
    for w in range(2):
        z = sv[w, 0] * R * np.cos(phi) + sv[w, 1] * R * np.sin(phi) + sv[w, 2] * HT     # rim heights along up
        head = np.maximum(sv[w, 3] - z, 0.0)
        ref = 2.0 / 3.0 * 0.62 * math.sqrt(2.0 * G) * np.sum(head ** 1.5) * R * (2.0 * math.pi / n)
        assert 0.0 < q[w] < 0.9e-3 and q[w] == pytest.approx(ref, rel=2e-3), (q[w], ref)


def test_a_tipped_glass_pours_down_to_its_capacity_over_the_rim():
    """Filled to 90 mm and tipped 30°: what stands above the rim's lowest point leaves as a weir over the wetted rim,
    on the pour edge, poured by the rim nozzle (into nothing: lost). It drains toward its capacity and never below it
    (the head and the wetted rim shrink together, so the tail is slow), and the network books every m³."""
    th, v0 = math.radians(30.0), 0.09 * A
    pool = Pool("glass", cavity=CYL, spill="pour")                        # catches over its rim, never its own pour
    lip = Nozzle("lip", edge="pour", rim_of="glass", exit_area=1e-4)
    p = StubFilm(charts=(FAR,), nozzles=(lip,), network={**_glass(v0, pour=True), "pools": [pool]})
    sim = _sim(p, _far, _tilted(30.0))
    for _ in range(50):
        sim.step_period()
    r = p.supply.read()
    cap, v = A * (HT - R * math.tan(th)), r["amount"][:, 0]
    assert (v >= cap * (1.0 - 1e-9)).all() and (v0 - v > 0.6 * (v0 - cap)).all(), (v, cap, v0)
    assert np.allclose(-r["edge_ledger"][:, 0], v0 - v, rtol=1e-9, atol=0.0)        # the pour edge carried it
    lhs, rhs, spout, sent = ledger_sides(p, edge=0)
    assert np.allclose(lhs, rhs, rtol=1e-12, atol=0.0) and np.allclose(spout, sent, rtol=1e-9, atol=0.0)
    assert (r["lost"] > 0).all() and np.allclose(r["lost"], r["sent"], rtol=1e-12, atol=0.0)


def test_an_upright_glass_filled_past_its_rim_spills_down_to_it():
    """Filled 20 % past its brim, upright (WP15's review of sf14: the surface sat at the rim with no head, so it never
    spilled and overfilled without limit): the excess stands over the mouth, spills over the whole rim on the pour edge
    and the glass drains to its brim, never below; the network books every m³."""
    v0, cap = 1.2 * A * HT, A * HT
    pool = Pool("glass", cavity=CYL, spill="pour")
    lip = Nozzle("lip", edge="pour", rim_of="glass", exit_area=1e-4)
    p = StubFilm(charts=(FAR,), nozzles=(lip,), network={**_glass(v0, pour=True), "pools": [pool]})
    sim = _sim(p, _far)
    for _ in range(50):
        sim.step_period()
    r = p.supply.read()
    v = r["amount"][:, 0]
    assert (v >= cap * (1.0 - BRIM_RTOL)).all() and (v0 - v > 0.9 * (v0 - cap)).all(), (v, cap, v0)
    assert np.allclose(-r["edge_ledger"][:, 0], v0 - v, rtol=1e-9, atol=0.0)
    lhs, rhs, spout, sent = ledger_sides(p, edge=0)
    assert np.allclose(lhs, rhs, rtol=1e-12, atol=0.0) and np.allclose(spout, sent, rtol=1e-9, atol=0.0)


def _glass_profile(wall: float) -> tuple[tuple[float, float], ...]:
    """Inside floor, inner wall, rim, outer wall (the outer wall ``wall`` beyond the inner)."""
    return (0.006, 0.0), (R, 0.0), (R, HT), (R + wall, HT), (R + wall, 0.0)


GLASS = _glass_profile(0.003)


def _welded_glass(z: float, q: tuple[float, float, float, float]):
    def build(b) -> None:
        link = b.add_link(xform=body_xform(0.0, 0.0, z, q), label="glass")
        b.add_shape_sphere(link, xform=body_xform(0.0, 0.0, -0.05), radius=0.005)
        b.add_articulation([b.add_joint_fixed(-1, link, parent_xform=body_xform(0.0, 0.0, z, q))])
        _far(b)
    return build


@pytest.mark.parametrize(("inverted", "wall"), [(False, 0.003), (True, 0.003), (False, 0.0005)])
def test_a_glass_dunked_in_a_sink_fills_over_its_rim_unless_inverted(inverted, wall):
    """A glass welded upright (floor at z = 0.02, rim at 0.12) or turned over (floor at 0.14, rim at 0.04) in a sink
    whose surface is at 0.16. Upright, it fills over its rim to the brim, its inside floods from its own water and
    its outside from the sink's (also a 0.5 mm wall, thinner than the 1 mm cavity tolerance: the outer wall faces
    out). Inverted, its air stays trapped: nothing goes in and its inside stays dry."""
    q = (1.0, 0.0, 0.0, 0.0) if inverted else (0.0, 0.0, 0.0, 1.0)
    z = 0.14 if inverted else 0.02
    build = _welded_glass(z, q)
    profile = _glass_profile(wall)
    glass = Pool("glass", body="glass", cavity=CYL, flood=True, weir="sink")
    sink = Pool("sink", floor=(0.0, 0.0, 0.0), half=(0.2, 0.2), flood=True)
    net = {"nodes": [{"name": "glass", "medium": "tank", "volume": 0.0, "area": A},
                     {"name": "sink", "medium": "tank", "volume": 0.16 * 0.16, "area": 0.16}],
           "edges": [], "pools": [glass, sink]}
    chart = RevolutionChart(profile=profile, n_theta=24, ds=2e-3, body="glass")
    p = StubFilm(charts=(FAR, chart), network=net)
    sim = _sim(p, build)
    for _ in range(20):
        sim.step_period()
    r = p.supply.read()
    v_glass, v_sink = r["amount"][:, 0], r["amount"][:, 1]
    cells = p.tables.cells_of(1)
    nx, ny = p.tables.shapes[1]
    rows = np.arange(nx)
    prof = np.asarray(profile)
    s_nodes = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(prof, axis=0).T))])
    s_rows = (rows + 0.5) * s_nodes[-1] / nx
    inside = s_rows < s_nodes[2] - 1e-3                                    # the floor and the inner wall
    outside = s_rows > s_nodes[3] + 1e-3
    flooded = p.flooded.numpy()[0, cells].reshape(nx, ny)
    if inverted:
        assert np.allclose(v_glass, 0.0, atol=0.0) and np.allclose(v_sink, 0.16 * 0.16, rtol=1e-12, atol=0.0)
        assert (flooded[inside] == -1).all()
    else:
        assert np.allclose(v_glass, A * HT, rtol=BRIM_RTOL, atol=0.0)                # to the brim
        assert np.allclose(v_sink + v_glass, 0.16 * 0.16, rtol=1e-12, atol=0.0)
        assert np.allclose(r["weir"], v_glass, rtol=1e-9, atol=0.0)
        assert (flooded[inside] == 0).all()
    assert (flooded[outside] == 1).all()                                   # below the sink's surface all round


def test_a_draining_glass_leaves_a_film_on_its_inner_wall():
    """An upright glass drains through its floor at Q (a source edge): its surface falls past the inner wall at
    U = Q/A, and every wall cell that comes out of the water keeps the Landau-Levich film of U (the wall does not
    move; its water does), taken from the glass."""
    flow, nu, sigma, rho = 1.0e-4, 1.0e-6, 0.0728, 998.0
    net = {"nodes": [{"name": "glass", "medium": "tank", "volume": 0.09 * A, "area": A}],
           "edges": [{"name": "plug", "kind": "source", "a": "glass", "flow": -flow}],
           "pools": [Pool("glass", body="glass", cavity=CYL, flood=True)]}
    chart = RevolutionChart(profile=GLASS, n_theta=24, ds=2e-3, body="glass")
    p = StubFilm(charts=(FAR, chart), network=net)
    sim = _sim(p, _welded_glass(0.02, (0.0, 0.0, 0.0, 1.0)))
    cells = p.tables.cells_of(1)
    sim.step_period()                                   # the first exchange has no surface motion to go by
    wet0, w0 = p.flooded.numpy()[:, cells] >= 0, p.supply.read()["withdrawn"]
    for _ in range(15):
        sim.step_period()
    left = wet0 & (p.flooded.numpy()[:, cells] < 0)
    mu_u, rho_g = rho * nu * flow / A, rho * G
    h_ll = min(0.94 * mu_u ** (2.0 / 3.0) / (sigma ** (1.0 / 6.0) * math.sqrt(rho_g)), math.sqrt(mu_u / rho_g))
    area = p.tables.area_np[cells]
    assert (left.sum(axis=1) >= 3 * 24).all()            # three rows or more came out
    got = p.supply.read()["withdrawn"] - w0
    assert np.allclose(got, h_ll * (left * area).sum(axis=1), rtol=1e-3, atol=0.0), (got, h_ll)


def test_an_empty_glass_starts_its_pool_from_floor_film_a_drained_one_would_not_keep():
    """JOB3's empty garbage can: an empty flooding glass's inner floor film over what a cell leaving its water keeps
    (withdraw_max 1 mm here: the chart has no contact angle) is its water. World 0's floor holds 0.9 mm (a drained
    glass's film: it stays, the pool stays empty), its inner wall and its outer wall's bottom rows 1.5 mm (above the
    floor; outside the vessel), world 1's floor 1.2 mm: the floor floods, all of it joins the pool (pooled = what the
    floor held, 1e-12; the tank, from an absolute pressure, 1e-9), and the floor stays under the water."""
    net = {"nodes": [{"name": "glass", "medium": "tank", "volume": 0.0, "area": A}], "edges": [],
           "pools": [Pool("glass", body="glass", cavity=CYL, flood=True)]}
    chart = RevolutionChart(profile=GLASS, n_theta=24, ds=2e-3, body="glass")
    p = StubFilm(charts=(FAR, chart), network=net)
    sim = _sim(p, _welded_glass(0.02, (0.0, 0.0, 0.0, 1.0)))
    cells = np.arange(p.tables.cell_count)[p.tables.cells_of(1)]
    local = p.tables.cpos.numpy()[cells]
    floor = cells[(local[:, 2] < 1e-4) & (np.hypot(local[:, 0], local[:, 1]) < R - 1e-4)]
    h = p.h.numpy()
    r_cell = np.hypot(local[:, 0], local[:, 1])
    wall = cells[(local[:, 2] > 5e-3) & (local[:, 2] < HT - 5e-3) & (np.abs(r_cell - R) < 1e-4)]
    outer = cells[(local[:, 2] < 2e-3) & (r_cell > R + 1e-3)]
    h[0, floor], h[1, floor], h[0, wall], h[0, outer] = 0.9e-3, 1.2e-3, 1.5e-3, 1.5e-3
    p.h.assign(h)
    held = p.h.numpy()[:, floor] @ p.tables.area_np[floor]
    for _ in range(3):
        sim.step_period()
    r = p.supply.read()
    assert len(floor) >= 24 and r["amount"][0, 0] == 0.0 and (p.flooded.numpy()[0, floor] < 0).all()
    assert np.allclose(p.h.numpy()[0, floor], 0.9e-3, rtol=1e-12, atol=0.0)          # kept: a drained glass's film
    assert len(wall) >= 24 and np.allclose(p.h.numpy()[0, wall], 1.5e-3, rtol=1e-12, atol=0.0)
    assert len(outer) >= 24 and np.allclose(p.h.numpy()[0, outer], 1.5e-3, rtol=1e-12, atol=0.0)
    assert r["amount"][1, 0] == pytest.approx(held[1], rel=1e-9, abs=0.0)
    assert r["pooled"][1] == pytest.approx(held[1], rel=1e-12, abs=0.0) and r["pooled"][0] == 0.0
    assert (p.flooded.numpy()[1, floor] == 0).all() and (p.h.numpy()[1, floor] == 0.0).all()
    lhs, rhs, *_ = ledger_sides(p, edge=0)
    assert np.allclose(lhs, rhs, rtol=1e-12, atol=0.0)


def _empty_start(profile, cavity, gravity, h_seed, contact_angle=None, floor_only=False, volume=0.0):
    """An empty flooding glass (welded upright; ``gravity`` may tip it) with ``h_seed`` [W] of film on every inner
    cell (or its level floor's), three exchanges on: the supply's read, the seeded cells and what they held."""
    net = {"nodes": [{"name": "glass", "medium": "tank", "volume": volume, "area": A}], "edges": [],
           "pools": [Pool("glass", body="glass", cavity=cavity, flood=True)]}
    chart = RevolutionChart(profile=profile, n_theta=24, ds=2e-3, body="glass", contact_angle=contact_angle)
    p = StubFilm(charts=(FAR, chart), network=net)
    sim = _sim(p, _welded_glass(0.02, (0.0, 0.0, 0.0, 1.0)), gravity)
    cells = np.arange(p.tables.cell_count)[p.tables.cells_of(1)]
    if floor_only:
        local = p.tables.cpos.numpy()[cells]
        cells = cells[(local[:, 2] < 1e-4) & (np.hypot(local[:, 0], local[:, 1]) < cavity[0][0] - 1e-4)]
    h = p.h.numpy()
    h[:, cells] = np.asarray(h_seed)[:, None]
    p.h.assign(h)
    held = p.h.numpy()[:, cells] @ p.tables.area_np[cells]
    for _ in range(3):
        sim.step_period()
    return p.supply.read(), p, cells, held


def test_a_tipped_cone_starts_at_its_lowest_point():
    """WP15's review: an empty cone (20 mm floor, 40 mm rim) tipped 20° with 1.5 mm on its inside starts its pool at
    its inside's lowest point (the wall at the rim's low side, above the bounding plane an empty surface sits on);
    what the started cells held is pooled (1e-12)."""
    cone = ((0.02, 0.0), (0.04, HT))
    r, p, cells, held = _empty_start(((0.004, 0.0), *cone), cone, _tilted(20.0), [1.5e-3, 1.5e-3])
    lost = held - p.h.numpy()[:, cells] @ p.tables.area_np[cells]
    assert (r["amount"][:, 0] > 0.0).all() and np.allclose(r["pooled"], lost, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("vessel", ["cone_5", "glass_20"])
def test_a_rounding_s_femtolitres_do_not_keep_a_tipped_vessel_from_starting(vessel):
    """WP15's re-check: a tank's absolute pressure rounds to ~1e-16 m³ in an empty vessel, a level above 0 with its
    surface a femtometre over the lowest point. With 1e-15 m³ in it, a cone tipped 5° and a glass tipped 20° still
    start from 1.5 mm on their insides (the start holds while the surface is within cavity_tol of the lowest point)."""
    cone = ((0.02, 0.0), (0.04, HT))
    profile, cavity, tilt = ((((0.004, 0.0), *cone), cone, 5.0) if vessel == "cone_5" else (GLASS, CYL, 20.0))
    r, p, cells, held = _empty_start(profile, cavity, _tilted(tilt), [1.5e-3, 1.5e-3], volume=1e-15)
    lost = held - p.h.numpy()[:, cells] @ p.tables.area_np[cells]
    assert (r["amount"][:, 0] > 1e-9).all() and np.allclose(r["pooled"], lost, rtol=1e-12, atol=0.0)


def test_a_level_floor_keeps_its_puddle_before_starting():
    """A glass charted at a 90° contact angle: its level floor keeps the static puddle 2·l_c·sin 45° (3.85 mm) when the
    water leaves, so 2 mm there (over withdraw_max, under the puddle) stays film (world 0) and 4 mm starts (world 1)."""
    r, p, cells, _ = _empty_start(GLASS, CYL, (0.0, 0.0, -G), [2e-3, 4e-3], contact_angle=90.0, floor_only=True)
    assert r["amount"][0, 0] == 0.0 and r["amount"][1, 0] > 0.0


def test_a_swinging_glass_pouring_from_its_rim_passes_the_hook_checks():
    """A glass on a hinge, tipped 30° and swinging, pours from its rim and floods its own walls: the pour's exit point
    and direction and the last surface (the withdrawal films' speed) move with it and are state (snapshots, resets
    and captured graphs keep them), so the hooks are deterministic, isolated per world and graph-safe."""
    from alloy.plugins.harness.hooks import hook_checks  # noqa: PLC0415
    from alloy.plugins.harness.run import run_scene  # noqa: PLC0415
    s15 = math.sin(math.radians(15.0))
    tip = body_xform(0.0, 0.0, 0.3, (s15, 0.0, 0.0, math.cos(math.radians(15.0))))

    def build(b) -> None:
        link = b.add_link(xform=tip, label="glass")
        b.add_shape_sphere(link, xform=body_xform(0.05, 0.0, 0.05), radius=0.01)
        j = b.add_joint_revolute(-1, link, parent_xform=tip, axis=(0.0, 1.0, 0.0), label="hinge")   # across the tip:
        # the rim's lowest point moves round the rim as it swings
        b.add_articulation([j], label="swing")
        _far(b)
    pool = Pool("glass", body="glass", cavity=CYL, spill="pour", flood=True)      # its walls flood and come out
    lip = Nozzle("lip", edge="pour", rim_of="glass", body="glass", exit_area=1e-4)
    chart = RevolutionChart(profile=GLASS, n_theta=12, ds=4e-3, body="glass")
    p = StubFilm(charts=(FAR, chart), nozzles=(lip,), network={**_glass(0.095 * A, pour=True), "pools": [pool]})
    run = run_scene(TestScene(build=build, ground=False, worlds=2, periods=4), [p], "cpu")
    assert run.error is None, run.error
    assert (p.supply.read()["sent"] > 0.0).all()
    results = hook_checks(run)
    assert results and all(r.passed for r in results), [(r.name, r.value, r.detail) for r in results]


def test_glasses_sharing_a_near_empty_sink_never_overdraw_it():
    """WP15's probe ([internal note] notes/wp15-probes/sf8_overdraw_probe.py): two empty glasses with their rims just
    under a near-empty sink (its area under a glass's rim area, one exchange a period), so each glass's weir alone
    could take most of the sink. Each takes at most what the sink has left after the other: the sink never goes
    below 0, the water is conserved, and the weir is what the glasses hold."""
    def build(b) -> None:
        _far(b)
        for name, x in (("g1", -0.06), ("g2", 0.06)):
            link = b.add_link(xform=body_xform(x, 0.0, 0.0201), label=name)
            b.add_shape_sphere(link, xform=body_xform(0.0, 0.0, -0.05), radius=0.005)
            b.add_articulation([b.add_joint_fixed(-1, link, parent_xform=body_xform(x, 0.0, 0.0201))])
    a_sink = 1.0e-3
    v0 = a_sink * 0.01
    pools = [Pool("g1", body="g1", cavity=CYL, weir="sink"), Pool("g2", body="g2", cavity=CYL, weir="sink"),
             Pool("sink", floor=(0.0, 0.0, 0.12), half=(0.2, 0.2))]
    net = {"nodes": [{"name": "g1", "medium": "tank", "volume": 0.0, "area": A},
                     {"name": "g2", "medium": "tank", "volume": 0.0, "area": A},
                     {"name": "sink", "medium": "tank", "volume": v0, "area": a_sink}], "edges": [], "pools": pools}
    p = StubFilm(charts=(FAR,), network=net)
    sim = build_simulation(TestScene(build=build, ground=False, worlds=2, dt=0.02, control_dt=0.02), [p], "cpu")
    low = np.inf
    for _ in range(15):
        sim.step_period()
        v = p.supply.read()["amount"]
        low = min(low, v[:, 2].min())
    r = p.supply.read()
    assert low >= -1e-15 and (v[:, :2] > 0.0).all()
    assert np.allclose(v.sum(axis=1), v0, rtol=1e-12, atol=0.0)
    assert np.allclose(r["weir"], v[:, :2].sum(axis=1), rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("off", [0.5, 0.8])
def test_a_glass_catches_over_its_round_mouth(off):
    """WP15's nit: a jet falling straight down at (off·R, off·R) lands in an empty glass at 0.5 (0.71 R from its axis)
    and misses it at 0.8 (1.13 R: inside the square around the rim, outside the mouth)."""
    from alloy.plugins.fluid import P_ATM  # noqa: PLC0415
    net = {"nodes": [{"name": "line", "medium": "liquid", "volume": 1e-4},
                     {"name": "glass", "medium": "tank", "volume": 0.0, "area": A}],
           "edges": [{"name": "valve", "kind": "orifice", "a": "line", "reservoir": P_ATM + 2e5, "cd_area": 5e-6,
                      "dp_lam": 1.0},
                     {"name": "spout", "kind": "orifice", "a": "line", "reservoir": P_ATM, "cd_area": 1e-4,
                      "dp_lam": 1.0}],
           "pools": [Pool("glass", cavity=CYL)]}
    jet = Nozzle("jet", edge="spout", pos=(off * R, off * R, 0.3), exit_area=1e-4)
    p = StubFilm(charts=(FAR,), nozzles=(jet,), network=net)
    sim = _sim(p, _far)
    for _ in range(3):
        sim.step_period()
    r = p.supply.read()
    assert (r["sent"] > 0.0).all()
    if off < 0.7:
        assert np.allclose(r["to_tank"], r["sent"], rtol=1e-12, atol=0.0)
    else:
        assert (r["to_tank"] == 0.0).all() and np.allclose(r["lost"], r["sent"], rtol=1e-12, atol=0.0)


def test_an_empty_weir_list_is_refused():
    """``weir=()`` names no pool: refused (None means no weir)."""
    with pytest.raises(PluginError, match="weir"):
        Pool("glass", cavity=CYL, weir=())


@pytest.mark.parametrize("x", [0.0, 0.4])
def test_a_glass_dunked_in_one_of_three_bowls_fills_from_that_one(x):
    """WP8's rinse ask: weir=("wash", "rinse", "sanitize") tries the bowls in order. Welded upright in the wash bowl
    (x = 0) or the rinse bowl (x = 0.4), the glass fills to the brim from that bowl alone; the others keep theirs."""
    bowls = {"wash": 0.0, "rinse": 0.4, "sanitize": 0.8}
    v_bowl = 0.09 * 0.16                                                   # 0.3 m square bowls, surface at 0.16
    pools = [Pool("glass", body="glass", cavity=CYL, flood=False, weir=tuple(bowls))]
    pools += [Pool(name, floor=(bx, 0.0, 0.0), half=(0.15, 0.15)) for name, bx in bowls.items()]
    nodes = [{"name": "glass", "medium": "tank", "volume": 0.0, "area": A}]
    nodes += [{"name": name, "medium": "tank", "volume": v_bowl, "area": 0.09} for name in bowls]
    p = StubFilm(charts=(FAR,), network={"nodes": nodes, "edges": [], "pools": pools})

    def build(b) -> None:
        link = b.add_link(xform=body_xform(x, 0.0, 0.02), label="glass")
        b.add_shape_sphere(link, xform=body_xform(0.0, 0.0, -0.05), radius=0.005)
        b.add_articulation([b.add_joint_fixed(-1, link, parent_xform=body_xform(x, 0.0, 0.02))])
        _far(b)
    sim = _sim(p, build)
    for _ in range(20):
        sim.step_period()
    v = p.supply.read()["amount"]
    held = {name: v[:, 1 + k] for k, name in enumerate(bowls)}
    into = "wash" if x == 0.0 else "rinse"
    assert np.allclose(v[:, 0], A * HT, rtol=BRIM_RTOL, atol=0.0)                    # to the brim
    for name in bowls:
        expect = v_bowl - v[:, 0] if name == into else v_bowl                    # what the glass holds, exactly
        assert np.allclose(held[name], expect, rtol=1e-12, atol=0.0), name
