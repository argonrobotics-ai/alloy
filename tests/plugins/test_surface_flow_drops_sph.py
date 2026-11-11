# SPDX-License-Identifier: Apache-2.0
"""Surface-flow drops with SPH (design §4.4, §4.5, §4.7; WP9): thin liquid is absorbed into the film, a resting pool
is not, and the film sheds SPH particles at its open edge (laws D-1, D-2, D-5; CPU, stepped by hand)."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton

from alloy.plugins.host import PluginHost
from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow
from alloy.solvers.sph.build import add_fluid_points, add_held_spares
from alloy.solvers.sph.config import DFSPHConfig, register_custom_attributes
from alloy.solvers.sph.solver import SolverDFSPH

R, G, DT = 1.0e-3, 9.81, 5.0e-4
RHO = 998.2                                  # the film material "water"
TOP = 0.01                                   # the plate's top face (the chart)


def _scene(points: np.ndarray, gravity=(0.0, 0.0, -G), held: int = 64, on_body: bool = False, velocity=(0.0, 0.0, 0.0),
           film: dict | None = None):
    cfg = DFSPHConfig(particle_radius=R, world_bounds=((-0.2, -0.2, -0.1), (0.2, 0.2, 0.3)))
    b = newton.ModelBuilder(gravity=gravity)
    register_custom_attributes(b)
    b.begin_world()
    if on_body:
        body = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), label="carrier")
        b.add_shape_box(body, hx=0.05, hy=0.05, hz=0.01, label="plate")
    else:
        b.add_shape_box(-1, hx=0.05, hy=0.05, hz=0.01, label="plate")
    add_fluid_points(b, cfg, points, rest_density=RHO, velocity=velocity)
    add_held_spares(b, cfg, held, rest_density=RHO, park=(0.0, 0.0, -0.05))
    b.end_world()
    m = b.finalize(device="cpu")
    flow = SurfaceFlow(**{"charts": (PlaneChart("plate", cells=(20, 20)),), "material": "water", "sph": True,
                          "drops": {}, **(film or {})})
    host = PluginHost(m, [flow], control_dt=0.01)
    solver = SolverDFSPH(m, cfg)
    flow.check_dt(DT)
    return m, flow, host, solver, cfg


def _step(m, flow, solver, a, b, n: int):
    for _ in range(n):
        solver.step(a, b, None, None, DT)
        flow.update(a, b, DT)
        a, b = b, a
    return a, b


def _sheet(nx: int, z: float) -> np.ndarray:
    ax = (np.arange(nx) - 0.5 * (nx - 1)) * 2.0 * R
    g = np.stack(np.meshgrid(ax, ax, indexing="ij"), -1).reshape(-1, 2)
    return np.concatenate([g, np.full((len(g), 1), z)], 1)


def _ids_liquid(m, solver) -> np.ndarray:
    ph, fl = solver.layout.phase.numpy(), m.particle_flags.numpy()
    return np.flatnonzero((ph == 1) & (fl & int(newton.ParticleFlags.ACTIVE) != 0))


def _liquid(m, solver) -> int:
    return len(_ids_liquid(m, solver))


def test_a_thin_sheet_joins_the_film_exactly():
    """D-5: a one-particle-deep sheet resting on the chart is absorbed in one update; the film gains exactly its
    volume and SPH's ledger books it out."""
    m, flow, host, solver, cfg = _scene(_sheet(4, TOP + R))
    a, b = m.state(), m.state()
    a, b = _step(m, flow, solver, a, b, 1)
    assert _liquid(m, solver) == 0 and solver.port.counts()["absorbed"].tolist() == [16]
    r, d = flow.read(), flow.drops.read()
    vp = solver.port.particle_volume
    assert r["film_volume"][0] + r["pending_volume"][0] == pytest.approx(16 * vp, rel=1e-12, abs=0.0)
    assert d["from_sph_volume"][0] == pytest.approx(16 * vp, rel=1e-12, abs=0.0)
    led = solver.mass_ledger()
    assert np.array_equal(led["active"] + led["out"], led["initial"] + led["in"])


def test_a_resting_pool_stays_liquid():
    """D-5: a pool four particles deep on the chart keeps every particle (each bottom particle has a column above)."""
    ax = (np.arange(6) - 2.5) * 2.0 * R
    pts = np.stack(np.meshgrid(ax, ax, TOP + R + 2.0 * R * np.arange(4), indexing="ij"), -1).reshape(-1, 3)
    m, flow, host, solver, cfg = _scene(pts)
    a, b = m.state(), m.state()
    solver.prime(a, DT)
    _step(m, flow, solver, a, b, 20)
    assert solver.port.counts()["absorbed"].tolist() == [0] and _liquid(m, solver) == 144


def test_absorbed_momentum_goes_to_the_chart_body():
    """D-5: a sheet arriving at 0.2 m/s on a plate carried by a body gives the body exactly Σ m·v_n, and its
    tangential momentum Σ m·v_t joins the film as pending momentum (src_mom, chart frame; WP15 review)."""
    m, flow, host, solver, cfg = _scene(_sheet(3, TOP + R), on_body=True, velocity=(0.1, 0.0, -0.2),
                                        gravity=(0.0, 0.0, 0.0))
    a, b = m.state(), m.state()
    v = a.particle_qd.numpy()[:9].copy()
    flow.drops.absorb(a, a, DT)                               # one absorb call on the start state
    J = flow.drops.forces.impulse.numpy()[0, 0]
    mass = float(m.particle_mass.numpy()[0])
    assert solver.port.counts()["absorbed"].tolist() == [9]
    assert np.allclose(J, [0.0, 0.0, mass * v[:, 2].sum()], rtol=1e-6, atol=1e-12)
    film_mom = (flow.src_mom.numpy()[0] * flow.tables.area_np[:, None]).sum(axis=0)
    assert np.allclose(film_mom, [mass * v[:, 0].sum(), 0.0, 0.0], rtol=1e-5, atol=1e-12)


def test_an_open_edge_sheds_sph_particles_and_the_ledger_closes():
    """D-1, D-2: a film running off an inclined plate's open edge leaves as SPH particles from the held reservoir;
    every shed m³ is SPH liquid or still pending, and SPH's mass ledger books what came in."""
    film = {"charts": (PlaneChart("plate", cells=(20, 20), edges=("open", "wall", "wall", "wall")),),
            "max_depth": 3.0e-3}
    m, flow, host, solver, cfg = _scene(np.zeros((0, 3)), gravity=(-4.0, 0.0, -G), held=400, film=film)
    flow.set_depth(1.5e-3, chart=0)
    a, b = m.state(), m.state()
    a, b = _step(m, flow, solver, a, b, 200)
    r, d = flow.read(), flow.drops.read()
    vp = solver.port.particle_volume
    made = solver.port.counts()["emitted"][0]
    assert made > 10 and d["to_sph_volume"][0] == pytest.approx(made * vp, rel=1e-12, abs=0.0)
    assert r["spilled_volume"][0] == pytest.approx(d["to_sph_volume"][0] + d["pending_volume"][0], rel=1e-12, abs=0.0)
    led = solver.mass_ledger()
    assert led["in"][0] / led["scale"] == pytest.approx(made * float(m.particle_mass.numpy()[0]), rel=1e-6, abs=0.0)
    assert np.array_equal(led["active"] + led["out"], led["initial"] + led["in"])
    x = a.particle_q.numpy()[_ids_liquid(m, solver)]
    assert (x[:, 0] < -0.05 + 3.0 * R).all()                    # they left over the x- edge


def test_a_soapy_film_shed_as_sph_particles_books_its_soap_lost():
    """WP15's S1 on drops carrying species: SPH particles carry no species, so the soap of a 3 kg/m³ film shed over an
    open edge as SPH particles is booked lost, species_to_sph = c x to_sph volume, and the rest waits with the pending
    volume (species_pending = c x pending); species_shed = c x spilled = to SPH + pending (1e-12)."""
    c = 3.0
    film = {"charts": (PlaneChart("plate", cells=(20, 20), edges=("open", "wall", "wall", "wall")),),
            "max_depth": 3.0e-3, "species": ("soap",)}
    m, flow, host, solver, cfg = _scene(np.zeros((0, 3)), gravity=(-4.0, 0.0, -G), held=400, film=film)
    flow.set_depth(1.5e-3, chart=0)
    flow.set_species(c * 1.5e-3, "soap", chart=0)
    a, b = m.state(), m.state()
    _step(m, flow, solver, a, b, 200)
    r, d = flow.read(), flow.drops.read()
    assert d["to_sph_volume"][0] > 0.0 and d["pending_volume"][0] > 0.0
    assert d["species_to_sph"][0, 0] == pytest.approx(c * d["to_sph_volume"][0], rel=1e-12, abs=0.0)
    assert d["species_pending"][0, 0] == pytest.approx(c * d["pending_volume"][0], rel=1e-12, abs=0.0)
    assert d["species_shed"][0, 0] == pytest.approx(c * r["spilled_volume"][0], rel=1e-12, abs=0.0)
    assert d["species_shed"][0, 0] == pytest.approx(d["species_to_sph"][0, 0] + d["species_pending"][0, 0],
                                                    rel=1e-12, abs=0.0)


def test_many_drops_from_one_edge_cell_never_overlap():
    """WP15 review: ten particles leaving one 5 mm edge cell at once stand 2r apart (rows across the face, then
    further out), not stacked on the face's ends."""
    film = {"charts": (PlaneChart("plate", cells=(20, 20), edges=("open", "wall", "wall", "wall")),),
            "max_depth": 3.0e-3}
    m, flow, host, solver, cfg = _scene(np.zeros((0, 3)), held=400, film=film)
    flow.set_depth(1.0e-3, chart=0)
    vp = solver.port.particle_volume
    pend = flow.drops.pend.vol.numpy()
    pend[0, 10] = 10.0 * vp + 1.0e-3 * vp                                # cell (0, 10) on the x- edge
    flow.drops.pend.vol.assign(pend)
    tiles = flow.drops.pend.tile.numpy()
    tiles[0, flow.cells.tile.numpy()[10]] = 1                            # as collect marks it (any film stride)
    flow.drops.pend.tile.assign(tiles)
    a, b = m.state(), m.state()
    solver.step(a, b, None, None, DT)
    flow.update(a, b, DT)
    x = b.particle_q.numpy()[_ids_liquid(m, solver)]
    assert len(x) >= 10
    d = np.linalg.norm(x[:, None] - x[None, :], axis=-1) + np.eye(len(x))
    assert d.min() >= 2.0 * R * (1.0 - 1.0e-5)


def test_a_hanging_film_drips_sph_particles_it_does_not_take_back():
    """D-3 with SPH (WP15 review): a 6 mm film under a plate drips SPH particles and drains to h_ret = 2 l_c sin 45°.
    A drip leaves at rest with gravity pulling it off the chart, so it is not absorbed again (before, every drip was
    taken back in the next exchange and the film stayed deep)."""
    cfg = DFSPHConfig(particle_radius=R, world_bounds=((-0.1, -0.1, -0.2), (0.1, 0.1, 0.3)))
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -G))
    register_custom_attributes(b)
    b.begin_world()
    b.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat(1.0, 0.0, 0.0, 0.0)), hx=0.02, hy=0.02,
                    hz=0.01, label="plate")                               # its +z face (the chart) faces down
    add_held_spares(b, cfg, 600, rest_density=RHO, park=(0.0, 0.0, -0.15))
    b.end_world()
    m = b.finalize(device="cpu")
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(8, 8)),), material="water", max_depth=8.0e-3, sph=True,
                       drops={"tau_drip": 0.02})
    PluginHost(m, [flow], control_dt=0.01)
    solver = SolverDFSPH(m, cfg)
    flow.check_dt(DT)
    flow.set_depth(6.0e-3, chart=0)
    a, bb = m.state(), m.state()
    _step(m, flow, solver, a, bb, 400)                                      # 0.2 s = 10 tau_drip
    h_ret = 2.0 * np.sqrt(flow.mat.surface_tension / (flow.mat.density * G)) * np.sin(np.radians(45.0))
    c, r, d = solver.port.counts(), flow.read(), flow.drops.read()
    assert np.allclose(flow.depth(0)[0], h_ret, rtol=1e-4, atol=0.0)
    assert c["absorbed"].tolist() == [0] and c["emitted"][0] > 300
    assert r["dripped_volume"][0] == pytest.approx(d["to_sph_volume"][0] + d["pending_volume"][0], rel=1e-12, abs=0.0)


@pytest.mark.compile_heavy
def test_a_stream_becomes_film_and_leaves_as_a_stream():
    """D-6: SPH drops fall onto a 20° plate near its open edge, join the film, and the film sheds them back. At steady
    state (from about 1 s) the film's outflux equals what it takes in (3 % over 2 s). The particles it makes trail it by
    each edge cell's sub-particle remainder (one per edge cell at most). Every ledger stays exact."""
    theta = np.radians(20.0)
    film = {"charts": (PlaneChart("plate", cells=(40, 10), edges=("open", "wall", "wall", "wall")),),
            "max_depth": 4.0e-3}
    cfg = DFSPHConfig(particle_radius=R, world_bounds=((-0.2, -0.2, -0.05), (0.2, 0.2, 0.3)))
    b = newton.ModelBuilder(gravity=(-G * np.sin(theta), 0.0, -G * np.cos(theta)))
    register_custom_attributes(b)
    b.begin_world()
    b.add_shape_box(-1, hx=0.05, hy=0.025, hz=0.01, label="plate")
    from alloy.solvers.sph.build import add_spares  # noqa: PLC0415
    add_spares(b, cfg, 600, rest_density=RHO, park=(0.0, 0.0, -0.04))
    add_held_spares(b, cfg, 60, rest_density=RHO, park=(0.0, 0.0, -0.045))
    b.end_world()
    m = b.finalize(device="cpu")
    flow = SurfaceFlow(**{"charts": film["charts"], "material": "water", "sph": True, "drops": {},
                          "max_depth": film["max_depth"]})
    PluginHost(m, [flow], control_dt=0.01)
    solver = SolverDFSPH(m, cfg)
    flow.check_dt(DT)
    solver.add_emitter(0, wp.transform(wp.vec3(-0.03, 0.0, 0.03), wp.quat_identity()), wp.vec3(R, R, R), 1.0e-6,
                       wp.vec3(0.0, 0.0, -0.5))
    a, bb = m.state(), m.state()
    marks = []
    for steps in (int(round(1.0 / DT)), int(round(2.0 / DT))):           # t = 1 s, then t = 3 s
        a, bb = _step(m, flow, solver, a, bb, steps)
        c = solver.port.counts()
        marks.append((c["absorbed"][0], c["emitted"][0], flow.read()["spilled_volume"][0]))
    vp = solver.port.particle_volume
    took, gave, out = (marks[1][k] - marks[0][k] for k in range(3))
    assert took > 200 and out == pytest.approx(took * vp, rel=0.03, abs=0.0)
    assert abs(gave - took) <= 10 + 1                                    # 10 edge cells' remainders
    r, d = flow.read(), flow.drops.read()
    assert d["from_sph_volume"][0] == pytest.approx(marks[1][0] * vp, rel=1e-12, abs=0.0)
    assert r["spilled_volume"][0] == pytest.approx(d["to_sph_volume"][0] + d["pending_volume"][0], rel=1e-12, abs=0.0)
    led = solver.mass_ledger()
    assert np.array_equal(led["active"] + led["out"], led["initial"] + led["in"])
