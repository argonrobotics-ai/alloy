# SPDX-License-Identifier: Apache-2.0
"""Paper tear on the film route (alloy/solvers/film/seams.py, seam_vbd.py): pre-split tear lines (seams) are
held closed by the film's strain projection and carry the load across them; the topology controller tears a seam on its
force (strength in plane, Γ·h/2 at a crack tip or out of plane at a free edge) or where the sheet next to it is
stretched past config.SEAM_OVERLOAD_STRAIN. Kraft values: library/materials/paper_kraft.yaml; notes:
docs/notes/paper-pack.md (tear)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from alloy.cards.library import CardLibrary
from alloy.solvers.film.emit import add_film_presplit, spec_from_card
from alloy.solvers.film.rigs import FilmRig, grid_mesh, pin
from alloy.topology.presplit import SplitPlan
from alloy.topology.types import TopologyBudget

LIB = CardLibrary(assets=None)
KRAFT = spec_from_card(LIB.resolve(LIB.object("paper_kraft_sheet")))       # 100 g/m2, 135 um
STRENGTH = 64.4e6                                                            # Pa, the card's DOC tensile strength
GAMMA = 14.1e3                                                               # J/m2: 2 x Elmendorf 950 mN / 135 um
TEAR = 0.5 * GAMMA * KRAFT.thickness                                         # N, 0.95: the Elmendorf force
FAILURE = SimpleNamespace(fracture_strain=0.016, fracture_stress=STRENGTH, tear_energy=GAMMA, cut_min_strain=None,
                          cut_min_force=None)
ELASTIC = SimpleNamespace(youngs=KRAFT.youngs)
DX = 5e-3


def _cut_legs(v: np.ndarray, t: np.ndarray, x: float, z_tip: float) -> tuple[np.ndarray, np.ndarray]:
    """Cut the mesh along x = ``x`` below ``z_tip`` (a trouser's legs): those vertices get a copy for the triangles
    on the +x side."""
    on = np.flatnonzero((np.abs(v[:, 0] - x) < 1e-9) & (v[:, 2] < z_tip - 1e-9))
    copy = {int(i): len(v) + k for k, i in enumerate(on)}
    t = t.copy()
    for r in np.flatnonzero(v[t].mean(axis=1)[:, 0] > x):
        t[r] = [copy.get(int(i), int(i)) for i in t[r]]
    return np.concatenate([v, v[on]]), t


def _sheet(v, t, seam: tuple, *, gravity: float = -9.81, pins=lambda x: np.zeros(len(x), bool), load_g: float = 0.0,
           load_rows=lambda x: np.zeros(len(x), bool)) -> FilmRig:
    """A kraft sheet (mesh ``v``, ``t`` hanging in x-z) split along ``seam`` (a segment), as a FilmRig with named
    particle sets: "all", "x0" (rest positions, local), "loaded"."""

    def build(b):
        p0, _ = add_film_presplit(b, v, t, KRAFT, SplitPlan(creases=(seam,)), failure=FAILURE, elastic=ELASTIC,
                                  budget=TopologyBudget(spare_vertices=len(v)), tear=TEAR, pos=(0.0, 0.0, 0.5))
        x = np.asarray(b.particle_q[p0:], np.float64) - np.array([0.0, 0.0, 0.5])
        loaded = np.flatnonzero(load_rows(x))
        for p in loaded:
            b.particle_mass[p0 + int(p)] += load_g * 1e-3 / max(len(loaded), 1)
        pin(b, p0 + np.flatnonzero(pins(x)))
        return {"all": p0 + np.arange(len(x)), "x0": x, "loaded": p0 + loaded}

    return FilmRig(build, dt=1e-3, iterations=10, gravity=gravity)


def _strip(load_g: float = 0.0) -> FilmRig:
    """A 2 x 6 cm strip hanging from its top row, split across at mid height, ``load_g`` on its bottom row."""
    v, t = grid_mesh(4, 12, DX, plane="xz")
    return _sheet(v, t, ((-1e-3, 0.0, -6 * DX), (4 * DX + 1e-3, 0.0, -6 * DX)), pins=lambda x: np.abs(x[:, 2]) < 1e-6,
                  load_g=load_g, load_rows=lambda x: x[:, 2] < -12 * DX + 1e-6)


def _controller(rig):
    from alloy.topology.bonds import bonds_from_model
    from alloy.topology.controller import TopologyController

    return TopologyController(rig.model, bonds_from_model(rig.model), TopologyBudget(max_events_per_step=64),
                              entries={"film": rig.solver})


def _step(rig, ctl, n: int, before=None) -> None:
    for _ in range(n):
        rig.s0.clear_forces()
        if before is not None:
            before(rig)
        if ctl is not None:
            ctl.pre_step(rig.s0, rig.contacts, rig.dt)
        rig.solver.step(rig.s0, rig.s1, None, rig.contacts, rig.dt)
        if ctl is not None:
            ctl.apply(rig.s1)
        rig.s0, rig.s1 = rig.s1, rig.s0
        rig.t += rig.dt


def _edge_strain(rig) -> float:
    """Max engineering strain over the strip's triangle edges against the rest mesh."""
    ids, x0 = rig.names[0]["all"], rig.names[0]["x0"]
    tri = rig.model.tri_indices.numpy() - ids[0]
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    q = rig.q()[ids]
    return float((np.linalg.norm(q[e[:, 0]] - q[e[:, 1]], axis=1) / np.linalg.norm(x0[e[:, 0]] - x0[e[:, 1]], axis=1)
                  - 1.0).max())


def test_a_presplit_film_holds_its_seams_in_the_projection() -> None:
    rig = _strip()
    top = rig.solver.film_topology
    seams = top.seams
    assert seams is not None and len(seams.a) == 5                       # one twin per vertex across the strip
    assert sorted(seams.edge.tolist()) == [0, 0, 0, 1, 1]
    assert (top.pairs[seams.pair] == np.stack([seams.a, seams.b], 1)).all()
    assert (seams.cross_pair[seams.cross_start[:-1]] == seams.pair).all()  # each seam's own pair first, then crossings
    assert len(seams.cross_pair) > len(seams.a)                          # long-range limits span the seam


def test_seams_group_by_world_for_the_piece_join() -> None:
    """One piece-join thread per world: seams grouped by world, global seams (world -1) in a last group."""
    from alloy.solvers.film.seams import _world

    pw = np.array([0, 0, 1, 1, -1, 0, 1, -1], np.int32)
    model = SimpleNamespace(particle_world=SimpleNamespace(numpy=lambda: pw))
    g = _world(model, np.array([2, 0, 4, 5, 3, 7], np.int32))                 # seam a-ends: worlds 1, 0, -1, 0, 1, -1
    assert g["world"].tolist() == [1, 0, -1, 0, 1, -1]
    assert g["world_start"].tolist() == [0, 2, 4, 6]
    assert g["by_world"].tolist() == [1, 3, 0, 4, 2, 5]


@pytest.mark.compile_heavy
def test_vbd_pieces_join_across_intact_seams_and_part_when_torn() -> None:
    """VBD's pieces come from triangles only; while a seam holds, both halves are one (pinned) piece."""
    rig = _strip()
    sv, s = rig.solver, rig.solver.film_topology.seams
    piece = lambda: sv.particle_cloth_piece.numpy()                       # noqa: E731
    assert piece()[s.a[0]] != piece()[s.b[0]]                             # Newton's own labels: two pieces
    sv._join_pieces()
    assert (piece()[s.a] == piece()[s.b]).all()
    intact = sv._bond_intact.numpy()
    intact[s.bond[:-1]] = 0
    sv._bond_intact.assign(intact)
    sv._join_pieces()
    assert (piece()[s.a] == piece()[s.b]).all()                           # one intact twin still joins them
    intact[s.bond] = 0
    sv._bond_intact.assign(intact)
    sv._join_pieces()
    assert (piece()[s.a] != piece()[s.b]).all()


@pytest.mark.compile_heavy
def test_seam_force_thresholds() -> None:
    """Interior: total force against the strength over the seam; crack tip: against the tear force; free edge: its
    out-of-plane part against the tear force when that ratio is the larger."""
    import warp as wp

    from alloy.solvers.film import seams as S

    rig = _strip()
    sv, s = rig.solver, rig.solver.film_topology.seams
    inner = int(np.flatnonzero(s.edge == 0)[0])
    edge = int(np.flatnonzero(s.edge == 1)[0])
    imp = np.zeros((len(s.a), 3), np.float32)
    imp[:, 2] = 0.5 * 1e-6                                               # 0.5 N in plane (z) at dt = 1 ms
    imp[edge, 1] = 0.8 * 1e-6                                            # 0.8 N out of plane (y) at the edge seam
    sv._seams.impulse.assign(imp)

    def run():
        wp.launch(S.seam_force, dim=len(s.a), inputs=[sv._seams, sv._bond_intact, rig.s0.particle_q,
                                                      rig.model.tri_indices, 1.0e6, 1.0, sv.bond_force,
                                                      sv.bond_threshold], device=rig.model.device)
        return sv.bond_force.numpy()[s.bond], sv.bond_threshold.numpy()[s.bond]

    f, thr = run()
    assert f[inner] == pytest.approx(0.5, rel=1e-4) and thr[inner] == pytest.approx(s.break_force[inner], rel=1e-6)
    assert f[edge] == pytest.approx(0.8, rel=1e-4) and thr[edge] == pytest.approx(TEAR, rel=1e-3)
    nb = s.nbr[s.nbr_start[inner]:s.nbr_start[inner + 1]]
    intact = sv._bond_intact.numpy()
    intact[s.bond[nb[0]]] = 0                                            # a neighbour torn: ``inner`` is a crack tip
    sv._bond_intact.assign(intact)
    f, thr = run()
    assert thr[inner] == pytest.approx(TEAR, rel=1e-3) and f[nb[0]] == 0.0


@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
def test_a_split_strip_hangs_like_an_unsplit_one() -> None:
    """A strip hanging from its top row, split across at mid height, 0.5 g on its bottom row (4x its own weight):
    the seam stays shut, its twins carry the weight below them, the sheet stretches as little as an unsplit one
    (max edge strain 0.0009 at 1x and 0.0033 at 5x unsplit; a split strip whose far half was a VBD piece of its own
    sat at the cap, 0.014) and nothing tears."""
    rig = _strip(load_g=0.5)
    ctl = _controller(rig)
    worst = 0.0
    for _ in range(6):
        _step(rig, ctl, 100)
        worst = max(worst, _edge_strain(rig))
    s = rig.solver.film_topology.seams
    q = rig.q()
    assert float(np.linalg.norm(q[s.a] - q[s.b], axis=1).max()) < 1e-6
    assert worst < 0.3 * KRAFT.strain_cap, worst
    assert (ctl.bonds.intact.numpy()[s.bond] == 1).all()
    m = rig.model.particle_mass.numpy()
    x0 = rig.names[0]["x0"]
    below = rig.names[0]["all"][x0[:, 2] < -6 * DX - 1e-6]
    weight = 9.81 * (m[below].sum() + 0.5 * m[s.b].sum())
    carried = float(rig.solver.bond_force.numpy()[s.bond].sum())
    assert abs(carried - weight) <= 0.15 * weight, (carried, weight)


def _trouser():
    """A 2 x 6 cm strip with its legs cut from the bottom to mid height and a tear line on to the top edge; both legs'
    bottom rows pinned (the right one is the moving grip), no gravity."""
    v, t = grid_mesh(4, 12, DX, plane="xz")
    mid, tip = 2 * DX, -6 * DX
    v, t = _cut_legs(v, t, mid, tip)
    rig = _sheet(v, t, ((mid, 0.0, tip), (mid, 0.0, 1e-3)), gravity=0.0, pins=lambda x: x[:, 2] < -12 * DX + 1e-6)
    x0, ids = rig.names[0]["x0"], rig.names[0]["all"]
    tri = rig.model.tri_indices.numpy() - ids[0]
    right = np.unique(tri[x0[tri].mean(axis=1)[:, 0] > mid])
    grip = ids[np.intersect1d(np.flatnonzero(x0[:, 2] < -12 * DX + 1e-6), right)]
    return rig, grip, -tip


@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
def test_tear1_a_trouser_tears_by_half_the_grip_travel() -> None:
    """The right leg's bottom row is swung over (an arc about the tip line, 0.4 s) and then pulled up at 5 cm/s: the
    fold tears nothing, and once the legs are taut the tear runs from the tip up the line by half the grip's travel
    (inextensible legs: crack length = travel / 2), one seam at a time."""
    rig, grip, leg = _trouser()
    ctl = _controller(rig)
    s = rig.solver.film_topology.seams
    order = np.argsort(rig.q()[s.a, 2])                                  # tip first
    q0, arc, r, v = rig.q()[grip].copy(), 0.4, leg - 0.5e-3, 0.05

    def at(tt: float) -> np.ndarray:
        if tt <= arc:
            th = np.pi * 0.5 * (1.0 - np.cos(np.pi * tt / arc))
            return np.array([0.0, r * np.sin(th) + 2e-3 * th / np.pi, r * (1.0 - np.cos(th))])
        return np.array([0.0, 2e-3, 2.0 * r + v * (tt - arc)])

    def move(rg):
        q, qd = rg.s0.particle_q.numpy(), rg.s0.particle_qd.numpy()
        q[grip], qd[grip] = q0 + at(rg.t + rg.dt), (at(rg.t + rg.dt) - at(rg.t)) / rg.dt
        rg.s0.particle_q.assign(q)
        rg.s0.particle_qd.assign(qd)

    _step(rig, ctl, int(round(arc / rig.dt)), before=move)
    assert (ctl.bonds.intact.numpy()[s.bond] == 1).all()
    for travel in (0.015, 0.035):
        _step(rig, ctl, int(round((arc + travel / v - rig.t) / rig.dt)), before=move)
        torn = ctl.bonds.intact.numpy()[s.bond][order] == 0
        n = int(torn.sum())
        assert torn[:n].all(), torn                                     # from the tip, in order
        assert travel / (2 * DX) - 1.0 <= n <= travel / (2 * DX) + 1.5, (travel, n)


@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
def test_the_tearable_wrap_sheet_lands_on_a_table_whole() -> None:
    """The library's tearable sheet (tear lines on a 5 cm grid) dropped 2 cm flat onto a table through the public
    API: its controller reads the film's seam forces, and the landing tears nothing (one substep's impulse between
    twins touching down apart read 0.58 N, its tear force, before the force was averaged over 10 ms)."""
    import alloy

    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_box("table", size=(0.8, 0.8, 0.4), pos=(0, 0, 0.2), static=True)
    scene.add_object("wrap", "paper_wrap_tear_sheet", pos=(0, 0, 0.42))
    sim = alloy.build(scene, device="cpu")
    assert sim.routes()["wrap"]["route"] == "cloth_film"
    topo = sim.engine.topology
    assert topo is not None and topo._linked and topo.bonds.count > 500
    sim.step(30)
    assert (topo.bonds.intact.numpy() == 1).all()
    assert sim.check()["ok"]
    assert float(sim.state(world=0)["wrap.points"][:, 2].min()) > 0.4


@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
def test_seam_tables_repeat_per_world() -> None:
    """Two worlds of the tearable sheet: each world's seams cross only its own long-range limits, the same ones in
    each (the crossing table is built once per distinct world and shifted; a cross-world table was quadratic in
    worlds: 80 s of setup at 8 worlds, a 64-world build ran past 25 min)."""
    import alloy

    scene = alloy.Scene(worlds=2, dt=0.01)
    scene.add_ground()
    scene.add_object("wrap", "paper_wrap_tear_sheet", pos=(0, 0, 0.42))
    sim = alloy.build(scene, device="cpu")
    ft = next(s for s in sim.engine.solvers if hasattr(s, "film_topology")).film_topology
    s = ft.seams
    n0 = int((s.world == 0).sum())
    assert 2 * n0 == len(s.a) and (s.world[n0:] == 1).all()
    off = int(s.a[n0] - s.a[0])
    assert (s.a[n0:] - s.a[:n0] == off).all()
    world = sim.engine.model.particle_world.numpy()
    for k in range(n0):
        r0 = ft.pairs[s.cross_pair[s.cross_start[k]:s.cross_start[k + 1]]]
        r1 = ft.pairs[s.cross_pair[s.cross_start[k + n0]:s.cross_start[k + n0 + 1]]]
        assert np.array_equal(r1, r0 + off) and (world[r1] == 1).all()
