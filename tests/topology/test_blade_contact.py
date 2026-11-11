# SPDX-License-Identifier: Apache-2.0
"""blade_contact.py (SCR-SC5-1): side-aware blade contact for pre-split twins. CPU, fast."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.core.errors import AssemblyError
from alloy.testing.topology import blade_pose, clamp_particles, grid_tet_mesh, presplit_block, soft_kwargs
from alloy.topology.blade_contact import BladeContact, BladeContactConfig, side_a_tets
from alloy.topology.bonds import bonds_from_model
from alloy.topology.controller import TopologyController
from alloy.topology.emit import add_presplit_soft
from alloy.topology.presplit import SplitPlan
from alloy.topology.types import Blade, BondKind, TopologyBudget

CELL, N = 5.0e-3, 4                        # 20 mm cube, one seam plane at x = 10 mm (mesh frame)
SEAM = SplitPlan(planes=(((1.0, 0.0, 0.0), 0.01),))
HALF = 5.0e-4
GAP = 2.0e-3
CONTACT_FIELDS = ("soft_contact_count", "soft_contact_indices", "soft_contact_barycentric", "soft_contact_shape",
                  "soft_contact_body_pos", "soft_contact_body_vel", "soft_contact_normal", "soft_contact_rigid_indices",
                  "soft_contact_particle")


def _quat(axis, angle: float) -> wp.quat:
    return wp.quat_from_axis_angle(wp.normalize(wp.vec3(*axis)), float(angle))


def _mat(q: wp.quat) -> np.ndarray:
    return np.array(wp.quat_to_matrix(q), dtype=np.float64).reshape(3, 3)


def _frame(u: np.ndarray, spin: float) -> np.ndarray:
    """Orthonormal columns (u, v, w), v and w spun by ``spin`` about u."""
    u = u / np.linalg.norm(u)
    h = np.eye(3)[int(np.argmin(np.abs(u)))]
    v = np.cross(u, h)
    v /= np.linalg.norm(v)
    w = np.cross(u, v)
    v2 = np.cos(spin) * v + np.sin(spin) * w
    return np.stack([u, v2, np.cross(u, v2)], axis=1)


def _seam_scene(device: str, *, rot: wp.quat | None = None, box_q: wp.quat | None = None, worlds: int = 1,
                with_bonds: bool = True):
    """A 20 mm tet cube pre-split at x = 10 mm (rotated by ``rot``) and one blade body with a 0.5 mm box per world."""
    t = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    v, tets = grid_tet_mesh(N, N, N, CELL)
    kw = soft_kwargs()
    kw["rot"] = rot if rot is not None else wp.quat_identity()
    if with_bonds:
        add_presplit_soft(t, v, tets, SEAM, kw, budget=TopologyBudget(spare_vertices=1000))
    else:
        t.add_soft_mesh(vertices=[wp.vec3(*p) for p in v], indices=tets.ravel().tolist(), **kw)
    body = t.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()), mass=0.0)
    t.add_shape_box(body, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), box_q or wp.quat_identity()), hx=HALF, hy=0.03,
                    hz=0.03, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    b.replicate(t, worlds)
    b.color()
    m = b.finalize(device=device)
    bodies = np.flatnonzero(m.body_world.numpy() >= 0)
    blades = [Blade(body=int(bb), shape=-1, edge_a=(0.0, -0.03, -0.03), edge_b=(0.0, 0.03, -0.03)) for bb in bodies]
    return m, blades, bodies


def _pose(state: newton.State, body: int, pos, rmat: np.ndarray) -> None:
    bq = state.body_q.numpy()
    q = wp.quat_from_matrix(wp.mat33(*rmat.ravel().tolist()))
    bq[body] = [*pos, q[0], q[1], q[2], q[3]]
    state.body_q.assign(bq)


def _collide(m: newton.Model, state: newton.State) -> newton.Contacts:
    pipe = newton.CollisionPipeline(m, soft_contact_gap=GAP)
    c = pipe.contacts()
    pipe.collide(state, c)
    return c


def _snap(c: newton.Contacts) -> dict[str, np.ndarray]:
    return {f: getattr(c, f).numpy().copy() for f in CONTACT_FIELDS}


def _changed(before: dict[str, np.ndarray], c: newton.Contacts) -> np.ndarray:
    """Contact rows that differ in any field."""
    n = min(int(before["soft_contact_count"][0]), c.soft_contact_max)
    rows = np.zeros(n, bool)
    for f in CONTACT_FIELDS[1:]:
        a, b = before[f][:n].reshape(n, -1), getattr(c, f).numpy()[:n].reshape(n, -1)
        rows |= np.any(a != b, axis=1)
    return np.flatnonzero(rows)


def _assert_bitwise(before: dict[str, np.ndarray], c: newton.Contacts) -> None:
    for f in CONTACT_FIELDS:
        np.testing.assert_array_equal(before[f], getattr(c, f).numpy(), err_msg=f)


def _twin_side(bonds) -> dict[int, int]:
    cut = bonds.kind.numpy() == int(BondKind.TET_CUT)
    side = {int(p): -1 for p in bonds.a.numpy()[cut]}
    side.update({int(p): 1 for p in bonds.b.numpy()[cut]})
    return side


def _edge_through(point, rmat: np.ndarray, edge_mid=(0.0, 0.0, -0.03)) -> np.ndarray:
    """Body position that puts the blade's edge midpoint (body frame ``edge_mid``) at ``point``."""
    return np.asarray(point, np.float64) - rmat @ np.asarray(edge_mid, np.float64)


def _box_blades(blades: list[Blade], box_q: wp.quat | None) -> list[Blade]:
    """The scene's blades with the edge on the box when the box is turned on its body (box frame (0, +-0.03, -0.03))."""
    rq = _mat(box_q or wp.quat_identity())
    return [Blade(body=b.body, shape=b.shape, edge_a=tuple(rq @ np.array([0.0, -0.03, -0.03])),
                  edge_b=tuple(rq @ np.array([0.0, 0.03, -0.03]))) for b in blades]


def _in_zone(m: newton.Model, s: newton.State, blade: Blade, parts: np.ndarray) -> np.ndarray:
    """Particles within the cut-front cap 2h + r of the blade's edge segment (SCR-JOB1-2's far-face zone)."""
    xf = s.body_q.numpy()[blade.body]
    r = _mat(wp.quat(*xf[3:7]))
    ea, eb = xf[:3] + r @ np.asarray(blade.edge_a), xf[:3] + r @ np.asarray(blade.edge_b)
    x = s.particle_q.numpy()[parts].astype(np.float64)
    t = np.clip((x - ea) @ (eb - ea) / np.dot(eb - ea, eb - ea), 0.0, 1.0)
    d = np.linalg.norm(x - (ea + t[:, None] * (eb - ea)), axis=1)
    return d <= 2.0 * HALF + m.particle_radius.numpy()[parts]


def _ramp_zone(m: newton.Model, s: newton.State, blade: Blade, spine_body: np.ndarray,
               parts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SCR-JOB1-2 (item 9 step 1): where the rule may rewrite. (behind, ahead_in_ramp, allowed): a particle at or behind
    the edge along the spine, where the faces are; one ahead of it within h + r + edge_radius, where the face depth
    ramps up as the edge nears (``allowed`` = h + r + edge_radius - distance to the edge)."""
    xf = s.body_q.numpy()[blade.body]
    r = _mat(wp.quat(*xf[3:7]))
    ea, eb = xf[:3] + r @ np.asarray(blade.edge_a), xf[:3] + r @ np.asarray(blade.edge_b)
    x = s.particle_q.numpy()[parts].astype(np.float64)
    t = np.clip((x - ea) @ (eb - ea) / np.dot(eb - ea, eb - ea), 0.0, 1.0)
    foot = ea + t[:, None] * (eb - ea)
    d = np.linalg.norm(x - foot, axis=1)
    behind = (x - foot) @ (r @ spine_body) >= 0.0
    allowed = HALF + m.particle_radius.numpy()[parts] + blade.edge_radius - d
    return behind, ~behind & (allowed > 0.0), allowed


def _seam_pose(rot: wp.quat) -> tuple[np.ndarray, np.ndarray]:
    """World seam normal and seam-face centre for a mesh rotated by ``rot``."""
    r = _mat(rot)
    return r @ np.array([1.0, 0.0, 0.0]), r @ np.array([0.01, 0.01, 0.01])


# ---------------------------------------------------------------------------------------------------------------


def test_no_blades_or_no_cut_bonds_is_bitwise_noop(device):
    m, blades, bodies = _seam_scene(device)
    s = m.state()
    _pose(s, int(bodies[0]), (0.01, 0.01, 0.01), np.eye(3))
    c = _collide(m, s)
    bonds = bonds_from_model(m)
    bonds.damage.fill_(0.5)
    before = _snap(c)
    bc = BladeContact(m, bonds, [])
    assert not bc.enabled
    bc.adjust_contacts(s, c)
    _assert_bitwise(before, c)
    m2, blades2, bodies2 = _seam_scene(device, with_bonds=False)
    s2 = m2.state()
    _pose(s2, int(bodies2[0]), (0.01, 0.01, 0.01), np.eye(3))
    c2 = _collide(m2, s2)
    before2 = _snap(c2)
    assert int(before2["soft_contact_count"][0]) > 0
    bc2 = BladeContact(m2, bonds_from_model(m2), blades2)
    assert not bc2.enabled
    bc2.adjust_contacts(s2, c2)
    _assert_bitwise(before2, c2)


def test_inactive_twins_keep_stock_and_edge_gate_activates(device):
    m, blades, bodies = _seam_scene(device)
    bonds = bonds_from_model(m)
    side = _twin_side(bonds)
    s = m.state()
    _pose(s, int(bodies[0]), (0.01, 0.01, 0.01), np.eye(3))                  # box over the seam, edge at z=-20 mm
    c = _collide(m, s)
    before = _snap(c)
    touching = {int(p) for p in before["soft_contact_particle"][:int(before["soft_contact_count"][0])]}
    assert touching & side.keys()
    bc = BladeContact(m, bonds, blades)
    bc.adjust_contacts(s, c)
    _assert_bitwise(before, c)                                   # intact, undamaged, edge far: stock
    _pose(s, int(bodies[0]), (0.01, 0.01, 0.029), np.eye(3))                 # edge at z = -1 mm
    for gate, expect in ((False, 0), (True, 1)):
        c = _collide(m, s)
        before = _snap(c)
        BladeContact(m, bonds, blades, config=BladeContactConfig(edge_gate=gate)).adjust_contacts(s, c)
        rows = _changed(before, c)
        x = s.particle_q.numpy()
        z = x[before["soft_contact_particle"][rows], 2]
        assert (len(rows) > 0) == bool(expect)
        assert np.all(z < 1.0e-3)                                # only the bottom twin row is near the edge


def test_damage_or_break_activates_only_those_bonds(device):
    """Of the damaged or broken bonds, only twins within the cut zone (2h + r of the edge) are rewritten."""
    m, blades, bodies = _seam_scene(device)
    bonds = bonds_from_model(m)
    s = m.state()
    _pose(s, int(bodies[0]), (0.01, 0.01, 0.04), np.eye(3))                  # edge along y at z = 10 mm
    c = _collide(m, s)
    before = _snap(c)
    cut = np.flatnonzero(bonds.kind.numpy() == int(BondKind.TET_CUT))
    z = s.particle_q.numpy()[bonds.a.numpy()[cut], 2]
    mid, top = cut[np.abs(z - 0.01) < 1e-6], cut[np.abs(z - 0.02) < 1e-6]
    dmg, intact = bonds.damage.numpy(), bonds.intact.numpy()
    dmg[mid[0]], intact[mid[1]], dmg[top[0]] = 0.1, 0, 0.1                  # the top row is 10 mm from the edge
    bonds.damage.assign(dmg)
    bonds.intact.assign(intact)
    BladeContact(m, bonds, blades).adjust_contacts(s, c)
    rows = _changed(before, c)
    moved = set(before["soft_contact_particle"][rows].tolist())
    chosen = {int(bonds.a.numpy()[i]) for i in mid[:2]} | {int(bonds.b.numpy()[i]) for i in mid[:2]}
    assert moved and moved <= chosen


ORIENTS = [
    # mesh rotation, box shape rotation on the body, spin about the face normal, tilt, face axis flipped
    (wp.quat_identity(), None, 0.65, 0.0, False),
    (_quat((1.0, 1.0, 0.0), 0.5), _quat((0.0, 1.0, 0.0), 0.5 * np.pi), 1.9, 0.17, True),
    (_quat((0.2, -0.4, 1.0), 1.1), _quat((1.0, 0.0, 0.0), 0.3), -2.4, 0.1, False),
]


@pytest.mark.parametrize("rot,box_q,spin,tilt,flip", ORIENTS)
def test_rewrite_puts_a_on_minus_n_and_b_on_plus_n(device, rot, box_q, spin, tilt, flip):
    m, blades, bodies = _seam_scene(device, rot=rot, box_q=box_q)
    blades = _box_blades(blades, box_q)
    bonds = bonds_from_model(m)
    bonds.damage.fill_(0.5)                                      # every twin active; this test checks geometry
    n_w, centre = _seam_pose(rot)
    a_body = _mat(box_q or wp.quat_identity()) @ np.array([1.0, 0.0, 0.0])     # the box's thin axis, body frame
    target = (-1.0 if flip else 1.0) * (_frame(n_w, 0.0) @ np.array([np.cos(tilt), np.sin(tilt), 0.0]))
    rb = _frame(target, spin) @ _frame(a_body, 0.0).T
    s = m.state()
    edge_mid = 0.5 * (np.asarray(blades[0].edge_a) + np.asarray(blades[0].edge_b))
    _pose(s, int(bodies[0]), _edge_through(centre, rb, edge_mid), rb)     # the edge crosses the seam face centre
    c = _collide(m, s)
    before = _snap(c)
    bc = BladeContact(m, bonds, blades)
    assert bc.faces[0].half == pytest.approx(HALF)
    np.testing.assert_allclose(np.abs(bc.faces[0].axis @ a_body), 1.0, atol=1e-6)
    bc.adjust_contacts(s, c)
    side = _twin_side(bonds)
    n = int(before["soft_contact_count"][0])
    parts = before["soft_contact_particle"][:n]
    twin_rows = np.flatnonzero([int(p) in side for p in parts])
    behind, ramp, allowed = _ramp_zone(m, s, blades[0], bc._blades.spine.numpy()[0], parts[twin_rows])
    assert len(twin_rows) >= 20 and ramp.sum() >= 1
    changed = set(_changed(before, c))
    assert changed <= set(twin_rows[behind | ramp])                          # ahead, out of the ramp: Newton's
    near = behind & _in_zone(m, s, blades[0], parts[twin_rows])          # at the faces, within the crossing cap
    assert near.sum() >= 2 and len(changed & set(twin_rows[near])) >= near.sum() // 2   # one side may match stock
    xf = s.body_q.numpy()[int(bodies[0])]
    rq = _mat(wp.quat(*xf[3:7]))
    x = s.particle_q.numpy()
    radius = m.particle_radius.numpy()
    for k, at_face, cap in zip(twin_rows, behind, allowed):
        if k not in changed:
            continue
        p = int(parts[k])
        nrm = c.soft_contact_normal.numpy()[k]
        bx = rq @ c.soft_contact_body_pos.numpy()[k] + xf[:3]
        assert np.sign(nrm @ n_w) == side[p]                      # a pushed toward -n, b toward +n
        np.testing.assert_allclose(abs(nrm @ (rb @ a_body)), 1.0, atol=1e-5)
        np.testing.assert_allclose(np.cross(x[p] - bx, nrm), 0.0, atol=2e-6)    # the twin projected onto the plane
        assert c.soft_contact_shape.numpy()[k] == bc.faces[0].face_shape
        if at_face:
            assert (bx - centre) @ nrm == pytest.approx(HALF - radius[p], abs=2e-6)  # the plane at h - r
            assert HALF < radius[p]                    # the plane is past the mid-plane (h 0.5 mm, r 1 mm), yet
            assert (bx - centre) @ nrm + radius[p] == pytest.approx(HALF, abs=2e-6)   # the twin rests on its own face
        else:                    # ahead: the full depth h - s d (plane at h - r), capped by the ramp, and no lower
            full = HALF - (x[p] - centre) @ nrm
            assert radius[p] - (x[p] - bx) @ nrm == pytest.approx(min(full, cap), abs=2e-6)


def test_misaligned_blade_keeps_stock(device):
    m, blades, bodies = _seam_scene(device)
    bonds = bonds_from_model(m)
    bonds.damage.fill_(0.5)
    s = m.state()
    _pose(s, int(bodies[0]), (0.01, 0.01, 0.01), _frame(np.array([0.0, 1.0, 0.0]), 0.0))   # face normal y
    c = _collide(m, s)
    before = _snap(c)
    BladeContact(m, bonds, blades).adjust_contacts(s, c)
    _assert_bitwise(before, c)


def test_per_world_only_world0_rewritten(device):
    m, blades, bodies = _seam_scene(device, worlds=2)
    bonds = bonds_from_model(m)
    s = m.state()
    _pose(s, int(bodies[0]), (0.01, 0.01, 0.029), np.eye(3))     # world 0: edge 1 mm below the bottom twins
    _pose(s, int(bodies[1]), (0.01, 0.01, 0.015), np.eye(3))     # world 1 perturbed: edge 15 mm below
    c = _collide(m, s)
    before = _snap(c)
    BladeContact(m, bonds, blades, config=BladeContactConfig(edge_gate=True)).adjust_contacts(s, c)
    rows = _changed(before, c)
    pw = m.particle_world.numpy()
    n = int(before["soft_contact_count"][0])
    parts = before["soft_contact_particle"][:n]
    assert len(rows) > 0 and np.all(pw[parts[rows]] == 0)
    assert np.any(pw[parts] == 1)                                  # world 1 has blade contacts, all stock


def _write_records(c: newton.Contacts, recs: list[tuple[tuple[int, int, int], tuple[float, float, float]]],
                   shape: int) -> None:
    n = len(recs)
    idx = c.soft_contact_indices.numpy()
    bary = c.soft_contact_barycentric.numpy()
    sh, nrm = c.soft_contact_shape.numpy(), c.soft_contact_normal.numpy()
    for k, (i, w) in enumerate(recs):
        idx[k], bary[k], sh[k], nrm[k] = i, w, shape, (0.0, 0.0, 1.0)
    for name, arr in (("indices", idx), ("barycentric", bary), ("shape", sh), ("normal", nrm)):
        getattr(c, f"soft_contact_{name}").assign(arr)
    c.soft_contact_count.assign(np.array([n], np.int32))


def test_multi_slot_records_need_one_side_of_active_twins(device):
    m, blades, bodies = _seam_scene(device)
    bonds = bonds_from_model(m)
    bonds.damage.fill_(0.5)
    s = m.state()
    _pose(s, int(bodies[0]), (0.01, 0.01, 0.04), np.eye(3))                  # edge along y at z = 10 mm
    c = _collide(m, s)
    cut = np.flatnonzero(bonds.kind.numpy() == int(BondKind.TET_CUT))
    cut = cut[np.abs(s.particle_q.numpy()[bonds.a.numpy()[cut], 2] - 0.01) < 1e-6]   # twins on the edge
    a, b = bonds.a.numpy()[cut], bonds.b.numpy()[cut]
    plain = int(np.setdiff1d(np.arange(m.particle_count), np.concatenate([a, b]))[0])
    recs = [((int(a[0]), int(a[1]), -1), (0.25, 0.75, 0.0)),            # edge, both side a: rewritten
            ((int(b[0]), int(b[1]), int(b[2])), (0.2, 0.3, 0.5)),       # face, all side b: rewritten
            ((int(a[0]), int(b[1]), -1), (0.5, 0.5, 0.0)),              # mixed sides: stock
            ((int(a[0]), plain, -1), (0.5, 0.5, 0.0))]                  # a non-twin slot: stock
    _write_records(c, recs, shape=int(np.flatnonzero(m.shape_body.numpy() == bodies[0])[0]))
    before = _snap(c)
    BladeContact(m, bonds, blades).adjust_contacts(s, c)
    np.testing.assert_array_equal(_changed(before, c), [0, 1])
    nrm = c.soft_contact_normal.numpy()
    assert nrm[0][0] == pytest.approx(-1.0) and nrm[1][0] == pytest.approx(1.0)
    x = s.particle_q.numpy()
    p0 = 0.25 * x[a[0]] + 0.75 * x[a[1]]
    bp = c.soft_contact_body_pos.numpy()[0] + s.body_q.numpy()[int(bodies[0])][:3]
    np.testing.assert_allclose(bp[1:], p0[1:], atol=1e-6)               # the barycentric point, on the -x face
    r = float(m.particle_radius.numpy()[a[:2]].max())                  # the plane at h - r: the twins rest on the face
    assert bp[0] == pytest.approx(0.01 - (HALF - r), abs=1e-6)


def test_seam_normal_follows_f_inverse_transpose(device):
    m, blades, _ = _seam_scene(device)
    bonds = bonds_from_model(m)
    bc = BladeContact(m, bonds, blades)
    s = m.state()
    x = s.particle_q.numpy().astype(np.float64)
    gamma = 0.8                                                          # shear x += gamma * z
    x[:, 0] += gamma * x[:, 2]
    s.particle_q.assign(x.astype(np.float32))
    cut = np.flatnonzero(bonds.kind.numpy() == int(BondKind.TET_CUT))
    n = bc.seam_normals(s).numpy()[cut]
    expect = np.array([1.0, 0.0, -gamma]) / np.hypot(1.0, gamma)         # F^-T x; F x would stay (1, 0, 0)
    np.testing.assert_allclose(n, np.broadcast_to(expect, n.shape), atol=1e-5)
    tets = side_a_tets(m.tet_indices.numpy(), bonds.a.numpy()[cut].astype(np.int64))
    assert np.all(tets >= 0) and np.all(np.any(m.tet_indices.numpy()[tets] == bonds.a.numpy()[cut, None], axis=1))


def test_config_and_half_thickness_rules(device):
    with pytest.raises(AssemblyError):
        BladeContactConfig(reach=-1.0e-3)
    with pytest.raises(AssemblyError):
        BladeContactConfig(min_alignment=1.5)
    m, blades, _ = _seam_scene(device)
    bonds = bonds_from_model(m)
    assert BladeContact(m, bonds, blades, half_thickness=[2.0e-3]).faces[0].half == pytest.approx(2.0e-3)
    sphere_only = [Blade(body=blades[0].body, shape=-1, edge_a=(0.0, 0.0, 0.0), edge_b=(0.0, 1.0, 0.0))]
    m.shape_type.assign(np.full(m.shape_count, int(newton.GeoType.SPHERE), np.int32))     # no box on the body
    with pytest.raises(AssemblyError):
        BladeContact(m, bonds, sphere_only)
    face = BladeContact(m, bonds, sphere_only, half_thickness=[1.0e-3]).faces[0]
    assert face.face_shape == -1 and face.half == 1.0e-3 and np.allclose(face.axis, (1.0, 0.0, 0.0))


def _small_cut(device: str, use_module: bool) -> int:
    sc = presplit_block(device, cells=(4, 3, 3), plan=SEAM, k_mu=2.0e4, k_lambda=2.0e4)
    m = sc.model
    clamp_particles(m, np.flatnonzero(np.isclose(m.particle_q.numpy()[:, 0], 4 * CELL)))   # both ends held
    m.soft_contact_ke = 2.0e3
    bonds = bonds_from_model(m)
    ctl = TopologyController(m, bonds, TopologyBudget(), sc.blades)
    bc = BladeContact(m, bonds, sc.blades) if use_module else None
    solver = newton.solvers.SolverVBD(m, iterations=10)
    pipe = newton.CollisionPipeline(m, soft_contact_gap=1.0e-3)
    con = pipe.contacts()
    s0, s1, ctrl = m.state(), m.state(), m.control()
    for k in range(300):
        z_edge = max(3 * CELL + 1.0e-3 - 0.08 * k * 1.0e-3, 5.0e-4)
        bq = s0.body_q.numpy()
        bq[sc.blade_bodies[0]] = blade_pose(np.array([0.01, 1.5 * CELL, z_edge + 0.02]), np.array([0.0, 1.0, 0.0]),
                                            np.array([0.0, 0.0, -1.0]))
        s0.body_q.assign(bq)
        s1.body_q.assign(bq)
        pipe.collide(s0, con)
        if bc is not None:
            bc.adjust_contacts(s0, con)
        ctl.pre_step(s0, con, 1.0e-3)
        solver.step(s0, s1, ctrl, con, 1.0e-3)
        ctl.apply(s1)
        s0, s1 = s1, s0
    assert np.isfinite(s0.particle_q.numpy()).all()
    cut = bonds.kind.numpy() == int(BondKind.TET_CUT)
    return int((bonds.intact.numpy()[cut] == 0).sum())


def test_small_cut_breaks_more_bonds_with_module(device):
    stock, module = _small_cut(device, False), _small_cut(device, True)
    assert module > stock, (stock, module)


@pytest.mark.gpu
def test_adjust_contacts_graph_capture_matches_eager(device):
    if not device.startswith("cuda"):
        pytest.skip("needs CUDA (ALLOY_DEVICE=cuda:0)")
    m, blades, bodies = _seam_scene(device, rot=ORIENTS[1][0], box_q=ORIENTS[1][1])
    blades = _box_blades(blades, ORIENTS[1][1])
    bonds = bonds_from_model(m)
    bonds.damage.fill_(0.5)
    n_w, centre = _seam_pose(ORIENTS[1][0])
    s = m.state()
    rb = _frame(n_w, 0.4) @ _frame(np.array([0.0, 0.0, -1.0]), 0.0).T
    _pose(s, int(bodies[0]), _edge_through(centre, rb, 0.5 * (np.asarray(blades[0].edge_a) + blades[0].edge_b)), rb)
    c = _collide(m, s)
    before = _snap(c)
    bc = BladeContact(m, bonds, blades)
    bc.adjust_contacts(s, c)                                     # eager; also loads the module before capture
    eager = _snap(c)
    assert len(_changed(before, c)) > 0
    for f, arr in before.items():
        getattr(c, f).assign(arr)
    with wp.ScopedCapture(device=device) as cap:
        bc.adjust_contacts(s, c)
    wp.capture_launch(cap.graph)
    _assert_bitwise(eager, c)
