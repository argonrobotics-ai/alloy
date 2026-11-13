# SPDX-License-Identifier: Apache-2.0
"""The orthotropic woven membrane on CLOTH_VBD (alloy.scene.ortho_membrane, gap #3): coupons on the kernel (warp and
weft stretch, shear), its force against the energy's gradient, the hinge scale, the route (opt-in only), and a woven
sheet that touches a box staying valid (it did not without the contact regulariser)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import warp as wp

import alloy
from alloy.scene import ortho_membrane as OM
from alloy.scene.twins import drape_route as DR
from alloy.scene.twins import fabric_rig as FR


def _scene(tmp_path, arm: str = "O", size=(0.04, 0.03), res=(4, 3), stand: bool = False):
    pack = json.loads(DR.PACK.read_text())
    rigs = FR.rigs_for(pack, arm, "F1")
    root = FR.write_cards(tmp_path / "F1", rigs, pack["id"])
    s = alloy.Scene(dt=0.01, roots=[str(root)])
    s.add_ground()
    if stand:
        s.add_box("stand", size=(0.06, 0.06, 0.3), pos=(0, 0, 0.15), static=True)
    s.add_cloth("c", size=size, pos=(0, 0, 0.31), resolution=res, thickness=FR.thickness(rigs[0].fab),
                material=rigs[0].name, self_contact=False)
    if arm == "O":
        s.items["c"]["card_extra"]["route"] = {"route": "cloth_vbd", "reason": "test: the orthotropic membrane"}
    return s, rigs[0].fab


def test_every_worlds_triangles_are_woven_and_coloured(tmp_path):
    """Two worlds: world 1's triangles carry the orthotropic rows too (replicate merges them) and the model has more
    than one colour group (WP11's review)."""
    s, _ = _scene(tmp_path)
    s.worlds = 2
    sim = alloy.build(s, device="cpu")
    m = _vbd(sim).model
    k = m.alloy.tri_ortho_k.numpy()
    half = m.tri_count // 2
    assert m.tri_count == 2 * half and np.all(k[half:, :3] == k[:half, :3]) and np.all(k[half:, 0] > 0.0)
    assert len(m.particle_color_groups) > 1
    sim.close()


def test_youngs_randomization_of_a_woven_card_is_refused(tmp_path):
    """randomize youngs scales tri_materials (0 here) and the hinges, not the orthotropic rows: refused, not half done."""
    s, _ = _scene(tmp_path)
    s.worlds = 2
    s.randomize("c", "youngs", 0.9e6, 1.1e6)
    with pytest.raises(alloy.BuildError, match="youngs randomization of a woven card"):
        alloy.build(s, device="cpu")


def _vbd(sim):
    return [x for x in sim.engine.solvers if hasattr(x, "_colour_callbacks_after")][0]


def _forces(solver, x: np.ndarray, x_prev: np.ndarray | None = None, dt: float = 0.01):
    m = solver.model
    pos = wp.array(x.astype(np.float32), dtype=wp.vec3, device=m.device)
    prev = pos if x_prev is None else wp.array(x_prev.astype(np.float32), dtype=wp.vec3, device=m.device)
    f = wp.zeros(len(x), dtype=wp.vec3, device=m.device)
    H = wp.zeros(len(x), dtype=wp.mat33, device=m.device)
    for g in m.particle_color_groups:
        wp.launch(OM.accumulate_ortho_membrane, dim=g.size,
                  inputs=[dt, g, pos, prev, m.tri_indices, m.tri_areas, m.alloy.tri_ortho_k, m.alloy.tri_ortho_dm,
                          m.alloy.tri_ortho_h, solver.particle_adjacency, solver.particle_cloth_piece,
                          solver.piece_near_self_contact, solver.piece_near_body_contact], outputs=[f, H],
                  device=m.device)
    return f.numpy().astype(np.float64), H.numpy().astype(np.float64)


@pytest.fixture(scope="module")
def coupon(tmp_path_factory):
    s, fab = _scene(tmp_path_factory.mktemp("ortho"), size=(0.05, 0.04), res=(6, 5))
    sim = alloy.build(s, device="cpu")
    yield sim, _vbd(sim), fab
    sim.close()


def test_the_card_routes_to_the_membrane_and_newtons_is_off(coupon):
    sim, solver, fab = coupon
    m = solver.model
    assert sim.routes()["c"]["route"] == "cloth_vbd"
    assert [type(c).__name__ for c in solver._colour_callbacks_after] == ["OrthoMembrane"]
    assert np.all(m.tri_materials.numpy()[:, :3] == 0.0)
    assert len(m.particle_color_groups) > 1           # coloured as a sheet, not one Jacobi group (ortho_membrane.colour)
    k = m.alloy.tri_ortho_k.numpy()
    np.testing.assert_allclose(k[:, :3], [[FR.tensile(fab, "warp"), FR.tensile(fab, "weft"), fab["shear_G_N_m"]]] *
                               len(k), rtol=1e-6)


def test_a_woven_grid_sheet_routes_to_the_membrane_by_default(tmp_path):
    """The default (D16 after gap #3): a woven card on an add_cloth sheet (a warp / weft grid) goes to CLOTH_VBD with
    the orthotropic membrane, alone or beside a free body, and no route override; routes() says why."""
    for beside in (False, True):
        s, fab = _scene(tmp_path / str(beside), arm="A")
        if beside:
            s.add_box("block", size=(0.02, 0.02, 0.02), pos=(0.3, 0.0, 0.05), mass=0.05)
        sim = alloy.build(s, device="cpu")
        r = sim.routes()["c"]
        assert r["route"] == "cloth_vbd" and "woven grid sheet" in " ".join(r["reasons"]), r
        k = _vbd(sim).model.alloy.tri_ortho_k.numpy()
        np.testing.assert_allclose(k[:, :2], [[FR.tensile(fab, "warp"), FR.tensile(fab, "weft")]] * len(k), rtol=1e-6)
        sim.close()


def test_an_isotropic_sheet_has_no_woven_rows_or_callback():
    """The add_cloth default card (cotton_woven: youngs / poisson, no elastic.cloth) is untouched by the woven default:
    cloth_vbd as before, Newton's membrane on, no orthotropic rows and no membrane callback on its solver."""
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_cloth("c", size=(0.04, 0.03), pos=(0, 0, 0.31), resolution=(4, 3))
    sim = alloy.build(s, device="cpu")
    r, solver = sim.routes()["c"], _vbd(sim)
    m = solver.model
    assert r["route"] == "cloth_vbd" and "woven" not in " ".join(r["reasons"])
    assert np.all(m.tri_materials.numpy()[:, 0] > 0.0)
    k = getattr(getattr(m, "alloy", None), "tri_ortho_k", None)
    assert k is None or not np.any(k.numpy())
    callbacks = solver._colour_callbacks + solver._colour_callbacks_after
    assert not any(isinstance(c, OM.OrthoMembrane) for c in callbacks)
    sim.close()


def _mesh_sheet(s, v, t):
    it = s.items["c"]
    it["shape"] = {"mesh": {"path": "", "sha256": "", "scale": [1.0, 1.0, 1.0], "collision": "convex"}}
    it["mesh"] = {"vertices": np.asarray(v).tolist(), "faces": np.asarray(t).tolist()}
    it["card_extra"].setdefault("budget", {})["particles"] = len(v)


def test_a_woven_mesh_sheet_keeps_the_opt_in(tmp_path):
    """A woven card on a mesh (not a grid sheet) keeps the old rule: Style3D alone, and beside a free body CLOTH_VBD
    only with the route override (the error names it)."""
    from bench.scenarios import fabric_drape as FD

    v, t = FD.specimen_mesh(0.05, 0.008, 7, "r6")
    s, _ = _scene(tmp_path / "alone", arm="A")
    _mesh_sheet(s, v, t)
    sim = alloy.build(s, device="cpu")
    assert sim.routes()["c"]["route"] == "cloth_style3d"
    sim.close()
    s, _ = _scene(tmp_path / "beside", arm="A")
    _mesh_sheet(s, v, t)
    s.add_box("block", size=(0.02, 0.02, 0.02), pos=(0.3, 0.0, 0.05), mass=0.05)
    with pytest.raises(alloy.BuildError, match="opt-in orthotropic membrane"):
        alloy.build(s, device="cpu")


def _hex(edge: float = 0.008, radius: float = 0.05, seed: int = 7):
    from bench.scenarios import fabric_drape as FD

    v, t = FD.specimen_mesh(radius, edge, seed, "r6")
    return np.asarray(v, np.float64), np.asarray(t, np.int64)


def _neighbour_pairs(t):
    edge, pairs = {}, []
    for i, tri in enumerate(t.tolist()):
        for a, c in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            j = edge.setdefault((min(a, c), max(a, c)), i)
            if j != i:
                pairs.append((j, i))
    return np.array(pairs)


def _rim(t):
    """Vertex ids on edges that belong to one triangle only."""
    e = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    u, n = np.unique(e, axis=0, return_counts=True)
    return u[n == 1]


def test_the_yarn_split_alternates_and_balances():
    """On the R6 jittered-hex disc neighbours carry different yarns across at least 85 % of the inner edges (odd rings
    of triangles keep the rest), each yarn gets half the area within 2 %, and an exact hex lattice alternates fully."""
    v, t = _hex()
    side = OM.yarn_split(t)
    pairs = _neighbour_pairs(t)
    d1, d2 = v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]]
    area = 0.5 * np.abs(d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1])
    assert set(np.unique(side)) == {-1, 1}
    assert np.mean(side[pairs[:, 0]] != side[pairs[:, 1]]) > 0.85
    assert abs(area[side > 0].sum() / area.sum() - 0.5) < 0.02
    inner = np.setdiff1d(np.arange(len(v)), np.unique(_rim(t)))            # every inner vertex keeps both yarns
    for sign in (1, -1):
        assert np.all(np.isin(inner, np.unique(t[side == sign])))
    n = 6                                                # an exact triangle lattice: rows of up / down triangles
    idx = lambda i, j: i * (n + 1) + j                   # noqa: E731
    lat = np.array([tri for i in range(n) for j in range(n) for tri in (
        (idx(i, j), idx(i + 1, j), idx(i, j + 1)), (idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)))])
    lp, ls = _neighbour_pairs(lat), OM.yarn_split(lat)
    assert np.all(ls[lp[:, 0]] != ls[lp[:, 1]])


def _yarn_strain(x, t, dm):
    d1, d2 = x[t[:, 1]] - x[t[:, 0]], x[t[:, 2]] - x[t[:, 0]]
    wu, wv = d1 * dm[:, [0]] + d2 * dm[:, [2]], d1 * dm[:, [1]] + d2 * dm[:, [3]]
    return np.linalg.norm(wu, axis=1) - 1.0, np.linalg.norm(wv, axis=1) - 1.0


def test_a_yarn_preserving_shape_strains_a_hex_mesh_to_first_order():
    """The cause of the lock (gap #3): x(u, v) = a(u) + b(v) with unit-speed arcs keeps every yarn's length while the
    yarn angle varies. A warp / weft grid's triangles read its yarn strain to second order in the cell (the chord of
    an arc: (h / R)^2 / 24), a hex mesh's to first order, 5x or more above the grid at 8 mm; hence the split."""
    from alloy.scene.emitters.common import grid_mesh

    R, h = 0.05, 0.008

    def shape(uv):
        return np.column_stack([R * np.sin(uv[:, 0] / R), R * np.sin(uv[:, 1] / R),
                                R * (2.0 - np.cos(uv[:, 0] / R) - np.cos(uv[:, 1] / R))])

    g = grid_mesh(0.096, 0.096, h)
    gu, gt = g.vertices[:, :2].astype(np.float64), g.indices.reshape(-1, 3)
    eu, ev = _yarn_strain(shape(gu), gt, OM.panel_dm_inv(gu, gt))
    chord = (h / R) ** 2 / 24.0
    np.testing.assert_allclose(np.abs(np.concatenate([eu, ev])), chord, rtol=0.05)
    v, t = _hex(h)
    hu, hv = _yarn_strain(shape(v[:, :2]), t, OM.panel_dm_inv(v[:, :2], t))
    rms = float(np.sqrt(np.mean(np.concatenate([hu, hv]) ** 2)))
    assert rms > 5.0 * chord, (rms, chord)


def test_the_split_constants_are_the_measured_ones():
    """SPLIT and H_FLOOR carry the default on meshes without warp / weft edges, and only GPU drape rows measured them
    (R6 / D03M arm O on the hex disc, notes in docs/known_gaps/cloth-woven-drape-isotropic.md): SPLIT sets the drape
    (Petrak F1 0.334 at 0.90, 0.263 at 0.99, 0.245 at 0.998) and 1.0 leaves zero-energy modes; H_FLOOR 0 let one F1
    world of 16 blow up at 60 iterations. A change to either needs those rows again (AUDIT-BENCH, 2026-11-10)."""
    assert (OM.SPLIT, OM.H_FLOOR) == (0.99, 0.2)


def test_a_hex_mesh_sheet_gets_split_rows_that_keep_the_cards_pair(tmp_path):
    """On a mesh without warp / weft edges each triangle carries one yarn at (1 + SPLIT) and the other at (1 - SPLIT)
    of the card's stiffness: per triangle the two add up to the card's pair, and the readback check passes."""
    s, fab = _scene(tmp_path)
    v, t = _hex()
    _mesh_sheet(s, np.column_stack([v[:, :2], np.zeros(len(v))]), t)
    sim = alloy.build(s, device="cpu")
    k = _vbd(sim).model.alloy.tri_ortho_k.numpy()
    ku, kv = FR.tensile(fab, "warp"), FR.tensile(fab, "weft")
    ru, rv = k[:, 0] / ku, k[:, 1] / kv
    np.testing.assert_allclose(ru + rv, 2.0, rtol=1e-6)
    np.testing.assert_allclose(np.abs(ru - rv), 2.0 * 0.99, rtol=1e-5)          # SPLIT
    np.testing.assert_allclose(k[:, 2], fab["shear_G_N_m"], rtol=1e-6)
    h = _vbd(sim).model.alloy.tri_ortho_h.numpy()                              # the given-up yarn, Hessian only
    warp = ru > 1.0
    np.testing.assert_allclose(h[warp, 1], 0.2 * 0.99 * kv, rtol=1e-6)          # H_FLOOR x SPLIT x k
    np.testing.assert_allclose(h[~warp, 0], 0.2 * 0.99 * ku, rtol=1e-6)
    assert np.all(h[warp, 0] == 0.0) and np.all(h[~warp, 1] == 0.0)
    sim.close()


def _rest(solver):
    x0 = solver.model.particle_q.numpy().astype(np.float64)
    return x0, x0[:, 0] - x0[:, 0].min(), x0[:, 1] - x0[:, 1].min()


@pytest.mark.parametrize("direction, k_index", [("warp", 0), ("weft", 1)])
def test_a_uniaxial_coupon_pulls_back_with_k_times_strain(coupon, direction, k_index):
    """x -> (1 + e) x along the warp (the sheet's x) or weft: the net force on the far edge is -k e W."""
    _, solver, _ = coupon
    x0, u, v = _rest(solver)
    e, ax = 1.0e-2, (0 if direction == "warp" else 1)
    x = x0.copy()
    x[:, ax] = x0[:, ax].min() + (1.0 + e) * (x0[:, ax] - x0[:, ax].min())
    f, _ = _forces(solver, x)
    along = u if ax == 0 else v
    far = along > along.max() - 1e-9
    width = float((v if ax == 0 else u).max())
    k = float(solver.model.alloy.tri_ortho_k.numpy()[0, k_index])
    assert f[far, ax].sum() == pytest.approx(-k * e * width, rel=2e-3)
    # no Poisson coupling (what is left is float32 rest positions times the stiff yarn k)
    assert np.abs(f[:, 1 - ax]).max() < 0.01 * np.abs(f[far, ax]).max()


def test_a_shear_coupon_pulls_back_with_g_times_angle(coupon):
    """Simple shear x -> x + g y: the top edge's net x force is -W (k_s g + k_v (r - 1) g / r), r = sqrt(1 + g^2): the
    shear term plus the weft's second-order stretch (k_v is 300x k_s, so at g = 0.02 that is 3 % of it)."""
    _, solver, _ = coupon
    x0, u, v = _rest(solver)
    g = 2.0e-2
    x = x0.copy()
    x[:, 0] += g * v
    f, _ = _forces(solver, x)
    top = v > v.max() - 1e-9
    _, kv, ks = (float(c) for c in solver.model.alloy.tri_ortho_k.numpy()[0, :3])
    r = float(np.sqrt(1.0 + g * g))
    assert f[top, 0].sum() == pytest.approx(-float(u.max()) * (ks * g + kv * (r - 1.0) * g / r), rel=5e-3)


def test_the_force_is_minus_the_energys_gradient(coupon):
    _, solver, _ = coupon
    m = solver.model
    k, dm, tri, area = (m.alloy.tri_ortho_k.numpy(), m.alloy.tri_ortho_dm.numpy(), m.tri_indices.numpy(),
                        m.tri_areas.numpy())
    x0 = m.particle_q.numpy().astype(np.float64)
    x = x0 + np.random.default_rng(1).normal(0.0, 0.002, x0.shape)

    def energy(y):
        d1, d2 = y[tri[:, 1]] - y[tri[:, 0]], y[tri[:, 2]] - y[tri[:, 0]]
        wu, wv = d1 * dm[:, [0]] + d2 * dm[:, [2]], d1 * dm[:, [1]] + d2 * dm[:, [3]]
        cu, cv = np.linalg.norm(wu, axis=1) - 1.0, np.linalg.norm(wv, axis=1) - 1.0
        return float(np.sum(area * 0.5 * (k[:, 0] * cu ** 2 + k[:, 1] * cv ** 2 + k[:, 2] * np.sum(wu * wv, 1) ** 2)))

    f, H = _forces(solver, x)
    fd, eps = np.zeros_like(x), 1e-6
    for i in range(len(x)):
        for a in range(3):
            xp, xm = x.copy(), x.copy()
            xp[i, a] += eps
            xm[i, a] -= eps
            fd[i, a] = -(energy(xp) - energy(xm)) / (2.0 * eps)
    assert np.abs(f - fd).max() < 1e-4 * np.abs(fd).max()
    assert np.linalg.eigvalsh(H).min() > -1e-6 * np.abs(H).max()     # PSD blocks


def test_hinges_take_the_plate_rigidity_across_them():
    """An edge along the weft (curvature along the warp) gets B_warp, one along the warp B_weft, a diagonal between."""
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    edges = np.array([[-1, -1, 0, 1], [-1, -1, 0, 2], [-1, -1, 0, 3]])
    s = OM.edge_bending_scale(uv, edges, 4.0, 1.0, 2.0)
    assert s == pytest.approx([1.0 / 2.0, 4.0 / 2.0, (0.5 * 2.0 + 0.5 * 1.0) ** 2 / 2.0])


def test_a_woven_disc_mesh_holds_its_shape_for_a_step(tmp_path):
    """A disc from a mesh (D03M's square-grid disc) at rest on the ground: with its triangles left out of the colouring
    it was one colour and reached 87 % strain in one step."""
    from bench.scenarios import fabric_drape as FD

    s, _ = _scene(tmp_path)
    v, t = FD.specimen_mesh(0.05, 0.008, 0, "grid")
    it = s.items["c"]
    it["shape"] = {"mesh": {"path": "", "sha256": "", "scale": [1.0, 1.0, 1.0], "collision": "convex"}}
    it["mesh"] = {"vertices": v.tolist(), "faces": t.tolist()}
    it["card_extra"].setdefault("budget", {})["particles"] = len(v)
    sim = alloy.build(s, device="cpu")
    e = np.unique(np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1), axis=0)
    q0 = sim.state(["c"], world=0)["c.points"]
    sim.step(2)
    q = sim.state(["c"], world=0)["c.points"]
    strain = np.linalg.norm(q[e[:, 0]] - q[e[:, 1]], axis=1) / np.linalg.norm(q0[e[:, 0]] - q0[e[:, 1]], axis=1) - 1.0
    assert len(_vbd(sim).model.particle_color_groups) > 1 and np.abs(strain).max() < 0.01
    sim.close()


def test_the_alignment_share_finds_warp_weft_grids():
    """A grid of any aspect has a warp and a weft leg in every triangle; the same grid turned 30 degrees has none (the
    build warns below ALIGNED_MIN: such a mesh locks the stiff yarns, DC 0.500 against 0.252 for Petrak F1)."""
    from alloy.scene.emitters.common import grid_mesh

    m = grid_mesh(0.15, 0.1, 0.0117)
    uv, tris = m.vertices[:, :2], m.indices.reshape(-1, 3)
    assert OM.aligned_share(uv, tris) == 1.0
    c, s_ = np.cos(np.pi / 6), np.sin(np.pi / 6)
    assert OM.aligned_share(uv @ np.array([[c, s_], [-s_, c]]), tris) == 0.0


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_a_woven_sheet_dropped_on_a_box_stays_valid(tmp_path):
    """F1 (KES stretch 7-18 kN/m, shear 23 N/m) dropped on a box: without the contact regulariser the energy jumped at
    0.17 s and a contact went 2 mm deep; with it the sheet drapes and check() stays ok."""
    s, _ = _scene(tmp_path, size=(0.15, 0.15), res=16, stand=True)
    sim = alloy.build(s, device="cpu")
    q = sim.run(0.6, record=["c"], every=10).world(0)["c.points"]
    assert sim.check()["ok"] and not sim.health()["degraded_worlds"]     # no CLOTH_STRETCH either
    assert q[-1, :, 2].min() < 0.27                      # the overhang hangs down past the box top (0.30)
    sim.close()
