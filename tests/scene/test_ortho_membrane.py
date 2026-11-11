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
                          solver.particle_adjacency, solver.particle_cloth_piece, solver.piece_near_self_contact,
                          solver.piece_near_body_contact], outputs=[f, H], device=m.device)
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


def test_the_router_still_picks_style3d_for_a_woven_card_alone(tmp_path):
    """Opt-in only: without the route override D16 still sends a lone anisotropic sheet to Style3D."""
    s, _ = _scene(tmp_path, arm="A")
    sim = alloy.build(s, device="cpu")
    assert sim.routes()["c"]["route"] == "cloth_style3d"
    sim.close()


def test_a_woven_card_beside_another_entry_needs_the_opt_in(tmp_path):
    """Beside a free body D16 sends the anisotropic card to CLOTH_VBD; without the route override that stays an error
    (it names the opt-in), so no scene changes behaviour without asking."""
    s, _ = _scene(tmp_path, arm="A")
    s.add_box("block", size=(0.02, 0.02, 0.02), pos=(0.3, 0.0, 0.05), mass=0.05)
    with pytest.raises(alloy.BuildError, match="opt-in orthotropic membrane"):
        alloy.build(s, device="cpu")


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
