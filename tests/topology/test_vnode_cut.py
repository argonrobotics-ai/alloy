# SPDX-License-Identifier: Apache-2.0
"""Item 9 step 3 prototype: per-tet virtual-node copies (alloy/topology/tet_copies.py, vnode_cut.py, vnode_rig.py).

CPU checks, the step-3 list in [internal note] scr/WP10.md:
- volume fractions against the clipped hull;
- recolouring keeps welded and sibling nodes apart;
- TOP-4: dormant copies leave the intact block's motion unchanged;
- TOP-1: a split keeps mass and momentum;
- SC5-3's stroke 1.7 mm off the nearest seam position splits exactly the tets the cut plane crosses and moves no
  material; the far piece then comes away.
GPU: the cutter's passes captured in a graph equal eager.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import warp as wp
from scipy.spatial import ConvexHull

from alloy._vendor import newton
from alloy.core.errors import AssemblyError
from alloy.testing.topology import blade_pose, soft_kwargs
from alloy.topology.tet_copies import (add_tet_copies, fraction_from_params, positive_fraction, positive_fraction_np,
                                       vec6)
from alloy.topology.types import Blade
from alloy.topology.vnode_cut import VNodeCutter
from alloy.topology.vnode_split import TetTable, crossing_params
from alloy.topology.vnode_rig import VNodeSim, vnode_block

X_CUT = 8.3e-3                 # [m] blade plane: 3.3 mm and 1.7 mm from the 5 mm grid planes


@wp.kernel
def _fraction(s: wp.array(dtype=wp.vec4), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = positive_fraction(s[i])


@wp.kernel
def _fraction_from_s(s: wp.array(dtype=wp.vec4), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    v = s[i]
    mask = int(0)
    t6 = vec6()
    k = int(0)
    for a in range(4):
        if v[a] >= 0.0:
            mask = mask | (1 << a)
        for b in range(a + 1, 4):
            if (v[a] >= 0.0) != (v[b] >= 0.0):
                t6[k] = v[a] / (v[a] - v[b])
            k += 1
    out[i] = fraction_from_params(mask, t6)


def _hull_fraction(x: np.ndarray, s: np.ndarray) -> float:
    pts = [x[i] for i in range(4) if s[i] >= 0.0]
    pts += [x[i] + s[i] / (s[i] - s[j]) * (x[j] - x[i]) for i in range(4) for j in range(i + 1, 4) if s[i] * s[j] < 0]
    return ConvexHull(np.array(pts)).volume / ConvexHull(x).volume


def test_positive_fraction_matches_the_clipped_hull() -> None:
    rng = np.random.default_rng(11)
    rows, ref = [], []
    while len(rows) < 300:
        x, n, p = rng.normal(size=(4, 3)), rng.normal(size=3), rng.normal(size=3) * 0.3
        s = (x - p) @ n
        if np.all(s >= 0) or np.all(s < 0) or abs(np.linalg.det(x[1:] - x[0])) < 0.05:
            continue
        rows.append(s)
        ref.append(_hull_fraction(x, s))
    s = np.array(rows)
    np.testing.assert_allclose(positive_fraction_np(s), ref, atol=1e-9)
    np.testing.assert_allclose(positive_fraction_np(s) + positive_fraction_np(-s), 1.0, atol=1e-12)
    out = wp.zeros(len(s), dtype=wp.float32, device="cpu")
    wp.launch(_fraction, dim=len(s), inputs=[wp.array(s.astype(np.float32), dtype=wp.vec4, device="cpu"), out],
              device="cpu")
    np.testing.assert_allclose(out.numpy(), ref, atol=2e-5)
    wp.launch(_fraction_from_s, dim=len(s), inputs=[wp.array(s.astype(np.float32), dtype=wp.vec4, device="cpu"), out],
              device="cpu")
    np.testing.assert_allclose(out.numpy(), ref, atol=2e-5)                           # from edge crossings


def test_recolouring_keeps_welded_and_sibling_copy_nodes_apart() -> None:
    sc = vnode_block("cpu", worlds=2)
    m, c = sc.model, sc.copies
    color = np.full(m.particle_count, -1)
    for g, ids in enumerate(m.particle_color_groups):
        assert np.all(color[ids.numpy()] < 0)                                        # one colour per particle
        color[ids.numpy()] = g
    assert np.all(color >= 0)
    assert np.all(color[c.node] != color[c.node_hub][:, None])                       # weld ends
    for k in range(2):
        assert all(len(set(r)) == 4 for r in color[c.node[c.hub_row, k]].tolist())   # a fragment's four nodes


def _sag(copies: bool) -> np.ndarray:
    sc = vnode_block("cpu", gravity=-9.81, copies=copies)
    sim = VNodeSim(sc, newton.solvers.SolverVBD(sc.model, iterations=4), control_dt=0.016, substeps=16)
    for _ in range(12):
        sim.step_period()
    return sim.state.particle_q.numpy()[:sc.hubs] - sc.model.particle_q.numpy()[:sc.hubs]


def test_dormant_copies_leave_the_intact_block_unchanged() -> None:
    plain, copied = _sag(False), _sag(True)
    assert np.abs(plain).max() > 1e-5                                                  # the cantilever sags
    np.testing.assert_array_equal(copied, plain)                                       # TOP-4: bit for bit


def _one_tet(blade_x: float):
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    x = np.array([[5, 0, 0], [5, 5, 0], [5, 5, 5], [10, 5, 5]], float) * 1e-3
    b.add_soft_mesh(vertices=[wp.vec3(*v) for v in x], indices=[0, 1, 2, 3], **soft_kwargs(k_mu=5e4, k_lambda=5e4))
    rec = add_tet_copies(b, [0], 1000.0)
    body = b.add_body(xform=wp.transform(wp.vec3(blade_x, 0.0, 0.024), wp.quat_identity()), mass=0.0)
    b.add_shape_box(body, hx=0.0005, hy=0.05, hz=0.02, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    b.color()
    m = b.finalize(device="cpu")
    blade = Blade(body=body, shape=-1, edge_a=(0.0, -0.05, -0.02), edge_b=(0.0, 0.05, -0.02))
    return m, rec, VNodeCutter(m, rec, [blade], 1.6e-3), body


def test_a_split_at_a_clamped_corner_books_its_added_mass() -> None:
    """A kinematic hub keeps zero mass, so its copies' mass is new: total = initial + VNodeCutter.added_mass()
    (TOP-1 ledger row); momentum is unchanged (the clamped hub is at rest)."""
    m, rec, cut, body = _one_tet(X_CUT)
    clamp = int(rec.hub[0, 0])
    for name in ("particle_mass", "particle_inv_mass"):
        a = getattr(m, name).numpy()
        a[clamp] = 0.0
        getattr(m, name).assign(a)
    s = m.state()
    qd = s.body_qd.numpy()
    qd[body, 2] = -0.05
    s.body_qd.assign(qd)
    m0 = float(m.particle_mass.numpy().astype(np.float64).sum())
    cut.pre_step(s)
    assert cut.split.numpy().tolist() == [1]
    added = cut.added_mass()
    assert added[0] == pytest.approx(0.5 * rec.mass[0] / 4.0, rel=1e-6)               # one corner's share
    assert float(m.particle_mass.numpy().astype(np.float64).sum()) == pytest.approx(m0 + added[0], rel=1e-6)


def test_the_copy_mask_flags_exactly_the_copies() -> None:
    m, rec, cut, _ = _one_tet(X_CUT)
    mask = cut.copy_mask.numpy()
    assert mask.dtype == np.int8 and mask.shape == (m.particle_count,)
    np.testing.assert_array_equal(np.flatnonzero(mask), np.sort(rec.node.reshape(-1)))


@wp.kernel
def _crossings(tt: TetTable, hub_s: wp.array(dtype=wp.float32), order: wp.array(dtype=wp.int32),
               edge_t: wp.array(dtype=wp.float32)):
    i = wp.tid()
    c = order[i]
    s = wp.vec4(hub_s[tt.hub_row[c, 0]], hub_s[tt.hub_row[c, 1]], hub_s[tt.hub_row[c, 2]], hub_s[tt.hub_row[c, 3]])
    crossing_params(tt, c, s, edge_t)


def test_shared_edge_crossings_do_not_depend_on_thread_order() -> None:
    """Every tet on a crossed edge computes its parameter in the edge's canonical order, so the stored bits are the
    same whichever thread writes first (forward vs reversed tet order)."""
    sc = vnode_block("cpu")
    cut = VNodeCutter(sc.model, sc.copies, sc.blades, 1.6e-3)
    rest = sc.model.particle_q.numpy()[sc.copies.node_hub]
    n = np.array([np.cos(0.37), np.sin(0.37), 0.21])
    hub_s = ((rest - np.array([8.3e-3, 7.0e-3, 6.1e-3])) @ (n / np.linalg.norm(n))).astype(np.float32)
    runs = []
    for order in (np.arange(sc.copies.count), np.arange(sc.copies.count)[::-1]):
        edge_t = wp.full(len(sc.copies.edge), -1.0, dtype=wp.float32, device="cpu")
        wp.launch(_crossings, dim=sc.copies.count, device="cpu",
                  inputs=[cut.tets, wp.array(hub_s, dtype=wp.float32, device="cpu"),
                          wp.array(order.astype(np.int32), dtype=wp.int32, device="cpu"), edge_t])
        runs.append(edge_t.numpy())
    assert (runs[0] >= 0).sum() > 20                                                   # many shared crossings
    np.testing.assert_array_equal(runs[0].view(np.int32), runs[1].view(np.int32))     # bit for bit


def test_a_split_keeps_mass_and_momentum() -> None:
    m, rec, cut, body = _one_tet(X_CUT)
    s = m.state()
    s.particle_qd.assign(np.random.default_rng(2).normal(scale=0.1, size=(m.particle_count, 3)).astype(np.float32))
    qd = s.body_qd.numpy()
    qd[body, 2] = -0.05
    s.body_qd.assign(qd)

    def totals():
        mass, v = m.particle_mass.numpy().astype(np.float64), s.particle_qd.numpy().astype(np.float64)
        return mass.sum(), (mass[:, None] * v).sum(axis=0)

    m0, p0 = totals()
    cut.pre_step(s)
    m1, p1 = totals()
    assert cut.split.numpy().tolist() == [1]
    assert m1 == pytest.approx(m0, rel=1e-6)
    np.testing.assert_allclose(p1, p0, rtol=1e-6, atol=1e-12)
    frac = m.particle_mass.numpy()[rec.node[rec.hub_row[0, 0]]] / ((0.5 * rec.mass[0]) / 4.0)
    assert frac.sum() == pytest.approx(1.0, rel=1e-6) and np.all(frac > 0.0)        # f+ + f- = 1


class _Descent:
    """Pre-substep pose writer: the blade straight down at the rate limit, as test_bully_target's hook."""

    def __init__(self, sim: VNodeSim, x: float) -> None:
        self.sim, self.body = sim, int(sim.scene.blade_bodies[0])
        self.z0 = self.z1 = sim.scene.size[2] + 2.0e-3 + 0.02
        self.low, self.k = -3.0e-3 + 0.02, 0
        sim.pre_substep.append(self)

    def period(self) -> None:
        self.z0, self.z1 = self.z1, max(self.z1 - 0.05 * 0.016, self.low)
        self.sim.step_period()

    def __call__(self, s) -> None:
        bq, bqd = s.body_q.numpy(), s.body_qd.numpy()
        bq[self.body, 2] = self.z0 + (self.k / 16) * (self.z1 - self.z0)
        bqd[self.body] = 0.0
        bqd[self.body, 2] = (self.z1 - self.z0) / 0.016
        s.body_q.assign(bq)
        s.body_qd.assign(bqd)
        self.k = (self.k + 1) % 16


@pytest.fixture(scope="module")
def stroked():
    sc = vnode_block("cpu")
    m = sc.model
    m.soft_contact_ke = 3.0e3
    q = m.joint_q.numpy()
    q[0:7] = blade_pose(np.array([X_CUT, 0.5 * sc.size[1], sc.size[2] + 2.0e-3 + 0.02]), np.array([0.0, 1.0, 0.0]),
                        np.array([0.0, 0.0, -1.0]))
    m.joint_q.assign(q)
    sim = VNodeSim(sc, newton.solvers.SolverVBD(m, iterations=4), control_dt=0.016, substeps=16)
    hook, m0, vmax = _Descent(sim, X_CUT), float(m.particle_mass.numpy().sum()), 0.0
    for _ in range(30):
        hook.period()
        vmax = max(vmax, float(np.linalg.norm(sim.state.particle_qd.numpy(), axis=1).max()))
        sim.dev = sim.cut_face_deviation()                                              # sampled per period
    sim.end = {"x": sim.state.particle_q.numpy().copy(), "split": sim.cutter.split.numpy().copy(),
               "mass": float(sim.model.particle_mass.numpy().sum())}                   # later tests move the sim
    return sim, m0, vmax


def _live_inverted(sim, x: np.ndarray | None = None) -> int:
    m = sim.model
    x = sim.state.particle_q.numpy() if x is None else x
    mat, tets = m.tet_materials.numpy(), m.tet_indices.numpy()
    live = (mat[:, 0] + mat[:, 1]) > 0.0
    xt = x[tets[live]]
    f = np.stack([xt[:, 1] - xt[:, 0], xt[:, 2] - xt[:, 0], xt[:, 3] - xt[:, 0]], axis=2) @ m.tet_poses.numpy()[live]
    return int((np.linalg.det(f) <= 0.0).sum())


def test_a_stroke_off_the_seams_splits_the_crossed_tets_and_moves_no_material(stroked) -> None:
    sim, m0, vmax = stroked
    c = sim.scene.copies
    hx = sim.model.particle_q.numpy()[c.hub][:, :, 0]
    crossed = (hx.min(axis=1) < X_CUT) & (hx.max(axis=1) > X_CUT)
    np.testing.assert_array_equal(sim.end["split"] == 1, crossed)                      # exactly the crossed tets
    assert int(crossed.sum()) == 54
    assert sim.end["mass"] == pytest.approx(m0, rel=1e-6)
    assert vmax < 1e-3 and _live_inverted(sim, sim.end["x"]) == 0
    hubs = slice(0, sim.scene.hubs)
    moved = np.linalg.norm(sim.end["x"][hubs] - sim.model.particle_q.numpy()[hubs], axis=1)
    assert moved.max() < 5e-5                                                           # the cut opened ahead
    assert sim.dev.shape == (1,) and 0.0 <= sim.dev[0] <= 1e-6                          # faces in the swept plane


def test_after_the_stroke_the_far_piece_comes_away(stroked) -> None:
    """The far piece (hubs beyond the cut, x > X_CUT, the blade face's + side, and the + copies of the hubs the cut
    touched) set moving at 0.02 m/s drifts freely; the clamped side stays. (Velocity on the hubs alone strains their
    welds, and VBD's few Gauss-Seidel iterations on a weld that stiff relative to a hub's inertia do not keep
    momentum.)"""
    sim, _, _ = stroked
    h, c = sim.scene.hubs, sim.scene.copies
    x0 = sim.state.particle_q.numpy()[:h].copy()
    far = x0[:, 0] > X_CUT
    touched = sim.cutter.hub_side.numpy() >= 0
    moving = np.concatenate([np.flatnonzero(far), c.node[touched, 0]])
    v = sim.state.particle_qd.numpy()
    v[moving] = (0.02, 0.0, 0.0)
    sim.state.particle_qd.assign(v)
    for _ in range(5):
        sim.step_period()                                                               # the blade stays down
    d = sim.state.particle_q.numpy()[:h] - x0
    travel = 0.02 * 5 * 0.016
    np.testing.assert_allclose(d[far, 0], travel, rtol=0.05)                            # free: a rigid drift
    assert np.abs(d[~far]).max() < 0.05 * travel                                        # the clamped side stays


@pytest.mark.gpu
@pytest.mark.smoke
def test_the_cutter_passes_captured_equal_eager() -> None:
    """Each mode twice: eager, eager, captured, captured give the same splits, masses, materials, welds and edge
    crossings (the shared-edge crossings are written by several threads in one launch: WP5 review). The blade moves
    down, so the edge leads and the split passes run (WP4 review)."""
    if not wp.get_cuda_device_count():
        pytest.skip("no CUDA device")
    out = []
    for capture in (False, False, True, True):
        sc = vnode_block("cuda:0")
        m = sc.model
        q = m.joint_q.numpy()
        q[0:7] = blade_pose(np.array([X_CUT, 0.5 * sc.size[1], 0.010 + 0.02]), np.array([0.0, 1.0, 0.0]),
                            np.array([0.0, 0.0, -1.0]))
        m.joint_q.assign(q)
        s = m.state()
        newton.eval_fk(m, s.joint_q, s.joint_qd, s)
        qd = s.body_qd.numpy()
        qd[sc.blade_bodies[0], 2] = -0.05                                              # the edge leads: splits run
        s.body_qd.assign(qd)
        cut = VNodeCutter(m, sc.copies, sc.blades, 1.6e-3)
        if capture:
            with wp.ScopedCapture(device="cuda:0") as cap:
                cut.pre_step(s)
            wp.capture_launch(cap.graph)
        else:
            cut.pre_step(s)
        out.append((cut.split.numpy(), m.particle_mass.numpy(), m.tet_materials.numpy(), m.spring_stiffness.numpy(),
                    cut.edge_t.numpy()))
        assert out[-1][0].sum() > 0                                                     # the passes ran (WP4 review)
    for run in out[1:]:
        for a, b in zip(out[0], run, strict=True):
            np.testing.assert_array_equal(a, b)


def test_an_oblique_stroke_cuts_without_kicks() -> None:
    """Free-class worlds 0 and 2 (yaw -24.8 / -29.9 deg, lean 3.7 / 2.7 deg), straight down: the cut plane crosses the
    grid obliquely. Material (hubs) stays under 0.2 m/s and virtual copies under 0.5 m/s (blade 0.05 m/s); no live
    tet inverts; mass is exact. (Copies private per tet ran virtual nodes at 4 m/s; stock blade contacts on hubs
    near the plane 1 m/s.)"""
    from alloy.topology import bully_target as bt

    sc = vnode_block("cpu", worlds=2)
    m = sc.model
    m.soft_contact_ke = bt.STROKE_KE
    ang = bt.free_angles(3)[[0, 2]]
    start = np.array([0.5 * sc.size[0], 0.5 * sc.size[1], sc.size[2] + 2.0e-3 + bt.EDGE_OFFSET])
    q, qs = m.joint_q.numpy(), m.joint_q_start.numpy()
    for w in range(2):
        q[qs[w]:qs[w] + 7] = bt.free_pose(start, *ang[w])
    m.joint_q.assign(q)
    sim = VNodeSim(sc, newton.solvers.SolverVBD(m, iterations=4), control_dt=0.016, substeps=16)
    hook, m0 = _Descent(sim, 0.0), float(m.particle_mass.numpy().sum())
    hook.body = sc.blade_bodies
    per_world = m.particle_count // 2
    hub = (np.arange(m.particle_count) % per_world) < sc.hubs
    v_hub = v_copy = 0.0
    for _ in range(26):
        hook.period()
        sp = np.linalg.norm(sim.state.particle_qd.numpy(), axis=1)
        v_hub, v_copy = max(v_hub, float(sp[hub].max())), max(v_copy, float(sp[~hub].max()))
        dev = sim.cut_face_deviation()
    split = sim.cutter.split.numpy() == 1
    assert np.all(np.bincount(sim.cutter.tets.world.numpy()[split], minlength=2) >= 54)  # cut through, each world
    assert v_hub < 0.2 and v_copy < 0.5
    # Diagnostic, not a physics limit (coordinator ruling 2026-11-06T18:59Z; path fidelity is judged in the material
    # frame, ruling 1): rest-frame crossings vs the world sweep also count how far the blade's passage moved the
    # material before an edge was crossed. 0.81 / 1.13 mm here at WELD_KE_RATIO 1 (0.90 / 1.04 mm at 100); step 3's
    # 0.1 h = 0.5 mm target is not met on this measure. The bound guards the measured level.
    assert np.all(dev <= 1.2e-3)
    assert _live_inverted(sim) == 0
    assert float(m.particle_mass.numpy().sum()) == pytest.approx(m0, rel=1e-6)


def test_the_copies_target_takes_the_free_class_band_in_the_blade_frame() -> None:
    """cut_block_free_copies has cut_block_free's blade-frame actions (+-1 mm face-first band, ruling
    2026-11-06T14:17Z) and the same per-world blades."""
    from alloy.topology import bully_target as bt
    from alloy.topology.vnode_target import cut_block_free_copies

    t, free = cut_block_free_copies(), bt.cut_block_free()
    np.testing.assert_array_equal(t.actions.low, free.actions.low)
    np.testing.assert_array_equal(t.actions.high, free.actions.high)
    np.testing.assert_allclose([t.actions.low[0], t.actions.high[0]], [-1.0e-3, 1.0e-3])
    sim, ref = t.make_sim(2, "cpu"), free.make_sim(2, "cpu")
    for got, want in zip(sim.bully_action_frame(), ref.bully_action_frame(), strict=True):
        np.testing.assert_allclose(got, want, atol=1e-7)
    q = sim.state.body_q.numpy()[sim.blades.numpy()]
    np.testing.assert_allclose(q[:, :3], np.tile(bt.free_start(), (2, 1)), atol=1e-7)



def test_the_copies_target_is_found_by_name_in_a_fresh_process() -> None:
    """get_target / target_names reach cut_block_free_copies through WP7's provider list (bully_target registers it;
    WP4 review: registered only on vnode_target's import, a fresh process could not find it)."""
    code = ("from alloy.bully.targets import get_target, target_names\n"
            "assert 'cut_block_free_copies' in target_names()\n"
            "assert get_target('cut_block_free_copies').name == 'cut_block_free_copies'\n")
    root = str(Path(__file__).resolve().parents[2])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([root, os.environ.get("PYTHONPATH", "")])}
    subprocess.run([sys.executable, "-c", code], check=True, env=env, cwd=root, timeout=300)


def test_copies_refuse_a_heavier_density_than_the_mesh() -> None:
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    x = np.array([[0, 0, 0], [5, 0, 0], [0, 5, 0], [0, 0, 5]], float) * 1e-3
    b.add_soft_mesh(vertices=[wp.vec3(*v) for v in x], indices=[0, 1, 2, 3], **soft_kwargs(density=1000.0))
    with pytest.raises(ValueError, match="corner shares"):
        add_tet_copies(b, [0], 2000.0)                         # a hub could hand over more mass than it has
    add_tet_copies(b, [0], 1000.0)


def test_the_cutter_refuses_two_blades_in_one_world() -> None:
    m, rec, _, body = _one_tet(X_CUT)
    blade = Blade(body=body, shape=-1, edge_a=(0.0, -0.05, -0.02), edge_b=(0.0, 0.05, -0.02))
    with pytest.raises(AssemblyError, match="one blade per world") as err:      # an Alloy error with its way out:
        VNodeCutter(m, rec, [blade, blade], 1.6e-3)                               # alloy.build raises BuildError
    assert "cut_model: seams" in err.value.fix


def test_mid_stroke_deviation_counts_only_crossings_the_edge_has_passed() -> None:
    """cut_block_free_copies' free descent through the bully runner (2 worlds, 8 ticks): splits run up to a cell
    ahead of the edge, and counted from the split their crossings read that look-ahead (5.6-6.3 mm, falling 0.8 mm a
    tick). Counted once the edge passes them (``behind_edges``): 0-0.2 mm."""
    from alloy.bully.replay import EpisodeRunner
    from alloy.bully.types import BullyLimits
    from alloy.topology import bully_target as bt

    t = bt.cut_block_free_copies()
    a = t.actions
    acts = np.tile(a.initial, (8, 1))
    acts[:, 2] = np.minimum(a.max_delta_per_tick[2] * np.arange(1, 9), a.high[2])
    runner = EpisodeRunner(t, 2, "cpu", BullyLimits())
    res = runner.run(acts)
    assert int(runner.sim.cutter.split.numpy().sum()) > 2 * 20                      # mid-stroke, splits ahead
    assert np.all(res.monitors["cut_face_deviation"] < 1.0e-3)


def test_behind_edges_uses_the_spine_side() -> None:
    from alloy.topology.vnode_rig import behind_edges

    sc = vnode_block("cpu")
    cut = VNodeCutter(sc.model, sc.copies, sc.blades, 1.6e-3)
    q = sc.model.body_q.numpy()                                       # blade at z = 1 m, edge 0.02 m below, spine up
    edge_z = float(q[sc.blade_bodies[0], 2]) - 0.02
    pts = np.array([[0.01, 0.0, edge_z + 1e-3], [0.01, 0.0, edge_z - 1e-3], [0.03, 0.2, edge_z + 1e-4]])
    np.testing.assert_array_equal(behind_edges(pts, np.zeros(3, np.int64), cut.blades, q), [True, False, True])
