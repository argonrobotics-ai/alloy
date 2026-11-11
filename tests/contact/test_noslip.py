# SPDX-License-Identifier: Apache-2.0
"""Scoped noslip (alloy.contact.noslip): MuJoCo Warp's noslip on the rows that touch a robot, per kinematic tree.

- scope "all" reproduces MuJoCo Warp's own noslip on the incline box (same rows, order, cone projection and stopping
  rule): one step from identical states, then 800 more;
- scope "robot" on T12 selects the two arms (dof friction rows, finger/arm geoms) and not the blocks, and on the
  T12 trace start (sparse Jacobian, nv 64) gives the same state as MuJoCo Warp's noslip;
- a geom pair (precision pair) scopes only its own contacts, and a tree no scoped row touches keeps the main
  solver's qacc bit for bit (its factor is skipped);
- surface geoms scope their contacts with geoms outside the set (not with each other): the cable-to-bench rows;
  scope robot leaves robot-robot contacts out (BUL-0010);
- a tree over 8 dofs (the warp-wide factor) also reproduces MuJoCo Warp's noslip under scope "all".
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
import warp as wp

from alloy._vendor.mujoco_warp._src.solver import _qcqp, _qcqp2

from alloy.contact.noslip import (ALL, ROBOT, NoslipSpec, build_scope, install_noslip, pair_geoms, robot_trees,
                                  shape_geoms)
from alloy.contact import noslip_kernels as nk
from alloy.contact.noslip_cone import cone_force, qcqp5, qcqp5_l0, qcqp5_warm, scaled_factor
from alloy.contact.types import PrecisionPairSpec
from alloy.core import newton_compat as nc
from alloy.core.errors import SolverOptionError
from alloy.testing.scenes import incline
from noslip_scenes import snake_on_slope, two_boxes_on_plank

T12_XML = Path(__file__).resolve().parents[2] / "library/robots/task_t12/scene.xml"


def _incline_pair(steps: int = 150):
    runners = [incline(2, 0.8, 0.4, device="cpu").runner("mujoco", cone="elliptic") for _ in range(2)]
    for r in runners:
        r.step(steps)
    return runners


@pytest.mark.compile_heavy
@pytest.mark.parametrize(("iterations", "tolerance"), [(10, 1e-30), (50, 1e-10)])
def test_scope_all_matches_mujoco_warp(iterations: int, tolerance: float) -> None:
    ref, alloy = _incline_pair()
    assert np.array_equal(ref.state.body_q.numpy(), alloy.state.body_q.numpy())
    nc.enable_mujoco_noslip(ref.solver, iterations, tolerance)
    inst = install_noslip(alloy.solver, NoslipSpec(iterations, tolerance, scope=ALL))
    ref.step(1)
    alloy.step(1)
    a, b = ref.solver.mjw_data, alloy.solver.mjw_data
    n = int(a.nefc.numpy()[0])
    assert n > 0 and np.array_equal(a.solver_niter.numpy(), b.solver_niter.numpy())
    assert np.allclose(a.efc.force.numpy()[:, :n], b.efc.force.numpy()[:, :n], rtol=0, atol=1e-5)
    assert np.allclose(a.qacc.numpy(), b.qacc.numpy(), rtol=0, atol=1e-4)
    assert np.allclose(a.efc.Ma.numpy(), b.efc.Ma.numpy(), rtol=0, atol=1e-4)
    ref.step(800)
    alloy.step(800)
    assert np.abs(ref.state.body_q.numpy() - alloy.state.body_q.numpy()).max() < 1e-7  # float32 summation order
    assert inst.overflow_worlds() == 0


@pytest.mark.compile_heavy
def test_pair_scope_touches_only_its_tree() -> None:
    """Scope "robot" (no robot here) plus the (plank, box A) pair: box A's contacts get noslip, box B keeps the main
    solver's qacc bit for bit (its tree is neither factored nor restarted)."""
    scene = two_boxes_on_plank()
    ref, alloy = (scene.runner("mujoco", cone="elliptic") for _ in range(2))
    for r in (ref, alloy):
        r.step(150)
    pairs = pair_geoms(alloy.solver, [PrecisionPairSpec(1, 0)])
    mjm = alloy.solver.mj_model
    box_a, box_b = (int(mjm.body_treeid[mjm.geom_bodyid[g]]) for g in (pairs[0][0], 2))
    assert sorted(pairs[0]) == [0, 1] and box_a != box_b
    inst = install_noslip(alloy.solver, NoslipSpec(50, 1e-10, scope=ROBOT, pairs=pairs))
    ref.step(1)
    alloy.step(1)
    qa, qb = ref.solver.mjw_data.qacc.numpy()[0], alloy.solver.mjw_data.qacc.numpy()[0]
    dofs = {t: np.asarray(mjm.dof_treeid) == t for t in (box_a, box_b)}
    assert int(inst.count.numpy()[0]) >= 1
    assert np.array_equal(qa[dofs[box_b]], qb[dofs[box_b]])
    assert not np.array_equal(qa[dofs[box_a]], qb[dofs[box_a]])
    assert inst.overflow_worlds() == 0


@pytest.mark.compile_heavy
@pytest.mark.parametrize(("surfaces", "moved", "kept"), [((1,), 1, 2), ((0, 1), 2, 1)])
def test_surface_scope_skips_contacts_inside_the_set(surfaces: tuple[int, ...], moved: int, kept: int) -> None:
    """Surface geoms scope their contacts with geoms outside the set: (box A) scopes A-plank; (plank, box A) scopes
    B-plank only, since A-plank has both geoms in the set (a cable's link-link contacts stay out the same way)."""
    scene = two_boxes_on_plank()
    ref, alloy = (scene.runner("mujoco", cone="elliptic") for _ in range(2))
    for r in (ref, alloy):
        r.step(150)
    mjm = alloy.solver.mj_model
    geoms = shape_geoms(alloy.solver, surfaces)
    inst = install_noslip(alloy.solver, NoslipSpec(50, 1e-10, scope=ROBOT, surfaces=geoms))
    ref.step(1)
    alloy.step(1)
    all_geoms = shape_geoms(alloy.solver, (0, 1, 2))
    tree = {s: int(mjm.body_treeid[mjm.geom_bodyid[g]]) for s, g in zip((0, 1, 2), all_geoms)}
    qa, qb = ref.solver.mjw_data.qacc.numpy()[0], alloy.solver.mjw_data.qacc.numpy()[0]
    dof_tree = np.asarray(mjm.dof_treeid)
    assert int(inst.count.numpy()[0]) >= 1 and inst.overflow_worlds() == 0
    assert np.array_equal(qa[dof_tree == tree[kept]], qb[dof_tree == tree[kept]])
    assert not np.array_equal(qa[dof_tree == tree[moved]], qb[dof_tree == tree[moved]])


@pytest.mark.compile_heavy
def test_scope_all_large_tree_matches_mujoco_warp() -> None:
    """One 10-dof tree (over MAXT8: the block's warp-wide factor and, with restart=True, MuJoCo's qacc restart)
    against MuJoCo Warp's noslip."""
    ref, alloy = (snake_on_slope().runner("mujoco", cone="elliptic") for _ in range(2))
    for r in (ref, alloy):
        r.step(150)
    assert int(alloy.solver.mj_model.tree_dofnum.max()) == 10
    nc.enable_mujoco_noslip(ref.solver, 50, 1e-10)
    inst = install_noslip(alloy.solver, NoslipSpec(50, 1e-10, scope=ALL, restart=True))
    ref.step(1)
    alloy.step(1)
    a, b = ref.solver.mjw_data, alloy.solver.mjw_data
    n = int(a.nefc.numpy()[0])
    assert n > 0 and int(inst.count.numpy()[0]) >= 5
    assert np.array_equal(a.solver_niter.numpy(), b.solver_niter.numpy())
    assert np.allclose(a.efc.force.numpy()[:, :n], b.efc.force.numpy()[:, :n], rtol=0, atol=1e-5)
    assert np.allclose(a.qacc.numpy(), b.qacc.numpy(), rtol=0, atol=1e-4)
    ref.step(400)
    alloy.step(400)
    assert np.abs(ref.state.body_q.numpy() - alloy.state.body_q.numpy()).max() < 1e-6


@pytest.mark.compile_heavy
def test_overflow_and_uninstall() -> None:
    (r,) = _incline_pair()[:1]
    inst = install_noslip(r.solver, NoslipSpec(10, 1e-10, scope=ALL, rows_per_world=1))
    r.step(2)
    assert inst.overflow_worlds() == 2  # four box corners per world, one slot
    inst.uninstall()
    assert r.solver.mjw_model.opt.noslip_iterations == 0


def test_dropped_solver_is_freed() -> None:
    """The noslip route registry holds nothing strongly (WP8 2026-11-05: it kept every dropped simulation's solver,
    MuJoCo Warp model and data alive, ~0.42 GiB per 512-world build on an L4)."""
    import gc  # noqa: PLC0415
    import weakref  # noqa: PLC0415

    from alloy.contact import noslip as NS  # noqa: PLC0415

    refs = []
    for _ in range(3):
        (r,) = _incline_pair(steps=1)[:1]
        install_noslip(r.solver, NoslipSpec(10, 1e-10, scope=ALL))
        r.step(1)
        assert id(r.solver.mjw_model) in NS._ROUTES
        refs.append((weakref.ref(r.solver), weakref.ref(r.solver.mjw_model), id(r.solver.mjw_model)))
        del r
    gc.collect()
    assert [(s(), m()) for s, m, _ in refs] == [(None, None)] * 3
    assert not any(key in NS._ROUTES for *_, key in refs)


def test_robot_scope_on_t12() -> None:
    mjm = mujoco.MjModel.from_xml_path(str(T12_XML))
    trees = robot_trees(mjm)
    arms = {int(mjm.body_treeid[mjm.body(f"{s}_link_1").id]) for s in ("left", "right")}
    assert set(trees.tolist()) == arms
    scope = build_scope(mjm, ROBOT)
    for name in ("left_finger_a", "right_finger_b", "left_link_6"):
        body = mjm.body(name).id
        assert scope.geom[mjm.geom_bodyid == body].all(), name
    block = mjm.body("block_6").id
    assert not scope.geom[mjm.geom_bodyid == block].any()
    assert not scope.geom[mjm.geom_bodyid == 0].any()
    arm_dofs = np.isin(mjm.dof_treeid, list(arms))
    assert np.array_equal(scope.dof.astype(bool), arm_dofs)
    assert scope.max_friction_rows == int(mjm.geom_condim.max()) - 1 == 5  # condim-6 finger pads
    assert scope.max_tree_dofs == 8
    assert build_scope(mjm, ALL).geom[mjm.geom_bodyid == block].all()
    pair = build_scope(mjm, ROBOT, [(int(np.nonzero(mjm.geom_bodyid == block)[0][0]), 0)])
    assert pair.pairs.tolist() == [[0, int(np.nonzero(mjm.geom_bodyid == block)[0][0])]]
    assert pair.geom[0] == 2 and not pair.geom[mjm.geom_bodyid == block][1:].any()


def test_rejects_unsupported() -> None:
    mjm = mujoco.MjModel.from_xml_path(str(T12_XML))
    with pytest.raises(SolverOptionError):
        build_scope(mjm, "grasp")
    with pytest.raises(SolverOptionError):
        build_scope(mjm, ROBOT, [(0, mjm.ngeom)])
    with pytest.raises(SolverOptionError):
        build_scope(mjm, ROBOT, surfaces=[mjm.ngeom])
    solver = incline(1, 0.8, 0.4, device="cpu").make_solver("mujoco", cone="pyramidal")
    with pytest.raises(SolverOptionError):
        install_noslip(solver)
    solver = incline(1, 0.8, 0.4, device="cpu").make_solver("mujoco", cone="elliptic")
    with pytest.raises(SolverOptionError):
        install_noslip(solver, NoslipSpec(iterations=0))


@pytest.mark.compile_heavy
def test_t12_robot_scope_matches_full_noslip_at_trace_start() -> None:
    """Sparse path on the recorded scene: the first 0.1 s (arm moving, blocks at rest) with the scoped rows equals
    MuJoCo Warp's noslip over every row, because the resting blocks' friction rows carry no slip to remove."""
    from alloy.tasks.cli import TRACES
    from alloy.tasks.replay import ReplayConfig, build_rig, import_scene, run_replay
    from alloy.tasks.trace import load_trace

    tt = load_trace(TRACES / "t12")
    cfg = ReplayConfig(seconds=0.1, device="cpu")
    out = []
    for scoped in (False, True):
        with wp.ScopedDevice("cpu"):
            rig = build_rig(import_scene(T12_XML, "cpu"), tt, cfg)
            if scoped:
                inst = install_noslip(rig.solver, NoslipSpec(50, 1e-10, scope=ROBOT))
            else:
                nc.enable_mujoco_noslip(rig.solver, 50, 1e-10)
            run_replay(rig, cfg)
            out.append(rig.states[0].body_q.numpy())
    assert inst.overflow_worlds() == 0
    assert int(inst.count.numpy()[0]) >= 16  # the two arms' dof friction rows
    assert np.abs(out[0][:, :3] - out[1][:, :3]).max() < 2e-6


@pytest.mark.compile_heavy
def test_force_space_sweep_matches_qacc_space_on_t12() -> None:
    """The force-space sweep (noslip_dl: Delassus matrix of the scoped rows, dof-row shortcut in C) and the qacc-space
    sweep give the same 0.1 s replay of the recorded scene (16 dof friction rows, the arms moving)."""
    from alloy.tasks.cli import TRACES
    from alloy.tasks.replay import ReplayConfig, build_rig, import_scene, run_replay
    from alloy.tasks.trace import load_trace

    tt = load_trace(TRACES / "t12")
    cfg = ReplayConfig(seconds=0.1, device="cpu")
    out = []
    for fs in (False, True):
        with wp.ScopedDevice("cpu"):
            rig = build_rig(import_scene(T12_XML, "cpu"), tt, cfg)
            inst = install_noslip(rig.solver, NoslipSpec(50, 1e-10, scope=ROBOT, force_space=fs))
            run_replay(rig, cfg)
            out.append(rig.states[0].body_q.numpy())
            assert int(inst.count.numpy()[0]) >= 16
    assert np.abs(out[0][:, :3] - out[1][:, :3]).max() < 2e-6


@wp.kernel(enable_backward=False)
def _qcqp_pair(A: wp.array[wp.types.matrix((5, 5), float)], b: wp.array[wp.types.vector(5, float)],
               d: wp.array[wp.types.vector(5, float)], r: wp.array[float],
               out: wp.array2d[wp.types.vector(5, float)], active: wp.array2d[int]):
    i = wp.tid()
    x0, a0 = _qcqp(A[i], b[i], d[i], r[i], 5)
    x1, a1 = qcqp5(A[i], b[i], d[i], r[i])
    out[i, 0] = x0
    out[i, 1] = x1
    active[i, 0] = wp.where(a0, 1, 0)
    active[i, 1] = wp.where(a1, 1, 0)


def test_qcqp5_matches_mujoco_warp() -> None:
    """The static-loop 5-D cone QCQP (condim-6 contacts) gives MuJoCo Warp's generic _qcqp result, inside and on
    the cone."""
    rng = np.random.default_rng(3)
    n = 256
    g = rng.normal(size=(n, 5, 5))
    A = (g @ g.transpose(0, 2, 1) + 0.1 * np.eye(5)).astype(np.float32)
    b = rng.normal(size=(n, 5)).astype(np.float32)
    d = rng.uniform(0.2, 1.0, size=(n, 5)).astype(np.float32)
    r = rng.uniform(0.01, 5.0, size=n).astype(np.float32)
    with wp.ScopedDevice("cpu"):
        out = wp.zeros((n, 2), dtype=wp.types.vector(5, float))
        act = wp.zeros((n, 2), dtype=int)
        wp.launch(_qcqp_pair, dim=n, inputs=[wp.array(A, dtype=wp.types.matrix((5, 5), float)),
                                             wp.array(b, dtype=wp.types.vector(5, float)),
                                             wp.array(d, dtype=wp.types.vector(5, float)), wp.array(r), out, act])
        x, a = out.numpy(), act.numpy()
    assert 0 < a[:, 0].sum() < n  # both regimes covered
    assert np.array_equal(a[:, 0], a[:, 1])
    assert np.array_equal(x[:, 0], x[:, 1])  # same operations in the same order


@wp.kernel(enable_backward=False)
def _qcqp_warm(A: wp.array[nk.mjt.mat55], b: wp.array[nk.mjt.vec5], d: wp.array[nk.mjt.vec5], r: wp.array[float],
               scale: wp.array[float], out: wp.array2d[nk.mjt.vec5], la: wp.array2d[float]):
    i, j = wp.tid()
    x0, a0, l0 = qcqp5_warm(A[i], b[i], d[i], r[i], 0.0)
    x1, a1, l1 = qcqp5_warm(A[i], b[i], d[i], r[i], wp.where(l0 > 0.0, l0 * scale[j], scale[j]))
    out[i, j] = x1
    la[i, j] = l1
    if j == 0:
        out[i, scale.shape[0]] = x0
        la[i, scale.shape[0]] = l0


def test_qcqp5_warm_start_same_root() -> None:
    """Started from the previous sweep's la (left of the root, on it, right of it) or from la > 0 when the solution is
    inside the cone, the condim-6 QCQP returns the cold-start (MuJoCo) result: same la = 0 inside the cone, the same
    point on it otherwise."""
    rng = np.random.default_rng(5)
    n = 256
    g = rng.normal(size=(n, 5, 5))
    A = (g @ g.transpose(0, 2, 1) + 0.1 * np.eye(5)).astype(np.float32)
    b = rng.normal(size=(n, 5)).astype(np.float32)
    d = rng.uniform(0.2, 1.0, size=(n, 5)).astype(np.float32)
    r = rng.uniform(0.01, 5.0, size=n).astype(np.float32)
    scale = np.array([1.0, 0.5, 0.9, 1.1, 2.0, 10.0], np.float32)  # la0 = la * scale (or = scale inside the cone)
    with wp.ScopedDevice("cpu"):
        out = wp.zeros((n, len(scale) + 1), dtype=nk.mjt.vec5)
        la = wp.zeros((n, len(scale) + 1), dtype=float)
        wp.launch(_qcqp_warm, dim=(n, len(scale)), inputs=[
            wp.array(A, dtype=nk.mjt.mat55), wp.array(b, dtype=nk.mjt.vec5), wp.array(d, dtype=nk.mjt.vec5),
            wp.array(r), wp.array(scale), out, la])
        x, lam = out.numpy(), la.numpy()
    cold, cold_la = x[:, -1:], lam[:, -1:]
    inside = cold_la[:, 0] == 0.0
    assert 0 < inside.sum() < n
    assert np.all(lam[inside, :-1] == 0.0)
    assert np.array_equal(x[inside, :-1], np.repeat(cold[inside], len(scale), axis=1))
    rel = np.abs(x[~inside, :-1] - cold[~inside]).max(axis=2) / np.abs(cold[~inside]).max(axis=2)
    assert rel.max() < 1e-4
    assert np.abs(lam[~inside, :-1] / cold_la[~inside] - 1.0).max() < 1e-3


@wp.kernel(enable_backward=False)
def _qcqp_l0(A: wp.array[nk.mjt.mat55], b: wp.array[nk.mjt.vec5], d: wp.array[nk.mjt.vec5], r: wp.array[float],
             out: wp.array2d[nk.mjt.vec5], la: wp.array2d[float]):
    i = wp.tid()
    x0, a0, l0 = qcqp5_warm(A[i], b[i], d[i], r[i], 0.0)
    L, ok = scaled_factor(A[i], d[i])
    x1, a1, l1 = qcqp5_l0(A[i], b[i], d[i], r[i], 0.0, L, ok)
    out[i, 0] = x0
    out[i, 1] = x1
    la[i, 0] = l0
    la[i, 1] = l1


def test_qcqp5_cached_factor_is_bit_identical() -> None:
    """With the scaled block's la = 0 factor computed once (scaled_factor, the force-space sweep's per-contact cache),
    the condim-6 QCQP returns qcqp5_warm's result bit for bit, inside the cone (no factorization) and on it."""
    rng = np.random.default_rng(7)
    n = 256
    g = rng.normal(size=(n, 5, 5))
    A = (g @ g.transpose(0, 2, 1) + 0.1 * np.eye(5)).astype(np.float32)
    b = rng.normal(size=(n, 5)).astype(np.float32)
    d = rng.uniform(0.2, 1.0, size=(n, 5)).astype(np.float32)
    r = rng.uniform(0.01, 5.0, size=n).astype(np.float32)
    with wp.ScopedDevice("cpu"):
        out = wp.zeros((n, 2), dtype=nk.mjt.vec5)
        la = wp.zeros((n, 2), dtype=float)
        wp.launch(_qcqp_l0, dim=n, inputs=[wp.array(A, dtype=nk.mjt.mat55), wp.array(b, dtype=nk.mjt.vec5),
                                           wp.array(d, dtype=nk.mjt.vec5), wp.array(r), out, la])
        x, lam = out.numpy(), la.numpy()
    assert 0 < (lam[:, 0] == 0).sum() < n
    assert np.array_equal(x[:, 0], x[:, 1]) and np.array_equal(lam[:, 0], lam[:, 1])


@wp.kernel(enable_backward=False)
def _keep_rows(rows: nk.Rows, con: nk.Contacts, out: wp.array[int]):
    r = wp.tid()
    out[r] = nk.keep_row(0, r, rows, con)


def test_keep_row_scope_bits() -> None:
    """Which efc rows start a slot under scope robot: robot-object contacts yes, robot-robot no (an arm folding onto
    itself: BUL-0010); surface-other yes, surface-surface no (a cable's own links); a listed pair yes; a scoped dof
    friction row yes. Geoms: 0, 1 robot; 2 world; 3, 4 surface; 5, 6 a pair."""
    rob, surf, pair = nk.ROBOT_BIT, nk.SURF_BIT, nk.PAIR_BIT
    contacts = [(0, 2), (0, 1), (3, 2), (3, 4), (5, 6), (5, 2)]
    expect = [1, 0, 1, 0, 1, 0, 1]
    ell, fdof = nk.ELLIPTIC, nk.FRICTION_DOF
    n = len(contacts)
    rows, con = nk.Rows(), nk.Contacts()
    rows.sparse = 1
    rows.type = wp.array([[ell] * n + [fdof]], dtype=int, device="cpu")
    rows.id = wp.array([list(range(n)) + [0]], dtype=int, device="cpu")
    z2i, z2f = wp.zeros((1, n + 1), dtype=int, device="cpu"), wp.zeros((1, n + 1), dtype=float, device="cpu")
    rows.J_rownnz, rows.J_rowadr, rows.D, rows.aref, rows.frictionloss, rows.force = z2i, z2i, z2f, z2f, z2f, z2f
    rows.J_colind = wp.zeros((1, 1, 1), dtype=int, device="cpu")
    rows.J = wp.zeros((1, 1, 1), dtype=float, device="cpu")
    con.dim = wp.array([3] * n, dtype=int, device="cpu")
    con.efc_address = wp.array([[c, 0, 0, 0, 0, 0] for c in range(n)], dtype=int, device="cpu")
    con.geom = wp.array(contacts, dtype=wp.vec2i, device="cpu")
    con.friction = wp.zeros(n, dtype=nk.mjt.vec5, device="cpu")
    con.geom_bodyid = wp.zeros(7, dtype=int, device="cpu")
    con.body_treeid = wp.zeros(1, dtype=int, device="cpu")
    con.geom_scope = wp.array([rob, rob, 0, surf, surf, pair, pair], dtype=int, device="cpu")
    con.dof_scope = wp.array([1], dtype=int, device="cpu")
    con.pairs = wp.array([(5, 6)], dtype=wp.vec2i, device="cpu")
    con.all_contacts = 0
    out = wp.zeros(n + 1, dtype=int, device="cpu")
    wp.launch(_keep_rows, dim=n + 1, inputs=[rows, con, out], device="cpu")
    assert out.numpy().tolist() == expect


@wp.kernel(enable_backward=False)
def _cone_vs_qcqp2(A: wp.array[nk.mjt.mat55], b: wp.array[nk.mjt.vec5], mu: nk.mjt.vec5, normal: float,
                   ours: wp.array[nk.mjt.vec5], theirs: wp.array[nk.mjt.vec5], active: wp.array[int]):
    i = wp.tid()
    f, a = _qcqp2(A[i], b[i], mu, normal)
    theirs[i] = f
    active[i] = wp.where(a, 1, 0)
    ours[i] = cone_force(A[i], b[i], mu, normal, 2)


def test_cone_force_stays_in_the_cone() -> None:
    """A friction block with one nearly uncontrolled direction (diag 1e4, 1e-10: the 1e-10 floor contact_force puts on
    a diagonal): MuJoCo Warp's _qcqp2 stops its Newton loop on the first step (delta < 1e-10 at la = 0) and returns the
    unconstrained solution, outside the cone and flagged inactive; un-clamped, that was a 1,570 m/s kick on T12
    (SB-7 100005:108). cone_force clamps any result outside the cone onto it; inside the cone it is unchanged."""
    A = np.zeros((2, 5, 5), np.float32)
    A[:, 0, 0], A[:, 1, 1] = 1e4, 1e-10
    b = np.zeros((2, 5), np.float32)
    b[0, 1] = 1e-6                          # unconstrained solution 1e4: far outside a unit cone
    b[1, 0] = 1e-1                          # unconstrained solution 1e-5: inside
    mu = nk.mjt.vec5(1.0, 1.0, 0.0, 0.0, 0.0)
    with wp.ScopedDevice("cpu"):
        ours, theirs = (wp.zeros(2, dtype=nk.mjt.vec5) for _ in range(2))
        active = wp.zeros(2, dtype=int)
        wp.launch(_cone_vs_qcqp2, dim=2, inputs=[wp.array(A, dtype=nk.mjt.mat55), wp.array(b, dtype=nk.mjt.vec5), mu,
                                                  1.0, ours, theirs, active])
        o, t, a = ours.numpy(), theirs.numpy(), active.numpy()
    assert a[0] == 0 and np.linalg.norm(t[0]) > 1e3              # the upstream QCQP leaves it outside
    assert np.linalg.norm(o[0]) == pytest.approx(1.0, rel=1e-5)   # clamped onto the cone
    assert np.array_equal(o[1], t[1]) and np.linalg.norm(o[1]) < 1.0
