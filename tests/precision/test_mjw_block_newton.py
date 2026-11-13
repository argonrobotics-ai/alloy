# SPDX-License-Identifier: Apache-2.0
"""Block-sparse Newton factorization (SCR-WP13-SPARSE-1): ``alloy.precision.mjw_block`` on the CPU.

Kernel level (fake model, data and solver context; rows synthesized from the Hessian's block pattern):
- recorded Hessians of settled 50- and 100-object piles (float32 condition 2e8-2e9): the block search matches the
  float64 solve in every world, its mean error within 3x MuJoCo Warp's own (a different elimination order: per world
  the two errors ranged 0.4-5x of each other), with search_dot and the Newton decrement; no overflow, no clamps;
- random SPD block systems with trees of 1..6 dofs (identity padding) and a different pattern per world;
- a singular block in one world: finite outputs, one clamp counted for that world only, the healthy world exact;
- the float64 redo solves in float64 (the error of a float32 store, not of a float32 factor) and counts its clamp;
- a cap overflow in one world: flagged, the dense path gets a mask that is False only there, the other world solved;
- a done world keeps its sentinels and its clamp count (WP5 / WP7 condition);
- a cone switched to pyramidal after ``put_model`` sends the call back to the dense path;
- the parallel graph build marks every tree pair of a row whose dofs are not grouped by tree;
- a solve on an unchanged tree graph keeps the symbolic lists (no ordering) and solves bit for bit as before; a world
  whose rows changed orders again; ``ALLOY_MJW_BLOCK_REUSE=0`` orders every solve;
- the buffers are keyed on the Data's ``solver_niter`` array, so a ``dataclasses.replace`` copy shares them.
Engine level (Newton + MuJoCo Warp): an eligible 20-box pile (elliptic cone) is marked, has its buffers from
``put_data`` (before any step or capture), runs without overflow or clamps and matches the dense path over the first
steps; ineligible scenes (small, a tree over 6 dofs with ALLOY_MJW_BLOCK_CHUNK=0, a pyramidal cone) stay dense, the
small one bitwise; the skip flags keep the dense path; a pile steps bit for bit with the symbolic reuse on and off,
and does reuse; the pin.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import warp as wp
from alloy._vendor import newton
from alloy._vendor.mujoco_warp._src.types import ConeType, SolverType

from alloy.precision import mjw_block, mjw_chol, mjw_pin
from alloy.precision import mjw_block_kernels as K

DATA = Path(__file__).with_name("data")
SENTINEL = 7.0


@dataclasses.dataclass
class Ctx:
    h: Any
    grad: Any
    search: Any
    search_dot: Any
    newton_decrement: Any
    done: Any


def _arr(a: Any, dtype: Any = float) -> wp.array:
    return wp.array(np.asarray(a), dtype=dtype, device="cpu")


def _sym(h: np.ndarray, nv: int) -> np.ndarray:
    u = np.triu(h[:nv, :nv]).astype(np.float64)
    return u + np.triu(u, 1).T


def _pairs_from_h(h: np.ndarray, adr: np.ndarray, num: np.ndarray) -> list[tuple[int, int]]:
    out = []
    for i in range(len(adr)):
        for j in range(i + 1, len(adr)):
            if np.any(h[adr[i]:adr[i] + num[i], adr[j]:adr[j] + num[j]] != 0):
                out.append((i, j))
    return out


def _rows(pairs: list[list[tuple[int, int]]], adr: np.ndarray) -> SimpleNamespace:
    """One two-entry efc row per tree pair: the first dof of each tree."""
    nw, r = len(pairs), max(max(len(p) for p in pairs), 1)
    nnz = 2 * r
    rownnz, rowadr = np.zeros((nw, r), np.int32), np.zeros((nw, r), np.int32)
    colind = np.zeros((nw, 1, nnz), np.int32)
    for w, plist in enumerate(pairs):
        for k, (i, j) in enumerate(plist):
            rownnz[w, k], rowadr[w, k] = 2, 2 * k
            colind[w, 0, 2 * k:2 * k + 2] = (adr[i], adr[j])
    nefc = np.array([len(p) for p in pairs], np.int32)
    efc = SimpleNamespace(J_rownnz=_arr(rownnz, int), J_rowadr=_arr(rowadr, int), J_colind=_arr(colind, int))
    return SimpleNamespace(efc=efc, nefc=_arr(nefc, int))


@dataclasses.dataclass
class Sys:
    m: Any
    d: Any
    ctx: Ctx
    clamps: wp.array
    masks: list


def _system(h, g, pairs, num, cfg=None, done=None, clamps0=None, nv=None) -> Sys:
    """Fake m / d / ctx for ``h`` [W, n, n] (upper triangle valid), ``g`` [W, n], trees of ``num`` dofs each."""
    nw, n = h.shape[0], h.shape[1]
    nv = n if nv is None else nv
    num = np.asarray(num, np.int32)
    adr = np.concatenate([[0], np.cumsum(num)[:-1]]).astype(np.int32)
    rows = _rows(pairs, adr)
    m = SimpleNamespace(dof_treeid=_arr(np.repeat(np.arange(len(num)), num), int), tree_dofadr=_arr(adr, int),
                        tree_dofnum=_arr(num, int), nv=nv, opt=SimpleNamespace(solver=SolverType.NEWTON,
                                                                               cone=ConeType.ELLIPTIC),
                        alloy_block_newton=cfg or mjw_block.config_for(len(num)))
    d = SimpleNamespace(nworld=nw, qpos=_arr(np.zeros((nw, 1))), solver_niter=_arr(np.zeros(nw, np.int32), int),
                        nefc=rows.nefc, efc=rows.efc)
    done = np.zeros(nw, bool) if done is None else np.asarray(done)
    ctx = Ctx(_arr(h), _arr(g), _arr(np.full((nw, nv), SENTINEL)), _arr(np.full(nw, SENTINEL)),
              _arr(np.full(nw, SENTINEL)), _arr(done, wp.bool))
    clamps = _arr(np.zeros(nw, np.int32) if clamps0 is None else clamps0, wp.int32)
    return Sys(m, d, ctx, clamps, [])


def _run(s: Sys) -> Sys:
    def dense(c: Any) -> None:
        s.masks.append(c.done.numpy().copy())

    assert mjw_block.factor_solve(s.m, s.d, s.ctx, s.clamps, dense)
    return s


def _want(h: np.ndarray, g: np.ndarray, nv: int) -> tuple[np.ndarray, float, float]:
    x = np.linalg.solve(_sym(h, nv), g[:nv].astype(np.float64))
    return -x, float(x @ x), float(g[:nv].astype(np.float64) @ x)


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, np.float64) - b) / np.linalg.norm(b))


@pytest.mark.parametrize("name", ["mjw_hslice50.npz", "mjw_hslice100.npz"])
def test_recorded_hessians_match_float64(name: str) -> None:
    z = np.load(DATA / name)
    nv, h, g = int(z["nv"]), z["h_10"], z["grad_10"]
    nt = nv // 6
    num, adr = np.full(nt, 6), 6 * np.arange(nt)
    s = _run(_system(h, g, [_pairs_from_h(h[w], adr, num) for w in range(h.shape[0])], num, done=z["done_10"], nv=nv))
    search, dot, dec = s.ctx.search.numpy(), s.ctx.search_dot.numpy(), s.ctx.newton_decrement.numpy()
    errs, stocks = [], []
    for w in range(h.shape[0]):
        want, sxx, sgx = _want(h[w], g[w], nv)
        errs.append(_rel(search[w], want))
        stocks.append(_rel(z["search_10"][w], want))
        assert errs[-1] <= 5e-3, f"world {w}: block {errs[-1]:.2e}, MuJoCo Warp {stocks[-1]:.2e}"
        assert dot[w] == pytest.approx(sxx, rel=5e-3) and dec[w] == pytest.approx(sgx, rel=5e-3)
    assert np.mean(errs) <= 3 * np.mean(stocks), f"block {errs}, MuJoCo Warp {stocks}"
    assert not s.clamps.numpy().any()
    assert not mjw_block.buffers_for(s.m, s.d).arrays["overflow"].numpy().any()


def _random_spd(rng, num: np.ndarray, pairs: list[tuple[int, int]], coupling: float = 0.5) -> np.ndarray:
    """Per-tree SPD blocks plus one rank-one coupling per pair (what J'DJ adds); upper triangle, NaN below."""
    n, adr = int(num.sum()), np.concatenate([[0], np.cumsum(num)[:-1]])
    h = np.zeros((n, n))
    for a, k in zip(adr, num):
        q = rng.standard_normal((k, k))
        h[a:a + k, a:a + k] = q @ q.T / k + np.eye(k)
    for i, j in pairs:
        v = np.zeros(n)
        v[adr[i]:adr[i] + num[i]] = rng.standard_normal(num[i])
        v[adr[j]:adr[j] + num[j]] = rng.standard_normal(num[j])
        h += coupling * np.outer(v, v)
    return np.where(np.triu(np.ones((n, n), bool)), h, np.nan).astype(np.float32)


def _random_pairs(rng, nt: int, p: float) -> list[tuple[int, int]]:
    return [(i, j) for i in range(nt) for j in range(i + 1, nt) if rng.random() < p]


def test_random_mixed_tree_sizes_match_float64() -> None:
    rng = np.random.default_rng(5)
    num = rng.integers(1, 7, size=24)
    pairs = [_random_pairs(rng, 24, p) for p in (0.05, 0.15, 0.4)]
    n = int(num.sum())
    h = np.stack([_random_spd(rng, num, p) for p in pairs])
    g = rng.standard_normal((3, n)).astype(np.float32)
    s = _run(_system(h, g, pairs, num))
    for w in range(3):
        want, sxx, sgx = _want(h[w], g[w], n)
        assert _rel(s.ctx.search.numpy()[w], want) <= 1e-4
        assert s.ctx.search_dot.numpy()[w] == pytest.approx(sxx, rel=1e-4)
        assert s.ctx.newton_decrement.numpy()[w] == pytest.approx(sgx, rel=1e-4)
    assert not s.clamps.numpy().any() and not mjw_block.buffers_for(s.m, s.d).arrays["overflow"].numpy().any()


def test_singular_block_is_clamped_in_its_world_only() -> None:
    rng = np.random.default_rng(6)
    num, nt = np.full(8, 6), 8
    pairs = [(i, i + 1) for i in range(nt - 1)] + [(0, 5), (2, 7)]
    n = 6 * nt
    h = np.stack([_random_spd(rng, num, pairs) for _ in range(2)])
    h[1, 12:18, :] = 0.0                                          # tree 2: rows ...
    h[1, :, 12:18] = 0.0                                          # ... and columns, so its pivots are 0
    g = rng.standard_normal((2, n)).astype(np.float32)
    s = _run(_system(h, g, [pairs, pairs], num))
    assert np.isfinite(s.ctx.search.numpy()).all() and np.isfinite(s.ctx.search_dot.numpy()).all()
    assert np.isfinite(s.ctx.newton_decrement.numpy()).all()
    assert s.clamps.numpy().tolist() == [0, 1]
    assert _rel(s.ctx.search.numpy()[0], _want(h[0], g[0], n)[0]) <= 1e-4


def test_float64_redo_runs_in_float64() -> None:
    rng = np.random.default_rng(10)
    num, nt = np.full(10, 6), 10
    pairs = [(i, i + 1) for i in range(nt - 1)] + [(0, 9), (3, 7)]
    h = np.stack([_random_spd(rng, num, pairs, coupling=200.0) for _ in range(2)])  # condition ~1e4
    g = rng.standard_normal((2, 60)).astype(np.float32)
    s = _run(_system(h, g, [pairs, pairs], num))
    f32 = [_rel(s.ctx.search.numpy()[w], _want(h[w], g[w], 60)[0]) for w in range(2)]
    b = mjw_block.buffers_for(s.m, s.d)
    b.arrays["bad"].assign(np.array([0, 1], np.int32))           # as if float32 had flagged world 1
    mjw_block.launch_factor(wp.float64, s.m, s.d, s.ctx, b, s.clamps)
    want, sxx, sgx = _want(h[1], g[1], 60)
    store = _rel(want.astype(np.float32), want)                   # the search is stored in float32
    assert f32[1] > 10 * store                                    # float32's factor error shows above that floor
    assert _rel(s.ctx.search.numpy()[1], want) <= 2 * store, (f32[1], store)
    assert s.ctx.search_dot.numpy()[1] == pytest.approx(sxx, rel=1e-6)
    assert s.clamps.numpy().tolist() == [0, 1]
    assert _rel(s.ctx.search.numpy()[0], _want(h[0], g[0], 60)[0]) == f32[0]  # world 0 untouched


def test_cap_overflow_goes_to_the_dense_path_for_that_world() -> None:
    rng = np.random.default_rng(7)
    nt, num = 8, np.full(8, 6)
    chain = [(i, i + 1) for i in range(nt - 1)]
    full = [(i, j) for i in range(nt) for j in range(i + 1, nt)]
    h = np.stack([_random_spd(rng, num, chain), _random_spd(rng, num, full)])
    g = rng.standard_normal((2, 6 * nt)).astype(np.float32)
    cfg = mjw_block.BlockConfig(nt, off_cap=10, pair_cap=1000)
    s = _run(_system(h, g, [chain, full], num, cfg=cfg))
    assert mjw_block.buffers_for(s.m, s.d).arrays["overflow"].numpy().tolist() == [0, 1]
    assert [m.tolist() for m in s.masks] == [[True, False]]
    assert _rel(s.ctx.search.numpy()[0], _want(h[0], g[0], 6 * nt)[0]) <= 1e-4
    assert (s.ctx.search.numpy()[1] == SENTINEL).all() and s.clamps.numpy().tolist() == [0, 0]


def test_done_world_keeps_sentinels_and_clamp_count() -> None:
    rng = np.random.default_rng(8)
    num, nt = np.full(6, 6), 6
    pairs = [(i, i + 1) for i in range(nt - 1)]
    h = np.stack([_random_spd(rng, num, pairs) for _ in range(2)])
    h[0, :6, :] = 0.0                                             # would clamp if it were factored
    h[0, :, :6] = 0.0
    g = rng.standard_normal((2, 36)).astype(np.float32)
    s = _run(_system(h, g, [pairs, pairs], num, done=[True, False], clamps0=[4, 0]))
    assert (s.ctx.search.numpy()[0] == SENTINEL).all()
    assert s.ctx.search_dot.numpy()[0] == SENTINEL and s.ctx.newton_decrement.numpy()[0] == SENTINEL
    assert s.clamps.numpy().tolist() == [4, 0]
    assert _rel(s.ctx.search.numpy()[1], _want(h[1], g[1], 36)[0]) <= 1e-4


def test_cone_switched_after_put_model_goes_dense() -> None:
    rng = np.random.default_rng(11)
    num = np.full(4, 6)
    s = _system(np.stack([_random_spd(rng, num, [(0, 1)])]), np.ones((1, 24), np.float32), [[(0, 1)]], num)
    s.m.opt.cone = ConeType.PYRAMIDAL                             # the incremental path would read hfactor back
    assert not mjw_block.factor_solve(s.m, s.d, s.ctx, s.clamps, lambda c: None)
    assert (s.ctx.search.numpy() == SENTINEL).all()


def test_graph_rows_mark_every_tree_pair_of_a_row() -> None:
    """A row whose dofs are not grouped by tree still marks each pair of its trees once, and no tree with itself."""
    trees = [3, 3, 5, 5, 3, 7]
    colind = np.array([[[6 * t + k for k, t in enumerate(trees)]]], np.int32)
    adj = wp.zeros((1, 8, 8), dtype=wp.int8, device="cpu")
    wp.launch(mjw_block.graph_rows, dim=(1, 1), device="cpu",
              inputs=[1, _arr([0], int), _arr([1], int), _arr([[len(trees)]], int), _arr([[0]], int),
                      _arr(colind, int), _arr(np.repeat(np.arange(8), 6), int)], outputs=[adj])
    got = {(a, b) for a, b in zip(*np.nonzero(adj.numpy()[0])) if a < b}
    assert got == {(3, 5), (3, 7), (5, 7)}


def _chain_system(rng, nt: int = 8) -> tuple[Sys, list[tuple[int, int]], np.ndarray]:
    num = np.full(nt, 6)
    chain = [(i, i + 1) for i in range(nt - 1)]
    h = np.stack([_random_spd(rng, num, chain) for _ in range(2)])
    return _run(_system(h, rng.standard_normal((2, 6 * nt)).astype(np.float32), [chain, chain], num)), chain, num


def _again(s: Sys) -> mjw_block.Buffers:
    b = mjw_block.buffers_for(s.m, s.d)
    b.arrays["deg"].fill_(-5)                                     # ordering scratch: only an ordering rewrites it
    s.ctx.search.fill_(SENTINEL)
    _run(s)
    return b


def test_an_unchanged_graph_keeps_the_symbolic_lists() -> None:
    """WP7's S1: a second solve on the same rows orders nothing and solves bit for bit as the first; a world whose
    rows changed orders again and solves its new system, the other keeps its lists."""
    rng = np.random.default_rng(11)
    s, chain, num = _chain_system(rng)
    first, n = s.ctx.search.numpy().copy(), int(num.sum())
    assert mjw_block.buffers_for(s.m, s.d).arrays["changed"].numpy().tolist() == [1, 1]
    b = _again(s)
    assert b.arrays["changed"].numpy().tolist() == [0, 0] and (b.arrays["deg"].numpy() == -5).all()
    assert np.array_equal(s.ctx.search.numpy(), first)
    other = chain + [(0, 7), (2, 5)]
    h1 = _random_spd(rng, num, other)
    rows = _rows([chain, other], 6 * np.arange(len(num)))
    s.d.nefc, s.d.efc = rows.nefc, rows.efc
    s.ctx.h = _arr(np.stack([s.ctx.h.numpy()[0], h1]))
    b = _again(s)
    assert b.arrays["changed"].numpy().tolist() == [0, 1]
    assert (b.arrays["deg"].numpy()[0] == -5).all() and not (b.arrays["deg"].numpy()[1] == -5).any()
    assert np.array_equal(s.ctx.search.numpy()[0], first[0])
    assert _rel(s.ctx.search.numpy()[1], _want(h1, s.ctx.grad.numpy()[1], n)[0]) <= 1e-4


def test_reuse_off_orders_every_solve(monkeypatch: pytest.MonkeyPatch) -> None:
    s, _, _ = _chain_system(np.random.default_rng(12))
    first = s.ctx.search.numpy().copy()
    monkeypatch.setenv("ALLOY_MJW_BLOCK_REUSE", "0")
    b = _again(s)
    assert b.arrays["changed"].numpy().tolist() == [1, 1] and not (b.arrays["deg"].numpy() == -5).any()
    assert np.array_equal(s.ctx.search.numpy(), first)


def test_buffers_are_keyed_on_solver_niter() -> None:
    rng = np.random.default_rng(9)
    num = np.full(4, 6)
    s = _system(np.stack([_random_spd(rng, num, [(0, 1)])]), np.ones((1, 24), np.float32), [[(0, 1)]], num)
    copy = SimpleNamespace(**vars(s.d))                           # like dataclasses.replace(d): same arrays
    first = mjw_block.buffers_for(s.m, s.d)
    assert first is not None and mjw_block.buffers_for(s.m, copy) is first
    assert mjw_block.buffers_for(s.m, SimpleNamespace(**{**vars(s.d), "solver_niter": _arr([0], int)})) is not first


def _stack(columns: int, height: int, worlds: int = 2) -> newton.Model:
    b = newton.ModelBuilder()
    for c in range(columns):
        for k in range(height):
            pos = wp.vec3(0.3 * c, 0.0, 0.05 + 0.1005 * k)
            b.add_shape_box(b.add_body(xform=wp.transform(pos, wp.quat_identity())), hx=0.05, hy=0.05, hz=0.05)
    s = newton.ModelBuilder()
    s.replicate(b, worlds)
    s.add_ground_plane()
    return s.finalize(device="cpu")


def _solver(model: newton.Model, cone: str = "elliptic") -> newton.solvers.SolverMuJoCo:
    assert mjw_chol.install()
    return newton.solvers.SolverMuJoCo(model, njmax=1500, nconmax=400, solver="newton", cone=cone)


def _advance(solver, model, n: int) -> np.ndarray:
    s0, s1, ctl = model.state(), model.state(), model.control()
    for _ in range(n):
        solver.step(s0, s1, ctl, None, 1e-3)
        s0, s1 = s1, s0
    return s0.body_q.numpy().copy()


def test_eligible_pile_is_marked_and_matches_dense(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _stack(5, 4)
    solver = _solver(model)
    assert solver.mjw_model.alloy_block_newton is not None, mjw_block.eligible(solver.mj_model)
    buf = mjw_block._BUFFERS.get(id(solver.mjw_data.solver_niter))
    assert buf is not None                                        # from put_data: a first step under capture finds it
    q3 = _advance(solver, model, 3)
    assert mjw_block.buffers_for(solver.mjw_model, solver.mjw_data) is buf
    assert not buf.arrays["overflow"].numpy().any()
    q20 = _advance(solver, model, 17)
    assert np.isfinite(q20).all() and not mjw_chol.clamp_counts(solver.mjw_data).numpy().any()
    monkeypatch.setenv("ALLOY_MJW_BLOCK", "0")
    dense_model = _stack(5, 4)
    dense = _solver(dense_model)
    assert getattr(dense.mjw_model, "alloy_block_newton", None) is None
    assert np.abs(_advance(dense, dense_model, 3) - q3).max() <= 1e-6


def _pile_steps(n: int = 20) -> tuple[np.ndarray, list[int]]:
    """A 20-box pile's body state after ``n`` steps, and per step how many worlds ran an ordering (the ordering
    scratch ``deg``, poisoned before the step, was rewritten)."""
    model = _stack(5, 4)
    solver = _solver(model)
    deg = mjw_block._BUFFERS[id(solver.mjw_data.solver_niter)].arrays["deg"]
    s0, s1, ctl = model.state(), model.state(), model.control()
    ordered = []
    for _ in range(n):
        deg.fill_(-5)
        solver.step(s0, s1, ctl, None, 1e-3)
        s0, s1 = s1, s0
        ordered.append(int((deg.numpy() != -5).any(axis=1).sum()))
    return np.concatenate([s0.body_q.numpy().ravel(), s0.body_qd.numpy().ravel()]), ordered


def test_pile_steps_bitwise_with_the_symbolic_reuse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reuse on: the lists of the first ordering serve the later steps (some step orders nothing), bit for bit as
    ordering every solve (``ALLOY_MJW_BLOCK_REUSE=0``)."""
    reuse, ordered = _pile_steps()
    monkeypatch.setenv("ALLOY_MJW_BLOCK_REUSE", "0")
    every, forced = _pile_steps()
    assert ordered[0] == 2 and 0 in ordered and set(forced) == {2}, (ordered, forced)
    assert np.array_equal(reuse, every)


def _steps_with_block(block: bool, n: int = 15) -> np.ndarray:
    mjw_block.install() if block else mjw_block.uninstall()
    try:
        model = _stack(1, 3)
        solver = _solver(model)
        assert getattr(solver.mjw_model, "alloy_block_newton", None) is None
        s0, s1, ctl = model.state(), model.state(), model.control()
        for _ in range(n):
            solver.step(s0, s1, ctl, None, 1e-3)
            s0, s1 = s1, s0
        return np.concatenate([s0.body_q.numpy().ravel(), s0.body_qd.numpy().ravel()])
    finally:
        mjw_block.install()


def test_small_scene_stays_dense_and_bitwise() -> None:
    assert np.array_equal(_steps_with_block(True), _steps_with_block(False))


def test_tree_over_six_dofs_is_not_eligible_with_chunking_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """``ALLOY_MJW_BLOCK_CHUNK=0`` keeps the rule before 0.1.2 (Gemini audit T2-009): a tree over 6 dofs makes the
    model dense. Default: it is split into runs (tests/precision/test_mjw_block.py)."""
    monkeypatch.setenv(mjw_block.CHUNK_ENV, "0")
    b = newton.ModelBuilder()
    for k in range(16):
        b.add_shape_box(b.add_body(xform=wp.transform(wp.vec3(0.3 * k, 1.0, 0.5), wp.quat_identity())),
                        hx=0.05, hy=0.05, hz=0.05)
    links, joints = [], []
    for i in range(7):
        link = b.add_link(xform=wp.transform(wp.vec3(0.1 * i, 0.0, 2.0), wp.quat_identity()), mass=1.0)
        b.add_shape_box(link, hx=0.04, hy=0.02, hz=0.02, cfg=newton.ModelBuilder.ShapeConfig(has_shape_collision=False))
        joints.append(b.add_joint_revolute(links[-1] if links else -1, link, axis=(0.0, 1.0, 0.0),
                                           parent_xform=wp.transform(wp.vec3(0.1 if links else 0.0, 0.0, 2.0),
                                                                     wp.quat_identity())))
        links.append(link)
    b.add_articulation(joints)
    solver = _solver(b.finalize(device="cpu"))
    assert solver.mj_model.nv >= mjw_block.NV_MIN
    assert "more than 6 dofs" in mjw_block.eligible(solver.mj_model)
    assert getattr(solver.mjw_model, "alloy_block_newton", None) is None


def test_pyramidal_cone_is_not_eligible() -> None:
    solver = _solver(_stack(5, 4), cone="pyramidal")              # its incremental solve reuses the dense hfactor
    assert "pyramidal" in mjw_block.eligible(solver.mj_model)
    assert getattr(solver.mjw_model, "alloy_block_newton", None) is None
    assert mjw_block._BUFFERS.get(id(solver.mjw_data.solver_niter)) is None


def test_skip_flags_keep_the_dense_path(monkeypatch: pytest.MonkeyPatch) -> None:
    solver = _solver(_stack(5, 4))
    m, d = solver.mjw_model, solver.mjw_data
    assert m.alloy_block_newton is not None
    calls = {"block": 0, "dense": 0}

    def block(*args: Any) -> bool:
        calls["block"] += 1
        return True

    monkeypatch.setattr(mjw_block, "factor_solve", block)
    monkeypatch.setitem(mjw_chol._STATE, "original", lambda *a: calls.__setitem__("dense", calls["dense"] + 1))
    monkeypatch.setattr(mjw_chol.wp, "launch", lambda *a, **k: None)
    ctx = SimpleNamespace(done=None, h=None, hfactor=None, grad=None, search=None, search_dot=None,
                          newton_decrement=None)
    mjw_chol._guarded(m, d, ctx, skip_unchanged=True)
    mjw_chol._guarded(m, d, ctx, skip_noflip=True)
    assert calls == {"block": 0, "dense": 2}
    mjw_chol._guarded(m, d, ctx)
    assert calls == {"block": 1, "dense": 2}


def test_block_sources_match_the_pin() -> None:
    assert mjw_pin.changed(mjw_pin.BLOCK) == [], "mujoco_warp's Newton solve changed: recheck mjw_block"
