# SPDX-License-Identifier: Apache-2.0
"""Block-sparse Newton for trees over 6 dofs (Gemini audit T2-009, 0.1.2; ``alloy.precision.mjw_block.units``), on the
CPU. tests/precision/test_mjw_block_newton.py keeps SPARSE-1's laws (trees of at most 6 dofs, bit for bit).

Until 0.1.2 one tree over 6 dofs (a 7-dof arm, the Panda's 9, a CHAIN cable's 15-21) sent the whole entry to MuJoCo
Warp's dense nv^3 factor: one 6-link cable beside 200 boxes and a 1-dof arm stepped 42-77x slower on an L4. Now such a
tree is split into runs of at most 6 dofs, every two runs of one tree tied in the graph.
- ``units``: the runs cover each tree in order; a model whose trees all fit gets None (its own arrays, bit for bit).
- Kernel level: trees of 1-15 dofs, dense inside, coupled across: the block search matches a float64 solve; without
  the ties (the control) it does not.
- The chain-grip pivot floor on the block path: WP9's R4 rig Hessian (indefinite by float32 rounding) as runs of one
  tree: the float64 redo at mjMINVAL is not a descent step, at the floor it is (as ``mjw_chol_f64.factor64_floor``).
- Engine level: a 7-hinge arm driven past its limits beside 16 boxes, the shipped Panda beside 16 boxes and a clamped
  CHAIN beside a 1-dof arm and 16 boxes (pivot floor marked) take the block path and step like the dense path within
  float32 tolerance, in no more Newton factorizations (an untied factor converges too, in more: WP13's review);
  ``ALLOY_MJW_BLOCK_CHUNK=0`` keeps them dense.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import warp as wp
from alloy._vendor.mujoco_warp._src import solver as mjw_solver
from alloy._vendor.mujoco_warp._src.types import ConeType, SolverType

from alloy.precision import mjw_block, mjw_chol

R4 = Path(__file__).parents[1] / "contact" / "chaingrip_r4_hessian.npz"


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


def _mjm(num: list[int]) -> SimpleNamespace:
    num_ = np.asarray(num, np.int32)
    adr = np.concatenate([[0], np.cumsum(num_)[:-1]]).astype(np.int32)
    return SimpleNamespace(ntree=len(num_), tree_dofadr=adr, tree_dofnum=num_, nv=int(num_.sum()))


def _system(h: np.ndarray, g: np.ndarray, num: list[int], pairs: list[list[tuple[int, int]]]):
    """Fake m / d / ctx: trees of ``num`` dofs (runs from ``mjw_block.units``), one efc row per tree pair of each
    world over every dof of both trees (as a contact row lists every dof it moves)."""
    mjm = _mjm(num)
    adr, nw = mjm.tree_dofadr, h.shape[0]
    u = mjw_block.units(mjm)
    m = SimpleNamespace(dof_treeid=_arr(np.repeat(np.arange(len(num)), num), int), tree_dofadr=_arr(adr, int),
                        tree_dofnum=_arr(mjm.tree_dofnum, int), nv=mjm.nv,
                        opt=SimpleNamespace(solver=SolverType.NEWTON, cone=ConeType.ELLIPTIC),
                        alloy_block_newton=mjw_block.config_for(mjw_block.n_blocks(mjm)))
    if u is not None:
        m.alloy_block_units = mjw_block.Units(*(_arr(a, int) for a in u))
    r = max(max((len(p) for p in pairs), default=0), 1)
    rownnz, rowadr = np.zeros((nw, r), np.int32), np.zeros((nw, r), np.int32)
    colind = np.zeros((nw, 1, 2 * r * mjm.nv), np.int32)
    for w, plist in enumerate(pairs):
        at = 0
        for k, (i, j) in enumerate(plist):
            dofs = [*range(adr[i], adr[i] + num[i]), *range(adr[j], adr[j] + num[j])]
            rownnz[w, k], rowadr[w, k] = len(dofs), at
            colind[w, 0, at:at + len(dofs)] = dofs
            at += len(dofs)
    efc = SimpleNamespace(J_rownnz=_arr(rownnz, int), J_rowadr=_arr(rowadr, int), J_colind=_arr(colind, int))
    d = SimpleNamespace(nworld=nw, qpos=_arr(np.zeros((nw, 1))), solver_niter=_arr(np.zeros(nw, np.int32), int),
                        nefc=_arr([len(p) for p in pairs], int), efc=efc)
    n = h.shape[1]
    ctx = Ctx(_arr(h), _arr(g), _arr(np.full((nw, n), 7.0)), _arr(np.full(nw, 7.0)), _arr(np.full(nw, 7.0)),
              _arr(np.zeros(nw, bool), wp.bool))
    return m, d, ctx, _arr(np.zeros(nw, np.int32), wp.int32)


def _solve(h, g, num, pairs, floor: float = 0.0):
    m, d, ctx, clamps = _system(h, g, num, pairs)
    assert mjw_block.factor_solve(m, d, ctx, clamps, lambda c: None, floor)
    return m, d, ctx, clamps


def _full(h: np.ndarray) -> np.ndarray:
    u = np.triu(h).astype(np.float64)
    return u + np.triu(u, 1).T


def _spd(rng, num: list[int], pairs: list[tuple[int, int]]) -> np.ndarray:
    """Symmetric, diagonally dominant: dense inside every tree, and on the blocks of ``pairs``."""
    mjm = _mjm(num)
    tree = np.repeat(np.arange(len(num)), num)
    allowed = tree[:, None] == tree[None, :]
    for i, j in pairs:
        allowed |= (tree[:, None] == i) & (tree[None, :] == j) | (tree[:, None] == j) & (tree[None, :] == i)
    s = np.where(allowed, rng.standard_normal((mjm.nv, mjm.nv)), 0.0)
    s = 0.5 * (s + s.T)
    return (s + np.diag(np.abs(s).sum(1) + 1.0)).astype(np.float32)


def test_units_cover_each_tree_in_runs_of_at_most_six() -> None:
    dofadr, dofnum, tree, dof_unit = mjw_block.units(_mjm([6, 9, 1, 15]))
    assert dofadr.tolist() == [0, 6, 12, 15, 16, 22, 28] and dofnum.tolist() == [6, 6, 3, 1, 6, 6, 3]
    assert tree.tolist() == [0, 1, 1, 2, 3, 3, 3]
    assert dof_unit.tolist() == np.repeat(np.arange(7), dofnum).tolist()
    assert mjw_block.units(_mjm([6, 1, 3])) is None and mjw_block.n_blocks(_mjm([6, 1, 3])) == 3
    assert mjw_block.n_blocks(_mjm([6, 9, 1, 15])) == 7


@pytest.mark.parametrize("ties", [True, False], ids=["tied", "control_untied"])
def test_split_trees_solve_like_float64(ties: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """Trees of 9, 15, 6, 1 and 4 dofs, dense inside, a different coupling per world: the block search is the float64
    solve. A row lists every dof it moves, so a touched tree's runs meet in the graph anyway; the 15-dof tree touches
    nothing (an arm in free air), and without the ties (the control) its runs lose their coupling."""
    if not ties:
        blocks = mjw_block._blocks
        monkeypatch.setattr(mjw_block, "_blocks", lambda m: (*blocks(m)[:4], 0))
    rng = np.random.default_rng(3)
    num = [9, 15, 6, 1, 4]
    pairs = [[(0, 2), (3, 4)], [(0, 4), (2, 3)]]                 # the 15-dof tree touches nothing: only ties
    h = np.stack([_spd(rng, num, p) for p in pairs])
    g = rng.standard_normal((2, sum(num))).astype(np.float32)
    _, _, ctx, clamps = _solve(h, g, num, pairs)
    for w in range(2):
        want = -np.linalg.solve(_full(h[w]), g[w].astype(np.float64))
        err = np.linalg.norm(ctx.search.numpy()[w] - want) / np.linalg.norm(want)
        assert (err <= 1e-5) if ties else (err > 1e-2), (w, err)
    assert not clamps.numpy().any()


def test_the_pivot_floor_holds_on_the_block_path() -> None:
    """WP9's R4 rig Hessian (a CHAIN gripped at impratio 300, float32 assembled, indefinite by rounding) as one
    60-dof tree in ten tied runs. The float64 redo at mjMINVAL gives no descent step; at the chain-grip floor
    (10 x nv * eps32 * max diag H) it gives a finite one, as the dense redo does."""
    z = np.load(R4)
    h, g = z["h"][None].astype(np.float32), z["grad"][None].astype(np.float32)
    out = {}
    for floor in (0.0, 10.0):
        m, d, ctx, clamps = _system(h, g, [60], [[]])
        assert mjw_block.factor_solve(m, d, ctx, clamps, lambda c: None, floor)
        b = mjw_block.buffers_for(m, d)
        b.arrays["bad"].assign(np.array([1], np.int32))           # as if float32 had flagged it
        mjw_block.launch_factor(wp.float64, m, d, ctx, b, clamps, floor)
        out[floor] = (ctx.search.numpy()[0].astype(np.float64), float(ctx.newton_decrement.numpy()[0]))
    x0, dec0 = out[0.0]
    assert not (np.isfinite(x0).all() and np.isfinite(dec0) and abs(dec0) < 1e12), dec0
    x, dec = out[10.0]
    assert np.isfinite(x).all() and dec > 0.0                       # -search is a descent direction
    assert float(g[0].astype(np.float64) @ -x) == pytest.approx(dec, rel=1e-4)


# --- engine level ---------------------------------------------------------------------------------------------------

ARM1 = """<mujoco><compiler angle="radian"/><worldbody><body name="b1" pos="0 0 0.05"><joint name="j1" type="hinge"
  axis="0 0 1" range="-1.5 1.5" damping="5"/><geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.015" mass="0.5"/>
  </body></worldbody><actuator><position joint="j1" kp="100" kv="10"/></actuator></mujoco>"""
_LINKS = "".join(f'<body name="l{i}" pos="0.05 0 0"><joint name="h{i}" type="hinge" axis="0 0 1" range="-1 1" '
                 f'damping="1"/><geom type="capsule" fromto="0 0 0 0.05 0 0" size="0.01" mass="0.1"/>'
                 for i in range(7))
_SERVOS = "".join(f'<position joint="h{i}" kp="5" kv="0.5"/>' for i in range(7))
ARM7 = (f'<mujoco><compiler angle="radian"/><worldbody><body name="base" pos="0 0 0.05">{_LINKS}{"</body>" * 7}'
        f'</body></worldbody><actuator>{_SERVOS}</actuator></mujoco>')


def _scene(kind: str):
    import alloy
    from alloy.cards.library import default_root

    s = alloy.Scene(worlds=2, dt=0.01)
    if kind == "panda":
        s.add_robot("panda", default_root() / "robots" / "emb_e2_panda" / "scene.xml")
    else:
        s.add_ground()
        s.add_robot("arm", mjcf=ARM7 if kind == "arm7" else ARM1, pos=(-0.45, 0, 0))
    if kind == "chain":
        s.add_rod("cable", length=0.2, radius=0.004, pos=(-0.45, 0.25, 0.02), direction=(1, 0, 0), segments=6,
                  clamp="start", route="chain")
    x0 = 1.0 if kind == "panda" else 0.3                     # the Panda's file has its table near the origin
    for i in range(16):
        r, c = divmod(i, 4)
        s.add_box(f"box{i}", size=(0.025,) * 3, pos=(x0 + 0.04 * c, -0.3 + 0.04 * r, 0.0125), mass=0.05)
    return s


def _steps(kind: str, chunk: bool, monkeypatch: pytest.MonkeyPatch) -> tuple[np.ndarray, Any]:
    import alloy

    monkeypatch.setenv("ALLOY_CPU_ROUTE", "warp")
    monkeypatch.setenv(mjw_block.CHUNK_ENV, "1" if chunk else "0")
    calls, solve = [0], mjw_block.factor_solve
    monkeypatch.setattr(mjw_block, "factor_solve", lambda *a, **k: calls.__setitem__(0, calls[0] + 1) or solve(*a, **k))
    sim = alloy.build(_scene(kind), device="cpu", seed=0)
    factors, guarded = [0], mjw_solver._cholesky_factorize_solve                # installed by the build (mjw_chol)
    monkeypatch.setattr(mjw_solver, "_cholesky_factorize_solve",
                        lambda *a, **k: factors.__setitem__(0, factors[0] + 1) or guarded(*a, **k))
    try:
        sv = next(s for s in sim.engine.solvers if getattr(s, "mjw_model", None) is not None)
        m = sv.mjw_model
        info = SimpleNamespace(block=getattr(m, "alloy_block_newton", None) is not None,
                               units=getattr(m, mjw_block.UNITS_ATTR, None) is not None,
                               floor=mjw_chol.pivot_floor(m, sv.mjw_data), nv=int(sv.mj_model.nv))
        if kind == "arm7":                                     # past its limits: one-dof rows on both runs of the arm
            sim.set_targets({f"arm/h{i}": 1.5 * (-1) ** i for i in range(7)})
        sim.step(5)
        bodies = sim.names()["bodies"]
        st = sim.state(bodies)
        info.clamps = int(mjw_chol.clamp_counts(sv.mjw_data).numpy().sum())
        buf = mjw_block.buffers_for(m, sv.mjw_data)
        info.calls, info.factors = calls[0], factors[0]
        info.overflow = None if buf is None else int(buf.arrays["overflow"].numpy().sum())
        info.ok = sim.check()["ok"]
        return np.concatenate([st[f"{b}.pos"].ravel() for b in bodies]), info
    finally:
        sim.close()


@pytest.mark.compile_heavy
@pytest.mark.timeout(2400)
@pytest.mark.parametrize("kind", ["arm7", "panda", "chain"])
def test_a_tree_over_six_dofs_takes_the_block_path_and_steps_like_dense(kind: str,
                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    block, bi = _steps(kind, True, monkeypatch)
    dense, di = _steps(kind, False, monkeypatch)
    print(f"{kind}: nv {bi.nv}, block {bi.block} (units {bi.units}), dense model block {di.block}, "
          f"floor {bi.floor:g}, max |dpos| {np.abs(block - dense).max():.3g} m, clamps {bi.clamps}/{di.clamps}, "
          f"block solves {bi.calls}/{di.calls}, Newton factorizations {bi.factors}/{di.factors}, "
          f"overflow {bi.overflow}")
    assert bi.nv >= mjw_block.NV_MIN and bi.block and bi.units and not di.block
    assert (bi.floor > 0.0) == (kind == "chain") and bi.floor == di.floor
    assert bi.ok and di.ok and bi.clamps == 0
    assert bi.calls > 0 and bi.overflow == 0 and di.calls == 0          # the block path ran, every world in it
    assert np.abs(block - dense).max() <= 1e-5
    # an inexact factor still converges, in more Newton iterations (WP13's review: untied runs took 6x on the chain)
    assert bi.factors <= 1.1 * di.factors + 2, (bi.factors, di.factors)
