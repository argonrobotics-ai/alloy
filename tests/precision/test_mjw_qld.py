# SPDX-License-Identifier: Apache-2.0
"""SCR-DC5-4 (ii): the ordered sparse L'DL factorization (``alloy.precision.mjw_qld``) on a branching chain (CPU).

- Its L'DL rebuilds M to float32 rounding, as MuJoCo Warp's own ``smooth._factor_i_sparse`` does, and its entries
  agree with the stock factor (joint armature keeps M well conditioned, as Newton's chain armature does; without it
  thin links make single L entries ill-determined while L'DL still rebuilds M).
- It gives the same bits when every level's update list is shuffled, which stands in for the racing launch order.
- The ordered fused solve (x = inv(L'DL) y) agrees with MuJoCo Warp's own and gives the same bits with shuffled
  levels.
- ``prepare_qld`` refuses a level that does not have the layout the ordering relies on.
"""

from __future__ import annotations

import mujoco
from alloy._vendor import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp
from alloy._vendor.mujoco_warp._src import smooth

from alloy.core.errors import PrecisionError
from alloy.precision import mjw_qld
from alloy.precision.mjw_launch import Launch

LINKS = 30


def _chain_xml() -> str:
    """A 30-link ball-joint chain (a cable) with a 6-link hinge branch at link 12: one tree of 102 dofs, above
    MuJoCo Warp's 64-dof dense-block limit, so it takes the sparse factor path."""
    def link(joint: str) -> str:
        return f'<body pos=".05 0 0">{joint}<geom type="capsule" fromto="0 0 0 .05 0 0" size=".01"/>'
    body = ""
    for n in range(LINKS):
        body += link('<joint type="ball"/>')
        if n == 12:
            body += "".join(link('<joint type="hinge" axis="1 0 0"/>') for _ in range(6)) + "</body>" * 6
    body += "</body>" * LINKS
    return f'<mujoco><default><joint armature="0.001"/></default><worldbody><body pos="0 0 1"><freejoint/><geom type="sphere" size=".02"/>{body}</body>' \
           f"</worldbody></mujoco>"


@pytest.fixture(scope="module")
def chain():
    mjm = mujoco.MjModel.from_xml_string(_chain_xml())
    mjd = mujoco.MjData(mjm)
    rng = np.random.default_rng(3)
    mjd.qpos[7:] = rng.uniform(-0.5, 0.5, mjm.nq - 7)
    mujoco.mj_normalizeQuat(mjm, mjd.qpos)
    mujoco.mj_forward(mjm, mjd)
    with wp.ScopedDevice("cpu"):
        m = mjw.put_model(mjm)
        d = mjw.put_data(mjm, mjd, nworld=2)
        mjw.forward(m, d)
    assert len(m.qLD_updates), "the chain must take the sparse factor path"
    yield m, d, mjm, mjd
    mjw_qld.release()


def _stock(m, d) -> np.ndarray:
    L = wp.zeros_like(d.qLD)
    D = wp.zeros_like(d.qLDiagInv)
    smooth._factor_i_sparse(m, d, d.M, L[:, m.qLD_block_total:], D)
    return L.numpy()[:, m.qLD_block_total:]


def _ordered(m, d) -> np.ndarray:
    L = wp.zeros_like(d.qLD)
    region = L[:, m.qLD_block_total:]
    wp.copy(region, d.M)
    mjw_qld.prepare_qld(m)
    for level in reversed(m.qLD_updates):
        mjw_qld.qld_handler((), Launch(dim=(2, level.size), inputs=[m.M_rownnz, m.M_rowadr, level, region],
                                       outputs=[region], block_dim=None))
    return L.numpy()[:, m.qLD_block_total:]


def _rebuild_error(mjm, mjd, x: np.ndarray) -> float:
    """max |L'DL - M| / max |M| for one world's factor in MuJoCo's CSR layout (diagonal = D)."""
    nv, adr, nnz, col = mjm.nv, mjm.M_rowadr, mjm.M_rownnz, mjm.M_colind
    full, L, D = np.zeros((nv, nv)), np.eye(nv), np.zeros(nv)
    for i in range(nv):
        for t in range(nnz[i]):
            j, a = col[adr[i] + t], adr[i] + t
            full[i, j] = full[j, i] = mjd.M[a]
            if j == i:
                D[i] = x[a]
            else:
                L[i, j] = x[a]
    return float(np.abs(L.T @ np.diag(D) @ L - full).max() / np.abs(full).max())


def test_ordered_factors_m(chain) -> None:
    m, d, mjm, mjd = chain
    stock, ordered = _stock(m, d), _ordered(m, d)
    err_stock, err_ordered = _rebuild_error(mjm, mjd, stock[0]), _rebuild_error(mjm, mjd, ordered[0])
    assert err_ordered <= max(2.0 * err_stock, 1.0e-6), (err_ordered, err_stock)
    np.testing.assert_allclose(ordered, stock, rtol=1e-3, atol=1e-6)


def test_shuffled_levels_same_bits(chain) -> None:
    m, d, _, _ = chain
    ref = _ordered(m, d)
    rng = np.random.default_rng(11)
    shuffled = []
    for level in m.qLD_updates:
        upd = level.numpy().reshape(-1, 3).copy()
        rng.shuffle(upd)
        shuffled.append(wp.array(upd, dtype=wp.vec3i, device="cpu"))
    m.qLD_updates, keep = tuple(shuffled), m.qLD_updates
    try:
        assert np.array_equal(_ordered(m, d), ref)
    finally:
        m.qLD_updates = keep


def _solve(m, d, ordered: bool) -> np.ndarray:
    """x = inv(L'DL) y for a fixed y with the stock factor of world 0/1 (``d.qLD``), stock or ordered kernel."""
    y = wp.array(np.random.default_rng(5).normal(size=(2, m.nv)).astype(np.float32), device="cpu")
    x = wp.zeros_like(y)
    L = d.qLD[:, m.qLD_block_total:]
    if not ordered:
        smooth._solve_LD_sparse(m, d, L, d.qLDiagInv, x, y)
        return x.numpy()
    mjw_qld.release()
    mjw_qld.prepare_qld(m)
    mjw_qld.solve_handler((m.nv, len(m.qLD_updates)), Launch(
        dim=(2, 32), inputs=[m.qLD_block_adr, L, d.qLDiagInv, m.qLD_all_updates, m.qLD_level_offsets, y],
        outputs=[x], block_dim=32))
    return x.numpy()


def test_ordered_solve_matches_stock(chain) -> None:
    m, d, _, _ = chain
    np.testing.assert_allclose(_solve(m, d, True), _solve(m, d, False), rtol=1e-3, atol=1e-4)


def test_ordered_solve_shuffled_levels_same_bits(chain) -> None:
    m, d, _, _ = chain
    ref = _solve(m, d, True)
    rng = np.random.default_rng(13)
    keep = m.qLD_updates
    shuffled = []
    for level in keep:
        upd = level.numpy().reshape(-1, 3).copy()
        rng.shuffle(upd)
        shuffled.append(wp.array(upd, dtype=wp.vec3i, device="cpu"))
    m.qLD_updates = tuple(shuffled)
    try:
        assert np.array_equal(_solve(m, d, True), ref)
    finally:
        m.qLD_updates = keep


def test_refuses_a_level_with_repeated_k() -> None:
    bad = np.array([[0, 5, 10], [1, 5, 11]], dtype=np.int32)  # k = 5 twice in one level
    with pytest.raises(PrecisionError, match="not grouped by target depth"):
        mjw_qld._check_level(bad, 0)
