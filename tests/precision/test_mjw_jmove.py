# SPDX-License-Identifier: Apache-2.0
"""The route's dense-J row permutation runs in place (coordinator ruling 2026-11-04 05:40Z): no second J, and the
result equals the gather it replaces bit for bit (CPU).

- random per-world permutations of the live rows (fixed points, 2-cycles, one long cycle), random float J: equal to
  ``J[w, perm[w, r]]`` for live rows; rows past ``nefc`` untouched;
- a routed dense-J SolverMuJoCo keeps no J copy in its row-sort scratch (sparse J on the route keeps none either).
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.precision import mjw_rows
from alloy.precision.mjw_jmove import _cycle_leaders, _dense_permute


def _perm(rng, n: int, kind: str) -> np.ndarray:
    if kind == "identity":
        return np.arange(n)
    if kind == "long":
        return np.roll(np.arange(n), 1)                 # one cycle through every row
    return rng.permutation(n)


@pytest.mark.parametrize("kind", ["random", "identity", "long"])
def test_in_place_permute_equals_the_gather(kind: str) -> None:
    rng = np.random.default_rng(5)
    nw, nj, nv = 6, 40, 13
    nefc = np.array([0, 1, 7, 40, 23, 39], dtype=np.int32)
    J0 = rng.standard_normal((nw, nj, nv)).astype(np.float32)
    perm = np.tile(np.arange(nj), (nw, 1))
    for w in range(nw):
        perm[w, :nefc[w]] = _perm(rng, int(nefc[w]), kind)
    want = J0.copy()
    for w in range(nw):
        want[w, :nefc[w]] = J0[w, perm[w, :nefc[w]]]
    J = wp.array(J0, dtype=float, device="cpu")
    lead = wp.zeros((nw, nj), dtype=wp.int8, device="cpu")
    perm_w = wp.array(perm, dtype=int, device="cpu")
    nefc_w = wp.array(nefc, dtype=int, device="cpu")
    wp.launch(_cycle_leaders, dim=nw, inputs=[nefc_w, nj, perm_w], outputs=[lead], device="cpu")
    wp.launch(_dense_permute, dim=(nw, nv), inputs=[nefc_w, nj, perm_w, lead], outputs=[J], device="cpu")
    assert np.array_equal(J.numpy(), want)                 # bitwise, and rows past nefc untouched


@pytest.mark.parametrize("jacobian", ["dense", "sparse"])
def test_routed_row_sort_keeps_no_second_j(jacobian: str) -> None:
    from alloy._vendor import newton

    b = newton.ModelBuilder()
    for i in range(3):
        body = b.add_body(xform=wp.transform(wp.vec3(0.03 * i, 0.0, 0.05 + 0.099 * i), wp.quat_identity()))
        b.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    s = newton.ModelBuilder()
    s.replicate(b, 2)
    s.add_ground_plane()
    model = s.finalize(device="cpu")
    solver = newton.solvers.SolverMuJoCo(model, njmax=100, nconmax=32, solver="newton", jacobian=jacobian)
    scratch = mjw_rows._scratch(solver.mjw_data)
    assert scratch.j_copy is None and scratch.c_copy is None
    assert (scratch.lead is not None) == (jacobian == "dense")
