# SPDX-License-Identifier: Apache-2.0
"""Tripwire for a mujoco_warp pin bump (P3-1 round 5): the source the ordered route was proven against.

``mjw_route`` copies kernel bodies from, reads launch arguments of, or calls safe the functions below. A hash of
each one's source at mujoco_warp@f34c6dd5 (v3.17.0; P3-1 rerun passed 2026-11-09) is kept here. ``install`` refuses the route (SolverMuJoCo stays NONE
with this reason) when any differs, and ``tests/precision/test_mjw_route.py::test_pin_sources`` fails, so a pin bump
reruns P3-1 (``tests/precision/test_mjw_route_gpu.py``) before the route comes back. Regenerate with
``python -m alloy.precision.mjw_pin`` after that rerun passes.
"""

import hashlib
import importlib
import inspect
import sys
from typing import Any

from alloy._vendor._warp_calls import upstream

PIN = "f34c6dd52813852d0f1a2ab6d1f4d21ef1f0829f"
_PKG = "alloy._vendor.mujoco_warp._src"
# module.name -> sha256 prefix of inspect.getsource at PIN
SOURCES: dict[str, str] = {
    "constraint._efc_contact_jac_sparse": "e546a54d4942c7b3",
    "constraint.make_constraint": "547c7abfcc71eea2",
    "derivative._qderiv_actuator_passive_actuation_sparse": "aaecb986f773fcc0",
    "forward._qfrc_actuator": "3be540e876f2e604",
    "passive._flex_bending": "129e351bf6fc499b",
    "passive._flex_elasticity": "eb2b3668815b160b",
    "passive._flex_passive_bend_interp": "0fc0da95638437f0",
    "passive._gravity_force": "ef7ad2c13d36240a",
    "passive._flex_rod_bending": "a517ff9963b1803f",
    "passive.passive": "908ad56584892abc",
    "passive._spring_damper_flexedge_passive": "3e6d941b6ae82840",
    "sensor._sensor_touch": "1f7c546cf6ef3a15",
    "sensor._tendon_actuator_force": "2a07ddc2f1064afa",
    "set_const._accumulate_subtreemass": "a964a941492bc5d9",
    "set_const.set_const_0": "abdb921f9856feb3",
    "set_const.set_const_fixed": "192d29600a90aa1b",
    "smooth._cfrc_backward": "1bedd43719cef7d2",
    "smooth._crb_accumulate": "b8840a43269f4f8c",
    "smooth._factor_i_sparse": "dcc45c595993ee76",
    "smooth._qLD_acc": "eaaf6465f76159f4",
    "smooth._rne_cfrc_backward": "feb2bc201b9b1c08",
    "smooth._solve_LD_sparse": "3e14e7bb0eff78fe",
    "smooth._solve_LD_sparse_copy_fn": "c93ae219e56edd94",
    "smooth._solve_LD_sparse_diag_fn": "c4b8c3ece19b1f24",
    "smooth._solve_LD_sparse_fused": "76ca1e40a60cabf1",
    "smooth._subtree_com_acc": "8400d9f255c45b90",
    "smooth._tendon_armature": "cfcc1392b15151cd",
    "smooth._tendon_bias_coef": "c81192c77509c4df",
    "smooth._tendon_bias_qfrc": "f1bd88666d3ef208",
    "smooth._transmission": "dd7f15b72dacb4c9",
    "smooth.com_pos": "15a9a1257f702f46",
    "smooth.crb": "4c20935302237195",
    "solver._JTDACJ_sparse": "27e3d830caede482",
    "solver._create_solver_context": "3236e32098027afb",
    "solver._elliptic_hessian_entry_from_projections": "ee6122f35e61d9ff",
    "solver._jtdaj_groups_per_world": "f02745037ef21486",
    "solver._linesearch_jv_fused_kernel": "85b390490a7564bb",
    "solver._solve_beta_finalize_tiled": "b999cdea3d55f223",
    "solver._solve_done": "f691f485dcca90ff",
    "solver._solve_init_jaref_kernel": "f47f3a9781e6ed4a",
    "solver._solve_init_search_cg_tiled": "c13b63a46e095ff1",
    "solver._update_constraint": "bb3241b42b78170d",
    "solver._update_constraint_efc": "312cecd678d2b7c8",
    "solver._update_constraint_init_qfrc_constraint_sparse": "8b083befa6ec69dd",
    "solver._update_gradient": "b14bd774cda8511c",
    "solver._update_gradient_grad": "ddc637ad5ebba5c5",
    "solver._update_gradient_grad_tiled": "0e5d49b2a037fb27",
    "solver.solve": "912386ccde41ccd7",
    "support.jac_dof": "8f48a613029a60fb",
    # mjw_chol (BUL-0016): the Newton Cholesky launcher it replaces and the kernels it copies or runs after
    "block_cholesky.solve_search_sums": "751d6b4ad1ff4f06",
    "solver._cholesky_factorize_solve": "5f04c3acf13da87e",
    "solver._update_gradient_cholesky": "85293d80abe9d74e",
    "solver._update_gradient_cholesky_blocked": "3af62edc0c41073d",
    "solver._update_gradient_cholesky_blocked_skip_unchanged": "a733e7cea08dcb7b",
    "solver._solver_iteration": "d7062671c414eb9c",
    "solver._update_gradient_zero_grad_dot": "501b70a04bd1dddf",
    "solver._use_incremental": "27435722c48c3f75",
    # mjw_block (SCR-WP13-SPARSE-1): where solver_niter is reset, the dense H's upper-triangle assembly, the solve
    "solver._solve_init_efc": "d090f518a298273a",
    "solver._update_gradient_init_h_sparse": "db6656e1a087e79e",
    "solver._solve": "ccdbc0e66f703d1c",
    "io.is_sparse": "f166c71957a67938",
}
# The pivot-clamp guard's sources (mjw_chol), checked apart from the route's: a change disables the guard only.
GUARD = ("block_cholesky.solve_search_sums", "solver._cholesky_factorize_solve", "solver._update_gradient_cholesky",
         "solver._update_gradient_cholesky_blocked", "solver._update_gradient_cholesky_blocked_skip_unchanged")


# The lazy last factorization's sources (mjw_lazy_factor, SCR-WP0-RIGIDDT-2 L1): a change disables that patch only.
LAZY = ("solver._solve_done", "solver._solver_iteration", "solver._update_gradient", "solver._update_gradient_grad",
        "solver._update_gradient_zero_grad_dot", "solver._use_incremental")


# The block-sparse Newton factorization's sources (mjw_block, SCR-WP13-SPARSE-1): the launcher it dispatches inside,
# what the outputs mean, the ctx fields it reads, where solver_niter is reset (0) and raised (after the gradient: its
# symbolic step runs while it is 0), the dense H it gathers (upper triangle), the sparse-Jacobian rule, and where the
# incremental path (which reads the dense hfactor back) is taken: never for an elliptic cone. A change turns it off.
BLOCK = ("block_cholesky.solve_search_sums", "io.is_sparse", "solver._JTDACJ_sparse",
         "solver._cholesky_factorize_solve", "solver._create_solver_context", "solver._solve", "solver._solve_done",
         "solver._solve_init_efc", "solver._solver_iteration", "solver._update_gradient",
         "solver._update_gradient_init_h_sparse", "solver._use_incremental")


def tracked() -> tuple[str, ...]:
    """Every function the route depends on: substituted factories, scoped stages, SAFE factories, launch sites."""
    from alloy.precision import mjw_route

    names = {f"{m.rsplit('.', 1)[1]}.{f}" for m, f in mjw_route.SUBSTITUTIONS}
    names |= {f"{m.rsplit('.', 1)[1]}.{f}" for m, f in mjw_route.SCOPES}
    names |= set(mjw_route.SAFE)
    names |= {"solver._update_gradient", "solver._update_constraint", "solver._elliptic_hessian_entry_from_projections",
              "solver._jtdaj_groups_per_world", "solver._create_solver_context", "solver._update_gradient_grad_tiled",
              # CG path (SCR-DC5-4 (ii)): its unmarked kernels, audited at PIN (tile reductions, integer counters)
              "solver._solve_init_search_cg_tiled", "solver._solve_beta_finalize_tiled", "solver._solve_done",
              # the ordered sparse solve calls these two MuJoCo Warp funcs
              "smooth._solve_LD_sparse_copy_fn", "smooth._solve_LD_sparse_diag_fn",
              # the ordered gravcomp kernel calls this MuJoCo Warp func (mjw_gravcomp)
              "support.jac_dof"}
    return tuple(sorted(names))


def _resolve(name: str) -> Any:
    mod, attr = name.split(".")
    return inspect.unwrap(getattr(importlib.import_module(f"{_PKG}.{mod}"), attr))


def source_hash(name: str) -> str:
    """Of the upstream source: the vendored call rewrite (SCR-WP0-COEXIST, ``alloy._vendor._warp_calls``) undone."""
    try:
        src = upstream(inspect.getsource(_resolve(name)))
    except (AttributeError, ImportError, OSError, TypeError):
        return "missing"
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def changed(names: tuple[str, ...] | None = None) -> list[str]:
    """Tracked functions (default: the route's) whose source differs from PIN (empty when mujoco_warp matches)."""
    return [n for n in (tracked() if names is None else names) if SOURCES.get(n) != source_hash(n)]


def _main() -> None:
    for name in sorted(set(tracked()) | set(GUARD) | set(LAZY) | set(BLOCK)):
        print(f'    "{name}": "{source_hash(name)}",')


if __name__ == "__main__":
    sys.exit(_main())
