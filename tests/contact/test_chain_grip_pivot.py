# SPDX-License-Identifier: Apache-2.0
"""SCR-WP3-CHAINGRIP-GPU: the pivot floor of `alloy.precision.mjw_chol` for models the chain-grip rule marks
(`alloy.contact.chain_grip.mark_pivot_floor`), on the CPU. WP5 owns mjw_chol (BUL-0016's guard, tests in
tests/precision/test_mjw_chol.py); these cover only the marked path and its keys.

- The rig's real Hessian (WP9's R4 rig, a CHAIN gripped at impratio 300, MuJoCo Warp CPU, t = 25 ms): indefinite by
  float32 rounding. mjMINVAL's clamp gives a non-finite step, the floor a finite descent step.
- Only a marked model launches ``_check_floor``; the mark and the guard's scratch survive the compact and DISCRETE
  solves' copies of the model and Data; a marked model takes the SPARSE-1 block path with its floor (0.1.2).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.precision import mjw_chol, mjw_chol_f64


@wp.kernel
def _fallback(h: wp.array3d[float], grad: wp.array2d[float], f64: wp.array3d[wp.float64], search: wp.array2d[float],
              dot: wp.array[float], dec: wp.array[float]):
    w = wp.tid()
    mjw_chol_f64.factor64(h, w, h.shape[1], f64)
    mjw_chol_f64.solve64(w, h.shape[1], f64, grad, search, dot, dec)


def _pile(worlds: int) -> newton.Model:
    b = newton.ModelBuilder()
    for i in range(3):
        body = b.add_body(xform=wp.transform(wp.vec3(0.03 * i, 0.0, 0.05 + 0.099 * i), wp.quat_identity()))
        b.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    s = newton.ModelBuilder()
    s.replicate(b, worlds)
    s.add_ground_plane()
    return s.finalize(device="cpu")


@wp.kernel
def _fallback_floor(h: wp.array3d[float], grad: wp.array2d[float], f64: wp.array3d[wp.float64],
                    search: wp.array2d[float], dot: wp.array[float], dec: wp.array[float]):
    w = wp.tid()
    mjw_chol_f64.factor64_floor(h, w, h.shape[1], f64, wp.float64(10.0))
    mjw_chol_f64.solve64(w, h.shape[1], f64, grad, search, dot, dec)


def test_pivot_floor_keeps_a_rounded_indefinite_hessian_finite() -> None:
    """SCR-WP3-CHAINGRIP-GPU: the Newton Hessian of WP9's R4 rig (a CHAIN gripped at impratio 300, MuJoCo Warp CPU,
    t = 25 ms, iteration 3) as float32 assembled it: indefinite by rounding (eigenvalues -1.9e-7 .. 98; the links'
    twist curvature is the 1e-6 armature floor). MuJoCo C's mjMINVAL clamp makes the step non-finite; the floor at
    10 * nv * eps32 * max diag H gives a finite descent step."""
    z = np.load(Path(__file__).with_name("chaingrip_r4_hessian.npz"))
    h, g = z["h"][None], z["grad"][None]
    nv = h.shape[1]
    full = np.triu(z["h"]).astype(np.float64)
    assert np.linalg.eigvalsh(full + np.triu(full, 1).T)[0] < 0.0             # the case: indefinite as rounded
    arr = lambda a, dt=float: wp.array(a, dtype=dt, device="cpu")  # noqa: E731
    out = {}
    for name, kern in (("minval", _fallback), ("floor", _fallback_floor)):
        f64, search = arr(np.zeros((1, nv + 1, nv)), wp.float64), arr(np.zeros((1, nv)))
        dot, dec = arr([0.0]), arr([0.0])
        wp.launch(kern, dim=1, inputs=[arr(h), arr(g), f64], outputs=[search, dot, dec], device="cpu")
        out[name] = (search.numpy()[0], float(dec.numpy()[0]))
    assert not np.isfinite(out["minval"][0]).all()
    x, newton_decrement = out["floor"]
    assert np.isfinite(x).all() and newton_decrement > 0.0                     # -x is a descent direction
    assert float(-(g[0].astype(np.float64) @ x)) == pytest.approx(newton_decrement, rel=1e-4)


def test_only_a_marked_model_takes_the_pivot_floor(monkeypatch) -> None:
    """Unmarked models launch ``_check`` (MuJoCo C's clamp), marked ones ``_check_floor``. The mark survives the
    compact solve's copies, which replace the model and its ``opt`` but share the Data's ``solver_niter`` (WP9's
    review)."""
    import dataclasses

    launched: list[str] = []
    launch0 = wp.launch

    def launch(kernel, *a, **k):
        if kernel in (mjw_chol._check, mjw_chol._check_floor):
            launched.append(kernel.key)
        return launch0(kernel, *a, **k)

    monkeypatch.setattr(mjw_chol.wp, "launch", launch)
    assert mjw_chol.install()
    for mark in (False, True):
        model = _pile(2)
        solver = newton.solvers.SolverMuJoCo(model, njmax=200, nconmax=64, solver="newton")
        m, d = solver.mjw_model, solver.mjw_data
        assert mjw_chol.pivot_floor(m, d) == 0.0
        if mark:
            mjw_chol.set_pivot_floor(m, 10.0, d)
        launched.clear()
        s0, s1, ctl = model.state(), model.state(), model.control()
        for _ in range(3):
            solver.step(s0, s1, ctl, None, 1e-3)
            s0, s1 = s1, s0
        want = {mjw_chol._check_floor.key} if mark else {mjw_chol._check.key}
        assert set(launched) == want
    d2 = dataclasses.replace(d, qacc=wp.zeros_like(d.qacc))
    m2 = dataclasses.replace(m, opt=dataclasses.replace(m.opt, tolerance=1.0))     # the compact copy's model and opt
    assert mjw_chol.pivot_floor(m2) == 0.0 and mjw_chol.pivot_floor(m2, d2) == 10.0
    assert mjw_chol._scratch_for(m2, d2) is mjw_chol._scratch_for(m, d)            # the shared slab
    wide = dataclasses.replace(m2, nv=m.nv + 4)
    s = mjw_chol._scratch_for(wide, d2)                                  # never narrower than the call's nv
    assert s is not None and s[1].shape[2] >= wide.nv


def test_a_marked_models_discrete_data_copy_finds_the_same_scratch() -> None:
    """MuJoCo Warp's DISCRETE solve runs on ``dataclasses.replace(d, qfrc_smooth=...)``, a new Data every step. A marked
    model (``set_pivot_floor``) finds the scratch prepared at construction through the shared ``solver_niter``, so it
    is guarded inside a CUDA graph capture and its fallbacks are counted."""
    import dataclasses

    assert mjw_chol.install()
    solver = newton.solvers.SolverMuJoCo(_pile(2), njmax=200, nconmax=64, solver="newton")
    m, d = solver.mjw_model, solver.mjw_data
    copy = dataclasses.replace(d, qfrc_smooth=wp.zeros_like(d.qfrc_smooth))
    mjw_chol.set_pivot_floor(m, 10.0, d)
    assert mjw_chol._scratch_for(m, copy) is mjw_chol._scratch_for(m, d)
    assert mjw_chol.clamp_counts(copy) is mjw_chol.clamp_counts(d)


def test_a_marked_model_takes_the_block_path_with_its_floor(monkeypatch) -> None:
    """Since 0.1.2 (Gemini audit T2-009) a model marked with ``set_pivot_floor`` takes the SPARSE-1 block path when it
    carries a block config, handing the block path its floor (mjw_block's float64 redo floors pivots as the dense
    redo does; tests/precision/test_mjw_block.py). Before, it stayed dense (WP5's review of SCR-WP3-CHAINGRIP-GPU), so
    a CHAIN beside clutter factored the whole entry densely."""
    from alloy.precision import mjw_block

    floors: list[float] = []
    monkeypatch.setattr(mjw_block, "factor_solve", lambda m, d, c, k, dense, floor=0.0: floors.append(floor) or True)
    assert mjw_chol.install()

    def steps(mark: bool) -> set[float]:
        floors.clear()
        model = _pile(2)
        solver = newton.solvers.SolverMuJoCo(model, njmax=200, nconmax=64, solver="newton", cone="elliptic")
        solver.mjw_model.alloy_block_newton = object()           # as if eligible
        if mark:
            mjw_chol.set_pivot_floor(solver.mjw_model, 10.0, solver.mjw_data)
        s0, s1, ctl = model.state(), model.state(), model.control()
        for _ in range(3):
            solver.step(s0, s1, ctl, None, 1e-3)
            s0, s1 = s1, s0
        return set(floors)

    assert steps(False) == {0.0}                                 # an unmarked model: mjMINVAL
    assert steps(True) == {10.0}
