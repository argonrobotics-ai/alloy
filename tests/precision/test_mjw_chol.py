# SPDX-License-Identifier: Apache-2.0
"""BUL-0016 guard (coordinator ruling 2026-11-04 09:32Z): MuJoCo C's pivot clamp on MuJoCo Warp's Newton Cholesky,
bitwise identical to MuJoCo Warp's own when no pivot is clamped (CPU; the GPU acceptance is the BUL-0016 replay and
T12 / T09 / T11 at 128 worlds).

- one tile (nv <= 32): MuJoCo Warp's kernel then the guard's check: random SPD Hessians keep search, search_dot and
  newton_decrement bit for bit; a Hessian whose pivot comes out <= 0 in float32 makes the stock outputs non-finite and
  the guard's finite, with the refactor counted, and the clean world beside it still bit for bit;
- blocked (nv > 32): the check after MuJoCo Warp's blocked factorization leaves a good world alone and refactors a
  world whose factor diagonal is NaN over the padded size; a reused guarded factor passes the check;
- a SolverMuJoCo Newton run steps bit for bit with the guard and without; its Data has the scratch from construction;
- HealthMonitor raises FALLBACK_USED (a notice: the world stays OK) for a world with clamps;
- the fallback runs in float64 (MuJoCo C's precision): E9 100072:1890's Hessian, which a float32 clamp overflowed to
  inf, solves finite and matches the float64 solve;
- the guard's mujoco_warp sources match the pin.
"""

from __future__ import annotations

from pathlib import Path

from alloy._vendor import newton
import numpy as np
import pytest
import warp as wp
from alloy._vendor.mujoco_warp._src import solver as mjw_solver

from alloy.precision import mjw_chol, mjw_chol_f64, mjw_pin
from alloy.precision.health import HealthMonitor
from alloy.precision.types import CheckPolicy, HealthFlags, Severity

NV = 22


def _spd(rng, n: int, cond: float = 1e3) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return (q @ np.diag(np.geomspace(1.0, 1.0 / cond, n)) @ q.T).astype(np.float32)


def _singular(n: int) -> np.ndarray:
    """SPD up to row 3, which repeats row 2 minus 1e-7 on the diagonal: pivot 3 is 0 or negative in float32."""
    h = np.eye(n, dtype=np.float32)
    h[2, 2] = h[3, 3] = 1.0
    h[2, 3] = h[3, 2] = 1.0
    h[3, 3] = np.float32(1.0) - np.float32(1e-7)
    return h


def _run(hs: np.ndarray, g: np.ndarray, guarded: bool):
    """MuJoCo Warp's one-tile kernel, then (guarded) the guard's check, as ``mjw_chol._guarded`` launches them."""
    nw = hs.shape[0]
    arr = lambda a, dt=float: wp.array(a, dtype=dt, device="cpu")  # noqa: E731
    search, dot, dec = arr(np.zeros((nw, NV))), arr(np.zeros(nw)), arr(np.zeros(nw))
    h, grad, done = arr(hs), arr(g), arr(np.zeros(nw, bool), wp.bool)
    wp.launch_tiled(mjw_solver._update_gradient_cholesky(NV, False), dim=nw,
                    inputs=[grad, h, arr(np.ones(nw, np.int32), int), done], outputs=[search, dot, dec],
                    block_dim=64, device="cpu")
    clamps = arr(np.zeros(nw, np.int32), wp.int32)
    if guarded:
        no_factor = arr(np.zeros((nw, 0, 0)))                  # one tile: MuJoCo Warp keeps no hfactor
        wp.launch(mjw_chol._check, dim=nw, inputs=[NV, done, h, no_factor, grad, clamps,
                                                   arr(np.zeros((nw, NV + 1, NV)), wp.float64)],
                  outputs=[search, dot, dec], device="cpu")
    return search.numpy(), dot.numpy(), dec.numpy(), clamps.numpy()


def _both(hs, g):
    return _run(hs, g, False), _run(hs, g, True)


def test_one_tile_is_bitwise_when_no_pivot_is_clamped() -> None:
    rng = np.random.default_rng(1)
    hs = np.stack([_spd(rng, NV, c) for c in (1e2, 1e4, 1e6)])
    g = rng.standard_normal((3, NV)).astype(np.float32)
    stock, guard = _both(hs, g)
    for a, b in zip(stock, guard[:3]):
        assert np.array_equal(a, b)
    assert not guard[3].any()


def test_one_tile_clamps_a_non_positive_pivot() -> None:
    rng = np.random.default_rng(2)
    hs = np.stack([_spd(rng, NV), _singular(NV)])
    g = rng.standard_normal((2, NV)).astype(np.float32)
    stock, guard = _both(hs, g)
    assert not np.isfinite(stock[0][1]).all() or not np.isfinite(stock[1][1]), "the stock factor must fail here"
    search, dot, dec, clamps = guard
    assert np.isfinite(search[1]).all() and np.isfinite(dot[1]) and np.isfinite(dec[1])
    assert clamps.tolist() == [0, 1]
    for a, b in zip(stock, guard[:3]):
        assert np.array_equal(a[0], b[0])                    # the clean world, bit for bit


def test_blocked_check_refactors_only_a_bad_world() -> None:
    """nv 40 padded to 48 (MuJoCo Warp pads ctx.h with identity); world 1's stock factor went NaN at pivot 5."""
    rng = np.random.default_rng(3)
    nv, pad, nw = 40, 48, 2
    h = np.stack([np.eye(pad, dtype=np.float32) for _ in range(nw)])
    for w in range(nw):
        h[w, :nv, :nv] = _spd(rng, nv)
    u = np.stack([np.linalg.cholesky(x.astype(np.float64)).T.astype(np.float32) for x in h])
    u[1, 5:, 5:] = np.nan
    g = np.zeros((nw, pad), np.float32)
    g[:, :nv] = rng.standard_normal((nw, nv))
    marker = np.full((nw, nv), 7.0, np.float32)
    arr = lambda a, dt=float: wp.array(a, dtype=dt, device="cpu")  # noqa: E731
    search, dot, dec = arr(marker), arr(np.full(nw, 7.0)), arr(np.full(nw, 7.0))
    hf, clamps = arr(u), arr(np.zeros(nw, np.int32), wp.int32)
    f64 = arr(np.zeros((nw, nv + 1, nv)), wp.float64)
    args = [nv, arr(np.zeros(nw, bool), wp.bool), arr(h), hf, arr(g), clamps, f64]
    wp.launch(mjw_chol._check, dim=nw, inputs=args, outputs=[search, dot, dec], device="cpu")
    s = search.numpy()
    assert np.array_equal(s[0], marker[0]) and dot.numpy()[0] == 7.0     # good world untouched
    want = -np.linalg.solve(h[1, :nv, :nv].astype(np.float64), g[1, :nv].astype(np.float64))
    np.testing.assert_allclose(s[1], want, rtol=2e-3, atol=1e-4)
    f = np.triu(hf.numpy()[1])
    assert np.isfinite(f).all() and clamps.numpy().tolist() == [0, 1]
    np.testing.assert_array_equal(f[nv:, nv:], np.eye(pad - nv))     # padding: identity
    assert not f[:nv, nv:].any()
    wp.launch(mjw_chol._check, dim=nw, inputs=args, outputs=[search, dot, dec], device="cpu")
    assert clamps.numpy().tolist() == [0, 1], "a reused guarded factor must pass the check"


def test_clamped_factor_passes_its_own_check() -> None:
    """A clamped diagonal is exactly sqrt(mjMINVAL) in float32, so the skip-unchanged reuse does not refactor."""
    h = _singular(NV)[None].repeat(1, axis=0)
    hf = wp.array(np.full((1, NV, NV), np.nan, np.float32), dtype=float, device="cpu")
    g = wp.array(np.ones((1, NV), np.float32), dtype=float, device="cpu")
    out = [wp.zeros((1, NV), dtype=float, device="cpu"), wp.zeros(1, dtype=float, device="cpu"),
           wp.zeros(1, dtype=float, device="cpu")]
    clamps = wp.zeros(1, dtype=wp.int32, device="cpu")
    f64 = wp.zeros((1, NV + 1, NV), dtype=wp.float64, device="cpu")
    args = [NV, wp.zeros(1, dtype=wp.bool, device="cpu"), wp.array(h, dtype=float, device="cpu"), hf, g, clamps, f64]
    for _ in range(2):
        wp.launch(mjw_chol._check, dim=1, inputs=args, outputs=out, device="cpu")
    d = np.diag(hf.numpy()[0])
    assert clamps.numpy().tolist() == [1] and d.min() == np.float32(np.sqrt(1e-15))


@wp.kernel
def _fallback(h: wp.array3d[float], grad: wp.array2d[float], f64: wp.array3d[wp.float64], search: wp.array2d[float],
              dot: wp.array[float], dec: wp.array[float]):
    w = wp.tid()
    mjw_chol_f64.factor64(h, w, h.shape[1], f64)
    mjw_chol_f64.solve64(w, h.shape[1], f64, grad, search, dot, dec)


def test_fallback_is_float64_on_e9_1890() -> None:
    """E9 100072:1890's Hessian at the step MuJoCo Warp broke (main f97b84be, L4): positive definite in float64
    (smallest eigenvalue 5e-6, condition 3.4e7). A float32 clamp overflowed the search to inf; the float64 fallback
    needs no clamp and matches the float64 solve."""
    z = np.load(Path(__file__).with_name("bul0016_1890_hessian.npz"))
    h, g = z["h"][None], z["grad"][None]
    nv = h.shape[1]
    arr = lambda a, dt=float: wp.array(a, dtype=dt, device="cpu")  # noqa: E731
    f64, search, dot, dec = arr(np.zeros((1, nv + 1, nv)), wp.float64), arr(np.zeros((1, nv))), arr([0.0]), arr([0.0])
    wp.launch(_fallback, dim=1, inputs=[arr(h), arr(g), f64], outputs=[search, dot, dec], device="cpu")
    full = np.triu(z["h"]).astype(np.float64)
    full = full + np.triu(full, 1).T
    want = -np.linalg.solve(full, z["grad"].astype(np.float64))
    x = search.numpy()[0]
    assert np.isfinite(x).all() and np.isfinite(dot.numpy()).all()
    np.testing.assert_allclose(x, want, rtol=1e-5, atol=1e-6 * np.abs(want).max())
    assert np.diag(f64.numpy()[0, :nv]).min() > np.sqrt(1e-15)            # no pivot needed the clamp


def _pile(worlds: int) -> newton.Model:
    b = newton.ModelBuilder()
    for i in range(3):
        body = b.add_body(xform=wp.transform(wp.vec3(0.03 * i, 0.0, 0.05 + 0.099 * i), wp.quat_identity()))
        b.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    s = newton.ModelBuilder()
    s.replicate(b, worlds)
    s.add_ground_plane()
    return s.finalize(device="cpu")


def _steps(guard: bool, n: int = 15) -> np.ndarray:
    if guard:
        assert mjw_chol.install()
    else:
        mjw_chol.uninstall()
    try:
        model = _pile(2)
        solver = newton.solvers.SolverMuJoCo(model, njmax=200, nconmax=64, solver="newton")
        s0, s1, ctl = model.state(), model.state(), model.control()
        for _ in range(n):
            solver.step(s0, s1, ctl, None, 1e-3)
            s0, s1 = s1, s0
        return np.concatenate([s0.body_q.numpy().ravel(), s0.body_qd.numpy().ravel()])
    finally:
        mjw_chol.install()


def test_solver_steps_bitwise_with_and_without_the_guard() -> None:
    assert np.array_equal(_steps(True), _steps(False))


def test_solver_data_is_prepared_at_construction() -> None:
    """Scratch exists before the first step, so a CUDA graph captured on that step runs the guard."""
    assert mjw_chol.install()
    solver = newton.solvers.SolverMuJoCo(_pile(2), njmax=200, nconmax=64, solver="newton")
    clamps = mjw_chol.clamp_counts(solver.mjw_data)
    assert clamps is not None and clamps.shape == (2,)


def test_clamps_raise_fallback_used() -> None:
    model = _pile(2)
    solver = newton.solvers.SolverMuJoCo(model, njmax=200, nconmax=64, solver="newton")
    monitor = HealthMonitor(model, CheckPolicy(level="cheap", every=1, full_every=1), solvers=[solver])
    clamps = mjw_chol.clamp_counts(solver.mjw_data)
    assert clamps is not None
    clamps.assign(np.array([0, 3], dtype=np.int32))
    state = model.state()
    monitor.update(0, state, state, None)
    rep = monitor.poll()
    flags = rep.flags.astype(np.int64) & int(HealthFlags.FALLBACK_USED)
    assert flags.tolist() == [0, int(HealthFlags.FALLBACK_USED)]
    # a notice: listed, and it adds nothing to the severity (both worlds carry the pile's START_PEN, DEGRADED)
    assert int(rep.severity[1]) == int(rep.severity[0]) and 1 not in rep.invalid_worlds


def test_guard_sources_match_the_pin() -> None:
    assert mjw_pin.changed(mjw_pin.GUARD) == [], "mujoco_warp's Newton Cholesky changed: recheck mjw_chol"
