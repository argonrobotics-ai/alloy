# SPDX-License-Identifier: Apache-2.0
"""Standalone repro: MuJoCo Warp's Newton solver factors a finite, positive definite Hessian into NaN (float32).

Only mujoco_warp, warp and numpy. ``hessian_route_on.npz`` holds the solver's Hessian tile ``h`` [1, 32, 32] (upper
triangle used, nv = 22) and gradient ``grad`` [1, 32] at Newton iteration 3 of one MuJoCo Warp step on a LEAP hand + cube
scene (Newton solver, elliptic cone, implicitfast, impratio 100, dt 0.625 ms; WP13's SB-7 episode 100002:867, filed in
the bully corpus). The Hessian's diagonal spans 1.5e-5 to 100 and its condition number is 2.8e7 in float64.

MuJoCo Warp's ``solver._update_gradient_cholesky`` calls Warp's ``tile_cholesky_inplace`` (fill_mode upper), a
sequential column Cholesky in float32 with no pivot guard. Pivot 15 comes out at -4.8e-7, ``sqrt`` gives NaN and the
NaN reaches ``qacc``. The same sequential sums in numpy float32 give the same negative pivot. LAPACK float32 (blocked),
float64, and the Jacobi-scaled matrix all factor it. MuJoCo C (float64) clamps a pivot below ``mjMINVAL`` in
``mju_cholFactor``; MuJoCo Warp has no clamp and no fallback.

``hessian_route_off.npz`` is the same iteration with the stock MuJoCo Warp kernels (Alloy's ordered route off): its
condition number is 1.7e7 and it factors, so the route's rounding only moves this step across the edge.

python bench/bully/repro/mjw_cholesky_nan/repro.py [npz] [--device cpu|cuda:0]
"""

import argparse
from pathlib import Path

import numpy as np
import warp as wp
from alloy._vendor.mujoco_warp._src import solver

NV = 22


def sequential_pivots(h: np.ndarray, dtype: type) -> list[float]:
    """Pivots of the column Cholesky Warp's tile_cholesky runs (upper, sequential sums), in ``dtype``."""
    upper = np.triu(h[:NV, :NV]).astype(dtype)
    a = upper + np.triu(upper, 1).T
    u = np.zeros((NV, NV), dtype)
    pivots = []
    for j in range(NV):
        s = a[j, j]
        for k in range(j):
            s = dtype(s - u[k, j] * u[k, j])
        pivots.append(float(s))
        if s <= 0:
            return pivots
        u[j, j] = np.sqrt(s)
        inv = dtype(1.0) / u[j, j]  # as Warp: one reciprocal per column, then multiply
        for i in range(j + 1, NV):
            si = a[j, i]
            for k in range(j):
                si = dtype(si - u[k, i] * u[k, j])
            u[j, i] = dtype(si * inv)
    return pivots


def mjw_search(h: np.ndarray, grad: np.ndarray, device: str) -> np.ndarray:
    """MuJoCo Warp's Cholesky gradient kernel, as solver.update_gradient launches it for nv <= the block size."""
    with wp.ScopedDevice(device):
        search = wp.zeros(grad.shape, dtype=float)
        wp.launch_tiled(solver._update_gradient_cholesky(NV, False), dim=1,
                        inputs=[wp.array(grad, dtype=float), wp.array(h, dtype=float), wp.array([28], dtype=int),
                                wp.array([False], dtype=bool)],
                        outputs=[search, wp.zeros(1, dtype=float), wp.zeros(1, dtype=float)], block_dim=64)
        return search.numpy()[0, :NV]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="?", default=str(Path(__file__).parent / "hessian_route_on.npz"))
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    z = np.load(a.npz)
    h, grad = z["h"], z["grad"]
    upper = np.triu(h[0, :NV, :NV]).astype(np.float64)
    full = upper + np.triu(upper, 1).T
    ev = np.linalg.eigvalsh(full)
    print(f"Hessian: finite {bool(np.isfinite(h).all())}, eigenvalues {ev.min():.3g} .. {ev.max():.3g}, "
          f"condition {ev.max() / ev.min():.3g}, diagonal {np.diag(full).min():.3g} .. {np.diag(full).max():.3g}")
    for dtype in (np.float32, np.float64):
        piv = sequential_pivots(h[0], dtype)
        print(f"sequential Cholesky in {dtype.__name__}: smallest pivot {min(piv):.3g} at column {int(np.argmin(piv))}")
    lapack = np.linalg.cholesky(full.astype(np.float32))
    print(f"LAPACK float32 Cholesky finite: {bool(np.isfinite(lapack).all())}")
    s = mjw_search(h, grad, a.device)
    print(f"MuJoCo Warp _update_gradient_cholesky ({a.device}): search direction finite {bool(np.isfinite(s).all())}")


if __name__ == "__main__":
    main()
