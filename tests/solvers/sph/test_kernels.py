# SPDX-License-Identifier: Apache-2.0
"""SPH kernels, half-space boundary volume and the wall fit (alloy.solvers.sph.kernels)."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.solvers.sph.kernels import (
    boundary_fit,
    halfspace_volume,
    halfspace_volume_np,
    kernel_grad,
    kernel_w,
    kernel_w_np,
    lattice_sum,
)

KINDS = [("cubic", 0), ("wendland_c2", 1)]


@pytest.mark.parametrize("name,kind", KINDS)
def test_kernel_normalised(name, kind):
    # integral of W over R^3 by a fine lattice sum: sum W(x_j) * dV -> 1
    h, dx = 1.0, 0.02
    ax = np.arange(-1.0, 1.0 + dx / 2, dx)
    g = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
    total = kernel_w_np(np.linalg.norm(g, axis=1), h, name).sum() * dx**3
    assert abs(total - 1.0) < 2e-3


@wp.kernel
def _grad_check(r: wp.array[wp.vec3], h: float, kind: int, out: wp.array[wp.vec2]):
    i = wp.tid()
    x = r[i]
    e = 1.0e-4 * h
    g = kernel_grad(x, h, kind)
    fd = wp.vec3(
        kernel_w(wp.length(x + wp.vec3(e, 0.0, 0.0)), h, kind) - kernel_w(wp.length(x - wp.vec3(e, 0.0, 0.0)), h, kind),
        kernel_w(wp.length(x + wp.vec3(0.0, e, 0.0)), h, kind) - kernel_w(wp.length(x - wp.vec3(0.0, e, 0.0)), h, kind),
        kernel_w(wp.length(x + wp.vec3(0.0, 0.0, e)), h, kind) - kernel_w(wp.length(x - wp.vec3(0.0, 0.0, e)), h, kind),
    ) / (2.0 * e)
    out[i] = wp.vec2(wp.length(g - fd), wp.length(g))


@pytest.mark.parametrize("name,kind", KINDS)
def test_kernel_gradient_matches_finite_difference(name, kind, rng):
    h = 1.0
    r = wp.array(rng.uniform(-h, h, (300, 3)).astype(np.float32), dtype=wp.vec3)
    out = wp.zeros(300, dtype=wp.vec2)
    wp.launch(_grad_check, dim=300, inputs=[r, h, kind, out])
    err, mag = out.numpy().T
    assert err.max() < 2e-3 * mag.max()


@wp.kernel
def _hv(d: wp.array[float], h: float, H: float, kind: int, out: wp.array[wp.vec3]):
    i = wp.tid()
    e = 1.0e-4 * h
    v = halfspace_volume(d[i], h, H, kind)
    fd = (halfspace_volume(d[i] + e, h, H, kind)[0] - halfspace_volume(d[i] - e, h, H, kind)[0]) / (2.0 * e)
    out[i] = wp.vec3(v[0], v[1], fd)


def _halfspace_quadrature(d: float, h: float, name: str) -> float:
    """Volume of W in the solid half-space z < -d (particle at the origin, wall plane at z = -d)."""
    dx = 0.01 * h
    ax = np.arange(-h + dx / 2, h, dx)  # cell centres: no sample lies on the wall plane
    g = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
    w = kernel_w_np(np.linalg.norm(g, axis=1), h, name)
    return float(w[g[:, 2] < -d].sum() * dx**3)


@pytest.mark.parametrize("name,kind", KINDS)
def test_halfspace_volume_closed_form(name, kind):
    h = 1.0
    ds = np.array([-1.5, -0.6, -0.25, 0.0, 0.25, 0.5, 0.8, 1.2], dtype=np.float32)
    out = wp.zeros(len(ds), dtype=wp.vec3)
    wp.launch(_hv, dim=len(ds), inputs=[wp.array(ds), h, 0.0, kind, out])
    psi, dpsi, fd = out.numpy().T
    for d, p in zip(ds, psi):
        assert abs(p - _halfspace_quadrature(float(d), h, name)) < 1e-3, d
        assert abs(p - halfspace_volume_np(float(d), h, name)) < 1e-6
    assert psi[0] == pytest.approx(1.0, abs=1e-6) and psi[3] == pytest.approx(0.5, abs=1e-6) and psi[-1] == 0.0
    assert np.abs(dpsi - fd).max() < 2e-3 * np.abs(dpsi).max()


def test_curvature_lowers_convex_volume():
    # a convex wall (H < 0) holds less solid inside the kernel than a plane at the same distance
    out = wp.zeros(2, dtype=wp.vec3)
    wp.launch(_hv, dim=1, inputs=[wp.array([0.3], dtype=float), 1.0, 0.0, 0, out[:1]])
    wp.launch(_hv, dim=1, inputs=[wp.array([0.3], dtype=float), 1.0, -1.0, 0, out[1:]])
    flat, convex = out.numpy()[:, 0]
    assert convex < flat


@pytest.mark.parametrize("name", ["cubic", "wendland_c2"])
def test_boundary_fit_matches_lattice_value(name):
    # the fitted wall volume equals the half-lattice deficit at the first layer (rest density exact at walls)
    sp, h = 2.0e-3, 4.0e-3
    delta, amp = boundary_fit(sp, h, name)
    n = 3
    ax = np.arange(-n, n + 1) * sp
    g = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
    w = kernel_w_np(np.linalg.norm(g, axis=1), h, name)
    deficit = 1.0 - w[g[:, 2] >= 0].sum() / w.sum()
    assert amp * halfspace_volume_np(0.5 * sp + delta, h, name) == pytest.approx(deficit, rel=1e-6)
    assert lattice_sum(sp, h, name) == pytest.approx(w.sum(), rel=1e-12)
