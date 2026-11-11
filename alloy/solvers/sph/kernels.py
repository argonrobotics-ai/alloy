# SPDX-License-Identifier: Apache-2.0
"""SPH smoothing kernels, half-space boundary volume and fixed-point helpers (Warp functions).

Kernels use the support radius ``h`` (W = 0 for r >= h), as SPlisHSPlasH does:
- cubic spline: W = 8/(pi h^3) f(q), f = 6q^3 - 6q^2 + 1 (q <= 1/2), 2(1-q)^3 (q <= 1);
- Wendland C2: W = 21/(2 pi h^3) (1-q)^4 (1+4q).

Boundary volume of a solid half-space at signed distance ``d`` (d > 0 outside the solid), with the order-1 tube-map
curvature correction of Nogina and Sellan 2026 (SPlisHSPlasH@dba6b26:SPlisHSPlasH/Utilities/TubeMapDensity.cpp,
area factor A(t) = 1 - H t / 2, H = -(k1 + k2), negative for convex solids):

    Psi(d) = int_{-h}^{min(h, -d)} P(t) (1 - H t / 2) dt,   P(t) = p(|t|/h) / h  (projected kernel)

With z = clamp(-d/h, -1, 1): Psi = 1/2 + sign(z) G0(|z|) - (H h / 2) (G1(|z|) - G1(1)) and
dPsi/dd = -p(|z|) (1 - H h z / 2) / h for |d| < h. G0(y) = int_0^y p, G1(y) = int_0^y q p. The cubic p matches
TubeMapDensity.cpp's P_SLICE_IN/OUT x 16; the polynomials were derived with numpy.polynomial (docs/notes/wp9-*.md)
and are checked against numerical quadrature in tests/solvers/sph/test_kernels.py.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

KERNEL_CUBIC = wp.constant(0)
KERNEL_WENDLAND = wp.constant(1)
KERNEL_IDS = {"cubic": 0, "wendland_c2": 1}

# Fixed-point scale for deterministic integer accumulation of forces [N], impulses and masses [kg] (2^40).
FIX_SCALE = wp.constant(1099511627776.0)
FIX_SCALE_HOST = float(2**40)

# Principal-curvature clamp |k| <= 2/h (TubeMapDensity.cpp MAX_PRINCIPAL_CURVATURE).
MAX_PRINCIPAL_CURVATURE = wp.constant(2.0)

_PI = math.pi


@wp.func
def kernel_w(r: float, h: float, kind: int) -> float:
    """Kernel value W(r) [1/m^3]."""
    q = r / h
    if q >= 1.0:
        return 0.0
    h3 = h * h * h
    if kind == 1:
        s = 1.0 - q
        return (21.0 / (2.0 * wp.pi * h3)) * s * s * s * s * (1.0 + 4.0 * q)
    k = 8.0 / (wp.pi * h3)
    if q <= 0.5:
        return k * (6.0 * q * q * q - 6.0 * q * q + 1.0)
    s = 1.0 - q
    return k * 2.0 * s * s * s


@wp.func
def kernel_grad(rv: wp.vec3, h: float, kind: int) -> wp.vec3:
    """Kernel gradient with respect to x_i, rv = x_i - x_j [1/m^4]."""
    r = wp.length(rv)
    q = r / h
    if q >= 1.0 or r <= 1.0e-9 * h:
        return wp.vec3(0.0, 0.0, 0.0)
    h3 = h * h * h
    dwdq = float(0.0)
    if kind == 1:
        s = 1.0 - q
        dwdq = (21.0 / (2.0 * wp.pi * h3)) * (-20.0 * q * s * s * s)
    else:
        k = 8.0 / (wp.pi * h3)
        if q <= 0.5:
            dwdq = k * (18.0 * q * q - 12.0 * q)
        else:
            s = 1.0 - q
            dwdq = k * (-6.0 * s * s)
    return rv * (dwdq / (r * h))


@wp.func
def projected_p(q: float, kind: int) -> float:
    """Dimensionless projected kernel p(q), q = |t|/h in [0, 1]; int_{-1}^{1} p = 1."""
    if q >= 1.0:
        return 0.0
    q2 = q * q
    if kind == 1:
        return 1.5 + q2 * (-10.5 + q2 * (52.5 + q * (-84.0 + q * (52.5 - 12.0 * q))))
    if q <= 0.5:
        return 1.4 + q2 * (-8.0 + q2 * (24.0 - 19.2 * q))
    return 1.6 + q2 * (-16.0 + q * (32.0 + q * (-24.0 + 6.4 * q)))


@wp.func
def _g0(y: float, kind: int) -> float:
    """G0(y) = int_0^y p(q) dq, y in [0, 1]."""
    y2 = y * y
    if kind == 1:
        return y * (1.5 + y2 * (-3.5 + y2 * (10.5 + y * (-14.0 + y * (7.5 - 1.5 * y)))))
    if y <= 0.5:
        return y * (1.4 + y2 * (-8.0 / 3.0 + y2 * (4.8 - 3.2 * y)))
    return -1.0 / 30.0 + y * (1.6 + y2 * (-16.0 / 3.0 + y * (8.0 + y * (-4.8 + y * (16.0 / 15.0)))))


@wp.func
def _g1(y: float, kind: int) -> float:
    """G1(y) = int_0^y q p(q) dq, y in [0, 1]."""
    y2 = y * y
    if kind == 1:
        return y2 * (0.75 + y2 * (-2.625 + y2 * (8.75 + y * (-12.0 + y * (6.5625 - (4.0 / 3.0) * y)))))
    if y <= 0.5:
        return y2 * (0.7 + y2 * (-2.0 + y2 * (4.0 - (19.2 / 7.0) * y)))
    return -1.0 / 280.0 + y2 * (0.8 + y2 * (-4.0 + y * (6.4 + y * (-4.0 + y * (6.4 / 7.0)))))


@wp.func
def clamp_principal_curvature(k: float, h: float) -> float:
    """Clamp one principal curvature to |k| <= 2/h, where the order-1 expansion stays valid."""
    kmax = MAX_PRINCIPAL_CURVATURE / h
    return wp.clamp(k, -kmax, kmax)


@wp.func
def halfspace_volume(d: float, h: float, H: float, kind: int) -> wp.vec2:
    """(Psi, dPsi/dd) of a solid half-space at signed distance d [m]; H = -(k1 + k2) [1/m], already clamped."""
    if d >= h:
        return wp.vec2(0.0, 0.0)
    Hc = H
    z = wp.clamp(-d / h, -1.0, 1.0)
    y = wp.abs(z)
    sgn = float(1.0)
    if z < 0.0:
        sgn = -1.0
    g1_full = _g1(1.0, kind)
    psi = 0.5 + sgn * _g0(y, kind) - 0.5 * Hc * h * (_g1(y, kind) - g1_full)
    dpsi = float(0.0)
    if d > -h:
        dpsi = -projected_p(y, kind) * (1.0 - 0.5 * Hc * h * z) / h
    return wp.vec2(psi, dpsi)


@wp.func
def to_fixed(x: float) -> wp.int64:
    """Quantize to int64 at FIX_SCALE (deterministic integer accumulation)."""
    return wp.int64(wp.round(wp.float64(x) * wp.float64(FIX_SCALE)))


@wp.func
def from_fixed(q: wp.int64) -> float:
    return float(wp.float64(q) / wp.float64(FIX_SCALE))


# ----------------------------------------------------------------------------------------------- host (numpy) mirrors


def kernel_w_np(r: np.ndarray, h: float, kind: str = "cubic") -> np.ndarray:
    """Host mirror of kernel_w (float64)."""
    q = np.asarray(r, dtype=np.float64) / h
    out = np.zeros_like(q)
    h3 = h**3
    if kind == "wendland_c2":
        m = q < 1.0
        s = 1.0 - q[m]
        out[m] = 21.0 / (2.0 * _PI * h3) * s**4 * (1.0 + 4.0 * q[m])
        return out
    k = 8.0 / (_PI * h3)
    m1 = q <= 0.5
    m2 = (q > 0.5) & (q < 1.0)
    out[m1] = k * (6.0 * q[m1] ** 3 - 6.0 * q[m1] ** 2 + 1.0)
    out[m2] = k * 2.0 * (1.0 - q[m2]) ** 3
    return out


def lattice_sum(spacing: float, h: float, kind: str = "cubic") -> float:
    """Sum of W over an infinite cubic lattice (self included) [1/m^3]; m = rho0 / S makes interior rho = rho0."""
    n = int(math.ceil(h / spacing)) + 1
    ax = np.arange(-n, n + 1, dtype=np.float64) * spacing
    gx, gy, gz = np.meshgrid(ax, ax, ax, indexing="ij")
    r = np.sqrt(gx**2 + gy**2 + gz**2)
    return float(kernel_w_np(r, h, kind).sum())


_G0 = {
    "cubic": ((0.5, [0.0, 1.4, 0.0, -8.0 / 3.0, 0.0, 4.8, -3.2]),
              (1.0, [-1.0 / 30.0, 1.6, 0.0, -16.0 / 3.0, 8.0, -4.8, 16.0 / 15.0])),
    "wendland_c2": ((1.0, [0.0, 1.5, 0.0, -3.5, 0.0, 10.5, -14.0, 7.5, -1.5]),),
}


def _g0_np(y: float, kind: str) -> float:
    for top, coeffs in _G0[kind]:
        if y <= top:
            return float(np.polynomial.polynomial.polyval(y, coeffs))
    return 0.5


def halfspace_volume_np(d: float, h: float, kind: str = "cubic") -> float:
    """Host mirror of halfspace_volume for a flat wall (H = 0)."""
    if d >= h:
        return 0.0
    z = min(max(-d / h, -1.0), 1.0)
    return 0.5 + math.copysign(_g0_np(abs(z), kind), z)


def _half_lattice(spacing: float, h: float, kind: str) -> tuple[float, float]:
    """(deficit 1 - F1, |dF1/dz|) of a particle on the first lattice layer next to a flat wall (fluid on layers
    >= 0, interior lattice normalised to density ratio 1)."""
    n = int(math.ceil(h / spacing)) + 1
    ax = np.arange(-n, n + 1, dtype=np.float64) * spacing
    g = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
    r = np.linalg.norm(g, axis=1)
    w = kernel_w_np(r, h, kind)
    s = float(w.sum())
    up = (g[:, 2] >= 0.0) & (r > 0.0)
    eps = 1.0e-7 * h
    dw = (kernel_w_np(r[up] + eps, h, kind) - kernel_w_np(r[up] - eps, h, kind)) / (2.0 * eps)
    grad_z = float(np.sum(dw * (-g[up, 2]) / r[up])) / s
    f1 = float(w[g[:, 2] >= 0.0].sum()) / s
    return 1.0 - f1, abs(grad_z)


def boundary_fit(spacing: float, h: float, kind: str = "cubic", slope_factor: float = 2.0) -> tuple[float, float]:
    """Wall offset delta [m] and amplitude A of the boundary volume Psi_eff(d) = A Psi(d + delta).

    At the rest layer (d = r) the fit matches the lattice deficit (A Psi = 1 - F1, so a lattice next to a flat
    wall has rho = rho0) and sets the slope to ``slope_factor`` times the half-lattice gradient |dF1/dz|. With the
    variationally consistent pressure force (wall term k_i dPsi, as SPlisHSPlasH), a uniform pressure exerts
    -k (2 G_f + G_b) on a wall particle, which vanishes for G_b = -2 G_f: slope_factor 2 makes the hydrostatic rest
    state force-free with a smooth pressure profile and the wall restoring. The plain integral is off on both counts
    (cubic, h = 4r: Psi(r) = 0.188 vs deficit 0.150; |Psi'(r)| = 0.84 |dF1/dz|), and a mirrored wall term (2 k_i)
    is not derivable from the density constraint, so it pumps energy (measured: hydrostatic columns blow up).
    """
    deficit, grad = _half_lattice(spacing, h, kind)
    r = 0.5 * spacing
    target = deficit / (slope_factor * grad)  # Psi / |Psi'| at the fitted distance

    def ratio(d: float) -> float:
        e = 1.0e-7 * h
        dpsi = (halfspace_volume_np(d + e, h, kind) - halfspace_volume_np(d - e, h, kind)) / (2.0 * e)
        return halfspace_volume_np(d, h, kind) / max(abs(dpsi), 1.0e-30)

    lo, hi = 0.0, 0.999 * h  # Psi/|Psi'| decreases with d
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if ratio(mid) > target:
            lo = mid
        else:
            hi = mid
    d = 0.5 * (lo + hi)
    return d - r, deficit / halfspace_volume_np(d, h, kind)


def laplacian_lattice_factor(spacing: float, h: float, kind: str = "cubic") -> float:
    """Ratio (discrete / exact) of the Weiler/Monaghan viscous Laplacian
    2 (d+2) sum_j V (v_ij . x_ij) / (|x_ij|^2 + 0.01 h^2) gradW_ij on the rest lattice (V = 1 / S), for the
    divergence-free shear v = (z^2 / 2, 0, 0) (exact: d^2 v_x / dz^2 = 1). With h = 2 spacings the cubic kernel gives
    0.69: the solver divides the operator by this factor so the card's viscosity is the effective one."""
    g, r, dw = _lattice_dw(spacing, h, kind)
    grad_x = dw * g[:, 0] / r
    v_ij_x = -0.5 * g[:, 2] ** 2  # v_i - v_j with v_x(z) = z^2 / 2, z_i = 0, z_j = -z_ij
    vol = 1.0 / lattice_sum(spacing, h, kind)
    lap = 10.0 * vol * float(np.sum(v_ij_x * g[:, 0] / (r**2 + 0.01 * h * h) * grad_x))
    return lap


def lattice_diagonal(spacing: float, h: float, kind: str = "cubic") -> float:
    """DFSPH diagonal sum_j V^2 |dW_ij|^2 (V = 1/S) of an interior particle of the rest lattice [1/m^2]: 1/alpha in
    the bulk, where |G|^2 = 0 by symmetry (pressure.py)."""
    _g, _r, dw = _lattice_dw(spacing, h, kind)
    vol = 1.0 / lattice_sum(spacing, h, kind)
    return float(np.sum((vol * dw) ** 2))


def _lattice_dw(spacing: float, h: float, kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Offsets x_ij (x_i at the origin), distances and dW/dr of the rest-lattice neighbours within h (self out)."""
    n = int(math.ceil(h / spacing)) + 1
    ax = np.arange(-n, n + 1, dtype=np.float64) * spacing
    g = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
    r = np.linalg.norm(g, axis=1)
    m = (r > 0.0) & (r < h)
    eps = 1.0e-7 * h
    dw = (kernel_w_np(r[m] + eps, h, kind) - kernel_w_np(r[m] - eps, h, kind)) / (2.0 * eps)
    return g[m], r[m], dw
