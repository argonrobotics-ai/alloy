# SPDX-License-Identifier: Apache-2.0
"""Akinci 2013 surface tension and adhesion for SolverDFSPH (``surface_tension="akinci2013"``, SCR-DC5-5 item 2).

Akinci, Akinci & Teschner, "Versatile Surface Tension and Adhesion for SPH Fluids", ACM TOG 32(6) 182, 2013, as
SPlisHSPlasH@dba6b26 (MIT): SPlisHSPlasH/SurfaceTension/SurfaceTension_Akinci2013.cpp:62-171. Per fluid pair, with
K_ij = 2 rho0 / (rho_i + rho_j) and the normal n_i = h sum_j (m_j / rho_j) dW_ij (it points into the liquid):
    a_i += K_ij gamma_ij [ -m_j C(|x_ij|) x_ij / |x_ij| - s(|x_ij|) (n_i - n_j) ]      cohesion + curvature
    C(r) = 32 / (pi h^9) { (h - r)^3 r^3 if h/2 < r <= h;  2 (h - r)^3 r^3 - h^6 / 64 if 0 < r <= h/2 }
The h^6/64 term sits inside the 32 / (pi h^9) factor (the paper's eq. 2); SPlisHSPlasH subtracts it unscaled
(SPHKernels.h:373), which drops the short-range repulsion.
s(r) = 4 (1 - r/h) is ours. Akinci and SPlisHSPlasH use s = 1, so the curvature force jumps by gamma (n_i - n_j)
whenever a neighbour crosses r = h. Measured on a resting drop (R = 1.6 mm, r = 0.1 mm, sigma 0.072 N/m, undamped
after settling), that discontinuity drives a jitter of mean speed 4.0 cm/s (p99 8.5), about 0.2 x the capillary
speed sqrt(sigma / (rho R)). With s it is 0.95 cm/s (p99 2.1). s averages to about 1 over a uniform neighbourhood;
the flat-lattice calibration below changes by 0.6 %. The paper's normal direction (into the liquid) is kept: the
outward normal measured worse (6.9 cm/s).

Adhesion to walls: Akinci sums a_i -= beta rho0 V_b A(|x_ib|) x_ib / |x_ib| over boundary particles. The walls here
are shape SDFs, so the sum is the integral over the half-space behind each boundary slot. For a wall at distance d
along the outward normal n it is a_i -= beta rho0 G(d) n, with
    G(d) = pi int_{max(d, h/2)}^{h} A(s) (s^2 - d^2) ds,   A(s) = 0.007 / h^3.25 (-4 s^2 / h + 6 s - 2 h)^(1/4).
G(d) = pi 0.007 g(d / h) with g dimensionless (table ``adhesion_table``). The reaction goes to the slot impulse, so
wrenches and the walls' work carry it.

Physical inputs, per particle: ``sph:surface_tension`` sigma [N/m] and ``sph:adhesion`` a = W_adh / sigma (Young:
cos theta = a - 1, so a = 1 + cos theta; 0 = no adhesion, 1 = 90 degrees, 2 = complete wetting).
- gamma = sigma / (RAYLEIGH_FACTOR c_lattice).
  - c_lattice is the Kirkwood-Buff surface tension per unit gamma of the rest lattice's flat free surface
    (``lattice_sigma_per_gamma``). Cohesion and curvature are pair forces, so
    sigma = (1/A) sum_{i<j} [z_ij F_ij,z - (x_ij F_ij,x + y_ij F_ij,y)/2].
  - c_lattice scales as h^2 rho0 (k1 rho0 + k2), because Akinci's single gamma multiplies terms of different units.
    Cubic, h = 4r: k1 = 5.34e-3, k2 = 2.51.
  - The unrelaxed lattice is not the simulated surface. RAYLEIGH_FACTOR = 1.55 is fitted on free water drops'
    undamped l = 2 frequency omega_0 = sqrt(omega_d^2 + lambda^2) against Rayleigh's sqrt(8 sigma / (rho R^3)).
    The trials were 1.0, 1.376 and 1.724 at R/h 5-10. The frequency is not proportional to sqrt(gamma): its log
    slope in gamma is about 0.9.
  - The damping lambda is numerical: lambda / omega_d is 0.5-1.0, and lambda is about 100x water's viscous rate
    5 nu / R^2.
- beta = a sigma / c_adh, where c_adh = W_adh per unit beta of the rest lattice next to a flat wall (layers at
  d = r, 3r, ...; ``adhesion_work_per_beta``).

Measured limits (wp9-results.md, SCR-DC5-5, SCR-WP9-7). This is Akinci's model with a fitted constant, not a
calibrated physical surface tension.
- omega_0 / Rayleigh is 0.815 / 0.942 / 0.936 / 1.008 at R/h 3.5 / 5 / 7 / 10 (water), and 0.936 at R/h 7 with
  rho0 = 1420. The 10 % law therefore holds for R/h 5-10.
- The cohesion pair forces give the resting liquid a small shear rigidity and yield stress:
  - a settled free drop stays a few % aspherical;
  - a sessile drop's contact line stays pinned (contact angle hysteresis of about +-17 degrees), so the angle
    misses Young's acos(a - 1) by up to 25 degrees;
  - static measures of the tension (flat-slab Kirkwood-Buff virial, Laplace pressure) read well below the dynamic
    one.
- Drops below R/h = 5 are qualitative. DC5's 12-30 uL spill at r = 0.4 mm has R/h ~ 1.
"""

from __future__ import annotations

import dataclasses
import logging
import math

import numpy as np
import warp as wp

from .kernels import kernel_grad, kernel_w_np
from .pressure import Particles, is_fluid, nbr_base, nbr_j

log = logging.getLogger("alloy.sph")

ADH_TABLE_N = 128  # samples of g(d/h) on [0, 1]
ADH_SCALE = math.pi * 0.007
RAYLEIGH_FACTOR = 1.55  # fitted on omega_0 / Rayleigh at R/h = 5-10 (module doc, wp9-results.md)


@wp.func
def cohesion_c(r: float, h: float) -> float:
    if r > h or r <= 0.0:
        return 0.0
    k = 32.0 / (wp.pi * wp.pow(h, 9.0))
    s = (h - r) * (h - r) * (h - r) * r * r * r
    if r > 0.5 * h:
        return k * s
    h3 = h * h * h
    return k * (2.0 * s - h3 * h3 / 64.0)


@wp.func
def curvature_weight(r: float, h: float) -> float:
    """s(r) = 4 (1 - r/h): the curvature pair term vanishes continuously at the support edge (module doc)."""
    return 4.0 * wp.max(1.0 - r / h, 0.0)


@wp.func
def adhesion_g(d: float, h: float, table: wp.array[wp.float32]) -> float:
    """pi 0.007 g(d/h) (linear in the table), 0 beyond h; walls closer than 0 count as 0."""
    u = wp.clamp(d / h, 0.0, 1.0) * float(ADH_TABLE_N)
    k = wp.min(int(u), ADH_TABLE_N - 1)
    t = u - float(k)
    return ADH_SCALE * ((1.0 - t) * table[k] + t * table[k + 1])


@wp.kernel(enable_backward=False)
def normals_kernel(p: Particles, ratio: wp.array[wp.float32], h: float, kind: int, normals: wp.array[wp.vec3]):
    i = wp.tid()
    if not is_fluid(p, i):
        normals[i] = wp.vec3(0.0, 0.0, 0.0)
        return
    xi = p.q[i]
    n = wp.vec3(0.0, 0.0, 0.0)
    b_ = nbr_base(p, i)
    for k in range(p.count[i]):
        j = nbr_j(p, b_, i, k)
        rho_j = ratio[j] * p.rho0[j]
        if rho_j > 0.0:
            n += (p.mass[j] / rho_j) * kernel_grad(xi - p.q[j], h, kind)
    normals[i] = h * n


@wp.func
def _wall_normal(x: wp.vec3, xb: wp.vec3, phi: float, gpsi: wp.vec3, h: float) -> wp.vec3:
    """Outward wall normal at the slot (x - x_b = phi n); from the volume gradient where phi ~ 0."""
    if wp.abs(phi) > 1.0e-6 * h:
        return (x - xb) / phi
    lg = wp.length(gpsi)
    if lg > 0.0:
        return -gpsi / lg
    return wp.vec3(0.0, 0.0, 0.0)


@wp.kernel(enable_backward=False)
def tension_kernel(
    p: Particles, ratio: wp.array[wp.float32], normals: wp.array[wp.vec3], gamma: wp.array[wp.float32],
    beta: wp.array[wp.float32], slot_shape: wp.array2d[wp.int32], slot_phi: wp.array2d[wp.float32],
    slot_xb: wp.array2d[wp.vec3], slot_gpsi: wp.array2d[wp.vec3], table: wp.array[wp.float32], h: float, dt: float,
    slot_impulse: wp.array2d[wp.vec3],
):
    """v_i += dt a_i (cohesion, curvature, adhesion); reads positions only, so the in-place update is race-free."""
    i = wp.tid()
    if not is_fluid(p, i):
        return
    xi = p.q[i]
    ni = normals[i]
    rho_i = ratio[i] * p.rho0[i]
    a = wp.vec3(0.0, 0.0, 0.0)
    b_ = nbr_base(p, i)
    for k in range(p.count[i]):
        j = nbr_j(p, b_, i, k)
        g = 0.5 * (gamma[i] + gamma[j])
        rho_j = ratio[j] * p.rho0[j]
        if g == 0.0 or rho_i + rho_j <= 0.0:
            continue
        rv = xi - p.q[j]
        r = wp.length(rv)
        acc = curvature_weight(r, h) * (ni - normals[j])
        if r > 1.0e-6 * h:
            acc += (p.mass[j] * cohesion_c(r, h) / r) * rv
        a -= (2.0 * p.rho0[i] / (rho_i + rho_j) * g) * acc
    b = beta[i]
    if b > 0.0:
        for s in range(4):
            if slot_shape[i, s] >= 0:
                phi = slot_phi[i, s]
                n = _wall_normal(xi, slot_xb[i, s], phi, slot_gpsi[i, s], h)
                da = -(b * p.rho0[i] * adhesion_g(phi, h, table)) * n
                a += da
                slot_impulse[i, s] = slot_impulse[i, s] - (p.mass[i] * dt) * da
    p.qd[i] = p.qd[i] + dt * a


# ------------------------------------------------------------------------------------------------- host helpers


def cohesion_c_np(r: np.ndarray, h: float) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    out = np.zeros_like(r)
    k = 32.0 / (math.pi * h**9)
    s = (h - r) ** 3 * r**3
    far = (r > 0.5 * h) & (r <= h)
    near = (r > 0.0) & (r <= 0.5 * h)
    out[far] = k * s[far]
    out[near] = k * (2.0 * s[near] - h**6 / 64.0)
    return out


def adhesion_table(n: int = ADH_TABLE_N) -> np.ndarray:
    """g(delta) = int_{max(delta, 1/2)}^1 (-4u^2 + 6u - 2)^(1/4) (u^2 - delta^2) du at delta = k/n, k = 0..n."""
    out = np.zeros(n + 1)
    for k in range(n + 1):
        lo = max(k / n, 0.5)
        if lo >= 1.0:
            continue
        u = lo + (1.0 - lo) * (np.arange(4000) + 0.5) / 4000.0  # midpoint rule; the 1/4 power is integrable
        f = np.maximum(-4.0 * u * u + 6.0 * u - 2.0, 0.0) ** 0.25 * (u * u - (k / n) ** 2)
        out[k] = float(f.sum() * (1.0 - lo) / 4000.0)
    return out


def _lattice(spacing: float, h: float) -> np.ndarray:
    n = int(math.ceil(h / spacing)) + 1
    ax = np.arange(-n, n + 1) * spacing
    g = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
    r = np.linalg.norm(g, axis=1)
    return g[(r < h) & (r > 0.0)]


def _slab_pairs(spacing: float, h: float, kind: str, rho0: float, mass: float, layers: int):
    """Per layer l < ``layers`` of the rest lattice's flat free surface (fluid on z = -l spacing, l >= 0): the
    offsets x_j - x_i to its existing neighbours and the per-unit-gamma pair forces on i (cohesion, curvature z)."""
    off = _lattice(spacing, h)  # x_j - x_i
    rr = np.linalg.norm(off, axis=1)
    dl = np.round(off[:, 2] / spacing).astype(int)  # layer index of j = l - dl
    depth = layers + 2 * int(math.ceil(h / spacing)) + 2
    w0 = float(kernel_w_np(np.zeros(1), h, kind)[0])
    rho = np.array([mass * (w0 + kernel_w_np(rr[l - dl >= 0], h, kind).sum()) for l in range(depth)])
    eps = 1.0e-7 * h
    dw = (kernel_w_np(rr + eps, h, kind) - kernel_w_np(rr - eps, h, kind)) / (2.0 * eps)
    grad_z = dw * (-off[:, 2]) / rr  # z of dW(x_i - x_j)
    nz = np.zeros(depth)
    for l in range(depth - int(math.ceil(h / spacing)) - 1):
        ok = l - dl >= 0
        nz[l] = h * np.sum(mass / rho[l - dl[ok]] * grad_z[ok])
    out = []
    for l in range(layers):
        ok = l - dl >= 0
        o, r, lj = off[ok], rr[ok], l - dl[ok]
        K = 2.0 * rho0 / (rho[l] + rho[lj])
        f_coh = (K * mass * mass * cohesion_c_np(r, h) / r)[:, None] * o  # -m^2 C x_ij / r, x_ij = -o
        f_curv_z = -K * mass * (nz[l] - nz[lj]) * 4.0 * np.maximum(1.0 - r / h, 0.0)
        out.append((o, f_coh, f_curv_z))
    return out


def layer_forces_per_gamma(spacing: float, h: float, kind: str, rho0: float, mass: float,
                           layers: int = 4) -> np.ndarray:
    """Net z force [N per unit gamma] on a particle of each top layer of the rest lattice's flat free surface."""
    return np.array([float(np.sum(f_coh[:, 2] + f_curv)) for _o, f_coh, f_curv in
                     _slab_pairs(spacing, h, kind, rho0, mass, layers)])


def lattice_sigma_per_gamma(spacing: float, h: float, kind: str, rho0: float, mass: float, layers: int = 8) -> float:
    """Kirkwood-Buff surface tension [N/m] per unit gamma of the rest lattice's flat free surface (module doc).
    Bulk pairs cancel by cubic symmetry, so ``layers`` > h / spacing + 2 suffices."""
    sigma = 0.0
    for o, f_coh, f_curv_z in _slab_pairs(spacing, h, kind, rho0, mass, layers):
        vir = -o[:, 2] * (f_coh[:, 2] + f_curv_z) - 0.5 * (-o[:, 0] * f_coh[:, 0] - o[:, 1] * f_coh[:, 1])
        sigma += 0.5 * float(vir.sum()) / spacing**2
    return sigma


def adhesion_work_per_beta(spacing: float, h: float, rho0: float, mass: float) -> float:
    """Work of adhesion [J/m^2] per unit beta of the rest lattice next to a flat wall (layers at d = r, 3r, ...):
    sum_l (m rho0 / spacing^2) int_{d_l}^h G(s) ds."""
    table = adhesion_table()
    s = np.linspace(0.0, h, len(table))
    G = ADH_SCALE * table
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (G[1:] + G[:-1]) * np.diff(s))])  # int_0^s G
    total = 0.0
    d = 0.5 * spacing
    while d < h:
        total += cum[-1] - np.interp(d, s, cum)
        d += spacing
    return mass * rho0 / spacing**2 * total


def capillary_dt(spacing: float, rho0: float, sigma: float) -> float:
    """Capillary step limit sqrt(rho dx^3 / (2 pi sigma)) [s] (inf without tension): Brackbill et al. 1992's bound
    with the particle spacing for the length, (dx / h)^1.5 = 0.35 of it with the kernel support h. Measured (csf): at
    the support-radius bound a wetting sessile drop blows up and a free drop's Laplace pressure is 10 % high."""
    return math.inf if sigma <= 0.0 else math.sqrt(rho0 * spacing**3 / (2.0 * math.pi * sigma))


class TensionBase:
    """The capillary step check and the report fields shared by the tension models."""

    NAME = "none"
    QUALITATIVE = True
    dt_limit = math.inf
    rh_range: tuple[float, float] | None = None
    violations = 0

    def check_dt(self, dt: float) -> None:
        """Count (and warn once about) a step above the capillary limit; never silent (SCR-WP9-7 condition 1)."""
        if dt <= self.dt_limit * (1.0 + 1.0e-6):
            return
        if self.violations == 0:
            log.warning("SPH %s: dt %.3g s is above the capillary step limit %.3g s; the tension laws do not cover "
                        "it (SolverDFSPH.stable_dt stays below the limit)", self.NAME, dt, self.dt_limit)
        self.violations += 1

    def annotate(self, rep):
        return dataclasses.replace(rep, tension=self.NAME, tension_qualitative=self.QUALITATIVE,
                                   tension_rh_range=self.rh_range, capillary_dt_violations=self.violations)


class AkinciTension(TensionBase):
    """Per-particle gamma and beta from the physical inputs, and the launches of one step."""

    NAME = "akinci2013"

    def __init__(self, model, layout, cfg, device):
        n = layout.n
        self.device = device
        if cfg.kernel != "cubic" or abs(cfg.support_factor - 4.0) > 1.0e-9:
            log.warning("SPH akinci2013: RAYLEIGH_FACTOR was measured for the cubic kernel at h = 4r; kernel %s, "
                        "h = %gr is uncalibrated", cfg.kernel, cfg.support_factor)
        sigma = model.sph.surface_tension.numpy()[:n].astype(np.float64)
        adh = model.sph.adhesion.numpy()[:n].astype(np.float64)
        rho0 = layout.rest_density.numpy()[:n].astype(np.float64)
        mass = model.particle_mass.numpy()[:n].astype(np.float64)
        gamma, beta = np.zeros(max(n, 1), np.float32), np.zeros(max(n, 1), np.float32)
        sph = layout.phase.numpy()[:n] >= 0
        for r0, m in {(float(a), float(b)) for a, b in zip(rho0[sph], mass[sph], strict=True)}:
            sel = sph & (rho0 == r0) & (mass == m)
            c_sigma = RAYLEIGH_FACTOR * lattice_sigma_per_gamma(cfg.spacing, cfg.h, cfg.kernel, r0, m)
            c_adh = adhesion_work_per_beta(cfg.spacing, cfg.h, r0, m)
            gamma[:n][sel] = sigma[sel] / c_sigma
            beta[:n][sel] = adh[sel] * sigma[sel] / c_adh
            log.info("SPH akinci2013: rho0 %g, sigma/gamma %.4g N/m, W_adh/beta %.4g J/m^2", r0, c_sigma, c_adh)
        self.gamma = wp.array(gamma, dtype=wp.float32, device=device)
        self.beta = wp.array(beta, dtype=wp.float32, device=device)
        self.table = wp.array(adhesion_table().astype(np.float32), dtype=wp.float32, device=device)
        self.normals = wp.zeros(max(n, 1), dtype=wp.vec3, device=device)
        self.n = n
        self.dt_limit = min((capillary_dt(cfg.spacing, float(r), float(s)) for r, s in zip(rho0[sph], sigma[sph],
                                                                                            strict=True)),
                            default=math.inf)

    def apply(self, p: Particles, slots, ratio: wp.array, h: float, kind: int, dt: float,
              slot_impulse: wp.array) -> None:
        if self.n == 0:
            return
        wp.launch(normals_kernel, dim=self.n, inputs=[p, ratio, h, kind, self.normals], device=self.device)
        wp.launch(tension_kernel, dim=self.n, device=self.device,
                  inputs=[p, ratio, self.normals, self.gamma, self.beta, slots.shape, slots.phi, slots.xb,
                          slots.gpsi, self.table, h, dt, slot_impulse])
