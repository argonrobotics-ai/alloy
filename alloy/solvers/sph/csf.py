# SPDX-License-Identifier: Apache-2.0
"""Continuum surface force with a wall contact angle for SolverDFSPH (``surface_tension="csf"``, SCR-WP9-7).

Method (single-phase free surface):
- colour gradient g_i = sum_j V_j gradW_ij + 1/2 sum_b gradPsi_b: the liquid's volume fraction gradient, the wall
  slots counting as liquid (the fitted wall volume's slope is twice the half-lattice gradient, kernels.boundary_fit),
  so a wall-wetted layer is not a free surface. |g| h > SURFACE_GRAD marks the surface particles; interior
  particles get no force.
- normal n = -<g> / |<g>|, <g> the colour gradient Shepard-smoothed NORMAL_PASSES + 1 times (a particle-deficient
  surface layer has a biased raw gradient; the smoothed one points along the macroscopic normal).
- contact angle (Breinlinger et al. 2013): at a surface particle whose nearest slot's shape has
  ``sph:shape_contact_angle`` theta >= 0, n is turned to cos(theta) n_w + sin(theta) t (n_w the wall normal into the
  liquid, t the tangential part of n). theta = -1 (default) leaves a free contact line. The pressure here is one-sided
  (it cannot pull), so the wall-normal part of the tension force that points away from the wall is taken by the wall
  (its reaction goes to the slot impulse); without that a wetting drop (theta < 90 deg) lifts off.
- curvature kappa = div_s n by a renormalised (least-squares) tangential divergence over the surface neighbours,
  then Shepard-smoothed over the surface CURVATURE_PASSES times, then moved from the particle centre to the
  interface (depth d = (rho/rho0 - c_s) / |g|: kappa / (1 + kappa d / 2)).
- force a_i = -sigma kappa |g| n / (rho_i I), I = the flat-lattice integral of |g| over the surface layers (the
  delta-function normalisation; flat_lattice_constants). No fitted constant.

Brackbill, Kothe & Zemach 1992 (CSF, capillary step bound); Morris 2000 and Adami et al. 2010 (SPH form);
Breinlinger, Polfer, Hashibon & Kraft 2013 (contact angle). Written from the papers; no third-party code.

Measured (wp9-results.md, SCR-WP9-7), no fitted constant:
- water (viscosity "xsph", run at xsph_c >= XSPH_MIN): Laplace pressure and the Rayleigh l = 2 frequency within 10 %
  at R/h 3.5, 5 and 7 (RH_RANGE_XSPH); at R/h 10 Laplace +8 %, Rayleigh +10.2 %.
- viscous (weiler2018, 0.05 Pa s): both within 10 % at R/h 7 and 10 (RH_RANGE_VISCOUS); the l = 2 amplitude decays to
  the noise floor around round (no permanent set) at R/h 3.5-10.
- sessile contact angle within 5 deg at 60, 90 and 120 deg (R/h 5 and 7).
Limits: an inviscid (xsph) drop's l = 2 amplitude grows (+18 % to x2.4 over 15 periods at R/h 7 to 3.5): the scheme
pumps the mode, so water drops do not settle back to round; 30 deg does not hold (the film is about 2 h thick);
below the R/h range the force is qualitative (the solver report gives the range).
"""

from __future__ import annotations

import logging
import math

import numpy as np
import warp as wp

from .config import resolve_viscosity
from .kernels import kernel_grad, kernel_w, kernel_w_np
from .pressure import Particles, is_fluid, nbr_base, nbr_j
from .tension import AkinciTension, TensionBase, capillary_dt

log = logging.getLogger("alloy.sph")

SURFACE_GRAD = 0.3  # |g| h above this marks a surface particle (a flat lattice: layers 0 and 1)
NORMAL_PASSES = 2  # extra Shepard passes of the colour gradient for the normal
CURVATURE_PASSES = 3  # Shepard passes of the curvature over the surface
XSPH_MIN = 0.3  # with viscosity "xsph", tension runs at xsph_c >= this (smaller values let the surface heat up)
RH_RANGE_XSPH = (3.5, 7.0)  # R/h range of the laws, viscosity "xsph" (water)
RH_RANGE_VISCOUS = (7.0, 10.0)  # R/h range of the laws, viscosity "weiler2018"
_THR = wp.constant(SURFACE_GRAD)


@wp.kernel(enable_backward=False)
def grad_kernel(p: Particles, slot_shape: wp.array2d[wp.int32], slot_gpsi: wp.array2d[wp.vec3], h: float, kind: int,
                grad: wp.array[wp.vec3]):
    i = wp.tid()
    grad[i] = wp.vec3(0.0, 0.0, 0.0)
    if not is_fluid(p, i):
        return
    xi = p.q[i]
    inv = 1.0 / p.rho0[i]
    g = wp.vec3(0.0, 0.0, 0.0)
    b_ = nbr_base(p, i)
    for k in range(p.count[i]):
        j = nbr_j(p, b_, i, k)
        g += (p.mass[j] * inv) * kernel_grad(xi - p.q[j], h, kind)
    for b in range(4):
        if slot_shape[i, b] >= 0:
            g += 0.5 * slot_gpsi[i, b]
    grad[i] = g


@wp.kernel(enable_backward=False)
def shepard_vec_kernel(p: Particles, h: float, kind: int, v_in: wp.array[wp.vec3], v_out: wp.array[wp.vec3]):
    """Shepard mean of v over the fluid neighbours (self included)."""
    i = wp.tid()
    v_out[i] = v_in[i]
    if not is_fluid(p, i):
        return
    xi = p.q[i]
    w0 = kernel_w(0.0, h, kind)
    s = w0 * v_in[i]
    den = w0
    b_ = nbr_base(p, i)
    for k in range(p.count[i]):
        j = nbr_j(p, b_, i, k)
        w = kernel_w(wp.length(xi - p.q[j]), h, kind)
        s += w * v_in[j]
        den += w
    v_out[i] = s / den


@wp.kernel(enable_backward=False)
def normal_kernel(p: Particles, grad: wp.array[wp.vec3], gdir: wp.array[wp.vec3], slot_shape: wp.array2d[wp.int32],
                  slot_phi: wp.array2d[wp.float32], slot_xb: wp.array2d[wp.vec3], h: float,
                  shape_cos: wp.array[wp.float32], shape_sin: wp.array[wp.float32], nhat: wp.array[wp.vec3],
                  surf: wp.array[wp.int32], nwall: wp.array[wp.vec3], wslot: wp.array[wp.int32]):
    """Surface flag, normal, and the Breinlinger turn at a wall with a contact angle (module doc)."""
    i = wp.tid()
    nhat[i] = wp.vec3(0.0, 0.0, 0.0)
    nwall[i] = wp.vec3(0.0, 0.0, 0.0)
    wslot[i] = -1
    surf[i] = 0
    lg = wp.length(gdir[i])
    if not is_fluid(p, i) or wp.length(grad[i]) * h <= _THR or lg <= 0.0:
        return
    n = -gdir[i] / lg
    best = int(-1)
    dmin = float(1.0e30)
    for b in range(4):
        if slot_shape[i, b] >= 0 and slot_phi[i, b] < dmin:
            dmin = slot_phi[i, b]
            best = b
    if best >= 0 and slot_shape[i, best] < shape_sin.shape[0] and shape_sin[slot_shape[i, best]] >= 0.0:
        nw = p.q[i] - slot_xb[i, best]
        lw = wp.length(nw)
        if lw > 1.0e-9:
            nw = nw / lw
            if dmin < 0.0:
                nw = -nw
            t = n - wp.dot(n, nw) * nw
            lt = wp.length(t)
            if lt > 1.0e-6:
                n = shape_cos[slot_shape[i, best]] * nw + (shape_sin[slot_shape[i, best]] / lt) * t
                nwall[i] = nw
                wslot[i] = best
    nhat[i] = n
    surf[i] = 1


@wp.kernel(enable_backward=False)
def curvature_kernel(p: Particles, nhat: wp.array[wp.vec3], surf: wp.array[wp.int32], h: float, kind: int,
                     kappa: wp.array[wp.float32]):
    """kappa = trace(B A^-1): the least-squares tangential gradient of n over the surface neighbours (renormalised
    for the one-sided neighbourhood), A = sum w t t^T, B = sum w (P dn) t^T, t = P (x_j - x_i), P = I - n n^T."""
    i = wp.tid()
    kappa[i] = 0.0
    if surf[i] == 0:
        return
    xi = p.q[i]
    ni = nhat[i]
    P = wp.identity(n=3, dtype=float) - wp.outer(ni, ni)
    A = wp.mat33(0.0)
    B = wp.mat33(0.0)
    cnt = int(0)
    b_ = nbr_base(p, i)
    for k in range(p.count[i]):
        j = nbr_j(p, b_, i, k)
        if surf[j] != 0:
            w = kernel_w(wp.length(xi - p.q[j]), h, kind)
            t = P * (p.q[j] - xi)
            A += w * wp.outer(t, t)
            B += w * wp.outer(P * (nhat[j] - ni), t)
            cnt += 1
    scale = wp.trace(A)
    if cnt < 4 or scale <= 0.0:
        return
    A += (1.0e-2 * scale) * wp.outer(ni, ni)  # A is rank 2 (tangent plane)
    kappa[i] = wp.trace(B * wp.inverse(A))


@wp.kernel(enable_backward=False)
def shepard_surface_kernel(p: Particles, surf: wp.array[wp.int32], h: float, kind: int, k_in: wp.array[wp.float32],
                           k_out: wp.array[wp.float32]):
    """Shepard mean of the curvature over the surface neighbours (self included)."""
    i = wp.tid()
    k_out[i] = k_in[i]
    if surf[i] == 0:
        return
    xi = p.q[i]
    w0 = kernel_w(0.0, h, kind)
    num = w0 * k_in[i]
    den = w0
    b_ = nbr_base(p, i)
    for k in range(p.count[i]):
        j = nbr_j(p, b_, i, k)
        if surf[j] != 0:
            w = kernel_w(wp.length(xi - p.q[j]), h, kind)
            num += w * k_in[j]
            den += w
    k_out[i] = num / den


@wp.kernel(enable_backward=False)
def csf_kernel(p: Particles, ratio: wp.array[wp.float32], grad: wp.array[wp.vec3], nhat: wp.array[wp.vec3],
               nwall: wp.array[wp.vec3], wslot: wp.array[wp.int32], surf: wp.array[wp.int32],
               kappa_in: wp.array[wp.float32], sigma: wp.array[wp.float32], c_surface: float, inv_i: float, dt: float,
               kappa_out: wp.array[wp.float32], slot_impulse: wp.array2d[wp.vec3]):
    """v_i += dt a_i (module doc); the away-from-wall part at a contact line goes to the wall's slot impulse."""
    i = wp.tid()
    kappa_out[i] = 0.0
    if surf[i] == 0 or sigma[i] == 0.0:
        return
    kap = kappa_in[i]
    lg = wp.length(grad[i])
    corr = 1.0 + 0.5 * kap * (ratio[i] - c_surface) / lg  # centre -> interface (module doc)
    if corr > 0.5:
        kap = kap / corr
    kappa_out[i] = kap
    rho_i = wp.max(ratio[i] * p.rho0[i], 1.0e-6)
    dv = -(dt * sigma[i] * kap * lg * inv_i / rho_i) * nhat[i]
    b = wslot[i]
    if b >= 0:
        nw = nwall[i]
        dn = wp.dot(dv, nw)
        if dn > 0.0:
            dv = dv - dn * nw
            slot_impulse[i, b] = slot_impulse[i, b] + (p.mass[i] * dn) * nw
    p.qd[i] = p.qd[i] + dv


def flat_lattice_constants(spacing: float, h: float, kind: str = "cubic", layers: int = 6) -> tuple[float, float]:
    """(I, c_s) of a flat rest-lattice surface: I = sum over the surface layers (|g| h > SURFACE_GRAD) of |g|
    spacing, the CSF delta normalisation; c_s = rho/rho0 extrapolated from the top layer to the interface."""
    n = int(math.ceil(h / spacing)) + 1
    ax = np.arange(-n, n + 1) * spacing
    off = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
    rr = np.linalg.norm(off, axis=1)
    keep = rr < h
    off, rr = off[keep], rr[keep]
    dl = np.round(off[:, 2] / spacing).astype(int)
    w = kernel_w_np(rr, h, kind)
    vol = 1.0 / float(w.sum())
    e = 1.0e-7 * h
    nz = rr > 0
    dw = np.zeros_like(rr)
    dw[nz] = (kernel_w_np(rr[nz] + e, h, kind) - kernel_w_np(rr[nz] - e, h, kind)) / (2.0 * e)
    gz_unit = np.where(nz, dw * (-off[:, 2]) / np.where(nz, rr, 1.0), 0.0)
    total, c0, g0 = 0.0, 0.0, 0.0
    for layer in range(layers):
        ok = layer - dl >= 0
        gz = abs(float(np.sum(vol * gz_unit[ok])))
        if gz * h > SURFACE_GRAD:
            total += gz * spacing
        if layer == 0:
            c0, g0 = float(np.sum(vol * w[ok])), gz
    return total, c0 - 0.5 * spacing * g0


def shape_angles(model, device) -> tuple[wp.array, wp.array]:
    """(cos, sin) of ``sph:shape_contact_angle`` per shape; sin = -1 marks a free contact line (angle < 0)."""
    ang = getattr(getattr(model, "sph", None), "shape_contact_angle", None)
    ang = ang.numpy().astype(np.float64) if ang is not None else np.zeros(0)
    if len(ang) == 0:
        ang = np.full(1, -1.0)
    bad = (ang > math.pi + 1.0e-6) | ((ang < 0.0) & (ang != -1.0))
    if bad.any():
        raise ValueError(f"sph:shape_contact_angle must be -1 (free) or in [0, pi] rad; shapes {np.nonzero(bad)[0]}")
    free = ang < 0.0
    cos = np.where(free, 0.0, np.cos(ang)).astype(np.float32)
    sin = np.where(free, -1.0, np.sin(np.clip(ang, 0.0, math.pi))).astype(np.float32)
    return wp.array(cos, dtype=wp.float32, device=device), wp.array(sin, dtype=wp.float32, device=device)


class CSFTension(TensionBase):
    """Per-particle sigma, the flat-lattice constants, per-shape contact angles and the launches of one step."""

    NAME = "csf"
    QUALITATIVE = False

    def __init__(self, model, layout, cfg, device):
        n = layout.n
        viscous = resolve_viscosity(cfg, layout.viscosity_max) == "weiler2018"
        self.rh_range = RH_RANGE_VISCOUS if viscous else RH_RANGE_XSPH
        if not viscous and n:
            log.warning("SPH csf: low-viscosity liquid (viscosity 'xsph'): a free drop's l = 2 oscillation grows "
                        "instead of decaying (energy injection, open in SCR-WP9-7); quantitative for the shape, the "
                        "Laplace pressure and the first oscillation periods at R/h %.1f-%.1f only; larger drops are "
                        "outside the validated range", *RH_RANGE_XSPH)
        self.device, self.n = device, n
        sigma = model.sph.surface_tension.numpy()[:n].astype(np.float32) if n else np.zeros(1, np.float32)
        self.sigma = wp.array(sigma if n else np.zeros(1, np.float32), dtype=wp.float32, device=device)
        self.I, self.c_surface = flat_lattice_constants(cfg.spacing, cfg.h, cfg.kernel)
        self.shape_cos, self.shape_sin = shape_angles(model, device)
        z = lambda dt: wp.zeros(max(n, 1), dtype=dt, device=device)  # noqa: E731
        self.grad, self.g1, self.g2, self.nhat, self.nwall = (z(wp.vec3) for _ in range(5))
        self.kraw, self.ktmp, self.kappa = (z(wp.float32) for _ in range(3))
        self.surf, self.wslot = z(wp.int32), z(wp.int32)
        rho0 = layout.rest_density.numpy()[:n] if n else np.zeros(0)
        liquid = layout.phase.numpy()[:n] >= 0 if n else np.zeros(0, bool)
        self.dt_limit = min((capillary_dt(cfg.spacing, float(r), float(s)) for r, s in zip(rho0[liquid], sigma[liquid],
                                                                                         strict=True)),
                            default=math.inf)

    def apply(self, p: Particles, slots, ratio: wp.array, h: float, kind: int, dt: float,
              slot_impulse: wp.array) -> None:
        if self.n == 0:
            return
        d, n = self.device, self.n
        wp.launch(grad_kernel, dim=n, inputs=[p, slots.shape, slots.gpsi, h, kind, self.grad], device=d)
        src, dst = self.grad, self.g1
        for _ in range(NORMAL_PASSES + 1):
            wp.launch(shepard_vec_kernel, dim=n, inputs=[p, h, kind, src, dst], device=d)
            src, dst = dst, (self.g2 if dst is self.g1 else self.g1)
        wp.launch(normal_kernel, dim=n, device=d,
                  inputs=[p, self.grad, src, slots.shape, slots.phi, slots.xb, h, self.shape_cos, self.shape_sin,
                          self.nhat, self.surf, self.nwall, self.wslot])
        wp.launch(curvature_kernel, dim=n, inputs=[p, self.nhat, self.surf, h, kind, self.kraw], device=d)
        src, dst = self.kraw, self.ktmp
        for _ in range(CURVATURE_PASSES):
            wp.launch(shepard_surface_kernel, dim=n, inputs=[p, self.surf, h, kind, src, dst], device=d)
            src, dst = dst, src
        wp.launch(csf_kernel, dim=n, device=d,
                  inputs=[p, ratio, self.grad, self.nhat, self.nwall, self.wslot, self.surf, src, self.sigma,
                          self.c_surface, 1.0 / self.I, dt, self.kappa, slot_impulse])


def make_tension(model, layout, cfg, device) -> TensionBase | None:
    """The tension model Config.surface_tension names (None for "none")."""
    cls = {"akinci2013": AkinciTension, "csf": CSFTension}.get(cfg.surface_tension)
    return cls(model, layout, cfg, device) if cls is not None else None


__all__ = ["CSFTension", "CURVATURE_PASSES", "NORMAL_PASSES", "RH_RANGE_VISCOUS", "RH_RANGE_XSPH", "SURFACE_GRAD",
           "XSPH_MIN", "flat_lattice_constants", "make_tension", "shape_angles"]
