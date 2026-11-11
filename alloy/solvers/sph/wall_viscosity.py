# SPDX-License-Identifier: Apache-2.0 AND MIT
# Portions ported from SPlisHSPlasH (MIT, Copyright (c) 2016 Jan Bender); see NOTICE.
"""Wall (boundary) viscosity for the Weiler 2018 implicit solve (SCR-SC1-1): no-slip / partial-slip walls.

Port of the volume-map boundary term of SPlisHSPlasH@dba6b26 (MIT, Copyright (c) 2016 Jan Bender):
``SPlisHSPlasH/Viscosity/Viscosity_Weiler2018.cpp:205-260`` (Bender2019 branch) with the boundary point and volume of
``SPlisHSPlasH/TimeStep.cpp:274-297``. For every boundary slot k of particle i (``boundary.py``: closest surface
point x_b, signed distance phi, boundary volume fraction psi):

- the representative boundary point x_k = x_i - d n with d = max(phi + r/2, 2r) and n the unit normal from the wall
  to the particle (TimeStep.cpp:293), and its volume V_k = psi_k / W(d) (psi = V W(|x_i - x_k|), TimeStep.cpp
  density sum ``density += Vj * W(xi - xj)``);
- four samples x_s = x_k +- c h t1, x_k +- c h t2 (t1, t2 tangent, c = 0.5 = SPlisHSPlasH's
  ``m_tangentialDistanceFactor``), each with V_k / 4 and the wall's own point velocity v_s (rigid twist);
- the term d_lap mu_b (V_k/4) (x_is . (v_i - v_s)) / (|x_is|^2 + 0.01 h^2) gradW(x_is).

In the solve A v = b it splits into a per-particle 3x3 block M_i = sum_s c_s gradW_is x_is^T (operator diagonal,
positive semi-definite, so A stays SPD and the block-Jacobi preconditioner takes it too) and a right-hand side
w_i = sum_s c_s (x_is . v_s) gradW_is:  (A v)_i -= dt/rho_i M_i v_i,  b_i = v*_i - dt/rho_i w_i. After the solve
the impulse the liquid received from each slot, m_i dt/rho_i (M_ik v_i - w_ik), goes to the slot with the opposite
sign (``slot_impulse`` holds impulses on the walls), so bodies feel the drag and ``alloy_shape_work`` counts the work.

Calibration (ours, like ``kernels.laplacian_lattice_factor`` for the bulk): with SPlisHSPlasH's coefficient d mu_b
only the first lattice layer (phi = r) reaches the samples and it takes 0.19 of the no-slip wall stress (cubic,
h = 4r), so a honey film on a wall slipped at ~7x its Nusselt speed. ``wall_coefficient_scale`` sets the coefficient
so that a first-layer particle of the rest lattice feels exactly the no-slip stress mu_b u / r on its (2r)^2 face,
i.e. M_tt = -mu_b / (2 r^2). Then mu_b = the liquid's mu gives no-slip walls; smaller values give partial slip.

SCR-SC1-3 (ours): the sample term above weakens ~8x between phi = r and 2r (|x_is| goes from 0.71 h to 0.90 h, where
the kernel gradient is small), and WP9's pressure boundary tapers psi to 0 by 3r, so a first layer that is not
exactly at r (a flowed or curved-wall layer) slipped: a honey coat on a rod drained 3x its rings-seeded speed at once.
Now (1) the volume is the untapered half-space volume (the volume map of Bender 2019, all of the support; slots reach
h when wall viscosity is on), (2) the tangential block is the stress it is calibrated to, M_tt = -mu_b / (max(phi, r)
2r), for an uneven first layer (full to 2r, faded out by 3r: the lattice's second layer at 3r is coupled through
the fluid), (3) the normal block keeps the sample form, and w = M v_b(x_b) (a wall moving with the liquid exerts no
impulse). At phi = r this is the calibration above, so rest-lattice results are unchanged.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from .boundary import point_velocity
from .kernels import _half_lattice, halfspace_volume, kernel_grad, kernel_w, kernel_w_np
from .pressure import Particles, is_fluid

TANGENTIAL_FACTOR = 0.5  # SPlisHSPlasH Viscosity_Weiler2018.cpp:29
TANGENT_FULL, TANGENT_ZERO = 2.0, 3.0  # x r: full no-slip block for an uneven first layer, faded out (SCR-SC1-3)


@wp.struct
class WallSlots:
    shape: wp.array2d[wp.int32]
    psi: wp.array2d[wp.float32]
    xb: wp.array2d[wp.vec3]
    phi: wp.array2d[wp.float32]
    shape_body: wp.array[wp.int32]
    body_q: wp.array[wp.transform]
    body_qd: wp.array[wp.spatial_vector]
    body_com: wp.array[wp.vec3]
    offset: float  # pressure boundary fit (ShapeArrays.offset/amplitude/taper0)
    amplitude: float
    taper0: float
    mu_b: float  # wall viscosity [Pa s]
    vshift: wp.array[wp.vec3]  # per particle: added to the wall velocity the term sees (the pressure guess, SCR-WP9-11)
    has_shift: int
    vb: wp.array2d[wp.vec3]  # slot wall velocities: read for soft faces (shape >= soft_base, soft_walls.py)
    soft_base: int


def wall_coefficient_scale(spacing: float, h: float, kind: str = "cubic") -> float:
    """Coefficient per unit mu_b [1/m^0, so coeff = mu_b * scale] making the wall block of a first-layer particle of
    the rest lattice (phi = r, boundary volume = the lattice deficit) equal -mu_b / (2 r^2) tangentially."""
    r = 0.5 * spacing
    psi = _half_lattice(spacing, h, kind)[0]
    d = max(r + 0.5 * r, 2.0 * r)
    wd = float(kernel_w_np(np.array([d]), h, kind)[0])
    rho = math.hypot(d, TANGENTIAL_FACTOR * h)
    eps = 1.0e-7 * h
    dw = float((kernel_w_np(np.array([rho + eps]), h, kind) - kernel_w_np(np.array([rho - eps]), h, kind))[0]) / (2 * eps)
    vol = 0.25 * psi / wd
    m_tt = 2.0 * vol / (rho * rho + 0.01 * h * h) * dw * (TANGENTIAL_FACTOR * h) ** 2 / rho  # unit coefficient
    if not m_tt < 0.0:
        raise ValueError(f"wall_coefficient_scale: no wall term reaches the first layer (h={h}, spacing={spacing})")
    return 1.0 / (2.0 * r * r * abs(m_tt))


def wall_slots(slots, shapes, body_q: wp.array, body_qd: wp.array, body_com: wp.array, cfg, vshift: wp.array,
               has_shift: int) -> WallSlots:
    """WallSlots view of the solver's BoundarySlots, ShapeArrays and body state (no copies); ``vshift`` (used when
    ``has_shift``) is added to each particle's wall velocities."""
    ws = WallSlots()
    ws.shape, ws.psi, ws.xb, ws.phi = slots.shape, slots.psi, slots.xb, slots.phi
    ws.shape_body, ws.body_q, ws.body_qd, ws.body_com = shapes.body, body_q, body_qd, body_com
    ws.offset, ws.amplitude, ws.taper0 = float(shapes.offset), float(shapes.amplitude), float(shapes.taper0)
    ws.mu_b = float(cfg.boundary_viscosity or 0.0)
    ws.vshift, ws.has_shift = vshift, int(has_shift)
    ws.vb, ws.soft_base = slots.vb, int(shapes.soft.base)
    return ws


@wp.func
def _tangents(n: wp.vec3) -> wp.mat33:
    """Rows t1, t2 orthonormal to the unit vector n (third row n)."""
    a = wp.vec3(1.0, 0.0, 0.0)
    if wp.abs(n[0]) > 0.6:
        a = wp.vec3(0.0, 1.0, 0.0)
    t1 = wp.normalize(wp.cross(n, a))
    t2 = wp.cross(n, t1)
    return wp.mat33(t1[0], t1[1], t1[2], t2[0], t2[1], t2[2], n[0], n[1], n[2])


@wp.func
def slot_pair(ws: WallSlots, i: int, k: int, xi: wp.vec3, r: float, h: float, kind: int, coeff: float):
    """(M_ik, w_ik) of slot k: normal block from the samples (M = sum_s c_s gradW_is x_is^T, coeff = d mu_b on the
    untapered volume), tangential block -mu_b / (max(phi, r) 2r) faded from 2r to 3r, w = M v_b(x_b), v_b plus the
    particle's shift when set."""
    m = wp.mat33(0.0)
    w = wp.vec3(0.0)
    s = ws.shape[i, k]
    if s < 0:
        return m, w
    phi = ws.phi[i, k]
    d = wp.max(phi + 0.5 * r, 2.0 * r)  # TimeStep.cpp:293
    wd = kernel_w(d, h, kind)
    n = xi - ws.xb[i, k]
    ln = wp.length(n)
    psi = ws.psi[i, k]
    if phi > ws.taper0:  # the pressure boundary's taper is not part of the volume map (SCR-SC1-3)
        psi = halfspace_volume(phi + ws.offset, h, 0.0, kind)[0] * ws.amplitude
    if wd <= 0.0 or psi <= 0.0 or ln < 1.0e-9:
        return m, w
    n = n / ln
    xk = xi - d * n
    tb = _tangents(n)
    vol = 0.25 * psi / wd
    for q in range(4):
        sg = 1.0 - 2.0 * float(q % 2)
        row = q // 2
        t = wp.vec3(tb[row, 0], tb[row, 1], tb[row, 2])
        xs = xk + (sg * TANGENTIAL_FACTOR * h) * t
        xis = xi - xs
        g = kernel_grad(xis, h, kind)
        c = coeff * vol / (wp.dot(xis, xis) + 0.01 * h * h)
        m += c * wp.outer(g, xis)
    u = wp.clamp((phi - TANGENT_FULL * r) / ((TANGENT_ZERO - TANGENT_FULL) * r), 0.0, 1.0)
    m_tt = -ws.mu_b / (wp.max(phi, r) * 2.0 * r) * (1.0 - u * u * (3.0 - 2.0 * u))
    nn = wp.outer(n, n)
    m = wp.dot(n, m * n) * nn + m_tt * (wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) - nn)
    vb = wp.vec3(0.0, 0.0, 0.0)
    if s >= ws.soft_base:
        vb = ws.vb[i, k]
    else:
        vb = point_velocity(ws.shape_body[s], ws.xb[i, k], ws.body_q, ws.body_qd, ws.body_com)
    if ws.has_shift != 0:
        vb = vb + ws.vshift[i]
    w = m * vb
    return m, w


@wp.kernel(enable_backward=False)
def wall_terms_kernel(p: Particles, ws: WallSlots, r: float, h: float, kind: int, coeff: float,
                      wall_m: wp.array[wp.mat33], wall_w: wp.array[wp.vec3]):
    """Per particle M_i = sum_k M_ik and w_i = sum_k w_ik (coeff = d_lap mu_b)."""
    i = wp.tid()
    m = wp.mat33(0.0)
    w = wp.vec3(0.0)
    if is_fluid(p, i):
        xi = p.q[i]
        for k in range(4):
            mk, wk = slot_pair(ws, i, k, xi, r, h, kind, coeff)
            m += mk
            w += wk
    wall_m[i] = m
    wall_w[i] = w


@wp.kernel(enable_backward=False)
def wall_reaction_kernel(p: Particles, ws: WallSlots, ratio: wp.array[wp.float32], r: float, h: float, kind: int,
                         coeff: float, dt: float, slot_impulse: wp.array2d[wp.vec3]):
    """Impulse on each slot's wall: -(m_i dt / rho_i)(M_ik v_i - w_ik), v_i the solved velocity."""
    i = wp.tid()
    if not is_fluid(p, i):
        return
    xi = p.q[i]
    vi = p.qd[i]
    rho_i = wp.max(ratio[i] * p.rho0[i], 1.0e-6)
    f = p.mass[i] * dt / rho_i
    for k in range(4):
        if ws.shape[i, k] >= 0:
            mk, wk = slot_pair(ws, i, k, xi, r, h, kind, coeff)
            slot_impulse[i, k] = slot_impulse[i, k] - f * (mk * vi - wk)


__all__ = ["TANGENTIAL_FACTOR", "TANGENT_FULL", "TANGENT_ZERO", "WallSlots", "slot_pair", "wall_coefficient_scale",
           "wall_reaction_kernel", "wall_slots", "wall_terms_kernel"]
