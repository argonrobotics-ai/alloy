# SPDX-License-Identifier: Apache-2.0
"""DFSPH density, factor, divergence and density solves (Bender & Koschier, TVCG 23(3) 2017, DOI
10.1109/TVCG.2016.2578335), matrix-free Jacobi with relaxation 1/2 as SPlisHSPlasH@dba6b26:
SPlisHSPlasH/DFSPH/TimeStepDFSPH.cpp:252-700, in mass form so several rest densities can share a solver.

Symbols, per particle i (ratio form, rho0_i its own rest density):
    r_i   = (sum_j m_j W_ij) / rho0_i + sum_b Psi_b                      density ratio (self included)
    G_i   = sum_j (m_j/rho0_i) dW_ij + sum_b dPsi_b                      grad of r_i
    alpha = 1 / (|G_i|^2 + sum_j m_i m_j/rho0_i^2 |dW_ij|^2)             Jacobi diagonal (BK17 eq. 11)
    a_i   = -sum_j m_j (k_i/rho0_i + k_j/rho0_j) dW_ij - k_i sum_b dPsi_b   pressure acceleration (k = p/rho0)
    Ar_i  = (1/rho0_i) sum_j m_j (a_i - a_j).dW_ij + a_i . sum_b dPsi_b   change of Dr/Dt per unit dt
Divergence solve: s = -max(Dr/Dt, 0) (0 with few neighbours), Ar*dt; density solve: s = 1 - r*, Ar*dt^2.
Update k <- max(k - w (s - Ar) alpha / dt^n, k_lo); error = complementarity residual: max(Ar - s, 0) where k = k_lo,
|s - Ar| where k > k_lo (surface deficiency never counts; over-pressure from a stale warm start does). k_lo = 0 except
in the divergence solve of pressurized liquid (solve_init_kernel) and, with an ambient pressure (the solve then runs
on gauge + p_a), in the density solve of liquid that is not sealed (ambient.py: k_lo = p_a / rho0).
The forces are -sum_i k_i dC_i/dx with C_i = r_i - 1, i.e. variationally consistent (conservative): the wall term
is k_i dPsi_b once, as SPlisHSPlasH (TimeStepDFSPH.cpp:1018-1036). A mirrored wall pressure (2 k_i) is not the
gradient of any potential and was measured to pump energy into a resting column. The wall volume is fitted so its
slope is twice the half-lattice gradient (kernels.boundary_fit), which makes the hydrostatic rest state force-free.
The reaction of slot b is -m_i (-k_i dPsi_b) = m_i k_i dPsi_b (pushes the wall away from the fluid).

Opposing walls (LIQ-8): between two walls closer than about 2r their gradients cancel, so |G_i| -> 0 while the
density error stays (the walls may leave no room at all). The solve then takes a Newton step of length err / |G_i|
along G_i, overshoots past the slit centre, and the particle rattles from wall to wall (measured: 3 m/s steady in a
static 2.6 mm slit at dt = 0.31 ms; 51 m/s and a wall crossing on the pour scene). Each particle gets a compliance
    lambda_i = sum_{b < c} max(0, -2 dPsi_b . dPsi_c)
(the cancelling part of |sum_b dPsi_b|^2) added to the diagonal of the operator (Ar_i -= lambda_i k_i) and of alpha,
so the boundary diagonal is at least sum_b |dPsi_b|^2 (two opposing walls: |dPsi_a|^2 + |dPsi_b|^2 instead of
(|dPsi_a| - |dPsi_b|)^2). It is zero unless two wall gradients make an angle above 90 degrees (single walls, seams and
box corners, whose face normals are orthogonal, are unchanged bit for bit), keeps the operator symmetric positive
definite and the forces variational, and bounds the step to the order of a single wall's: the constraint is
softened exactly where the walls leave no room.

Degenerate gradients (LIQ-8): the same Newton step blows up wherever |G_i|^2 + sum_j |dW_ij|^2 -> 0 with the density
error still there, e.g. a few particles almost on top of each other (dW -> 0 at r -> 0): measured, 4 particles 1 um
apart in free space leave at 2,666 m/s after one step (33 m/s at 0.1 mm; the step goes as 1/separation; 740 m/s 2r
above a floor), 7-9 m/s with the floor. The diagonal therefore has
a floor tot_min = Config.diagonal_floor x the interior rest-lattice value (kernels.lattice_diagonal); the shortfall
max(tot_min - tot_i, 0) joins lambda_i as compliance (operator diagonal and alpha), so the step stays bounded by about
err / sqrt(tot_min). Pressurized particles of a resting column and a dam break never come below 0.41 of the lattice
value (measured), so with the default 0.25 they are unchanged bit for bit.

Compression relief (flat impact, 2026-11-04): the density solve removes the predicted compression r* - 1 =
(r_i - 1) + dt Dr/Dt in one step, a displacement of about (r_i - 1)/sqrt(D) (D the interior rest-lattice diagonal),
so existing compression leaves at (r_i - 1)/(dt sqrt(D)). A particle the projection left in a wall's 1 um skin has
r_i 1.5-2.5; a 72 mm block dropped 20 mm flat into a 48 mm box glass (T4, r 1 mm) pinned such particles on the
glass's corner edges and the solve shot them and their neighbours out at 57 m/s (500 m/s converged, at the smaller dt
that followed). The positional part therefore enters the source capped at v dt sqrt(D),
v = Config.compression_relief_speed, like a rigid contact's bounded depenetration speed; the velocity part dt Dr/Dt
(the approach) is never capped. The cap is 8-20 % at dt 0.5-1.25 ms (r 1 mm), far above any converged step's
compression, so normal flow is bitwise unchanged. The drop then peaks at 5.6 m/s in wall jets (1.9 m/s at a
fixed 0.25 ms step) and no particle crosses a wall.

Lone particles (LIQ-8): a particle without fluid neighbours has r_i = self + walls, so its pressure is a wall contact
force. Where walls overlap or close in on it (the pour's cup corner pressed into the receiver wall, which MuJoCo
contact allows) r_i reaches 2-4 with a vanishing or flickering wall gradient, and the solve kicked such a particle
out at 3-7 m/s within single substeps (measured, two escapes over the tray rim). Its k is decoupled from every other
particle (no neighbours), so after each solve it is limited like a rigid contact's depenetration: the push -k dt G
may stop its approach to the walls and separate it at most Config.contact_separation_speed relative to them
(``contact_limit_kernel``). Particles with fluid neighbours are untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp
from alloy._vendor.newton import ParticleFlags

from .kernels import kernel_grad, kernel_w

_ACTIVE = wp.constant(int(ParticleFlags.ACTIVE))
MODE_DIVERGENCE = wp.constant(0)
MODE_DENSITY = wp.constant(1)


@wp.struct
class Particles:
    q: wp.array[wp.vec3]
    qd: wp.array[wp.vec3]
    mass: wp.array[wp.float32]
    rho0: wp.array[wp.float32]
    flags: wp.array[wp.int32]
    phase: wp.array[wp.int8]
    world: wp.array[wp.int32]
    nbr: wp.array2d[wp.uint16]  # [capacity, n] world-local offsets (neighbors.py, LIQ-9 L4b); decode with nbr_j
    nbr_hi: wp.array2d[wp.uint8]  # high byte of the offsets when a world has > 65,536 particles; else 1 x 1
    nbr_wide: int
    wstart: wp.array[wp.int32]  # first particle of each world
    count: wp.array[wp.int32]


@wp.func
def nbr_base(p: Particles, i: int) -> int:
    """The world start the offsets of particle i's row are counted from."""
    return p.wstart[p.world[i]]


@wp.func
def nbr_j(p: Particles, b: int, i: int, k: int) -> int:
    """Global id of the k-th neighbour of particle i (``b = nbr_base(p, i)``)."""
    j = b + wp.int32(p.nbr[k, i])
    if p.nbr_wide != 0:
        j = j + (wp.int32(p.nbr_hi[k, i]) << 16)
    return j


@wp.struct
class Slots:
    shape: wp.array2d[wp.int32]
    psi: wp.array2d[wp.float32]
    gpsi: wp.array2d[wp.vec3]
    vb: wp.array2d[wp.vec3]
    reg: wp.array[wp.float32]  # compliance: opposing walls + diagonal floor [1/m^2] (density_alpha_kernel)


@dataclass
class SolveArrays:
    """Per-particle work arrays of the pressure solves."""

    ratio: wp.array  # density ratio r_i
    alpha: wp.array
    neff: wp.array  # effective neighbour count (fluid + boundary-equivalent)
    src: wp.array  # divergence: Dr/Dt; density: predicted ratio r*
    kappa: wp.array  # current pressure k = p / rho0 [m^2/s^2]
    acc: wp.array  # pressure acceleration [m/s^2]
    err: wp.array  # per-particle residual of the last iteration (dimensionless)
    active: wp.array  # 1.0 for active SPH particles
    reg: wp.array  # opposing-wall compliance lambda_i [1/m^2] (exposed to the kernels as Slots.reg)

    @staticmethod
    def allocate(n: int, device: wp.context.Device) -> SolveArrays:
        n = max(n, 1)
        f = lambda: wp.zeros(n, dtype=wp.float32, device=device)  # noqa: E731
        return SolveArrays(f(), f(), f(), f(), f(), wp.zeros(n, dtype=wp.vec3, device=device), f(), f(), f())


@wp.func
def is_fluid(p: Particles, i: int) -> bool:
    return (p.flags[i] & _ACTIVE) != 0 and int(p.phase[i]) >= 1


@wp.kernel(enable_backward=False)
def density_alpha_kernel(
    p: Particles, sl: Slots, h: float, kind: int, n_full: float, tot_min: float,
    ratio: wp.array[wp.float32], alpha: wp.array[wp.float32], neff: wp.array[wp.float32],
    active: wp.array[wp.float32],
):
    i = wp.tid()
    if not is_fluid(p, i):
        ratio[i] = 0.0
        alpha[i] = 0.0
        neff[i] = 0.0
        active[i] = 0.0
        sl.reg[i] = 0.0
        return
    xi = p.q[i]
    mi = p.mass[i]
    inv_r0 = 1.0 / p.rho0[i]
    rho = mi * kernel_w(0.0, h, kind)
    G = wp.vec3(0.0, 0.0, 0.0)
    sq = float(0.0)
    cnt = p.count[i]
    b_ = nbr_base(p, i)
    for k in range(cnt):
        j = nbr_j(p, b_, i, k)
        rv = xi - p.q[j]
        mj = p.mass[j]
        rho += mj * kernel_w(wp.length(rv), h, kind)
        g = kernel_grad(rv, h, kind)
        G += (mj * inv_r0) * g
        sq += (mi * mj * inv_r0 * inv_r0) * wp.dot(g, g)
    psi = float(0.0)
    lam = float(0.0)  # opposing-wall compliance (module doc)
    for b in range(4):
        if sl.shape[i, b] >= 0:
            psi += sl.psi[i, b]
            G += sl.gpsi[i, b]
            for c in range(b + 1, 4):
                if sl.shape[i, c] >= 0:
                    lam += wp.max(-2.0 * wp.dot(sl.gpsi[i, b], sl.gpsi[i, c]), 0.0)
    tot = wp.dot(G, G) + sq + lam
    lam += wp.max(tot_min - tot, 0.0)  # degenerate gradients (module doc)
    tot = wp.max(tot, tot_min)
    sl.reg[i] = lam
    ratio[i] = rho * inv_r0 + psi
    a = float(0.0)
    if tot > 1.0e-12:
        a = 1.0 / tot
    alpha[i] = a
    neff[i] = float(cnt) + psi * n_full
    active[i] = 1.0


@wp.func
def density_rate(p: Particles, sl: Slots, i: int, h: float, kind: int) -> float:
    """D r_i / Dt from fluid neighbours and boundary slots [1/s]."""
    xi = p.q[i]
    vi = p.qd[i]
    inv_r0 = 1.0 / p.rho0[i]
    rate = float(0.0)
    b_ = nbr_base(p, i)
    for k in range(p.count[i]):
        j = nbr_j(p, b_, i, k)
        g = kernel_grad(xi - p.q[j], h, kind)
        rate += (p.mass[j] * inv_r0) * wp.dot(vi - p.qd[j], g)
    for b in range(4):
        if sl.shape[i, b] >= 0:
            rate += wp.dot(vi - sl.vb[i, b], sl.gpsi[i, b])
    return rate


@wp.kernel(enable_backward=False)
def solve_init_kernel(
    p: Particles, sl: Slots, h: float, kind: int, mode: int, dt: float, min_neighbors: float, warm: float,
    warm_den: float, two_sided: int, ratio: wp.array[wp.float32], neff: wp.array[wp.float32], kappa_store: wp.array[wp.float32],
    kd_store: wp.array[wp.float32], src: wp.array[wp.float32], kappa: wp.array[wp.float32],
    lower: wp.array[wp.float32], relief: float,
):
    """Source term, warm start (``warm`` x previous k) and the lower bound of k. Density solve: the existing
    compression r_i - 1 enters the source capped at ``relief`` (compression relief, module doc).

    Warm starts are always kept (SPlisHSPlasH keeps them only where rho* > 1, TimeStepDFSPH.cpp:285-292, so a column
    at exact rest density loses its whole hydrostatic pressure every step and free-falls while Jacobi rebuilds it);
    the complementarity residual iterates away any part that over-expands. Clamps as SPlisHSPlasH: 0.5 m^2/s
    (divergence), 2.5e-4 m^2 (density).

    Divergence solve with ``two_sided`` (Config.divergence_two_sided, default off): where the expected density
    pressure k_d (its warm start) is positive the liquid cannot expand either, so the source is the signed Dr/Dt and
    k_v may go negative down to -k_d (total pressure stays >= 0); elsewhere only compression counts. Measured on a
    resting column it does not reduce the residual motion at the default 1e-3 per-step tolerance (that tolerance
    itself admits strain rates up to 1e-3/dt), so it is off by default (docs/notes/wp9-results.md)."""
    i = wp.tid()
    if not is_fluid(p, i):
        src[i] = 0.0
        kappa[i] = 0.0
        lower[i] = 0.0
        return
    rate = density_rate(p, sl, i, h, kind)
    if mode == MODE_DIVERGENCE:
        kd = float(0.0)
        if two_sided != 0:
            kd = wp.max(warm_den * wp.min(kd_store[i], 2.5e-4) / (dt * dt), 0.0)  # tension (ambient.py) is no head
        lb = -kd
        r = rate
        if kd <= 0.0:
            r = wp.max(rate, 0.0)
        if neff[i] < min_neighbors:
            r = 0.0
            lb = 0.0
        src[i] = r
        lower[i] = lb
        kappa[i] = wp.max(warm * wp.min(kappa_store[i], 0.5) / dt, lb)
    else:
        rs = ratio[i] + dt * rate
        if ratio[i] - 1.0 > relief:  # compression relief (module doc); this branch only, so uncapped is bitwise
            rs = 1.0 + relief + dt * rate
        src[i] = rs
        lower[i] = 0.0
        kappa[i] = warm * wp.min(kappa_store[i], 2.5e-4) / (dt * dt)


@wp.func
def pressure_accel(p: Particles, sl: Slots, kappa: wp.array[wp.float32], i: int, h: float, kind: int) -> wp.vec3:
    xi = p.q[i]
    ki = kappa[i] / p.rho0[i]
    a = wp.vec3(0.0, 0.0, 0.0)
    b_ = nbr_base(p, i)
    for k in range(p.count[i]):
        j = nbr_j(p, b_, i, k)
        g = kernel_grad(xi - p.q[j], h, kind)
        a -= (p.mass[j] * (ki + kappa[j] / p.rho0[j])) * g
    for b in range(4):
        if sl.shape[i, b] >= 0:
            a -= kappa[i] * sl.gpsi[i, b]
    return a


@wp.kernel(enable_backward=False)
def accel_kernel(
    p: Particles, sl: Slots, h: float, kind: int, kappa: wp.array[wp.float32],
    converged: wp.array[wp.int32], acc: wp.array[wp.vec3],
):
    i = wp.tid()
    if not is_fluid(p, i):
        acc[i] = wp.vec3(0.0, 0.0, 0.0)
        return
    if converged[p.world[i]] != 0:
        return
    acc[i] = pressure_accel(p, sl, kappa, i, h, kind)


@wp.func
def accel_divergence(p: Particles, sl: Slots, acc: wp.array[wp.vec3], i: int, h: float, kind: int) -> float:
    """Ar_i = (1/rho0_i) sum_j m_j (a_i - a_j).dW_ij + a_i . sum_b dPsi_b: rate of change of Dr_i/Dt per unit time
    produced by the accelerations ``acc`` (the matrix-free product of the DFSPH system)."""
    xi = p.q[i]
    ai = acc[i]
    inv_r0 = 1.0 / p.rho0[i]
    ar = float(0.0)
    b_ = nbr_base(p, i)
    for k in range(p.count[i]):
        j = nbr_j(p, b_, i, k)
        g = kernel_grad(xi - p.q[j], h, kind)
        ar += (p.mass[j] * inv_r0) * wp.dot(ai - acc[j], g)
    for b in range(4):
        if sl.shape[i, b] >= 0:
            ar += wp.dot(ai, sl.gpsi[i, b])
    return ar


@wp.kernel(enable_backward=False)
def jacobi_kernel(
    p: Particles, sl: Slots, h: float, kind: int, mode: int, dt: float, min_neighbors: float,
    alpha: wp.array[wp.float32], neff: wp.array[wp.float32], src: wp.array[wp.float32],
    acc: wp.array[wp.vec3], converged: wp.array[wp.int32], omega: float, lower: wp.array[wp.float32],
    kappa: wp.array[wp.float32], err: wp.array[wp.float32],
):
    """One relaxed Jacobi sweep; ``err`` per particle (the world max comes from ``WorldReducer.sum``'s max)."""
    i = wp.tid()
    if not is_fluid(p, i):
        err[i] = 0.0
        return
    w = p.world[i]
    if converged[w] != 0:
        return
    ar = accel_divergence(p, sl, acc, i, h, kind) - sl.reg[i] * kappa[i]  # opposing-wall compliance
    s = float(0.0)
    scale = float(0.0)
    if mode == MODE_DIVERGENCE:
        ar *= dt
        s = -src[i]
        scale = 1.0 / dt
    else:
        ar *= dt * dt
        s = 1.0 - src[i]
        scale = 1.0 / (dt * dt)
    r = s - ar
    lb = lower[i]
    k_new = wp.max(kappa[i] - omega * r * alpha[i] * scale, lb)
    kappa[i] = k_new
    # complementarity residual of rho* <= 1, k >= 0, k (1 - rho*) = 0: compression always counts; where the pressure
    # is active (k > 0) any deviation counts, so a stale warm start that over-expands is iterated away.
    e = wp.max(-r, 0.0)
    if k_new > lb:
        e = wp.abs(r)
    if mode == MODE_DIVERGENCE:
        if neff[i] < min_neighbors:
            e = 0.0
        e *= dt
    err[i] = e


@wp.kernel(enable_backward=False)
def begin_solve_kernel(
    n_active: wp.array[wp.float64], converged: wp.array[wp.int32], iters: wp.array[wp.int32],
    err_mean: wp.array[wp.float32], n_unconverged: wp.array[wp.int32],
):
    w = wp.tid()
    iters[w] = 0
    err_mean[w] = 0.0
    if n_active[w] > wp.float64(0.0):
        converged[w] = 0
        wp.atomic_add(n_unconverged, 0, 1)
    else:
        converged[w] = 1


@wp.kernel(enable_backward=False)
def check_kernel(
    err_sum: wp.array[wp.float64], n_active: wp.array[wp.float64], err_max: wp.array[wp.float32], tol: float,
    tol_max: float, tol_scale: wp.array[wp.float32], min_it: int, max_it: int, converged: wp.array[wp.int32],
    iters: wp.array[wp.int32], err_mean: wp.array[wp.float32], n_unconverged: wp.array[wp.int32],
):
    """Per world: converged when mean <= tol and max <= tol_max, both x tol_scale[w] (after min_it), or at max_it.
    The max test catches compression concentrated in a few layers (a sinking column only compresses its floor layer,
    which a mean over all particles dilutes by the column height)."""
    w = wp.tid()
    if converged[w] != 0:
        return
    it = iters[w] + 1
    iters[w] = it
    mean = float(err_sum[w] / wp.max(n_active[w], wp.float64(1.0)))
    err_mean[w] = mean
    ts = tol_scale[w]
    if (it >= min_it and mean <= tol * ts and err_max[w] <= tol_max * ts) or it >= max_it:
        converged[w] = 1
    else:
        wp.atomic_add(n_unconverged, 0, 1)


@wp.kernel(enable_backward=False)
def contact_limit_kernel(p: Particles, sl: Slots, dt: float, v_sep: float, kappa: wp.array[wp.float32]):
    """Lone particles (module doc): k_i <= max(v_sep - u, 0) / (dt |G_i|), u = separation speed from the walls along
    the push -G_i (relative to their Psi-weighted velocity)."""
    i = wp.tid()
    if not is_fluid(p, i) or p.count[i] > 0 or kappa[i] <= 0.0:
        return
    G = wp.vec3(0.0, 0.0, 0.0)
    vb = wp.vec3(0.0, 0.0, 0.0)
    wsum = float(0.0)
    for b in range(4):
        if sl.shape[i, b] >= 0:
            G += sl.gpsi[i, b]
            vb += sl.psi[i, b] * sl.vb[i, b]
            wsum += sl.psi[i, b]
    g = wp.length(G)
    if g <= 0.0:
        return
    if wsum > 0.0:
        vb = vb / wsum
    u = wp.dot(p.qd[i] - vb, -G / g)
    kappa[i] = wp.min(kappa[i], wp.max(v_sep - u, 0.0) / (dt * g))


@wp.kernel(enable_backward=False)
def lone_friction_kernel(p: Particles, sl: Slots, dt: float, kappa: wp.array[wp.float32], mu: wp.array[wp.float32],
                         n_shapes: int, slot_impulse: wp.array2d[wp.vec3]):
    """Coulomb friction of a lone particle on its walls (Config.wall_friction, SCR-OSS3-2): the wall's push this solve,
    dv_n = dt k_i |G_i|, takes up to mu dv_n of its tangential speed relative to the walls (Psi-weighted velocity),
    never reversing it; mu is the shape of its largest-Psi slot (virtual soft faces: none). The impulse goes to that
    slot's wall like the pressure reaction. A droplet that left the liquid then slows at mu g on a floor."""
    i = wp.tid()
    if not is_fluid(p, i) or p.count[i] > 0 or kappa[i] <= 0.0:
        return
    G = wp.vec3(0.0, 0.0, 0.0)
    vb = wp.vec3(0.0, 0.0, 0.0)
    wsum = float(0.0)
    best = int(-1)
    pbest = float(0.0)
    for b in range(4):
        s = sl.shape[i, b]
        if s >= 0:
            G += sl.gpsi[i, b]
            vb += sl.psi[i, b] * sl.vb[i, b]
            wsum += sl.psi[i, b]
            if sl.psi[i, b] > pbest:
                pbest = sl.psi[i, b]
                best = b
    g = wp.length(G)
    if g <= 0.0 or wsum <= 0.0 or best < 0:
        return
    s = sl.shape[i, best]
    if s >= n_shapes:
        return
    m_u = mu[s]
    if m_u <= 0.0:
        return
    n = -G / g
    u = p.qd[i] - vb / wsum
    vt = u - wp.dot(u, n) * n
    lt = wp.length(vt)
    if lt <= 0.0:
        return
    dv = wp.min(1.0, m_u * dt * kappa[i] * g / lt) * vt
    p.qd[i] = p.qd[i] - dv
    slot_impulse[i, best] = slot_impulse[i, best] + p.mass[i] * dv


@wp.kernel(enable_backward=False)
def apply_pressure_kernel(
    p: Particles, sl: Slots, h: float, kind: int, mode: int, dt: float, kappa: wp.array[wp.float32],
    kappa_store: wp.array[wp.float32], slot_impulse: wp.array2d[wp.vec3],
):
    """v += dt a(k); boundary reactions m_i k_i dPsi_b dt to slots (also in one-way coupling: they give the walls'
    work on the liquid, bodies.py); store the dt-free warm start."""
    i = wp.tid()
    if not is_fluid(p, i):
        return
    a = pressure_accel(p, sl, kappa, i, h, kind)
    p.qd[i] = p.qd[i] + dt * a
    mk = p.mass[i] * kappa[i] * dt
    for b in range(4):
        if sl.shape[i, b] >= 0:
            slot_impulse[i, b] = slot_impulse[i, b] + mk * sl.gpsi[i, b]
    if mode == MODE_DIVERGENCE:
        kappa_store[i] = kappa[i] * dt
    else:
        kappa_store[i] = kappa[i] * dt * dt


@wp.kernel(enable_backward=False)
def max_err_kernel(p: Particles, err: wp.array[wp.float32], out: wp.array[wp.float32]):
    i = wp.tid()
    if is_fluid(p, i):
        wp.atomic_max(out, p.world[i], err[i])


@wp.kernel(enable_backward=False)
def scale_kernel(src: wp.array[wp.float32], factor: float, dst: wp.array[wp.float32]):
    """dst = factor * src (divergence warm start k dt -> density warm start k dt^2)."""
    i = wp.tid()
    dst[i] = factor * src[i]
