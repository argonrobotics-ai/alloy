# SPDX-License-Identifier: Apache-2.0
"""Warp kernels of the rod direct solve (SCR-WP2-10): per chain, one Gauss-Newton step of SolverVBD's backward-Euler
energy over all of the chain's bodies at once, by block Thomas on the 6N x 6N block-tridiagonal system.

**Energy per chain joint j** (parent p, child c; joint frames X_p, X_c; R_wp = R_p R(X_p)):
- stretch/shear: u = R_wp^T (x_c - x_p), E = 1/2 u^T diag(ke0, ke1, ke2) u;
- bend/twist: kappa = log(A0^T R_wp^T R_wc), E = 1/2 kappa^T diag(ke3, ke4, ke5) kappa.
- Each of the six slots also takes the damping term kd (r - r_prev) / dt, with r its residual u or kappa at the
  previous step's poses.
- **Geometric stiffness** (SCR-WP9-ROD R3c, `use_geo`). Gauss-Newton keeps J^T K J and drops the stretch springs'
  second-order term. A spring force f acting at an anchor r from a body's COM adds (f . r) I - sym(f r^T) to that
  body's rotation block. Under tension (f . r > 0) its part (f . r)(I - r^ r^T) is positive, and dropping it made
  tension act explicitly on segment rotation. A taut light rope then diverges once T h passes I_t / dt^2: the A10
  capstan rope at 1 N went non-finite at 1.25 ms. The kernel adds that positive part to both bodies. It adds nothing
  under compression (buckling stays the energy's own business), and it leaves out the frame-rotation part (u is
  measured in R_wp, which also turns with the parent; WP2's review). It changes the Hessian, not the forces, so
  statics are unchanged.

The stiffnesses come from the joint's `joint_target_ke` / `joint_target_kd` in add_rod's order (shear, shear,
stretch, bend, bend, twist). A0 is the rest relative frame. A FIXED joint (a welded end body) takes isotropic
k_lin / k_ang, SolverVBD's structural joint stiffness, and no damping.

**Body terms:** inertia exactly as SolverVBD's `solve_rigid_body` (COM target m/dt^2, rotation-vector target
I/dt^2), plus the contact and other-joint blocks Newton accumulated into the solver's scratch (`use_ext`). The other
body of a contact or a non-chain joint is held, as in Newton's sweep.
- An applied torque (`applied_t`, per slot, world frame about the COM) is a potential, -tau . dtheta: it adds tau to
  the angular descent force and nothing to the Hessian. It does not shift the inertial target, so a segment's tiny
  axial inertia cannot wrap the target rotation (SCR-WP9-ROD R1). A zero torque adds nothing, bit for bit.

**Coordinates:** per body (dc, dtheta), world frame, about the COM. The pose update is the exact exponential map.
Kinematic bodies (inverse mass 0) are held: their row is the identity with no coupling.

**Kernels** (shared per-slot functions, so the paths are the same arithmetic):
- `rod_assemble`, thread per slot;
- `rod_thomas`, thread per chain;
- `rod_update`, thread per slot;
- `rod_serial`: all three, thread per chain.
"""

from __future__ import annotations

import warp as wp

from alloy._vendor.newton._src.solvers.vbd.rigid_vbd_kernels import evaluate_joint_force_hessian

mat66 = wp.types.matrix(shape=(6, 6), dtype=wp.float32)
vec6 = wp.types.vector(length=6, dtype=wp.float32)
JOINT_FIXED = wp.constant(3)


@wp.struct
class RodArgs:
    """Model and solver arrays the rod kernels read."""

    dt: float
    k_lin: float
    k_ang: float
    use_ext: int
    use_geo: int
    rest_rel: wp.array(dtype=wp.quat)
    body_q_prev: wp.array(dtype=wp.transform)
    body_inertia_q: wp.array(dtype=wp.transform)
    body_com: wp.array(dtype=wp.vec3)
    body_mass: wp.array(dtype=float)
    body_inertia: wp.array(dtype=wp.mat33)
    body_inv_mass: wp.array(dtype=float)
    joint_type: wp.array(dtype=int)
    joint_parent: wp.array(dtype=int)
    joint_X_p: wp.array(dtype=wp.transform)
    joint_X_c: wp.array(dtype=wp.transform)
    joint_qd_start: wp.array(dtype=int)
    joint_ke: wp.array(dtype=float)
    joint_kd: wp.array(dtype=float)
    ext_f: wp.array(dtype=wp.vec3)
    ext_t: wp.array(dtype=wp.vec3)
    ext_ll: wp.array(dtype=wp.mat33)
    ext_al: wp.array(dtype=wp.mat33)
    ext_aa: wp.array(dtype=wp.mat33)
    applied_t: wp.array(dtype=wp.vec3)


@wp.struct
class RodChains:
    """Slot layout (alloy.solvers.rod.topology) and the per-slot scratch: D diagonal blocks (then their Cholesky
    factors), O[k] the coupling of slot k-1 (rows) and slot k (columns), F the descent force (then the step)."""

    body: wp.array(dtype=int)
    joint: wp.array(dtype=int)
    start: wp.array(dtype=int)
    count: wp.array(dtype=int)
    last: wp.array(dtype=int)
    D: wp.array(dtype=mat66)
    O: wp.array(dtype=mat66)
    F: wp.array(dtype=vec6)


@wp.func
def skew(v: wp.vec3) -> wp.mat33:
    return wp.mat33(0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0)


@wp.func
def jr_inv(phi: wp.vec3) -> wp.mat33:
    """Inverse right Jacobian of SO(3): log(R exp(d)) = log(R) + Jr^-1 d to first order."""
    th = wp.length(phi)
    K = skew(phi)
    c = 1.0 / 12.0
    if th > 1.0e-4:
        c = 1.0 / (th * th) - (1.0 + wp.cos(th)) / (2.0 * th * wp.sin(th))
    return wp.identity(n=3, dtype=float) + 0.5 * K + c * (K * K)


@wp.func
def rot_vec(q_in: wp.quat) -> wp.vec3:
    q = q_in
    if q[3] < 0.0:
        q = wp.quat(-q[0], -q[1], -q[2], -q[3])
    axis, ang = wp.quat_to_axis_angle(q)
    return axis * ang


@wp.func
def add_blk(M: mat66, r: int, c: int, B: wp.mat33) -> mat66:
    for i in range(3):
        for j in range(3):
            M[r + i, c + j] = M[r + i, c + j] + B[i, j]
    return M


@wp.func
def blocks66(ll: wp.mat33, la: wp.mat33, al: wp.mat33, aa: wp.mat33) -> mat66:
    M = mat66()
    M = add_blk(M, 0, 0, ll)
    M = add_blk(M, 0, 3, la)
    M = add_blk(M, 3, 0, al)
    return add_blk(M, 3, 3, aa)


@wp.func
def vec66(lin: wp.vec3, ang: wp.vec3) -> vec6:
    return vec6(lin[0], lin[1], lin[2], ang[0], ang[1], ang[2])


@wp.func
def chol6(A: mat66) -> mat66:
    L = mat66()
    for j in range(6):
        s = A[j, j]
        for k in range(j):
            s = s - L[j, k] * L[j, k]
        d = wp.sqrt(wp.max(s, 1.0e-30))
        L[j, j] = d
        for i in range(j + 1, 6):
            t = A[i, j]
            for k in range(j):
                t = t - L[i, k] * L[j, k]
            L[i, j] = t / d
    return L


@wp.func
def chol6_solve(L: mat66, b: vec6) -> vec6:
    y = vec6()
    for i in range(6):
        s = b[i]
        for k in range(i):
            s = s - L[i, k] * y[k]
        y[i] = s / L[i, i]
    x = vec6()
    for ii in range(6):
        i = 5 - ii
        s = y[i]
        for k in range(i + 1, 6):
            s = s - L[k, i] * x[k]
        x[i] = s / L[i, i]
    return x


@wp.func
def chol6_solve_mat(L: mat66, B: mat66) -> mat66:
    X = mat66()
    for c in range(6):
        col = vec6()
        for i in range(6):
            col[i] = B[i, c]
        x = chol6_solve(L, col)
        for i in range(6):
            X[i, c] = x[i]
    return X


@wp.func
def com_of(q: wp.transform, com_local: wp.vec3) -> wp.vec3:
    return wp.transform_get_translation(q) + wp.quat_rotate(wp.transform_get_rotation(q), com_local)


@wp.func
def tension_stiffness(f: wp.vec3, r: wp.vec3) -> wp.mat33:
    """The positive part of a spring force f's geometric stiffness at anchor r: (f . r)(I - r^ r^T) under tension,
    zero under compression (kernels docstring)."""
    fr = wp.dot(f, r)
    rr = wp.dot(r, r)
    if fr <= 0.0 or rr <= 0.0:
        return wp.mat33(0.0)
    return fr * (wp.identity(n=3, dtype=float) - wp.outer(r, r) / rr)


@wp.func
def joint_terms(A: RodArgs, body_q: wp.array(dtype=wp.transform), j: int, ba: int, bb: int):
    """(F_a, F_b, H_aa, H_bb, H_ab) of chain joint j between the bodies of consecutive slots a, b: descent forces and
    Gauss-Newton blocks, H_ab with rows a and columns b. The joint's parent may be either body."""
    flip = A.joint_parent[j] != ba
    bp = ba
    bc = bb
    if flip:
        bp = bb
        bc = ba
    qp = body_q[bp]
    qc = body_q[bc]
    Xp = A.joint_X_p[j]
    Xc = A.joint_X_c[j]
    qwp = wp.mul(wp.transform_get_rotation(qp), wp.transform_get_rotation(Xp))
    qwc = wp.mul(wp.transform_get_rotation(qc), wp.transform_get_rotation(Xc))
    Rwp = wp.quat_to_matrix(qwp)
    RpT = wp.transpose(Rwp)
    xp = wp.transform_point(qp, wp.transform_get_translation(Xp))
    xc = wp.transform_point(qc, wp.transform_get_translation(Xc))
    rp = xp - com_of(qp, A.body_com[bp])
    rc = xc - com_of(qc, A.body_com[bc])
    d = xc - xp
    o = A.joint_qd_start[j]
    Ks = wp.mat33(A.joint_ke[o], 0.0, 0.0, 0.0, A.joint_ke[o + 1], 0.0, 0.0, 0.0, A.joint_ke[o + 2])
    kb = wp.vec3(A.k_ang)
    kdb = wp.vec3(0.0)
    kds = wp.vec3(0.0)
    if A.joint_type[j] == JOINT_FIXED:
        Ks = A.k_lin * wp.identity(n=3, dtype=float)
    else:
        kb = wp.vec3(A.joint_ke[o + 3], A.joint_ke[o + 4], A.joint_ke[o + 5])
        kdb = wp.vec3(A.joint_kd[o + 3], A.joint_kd[o + 4], A.joint_kd[o + 5])
        kds = wp.vec3(A.joint_kd[o], A.joint_kd[o + 1], A.joint_kd[o + 2])
    # stretch / shear: residual u = R_wp^T d; d(u)/d(dc_p) = -R_wp^T, d(dc_c) = R_wp^T; damping kd (u - u_prev)/dt
    Jpa = RpT * (skew(rp) + skew(d))
    Jca = -RpT * skew(rc)
    qpp = A.body_q_prev[bp]
    qcp = A.body_q_prev[bc]
    qwp0 = wp.mul(wp.transform_get_rotation(qpp), wp.transform_get_rotation(Xp))
    u0 = wp.quat_rotate_inv(qwp0, wp.transform_point(qcp, wp.transform_get_translation(Xc))
                            - wp.transform_point(qpp, wp.transform_get_translation(Xp)))
    u = RpT * d
    fs = Ks * u + wp.cw_mul(kds, u - u0) / A.dt
    Ks = Ks + wp.mat33(kds[0], 0.0, 0.0, 0.0, kds[1], 0.0, 0.0, 0.0, kds[2]) / A.dt
    # bend / twist on the relative rotation, with its previous-step value for damping
    A0 = A.rest_rel[j]
    kap = rot_vec(wp.mul(wp.quat_inverse(A0), wp.mul(wp.quat_inverse(qwp), qwc)))
    qwc0 = wp.mul(wp.transform_get_rotation(qcp), wp.transform_get_rotation(Xc))
    kap0 = rot_vec(wp.mul(wp.quat_inverse(A0), wp.mul(wp.quat_inverse(qwp0), qwc0)))
    fk = wp.cw_mul(kb, kap) + wp.cw_mul(kdb, kap - kap0) / A.dt
    Kb = wp.mat33(kb[0] + kdb[0] / A.dt, 0.0, 0.0, 0.0, kb[1] + kdb[1] / A.dt, 0.0, 0.0, 0.0, kb[2] + kdb[2] / A.dt)
    Jbc = jr_inv(kap) * wp.transpose(wp.quat_to_matrix(qwc))
    Jbp = -(jr_inv(-kap) * wp.quat_to_matrix(wp.quat_inverse(A0))) * RpT
    Fp = vec66(Rwp * fs, -(wp.transpose(Jpa) * fs + wp.transpose(Jbp) * fk))
    Fc = vec66(-(Rwp * fs), -(wp.transpose(Jca) * fs + wp.transpose(Jbc) * fk))
    KsR = Ks * RpT
    KsJpa = Ks * Jpa
    KsJca = Ks * Jca
    Hpp = blocks66(Rwp * KsR, -(Rwp * KsJpa), -(wp.transpose(Jpa) * KsR),
                   wp.transpose(Jpa) * KsJpa + wp.transpose(Jbp) * Kb * Jbp)
    Hcc = blocks66(Rwp * KsR, Rwp * KsJca, wp.transpose(Jca) * KsR,
                   wp.transpose(Jca) * KsJca + wp.transpose(Jbc) * Kb * Jbc)
    if A.use_geo != 0:                       # tension's geometric stiffness at each anchor (docstring)
        fw = Rwp * fs                        # the spring's force on the parent; the child takes -fw
        Hpp = add_blk(Hpp, 3, 3, tension_stiffness(fw, rp))
        Hcc = add_blk(Hcc, 3, 3, tension_stiffness(-fw, rc))
    Hpc = blocks66(-(Rwp * KsR), -(Rwp * KsJca), wp.transpose(Jpa) * KsR,
                   wp.transpose(Jpa) * KsJca + wp.transpose(Jbp) * Kb * Jbc)
    if flip:
        return Fc, Fp, Hcc, Hpp, wp.transpose(Hpc)
    return Fp, Fc, Hpp, Hcc, Hpc


@wp.func
def assemble_slot(A: RodArgs, C: RodChains, body_q: wp.array(dtype=wp.transform), k: int):
    """Slot k's diagonal block, descent force and coupling to slot k-1: its body's inertia and external blocks and
    its side of the two adjacent chain joints."""
    b = C.body[k]
    inv_dt2 = 1.0 / (A.dt * A.dt)
    q = body_q[b]
    qs = A.body_inertia_q[b]
    cl = A.body_com[b]
    m = A.body_mass[b]
    rot = wp.transform_get_rotation(q)
    Rm = wp.quat_to_matrix(rot)
    Ib = A.body_inertia[b]
    Dk = blocks66((m * inv_dt2) * wp.identity(n=3, dtype=float), wp.mat33(0.0), wp.mat33(0.0),
                  inv_dt2 * (Rm * Ib * wp.transpose(Rm)))
    th_b = rot_vec(wp.mul(wp.quat_inverse(rot), wp.transform_get_rotation(qs)))
    Fk = vec66((com_of(qs, cl) - com_of(q, cl)) * (m * inv_dt2), wp.quat_rotate(rot, Ib * (th_b * inv_dt2)))
    if A.use_ext != 0:
        Fk = Fk + vec66(A.ext_f[b], A.ext_t[b])
        Dk = Dk + blocks66(A.ext_ll[b], wp.transpose(A.ext_al[b]), A.ext_al[b], A.ext_aa[b])
    tau = A.applied_t[k]
    if wp.length_sq(tau) > 0.0:
        Fk = Fk + vec66(wp.vec3(0.0), tau)
    held = A.body_inv_mass[b] == 0.0
    Ok = mat66()
    j = C.joint[k]
    if j >= 0:
        ba = C.body[k - 1]
        Fa, Fb, Haa, Hbb, Hab = joint_terms(A, body_q, j, ba, b)
        Dk = Dk + Hbb
        Fk = Fk + Fb
        if not held and A.body_inv_mass[ba] != 0.0:
            Ok = Hab
    if C.last[k] == 0:
        Fa2, Fb2, Haa2, Hbb2, Hab2 = joint_terms(A, body_q, C.joint[k + 1], b, C.body[k + 1])
        Dk = Dk + Haa2
        Fk = Fk + Fa2
    eps = 1.0e-9 * ((Dk[3, 3] + Dk[4, 4] + Dk[5, 5]) / 3.0 + 1.0)   # SolverVBD's angular regularisation
    for i in range(3, 6):
        Dk[i, i] = Dk[i, i] + eps
    if held:
        Dk = wp.identity(n=6, dtype=float)
        Fk = vec6()
    return Dk, Fk, Ok


@wp.func
def thomas_chain(C: RodChains, r: int):
    s = C.start[r]
    n = C.count[r]
    L = chol6(C.D[s])
    C.D[s] = L
    for k in range(s + 1, s + n):
        Ok = C.O[k]
        Ot = wp.transpose(Ok)
        X = chol6_solve_mat(L, Ok)
        y = chol6_solve(L, C.F[k - 1])
        L = chol6(C.D[k] - Ot * X)
        C.D[k] = L
        C.F[k] = C.F[k] - Ot * y
    x = chol6_solve(C.D[s + n - 1], C.F[s + n - 1])
    C.F[s + n - 1] = x
    for kk in range(n - 1):
        k = s + n - 2 - kk
        x = chol6_solve(C.D[k], C.F[k] - C.O[k + 1] * x)
        C.F[k] = x


@wp.func
def update_slot(A: RodArgs, C: RodChains, body_q: wp.array(dtype=wp.transform), k: int):
    b = C.body[k]
    if A.body_inv_mass[b] == 0.0:
        return
    xk = C.F[k]
    w = wp.vec3(xk[3], xk[4], xk[5])
    q = body_q[b]
    rot = wp.transform_get_rotation(q)
    ang = wp.length(w)
    dq = wp.quat_identity()
    if ang > 1.0e-12:
        dq = wp.quat_from_axis_angle(w / ang, ang)
    rot_new = wp.normalize(wp.mul(dq, rot))
    com_new = com_of(q, A.body_com[b]) + wp.vec3(xk[0], xk[1], xk[2])
    body_q[b] = wp.transform(com_new - wp.quat_rotate(rot_new, A.body_com[b]), rot_new)


@wp.kernel
def rod_assemble(A: RodArgs, C: RodChains, slots: wp.array(dtype=int), body_q: wp.array(dtype=wp.transform)):
    k = slots[wp.tid()]
    Dk, Fk, Ok = assemble_slot(A, C, body_q, k)
    C.D[k] = Dk
    C.F[k] = Fk
    C.O[k] = Ok


@wp.kernel
def rod_thomas(C: RodChains, rods: wp.array(dtype=int)):
    thomas_chain(C, rods[wp.tid()])


@wp.kernel
def rod_update(A: RodArgs, C: RodChains, slots: wp.array(dtype=int), body_q: wp.array(dtype=wp.transform)):
    update_slot(A, C, body_q, slots[wp.tid()])


@wp.kernel
def rod_serial(A: RodArgs, C: RodChains, rods: wp.array(dtype=int), body_q: wp.array(dtype=wp.transform)):
    """The three passes for one chain in one thread (fewer launches; faster on CPU)."""
    r = rods[wp.tid()]
    s = C.start[r]
    n = C.count[r]
    for k in range(s, s + n):
        Dk, Fk, Ok = assemble_slot(A, C, body_q, k)
        C.D[k] = Dk
        C.F[k] = Fk
        C.O[k] = Ok
    thomas_chain(C, r)
    for k in range(s, s + n):
        update_slot(A, C, body_q, k)


@wp.kernel
def rod_external_joints(
    ext_body: wp.array(dtype=int), ext_start: wp.array(dtype=int), ext_joint: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform), body_q_rest: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3), joint_type: wp.array(dtype=int), joint_enabled: wp.array(dtype=bool),
    joint_parent: wp.array(dtype=int), joint_child: wp.array(dtype=int), joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform), joint_axis: wp.array(dtype=wp.vec3),
    joint_rod_rest_kb_local: wp.array(dtype=wp.vec3), joint_rod_rest_twist: wp.array(dtype=float),
    joint_qd_start: wp.array(dtype=int), joint_target_q_start: wp.array(dtype=int),
    joint_constraint_start: wp.array(dtype=int), joint_penalty_k: wp.array(dtype=float),
    joint_rho: wp.array(dtype=float), joint_material_k: wp.array(dtype=float),
    joint_penalty_kd: wp.array(dtype=float), joint_sigma_start: wp.array(dtype=wp.vec3),
    joint_C_fric: wp.array(dtype=wp.vec3), joint_target_ke: wp.array(dtype=float),
    joint_target_kd: wp.array(dtype=float), joint_target_q: wp.array(dtype=float),
    joint_target_qd: wp.array(dtype=float), joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float), joint_limit_ke: wp.array(dtype=float),
    joint_limit_kd: wp.array(dtype=float), joint_drive_limit_support: wp.array(dtype=float),
    joint_drive_lambda: wp.array(dtype=float), joint_limit_lambda: wp.array(dtype=float),
    joint_lambda_lin: wp.array(dtype=wp.vec3), joint_lambda_ang: wp.array(dtype=wp.vec3),
    joint_C0_lin: wp.array(dtype=wp.vec3), joint_C0_ang: wp.array(dtype=wp.vec3),
    joint_is_hard: wp.array(dtype=wp.int32), stab_alpha: float, joint_compliant_alm: int,
    joint_dof_dim: wp.array2d(dtype=int), joint_rest_angle: wp.array(dtype=float),
    joint_angle_prev: wp.array(dtype=float), dt: float,
    body_forces: wp.array(dtype=wp.vec3), body_torques: wp.array(dtype=wp.vec3),
    body_hessian_ll: wp.array(dtype=wp.mat33), body_hessian_al: wp.array(dtype=wp.mat33),
    body_hessian_aa: wp.array(dtype=wp.mat33),
):
    """Newton's own force and Hessian of each non-chain joint on a chain body, with the other body held as in
    SolverVBD's sweep, added to the body's scratch; `assemble_slot` reads it as external blocks. One thread per body
    sums that body's joints in a fixed order (CSR ext_start / ext_joint, joints ascending). No atomics, so it repeats
    bit for bit (D3, SPEC §3.7). It runs on a freshly zeroed scratch, before the contact accumulation, so no other
    writer touches these bodies."""
    t = wp.tid()
    b = ext_body[t]
    f_sum = wp.vec3(0.0)
    tau_sum = wp.vec3(0.0)
    hll_sum = wp.mat33(0.0)
    hal_sum = wp.mat33(0.0)
    haa_sum = wp.mat33(0.0)
    for k in range(ext_start[t], ext_start[t + 1]):
        f, tau, hll, hal, haa = evaluate_joint_force_hessian(
            b, ext_joint[k], body_q, body_q_prev, body_q_rest, body_com, joint_type, joint_enabled, joint_parent,
            joint_child, joint_X_p, joint_X_c, joint_axis, joint_rod_rest_kb_local, joint_rod_rest_twist,
            joint_qd_start, joint_target_q_start, joint_constraint_start, joint_penalty_k, joint_rho, joint_material_k,
            joint_penalty_kd, joint_sigma_start, joint_C_fric, joint_target_ke, joint_target_kd, joint_target_q,
            joint_target_qd, joint_limit_lower, joint_limit_upper, joint_limit_ke, joint_limit_kd,
            joint_drive_limit_support, joint_drive_lambda, joint_limit_lambda, joint_lambda_lin, joint_lambda_ang,
            joint_C0_lin, joint_C0_ang, joint_is_hard, stab_alpha, joint_compliant_alm, joint_dof_dim,
            joint_rest_angle, joint_angle_prev, dt)
        f_sum = f_sum + f
        tau_sum = tau_sum + tau
        hll_sum = hll_sum + hll
        hal_sum = hal_sum + hal
        haa_sum = haa_sum + haa
    body_forces[b] = body_forces[b] + f_sum
    body_torques[b] = body_torques[b] + tau_sum
    body_hessian_ll[b] = body_hessian_ll[b] + hll_sum
    body_hessian_al[b] = body_hessian_al[b] + hal_sum
    body_hessian_aa[b] = body_hessian_aa[b] + haa_sum


__all__ = ["RodArgs", "RodChains", "mat66", "rod_assemble", "rod_external_joints", "rod_serial", "rod_thomas",
           "rod_update", "vec6"]
