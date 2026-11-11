# SPDX-License-Identifier: Apache-2.0
"""Film grip: friction for film vertices held by rigid contacts (SCR-WP13-GRIP-1; docs/notes/film-grip-c16.md).

C16's straps slipped out of a 30 N pinch carrying a 3 g bag. A static-pad pinch on a hanging HDPE 15 um strip shows two
gaps, each enough to let a loaded film walk out of any pinch:

1. **The strain-limit projection ignores friction.** SolverFilm carries in-plane stiffness in its projection (VBD
   membrane at 1e-4), which runs after the VBD solve and shifts every film vertex by inverse mass. A pinched vertex's
   shift becomes the next step's start, so friction never resists it. Once the strip reaches the strain cap the pinch
   walks out: 10 mm in 0.5 s, falling out by 1 s, at a 500x friction margin.
   Fix, ``FilmConfig.projection_friction``: on a vertex in body contact the projection's shift
   - never goes deeper into a contact (the contact guard): in a pinch, both normal directions are blocked; on a
     one-sided contact such as a table, the shift may still move away from the surface. Without the guard the
     projection pushed pinched vertices into the jaws, and the contact force pried force-limited jaws apart (WP4's
     coupled pinch: the gap opened past 1 mm and the strip fell);
   - keeps only the part of its tangential shift beyond the friction impulse its contacts can give in one step,
     mu·f_n·dt²/m. This is position-based friction on the projection, as PBD applies it to contacts. Below mu·N the
     vertex holds; above it, it slips by the excess.
     A sliding vertex gives none. It is sliding when it enters the step moving past its contacts faster than their
     friction impulse can stop it, |v_rel|·m > mu·f_n·dt; its friction is then VBD's kinetic friction, already
     applied, so a dragged film feels Coulomb's mu_k, not twice it (the whole capacity on a sliding vertex gave
     mu_k 0.40 against 0.3; SOFT-1 review). The test reads only the step's start, not VBD's result: on a light
     pinched vertex VBD's friction (a reaction at the stiffness 2·mu·f_n/eps_u) does not converge in 10
     iterations, and budgets that read it let WP4's coupled pinch walk 10.4 mm (the residual mu·f_n - |f_t|),
     14.9 mm (a friction disk around its 20 ms average) or fall out (VBD's slip past eps_u as the sliding test);
     the whole capacity holds it to 0.003 mm.
   The clamp runs once, after the projection's Jacobi sweeps, on their summed shift.
   The normal directions are the strong eigen-directions of the contacts' tensor sum(f_n·n nᵀ), which a pinch's
   opposite normals add to instead of cancelling.
2. **Newton's VBD friction has no stiffness at exactly zero slip.** ``compute_projected_isotropic_friction`` returns a
   zero Hessian when the tangential slip is exactly 0, though its limit is mu·f_n·2/eps_u. Away from the origin a
   stuck vertex's slip rounds to exactly 0 in float32: at z = 0.5 m one ulp is 6e-8 m, while a static slip is about
   1e-10 m. On the next iteration only inertia holds the vertex, so a pinch creeps millimetres per second. Near the
   origin, or with one VBD iteration, there is no creep.
   Fix, ``FilmConfig.static_friction_hessian``: an after_accumulate colour callback adds the limit on exactly those
   contacts, repeating Newton's slip arithmetic. Cloth (SolverVBDAlloy) has the same gap; the clean fix is the limit
   inside Newton's friction function.

Both fixes keep no state between steps (the start velocities are copied each step), so coupling iterations that
re-step the film are unaffected. Both act only
on particles with a body contact, so a free film steps bitwise as before. Only particle records
(p, -1, -1) take part: the edge/face records of full-surface contacts keep Newton's friction as is. The Newton
internals imported here are checked by canary tests; if they move, SolverFilm logs a warning and steps without grip.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from alloy._vendor.newton._src.geometry.soft_contacts_mesh import (_make_soft_contact_filter_data, _soft_contact_force_enabled,
                                                     _SoftContactFilterData)
from alloy._vendor.newton._src.solvers.vbd.rigid_vbd_kernels import (_evaluate_rigid_soft_contact_force_norm,
                                                       compute_projected_isotropic_friction)


FORCE_UNIT = 1.0e-9       # N per count of the int64 per-particle sums (no float atomics: SPEC §3.7 D3; max 9e9 N)
STIFF_UNIT = 1.0e-3       # N/m per count of the zero-slip Hessian sums (max 9e15 N/m)
CAP, TENSOR, SIDE, VREL = 0, 1, 7, 10  # acc columns: mu·f_n; f_n·n nᵀ (xx yy zz xy xz yz); f_n·n; f_n·v_rel,t
ACC_COLS = 13


@wp.func
def _pose(body_q: wp.array[wp.transform], b: int) -> wp.transform:
    if b >= 0:
        return body_q[b]
    return wp.transform_identity()


@wp.func
def _fix(x: float, unit: float) -> wp.int64:
    return wp.int64(wp.round(x / unit))


@wp.func
def _add_sym(acc: wp.array2d[wp.int64], p: int, col: int, M: wp.mat33, unit: float):
    """Integer atomics of a symmetric 3x3 (xx yy zz xy xz yz) into acc[p, col:col+6]: exact, order-free sums."""
    wp.atomic_add(acc, p, col + 0, _fix(M[0, 0], unit))
    wp.atomic_add(acc, p, col + 1, _fix(M[1, 1], unit))
    wp.atomic_add(acc, p, col + 2, _fix(M[2, 2], unit))
    wp.atomic_add(acc, p, col + 3, _fix(M[0, 1], unit))
    wp.atomic_add(acc, p, col + 4, _fix(M[0, 2], unit))
    wp.atomic_add(acc, p, col + 5, _fix(M[1, 2], unit))


@wp.func
def _sym(acc: wp.array2d[wp.int64], p: int, col: int, unit: float) -> wp.mat33:
    a = float(acc[p, col + 0]) * unit
    b = float(acc[p, col + 1]) * unit
    c = float(acc[p, col + 2]) * unit
    xy = float(acc[p, col + 3]) * unit
    xz = float(acc[p, col + 4]) * unit
    yz = float(acc[p, col + 5]) * unit
    return wp.mat33(a, xy, xz, xy, b, yz, xz, yz, c)


@wp.func
def _contact_body_velocity(b: int, X: wp.transform, bx: wp.vec3, cpos: wp.vec3, cvel: wp.vec3,
                           body_q_prev: wp.array[wp.transform], use_prev: int, body_qd: wp.array[wp.spatial_vector],
                           body_com: wp.array[wp.vec3], dt: float) -> wp.vec3:
    """The contact point's velocity exactly as Newton's _eval_body_particle_contact computes it (same operations, same
    order), so the zero-slip test agrees with Newton's bit for bit."""
    if use_prev != 0:
        bx_prev = wp.transform_point(_pose(body_q_prev, b), cpos)
        return (bx - bx_prev) / dt + wp.transform_vector(X, cvel)
    vs = wp.spatial_vector()
    com = wp.vec3(0.0)
    if b >= 0:
        vs = body_qd[b]
        com = body_com[b]
    rr = bx - wp.transform_point(X, com)
    return wp.spatial_top(vs) + wp.cross(wp.spatial_bottom(vs), rr) + wp.transform_vector(X, cvel)


@wp.struct
class ContactView:
    """A step's particle-body contacts and the body poses Newton's particle iteration reads (SolverVBD
    ``_solve_particle_iteration``), so the slip here is Newton's slip bit for bit."""
    count: wp.array[wp.int32]
    max_count: int
    corners: wp.array[wp.vec3i]
    shape: wp.array[wp.int32]
    body_pos: wp.array[wp.vec3]
    body_vel: wp.array[wp.vec3]
    normal: wp.array[wp.vec3]
    shape_body: wp.array[wp.int32]
    shape_margin: wp.array[float]
    radius: wp.array[float]
    ke: wp.array[float]
    mu: wp.array[float]
    log_barrier: int
    body_q: wp.array[wp.transform]
    body_q_prev: wp.array[wp.transform]
    use_prev: int
    body_qd: wp.array[wp.spatial_vector]
    body_com: wp.array[wp.vec3]
    anchor: wp.array[wp.vec3]          # the step's start positions (VBD's friction anchor)
    v0: wp.array[wp.vec3]              # the step's start velocities
    dt: float
    eps_u: float                       # friction_epsilon·dt


@wp.func
def _contact_force(cv: ContactView, c: int, p: int, x: wp.array[wp.vec3]) -> float:
    """Newton's normal force f_n of contact c on particle p at x (0 when not penetrating)."""
    s = cv.shape[c]
    margin = float(0.0)
    if cv.shape_margin.shape[0] > 0:
        margin = cv.shape_margin[s]
    r = cv.radius[p] + margin
    d = wp.dot(cv.normal[c], x[p] - wp.transform_point(_pose(cv.body_q, cv.shape_body[s]), cv.body_pos[c]))
    if r - d <= 0.0:
        return 0.0
    dE, d2E = _evaluate_rigid_soft_contact_force_norm(d, r, cv.ke[c], cv.log_barrier != 0)
    return wp.max(-dE, 0.0)


@wp.func
def _body_velocity(cv: ContactView, c: int) -> wp.vec3:
    """The velocity of contact c's body point over the step, as Newton's particle iteration takes it."""
    b = cv.shape_body[cv.shape[c]]
    X = _pose(cv.body_q, b)
    bx = wp.transform_point(X, cv.body_pos[c])
    return _contact_body_velocity(b, X, bx, cv.body_pos[c], cv.body_vel[c], cv.body_q_prev, cv.use_prev, cv.body_qd,
                                  cv.body_com, cv.dt)


@wp.func
def _slip(cv: ContactView, c: int, p: int, x: wp.array[wp.vec3]) -> wp.vec3:
    """Newton's relative translation over the step (rigid_vbd_kernels._eval_body_particle_contact)."""
    return (x[p] - cv.anchor[p]) - _body_velocity(cv, c) * cv.dt




@wp.kernel(enable_backward=False)
def contact_capacity(cv: ContactView, x: wp.array[wp.vec3], flt: _SoftContactFilterData, acc: wp.array2d[wp.int64]):
    """Per film particle over its penetrating contacts at the VBD result x, in FORCE_UNIT counts: mu·f_n (friction
    capacity), f_n·n nᵀ (sign-free: a pinch's opposite normals add up), f_n·n (signed: they cancel in a pinch) and
    f_n times the tangential velocity at which the vertex entered the step relative to the body point."""
    c = wp.tid()
    if c >= wp.min(cv.count[0], cv.max_count):
        return
    ids = cv.corners[c]
    if ids[1] >= 0 or not _soft_contact_force_enabled(c, flt):
        return
    p = ids[0]
    fn = _contact_force(cv, c, p, x)
    if fn <= 0.0:
        return
    n = cv.normal[c]
    wp.atomic_add(acc, p, CAP, _fix(cv.mu[c] * fn, FORCE_UNIT))
    _add_sym(acc, p, TENSOR, fn * wp.outer(n, n), FORCE_UNIT)
    v = cv.v0[p] - _body_velocity(cv, c)
    v = v - n * wp.dot(n, v)
    for k in range(3):
        wp.atomic_add(acc, p, SIDE + k, _fix(fn * n[k], FORCE_UNIT))
        wp.atomic_add(acc, p, VREL + k, _fix(fn * v[k], FORCE_UNIT))


@wp.func
def guarded_shift(M: wp.mat33, side: wp.vec3, budget: float, d: wp.vec3) -> wp.vec3:
    """The projection shift ``d`` of a contact vertex, after contact and friction. M = sum f_n·n nᵀ, side = sum f_n·n.
    Normal subspace: the eigen-directions q of M with at least half its largest eigenvalue (a pinch, a flat contact:
    the normal; a wedge: both). Along each q the shift may not go deeper into a contact: a two-sided q (a pinch:
    |side·q| < λ/2) loses its component, a one-sided q only an inward one (the contact guard). The tangential rest
    loses up to ``budget`` = mu·f_n·dt²/m, or nothing on a sliding vertex (zero below it: hold; the excess: slip)."""
    Q, e = wp.eig3(M)
    top = wp.max(e[0], wp.max(e[1], e[2]))
    t = d
    normal = wp.vec3(0.0)
    if top > 0.0:
        for i in range(3):
            if e[i] >= 0.5 * top:
                q = wp.vec3(Q[0, i], Q[1, i], Q[2, i])
                c = wp.dot(q, d)
                t = t - q * c
                s = wp.dot(side, q)
                if wp.abs(s) >= 0.5 * e[i] and c * s > 0.0:
                    normal = normal + q * c                 # one-sided, moving away from the contact: free
    lt = wp.length(t)
    if lt <= budget:
        return normal
    return normal + t * (1.0 - budget / lt)


@wp.kernel(enable_backward=False)
def friction_clamp(vtx_particle: wp.array[wp.int32], inv_mass: wp.array[float], acc: wp.array2d[wp.int64], dt2: float,
                   x: wp.array[wp.vec3], dx_total: wp.array[wp.vec3]):
    """Contact vertices: the projection's shift after the contact guard and friction (``guarded_shift``)."""
    v = wp.tid()
    p = vtx_particle[v]
    if acc[p, CAP] <= wp.int64(0) or inv_mass[p] <= 0.0:
        return
    cap = float(acc[p, CAP]) * FORCE_UNIT
    fn = float(acc[p, TENSOR] + acc[p, TENSOR + 1] + acc[p, TENSOR + 2]) * FORCE_UNIT      # trace: sum f_n
    v_rel = wp.vec3(float(acc[p, VREL]), float(acc[p, VREL + 1]), float(acc[p, VREL + 2])) * (FORCE_UNIT / fn)
    budget = cap * dt2 * inv_mass[p]
    if wp.length(v_rel) * wp.sqrt(dt2) > budget:          # entered faster than friction stops it in one step
        budget = 0.0
    d = dx_total[p]
    side = wp.vec3(float(acc[p, SIDE]), float(acc[p, SIDE + 1]), float(acc[p, SIDE + 2])) * FORCE_UNIT
    g = guarded_shift(_sym(acc, p, TENSOR, FORCE_UNIT), side, budget, d)
    x[p] = x[p] + (g - d)
    dx_total[p] = g


@wp.kernel(enable_backward=False)
def zero_slip_hessian(cv: ContactView, colour: int, colors: wp.array[wp.int32], pos: wp.array[wp.vec3],
                      flt: _SoftContactFilterData, acc: wp.array2d[wp.int64]):
    """VBD after_accumulate slot: where Newton's friction saw a tangential slip of exactly 0 (and so added no
    Hessian), add its limit mu·f_n·(2/eps_u) on the tangent plane. ``pos`` is the iterate; the slip uses Newton's own
    arithmetic (``_slip``) so the zero test matches it bit for bit."""
    c = wp.tid()
    if c >= wp.min(cv.count[0], cv.max_count):
        return
    ids = cv.corners[c]
    if ids[1] >= 0:
        return
    p = ids[0]
    if colors[p] != colour or not _soft_contact_force_enabled(c, flt):
        return
    fn = _contact_force(cv, c, p, pos)
    if fn <= 0.0:
        return
    n = cv.normal[c]
    u = _slip(cv, c, p, pos)
    if wp.length(u - n * wp.dot(n, u)) > 0.0:
        return
    k = cv.mu[c] * fn * 2.0 / cv.eps_u
    _add_sym(acc, p, 0, k * (wp.identity(3, float) - wp.outer(n, n)), STIFF_UNIT)


@wp.kernel(enable_backward=False)
def apply_hessian(colour: int, colors: wp.array[wp.int32], acc: wp.array2d[wp.int64],
                  hessians: wp.array[wp.mat33]):
    """Adds this colour's zero-slip sums to Newton's particle Hessians and clears them (one thread per particle)."""
    p = wp.tid()
    if colors[p] != colour:
        return
    if acc[p, 0] == 0 and acc[p, 1] == 0 and acc[p, 2] == 0 and acc[p, 3] == 0 and acc[p, 4] == 0 and acc[p, 5] == 0:
        return
    hessians[p] = hessians[p] + _sym(acc, p, 0, STIFF_UNIT)
    for k in range(6):
        acc[p, k] = wp.int64(0)


def _empty(dtype, dev) -> wp.array:
    return wp.zeros(1, dtype=dtype, device=dev)


class FilmGrip:
    """SolverFilm's grip terms (module doc): the colour-slot Hessian and the projection's friction clamp."""

    def __init__(self, solver: Any, projection: bool, hessian: bool) -> None:
        m, dev = solver.model, solver.model.device
        self.solver, self.projection, self.hessian = solver, projection, hessian
        self.acc = wp.zeros((int(m.particle_count), ACC_COLS), dtype=wp.int64, device=dev)
        self.v0 = wp.zeros(int(m.particle_count), dtype=wp.vec3, device=dev) if projection else None  # start vel.
        self.acc_h = wp.zeros((int(m.particle_count), 6), dtype=wp.int64, device=dev)      # zero-slip Hessians
        self.dt = 0.0
        self._no_q, self._no_qd = _empty(wp.transform, dev), _empty(wp.spatial_vector, dev)
        self._no_com = _empty(wp.vec3, dev)
        if hessian:
            solver.add_colour_callback(self.hessian_slot, external=False, when="after_accumulate")

    def _bodies(self, state_in: Any, state_out: Any) -> tuple[wp.array, wp.array | None, wp.array]:
        """(body_q, body_q_prev or None, body_qd) as Newton's particle iteration reads them."""
        s, m = self.solver, self.solver.model
        if m.body_count == 0 or state_in.body_q is None:
            return self._no_q, None, self._no_qd
        if s.integrate_with_external_rigid_solver:
            return state_out.body_q, state_in.body_q, state_out.body_qd
        return state_in.body_q, s.body_q_prev, state_in.body_qd

    def _view(self, contacts: Any) -> ContactView:
        """The step's contacts and body poses as Newton's particle iteration reads them (one struct per step)."""
        s, m = self.solver, self.solver.model
        bq, bqp, bqd = self._bodies(self._state_in, self._state_out)
        cv = ContactView()
        cv.count, cv.max_count, cv.corners = contacts.soft_contact_count, contacts.soft_contact_max, \
            contacts.soft_contact_indices
        cv.shape, cv.body_pos, cv.body_vel = contacts.soft_contact_shape, contacts.soft_contact_body_pos, \
            contacts.soft_contact_body_vel
        cv.normal, cv.shape_body, cv.shape_margin, cv.radius = contacts.soft_contact_normal, m.shape_body, \
            m.shape_margin, m.particle_radius
        cv.ke, cv.mu = s.body_particle_contact_penalty_k, s.body_particle_contact_material_mu
        cv.log_barrier = int(bool(s.rigid_soft_contact_use_log_barrier))
        cv.body_q, cv.body_q_prev, cv.use_prev = bq, bqp if bqp is not None else bq, int(bqp is not None)
        cv.body_qd, cv.body_com = bqd, m.body_com if m.body_count else self._no_com
        cv.anchor, cv.dt, cv.eps_u = s.particle_q_prev, float(self.dt), float(s.friction_epsilon * self.dt)
        cv.v0 = self.v0 if self.v0 is not None else self._no_com
        return cv

    def _filter(self, contacts: Any, x: wp.array) -> Any:
        m = self.solver.model
        bq = self._bodies(self._state_in, self._state_out)[0]
        return _make_soft_contact_filter_data(m, contacts, x, bq if m.body_count else None)

    def hessian_slot(self, solver: Any, colour: int, state: Any, contacts: Any) -> None:
        """after_accumulate callback: ``state`` is the iterate (Newton iterates in state_in)."""
        if contacts is None or contacts.soft_contact_max == 0 or self.dt <= 0.0:
            return
        m = solver.model
        wp.launch(zero_slip_hessian, dim=contacts.soft_contact_max,
                  inputs=[self._view(contacts), colour, m.particle_colors, state.particle_q,
                          self._filter(contacts, state.particle_q)],
                  outputs=[self.acc_h], device=m.device)
        wp.launch(apply_hessian, dim=m.particle_count, inputs=[colour, m.particle_colors, self.acc_h],
                  outputs=[solver.particle_hessians], device=m.device)

    def begin_step(self, state_in: Any, state_out: Any, dt: float) -> None:
        """SolverFilm.step, before the VBD solve: the slot and the clamp need dt and the step's body arrays."""
        self.dt, self._state_in, self._state_out = float(dt), state_in, state_out
        if self.v0 is not None and state_in.particle_qd is not None:
            wp.copy(self.v0, state_in.particle_qd)

    def capacity(self, state_out: Any, contacts: Any) -> bool:
        """Before the projection: each film particle's contact sums at the VBD result. False: no contacts."""
        if not self.projection or contacts is None or contacts.soft_contact_max == 0:
            return False
        m = self.solver.model
        self.acc.zero_()
        wp.launch(contact_capacity, dim=contacts.soft_contact_max,
                  inputs=[self._view(contacts), state_out.particle_q, self._filter(contacts, state_out.particle_q)],
                  outputs=[self.acc], device=m.device)
        return True

    def clamp(self, particles: wp.array, x: wp.array, dx: wp.array, dt: float) -> None:
        """After the projection's sweeps, before the velocity update reads ``dx``."""
        m = self.solver.model
        wp.launch(friction_clamp, dim=particles.shape[0],
                  inputs=[particles, m.particle_inv_mass, self.acc, float(dt * dt), x, dx],
                  device=m.device)
