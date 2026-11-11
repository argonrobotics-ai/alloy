# SPDX-License-Identifier: Apache-2.0
"""Contact-aware coarse step of SolverVBDMomentum (SCR-WP11-6 step 3 (D)).

Opt-in: ``make_vbd(..., coarse_contact=True)``.

Why. SolverVBDMomentum minimises the step objective along each free component's translation in closed form, but
skips a component that touches a body: the closed form leaves out the contact forces. Such a component is exactly
the one that converges slowest: a few vertices under a pad must pull a stiff sheet across a frictional table, and
the Gauss-Seidel sweep moves that near-rigid translation a few micrometres per iteration (tests/solvers/vbd/
test_cloth_drag_law.py: 0.17 of the pad's travel at 10 iterations, 0.98 at 160).

What. A component held only by body-contact records (no attachment, no self contact with another component, no pinned
or inactive particle) takes ``NEWTON_STEPS`` Newton steps on its translation delta, on the same objective:

    G(delta) = sum_i m_i / (2 dt^2) |x_i + delta - y_i|^2 + sum_k E_k(x + delta)

Elastic energies are translation-invariant and drop out. E_k is record k's contact (normal, damping, friction),
evaluated by Newton's own ``_eval_soft_ef_contact`` at the shifted contact point (per-vertex and full-surface edge/face
records alike). The record's force and Hessian at the point are the component's (barycentric weights sum to one).
A converged step has delta = 0. The sweep's truncation still writes every position.

When. The step acts from the iteration after SETTLE_ITERATIONS. Iteration 1 meets the inertial guess's deep normal
(contact forces hundreds of times the weight), and the contact step needs the settled normal loads. Each record's
friction load is floored at the normal part of the weight it carries (``record_terms``).

Limit. A light vertex's contact depth is below float32 position resolution away from the world origin, so its contact
force, and the friction load here, can be many times its weight: docs/known_gaps/cloth-contact-depth-float32.md.

Determinism and capture. Records are sorted by component once per step (Warp's stable radix sort), and each
component sums its run in that order in float64. Launch sizes are fixed by the contact buffer.
"""

from __future__ import annotations

import warp as wp

from alloy._vendor.newton._src.solvers.vbd.rigid_vbd_kernels import _eval_soft_ef_contact, compute_projected_isotropic_friction

NEWTON_STEPS = 3
NEAR = 0.25            # a record within this share of the particle radius past contact keeps its weight floor
SETTLE_ITERATIONS = 3  # sweep iterations of a step before the contact step acts (module docstring)


@wp.kernel(enable_backward=False)
def record_keys(count: wp.array[wp.int32], indices: wp.array[wp.vec3i], comp_of: wp.array[wp.int32], sentinel: int,
                keys: wp.array[wp.int32], values: wp.array[wp.int32]):
    """Sort key per record: its component (``sentinel`` for an unused slot or a particle in no component)."""
    k = wp.tid()
    values[k] = k
    keys[k] = sentinel
    if k < count[0]:
        p = indices[k][0]
        if p >= 0:
            c = comp_of[p]
            if c >= 0:
                keys[k] = c


@wp.kernel(enable_backward=False)
def segment_starts(keys: wp.array[wp.int32], n: int, starts: wp.array[wp.int32]):
    """starts[c] = first sorted slot with key >= c (binary search over the sorted keys), c in 0..C."""
    c = wp.tid()
    lo = int(0)
    hi = n
    while lo < hi:
        mid = (lo + hi) // 2
        if keys[mid] < c:
            lo = mid + 1
        else:
            hi = mid
    starts[c] = lo


@wp.kernel(enable_backward=False)
def shifted_positions(comp_of: wp.array[wp.int32], base: wp.array[wp.vec3], disp: wp.array[wp.vec3],
                      delta: wp.array[wp.vec3], active: wp.array[wp.int32], x: wp.array[wp.vec3]):
    i = wp.tid()
    c = comp_of[i]
    x[i] = base[i] + disp[i]
    if c >= 0:
        if active[c] != 0:
            x[i] = x[i] + delta[c]


@wp.struct
class BodyView:
    shape_body: wp.array[int]
    body_q: wp.array[wp.transform]
    body_q_prev: wp.array[wp.transform]
    body_qd: wp.array[wp.spatial_vector]
    body_com: wp.array[wp.vec3]
    shape_margin: wp.array[float]


@wp.struct
class RecordView:
    count: wp.array[wp.int32]
    indices: wp.array[wp.vec3i]
    bary: wp.array[wp.vec3]
    shape: wp.array[int]
    body_pos: wp.array[wp.vec3]
    body_vel: wp.array[wp.vec3]
    normal: wp.array[wp.vec3]
    ke: wp.array[float]
    kd: wp.array[float]
    mu: wp.array[float]


@wp.struct
class ParticleView:
    radius: wp.array[float]
    mass: wp.array[float]
    world: wp.array[wp.int32]
    gravity: wp.array[wp.vec3]


@wp.func
def _record_point(corners: wp.vec3i, bary: wp.vec3, x: wp.array[wp.vec3], pts: ParticleView):
    """The record's contact point, its radius and the mass it carries (barycentric)."""
    v0 = corners[0]
    p = bary[0] * x[v0]
    r = pts.radius[v0]
    m = bary[0] * pts.mass[v0]
    for j in range(1, 3):
        c = corners[j]
        if c >= 0:
            p = p + bary[j] * x[c]
            r = wp.max(r, pts.radius[c])
            m = m + bary[j] * pts.mass[c]
    return p, r, m


@wp.func
def _body_motion(k: int, rec: RecordView, bodies: BodyView, dt: float):
    """The body point of record k and its displacement over the step, as Newton's contact force builds it."""
    b = bodies.shape_body[rec.shape[k]]
    xw = wp.transform_identity()
    xw_prev = wp.transform_identity()
    if b >= 0:
        xw = bodies.body_q[b]
        xw_prev = bodies.body_q_prev[b]
    bx = wp.transform_point(xw, rec.body_pos[k])
    move = wp.transform_vector(xw, rec.body_vel[k]) * dt
    if b >= 0:
        move = move + bx - wp.transform_point(xw_prev, rec.body_pos[k])
    return bx, move


@wp.kernel(enable_backward=False)
def record_terms(rec: RecordView, bodies: BodyView, pts: ParticleView, x: wp.array[wp.vec3],
                 x_prev: wp.array[wp.vec3], eps: float, dt: float, barrier: bool,
                 f_out: wp.array[wp.vec3], h_out: wp.array[wp.mat33], n_out: wp.array[wp.vec3]):
    """Each record's friction force and Hessian at its shifted contact point, with the normal load floored at the
    normal part of the weight the record carries (zero for an unused slot or a separated record).

    The sweep resolves the stiff normal direction; under the log barrier a resting sheet floats ~1 um deep and its
    records flicker to zero normal force between iterations, which would hand the whole component a free slide here.
    The floor keeps the friction a resting or pressed component really has; a record only counts while its point is
    within NEAR of a radius past contact. The weight is ``n_out``'s magnitude, for the component's mean normal."""
    k = wp.tid()
    f_out[k] = wp.vec3(0.0)
    h_out[k] = wp.mat33(0.0)
    n_out[k] = wp.vec3(0.0)
    if k >= rec.count[0] or rec.indices[k][0] < 0:
        return
    corners = rec.indices[k]
    p, radius, mass = _record_point(corners, rec.bary[k], x, pts)
    p_prev, r0, m0 = _record_point(corners, rec.bary[k], x_prev, pts)
    bx, move = _body_motion(k, rec, bodies, dt)
    n = rec.normal[k]
    margin = float(0.0)
    if bodies.shape_margin.shape[0] > 0:
        margin = bodies.shape_margin[rec.shape[k]]
    gap = wp.dot(n, p - bx) - (radius + margin)
    if gap > NEAR * radius:
        return
    f, h, bxx = _eval_soft_ef_contact(k, corners, rec.bary[k], x, x_prev, pts.radius, rec.ke[k], rec.kd[k],
                                      rec.mu[k], eps, bodies.shape_body, bodies.body_q, bodies.body_q_prev,
                                      bodies.body_qd, bodies.body_com, rec.shape, rec.body_pos, rec.body_vel,
                                      rec.normal, bodies.shape_margin, dt, barrier)
    w = pts.world[corners[0]]
    g = pts.gravity[wp.max(w, 0)]
    load = wp.max(wp.dot(f, n), wp.max(0.0, -wp.dot(g, n)) * mass)
    u = (p - p_prev) - move
    ff, kf = compute_projected_isotropic_friction(rec.mu[k], load, n, u, eps * dt)
    f_out[k] = ff
    h_out[k] = kf
    n_out[k] = load * n


@wp.kernel(enable_backward=False)
def component_newton(active: wp.array[wp.int32], comp_chunk_start: wp.array[wp.int32], g_chunk: wp.array[wp.vec3d],
                     m_chunk: wp.array[wp.float64], starts: wp.array[wp.int32], order: wp.array[wp.int32],
                     f_rec: wp.array[wp.vec3], h_rec: wp.array[wp.mat33], n_rec: wp.array[wp.vec3], dt: float,
                     delta: wp.array[wp.vec3]):
    """One Newton step on an active component's translation within the plane normal to its force-weighted contact
    normal: grad = P ((sum m (x - y) + M delta) / dt^2) - F_t, Hess = P (M / dt^2) P + H_t + (M / dt^2) n n^T (the
    normal block only keeps it invertible), delta -= P Hess^-1 grad. Sums in float64, records in sorted order."""
    c = wp.tid()
    if active[c] == 0:
        return
    g = wp.vec3d(0.0, 0.0, 0.0)
    msum = wp.float64(0.0)
    for k in range(comp_chunk_start[c], comp_chunk_start[c + 1]):
        g = g + g_chunk[k]
        msum = msum + m_chunk[k]
    fsum = wp.vec3d(0.0, 0.0, 0.0)
    nsum = wp.vec3d(0.0, 0.0, 0.0)
    hsum = wp.mat33d(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for j in range(starts[c], starts[c + 1]):
        r = order[j]
        fsum = fsum + wp.vec3d(f_rec[r])
        nsum = nsum + wp.vec3d(n_rec[r])
        hsum = hsum + wp.mat33d(h_rec[r])
    nl = wp.length(nsum)
    if nl <= wp.float64(0.0):
        return                                            # no record pressing: the sweep handles it
    n = nsum / nl
    p = wp.identity(3, wp.float64) - wp.outer(n, n)
    w = msum / (wp.float64(dt) * wp.float64(dt))
    d = wp.vec3d(delta[c])
    grad = p * ((g / (wp.float64(dt) * wp.float64(dt))) + w * d) - fsum
    hess = hsum + w * wp.identity(3, wp.float64)
    det = wp.determinant(hess)
    if det > wp.float64(0.0):
        step = p * (wp.inverse(hess) * grad)
        delta[c] = wp.vec3(d - step)


@wp.kernel(enable_backward=False)
def mark_active(touching: wp.array[wp.int32], blocked: wp.array[wp.int32], comp_chunk_start: wp.array[wp.int32],
                pinned_chunk: wp.array[wp.int32], active: wp.array[wp.int32], delta: wp.array[wp.vec3]):
    """A component takes the contact step when it touches a body and nothing else blocks it (no pinned particle)."""
    c = wp.tid()
    pinned = int(0)
    for k in range(comp_chunk_start[c], comp_chunk_start[c + 1]):
        pinned = pinned + pinned_chunk[k]
    a = int(0)
    if touching[c] != 0 and blocked[c] == 0 and pinned == 0:
        a = 1
        delta[c] = wp.vec3(0.0)
    active[c] = a


@wp.kernel(enable_backward=False)
def merge_delta(active: wp.array[wp.int32], contact_delta: wp.array[wp.vec3], delta: wp.array[wp.vec3]):
    c = wp.tid()
    if active[c] != 0:
        delta[c] = contact_delta[c]


__all__ = ["NEAR", "NEWTON_STEPS", "SETTLE_ITERATIONS", "BodyView", "ParticleView", "RecordView", "component_newton",
           "mark_active", "merge_delta", "record_keys", "record_terms", "segment_starts", "shifted_positions"]
