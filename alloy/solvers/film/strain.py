# SPDX-License-Identifier: Apache-2.0
"""Strain-limit projection (Wang, O'Brien, Ramamoorthi 2010 isotropic strain limiting, Jacobi form).

Per limited triangle: deformation gradient F = Ds·Dm⁻¹ (3x2, Dm⁻¹ = ``model.tri_poses``), principal stretches
s1 >= s2 from FᵀF, clamped to [1 - compress_cap, 1 + strain_cap]; the clamped gradient F' = F·V·diag(s'/s)·Vᵀ maps
the rest triangle about its mass centre (Wang 2010: "the triangle's mass centre stays fixed"; with one pinned
vertex about that vertex, with two about their midpoint; three pinned: no change). Each vertex then moves by the
relaxed sum of its limited triangles' (and long-range pairs') corrections, each constraint scaled by the relaxation
over the active-constraint count of its busiest vertex (Jacobi that keeps every constraint's mass centre, so the
projection conserves momentum; gathered in CSR order: deterministic, no atomics). Pinned particles (inverse mass 0) never move. Velocities get Δx/dt so x and v stay consistent.
"""

from __future__ import annotations

import warp as wp

EPS = wp.constant(1.0e-12)


@wp.func
def principal_frame(f0: wp.vec3, f1: wp.vec3):
    """Eigen-decomposition of C = FᵀF (2x2, symmetric): (s1, s2, v1) with s1 >= s2 >= 0 and v1 the unit
    eigenvector of s1 (v2 = perp(v1))."""
    a = wp.dot(f0, f0)
    b = wp.dot(f0, f1)
    c = wp.dot(f1, f1)
    m = 0.5 * (a + c)
    d = wp.sqrt(0.25 * (a - c) * (a - c) + b * b)
    l1 = m + d
    l2 = wp.max(m - d, 0.0)
    v1 = wp.vec2(1.0, 0.0)
    if wp.abs(b) > 1.0e-12 * (a + c):
        v1 = wp.normalize(wp.vec2(b, l1 - a))
    elif c > a:
        v1 = wp.vec2(0.0, 1.0)
    return wp.sqrt(l1), wp.sqrt(l2), v1


@wp.func
def clamp_ratio(s: float, lo: float, hi: float) -> float:
    """s'/s for s clamped to [lo, hi]; 1 when s is degenerate (a collapsed triangle has no direction to scale)."""
    if s < 1.0e-9:
        return 1.0
    return wp.clamp(s, lo, hi) / s


@wp.kernel(enable_backward=False)
def tri_corrections(
    x: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    tri_indices: wp.array2d[wp.int32],
    tri_poses: wp.array[wp.mat22],
    strain_cap: wp.array[float],
    compress_cap: wp.array[float],
    mass_ratio_cap: float,
    delta: wp.array2d[wp.vec3],
    active: wp.array[wp.int32],
):
    """delta[t, k] = correction of triangle t's k-th vertex; active[t] = 1 if t was outside its limits."""
    t = wp.tid()
    delta[t, 0] = wp.vec3(0.0)
    delta[t, 1] = wp.vec3(0.0)
    delta[t, 2] = wp.vec3(0.0)
    active[t] = 0
    cap = strain_cap[t]
    if cap <= 0.0:
        return
    i = tri_indices[t, 0]
    j = tri_indices[t, 1]
    k = tri_indices[t, 2]
    dinv = tri_poses[t]
    e1 = x[j] - x[i]
    e2 = x[k] - x[i]
    f0 = e1 * dinv[0, 0] + e2 * dinv[1, 0]
    f1 = e1 * dinv[0, 1] + e2 * dinv[1, 1]
    s1, s2, v1 = principal_frame(f0, f1)
    hi = 1.0 + cap
    lo = 1.0 - compress_cap[t]
    if s1 <= hi and s2 <= hi and s2 >= lo:
        return
    r1 = clamp_ratio(s1, lo, hi)
    r2 = clamp_ratio(s2, lo, hi)
    v2 = wp.vec2(-v1[1], v1[0])
    m00 = r1 * v1[0] * v1[0] + r2 * v2[0] * v2[0]
    m01 = r1 * v1[0] * v1[1] + r2 * v2[0] * v2[1]
    m11 = r1 * v1[1] * v1[1] + r2 * v2[1] * v2[1]
    g0 = f0 * m00 + f1 * m01          # F' = F·M, M symmetric
    g1 = f0 * m01 + f1 * m11
    dm = wp.inverse(dinv)             # rest edge matrix: X_i = 0, X_j = dm[:,0], X_k = dm[:,1]
    X1 = wp.vec2(dm[0, 0], dm[1, 0])
    X2 = wp.vec2(dm[0, 1], dm[1, 1])
    wi = inv_mass[i]
    wj = inv_mass[j]
    wk = inv_mass[k]
    pinned = int(wi == 0.0) + int(wj == 0.0) + int(wk == 0.0)
    if pinned == 3:
        return
    # weights: inverse of inverse mass, or the pinned set alone (infinite mass)
    mi = float(0.0)
    mj = float(0.0)
    mk = float(0.0)
    if pinned > 0:
        mi = float(wi == 0.0)
        mj = float(wj == 0.0)
        mk = float(wk == 0.0)
    else:
        mi = 1.0 / wi
        mj = 1.0 / wj
        mk = 1.0 / wk
        if mass_ratio_cap > 0.0:
            mmax = mass_ratio_cap * wp.min(mi, wp.min(mj, mk))
            mi = wp.min(mi, mmax)
            mj = wp.min(mj, mmax)
            mk = wp.min(mk, mmax)
    msum = mi + mj + mk
    # everything relative to x_i: positions are ~1 m in float32, corrections ~1 um (no cancellation)
    c = (e1 * mj + e2 * mk) / msum
    C = (X1 * mj + X2 * mk) / msum
    Xi = -C
    Xj = X1 - C
    Xk = X2 - C
    if wi > 0.0:
        delta[t, 0] = c + g0 * Xi[0] + g1 * Xi[1]
    if wj > 0.0:
        delta[t, 1] = c + g0 * Xj[0] + g1 * Xj[1] - e1
    if wk > 0.0:
        delta[t, 2] = c + g0 * Xk[0] + g1 * Xk[1] - e2
    active[t] = 1


@wp.kernel(enable_backward=False)
def vertex_count(
    vtx_start: wp.array[wp.int32],
    vtx_tri: wp.array[wp.int32],
    vtx_particle: wp.array[wp.int32],
    active: wp.array[wp.int32],
    pair_start: wp.array[wp.int32],
    pair_id: wp.array[wp.int32],
    pair_active: wp.array[wp.int32],
    count: wp.array[wp.int32],
):
    """count[p] = active constraints (limited triangles and long-range pairs) touching film particle p."""
    v = wp.tid()
    n = int(0)
    for e in range(vtx_start[v], vtx_start[v + 1]):
        n = n + active[vtx_tri[e]]
    for e in range(pair_start[v], pair_start[v + 1]):
        n = n + pair_active[pair_id[e]]
    count[vtx_particle[v]] = n


@wp.func
def constraint_scale(mode: int, relaxation: float, n_self: int, n_max: int) -> float:
    """mode 0 (momentum): every correction of a constraint scaled by relaxation / (its busiest vertex's count),
    so each constraint's correction set keeps its mass centre and the projection conserves momentum; mode 1
    (vertex averaging): relaxation / this vertex's count (faster, not momentum-conserving)."""
    if mode == 1:
        return relaxation / float(n_self)
    return relaxation / float(n_max)


@wp.kernel(enable_backward=False)
def vertex_apply(
    vtx_start: wp.array[wp.int32],
    vtx_tri: wp.array[wp.int32],
    vtx_slot: wp.array[wp.int32],
    vtx_particle: wp.array[wp.int32],
    tri_indices: wp.array2d[wp.int32],
    delta: wp.array2d[wp.vec3],
    active: wp.array[wp.int32],
    pair_start: wp.array[wp.int32],
    pair_id: wp.array[wp.int32],
    pair_end: wp.array[wp.int32],
    pairs: wp.array2d[wp.int32],
    pair_delta: wp.array2d[wp.vec3],
    pair_active: wp.array[wp.int32],
    count: wp.array[wp.int32],
    relaxation: float,
    mode: int,
    x: wp.array[wp.vec3],
    dx_total: wp.array[wp.vec3],
):
    """One thread per film particle: x += Σ over its active constraints of scale · correction (``constraint_scale``)."""
    v = wp.tid()
    p = vtx_particle[v]
    n_self = count[p]
    if n_self == 0:
        return
    acc = wp.vec3(0.0)
    for e in range(vtx_start[v], vtx_start[v + 1]):
        t = vtx_tri[e]
        if active[t] != 0:
            nm = wp.max(count[tri_indices[t, 0]], wp.max(count[tri_indices[t, 1]], count[tri_indices[t, 2]]))
            acc = acc + delta[t, vtx_slot[e]] * constraint_scale(mode, relaxation, n_self, nm)
    for e in range(pair_start[v], pair_start[v + 1]):
        c = pair_id[e]
        if pair_active[c] != 0:
            nm = wp.max(count[pairs[c, 0]], count[pairs[c, 1]])
            acc = acc + pair_delta[c, pair_end[e]] * constraint_scale(mode, relaxation, n_self, nm)
    x[p] = x[p] + acc
    dx_total[p] = dx_total[p] + acc


@wp.kernel(enable_backward=False)
def vertex_reaction(
    vtx_start: wp.array[wp.int32],
    vtx_tri: wp.array[wp.int32],
    vtx_particle: wp.array[wp.int32],
    tri_indices: wp.array2d[wp.int32],
    delta: wp.array2d[wp.vec3],
    active: wp.array[wp.int32],
    pair_start: wp.array[wp.int32],
    pair_id: wp.array[wp.int32],
    pair_end: wp.array[wp.int32],
    pairs: wp.array2d[wp.int32],
    pair_delta: wp.array2d[wp.vec3],
    pair_active: wp.array[wp.int32],
    count: wp.array[wp.int32],
    inv_mass: wp.array[float],
    relaxation: float,
    mode: int,
    inv_dt2: float,
    reaction: wp.array[wp.vec3],
):
    """One thread per film particle; pinned ones only (inverse mass 0): the force this sweep's constraints put on it.
    A constraint moves only its free vertices (a pinned vertex's correction is 0), so the momentum it gives them,
    Σ m_k·δ_k·scale_k, comes through its pinned vertices: each takes −1/(their number) of it, over dt² (the XPBD
    reaction λ∇C of a translation-invariant constraint). Read only (``SolverFilm.projection_reaction``). The equal
    split among a constraint's pinned vertices fixes their sum, not each one (that also needs the moment balance):
    sum over a body's pinned vertices before reading a force, not a torque (WP4's review of slice 1)."""
    v = wp.tid()
    p = vtx_particle[v]
    if inv_mass[p] > 0.0 or count[p] == 0:
        return
    acc = wp.vec3(0.0)
    for e in range(vtx_start[v], vtx_start[v + 1]):
        t = vtx_tri[e]
        if active[t] != 0:
            nm = wp.max(count[tri_indices[t, 0]], wp.max(count[tri_indices[t, 1]], count[tri_indices[t, 2]]))
            moved = wp.vec3(0.0)
            pinned = int(0)
            for k in range(3):
                q = tri_indices[t, k]
                if inv_mass[q] > 0.0:
                    moved = moved + delta[t, k] * (constraint_scale(mode, relaxation, count[q], nm) / inv_mass[q])
                else:
                    pinned = pinned + 1
            acc = acc - moved / float(pinned)
    for e in range(pair_start[v], pair_start[v + 1]):
        c = pair_id[e]
        if pair_active[c] != 0:
            o = 1 - pair_end[e]
            q = pairs[c, o]
            if inv_mass[q] > 0.0:
                nm = wp.max(count[pairs[c, 0]], count[pairs[c, 1]])
                acc = acc - pair_delta[c, o] * (constraint_scale(mode, relaxation, count[q], nm) / inv_mass[q])
    reaction[p] = reaction[p] + acc * inv_dt2


@wp.kernel(enable_backward=False)
def velocity_from_shift(vtx_particle: wp.array[wp.int32], dx_total: wp.array[wp.vec3], inv_dt: float,
                        qd: wp.array[wp.vec3]):
    """v += Δx/dt for the projection's total shift (x and v stay the backward-Euler pair VBD produced); the caller
    zeroes dx_total after the ledger has read it."""
    v = wp.tid()
    p = vtx_particle[v]
    qd[p] = qd[p] + dx_total[p] * inv_dt


@wp.kernel(enable_backward=False)
def tri_max_strain(x: wp.array[wp.vec3], tri_indices: wp.array2d[wp.int32], tri_poses: wp.array[wp.mat22],
                   strain_cap: wp.array[float], out: wp.array[float]):
    """out[t] = (s1 - 1)/strain_cap of each limited triangle (> 1: over its cap), 0 for unlimited triangles."""
    t = wp.tid()
    out[t] = 0.0
    cap = strain_cap[t]
    if cap <= 0.0:
        return
    i = tri_indices[t, 0]
    dinv = tri_poses[t]
    e1 = x[tri_indices[t, 1]] - x[i]
    e2 = x[tri_indices[t, 2]] - x[i]
    s1, s2, v1 = principal_frame(e1 * dinv[0, 0] + e2 * dinv[1, 0], e1 * dinv[0, 1] + e2 * dinv[1, 1])
    out[t] = (s1 - 1.0) / cap
