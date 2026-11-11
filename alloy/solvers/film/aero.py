# SPDX-License-Identifier: Apache-2.0
"""Air force on film triangles: quadratic pressure drag along the normal plus skin friction along the face.

Per triangle (area A, unit normal n, relative air velocity v = mean vertex velocity - wind, v_n = (v·n)n,
v_t = v - v_n): F = -½·ρ·A·s·(Cd·|v·n|·v_n + Ct·|v_t|·v_t), s = ``film:air``. F/3 goes to each vertex, gathered in
CSR order (deterministic). Power Σ F·v_mean ≤ 0 per triangle, so the force only removes energy at the current
velocity. With ``drag_cap`` the per-vertex impulse |F|·dt/m is clamped to the vertex's relative speed (explicit
drag can never reverse a light vertex in one step). The force enters SolverVBD through ``state.particle_f`` for one
step and the caller's forces are restored after it (solver.py).
"""

from __future__ import annotations

import warp as wp


@wp.struct
class AirParams:
    rho: float
    cd: float
    ct: float
    wind: wp.vec3
    dt: float
    cap: int


@wp.kernel(enable_backward=False)
def tri_air_force(
    p: AirParams,
    x: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    air: wp.array[float],
    f_tri: wp.array[wp.vec3],
):
    t = wp.tid()
    f_tri[t] = wp.vec3(0.0)
    s = air[t]
    if s <= 0.0:
        return
    i = tri_indices[t, 0]
    j = tri_indices[t, 1]
    k = tri_indices[t, 2]
    n = wp.cross(x[j] - x[i], x[k] - x[i])
    a2 = wp.length(n)
    if a2 < 1.0e-14:
        return
    nh = n / a2
    v = (qd[i] + qd[j] + qd[k]) / 3.0 - p.wind
    vn = wp.dot(v, nh)
    vt = v - nh * vn
    f_tri[t] = -0.25 * p.rho * a2 * s * (nh * (p.cd * wp.abs(vn) * vn) + vt * (p.ct * wp.length(vt)))


@wp.kernel(enable_backward=False)
def vertex_air_force(
    p: AirParams,
    vtx_start: wp.array[wp.int32],
    vtx_tri: wp.array[wp.int32],
    vtx_particle: wp.array[wp.int32],
    f_tri: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    qd: wp.array[wp.vec3],
    particle_f: wp.array[wp.vec3],
    f_air: wp.array[wp.vec3],
):
    """f_air[p] = Σ f_tri/3 over p's triangles (impulse-capped); particle_f[p] += f_air[p]."""
    v = wp.tid()
    q = vtx_particle[v]
    f = wp.vec3(0.0)
    for e in range(vtx_start[v], vtx_start[v + 1]):
        f = f + f_tri[vtx_tri[e]] / 3.0
    w = inv_mass[q]
    if w == 0.0:
        f = wp.vec3(0.0)
    elif p.cap != 0:
        dv = wp.length(f) * p.dt * w
        vrel = wp.length(qd[q] - p.wind)
        if dv > vrel:
            f = f * (vrel / dv)
    f_air[q] = f
    particle_f[q] = particle_f[q] + f
