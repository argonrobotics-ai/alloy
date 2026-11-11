# SPDX-License-Identifier: Apache-2.0
"""The film's stretch limit as a membrane stress inside VBD's solve (``FilmConfig.limiter="stress"``;
docs/known_gaps/film-projection-force.md).

The projection (strain.py) limits stretch by moving vertices after the VBD solve, each constraint by inverse mass.
A film vertex weighs under a milligram, so it gives way and a load on it (grams, or a body through contact) hardly
moves: a strip carrying 2-500 g stood 11.5-34x past its cap and rang at 0.1-0.5 m/s. The tension the projection
enforces is never a force, so no contact, bond or body feels it.

Here a limited triangle past its cap is an elastic membrane of the film's own stiffness scale, E h / (1 - ν²). With
F the triangle's 3x2 deformation gradient and G = (FᵀF - (1 + cap)² I) / 2 its Green strain past the cap, the energy
is ψ = k |P+(G)|² / 2, k = E h A / (1 - ν²) [J], P+ the projection on positive semi-definite 2x2 matrices (tension
only: below the cap, and across a wrinkle, the film carries none). With S = k P+(G) the stress [J]:

    f_i = -F S b_i,       H_i = (b_i·S b_i) I + k Σ_ab θ_ab g_ab g_abᵀ,
    g_ab = ((F w_a)(b_i·w_b) + (F w_b)(b_i·w_a)) / 2

w_a the eigenvectors of G and l_a its eigenvalues, θ_aa = 1 where l_a > 0, θ_12 = θ_21 = (l1+ - l2+) / (l1 - l2)
(the derivative of P+), b_i the rest-frame gradient of vertex i (rows of ``tri_poses``). The force is the exact
gradient, so it conserves linear and angular momentum; the Hessian is positive semi-definite: its first term is the
tension's own stiffness (a taut string's, in every direction), the rest the Gauss-Newton stiffness of the stretch.
Both join VBD's vertex solve in its after-accumulate colour slot, beside inertia, contact and friction. So the
tension is a force in the same solve as the contacts (what a body feels of it through contact is not measured yet).
Below the cap nothing changes (VBD's own soft membrane, bending and creases act as before).

The model's limits:
- **No Poisson coupling.** ψ depends on |P+(G)|² alone. In a one-way pull with the cross direction slack the real
  film's modulus is E; this energy gives E (1 + cap)² / (1 - ν²): the strip's stretch past the cap is 0.79-0.82 of
  m g / (E h b) at ν 0.4 (27 % stiff; 0.78 predicted). Pulled both ways equally it is (1 + ν) soft. The stretch
  past the cap is under 1 % of the length in every law, so the error is of that.
- **Soft below the cap.** A real film is elastic from zero strain. This one has VBD's soft membrane (1e-4 of the
  stiffness) until the cap: a load let go at rest length falls the cap's slack first (cap x length) and is caught
  there, and with the cross direction slack the limit's shear stiffness is θ_12 = l1 / (l1 - l2), a few percent of
  k, so loaded vertices move against each other across the pull at mm/s.
- **float32.** Strain is read from positions, so the stress per triangle is quantised at about
  E h eps32 |x| / cell [N/m] and a particle's ``force`` at about E h eps32 |x| (HDPE 15 um: 1.8 mN at 1 m from the
  origin, more than a 2 g strip's share per pin). Sums are right: the pins of the 2 g strip read 0.742 of its
  weight at the origin, 0.738 at 1 m and 0.736 at 4 m.
- **``force`` leaves out the soft membrane.** VBD's soft membrane carries a share of a light load below the cap, so
  at loads near the film's own weight ``force`` under-reads the pins' reaction by up to a quarter (a 2 g strip: 0.74
  of the weight); from 50 g it reads 0.99 or more.
- **No yield.** Past the cap the film stays elastic at any load: an overload keeps stretching and nothing flags it
  (5 kg on the 40 mm HDPE strip, 2.2x its 22 N yield force: 3.8x past the cap after 2 s, still at 36 mm/s; health's
  CLOTH_STRETCH warns only past 50 %).

``colour_entry`` (in each colour's truncation): VBD's Hessian has no stiffness for a slack triangle, so a step that
takes one past its cap overshoots, and the film rings (a 2 g load at 10 iterations: 9x past the cap). Such a step is
cut where the triangle reaches ``entry`` of its cap past it; the next iteration solves it with the limit in its
Hessian. The same reason and cure as the contact skin's entry limit (skin.py). The cut never lets the triangle past
``entry``: along a step G(t) = G0 + t A + t² B with B positive semi-definite, so G's top eigenvalue is convex in t and
lies under its chord, which is what the cut follows. Same-colour vertices share no triangle, so the other two corners
stand still in the test. A taut triangle (already past the cap) is not cut: its stiffness is in that step's Hessian.

Past the cap the film stretches elastically (a 40 mm HDPE 15 um strip under 500 g: 0.64 % of its length). ``stress``
holds S per triangle after a step and ``force`` the limit's pull on each particle [N], pinned ones included (the pins
of a hung strip sum to minus the weight the limit carries; VBD's soft membrane carries its own small share).
No multipliers and no state between steps. One thread per particle of the colour or per triangle: no atomics, fixed
launch sizes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp


@wp.func
def tension_part(m: wp.vec3):
    """P+ of the symmetric 2x2 (m00, m01, m11): (the projected matrix, eigenvalue 1 >= eigenvalue 2, unit eigenvector
    1); eigenvector 2 is its perpendicular."""
    mean = 0.5 * (m[0] + m[2])
    dev = wp.sqrt(0.25 * (m[0] - m[2]) * (m[0] - m[2]) + m[1] * m[1])
    l1 = mean + dev
    l2 = mean - dev
    w = wp.vec2(1.0, 0.0)
    if wp.abs(m[1]) > 1.0e-12 * (wp.abs(m[0]) + wp.abs(m[2])):
        w = wp.normalize(wp.vec2(m[1], l1 - m[0]))
    elif m[2] > m[0]:
        w = wp.vec2(0.0, 1.0)
    p1 = wp.max(l1, 0.0)
    p2 = wp.max(l2, 0.0)
    s = wp.vec3(p1 * w[0] * w[0] + p2 * w[1] * w[1], (p1 - p2) * w[0] * w[1], p1 * w[1] * w[1] + p2 * w[0] * w[0])
    return s, l1, l2, w


@wp.func
def strain_past_cap(xi: wp.vec3, xj: wp.vec3, xk: wp.vec3, dinv: wp.mat22, cap: float):
    """(f0, f1, G) of a triangle at (xi, xj, xk): G = (FᵀF - (1 + cap)² I) / 2 as (G00, G01, G11)."""
    e1 = xj - xi
    e2 = xk - xi
    f0 = e1 * dinv[0, 0] + e2 * dinv[1, 0]
    f1 = e1 * dinv[0, 1] + e2 * dinv[1, 1]
    hi = 1.0 + cap
    return f0, f1, wp.vec3(0.5 * (wp.dot(f0, f0) - hi * hi), 0.5 * wp.dot(f0, f1), 0.5 * (wp.dot(f1, f1) - hi * hi))


@wp.func
def top_strain(xi: wp.vec3, xj: wp.vec3, xk: wp.vec3, dinv: wp.mat22, cap: float) -> float:
    """The largest eigenvalue of G: > 0 past the cap."""
    f0, f1, g = strain_past_cap(xi, xj, xk, dinv, cap)
    return 0.5 * (g[0] + g[2]) + wp.sqrt(0.25 * (g[0] - g[2]) * (g[0] - g[2]) + g[1] * g[1])


@wp.func
def rest_gradient(dinv: wp.mat22, slot: int) -> wp.vec2:
    """b of the triangle's vertex ``slot``: F = Σ x_i ⊗ b_i."""
    bj = wp.vec2(dinv[0, 0], dinv[0, 1])
    bk = wp.vec2(dinv[1, 0], dinv[1, 1])
    if slot == 1:
        return bj
    if slot == 2:
        return bk
    return -(bj + bk)


@wp.kernel(enable_backward=False)
def colour_stress(group: wp.array[wp.int32], vertex_of: wp.array[wp.int32], vtx_start: wp.array[wp.int32],
                  vtx_tri: wp.array[wp.int32], vtx_slot: wp.array[wp.int32], x: wp.array[wp.vec3],
                  tri_indices: wp.array2d[wp.int32], tri_poses: wp.array[wp.mat22], strain_cap: wp.array[float],
                  stiffness: wp.array[float], forces: wp.array[wp.vec3], hessians: wp.array[wp.mat33]):
    """VBD after-accumulate slot, one thread per particle of the colour: the limit's force and Hessian (module doc)."""
    p = group[wp.tid()]
    v = vertex_of[p]
    if v < 0:
        return
    f = wp.vec3(0.0)
    h = wp.mat33(0.0)
    for e in range(vtx_start[v], vtx_start[v + 1]):
        t = vtx_tri[e]
        k = stiffness[t]
        if k <= 0.0:
            continue
        f0, f1, g = strain_past_cap(x[tri_indices[t, 0]], x[tri_indices[t, 1]], x[tri_indices[t, 2]], tri_poses[t],
                                    strain_cap[t])
        s, l1, l2, w = tension_part(g * k)
        if l1 <= 0.0:
            continue
        b = rest_gradient(tri_poses[t], vtx_slot[e])
        sb = wp.vec2(s[0] * b[0] + s[1] * b[1], s[1] * b[0] + s[2] * b[1])
        f = f - (f0 * sb[0] + f1 * sb[1])
        h = h + wp.identity(3, float) * wp.dot(b, sb)
        bw = wp.dot(b, w)
        bw2 = -b[0] * w[1] + b[1] * w[0]
        fw = f0 * w[0] + f1 * w[1]
        fw2 = f1 * w[0] - f0 * w[1]
        h = h + wp.outer(fw, fw) * (k * bw * bw)
        shear = float(1.0)                          # dP+ between the two eigen-directions: (l1+ - l2+) / (l1 - l2)
        if l2 > 0.0:
            h = h + wp.outer(fw2, fw2) * (k * bw2 * bw2)
        else:
            shear = l1 / (l1 - l2)
        q = fw * bw2 + fw2 * bw
        h = h + wp.outer(q, q) * (0.5 * k * shear)
    forces[p] = forces[p] + f
    hessians[p] = hessians[p] + h


@wp.kernel(enable_backward=False)
def colour_entry(group: wp.array[wp.int32], vertex_of: wp.array[wp.int32], vtx_start: wp.array[wp.int32],
                 vtx_tri: wp.array[wp.int32], vtx_slot: wp.array[wp.int32], x: wp.array[wp.vec3],
                 ref: wp.array[wp.vec3], tri_indices: wp.array2d[wp.int32], tri_poses: wp.array[wp.mat22],
                 strain_cap: wp.array[float], stiffness: wp.array[float], entry: float, disp: wp.array[wp.vec3]):
    """In a colour's truncation, one thread per particle of the colour: a step that takes a slack triangle past its
    cap is cut where the triangle reaches ``entry`` of its cap past it (module doc). ``x`` holds the iterate before
    the colour's step, ``ref + disp`` the step's end."""
    p = group[wp.tid()]
    v = vertex_of[p]
    if v < 0:
        return
    xo = x[p]
    xn = ref[p] + disp[p]
    if wp.length_sq(xn - xo) == 0.0:
        return
    cut = float(1.0)
    for e in range(vtx_start[v], vtx_start[v + 1]):
        t = vtx_tri[e]
        if stiffness[t] <= 0.0:
            continue
        xa = x[tri_indices[t, 0]]
        xb = x[tri_indices[t, 1]]
        xc = x[tri_indices[t, 2]]
        g0 = top_strain(xa, xb, xc, tri_poses[t], strain_cap[t])
        if g0 > 0.0:
            continue                                         # already taut: the limit was in this step's Hessian
        slot = vtx_slot[e]
        if slot == 0:
            xa = xn
        elif slot == 1:
            xb = xn
        else:
            xc = xn
        g1 = top_strain(xa, xb, xc, tri_poses[t], strain_cap[t])
        lim = entry * strain_cap[t]
        if g1 > lim:
            cut = wp.min(cut, (lim - g0) / (g1 - g0))
    if cut < 1.0:
        disp[p] = xo + (xn - xo) * cut - ref[p]


@wp.kernel(enable_backward=False)
def triangle_stress(x: wp.array[wp.vec3], tri_indices: wp.array2d[wp.int32], tri_poses: wp.array[wp.mat22],
                    strain_cap: wp.array[float], stiffness: wp.array[float], stress: wp.array[wp.vec3]):
    """stress[t] = S = k P+(G) at positions x [J] (one thread per triangle)."""
    t = wp.tid()
    stress[t] = wp.vec3(0.0)
    k = stiffness[t]
    if k <= 0.0:
        return
    f0, f1, g = strain_past_cap(x[tri_indices[t, 0]], x[tri_indices[t, 1]], x[tri_indices[t, 2]], tri_poses[t],
                                strain_cap[t])
    s, l1, l2, w = tension_part(g * k)
    stress[t] = s


@wp.kernel(enable_backward=False)
def stress_force(vtx_particle: wp.array[wp.int32], vtx_start: wp.array[wp.int32], vtx_tri: wp.array[wp.int32],
                 vtx_slot: wp.array[wp.int32], x: wp.array[wp.vec3], tri_indices: wp.array2d[wp.int32],
                 tri_poses: wp.array[wp.mat22], stress: wp.array[wp.vec3], force: wp.array[wp.vec3]):
    """force[p] = -Σ F S b over the film particle's triangles [N] (one thread per film particle)."""
    v = wp.tid()
    f = wp.vec3(0.0)
    for e in range(vtx_start[v], vtx_start[v + 1]):
        t = vtx_tri[e]
        s = stress[t]
        if s[0] == 0.0 and s[1] == 0.0 and s[2] == 0.0:
            continue
        i = tri_indices[t, 0]
        dinv = tri_poses[t]
        e1 = x[tri_indices[t, 1]] - x[i]
        e2 = x[tri_indices[t, 2]] - x[i]
        b = rest_gradient(dinv, vtx_slot[e])
        sb = wp.vec2(s[0] * b[0] + s[1] * b[1], s[1] * b[0] + s[2] * b[1])
        f = f - ((e1 * dinv[0, 0] + e2 * dinv[1, 0]) * sb[0] + (e1 * dinv[0, 1] + e2 * dinv[1, 1]) * sb[1])
    force[vtx_particle[v]] = f


@wp.kernel(enable_backward=False)
def tension(tri_areas: wp.array[float], stress: wp.array[wp.vec3], out: wp.array[float]):
    """out[t] = the triangle's largest membrane tension [N/m]: the top eigenvalue of S over its rest area."""
    t = wp.tid()
    s, l1, l2, w = tension_part(stress[t])
    out[t] = wp.max(l1, 0.0) / wp.max(tri_areas[t], 1.0e-30)


class StressLimiter:
    """One SolverFilm's stress limit: the per-triangle stiffness, the launches, and the readouts (module doc)."""

    def __init__(self, solver: Any, topology: Any, device_topology: Any, stiffness: np.ndarray, entry: float) -> None:
        m, dev = solver.model, solver.model.device
        vertex_of = np.full(int(m.particle_count), -1, np.int32)
        vertex_of[np.asarray(topology.particles, np.int64)] = np.arange(len(topology.particles), dtype=np.int32)
        self.solver, self.d, self.entry = solver, device_topology, float(entry)
        self.vertex_of = wp.array(vertex_of, dtype=wp.int32, device=dev)
        self.stiffness = wp.array(np.ascontiguousarray(stiffness, np.float32), dtype=float, device=dev)  # k [J]
        self.stress = wp.zeros(int(m.tri_count), dtype=wp.vec3, device=dev)        # S (00, 01, 11) [J], last step
        self.force = wp.zeros(int(m.particle_count), dtype=wp.vec3, device=dev)    # N per particle, last step
        solver.add_colour_callback(self.colour_slot, external=False, when="after_accumulate")

    def _tri(self) -> list:
        m = self.solver.model
        return [m.tri_indices, m.tri_poses, m.film.strain_cap, self.stiffness]

    def colour_slot(self, solver: Any, colour: int, state: Any, contacts: Any) -> None:
        """after_accumulate callback: ``state`` holds the iterate."""
        m, d = solver.model, self.d
        group = m.particle_color_groups[colour]
        wp.launch(colour_stress, dim=group.shape[0],
                  inputs=[group, self.vertex_of, d.vtx_start, d.vtx_tri, d.vtx_slot, state.particle_q, *self._tri()],
                  outputs=[solver.particle_forces, solver.particle_hessians], device=m.device)

    def cut_entry(self, solver: Any, state: Any, group: Any) -> None:
        """A colour's truncation; ``group`` None is the truncation after the initial guess, where every vertex has
        moved and the first iteration starts with the taut triangles in its Hessian: no cut."""
        if group is None or self.entry <= 0.0:
            return
        m, d = solver.model, self.d
        wp.launch(colour_entry, dim=group.shape[0],
                  inputs=[group, self.vertex_of, d.vtx_start, d.vtx_tri, d.vtx_slot, state.particle_q,
                          solver.pos_prev_collision_detection, *self._tri(), self.entry],
                  outputs=[solver.particle_displacements], device=m.device)

    def after_step(self, x: wp.array) -> None:
        """The readouts at the step's end positions."""
        m, d = self.solver.model, self.d
        wp.launch(triangle_stress, dim=m.tri_count, inputs=[x, *self._tri()], outputs=[self.stress], device=m.device)
        wp.launch(stress_force, dim=d.particles.shape[0],
                  inputs=[d.particles, d.vtx_start, d.vtx_tri, d.vtx_slot, x, m.tri_indices, m.tri_poses, self.stress],
                  outputs=[self.force], device=m.device)


__all__ = ["StressLimiter", "colour_entry", "colour_stress", "stress_force", "tension", "tension_part",
           "triangle_stress"]
