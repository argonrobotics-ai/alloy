# SPDX-License-Identifier: Apache-2.0
"""Static-shape Divide-and-Truncate for VBD particles (SCR-SC8-3).

Newton's SolverVBD with self-contact truncates every particle displacement at the division planes to its self-contact
neighbours (Divide-and-Truncate, DAT), but nothing truncates a particle's motion into a static shape unless rigid-soft
DAT is on (`rigid_soft_enable_dat`, off by default, and it needs a solver-owned collision pipeline). Each step the
inertia guess carries a resting pile down coherently (no relative approach, so no self-contact truncation), and the
static shape's push back up on a squeezed bottom-layer vertex is truncated at the plane to the layer above. The pile
ratchets into the shape: a scrunched film pile sank 4-7 mm through a static ground plane at a 5 ms step, with stock
SolverVBD too (docs/notes/wp11-results.md, SCR-SC8-3).

This module truncates the particle side of every soft contact with a static shape (shape_body == -1) against a
division plane, the same way Newton's rigid-soft DAT treats a static analytic contact row: the plane is normal to the
stored contact normal, placed by Newton's `place_dat_division_plane` between the stored surface point and the
particle's reference position (the static side never approaches, so it sits at 5 % of the gap from the surface), and
`planar_truncation_t` keeps each particle of the record on its side (a particle that starts below may move out, never
deeper). The scalar t of each particle is the minimum over its records; the displacement is scaled by it before
Newton's own truncation, which then applies the self-contact planes and the isotropic bound on top (a product of
truncations keeps every constraint: each is a scaling of the same straight trajectory toward the reference).

Fixed launch sizes (soft_contact_max, particle_count), a float atomic_min (order independent): capture-safe and
deterministic. Dynamic bodies are not covered (their planes move with the body; Newton's rigid-soft DAT does that).
"""

from __future__ import annotations

import warp as wp
from alloy._vendor.newton._src.solvers.vbd.rigid_vbd_kernels import (
    dat_separation_epsilon,
    place_dat_division_plane,
    planar_truncation_t,
)


@wp.kernel
def static_contact_truncation(
    soft_contact_count: wp.array[wp.int32],
    soft_contact_indices: wp.array[wp.vec3i],
    soft_contact_barycentric: wp.array[wp.vec3],
    soft_contact_shape: wp.array[wp.int32],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    pos_ref: wp.array[wp.vec3],
    displacement: wp.array[wp.vec3],
    gamma: float,
    truncation_t: wp.array[float],
):
    """One soft contact record: truncate its particles against the static shape's division plane (module docstring)."""
    i = wp.tid()
    if i >= soft_contact_count[0]:
        return
    idx = soft_contact_indices[i]
    if idx[0] < 0 or shape_body[soft_contact_shape[i]] >= 0:
        return
    bary = soft_contact_barycentric[i]
    x_ref = wp.vec3(0.0)
    scale = float(1.0)
    approach = float(0.0)
    n = soft_contact_normal[i]
    for k in range(3):
        v = idx[k]
        if v >= 0:
            x_ref += bary[k] * pos_ref[v]
            scale = wp.max(scale, wp.max(wp.abs(pos_ref[v])))
            approach = wp.max(approach, -wp.dot(n, displacement[v]))
    surface = soft_contact_body_pos[i]               # world frame for a static shape
    scale = wp.max(scale, wp.max(wp.abs(surface)))
    eps = dat_separation_epsilon(scale)
    gap = wp.max(wp.dot(n, x_ref - surface), 0.0)    # an existing penetration gives a zero gap, not a deeper target
    plane, _lmbd = place_dat_division_plane(n, surface, gap, approach, 0.0, eps)
    for k in range(3):
        v = idx[k]
        if v >= 0:
            t = planar_truncation_t(pos_ref[v], displacement[v], n, plane, gamma, eps)
            if t < 1.0:
                wp.atomic_min(truncation_t, v, t)


@wp.kernel
def scale_displacements(truncation_t: wp.array[float], displacement: wp.array[wp.vec3]):
    i = wp.tid()
    t = truncation_t[i]
    if t < 1.0:
        displacement[i] = displacement[i] * t


def truncate_against_static_shapes(solver, contacts, truncation_t: wp.array) -> None:
    """Scale ``solver.particle_displacements`` in place so no particle crosses a static shape's division plane."""
    truncation_t.fill_(1.0)
    wp.launch(static_contact_truncation, dim=contacts.soft_contact_max,
              inputs=[contacts.soft_contact_count, contacts.soft_contact_indices, contacts.soft_contact_barycentric,
                      contacts.soft_contact_shape, contacts.soft_contact_body_pos, contacts.soft_contact_normal,
                      solver.model.shape_body, solver.pos_prev_collision_detection, solver.particle_displacements,
                      solver.dat_conservative_bound_relaxation],
              outputs=[truncation_t], device=solver.device)
    wp.launch(scale_displacements, dim=solver.model.particle_count, inputs=[truncation_t],
              outputs=[solver.particle_displacements], device=solver.device)


__all__ = ["scale_displacements", "static_contact_truncation", "truncate_against_static_shapes"]
