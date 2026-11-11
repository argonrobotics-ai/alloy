# SPDX-License-Identifier: Apache-2.0
"""Pass-through detection: a particle whose centre is inside a rigid or proxy shape (SCR-SC4-4 ruling (1a)).

A soft contact (Newton's ``create_soft_contacts``) stores the closest surface point in the shape's body frame
(``soft_contact_body_pos`` = X_bs (x_local - n d)) and the world normal, so the particle centre's signed distance to
the shape is exactly d = (x - X_wb body_pos) . n. d < 0 means the centre is inside the shape (depth > radius): the
contact can then push the particle out of the far face, which threads cloth through a pad or a sheet.

``PassThroughFlags`` keeps a per-world int32 flag on the device, set by ``check`` after each collide of the proxy (or
fallback) pipeline, sticky until ``reset``. As a HealthMonitor capacity flag ``(flags, PASS_THROUGH_BITS)`` it turns
that world INVALID (PAIR_PEN_FAIL, the pair-penetration family; fail closed). The check sees the particles at
collide time, so a pass-through during an exchange is flagged at the next one. Range: a primitive (box, sphere,
capsule, ...) is checked at any depth; a mesh only while the centre is within margin + radius of its surface,
where Newton's mesh query still returns a closest point (deeper inside a mesh no contact is generated at all), except
under full-surface contact, whose mesh pass adds a recovery record for a deep vertex (any depth). Full-surface mesh
records with an unresolved inside/outside sign are skipped (Newton applies no force for them either).
"""

from __future__ import annotations

from typing import Any

import warp as wp

from alloy.coupling.kinematic import clear_masked_kernel
from alloy.precision.types import HealthFlags

PASS_THROUGH_BITS = int(HealthFlags.PAIR_PEN_FAIL)
TOL_FRAC = 0.05          # centre inside by more than 5 % of the particle radius (grazing round-off is not an event)
MESH_SIGN_UNKNOWN = wp.constant(32)   # newton geometry/soft_contacts_mesh.py _MESH_FEATURE_SIGN_UNKNOWN


@wp.kernel(enable_backward=False)
def centre_inside_kernel(count: wp.array[wp.int32], particle: wp.array[wp.int32], indices: wp.array[wp.vec3i],
                         shape: wp.array[wp.int32], body_pos: wp.array[wp.vec3], normal: wp.array[wp.vec3],
                         particle_q: wp.array[wp.vec3], radius: wp.array[float], particle_world: wp.array[wp.int32],
                         shape_body: wp.array[wp.int32], body_q: wp.array[wp.transform], tol_frac: float,
                         mesh_features: wp.array[wp.vec3i], has_features: int, out: wp.array[wp.int32]):
    """Flag the world of each particle contact whose particle centre is inside its shape."""
    i = wp.tid()
    if i >= count[0] or indices[i][1] != -1:        # past the count, or a full-surface edge / face contact
        return
    if has_features != 0:                           # full-surface mesh record with an unresolved inside/outside sign
        f = mesh_features[i][0]
        if f >= 0 and (f & MESH_SIGN_UNKNOWN) != 0:
            return
    p = particle[i]
    b = shape_body[shape[i]]
    xs = body_pos[i]
    if b >= 0:
        xs = wp.transform_point(body_q[b], xs)
    d = wp.dot(particle_q[p] - xs, normal[i])
    if d < -tol_frac * radius[p]:
        w = particle_world[p]
        if w >= 0:
            out[w] = 1
        else:
            for k in range(out.shape[0]):            # a global particle: every world
                out[k] = 1


class PassThroughFlags:
    """Per-world pass-through flags of one contact pipeline (module docstring)."""

    def __init__(self, world_count: int, device: Any, tol_frac: float = TOL_FRAC) -> None:
        self.flags = wp.zeros(max(int(world_count or 0), 1), dtype=wp.int32, device=device)
        self.tol_frac = float(tol_frac)
        self._no_bodies = wp.zeros(1, dtype=wp.transform, device=device)   # body_q of a body-less state
        self._no_features = wp.full(1, wp.vec3i(-1, -1, -1), dtype=wp.vec3i, device=device)

    def check(self, view: Any, state: Any, contacts: Any) -> None:
        """Device-only, capture-safe: launch over the soft contact buffer just filled by ``collide``."""
        n = int(contacts.soft_contact_max or 0)
        if n == 0 or not int(view.particle_count or 0):
            return
        features = getattr(contacts, "_soft_contact_mesh_features", None)    # full-surface mesh records (Newton)
        wp.launch(centre_inside_kernel, dim=n, device=self.flags.device,
                  inputs=[contacts.soft_contact_count, contacts.soft_contact_particle, contacts.soft_contact_indices,
                          contacts.soft_contact_shape, contacts.soft_contact_body_pos, contacts.soft_contact_normal,
                          state.particle_q, view.particle_radius, view.particle_world, view.shape_body,
                          self._no_bodies if state.body_q is None else state.body_q, self.tol_frac,
                          self._no_features if features is None else features, int(features is not None),
                          self.flags])

    def health_source(self) -> tuple[wp.array, int]:
        return self.flags, PASS_THROUGH_BITS

    def reset(self, world_mask: Any = None) -> None:
        if world_mask is None:
            self.flags.zero_()
        else:
            wp.launch(clear_masked_kernel, dim=self.flags.shape[0], device=self.flags.device,
                      inputs=[world_mask, self.flags])
