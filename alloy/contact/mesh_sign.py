# SPDX-License-Identifier: Apache-2.0
"""Re-check of inside soft-contact rows on parity-signed meshes (WP2's trace, coordinator 2026-11-07).

Newton's ``create_soft_contacts`` (geometry/kernels.py, MESH branch) writes ``n = normalize(x - p) * sign`` with the
sign from ``mesh_query_point_sign``, which is ray-crossing parity for a watertight mesh (``resolve_mesh_sign_method``).
Warp 1.18.0's ``mesh_query_point_sign_parity`` says INSIDE at degenerate points where its ray grazes a shared edge or
vertex: in ``alloy.api.meshes.open_cylinder(0.03, 0.06, 0.004)`` a particle in the cavity 4.3 mm from the inner wall
got a row with n.(x - p) = -4.26 mm (the pseudo-normal and the winding number both say outside; one ulp away parity
agrees; 0 errors in 200,000 random cavity points). Health read it as a 6.28 mm penetration (WP5 now skips SPH / MPM
rows), and a VBD-family particle there is pushed into the wall for that substep.

``ParityRecheck.apply`` runs after each collide on the rows Newton's per-particle pass wrote (not the full-surface mesh
block, whose records carry their own feature flags). Every row on a watertight mesh shape is signed again by a
perturbed-parity vote: three parity queries at x + delta d_k (delta = 1e-3 |x - p|, three fixed generic directions),
the sign of their sum; where it disagrees with the row (n.(x - p) < 0 is inside) the normal flips, in either direction
(the grazing sheet also crosses the wall's solid, where parity can say outside for an inside particle). The body point
is unchanged (``x_local - n * d`` keeps n * d), so the row becomes what Newton writes for that side at that distance.
Rows closer than ``MIN_GAP`` to the surface are left alone (the sign is ill-defined there), and so is a row no vote
query answered. Below ~10 um the vote's step falls under a float32 ulp at 3 cm coordinates and cannot leave a grazing
sheet; those rows touch the surface, where either sign gives about the same force. ``flips`` counts the flipped rows
on the device.

Where it runs: after the bulk collide of ``AlloyContactLayer`` (plain scenes; soft rows come from the bulk pass only)
and after each collide of ``ProxyContactPipeline`` (the stepper's proxy pipelines and the fallback buffer). Task rigs
that build their own ``newton.CollisionPipeline`` (sew_seam, sew_ply_pinch's outer buffer, laundry_crossings,
job_slice_sim) do not get it; of those only sew_ply_pinch has mesh shapes (its pads), and its cloth reads the proxy
pipeline, which does.

Why a vote (WP2's review): the pseudo-normal is no arbiter on these meshes. Of 200,000 points inside the cup's 4 mm wall
parity says inside for all, Warp's pseudo-normal sign says outside for 907 (0.45 %, at the wall's mid-plane where the
inner and outer surfaces tie). The vote stays on parity, which is right for a watertight mesh, and steps off the
measure-zero sheet: 0 errors in 200,000 cavity and 200,000 wall points. The collision meshes carry no winding-number
support. Cost: three ray casts per row on a parity-signed mesh.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy._vendor.newton._src.geometry.flags import MeshProperties, MeshSignMethod
from alloy._vendor.newton._src.geometry.kernels import resolve_mesh_sign_method
from alloy._vendor.newton._src.geometry.soft_contacts_sdf import _shape_frames

MESH = wp.constant(int(newton.GeoType.MESH))
CONVEX_MESH = wp.constant(int(newton.GeoType.CONVEX_MESH))
PARITY = wp.constant(int(MeshSignMethod.PARITY))


# fixed generic (non-axis) unit directions for the perturbed-parity vote (WP2's review)
VOTE_D0 = wp.constant(wp.vec3(0.5773, 0.6241, 0.5263))
VOTE_D1 = wp.constant(wp.vec3(-0.7071, 0.1961, 0.6794))
VOTE_D2 = wp.constant(wp.vec3(0.2357, -0.9129, 0.3333))
VOTE_STEP = 1.0e-3          # the vote's offset, a fraction of the particle's distance to the surface
MIN_GAP = 1.0e-6            # m: closer than this the sign is ill-defined; the row is left as Newton wrote it


@wp.func
def parity_vote(mesh: wp.uint64, x: wp.vec3, step: float, reach: float) -> float:
    """Sign of the sum of three parity queries at x + step * d_k: off the measure-zero sheet where a ray grazes an
    edge or a vertex (WP2: 0 errors in 200,000 cavity and 200,000 wall points of the cup); 0 when none answered."""
    a = wp.mesh_query_point_sign_parity(mesh, x + step * VOTE_D0, reach)
    b = wp.mesh_query_point_sign_parity(mesh, x + step * VOTE_D1, reach)
    c = wp.mesh_query_point_sign_parity(mesh, x + step * VOTE_D2, reach)
    s = wp.where(a.result, a.sign, 0.0) + wp.where(b.result, b.sign, 0.0) + wp.where(c.result, c.sign, 0.0)
    if s == 0.0:
        return 0.0                                   # no query answered (or a tie): leave the row alone
    return wp.where(s < 0.0, -1.0, 1.0)


@wp.kernel(enable_backward=False)
def _recheck_parity_rows(
    count: wp.array[int], capacity: int, particle: wp.array[int], shape: wp.array[int], normal: wp.array[wp.vec3],
    body_pos: wp.array[wp.vec3], features: wp.array[wp.vec3i], has_features: int, particle_q: wp.array[wp.vec3],
    body_q: wp.array[wp.transform], shape_body: wp.array[wp.int32], shape_transform: wp.array[wp.transform],
    shape_type: wp.array[wp.int32], shape_scale: wp.array[wp.vec3], shape_source_ptr: wp.array[wp.uint64],
    shape_mesh_properties: wp.array[wp.int32], flips: wp.array[wp.int32],
):
    tid = wp.tid()
    if tid >= wp.min(count[0], capacity):
        return
    s = shape[tid]
    if s < 0:
        return
    kind = shape_type[s]
    if kind != MESH and kind != CONVEX_MESH:
        return
    if resolve_mesh_sign_method(shape_mesh_properties[s]) != PARITY:
        return
    if has_features != 0:
        if features[tid][0] >= 0:                    # a full-surface mesh record, signed by its own features
            return
    x = particle_q[particle[tid]]
    b = shape_body[s]
    p = body_pos[tid]
    if b >= 0:
        p = wp.transform_point(body_q[b], p)
    n = normal[tid]
    gap = x - p
    dist = wp.length(gap)
    if dist < MIN_GAP:
        return
    row_sign = wp.where(wp.dot(n, gap) < 0.0, -1.0, 1.0)
    _X_bs, _X_ws, X_sw = _shape_frames(shape_body, body_q, shape_transform, s)
    scale = shape_scale[s]
    min_scale = wp.min(wp.min(wp.abs(scale[0]), wp.abs(scale[1])), wp.abs(scale[2]))
    x_mesh = wp.cw_div(wp.transform_point(X_sw, x), scale)
    step = VOTE_STEP * dist / min_scale
    vote = parity_vote(shape_source_ptr[s], x_mesh, step, 1.01 * dist / min_scale + 2.0 * step + 1.0e-6)
    if vote != 0.0 and vote != row_sign:
        normal[tid] = -n
        wp.atomic_add(flips, 0, 1)


def has_parity_meshes(model: Any) -> bool:
    """Whether any shape is a mesh Newton signs by parity (watertight)."""
    if not int(model.shape_count or 0):
        return False
    kinds = model.shape_type.numpy()
    props = model._shape_mesh_properties.numpy()
    mesh = (kinds == int(newton.GeoType.MESH)) | (kinds == int(newton.GeoType.CONVEX_MESH))
    return bool(np.any(mesh & ((props & int(MeshProperties.WATERTIGHT)) != 0)))


class ParityRecheck:
    """The re-check for one model or view (module docstring); a no-op without parity-signed meshes."""

    def __init__(self, model: Any) -> None:
        self.enabled = has_parity_meshes(model)
        self.flips = wp.zeros(1, dtype=wp.int32, device=model.device)
        self._no_features = wp.zeros(0, dtype=wp.vec3i, device=model.device)

    def apply(self, model: Any, state: Any, contacts: Any) -> None:
        if not self.enabled or not int(contacts.soft_contact_max):
            return
        features = getattr(contacts, "_soft_contact_mesh_features", None)
        has = features is not None and features.shape[0] >= contacts.soft_contact_max
        wp.launch(_recheck_parity_rows, dim=int(contacts.soft_contact_max), device=model.device, record_tape=False,
                  inputs=[contacts.soft_contact_count, int(contacts.soft_contact_max), contacts.soft_contact_particle,
                          contacts.soft_contact_shape, contacts.soft_contact_normal, contacts.soft_contact_body_pos,
                          features if has else self._no_features, int(has), state.particle_q, state.body_q,
                          model.shape_body, model.shape_transform, model.shape_type, model.shape_scale,
                          model.shape_source_ptr, model._shape_mesh_properties, self.flips])


__all__ = ["ParityRecheck", "has_parity_meshes"]
