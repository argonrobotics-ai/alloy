# SPDX-License-Identifier: Apache-2.0
"""Culled mesh soft-contact candidates on CUDA (WP10's mixed-speed lever #4, coordinator 2026-11-07).

Newton's full-surface soft contacts test every particle against every mesh shape of its world
(``CollisionPipeline``'s vertex pairs, newton ``soft_contacts_mesh.py`` ``_detect_mesh_vertex_contacts``). On CUDA each
pair takes a 32-lane group that splits the mesh BVH, and almost every group exits at the shape's expanded bounding box:
in WP10's robot + cloth + water scene (robot_cloth_liquid, L4) that query took 4.3 of 11.1 s of kernel time per period
at 64 worlds and 9.4 of 22.1 s at 128, and at 256 worlds 32 x pairs passed 2^31 threads, so the build refused. Its
destination proxy also gave the SPH water candidates (2,940 of 3,616 particles per world); the stepper now gives a
particle destination candidates for its own particles only (``alloy.integration.stepper._proxy_pipeline``).

``attach(pipeline)`` gives a pipeline's mesh contact data a cull; Newton's launcher is replaced in its collide module by
``launch_soft_mesh_contacts``, which keeps Newton's launcher for every pipeline without one (and on the CPU) and
otherwise
1. runs Newton's launcher with no vertex pairs (face-vertex, edge-edge and penetration records, begin / end and the
   geometry evaluation, all unchanged);
2. culls the vertex pairs one thread per pair with the narrow kernel's own early exits (COLLIDE_PARTICLES and the
   shape-local bounds widened by gap + margin + radius, the same expressions), appending the survivors' indices;
3. runs Newton's vertex query on the survivors only, from a fixed grid (``SLOTS`` 32-lane groups striding over the
   survivors, so the launch no longer scales with the candidates), appending to the same record counter;
4. ends the mesh block again and re-evaluates its records (the face-vertex and edge-edge ones give the same values).
The records are those Newton writes, in another slot order; Newton's slots already come from atomic counters, and the
pipelines that use a cull put their soft records in a canonical order after each collide
(``alloy.coupling.contact_order``).

The vertex query's body is Newton's (newton ``_detect_mesh_vertex_contacts``, Apache-2.0), moved into a function so a
strided loop can run it. ``NEWTON_SOURCES`` pins the fingerprints of the Newton code this relies on; when the pin moves
and one differs, ``attach`` leaves the pipeline on Newton's path and logs why.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
from typing import Any

import warp as wp

from alloy._vendor import newton
from alloy._vendor.newton._src.geometry import soft_contacts_mesh as _scm
from alloy._vendor.newton._src.geometry.flags import MeshSignMethod
from alloy._vendor.newton._src.geometry.kernels import resolve_mesh_sign_method, triangle_closest_point
from alloy._vendor.newton._src.geometry.soft_contacts_mesh import (
    _MESH_FEATURE_INSIDE,
    _MESH_FEATURE_RECOVERY,
    _MESH_FEATURE_VT,
    _append_mesh_contact,
    _mesh_feature_flags,
    _mesh_partition_sign,
    _mesh_query_partition,
)
from alloy._vendor.newton._src.geometry.soft_contacts_sdf import _shape_frames
from alloy._vendor.newton._src.sim import collide as _collide

log = logging.getLogger("alloy.contact")

SLOTS = 1 << 15                     # 32-lane groups in the strided vertex query (1 M threads on CUDA)
QUERY_DEPTH = 5                     # Newton's 32-lane query group: one warp, which _sync_lanes relies on
CUDA_ONLY = True                    # the CPU keeps Newton's path (one lane per pair there); tests turn it off
COLLIDE_PARTICLES = wp.constant(int(newton.ShapeFlags.COLLIDE_PARTICLES))
# sha256[:16] of the Newton sources this module copies or calls (newton pin of 2026-11-07)
NEWTON_SOURCES = {"_detect_mesh_vertex_contacts": "98676e99637ac6ac", "launch_soft_mesh_contacts": "cf0f768b70edae25",
                  "_end_mesh_contacts": "9fbbd62efc72b9fd", "_launch_evaluate_mesh_contacts": "97c88ada3901fa40"}
_NEWTON_LAUNCH = _scm.launch_soft_mesh_contacts


def _fingerprint(name: str) -> str:
    obj = getattr(_scm, name)
    return hashlib.sha256(inspect.getsource(getattr(obj, "func", obj)).encode()).hexdigest()[:16]


def newton_drift() -> list[str]:
    """Names in ``NEWTON_SOURCES`` whose Newton source changed, and the query group size if it is no longer one warp
    (``_sync_lanes`` syncs the full warp; empty: the copy matches the pin)."""
    drift = [name for name, digest in NEWTON_SOURCES.items() if _fingerprint(name) != digest]
    if int(_scm._MESH_QUERY_PARTITION_DEPTH) != QUERY_DEPTH:
        drift.append("_MESH_QUERY_PARTITION_DEPTH")
    return drift


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__syncwarp();
#endif
""")
def _sync_lanes():
    """Reconverge the 32 lanes of a query group before the next strided pair (its sign query is warp-collective)."""
    ...


@wp.func
def _outside_bounds(x_local: wp.vec3, lower: wp.vec3, upper: wp.vec3, threshold: float) -> bool:
    lower_bound = lower - wp.vec3(threshold)
    upper_bound = upper + wp.vec3(threshold)
    return (x_local[0] < lower_bound[0] or x_local[1] < lower_bound[1] or x_local[2] < lower_bound[2]
            or x_local[0] > upper_bound[0] or x_local[1] > upper_bound[1] or x_local[2] > upper_bound[2])


@wp.kernel(enable_backward=False)
def _cull_vertex_pairs(
    vt_pairs: wp.array[wp.vec2i], particle_q: wp.array[wp.vec3], particle_radius: wp.array[float],
    body_q: wp.array[wp.transform], shape_transform: wp.array[wp.transform], shape_body: wp.array[wp.int32],
    shape_flags: wp.array[wp.int32], shape_lower: wp.array[wp.vec3], shape_upper: wp.array[wp.vec3],
    shape_margin: wp.array[float], gap: float, survivors: wp.array[wp.int32], survivor_count: wp.array[wp.int32],
):
    """Keep the pairs Newton's vertex query does not reject before its mesh queries (same tests, same arithmetic)."""
    tid = wp.tid()
    pair = vt_pairs[tid]
    particle_index, shape_index = pair[0], pair[1]
    if (shape_flags[shape_index] & COLLIDE_PARTICLES) == 0:
        return
    _X_bs, _X_ws, X_sw = _shape_frames(shape_body, body_q, shape_transform, shape_index)
    x_local = wp.transform_point(X_sw, particle_q[particle_index])
    s_margin = shape_margin[shape_index] if shape_margin.shape[0] > 0 else 0.0
    threshold = gap + s_margin + particle_radius[particle_index]
    if _outside_bounds(x_local, shape_lower[shape_index], shape_upper[shape_index], threshold):
        return
    survivors[wp.atomic_add(survivor_count, 0, 1)] = tid


@wp.func
def _vertex_pair_contacts(
    pair: wp.vec2i, lane: int, partition_depth: int, particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float], body_q: wp.array[wp.transform], shape_transform: wp.array[wp.transform],
    shape_body: wp.array[wp.int32], shape_flags: wp.array[wp.int32], shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64], shape_mesh_properties: wp.array[wp.int32],
    shape_lower: wp.array[wp.vec3], shape_upper: wp.array[wp.vec3], shape_margin: wp.array[float], gap: float,
    contact_max: wp.int32, contact_count: wp.array[wp.int64], features: wp.array[wp.vec3i],
    contact_shapes: wp.array[int],
):
    """Newton's ``_detect_mesh_vertex_contacts`` body for one pair (module docstring)."""
    partitions = 1 << partition_depth
    particle_index = pair[0]
    shape_index = pair[1]
    if (shape_flags[shape_index] & COLLIDE_PARTICLES) == 0:
        return
    px = particle_q[particle_index]
    radius = particle_radius[particle_index]
    _X_bs, _X_ws, X_sw = _shape_frames(shape_body, body_q, shape_transform, shape_index)
    x_local = wp.transform_point(X_sw, px)
    scale = shape_scale[shape_index]
    s_margin = shape_margin[shape_index] if shape_margin.shape[0] > 0 else 0.0
    threshold = gap + s_margin + radius
    lower_bound = shape_lower[shape_index] - wp.vec3(threshold)
    upper_bound = shape_upper[shape_index] + wp.vec3(threshold)
    if _outside_bounds(x_local, shape_lower[shape_index], shape_upper[shape_index], threshold):
        return
    mesh = shape_source_ptr[shape_index]
    min_scale = wp.min(wp.min(wp.abs(scale[0]), wp.abs(scale[1])), wp.abs(scale[2]))
    x_mesh = wp.cw_div(x_local, scale)
    r_mesh = threshold / min_scale
    sign_method = resolve_mesh_sign_method(shape_mesh_properties[shape_index])
    recovery_radius = wp.length(upper_bound - lower_bound) / min_scale
    sign_and_face = _mesh_partition_sign(
        mesh, x_mesh, r_mesh, recovery_radius, partitions, sign_method == MeshSignMethod.PARITY
    )
    vertex_sign = sign_and_face[0]
    recovery_face = sign_and_face[1]
    if lane == 0 and recovery_face >= 0:
        _append_mesh_contact(
            _MESH_FEATURE_VT + _MESH_FEATURE_INSIDE + _MESH_FEATURE_RECOVERY, particle_index,
            shape_index, recovery_face, contact_max, contact_count, features, contact_shapes,
        )
    root = _mesh_query_partition(mesh, lane, partition_depth)
    if root == -2:
        return
    query = wp.bvh_query_sphere(wp.mesh_get_bvh(mesh), x_mesh, r_mesh, root)
    face = wp.int32(0)
    while wp.bvh_query_next(query, face):
        a = wp.cw_mul(wp.mesh_get_point(mesh, face * 3 + 0), scale)
        b = wp.cw_mul(wp.mesh_get_point(mesh, face * 3 + 1), scale)
        c = wp.cw_mul(wp.mesh_get_point(mesh, face * 3 + 2), scale)
        cp, _rigid_bary, _feature = triangle_closest_point(a, b, c, x_local)
        if wp.length(x_local - cp) < threshold:
            _append_mesh_contact(
                _mesh_feature_flags(_MESH_FEATURE_VT, vertex_sign), particle_index, shape_index, face,
                contact_max, contact_count, features, contact_shapes,
            )


@wp.kernel(enable_backward=False)
def _detect_culled_vertex_contacts(
    vt_pairs: wp.array[wp.vec2i], survivors: wp.array[wp.int32], survivor_count: wp.array[wp.int32], slots: int,
    partition_depth: int, particle_q: wp.array[wp.vec3], particle_radius: wp.array[float],
    body_q: wp.array[wp.transform], shape_transform: wp.array[wp.transform], shape_body: wp.array[wp.int32],
    shape_flags: wp.array[wp.int32], shape_scale: wp.array[wp.vec3], shape_source_ptr: wp.array[wp.uint64],
    shape_mesh_properties: wp.array[wp.int32], shape_lower: wp.array[wp.vec3], shape_upper: wp.array[wp.vec3],
    shape_margin: wp.array[float], gap: float, contact_max: wp.int32, contact_count: wp.array[wp.int64],
    features: wp.array[wp.vec3i], contact_shapes: wp.array[int],
):
    """Newton's vertex query over the survivors: each 32-lane group strides over them from a fixed grid."""
    slot = wp.tid() >> partition_depth
    lane = wp.tid() & ((1 << partition_depth) - 1)
    for k in range(slot, survivor_count[0], slots):
        _vertex_pair_contacts(
            vt_pairs[survivors[k]], lane, partition_depth, particle_q, particle_radius, body_q, shape_transform,
            shape_body, shape_flags, shape_scale, shape_source_ptr, shape_mesh_properties, shape_lower, shape_upper,
            shape_margin, gap, contact_max, contact_count, features, contact_shapes,
        )
        _sync_lanes()


class MeshCull:
    """Survivor buffer for one pipeline's mesh vertex pairs (capacity: every pair, so it cannot overflow): 4 B per pair
    beside Newton's 8 B (89 MB for robot_cloth_liquid's 22 M pairs at 256 worlds)."""

    def __init__(self, data: Any, device: Any) -> None:
        n = int(data.vertex_pairs.shape[0])
        self.survivors = wp.empty(max(n, 1), dtype=wp.int32, device=device)
        self.count = wp.zeros(1, dtype=wp.int32, device=device)
        self.slots = max(min(n, SLOTS), 1)
        self.no_pairs = wp.empty(0, dtype=wp.vec2i, device=device)

    def launch(self, model: Any, state: Any, contacts: Any, gap: float, data: Any) -> None:
        """Steps 2-3 of the module docstring: cull, then the strided vertex query appending to the mesh block."""
        vt_pairs = data.vertex_pairs
        if vt_pairs.shape[0] == 0:
            return
        device = model.device
        self.count.zero_()
        bounds = [model.shape_collision_aabb_lower, model.shape_collision_aabb_upper, model.shape_margin]
        wp.launch(_cull_vertex_pairs, dim=vt_pairs.shape[0], device=device, record_tape=False, inputs=[
            vt_pairs, state.particle_q, model.particle_radius, state.body_q, model.shape_transform, model.shape_body,
            model.shape_flags, *bounds, gap], outputs=[self.survivors, self.count])
        depth = QUERY_DEPTH
        wp.launch(_detect_culled_vertex_contacts, dim=self.slots << depth, device=device, record_tape=False,
                  block_dim=64, inputs=[
                      vt_pairs, self.survivors, self.count, self.slots, depth, state.particle_q, model.particle_radius,
                      state.body_q, model.shape_transform, model.shape_body, model.shape_flags, model.shape_scale,
                      model.shape_source_ptr, model._shape_mesh_properties, *bounds, gap,
                      int(contacts._soft_contact_mesh_features.shape[0])],
                  outputs=[data.contact_count, contacts._soft_contact_mesh_features, contacts.soft_contact_shape])


def launch_soft_mesh_contacts(*, model: Any, state: Any, contacts: Any, gap: float, data: Any) -> None:
    """Newton's mesh soft-contact launcher, with the vertex pairs culled where ``attach`` gave ``data`` a cull."""
    cull = getattr(data, "_alloy_cull", None)
    grad = bool(getattr(contacts.soft_contact_body_pos, "requires_grad", False))   # backward replays Newton's slots
    if cull is None or grad or (CUDA_ONLY and not model.device.is_cuda):
        _NEWTON_LAUNCH(model=model, state=state, contacts=contacts, gap=gap, data=data)
        return
    vt_pairs = data.vertex_pairs
    data.vertex_pairs = cull.no_pairs                      # 1. Newton's other records, begin / end, evaluation
    try:
        _NEWTON_LAUNCH(model=model, state=state, contacts=contacts, gap=gap, data=data)
    finally:
        data.vertex_pairs = vt_pairs
    contact_max = int(contacts._soft_contact_mesh_features.shape[0])
    if contact_max == 0:
        return
    cull.launch(model, state, contacts, gap, data)         # 2-3. cull, strided vertex query
    wp.launch(_scm._end_mesh_contacts, dim=1, device=model.device, record_tape=False,  # 4. end and evaluate again
              inputs=[data.contact_count, contacts.soft_contact_count, contact_max])
    _scm._launch_evaluate_mesh_contacts(model, state, contacts, data)


def install() -> None:
    """Route Newton's collide module through ``launch_soft_mesh_contacts`` (idempotent; a pipeline without a cull keeps
    Newton's launcher)."""
    _collide.launch_soft_mesh_contacts = launch_soft_mesh_contacts


def attach(pipeline: Any) -> bool:
    """Give ``pipeline``'s mesh vertex pairs a cull on CUDA; False (Newton's path) without mesh pairs, on the CPU, or
    when the Newton code this copies has changed (logged)."""
    data = getattr(pipeline, "_soft_mesh_contact_data", None)
    model = getattr(pipeline, "model", None)
    cpu_off = CUDA_ONLY and model is not None and not model.device.is_cuda
    if data is None or model is None or cpu_off or not int(data.vertex_pairs.shape[0]):
        return False
    drift = newton_drift()
    if drift:
        log.warning("mesh soft-contact cull off: Newton's %s changed since the copy (alloy.contact.mesh_cull)", drift)
        return False
    install()
    data._alloy_cull = MeshCull(data, model.device)
    return True


__all__ = ["CUDA_ONLY", "NEWTON_SOURCES", "QUERY_DEPTH", "SLOTS", "MeshCull", "attach", "install",
           "launch_soft_mesh_contacts", "newton_drift"]
