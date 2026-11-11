# SPDX-License-Identifier: Apache-2.0
"""Contacts buffer helpers: append the rows of one ``newton.Contacts`` to another at a device-side offset.

Used when one solver-facing buffer carries several streams (bulk + precision under ``native="newton"``, or several
precision pipelines when worlds are split to stay under Newton's deterministic reducer cap). Offsets and counts
stay on the device (``start`` int32[1]) so the sequence is CUDA-graph safe. Rows that do not fit set an overflow
counter instead of being written. Only the fields solvers read are copied (shapes, points, offsets, normal, margins,
per-contact stiffness/damping/friction); ``rigid_contact_match_index`` of copied rows is -1.
"""

from __future__ import annotations

import warp as wp

from alloy._vendor import newton


@wp.kernel
def _begin_append(dst_count: wp.array[wp.int32], src_count: wp.array[wp.int32], dst_max: int,
                  start: wp.array[wp.int32], overflow: wp.array[wp.int32]):
    s = dst_count[0]
    n = src_count[0]
    start[0] = s
    if s + n > dst_max:
        wp.atomic_add(overflow, 0, s + n - dst_max)


@wp.kernel
def _copy_rows(start: wp.array[wp.int32], src_count: wp.array[wp.int32], dst_max: int,
               s0: wp.array[wp.int32], s1: wp.array[wp.int32], p0: wp.array[wp.vec3], p1: wp.array[wp.vec3],
               o0: wp.array[wp.vec3], o1: wp.array[wp.vec3], nrm: wp.array[wp.vec3], m0: wp.array[wp.float32],
               m1: wp.array[wp.float32], d_s0: wp.array[wp.int32], d_s1: wp.array[wp.int32],
               d_p0: wp.array[wp.vec3], d_p1: wp.array[wp.vec3], d_o0: wp.array[wp.vec3], d_o1: wp.array[wp.vec3],
               d_nrm: wp.array[wp.vec3], d_m0: wp.array[wp.float32], d_m1: wp.array[wp.float32]):
    i = wp.tid()
    k = start[0] + i
    if i >= src_count[0] or k >= dst_max:
        return
    d_s0[k] = s0[i]
    d_s1[k] = s1[i]
    d_p0[k] = p0[i]
    d_p1[k] = p1[i]
    d_o0[k] = o0[i]
    d_o1[k] = o1[i]
    d_nrm[k] = nrm[i]
    d_m0[k] = m0[i]
    d_m1[k] = m1[i]


@wp.kernel
def _copy_props(start: wp.array[wp.int32], src_count: wp.array[wp.int32], dst_max: int,
                k_in: wp.array[wp.float32], d_in: wp.array[wp.float32], f_in: wp.array[wp.float32], has_src: int,
                k_out: wp.array[wp.float32], d_out: wp.array[wp.float32], f_out: wp.array[wp.float32],
                match_out: wp.array[wp.int32], has_match: int):
    i = wp.tid()
    k = start[0] + i
    if i >= src_count[0] or k >= dst_max:
        return
    if has_src != 0:
        k_out[k] = k_in[i]
        d_out[k] = d_in[i]
        f_out[k] = f_in[i]
    else:
        k_out[k] = 0.0
        d_out[k] = 0.0
        f_out[k] = 0.0
    if has_match != 0:
        match_out[k] = -1


@wp.kernel
def _end_append(start: wp.array[wp.int32], src_count: wp.array[wp.int32], dst_max: int,
                dst_count: wp.array[wp.int32]):
    dst_count[0] = wp.min(start[0] + src_count[0], dst_max)


def append_rows(dst: newton.Contacts, src: newton.Contacts, start: wp.array, overflow: wp.array) -> None:
    """Copy ``src``'s live rows to ``dst`` after its live rows; ``start`` receives the offset (device int32[1]),
    ``overflow`` accumulates rows that did not fit. ``dst`` must have per-contact shape properties."""
    dev = dst.rigid_contact_shape0.device
    cap = dst.rigid_contact_max
    wp.launch(_begin_append, dim=1, inputs=[dst.rigid_contact_count, src.rigid_contact_count, cap, start, overflow],
              device=dev)
    wp.launch(_copy_rows, dim=src.rigid_contact_max, device=dev, inputs=[
        start, src.rigid_contact_count, cap, src.rigid_contact_shape0, src.rigid_contact_shape1,
        src.rigid_contact_point0, src.rigid_contact_point1, src.rigid_contact_offset0, src.rigid_contact_offset1,
        src.rigid_contact_normal, src.rigid_contact_margin0, src.rigid_contact_margin1, dst.rigid_contact_shape0,
        dst.rigid_contact_shape1, dst.rigid_contact_point0, dst.rigid_contact_point1, dst.rigid_contact_offset0,
        dst.rigid_contact_offset1, dst.rigid_contact_normal, dst.rigid_contact_margin0, dst.rigid_contact_margin1])
    has_src = src.rigid_contact_stiffness is not None
    match = getattr(dst, "rigid_contact_match_index", None)
    wp.launch(_copy_props, dim=src.rigid_contact_max, device=dev, inputs=[
        start, src.rigid_contact_count, cap,
        src.rigid_contact_stiffness if has_src else dst.rigid_contact_stiffness,
        src.rigid_contact_damping if has_src else dst.rigid_contact_damping,
        src.rigid_contact_friction if has_src else dst.rigid_contact_friction, int(has_src),
        dst.rigid_contact_stiffness, dst.rigid_contact_damping, dst.rigid_contact_friction,
        match if match is not None else dst.rigid_contact_shape0, int(match is not None)])
    wp.launch(_end_append, dim=1, inputs=[start, src.rigid_contact_count, cap, dst.rigid_contact_count], device=dev)
