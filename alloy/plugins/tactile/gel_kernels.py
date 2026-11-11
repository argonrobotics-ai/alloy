# SPDX-License-Identifier: Apache-2.0
"""Warp kernels of ``GelSensor`` (tactile v2 physics).

Indexing:
- sensor g = w·S + s;
- depth cell (i, j), row-major, with row 0 at +y;
- coarse load cell (ci, cj), likewise;
- marker m.

The pipeline per step:
1. contacts -> the engine force and twist on each pad (int64 fixed point), plus the touching shapes (slots);
2. depth: sphere-trace each cell's ray (+z from -max_depth) against the touching shapes' SDFs (Newton's public
   ``sdf_*``);
3. coarse loads: depth summed per block;
4. per sensor: loads proportional to depth, carrying the engine's normal force, shear force and twist exactly, and
   the scale C that makes the normal displacement at the peak equal its depth;
5. markers: u = C·Σ G(q - c)·load_c.
"""

from __future__ import annotations

import warp as wp

from alloy._vendor.newton.geometry import sdf_box, sdf_capsule, sdf_cylinder, sdf_mesh, sdf_sphere

FIX = 16777216.0     # 2^24 counts per N (or N·m)
SLOTS = 8            # touching shapes per sensor
SPHERE, CAPSULE, CYLINDER, BOX, MESH = 3, 4, 6, 7, 8     # newton.GeoType
Z_AXIS = 2


@wp.func
def _fix(x: float) -> wp.int64:
    return wp.int64(wp.round(x * FIX))


@wp.func
def active(stamp: wp.array(dtype=wp.int32), w: int, every: int) -> bool:
    """True on the steps the fields are computed: the last step of each ``every`` (the stamp counts steps)."""
    return stamp[w] % every == every - 1


@wp.func
def shape_sdf(s: int, p_world: wp.vec3, body_q: wp.array(dtype=wp.transform), shape_body: wp.array(dtype=wp.int32),
              shape_xform: wp.array(dtype=wp.transform), shape_type: wp.array(dtype=wp.int32),
              shape_scale: wp.array(dtype=wp.vec3), shape_src: wp.array(dtype=wp.uint64)) -> float:
    """Signed distance from a world point to shape ``s`` (Newton model shapes, axis z for capsule and cylinder)."""
    x = shape_xform[s]
    b = shape_body[s]
    if b >= 0:
        x = wp.transform_multiply(body_q[b], x)
    p = wp.transform_point(wp.transform_inverse(x), p_world)
    t = shape_type[s]
    sc = shape_scale[s]
    if t == SPHERE:
        return sdf_sphere(p, sc[0])
    if t == BOX:
        return sdf_box(p, sc[0], sc[1], sc[2])
    if t == CAPSULE:
        return sdf_capsule(p, sc[0], sc[1], Z_AXIS)
    if t == CYLINDER:
        return sdf_cylinder(p, sc[0], sc[1], Z_AXIS)
    if t == MESH:
        smin = wp.min(sc[0], wp.min(sc[1], sc[2]))
        return sdf_mesh(shape_src[s], wp.cw_div(p, sc), 1.0) * smin
    return 1.0e6


@wp.func
def _side(g: int, x: wp.vec3, f_body: wp.vec3, rp: wp.quat, other: int, capture: float,
          force_acc: wp.array(dtype=wp.int64), torque_acc: wp.array(dtype=wp.int64),
          slots: wp.array(dtype=wp.int32)):
    if wp.abs(x[2]) > capture:
        return
    f = wp.quat_rotate_inv(rp, f_body)
    wp.atomic_add(force_acc, g * 3 + 0, _fix(-f[2]))
    wp.atomic_add(force_acc, g * 3 + 1, _fix(f[0]))
    wp.atomic_add(force_acc, g * 3 + 2, _fix(f[1]))
    wp.atomic_add(torque_acc, g, _fix(x[0] * f[1] - x[1] * f[0]))
    if other < 0:
        return
    for k in range(SLOTS):
        old = wp.atomic_cas(slots, g * SLOTS + k, -1, other)
        if old == -1 or old == other:
            return


@wp.kernel
def gel_contacts(count: wp.array(dtype=wp.int32), shape0: wp.array(dtype=wp.int32),
                 shape1: wp.array(dtype=wp.int32), point0: wp.array(dtype=wp.vec3), point1: wp.array(dtype=wp.vec3),
                 contact_f: wp.array(dtype=wp.spatial_vector), shape_body: wp.array(dtype=wp.int32),
                 body_q: wp.array(dtype=wp.transform), pad_of_body: wp.array(dtype=wp.int32), sensors: int,
                 pad_origin: wp.array(dtype=wp.vec3), pad_rot: wp.array(dtype=wp.quat), capture: float,
                 stamp: wp.array(dtype=wp.int32), every: int, force_acc: wp.array(dtype=wp.int64),
                 torque_acc: wp.array(dtype=wp.int64), slots: wp.array(dtype=wp.int32)):
    """One thread per exported contact: force and twist on each gel pad it touches, and the shape it touches."""
    i = wp.tid()
    if i >= count[0]:
        return
    f0 = wp.spatial_top(contact_f[i])
    s0 = shape0[i]
    s1 = shape1[i]
    for side in range(2):
        s = s0
        o = s1
        pt = point0[i]
        fw = f0
        if side == 1:
            s = s1
            o = s0
            pt = point1[i]
            fw = -f0
        if s >= 0:
            b = shape_body[s]
            if b >= 0:
                g = pad_of_body[b]
                if g >= 0 and active(stamp, g // sensors, every):
                    p = g % sensors
                    rp = pad_rot[p]
                    x = wp.quat_rotate_inv(rp, pt - pad_origin[p])
                    fb = wp.quat_rotate_inv(wp.transform_get_rotation(body_q[b]), fw)
                    _side(g, x, fb, rp, o, capture, force_acc, torque_acc, slots)


@wp.func
def _min_sdf(g: int, p: wp.vec3, slots: wp.array(dtype=wp.int32), body_q: wp.array(dtype=wp.transform),
             shape_body: wp.array(dtype=wp.int32), shape_xform: wp.array(dtype=wp.transform),
             shape_type: wp.array(dtype=wp.int32), shape_scale: wp.array(dtype=wp.vec3),
             shape_src: wp.array(dtype=wp.uint64)) -> float:
    d = float(1.0e6)
    for k in range(SLOTS):
        s = slots[g * SLOTS + k]
        if s >= 0:
            d = wp.min(d, shape_sdf(s, p, body_q, shape_body, shape_xform, shape_type, shape_scale, shape_src))
    return d


@wp.kernel
def gel_depth(slots: wp.array(dtype=wp.int32), pad_body: wp.array(dtype=wp.int32), sensors: int,
              pad_origin: wp.array(dtype=wp.vec3), pad_rot: wp.array(dtype=wp.quat), rows: int, cols: int,
              cell: float, max_depth: float, body_q: wp.array(dtype=wp.transform),
              shape_body: wp.array(dtype=wp.int32), shape_xform: wp.array(dtype=wp.transform),
              shape_type: wp.array(dtype=wp.int32), shape_scale: wp.array(dtype=wp.vec3),
              shape_src: wp.array(dtype=wp.uint64), stamp: wp.array(dtype=wp.int32), every: int,
              depth: wp.array(dtype=wp.float32)):
    """One thread per (sensor, cell): penetration of the touching shapes into the undeformed gel surface."""
    k = wp.tid()
    n = rows * cols
    g = k // n
    if not active(stamp, g // sensors, every):
        return
    c = k - g * n
    i = c // cols
    j = c - i * cols
    depth[k] = 0.0
    if slots[g * SLOTS] < 0:
        return
    p = g % sensors
    xs = wp.transform_multiply(body_q[pad_body[g]], wp.transform(pad_origin[p], pad_rot[p]))
    x = (float(j) + 0.5) * cell - float(cols) * cell * 0.5
    y = float(rows) * cell * 0.5 - (float(i) + 0.5) * cell
    z = -max_depth
    d = _min_sdf(g, wp.transform_point(xs, wp.vec3(x, y, z)), slots, body_q, shape_body, shape_xform, shape_type,
                 shape_scale, shape_src)
    if d <= 0.0:
        depth[k] = max_depth
        return
    for _it in range(64):
        if d < 1.0e-7 or z >= 0.0:
            break
        z = z + d
        d = _min_sdf(g, wp.transform_point(xs, wp.vec3(x, y, z)), slots, body_q, shape_body, shape_xform,
                     shape_type, shape_scale, shape_src)
    if z < 0.0 and d < 1.0e-6:
        depth[k] = -z
