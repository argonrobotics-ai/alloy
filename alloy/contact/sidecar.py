# SPDX-License-Identifier: Apache-2.0
"""Sidecar logic behind ``AlloyContactSidecar.from_pipeline`` / ``assemble`` (types in alloy.contact.types, WP0).

The sidecar holds Alloy's per-contact extras indexed by the rigid slot of the layer's ``newton.Contacts``. Streams
are written in the fixed order NATIVE, RIGID, PRECISION, CROSS by ``fill_stream`` and counted by
``assemble_streams`` (integer atomics only: deterministic, §3.7). ``weight`` is the D22 sdf-mode share of the pair
stiffness, normalised over the load-bearing rows of its contact patch: ``n`` counts the rows of the same (pair,
world, normal bucket) with separation ≤ 0. Speculative rows carry no force (counting them gave the bottom face of the
CT-1 peg 0.35 K: 74 µm sink at 50 N instead of 25), and each patch of the pair gets the full K (one shared K gave
each of a leaning peg's two wall patches K/2). The normal bucket is the nearest of the 26 cube directions (a cone of
~27° about each axis). A new row gets ``1/n``; a matched (PERSISTENT) row keeps its weight and relaxes toward ``1/n``
by ``WEIGHT_RELAX`` per step, so K neither jumps when one row appears under load nor stays inflated on the few rows
of a first touch.
"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from alloy._vendor import newton
from alloy.contact.types import AlloyContactSidecar, ContactClass, ContactFlags
from alloy.core.log import get_logger

log = get_logger("contact.sidecar")

F_SPECULATIVE = wp.constant(wp.uint8(int(ContactFlags.SPECULATIVE)))
F_NEW = wp.constant(wp.uint8(int(ContactFlags.NEW)))
F_PERSISTENT = wp.constant(wp.uint8(int(ContactFlags.PERSISTENT)))
WEIGHT_RELAX = 0.05     # per step: persistent weights reach 1/n with a ~20-step time constant (no force jumps)
PATCH_BUCKETS = 27      # normal buckets per (pair, world): cube directions, index 13 (zero) unused


def _ensure_contact_arrays(contacts: newton.Contacts) -> None:
    """Upgrade a pipeline-made ``Contacts`` in place: per-contact shape properties (D13/D22) and ``force``."""
    n = contacts.rigid_contact_max
    dev = contacts.rigid_contact_shape0.device
    if contacts.rigid_contact_stiffness is None:
        contacts.rigid_contact_stiffness = wp.zeros(n, dtype=wp.float32, device=dev)
        contacts.rigid_contact_damping = wp.zeros(n, dtype=wp.float32, device=dev)
        contacts.rigid_contact_friction = wp.zeros(n, dtype=wp.float32, device=dev)
        contacts.per_contact_shape_properties = True
    if getattr(contacts, "rigid_contact_force", None) is None:
        contacts.rigid_contact_force = wp.zeros(n, dtype=wp.vec3, device=dev)


def sidecar_from_pipeline(cls: type[AlloyContactSidecar], pipeline: newton.CollisionPipeline, *,
                          xp_contact_max: int = 0, world_count: int = 1,
                          per_world_capacity: int | None = None) -> tuple[newton.Contacts, AlloyContactSidecar]:
    """``pipeline.contacts()`` (so every pipeline buffer check passes) upgraded with per-contact properties and
    ``force``, plus a sidecar of the same capacity. ``per_world_capacity`` is checked, not enforced (Newton has no
    per-world partitions, §8 upstream ask)."""
    contacts = pipeline.contacts()
    _ensure_contact_arrays(contacts)
    cap = contacts.rigid_contact_max
    if per_world_capacity is not None and cap < per_world_capacity * world_count:
        log.warning("contact capacity %d < per_world_capacity %d x %d worlds", cap, per_world_capacity, world_count)
    side = cls.allocate(cap, xp_contact_max=xp_contact_max, device=contacts.rigid_contact_shape0.device)
    return contacts, side


@wp.kernel
def _count_streams(count: wp.array[wp.int32], cls: wp.array[wp.int8], stream_count: wp.array[wp.int32]):
    i = wp.tid()
    if i < count[0]:
        wp.atomic_add(stream_count, int(cls[i]), 1)


def assemble_streams(sidecar: AlloyContactSidecar, contacts: newton.Contacts) -> None:
    """Recount ``stream_count[cls]`` over the live rows of ``contacts`` (streams were written in fixed order)."""
    sidecar.stream_count.zero_()
    wp.launch(_count_streams, dim=contacts.rigid_contact_max,
              inputs=[contacts.rigid_contact_count, sidecar.cls, sidecar.stream_count],
              device=sidecar.stream_count.device)


@dataclass(frozen=True)
class StreamTables:
    """Per-shape lookups the fill kernel needs (built once by the layer)."""

    shape_pair: wp.array        # [S] int32, precision pair index or -1
    pair_class: wp.array        # [P] int8, PRECISION_SDF or PRECISION_HYDRO
    pair_count: wp.array        # [P*W*PATCH_BUCKETS] int32 scratch: load-bearing rows per (pair, world, bucket)
    world_count: int


@wp.func
def _row_world(shape_a: int, shape_b: int, shape_world: wp.array[wp.int32]) -> int:
    w = shape_world[shape_a]
    if w < 0:
        w = shape_world[shape_b]
    return w


@wp.func
def _row_separation(i: int, shape_body: wp.array[wp.int32], body_q: wp.array[wp.transform],
                    shape0: wp.array[wp.int32], shape1: wp.array[wp.int32], point0: wp.array[wp.vec3],
                    point1: wp.array[wp.vec3], normal: wp.array[wp.vec3], margin0: wp.array[wp.float32],
                    margin1: wp.array[wp.float32]) -> float:
    b0 = shape_body[shape0[i]]
    b1 = shape_body[shape1[i]]
    x0 = point0[i]
    x1 = point1[i]
    if b0 >= 0:
        x0 = wp.transform_point(body_q[b0], x0)
    if b1 >= 0:
        x1 = wp.transform_point(body_q[b1], x1)
    return wp.dot(normal[i], x1 - x0) - (margin0[i] + margin1[i])


@wp.func
def _patch(p: int, w: int, world_count: int, n: wp.vec3) -> int:
    """Slot of (pair, world, normal bucket) in ``pair_count``; the bucket is the nearest cube direction of n."""
    m = wp.max(wp.max(wp.abs(n[0]), wp.abs(n[1])), wp.max(wp.abs(n[2]), 1.0e-12))
    b = (int(wp.round(n[0] / m)) + 1) * 9 + (int(wp.round(n[1] / m)) + 1) * 3 + int(wp.round(n[2] / m)) + 1
    return (p * world_count + wp.max(w, 0)) * PATCH_BUCKETS + b


@wp.kernel
def _fill_rows(start: wp.array[wp.int32], count: wp.array[wp.int32], default_cls: wp.int8,
               shape_world: wp.array[wp.int32],
               shape_body: wp.array[wp.int32], body_q: wp.array[wp.transform], shape0: wp.array[wp.int32],
               shape1: wp.array[wp.int32], point0: wp.array[wp.vec3], point1: wp.array[wp.vec3],
               normal: wp.array[wp.vec3], margin0: wp.array[wp.float32], margin1: wp.array[wp.float32],
               shape_pair: wp.array[wp.int32], pair_class: wp.array[wp.int8], world_count: int,
               pair_count: wp.array[wp.int32], out_cls: wp.array[wp.int8], out_pair: wp.array[wp.int32],
               out_flags: wp.array[wp.uint8], out_world: wp.array[wp.int32], out_area: wp.array[wp.float32],
               out_toi: wp.array[wp.float32]):
    i = wp.tid()
    k = start[0] + i
    if i >= count[0] or k >= out_cls.shape[0]:
        return                      # rows past the dst capacity were dropped by append_rows (counted there)
    sa = shape0[k]
    sb = shape1[k]
    w = _row_world(sa, sb, shape_world)
    p = shape_pair[sa]
    if p >= 0 and shape_pair[sb] != p:
        p = -1                      # a precision shape meeting an ordinary partner is an ordinary row
    c = default_cls
    flags = wp.uint8(0)
    if _row_separation(k, shape_body, body_q, shape0, shape1, point0, point1, normal, margin0, margin1) > 0.0:
        flags = flags | F_SPECULATIVE
    if p >= 0:
        c = pair_class[p]
        if flags == wp.uint8(0):                                            # load-bearing rows only
            wp.atomic_add(pair_count, _patch(p, w, world_count, normal[k]), 1)
    out_cls[k] = c
    out_pair[k] = p
    out_flags[k] = flags
    out_world[k] = w
    out_area[k] = 0.0
    out_toi[k] = 1.0


@wp.kernel
def _fill_weights(start: wp.array[wp.int32], count: wp.array[wp.int32], world_count: int, normal: wp.array[wp.vec3],
                  pair: wp.array[wp.int32],
                  world: wp.array[wp.int32], match_index: wp.array[wp.int32], has_match: int,
                  pair_count: wp.array[wp.int32], weight_prev: wp.array[wp.float32], flags: wp.array[wp.uint8],
                  weight: wp.array[wp.float32]):
    i = wp.tid()
    k = start[0] + i
    if i >= count[0] or k >= weight.shape[0]:
        return
    p = pair[k]
    if p < 0:
        weight[k] = 1.0
        return
    m = int(-1)
    if has_match != 0:
        m = match_index[i]                  # the source buffer's own slot i (k is the slot in dst)
    target = 1.0 / float(wp.max(pair_count[_patch(p, world[k], world_count, normal[k])], 1))
    if m >= 0:
        wprev = weight_prev[m]              # previous step's row m of the same source buffer
        weight[k] = wprev + WEIGHT_RELAX * (target - wprev)
        flags[k] = flags[k] | F_PERSISTENT
    else:
        weight[k] = target
        flags[k] = flags[k] | F_NEW


@wp.kernel
def _save_weights(start: wp.array[wp.int32], count: wp.array[wp.int32], weight: wp.array[wp.float32],
                  weight_prev: wp.array[wp.float32]):
    i = wp.tid()
    k = start[0] + i
    if i < count[0] and k < weight.shape[0]:
        weight_prev[i] = weight[k]


def fill_stream(side: AlloyContactSidecar, tables: StreamTables, model: newton.Model, state: newton.State,
                dst: newton.Contacts, src: newton.Contacts, start: wp.array, weight_prev: wp.array) -> None:
    """Fill sidecar rows ``[start[0], start[0] + src.count)`` of ``dst`` (the rows of ``src`` already copied there;
    ``dst is src`` with ``start = 0`` when the stream is the whole buffer). ``start`` is a device int32[1] so the
    offset (the bulk count under ``native="newton"``) never needs a host readback (graph-safe). ``weight_prev``
    [src capacity] holds the previous step's weights of ``src`` by its own slots (matched rows keep theirs)."""
    dev = side.cls.device
    tables.pair_count.zero_()
    cap = src.rigid_contact_max
    wp.launch(_fill_rows, dim=cap, device=dev, inputs=[
        start, src.rigid_contact_count, wp.int8(int(ContactClass.RIGID)), model.shape_world, model.shape_body,
        state.body_q, dst.rigid_contact_shape0, dst.rigid_contact_shape1, dst.rigid_contact_point0,
        dst.rigid_contact_point1, dst.rigid_contact_normal, dst.rigid_contact_margin0, dst.rigid_contact_margin1,
        tables.shape_pair,
        tables.pair_class, tables.world_count, tables.pair_count, side.cls, side.pair, side.flags, side.world,
        side.area, side.toi])
    match = getattr(src, "rigid_contact_match_index", None)
    has_match = int(match is not None)
    match_arr = match if match is not None else side.pair
    wp.launch(_fill_weights, dim=cap, device=dev, inputs=[
        start, src.rigid_contact_count, tables.world_count, dst.rigid_contact_normal, side.pair, side.world,
        match_arr, has_match,
        tables.pair_count, weight_prev, side.flags, side.weight])
    wp.launch(_save_weights, dim=cap, device=dev, inputs=[start, src.rigid_contact_count, side.weight, weight_prev])
