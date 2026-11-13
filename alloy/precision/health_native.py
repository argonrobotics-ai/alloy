# SPDX-License-Identifier: Apache-2.0
"""MuJoCo's own contacts in the HealthMonitor (SCR-SC0-1; GAP-E1's PEN rule, ``health_pen``): one kernel per
SolverMuJoCo's contact buffer, device only and graph-capturable.

Per contact: the true depth ``-dist`` raises ``Accum.pen_native`` (``max_penetration_m``, the deepest contact's
names); the depth past its allowance (``health_pen.contact_allowance``: 0 unless ``relax_soft_contacts`` marked one of
its shapes) raises ``Accum.pen_native_flag`` (PEN_WARN past 1 mm, the 1 cm dwell rule in ``classify``) and the
contact's level ``Accum.pen_native_fail`` (``health_pen.past_limit``). Unmarked contacts: flag depth = true depth, so
the rule is the one of 6cb983107 bit for bit.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from alloy.precision.health_failure import resolve_native
from alloy.precision.health_kernels import PEN_NATIVE, Accum, note_deepest, pen_tag
from alloy.precision.health_pen import contact_allowance, designed_pair, moving_extent, past_limit
from alloy.precision.health_source_shapes import SolidTable, contact_thickness

_VEC5 = wp.types.vector(length=5, dtype=wp.float32)        # MuJoCo Warp's contact.solimp rows


@wp.func
def geom_to_shape(geom_shape: wp.array2d[wp.int32], w: int, g: int) -> int:
    """The Newton shape behind MuJoCo geom ``g`` of world ``w`` (-1 when it has none)."""
    if geom_shape.shape[0] == 0 or g < 0 or g >= geom_shape.shape[1]:
        return -1
    return geom_shape[w % geom_shape.shape[0], g]


@wp.kernel
def check_mjw_contacts(nacon: wp.array[wp.int32], dist: wp.array[wp.float32], worldid: wp.array[wp.int32],
                       geom: wp.array[wp.vec2i], geom_shape: wp.array2d[wp.int32], shape_thick: wp.array[wp.float32],
                       pair_mark: wp.array[wp.int32], world_count: wp.int32, pen_fail: wp.float32, hard: wp.float32,
                       frame: wp.array[wp.mat33], solimp: wp.array[_VEC5],
                       shape_allow: wp.array[wp.float32],
                       solids: SolidTable, shape_type: wp.array[wp.int32], shape_scale: wp.array[wp.vec3],
                       shape_transform: wp.array[wp.transform], shape_body: wp.array[wp.int32],
                       body_q: wp.array[wp.transform], acc: Accum):
    """MuJoCo Warp's own contacts (SCR-SC0-1): penetration = -dist (signed surface distance); maxima only (D3)."""
    k = wp.tid()
    if k >= nacon[0]:
        return
    w = worldid[k]
    if w < 0 or w >= world_count:
        return
    g = geom[k]
    n = frame[k][0]  # MuJoCo's contact normal (frame row 0, world)
    sa, sb = geom_to_shape(geom_shape, w, g[0]), geom_to_shape(geom_shape, w, g[1])
    pen = -dist[k]
    if pen <= 0.0:
        return
    wp.atomic_max(acc.pen_native, w, pen)
    note_deepest(acc, w, pen, pen_tag(PEN_NATIVE, k))
    flag = pen - contact_allowance(shape_allow, sa, sb, solimp[k][2])     # soft on purpose: past its allowance
    if flag <= 0.0:
        return
    wp.atomic_max(acc.pen_native_flag, w, flag)
    ta = contact_thickness(solids, shape_type, shape_scale, shape_transform, shape_body, shape_thick, body_q, sa, n)
    tb = contact_thickness(solids, shape_type, shape_scale, shape_transform, shape_body, shape_thick, body_q, sb, n)
    thin = designed_pair(wp.min(ta, tb), pair_mark, sa, sb)                # GAP-E1: health_pen's one rule
    size = moving_extent(sa, sb, shape_thick, shape_body)
    wp.atomic_max(acc.pen_native_fail, w, past_limit(flag, wp.min(pen_fail, 0.5 * thin), hard, size))


def check_native(monitor: Any, state: Any) -> None:
    """Penetration from each SolverMuJoCo's MuJoCo Warp contacts of the last step (device only, capturable)."""
    m, pen_fail = monitor.model, float(monitor.thresholds.pen_fail_m)
    body_q = state.body_q if state.body_q is not None else monitor._empty[wp.transform]
    for src in monitor._mjw_contacts:
        d = src.data
        wp.launch(check_mjw_contacts, dim=int(d.naconmax),
                  inputs=[d.nacon, d.contact.dist, d.contact.worldid, d.contact.geom, src.geom_shape, monitor._thick,
                          monitor._pair_mark, monitor.world_count, pen_fail, float(monitor.cfg.native_pen_hard),
                          d.contact.frame, d.contact.solimp, monitor._shape_allow, monitor._solids, m.shape_type,
                          m.shape_scale, m.shape_transform, m.shape_body, body_q, monitor._acc], device=monitor.device)
        wp.launch(resolve_native, dim=monitor.world_count, inputs=[d.contact.geom, src.geom_shape, monitor._acc],
                  device=monitor.device)
        monitor._mark_pen_measured()
