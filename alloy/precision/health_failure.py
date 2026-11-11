# SPDX-License-Identifier: Apache-2.0
"""First-failure snapshot of the HealthMonitor record (coordinator ruling 2026-11-04 05:40Z).

A world that goes INVALID is latched: held at its last good pose, so its later checks read clean. World 2 of T09's
route bug showed ``max_pair_pen_m`` 0 next to a real 256 um PAIR_PEN_FAIL. The record therefore keeps, beside the
latest values, the values of the check that first made the world INVALID (``StepHealth.fail_*``; the check index is
``first_bad_step``):
- flags, sim time;
- the deepest penetration, with its contact's shapes or particle;
- the precision-pair penetration, with its pair;
- speed and its body, joint errors, energy;
- the plug-in depth check that tripped, with its depth (``health_plugin``).

Kernels here, launched by ``HealthMonitor.update``:
- ``resolve_*``: after each contact source's check, turn the deepest-contact tag (``Accum.pen_arg``, a deterministic
  argmax) into shape / particle ids while that source's arrays are at hand;
- ``advance_time``: per-world sim time, when the caller passes ``dt`` (rolled back with the world);
- ``first_failure``: after ``classify``, copy the record's latest values into ``fail_*`` once, on the check where the
  world became INVALID.

Valid worlds' records are unchanged.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from alloy.precision.health_kernels import PEN_NATIVE, PEN_PARTICLE, PEN_RIGID, PEN_SOFT, SEV_INVALID, Accum
from alloy.precision.types import FAIL_FIELDS, FirstFailure, StepHealth

_INDEX = wp.constant(wp.uint64((1 << 29) - 1))


@wp.func
def _deepest(acc: Accum, w: int, source: int) -> int:
    """Index of world ``w``'s deepest contact if it came from ``source``, else -1."""
    a = acc.pen_arg[w]
    if a == wp.uint64(0) or int((a >> wp.uint64(29)) & wp.uint64(7)) != source:
        return -1
    return int(a & _INDEX)


@wp.func
def _set(acc: Accum, w: int, a: int, b: int, particle: int):
    acc.pen_shape_a[w] = a
    acc.pen_shape_b[w] = b
    acc.pen_particle[w] = particle


@wp.kernel
def resolve_rigid(shape0: wp.array[wp.int32], shape1: wp.array[wp.int32], acc: Accum):
    w = wp.tid()
    k = _deepest(acc, w, PEN_RIGID)
    if k >= 0:
        _set(acc, w, shape0[k], shape1[k], -1)


@wp.kernel
def resolve_soft(particle: wp.array[wp.int32], shape: wp.array[wp.int32], acc: Accum):
    w = wp.tid()
    k = _deepest(acc, w, PEN_SOFT)
    if k >= 0:
        _set(acc, w, shape[k], -1, particle[k])


@wp.kernel
def resolve_particle(acc: Accum):
    """Cut-front growth (SCR-SC5-3): the tag is the particle."""
    w = wp.tid()
    k = _deepest(acc, w, PEN_PARTICLE)
    if k >= 0:
        _set(acc, w, -1, -1, k)


@wp.func
def _geom_shape(geom_shape: wp.array2d[wp.int32], w: int, g: int) -> int:
    if geom_shape.shape[0] == 0 or g < 0 or g >= geom_shape.shape[1]:
        return -1
    return geom_shape[w % geom_shape.shape[0], g]


@wp.kernel
def resolve_native(geom: wp.array[wp.vec2i], geom_shape: wp.array2d[wp.int32], acc: Accum):
    """MuJoCo Warp's own contacts: geoms mapped to Newton shapes (-1 where a geom has none)."""
    w = wp.tid()
    k = _deepest(acc, w, PEN_NATIVE)
    if k >= 0:
        g = geom[k]
        _set(acc, w, _geom_shape(geom_shape, w, g[0]), _geom_shape(geom_shape, w, g[1]), -1)


@wp.kernel
def advance_time(dt: wp.float64, time_s: wp.array[wp.float64]):
    w = wp.tid()
    t = time_s[w]
    if wp.isnan(t):
        t = wp.float64(0.0)
    time_s[w] = t + dt


@wp.kernel
def first_failure(step_counter: wp.array[wp.int64], time_s: wp.array[wp.float64], acc: Accum,
                  record: wp.array[StepHealth]):
    """Once per INVALID episode: on the check where the world became INVALID (``first_bad_step`` is this check),
    keep that check's values beside the latest ones."""
    w = wp.tid()
    h = record[w]
    if h.severity != SEV_INVALID or h.first_bad_step != step_counter[0] or h.fail_flags != wp.uint32(0):
        return
    h.fail_flags = h.flags
    h.fail_time_s = time_s[w]
    h.fail_pen_m = h.max_pen_m
    h.fail_shape_a = acc.pen_shape_a[w]
    h.fail_shape_b = acc.pen_shape_b[w]
    h.fail_particle = acc.pen_particle[w]
    h.fail_pair_pen_m = h.max_pair_pen_m
    h.fail_pair_id = h.max_pair_id
    h.fail_speed_mps = h.max_speed_mps
    h.fail_body = h.argmax_body
    h.fail_joint_err_m = h.max_joint_err_m
    h.fail_joint_err_rad = h.max_joint_err_rad
    h.fail_energy_j = h.energy_j
    h.fail_plugin_check = h.plugin_check
    h.fail_plugin_depth_m = h.plugin_depth_m
    record[w] = h


def first_failure_report(h: np.ndarray, plugin_names: tuple[str, ...] = ()) -> FirstFailure:
    """The records' first-failure snapshots (``HealthMonitor.poll``)."""
    return FirstFailure(h["first_bad_step"].astype(np.int64),
                        *(h[f"fail_{name}"].copy() for name in FAIL_FIELDS), plugin_names=plugin_names)


@wp.kernel
def masked_copy(src: wp.array[wp.float64], mask: wp.array[wp.int32], dst: wp.array[wp.float64]):
    w = wp.tid()
    if mask[w] != 0:
        dst[w] = src[w]

