# SPDX-License-Identifier: Apache-2.0
"""The PEN rule for MuJoCo's own contacts (rigid bodies and robots; GAP-E1, coordinator rulings 2026-11-10).

One limit per contact: ``limit = min(pen_s1, half the thinner part along the contact normal)`` (1 cm unless a part
is thinner than 2 cm; a shape marked by ``HealthMonitor.relax_thin_part_rule`` counts as thick, and so do both
shapes of a contact inside a pair marked by ``relax_designed_overlap``).
- PEN_WARN (DEGRADED) at once past ``pen_warn`` (1 mm), or past ``limit``.
- PEN_FAIL (INVALID) when a contact of the world has stayed past its ``limit`` for ``CheckPolicy.native_pen_dwell_s``
  (0.1 s of simulated time), or at the fourth check in a row past its hard limit (``hard_up``; rulings
  2026-11-10 06:35Z, 07:08Z: a fast ball's dip into a thin plate is FAST_CONTACT / TUNNEL's). The hard limit,
  ``at_once_limit``: ``native_pen_hard x limit`` (2 cm, or through the thinner part), and for large bodies
  ``LARGE_BODY`` (10 %) of the smaller moving body's smallest extent (``moving_extent``; static shapes set none)
  when that
  is more (a 1 m crate: 10 cm; rulings 2026-11-10 03:13Z and 03:27Z: a landing dip grows with the body and its
  speed, a 1 m cube at 10 kg/m3 dips 19 mm from 0.5 m and 87 mm from 10 m, each for under 50 ms; a real sink fails
  by the dwell within 0.1 s anyway). The ground and static shapes set no extent.
A landing or a bounce compresses a soft contact past the limit for a few steps by design; a resting or held overlap,
or a part pushed through a thin one, is wrong. ``check_mjw_contacts`` raises ``Accum.pen_native_fail`` to the worst
contact's level (1 past its limit, ``NATIVE_HARD`` past its at-once limit); ``classify`` keeps the per-world clock in
``StepHealth.pen_deep_s`` (device-side, in the record: restored with it, graph-capturable).
Contacts soft on purpose (physics ruling D, 2026-11-10 23:41Z; Menagerie ANYmal C's feet, solimp 0.015 1 0.03, rest
17.2 mm deep): shapes marked by ``relax_soft_contacts`` (``scene.relax_contact_depth`` in the API) are judged on their
depth past an allowance (``contact_allowance``: the contact's own solimp width when wider than MuJoCo's 1 mm, or a
stated depth), for PEN_WARN, PEN_FAIL and START_PEN alike; the reported depths stay the true ones.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import warp as wp

from alloy.precision.types import HealthFlags

HARD_TIMES = wp.constant(2.0)       # Newton / cloth contacts: hard limit = this x the limit (MuJoCo's own:
                                    # CheckPolicy.native_pen_hard, the same value, kept equal by a test)
NATIVE_HARD = wp.constant(2)        # contact level past its hard limit (1: past its limit; 0: within)
_WARN = wp.constant(wp.uint32(int(HealthFlags.PEN_WARN)))
_FAIL = wp.constant(wp.uint32(int(HealthFlags.PEN_FAIL)))


LARGE_BODY = wp.constant(0.10)      # at-once limit as a share of the smaller body's extent along the normal


@wp.func
def at_once_limit(hard_limit: wp.float32, thin: wp.float32) -> wp.float32:
    """``hard_limit`` (2 cm, or the thinner part's thickness), or 10 % of the smaller body's extent ``thin`` when
    that is more; ``thin`` inf (ground against a relaxed or static shape) keeps ``hard_limit``."""
    if thin < 1.0e30:
        return wp.max(hard_limit, LARGE_BODY * thin)
    return hard_limit


@wp.func
def past_limit(pen: wp.float32, limit: wp.float32, hard: wp.float32, thin: wp.float32) -> wp.int32:
    """The level a contact ``pen`` deep raises ``Accum.pen_native_fail`` to (an ``atomic_max``, so no count can
    wrap: WP5's review): 0 within ``limit`` (1 cm, or half the thinner part), 1 past it (PEN_FAIL once that has
    lasted the dwell), ``NATIVE_HARD`` past the at-once limit."""
    if pen > at_once_limit(hard * limit, thin):
        return NATIVE_HARD
    if pen > limit:
        return 1
    return 0


HARD_CHECKS = wp.constant(4)        # PEN_FAIL at this consecutive check past the hard limit (ruling 07:08Z, data)


@wp.func
def pen_level(native_on: wp.int32, native_level: wp.int32, native_deep: bool, other_level: wp.int32) -> wp.int32:
    """The world's worst contact this check: 0 within its limit, 1 past it, ``NATIVE_HARD`` past its hard limit;
    MuJoCo's contacts (when their flags are on) and the Newton / cloth contacts (``Accum.pen_fail``) together."""
    level = other_level
    if native_on != 0:
        level = wp.max(level, native_level)
        if native_deep:
            level = wp.max(level, 1)
    return level


@wp.func
def hard_up(hard_n: wp.int32, check_dt: wp.float32) -> bool:
    """True at the ``HARD_CHECKS``-th (4th) consecutive check a contact stays past its hard limit (ruling 2026-11-10
    07:08Z, in checks so it scales with the rigid step; 4 on the data). Measured at 1.25 ms checks, both CPU
    routes: a 2 cm ball landing on a 1 cm plate from 2 / 3-4 m stays past it 1 / 3 checks and bounces back
    (FAST_CONTACT), one tunnelling through it 1 check (TUNNEL); the AGENTS.md arm pushed through a 5 / 20 / 50 mm
    static plate 6 / 4 / 10 checks, wedged in a 10 mm one for good.
    ``check_dt`` inf (no dt given): at the first check."""
    return hard_n >= HARD_CHECKS or (hard_n >= 1 and check_dt > 1.0e30)


@wp.func
def moving_extent(sa: int, sb: int, shape_thick: wp.array[wp.float32], shape_body: wp.array[wp.int32]) -> wp.float32:
    """The smaller smallest extent (``health_setup.shape_thickness``) of the contact's moving shapes; a static shape
    (and the ground) sets none (ruling 2026-11-10 03:13Z): a 1 m crate on a 1 cm shelf gets its own 10 cm, an arm
    link 4 cm across its 4 mm, whichever way it lies."""
    size = wp.float32(wp.inf)
    if sa >= 0 and shape_body[sa] >= 0:
        size = shape_thick[sa]
    if sb >= 0 and shape_body[sb] >= 0:
        size = wp.min(size, shape_thick[sb])
    return size


@wp.func
def pen_fail_level(pen: wp.float32, limit: wp.float32, hard: wp.float32) -> wp.int32:
    """A Newton contact's level (``note_pen``): 0 within ``limit``; past it 1 (the 0.1 s dwell), ``NATIVE_HARD``
    past ``hard x limit`` (rigid pairs: ``HARD_TIMES``; cloth and soft particles: inf, the dwell only, audit COUPLE-6:
    a 4 cm box dropped 3 cm onto a 2 mm sheet pushes particles 2.7 mm in for a few steps), and at once
    (above ``NATIVE_HARD``) for ``hard`` 0: past a cut front (SCR-SC5-3, unchanged)."""
    if pen <= limit:
        return 0
    if hard <= 0.0:
        return NATIVE_HARD + 1
    if pen > hard * limit:
        return NATIVE_HARD
    return 1


@wp.func
def deep_time(deep_s: wp.float32, check_dt: wp.float32, deep: bool) -> wp.float32:
    """The world's clock: time a native contact has stayed past its limit, this check (``check_dt``) included; 0
    when none is. ``check_dt`` inf (a caller that gives ``update`` no ``dt``): PEN_FAIL at the first check."""
    if deep:
        return deep_s + check_dt
    return 0.0


@wp.func
def native_pen_flags(warn: bool, hard: bool, deep_s: wp.float32, dwell_s: wp.float32) -> wp.uint32:
    """``warn``: deepest contact past ``pen_warn``; ``hard``: one past its at-once limit; ``deep_s``: the clock."""
    f = wp.uint32(0)
    if warn or deep_s > 0.0:
        f = f | _WARN
    if deep_s > 0.0 and (deep_s >= dwell_s or hard):
        f = f | _FAIL
    return f


@wp.func
def designed_pair(thin: wp.float32, pair_mark: wp.array[wp.int32], sa: int, sb: int) -> wp.float32:
    """``thin`` (the thinner part's extent) of a contact, or inf when both shapes carry a common bit of
    ``pair_mark``: a pair marked by ``relax_designed_overlap`` is held to the 1 cm / 2 cm limits alone."""
    if sa >= 0 and sb >= 0:
        if (pair_mark[sa] & pair_mark[sb]) != 0:
            return wp.inf
    return thin


Pairs = Sequence[tuple[Sequence[str], str]]


def health_monitor(sim: object) -> Any:
    """The ``HealthMonitor`` of an engine simulation (or the monitor itself); None when health is off."""
    health = getattr(sim, "health", None)
    return next((m for m in (sim, health, getattr(health, "health", None)) if hasattr(m, "_pair_mark")), None)


def relax_designed_overlap(sim: object, pairs: Pairs, *, require_all: bool = True) -> list[list[int]]:
    """Hold the contacts WITHIN each designed pair to the 1 cm / 2 cm limits alone: the thin-part limit no longer
    applies between two shapes of the same entry, and still applies between each of them and every other shape
    (DC5's review: a marked module card pushed through a cage wall must still fail).

    ``pairs``: one entry per designed pair, ``(label suffixes of its shapes, why)``: both sides of the pair (a
    sprung member and its partner; mirrored copies may share an entry), with the overlap in mm in ``why``. Up to 31
    entries per monitor. ``sim``: an engine simulation or its ``HealthMonitor``. Raises ``ValueError`` for a suffix
    that names no shape, so a renamed part cannot silently lose its mark; ``require_all=False`` (a rig whose scenes
    leave parts out) skips an entry with a missing shape and raises only when no entry is whole.
    Example: ``relax_designed_overlap(sim, [(("contact0_top", "module_card"), "0.20 mm drawn preload")])``."""
    monitor = health_monitor(sim)
    if monitor is None:                                   # health off: nothing to relax
        return []
    labels = [str(lb) for lb in getattr(monitor.model, "shape_label", None) or monitor.model.shape_key]
    mark, groups = monitor._pair_mark.numpy(), []
    used = int(np.bitwise_or.reduce(mark)) if len(mark) else 0
    for names, _why in pairs:
        missing = [n for n in names if not any(lb.endswith(n) for lb in labels)]
        if missing and require_all:
            raise ValueError(f"relax_designed_overlap: no shape label ends with {missing}")
        if missing:
            continue
        bit = next(b for b in range(31) if not used >> b & 1)       # StopIteration: more than 31 entries
        used |= 1 << bit
        ids = [i for i, lb in enumerate(labels) if any(lb.endswith(n) for n in names)]
        mark[ids] |= 1 << bit
        groups.append(ids)
    if pairs and not groups:
        raise ValueError(f"relax_designed_overlap: no entry has all its shapes in this scene ({[p[0] for p in pairs]})")
    monitor._pair_mark.assign(mark)
    return groups


SOLIMP_WIDTH = wp.constant(0.001)   # MuJoCo's default solimp width (m); a wider one is a contact made soft on purpose
FROM_SOLIMP = -1.0                  # ``_shape_allow`` entry: the contact's own solimp width (``relax_soft_contacts``)


@wp.func
def shape_allowance(shape_allow: wp.array[wp.float32], s: int, width: wp.float32) -> wp.float32:
    """One shape's allowance in a contact of solimp width ``width``: 0 unmarked, its stated depth, or (marked
    ``FROM_SOLIMP``) ``width`` when wider than MuJoCo's 1 mm default."""
    if s < 0 or s >= shape_allow.shape[0]:
        return 0.0
    a = shape_allow[s]
    if a < 0.0:
        return wp.where(width > SOLIMP_WIDTH, width, 0.0)
    return a


@wp.func
def contact_allowance(shape_allow: wp.array[wp.float32], sa: int, sb: int, width: wp.float32) -> wp.float32:
    """How deep a contact may go before the PEN rule counts it: the larger allowance of its two shapes."""
    return wp.max(shape_allowance(shape_allow, sa, width), shape_allowance(shape_allow, sb, width))


def allowance_host(shape_allow: np.ndarray | None, sa: np.ndarray, sb: np.ndarray, width: np.ndarray) -> np.ndarray:
    """``contact_allowance`` for host arrays of contacts (START_PEN's one-time judge); zeros when none is marked."""
    out = np.zeros(len(sa), np.float64)
    if shape_allow is None or not len(shape_allow) or not len(sa):
        return out
    for s in (sa, sb):
        a = np.where(s >= 0, shape_allow[np.clip(s, 0, len(shape_allow) - 1)], 0.0).astype(np.float64)
        own = np.where(width > float(SOLIMP_WIDTH), width, 0.0)
        out = np.maximum(out, np.where(a < 0.0, own, a))
    return out


def relax_soft_contacts(sim: object, shape_ids: Sequence[int], depth: float | None = None) -> None:
    """Judge the contacts of ``shape_ids`` on their depth past an allowance: ``depth`` (m), or with None each
    contact's own MuJoCo solimp width when wider than the 1 mm default (a foot made soft on purpose: Menagerie ANYmal
    C's solimp 0.015 1 0.03 gives 30 mm; a shape at the default keeps the full rule). PEN_WARN then fires 1 mm past
    the allowance and PEN_FAIL 1 cm past it (0.1 s), and START_PEN alike; ``max_penetration_m`` and
    ``peak_penetration_m`` stay the true depths. MuJoCo's own contacts only (rigid bodies and robots).
    ``sim``: an engine simulation or its ``HealthMonitor``; nothing happens with health off.
    Example: ``relax_soft_contacts(sim.engine, feet)``; in the API, ``scene.relax_contact_depth("anymal")``."""
    monitor = health_monitor(sim)
    if monitor is None:
        return
    allow = monitor._shape_allow.numpy()
    allow[np.asarray(list(shape_ids), dtype=np.int64)] = FROM_SOLIMP if depth is None else float(depth)
    monitor._shape_allow.assign(allow)
    monitor.start.shape_allow = allow
