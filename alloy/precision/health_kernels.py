# SPDX-License-Identifier: Apache-2.0
"""Device kernels of the HealthMonitor (SPEC §3.7, §4.3). Lint-clean: only integer ``atomic_add``/``atomic_or``,
float ``atomic_max`` and per-world serial float64 sums, so every reduction is order-independent (D9 rule D3).

Per step: ``reset`` -> item kernels (bodies, joint coordinates, particles, rigid and soft contacts, overflow
sources) write per-world accumulators -> optional ``joint``/``weld`` (full level) and ``energy`` -> ``classify``
turns the accumulators into ``StepHealth`` flags and severity.
"""

from __future__ import annotations

import warp as wp

from alloy.precision.health_pen import (  # noqa: I001
    HARD_TIMES, NATIVE_HARD, deep_time, designed_pair, hard_up, moving_extent, native_pen_flags, past_limit,
    pen_fail_level, pen_level)
from alloy.precision.health_overflow import OverflowNote, overflow_line, overflow_note  # noqa: F401 (health.py)
from alloy.precision.health_source_shapes import SolidTable, contact_thickness
from alloy.precision.ladder import DEGRADED_MASK, INVALID_MASK
from alloy.precision.types import HealthFlags, Severity, StepHealth

FL_NONFINITE_Q = wp.constant(wp.uint32(int(HealthFlags.NONFINITE_Q)))
FL_NONFINITE_QD = wp.constant(wp.uint32(int(HealthFlags.NONFINITE_QD)))
FL_NONFINITE_PARTICLE = wp.constant(wp.uint32(int(HealthFlags.NONFINITE_PARTICLE)))
FL_CONTACT_OVERFLOW = wp.constant(wp.uint32(int(HealthFlags.CONTACT_OVERFLOW)))
FL_ROW_OVERFLOW = wp.constant(wp.uint32(int(HealthFlags.ROW_OVERFLOW)))
FL_UNCONVERGED = wp.constant(wp.uint32(int(HealthFlags.UNCONVERGED)))
FL_PEN_WARN = wp.constant(wp.uint32(int(HealthFlags.PEN_WARN)))
FL_PEN_FAIL = wp.constant(wp.uint32(int(HealthFlags.PEN_FAIL)))
FL_PAIR_PEN_FAIL = wp.constant(wp.uint32(int(HealthFlags.PAIR_PEN_FAIL)))
FL_PAIR_PEN_WARN = wp.constant(wp.uint32(int(HealthFlags.PAIR_PEN_WARN)))
FL_JOINT_FAIL = wp.constant(wp.uint32(int(HealthFlags.JOINT_FAIL)))
FL_ENERGY_JUMP = wp.constant(wp.uint32(int(HealthFlags.ENERGY_JUMP)))
FL_SPEED_FAIL = wp.constant(wp.uint32(int(HealthFlags.SPEED_FAIL)))
FL_FAR = wp.constant(wp.uint32(int(HealthFlags.FAR_FROM_ORIGIN)))
FL_LATCHED = wp.constant(wp.uint32(int(HealthFlags.LATCHED)))
MASK_INVALID = wp.constant(wp.uint32(INVALID_MASK))
MASK_DEGRADED = wp.constant(wp.uint32(DEGRADED_MASK))
SEV_OK = wp.constant(int(Severity.OK))
SEV_DEGRADED = wp.constant(int(Severity.DEGRADED))
SEV_INVALID = wp.constant(int(Severity.INVALID))
PARTICLE_ACTIVE = wp.constant(1)  # newton.ParticleFlags.ACTIVE (newton/_src/geometry/flags.py:13)


@wp.struct
class Accum:
    """Per-world accumulators of one check (reset every check)."""

    bad_q: wp.array[wp.int32]
    bad_qd: wp.array[wp.int32]
    bad_particle: wp.array[wp.int32]
    overflow: wp.array[wp.uint32]          # flag bits from capacity sources and adapters
    speed: wp.array[wp.float32]
    abs_pos: wp.array[wp.float32]
    arg: wp.array[wp.uint64]               # (speed bits << 32) | body: deterministic argmax
    pen: wp.array[wp.float32]
    pen_fail: wp.array[wp.int32]
    pen_src: wp.array[wp.int32]            # 1 once a contact source covered the world in this check (SCR-SC0-1)
    pen_native: wp.array[wp.float32]       # max penetration from MuJoCo Warp's own contacts (SCR-SC0-1)
    pen_native_fail: wp.array[wp.int32]    # its contacts above their S1 limit
    pen_arg: wp.array[wp.uint64]           # (pen bits << 32) | pen_tag(source, index) of the deepest contact
    pen_shape_a: wp.array[wp.int32]        # that contact's shapes / particle (health_failure resolves them; -1: none)
    pen_shape_b: wp.array[wp.int32]
    pen_particle: wp.array[wp.int32]
    cut_front: wp.array[wp.float32]        # deepest exempt cut-front soft contact (SCR-SC5-3)
    cut_front_hit: wp.array[wp.int32]      # exempt cut-front contacts past the normal PEN limit
    plug_check: wp.array[wp.int32]         # first plug-in depth check over its limit (health_plugin; -1: none)
    plug_depth: wp.array[wp.float32]       # its depth, else the deepest plug-in depth
    joint_m: wp.array[wp.float32]
    joint_rad: wp.array[wp.float32]
    joint_fail: wp.array[wp.int32]
    iters: wp.array[wp.int32]
    residual: wp.array[wp.float32]         # NaN = no solver reports one
    energy: wp.array[wp.float64]
    stretch_fail: wp.array[wp.int32]       # a cloth past cloth_stretch_fail, or an ``invalid`` source: INVALID


@wp.struct
class Limits:
    pen_warn: wp.float32
    pen_fail: wp.float32
    pair_warn: wp.float32
    pair_fail: wp.float32
    speed_fail: wp.float32
    energy_jump: wp.float64
    energy_s1: wp.float64
    r_warn: wp.float32
    r_fail: wp.float32
    far_invalid: wp.int32
    check_energy: wp.int32
    static_flags: wp.uint32
    native_pen_flags: wp.int32             # MuJoCo Warp contacts raise PEN_WARN / PEN_FAIL (CheckPolicy)
    pen_dwell_s: wp.float32                # ... PEN_FAIL once past the limit for this long (GAP-E1)


@wp.struct
class EnergyRef:
    """Per-world energy bookkeeping (D20): reference height, E0, running max of E - W, W_ext, E_scale."""

    z_ref: wp.array[wp.float64]
    e0: wp.array[wp.float64]
    e_max: wp.array[wp.float64]
    w_cum: wp.array[wp.float64]
    e_scale: wp.array[wp.float64]


@wp.kernel
def reset_accum(acc: Accum):
    w = wp.tid()
    acc.bad_q[w] = 0
    acc.bad_qd[w] = 0
    acc.bad_particle[w] = 0
    acc.overflow[w] = wp.uint32(0)
    acc.speed[w] = 0.0
    acc.abs_pos[w] = 0.0
    acc.arg[w] = wp.uint64(0)
    acc.pen[w] = 0.0
    acc.pen_fail[w] = 0
    acc.pen_src[w] = 0
    acc.pen_native[w] = 0.0
    acc.pen_native_fail[w] = 0
    acc.pen_arg[w] = wp.uint64(0)
    acc.pen_shape_a[w] = -1
    acc.pen_shape_b[w] = -1
    acc.pen_particle[w] = -1
    acc.cut_front[w] = 0.0
    acc.cut_front_hit[w] = 0
    acc.stretch_fail[w] = 0
    acc.plug_check[w] = -1
    acc.plug_depth[w] = 0.0
    acc.joint_m[w] = 0.0
    acc.joint_rad[w] = 0.0
    acc.joint_fail[w] = 0
    acc.iters[w] = 0
    acc.residual[w] = wp.nan
    acc.energy[w] = wp.float64(0.0)


@wp.func
def note_motion(acc: Accum, w: int, item: int, p: wp.vec3, v: wp.vec3):
    """Speed and |p| of one finite item into world ``w``."""
    s = wp.length(v)
    wp.atomic_max(acc.speed, w, s)
    wp.atomic_max(acc.arg, w, (wp.uint64(wp.cast(s, wp.uint32)) << wp.uint64(32)) | wp.uint64(item))
    wp.atomic_max(acc.abs_pos, w, wp.length(p))


@wp.kernel
def check_bodies(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector],
                 body_world: wp.array[wp.int32], acc: Accum):
    b = wp.tid()
    w = body_world[b]
    if w < 0:
        return
    X = body_q[b]
    p = wp.transform_get_translation(X)
    ok_q = wp.isfinite(p) and wp.isfinite(wp.transform_get_rotation(X))
    qd = body_qd[b]
    v = wp.spatial_top(qd)  # COM linear velocity, world frame (newton/_src/sim/state.py:135)
    ok_qd = wp.isfinite(v) and wp.isfinite(wp.spatial_bottom(qd))
    if not ok_q:
        wp.atomic_add(acc.bad_q, w, 1)
    if not ok_qd:
        wp.atomic_add(acc.bad_qd, w, 1)
    if ok_q and ok_qd:
        note_motion(acc, w, b, p, v)


@wp.kernel
def check_coords(values: wp.array[wp.float32], coord_world: wp.array[wp.int32], bad: wp.array[wp.int32]):
    i = wp.tid()
    w = coord_world[i]
    if w >= 0 and not wp.isfinite(values[i]):
        wp.atomic_add(bad, w, 1)


@wp.kernel
def check_particles(q: wp.array[wp.vec3], qd: wp.array[wp.vec3], flags: wp.array[wp.int32],
                    particle_world: wp.array[wp.int32], acc: Accum):
    i = wp.tid()
    w = particle_world[i]
    if w < 0 or (flags[i] & PARTICLE_ACTIVE) == 0:
        return
    if not (wp.isfinite(q[i]) and wp.isfinite(qd[i])):
        wp.atomic_add(acc.bad_particle, w, 1)
        return
    wp.atomic_max(acc.speed, w, wp.length(qd[i]))
    wp.atomic_max(acc.abs_pos, w, wp.length(q[i]))


@wp.func
def body_xform(body_q: wp.array[wp.transform], b: int) -> wp.transform:
    if b < 0:
        return wp.transform_identity()
    return body_q[b]


PEN_RIGID, PEN_SOFT, PEN_NATIVE, PEN_PARTICLE = 1, 2, 3, 4  # pen_tag sources


@wp.func
def pen_tag(source: int, index: int) -> wp.uint64:
    return (wp.uint64(source) << wp.uint64(29)) | wp.uint64(index)


@wp.func
def note_deepest(acc: Accum, w: int, pen: float, tag: wp.uint64):
    """Deterministic argmax: the deepest contact (ties: the larger tag) of world ``w`` this check."""
    wp.atomic_max(acc.pen_arg, w, (wp.uint64(wp.cast(pen, wp.uint32)) << wp.uint64(32)) | tag)


@wp.func
def note_pen(acc: Accum, w: int, pen: float, limit: float, tag: wp.uint64, hard: float):
    if pen > 0.0:
        wp.atomic_max(acc.pen, w, pen)
        note_deepest(acc, w, pen, tag)
        wp.atomic_max(acc.pen_fail, w, pen_fail_level(pen, limit, hard))     # health_pen's rule


@wp.kernel
def check_rigid_contacts(count: wp.array[wp.int32], shape0: wp.array[wp.int32], shape1: wp.array[wp.int32],
                         point0: wp.array[wp.vec3], point1: wp.array[wp.vec3], normal: wp.array[wp.vec3],
                         margin0: wp.array[wp.float32], margin1: wp.array[wp.float32],
                         shape_body: wp.array[wp.int32], shape_world: wp.array[wp.int32],
                         shape_thick: wp.array[wp.float32], body_q: wp.array[wp.transform], pen_fail: wp.float32,
                         solids: SolidTable, shape_type: wp.array[wp.int32], shape_scale: wp.array[wp.vec3],
                         shape_transform: wp.array[wp.transform], acc: Accum):
    k = wp.tid()
    if k >= count[0]:
        return
    s0 = shape0[k]
    s1 = shape1[k]
    if s0 < 0 or s1 < 0:
        return
    w = shape_world[s0]
    if w < 0:
        w = shape_world[s1]
    if w < 0:
        return
    # d = n·(X_b p1 - X_a p0) - (margin0 + margin1) (newton/_src/geometry/differentiable_contacts.py:84-89)
    pa = wp.transform_point(body_xform(body_q, shape_body[s0]), point0[k])
    pb = wp.transform_point(body_xform(body_q, shape_body[s1]), point1[k])
    d = wp.dot(normal[k], pb - pa) - (margin0[k] + margin1[k])
    n = normal[k]  # thin-part rule: each body's solid along the normal (health_source_shapes)
    t0 = contact_thickness(solids, shape_type, shape_scale, shape_transform, shape_body, shape_thick, body_q, s0, n)
    t1 = contact_thickness(solids, shape_type, shape_scale, shape_transform, shape_body, shape_thick, body_q, s1, n)
    limit = wp.min(pen_fail, 0.5 * wp.min(t0, t1))
    note_pen(acc, w, -d, limit, pen_tag(PEN_RIGID, k), HARD_TIMES)


@wp.kernel
def check_soft_contacts(count: wp.array[wp.int32], particle: wp.array[wp.int32], shape: wp.array[wp.int32],
                        body_pos: wp.array[wp.vec3], normal: wp.array[wp.vec3], particle_q: wp.array[wp.vec3],
                        particle_radius: wp.array[wp.float32], particle_world: wp.array[wp.int32],
                        shape_body: wp.array[wp.int32], shape_thick: wp.array[wp.float32],
                        cut_blade: wp.array[wp.int32], blade_half: wp.array[wp.float32],
                        cut_now: wp.array[wp.float32], cut_floor: wp.array[wp.float32],
                        body_q: wp.array[wp.transform], pen_fail: wp.float32, acc: Accum):
    """Soft contacts: penetration past half the thinner of particle and shape. SCR-SC5-3 (re-ruled 19:05Z): a contact
    of particle i with a shape of body ``cut_blade[i]`` (the Blade cutting a phase-1 bond that i twins) is outside that
    rule: it is held to ``2 * blade_half[body] + radius`` (a twin may start anywhere between the faces), reported as
    cut-front depth, and its deepest value this check goes to ``cut_now[i]`` for the growth rule, with the limit a
    normal contact would have in ``cut_floor[i]`` (the growth rule's floor, coordinator ruling 2026-11-04 11:30Z)."""
    k = wp.tid()
    if k >= count[0]:
        return
    i = particle[k]
    s = shape[k]
    if i < 0 or s < 0:
        return
    w = particle_world[i]
    if w < 0:
        return
    ps = wp.transform_point(body_xform(body_q, shape_body[s]), body_pos[k])
    r = particle_radius[i]
    d = wp.dot(normal[k], particle_q[i] - ps) - r
    limit = wp.min(pen_fail, 0.5 * wp.min(2.0 * r, shape_thick[s]))
    if cut_blade.shape[0] > i:
        blade = cut_blade[i]
        if blade >= 0 and shape_body[s] == blade:
            front = 2.0 * blade_half[blade] + r
            wp.atomic_max(cut_now, i, -d)
            wp.atomic_max(cut_floor, i, limit)
            if -d > front:
                note_pen(acc, w, -d, front, pen_tag(PEN_SOFT, k), 0.0)  # past the cut front: PEN_FAIL at once
                return
            wp.atomic_max(acc.cut_front, w, -d)
            if -d > limit:
                wp.atomic_add(acc.cut_front_hit, w, 1)
            return
    note_pen(acc, w, -d, limit, pen_tag(PEN_SOFT, k), wp.inf)       # cloth / soft: the dwell only


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
                       frame: wp.array[wp.mat33],
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
    ta = contact_thickness(solids, shape_type, shape_scale, shape_transform, shape_body, shape_thick, body_q, sa, n)
    tb = contact_thickness(solids, shape_type, shape_scale, shape_transform, shape_body, shape_thick, body_q, sb, n)
    thin = wp.min(ta, tb)
    pen = -dist[k]
    if pen > 0.0:
        wp.atomic_max(acc.pen_native, w, pen)
        note_deepest(acc, w, pen, pen_tag(PEN_NATIVE, k))
        thin = designed_pair(thin, pair_mark, sa, sb)                # GAP-E1: health_pen's one rule
        size = moving_extent(sa, sb, shape_thick, shape_body)
        wp.atomic_max(acc.pen_native_fail, w, past_limit(pen, wp.min(pen_fail, 0.5 * thin), hard, size))


@wp.kernel
def mark_pen_measured(world_count: wp.int32, acc: Accum):
    """Every world of the model is covered by the contact source just read."""
    w = wp.tid()
    if w < world_count:
        acc.pen_src[w] = 1


@wp.kernel
def merge_sample(iters: wp.array[wp.int32], residual: wp.array[wp.float32], acc: Accum):
    """One solver's health sample (iterations, residual normalised by its tolerance) into the world record."""
    w = wp.tid()
    acc.iters[w] = wp.max(acc.iters[w], iters[w])
    r = residual[w]
    if wp.isnan(r):
        return
    if wp.isnan(acc.residual[w]) or r > acc.residual[w]:
        acc.residual[w] = r


@wp.func
def energy_flags(w: int, e_now: wp.float64, lim: Limits, ref: EnergyRef) -> wp.vec2i:
    """(per-step jump, cumulative S1) from E - W_ext against its running maximum (D20). Updates e_max."""
    net = e_now - ref.w_cum[w]
    scale = ref.e_scale[w]
    jump = 0
    s1 = 0
    if net - ref.e_max[w] > lim.energy_jump * scale:
        jump = 1
    if net - ref.e0[w] > lim.energy_s1 * scale:
        s1 = 1
    if net > ref.e_max[w]:
        ref.e_max[w] = net
    return wp.vec2i(jump, s1)


@wp.func
def motion_flags(w: int, acc: Accum, lim: Limits) -> wp.uint32:
    f = wp.uint32(0)
    if acc.bad_q[w] > 0:
        f = f | FL_NONFINITE_Q
    if acc.bad_qd[w] > 0:
        f = f | FL_NONFINITE_QD
    if acc.bad_particle[w] > 0:
        f = f | FL_NONFINITE_PARTICLE
    if acc.speed[w] > lim.speed_fail:
        f = f | FL_SPEED_FAIL
    if acc.abs_pos[w] > lim.r_warn:
        f = f | FL_FAR
    if acc.pen[w] > lim.pen_warn:
        f = f | FL_PEN_WARN
    if acc.pen_fail[w] > NATIVE_HARD or acc.pen[w] > lim.pen_fail or acc.plug_check[w] >= 0:
        f = f | FL_PEN_FAIL
    if acc.joint_fail[w] > 0:
        f = f | FL_JOINT_FAIL
    if acc.residual[w] > 1.0:
        f = f | FL_UNCONVERGED
    return f


@wp.func
def pair_flags(pair_pen: wp.float32, lim: Limits) -> wp.uint32:
    f = wp.uint32(0)
    if pair_pen > lim.pair_warn:
        f = f | FL_PAIR_PEN_WARN
    if pair_pen > lim.pair_fail:
        f = f | FL_PAIR_PEN_FAIL
    return f


@wp.func
def neutral_record() -> StepHealth:
    """The inbox a GapMonitor writes into during a step: no flags, no pair penetration, gap +inf."""
    h = StepHealth()
    h.flags = wp.uint32(0)
    h.severity = SEV_OK
    h.first_bad_step = wp.int64(-1)
    h.residual = wp.nan
    h.max_pair_pen_m = 0.0
    h.max_pair_id = -1
    h.min_pair_gap_m = wp.inf
    return h


@wp.kernel
def classify(step_counter: wp.array[wp.int64], acc: Accum, lim: Limits, ref: EnergyRef,
             latched: wp.array[wp.int32], inbox: wp.array[StepHealth], record: wp.array[StepHealth],
             note: OverflowNote, hold_joints: wp.int32, check_dt: wp.float32):
    """inbox (GapMonitor bits and pair values of this step, D11) + accumulators -> record; inbox reset; note.

    ``hold_joints`` != 0 on a check that did not measure joints and welds (they are measured every ``full_every``
    checks): the record keeps the last measured JOINT_FAIL bit and joint errors, so a read between two measurements
    sees the verdict (audit ERR-11: it was erased by the next check and a period-end read never saw it)."""
    w = wp.tid()
    step = step_counter[0]
    got = inbox[w]
    h = record[w]
    flags = lim.static_flags | acc.overflow[w] | motion_flags(w, acc, lim) | got.flags
    lvl = pen_level(lim.native_pen_flags, acc.pen_native_fail[w], acc.pen_native[w] > lim.pen_fail, acc.pen_fail[w])
    h.pen_deep_s = deep_time(h.pen_deep_s, check_dt, lvl > 0)
    h.pen_hard_n = wp.where(lvl >= NATIVE_HARD, h.pen_hard_n + 1, 0)
    flags = flags | native_pen_flags(lim.native_pen_flags != 0 and acc.pen_native[w] > lim.pen_warn,
                                     hard_up(h.pen_hard_n, check_dt), h.pen_deep_s, lim.pen_dwell_s)
    flags = flags | pair_flags(got.max_pair_pen_m, lim)
    if hold_joints != 0:
        flags = flags | (h.flags & FL_JOINT_FAIL)
    e_s1 = 0
    if lim.check_energy != 0:
        ej = energy_flags(w, acc.energy[w], lim, ref)
        if ej[0] != 0 or ej[1] != 0:
            flags = flags | FL_ENERGY_JUMP
        e_s1 = ej[1]
    far_fail = lim.far_invalid != 0 and acc.abs_pos[w] > lim.r_fail
    sev = SEV_OK
    if (flags & MASK_DEGRADED) != wp.uint32(0):
        sev = SEV_DEGRADED
    blown = acc.stretch_fail[w] != 0              # a cloth triangle past cloth_stretch_fail (SCR-WP5-17)
    if (flags & MASK_INVALID) != wp.uint32(0) or e_s1 != 0 or far_fail or latched[w] != 0 or blown:
        sev = SEV_INVALID
    if latched[w] != 0:
        flags = flags | FL_LATCHED | h.flags  # sticky: keep the bits that latched the world
    if h.severity == SEV_INVALID:
        # INVALID is sticky in the record whether or not the caller latches (SCR-SC0-1: a caller that polls once at
        # the end must not read OK after a mid-run overflow); a rollback or reset restores the record with the state
        sev = SEV_INVALID
        flags = flags | h.flags
    if sev == SEV_INVALID and h.first_bad_step < wp.int64(0):
        h.first_bad_step = step
    h.flags = flags
    h.seen_flags = h.seen_flags | flags   # a flag of one check stays readable after the next (ruling 11-10)
    h.severity = sev
    h.solver_iters = acc.iters[w]
    h.residual = acc.residual[w]
    h.pen_measured = acc.pen_src[w]
    h.max_pen_m = wp.max(acc.pen[w], acc.pen_native[w])
    h.peak_pen_m = wp.max(h.peak_pen_m, h.max_pen_m)     # largest depth since the record's reset (GAP-E1)
    h.cut_front_pen_m = wp.max(h.cut_front_pen_m, acc.cut_front[w])
    if acc.cut_front_hit[w] > 0:
        h.cut_front_steps = h.cut_front_steps + 1
    if acc.pen_src[w] == 0:
        h.max_pen_m = wp.nan  # not measured: never 0 (SCR-SC0-1)
    h.max_pair_pen_m = got.max_pair_pen_m
    h.max_pair_id = got.max_pair_id
    h.plugin_check = acc.plug_check[w]
    h.plugin_depth_m = acc.plug_depth[w]
    h.min_pair_gap_m = got.min_pair_gap_m
    if hold_joints == 0:
        h.max_joint_err_m = acc.joint_m[w]
        h.max_joint_err_rad = acc.joint_rad[w]
    h.max_speed_mps = acc.speed[w]
    h.max_abs_pos_m = acc.abs_pos[w]
    h.energy_j = acc.energy[w]
    h.argmax_body = wp.int32(acc.arg[w] & wp.uint64(0xFFFFFFFF))
    record[w] = h
    inbox[w] = neutral_record()
    if (flags & FL_CONTACT_OVERFLOW) != wp.uint32(0):     # nested: Warp evaluates both sides of an `and`
        if wp.atomic_max(note.printed, 0, 1) == 0:
            overflow_line(w, note)


@wp.kernel
def latch_masks(record: wp.array[StepHealth], latched: wp.array[wp.int32], bad: wp.array[wp.int32]):
    """INVALID worlds latch: they are restored from the caller's last-good snapshot every step (frozen)."""
    w = wp.tid()
    h = record[w]
    if h.severity == SEV_INVALID:
        latched[w] = 1
        h.flags = h.flags | FL_LATCHED
        record[w] = h
    bad[w] = latched[w]
