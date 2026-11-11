# SPDX-License-Identifier: Apache-2.0
"""Warp kernels of ``ModalSound`` (engine item 10, sound). One step runs six launches:

1. ``classify`` (one thread per contact): an active contact (normal force > 0) on a sounding body marks its
   (object, partner body) pair active, and fast if |v_n| > v_rest. A fast contact of a pair that was not touching
   counts toward that pair's new contacts. The contact's force enters the object's modal forces (sustained path)
   only if its pair is settled, or new and slow. Never during an impact episode.
2. ``impacts`` (one thread per contact): each new fast contact adds its share of the predicted impact impulse
   J = (1 + e)·m_eff·v_n as modal velocity kicks, weighted by the Hertz pulse spectrum H(ω_k τ). The pair's later
   steps (MuJoCo's soft impact, including its v_n = 0 turning point) add nothing until the pair separates or has
   been slow for ``settle`` steps, so the impact is never counted twice.
3. ``synth`` (one thread per world and mode): the kick, then exact damped-oscillator steps at the audio rate. The
   sustained force is interpolated linearly over the step. Each mode writes its radiated acceleration per sample.
4. ``source`` (one thread per world, object and sample): sums its modes in a fixed order into the object's ring.
5. ``mix`` (one thread per world, microphone and sample): each object's ring, read r/c earlier (optional delay)
   and scaled by 1/r, into the microphone's audio ring.
6. ``advance_pairs`` and ``advance_clock``: the pair states and the sample clock for the next step. A pair state
   is 0 (not touching), 1 (settled: sustained path) or 2 + n (an impact episode, slow for the last n steps).

Sums over contacts use int64 fixed point (exact and order-free; no float atomics). Index conventions:
- object instance g = w·O + o (world w, object o);
- mode slot w·K + m (m indexes all objects' modes, objects back to back);
- partner slot: the partner body's index within its world, or S - 1 for static bodies and global ones.
"""

from __future__ import annotations

import warp as wp

FIX_F = 4294967296.0          # 2^32 counts per N:   2.3e-10 N resolution, ±2e9 N range
FIX_J = 1099511627776.0       # 2^40 counts per N·s: 9.1e-13 N·s resolution, ±8e6 N·s range
PI = 3.14159265358979


@wp.func
def n_sub(frac: wp.float64, spf: wp.float64) -> int:
    """Audio samples that fall in this step (``frac`` = fractional samples carried over, ``spf`` = dt·fs)."""
    return int(wp.floor(frac + spf))


@wp.func
def pulse_weight(x: float) -> float:
    """|F(ω)|/J of a half-sine pulse, x = ω·τ (``hertz.pulse_weight``)."""
    den = 1.0 - (x / PI) * (x / PI)
    if wp.abs(den) < 1.0e-6:
        return PI / 4.0
    return wp.abs(wp.cos(0.5 * x) / den)


@wp.func
def world_point(body_q: wp.array(dtype=wp.transform), b: int, p: wp.vec3) -> wp.vec3:
    if b < 0:
        return p
    return wp.transform_point(body_q[b], p)


@wp.func
def point_vel(body_q: wp.array(dtype=wp.transform), body_qd: wp.array(dtype=wp.spatial_vector),
              body_com: wp.array(dtype=wp.vec3), b: int, p: wp.vec3) -> wp.vec3:
    if b < 0:
        return wp.vec3(0.0)
    com = wp.transform_point(body_q[b], body_com[b])
    qd = body_qd[b]
    return wp.spatial_top(qd) + wp.cross(wp.spatial_bottom(qd), p - com)


@wp.func
def inv_mass_along(body_q: wp.array(dtype=wp.transform), body_com: wp.array(dtype=wp.vec3),
                   inv_m: wp.array(dtype=float), inv_i: wp.array(dtype=wp.mat33), b: int, p: wp.vec3,
                   n: wp.vec3) -> float:
    """1/m + (r×n)ᵀ I⁻¹ (r×n) of body ``b`` at world point ``p`` along ``n`` (0 for static bodies)."""
    if b < 0:
        return 0.0
    im = inv_m[b]
    if im == 0.0:
        return 0.0
    rot = wp.transform_get_rotation(body_q[b])
    rn = wp.quat_rotate_inv(rot, wp.cross(p - wp.transform_point(body_q[b], body_com[b]), n))
    return im + wp.dot(rn, inv_i[b] * rn)


@wp.func
def partner_slot(pb: int, w: int, body_world: wp.array(dtype=wp.int32), world_start: wp.array(dtype=wp.int32),
                 slots: int) -> int:
    if pb < 0:
        return slots - 1
    if body_world[pb] != w:
        return slots - 1
    return wp.min(pb - world_start[w], slots - 1)


@wp.func
def nearest(x: wp.vec3, pts: wp.array(dtype=wp.vec3), p0: int, p1: int) -> int:
    best = int(p0)
    bd = float(1.0e30)
    for p in range(p0, p1):
        d = wp.length_sq(x - pts[p])
        if d < bd:
            bd = d
            best = p
    return best


@wp.struct
class Objects:
    """Per-object tables (shared by every world) and the scene's body tables."""

    sound_of_body: wp.array(dtype=wp.int32)    # [B] g = w·O + o, or -1
    mode_start: wp.array(dtype=wp.int32)       # [O + 1]
    point_start: wp.array(dtype=wp.int32)      # [O + 1]
    gain_start: wp.array(dtype=wp.int32)       # [K]: index of (mode m, point point_start[o]) in gains
    points: wp.array(dtype=wp.vec3)            # body frame
    gains: wp.array(dtype=wp.vec3)             # body frame
    omega: wp.array(dtype=float)               # [K] rad/s
    hertz: wp.array(dtype=wp.vec4)             # [O] (E, ν, R, e)
    partner: wp.vec3                           # default partner (E, ν, R)
    body_world: wp.array(dtype=wp.int32)
    world_start: wp.array(dtype=wp.int32)
    objects: int
    modes: int
    slots: int


@wp.func
def modal_add(ob: Objects, o: int, w: int, b: int, p_body: wp.vec3, f_world: wp.vec3,
              body_q: wp.array(dtype=wp.transform), tau: float, fix: float, acc: wp.array(dtype=wp.int64)):
    """Add gains·F (body frame, nearest sample point) to object o's modes in world w. ``tau`` > 0 weights each
    mode by the Hertz pulse spectrum H(ω τ) (an impact kick); ``tau`` = 0 adds the plain modal force."""
    p = nearest(p_body, ob.points, ob.point_start[o], ob.point_start[o + 1]) - ob.point_start[o]
    fb = wp.quat_rotate_inv(wp.transform_get_rotation(body_q[b]), f_world)
    for m in range(ob.mode_start[o], ob.mode_start[o + 1]):
        x = wp.dot(ob.gains[ob.gain_start[m] + p], fb)
        if tau > 0.0:
            x = x * pulse_weight(ob.omega[m] * tau)
        wp.atomic_add(acc, w * ob.modes + m, wp.int64(wp.round(x * fix)))


@wp.kernel
def classify(count: wp.array(dtype=wp.int32), shape0: wp.array(dtype=wp.int32), shape1: wp.array(dtype=wp.int32),
             point0: wp.array(dtype=wp.vec3), point1: wp.array(dtype=wp.vec3), normal: wp.array(dtype=wp.vec3),
             contact_f: wp.array(dtype=wp.spatial_vector), shape_body: wp.array(dtype=wp.int32),
             body_q: wp.array(dtype=wp.transform), body_qd: wp.array(dtype=wp.spatial_vector),
             body_com: wp.array(dtype=wp.vec3), ob: Objects, v_rest: float, prev: wp.array(dtype=wp.int32),
             active: wp.array(dtype=wp.int32), fast: wp.array(dtype=wp.int32), new_count: wp.array(dtype=wp.int32),
             acc_f: wp.array(dtype=wp.int64)):
    i = wp.tid()
    if i >= count[0] or shape0[i] < 0 or shape1[i] < 0:
        return
    b0 = shape_body[shape0[i]]
    b1 = shape_body[shape1[i]]
    g0 = int(-1)
    g1 = int(-1)
    if b0 >= 0:
        g0 = ob.sound_of_body[b0]
    if b1 >= 0:
        g1 = ob.sound_of_body[b1]
    f0 = wp.spatial_top(contact_f[i])
    n = normal[i]
    if (g0 < 0 and g1 < 0) or -wp.dot(f0, n) <= 0.0:
        return
    pc = 0.5 * (world_point(body_q, b0, point0[i]) + world_point(body_q, b1, point1[i]))
    vn = wp.dot(point_vel(body_q, body_qd, body_com, b0, pc) - point_vel(body_q, body_qd, body_com, b1, pc), n)
    for side in range(2):
        g = g0
        b = b0
        pb = b1
        pt = point0[i]
        f = f0
        if side == 1:
            g = g1
            b = b1
            pb = b0
            pt = point1[i]
            f = -f0
        if g >= 0:
            w = g / ob.objects
            o = g - w * ob.objects
            slot = g * ob.slots + partner_slot(pb, w, ob.body_world, ob.world_start, ob.slots)
            active[slot] = 1
            st = prev[slot]
            if wp.abs(vn) > v_rest:
                fast[slot] = 1
                if st == 0 and vn > 0.0:
                    wp.atomic_add(new_count, slot, 1)
            if st == 1 or (st == 0 and wp.abs(vn) <= v_rest):
                modal_add(ob, o, w, b, pt, f, body_q, 0.0, FIX_F, acc_f)


@wp.kernel
def impacts(count: wp.array(dtype=wp.int32), shape0: wp.array(dtype=wp.int32), shape1: wp.array(dtype=wp.int32),
            point0: wp.array(dtype=wp.vec3), point1: wp.array(dtype=wp.vec3), normal: wp.array(dtype=wp.vec3),
            contact_f: wp.array(dtype=wp.spatial_vector), shape_body: wp.array(dtype=wp.int32),
            body_q: wp.array(dtype=wp.transform), body_qd: wp.array(dtype=wp.spatial_vector),
            body_com: wp.array(dtype=wp.vec3), inv_m: wp.array(dtype=float), inv_i: wp.array(dtype=wp.mat33),
            ob: Objects, v_rest: float, prev: wp.array(dtype=wp.int32), new_count: wp.array(dtype=wp.int32),
            acc_j: wp.array(dtype=wp.int64)):
    i = wp.tid()
    if i >= count[0] or shape0[i] < 0 or shape1[i] < 0:
        return
    b0 = shape_body[shape0[i]]
    b1 = shape_body[shape1[i]]
    g0 = int(-1)
    g1 = int(-1)
    if b0 >= 0:
        g0 = ob.sound_of_body[b0]
    if b1 >= 0:
        g1 = ob.sound_of_body[b1]
    n = normal[i]
    if (g0 < 0 and g1 < 0) or -wp.dot(wp.spatial_top(contact_f[i]), n) <= 0.0:
        return
    pc = 0.5 * (world_point(body_q, b0, point0[i]) + world_point(body_q, b1, point1[i]))
    vn = wp.dot(point_vel(body_q, body_qd, body_com, b0, pc) - point_vel(body_q, body_qd, body_com, b1, pc), n)
    if vn <= v_rest:
        return
    inv_sum = inv_mass_along(body_q, body_com, inv_m, inv_i, b0, pc, n) + \
        inv_mass_along(body_q, body_com, inv_m, inv_i, b1, pc, n)
    if inv_sum <= 0.0:
        return
    for side in range(2):
        g = g0
        b = b0
        pb = b1
        gp = g1
        pt = point0[i]
        sgn = float(-1.0)                      # the impulse on shape0 points along -n
        if side == 1:
            g = g1
            b = b1
            pb = b0
            gp = g0
            pt = point1[i]
            sgn = 1.0
        if g >= 0:
            w = g / ob.objects
            o = g - w * ob.objects
            slot = g * ob.slots + partner_slot(pb, w, ob.body_world, ob.world_start, ob.slots)
            if prev[slot] == 0 and new_count[slot] > 0:
                h = ob.hertz[o]
                pe = ob.partner
                if gp >= 0:
                    hp = ob.hertz[gp - (gp / ob.objects) * ob.objects]
                    pe = wp.vec3(hp[0], hp[1], hp[2])
                e_eff = 1.0 / ((1.0 - h[1] * h[1]) / h[0] + (1.0 - pe[1] * pe[1]) / pe[0])
                r_eff = 1.0 / (1.0 / h[2] + 1.0 / pe[2])
                m_eff = 1.0 / inv_sum
                tau = 2.87 * wp.pow(m_eff * m_eff / (r_eff * e_eff * e_eff * vn), 0.2)
                jn = (1.0 + h[3]) * m_eff * vn / float(new_count[slot])
                modal_add(ob, o, w, b, pt, sgn * jn * n, body_q, tau, FIX_J, acc_j)


@wp.kernel
def synth(acc_f: wp.array(dtype=wp.int64), acc_j: wp.array(dtype=wp.int64), frac: wp.array(dtype=wp.float64),
          spf: wp.float64, dt: float, modes: int, ns: int, inv_mass: wp.array(dtype=float),
          trans: wp.array(dtype=wp.vec4), drive: wp.array(dtype=wp.vec2), damp: wp.array(dtype=wp.vec2),
          rad: wp.array(dtype=float), q: wp.array(dtype=float), v: wp.array(dtype=float),
          f_prev: wp.array(dtype=float), kicks: wp.array(dtype=float), sustained: wp.array(dtype=float),
          out: wp.array(dtype=float)):
    """One thread per (world, mode). ``trans`` = (Φ11, Φ12, Φ21, Φ22) and ``drive`` = (Γ1, Γ2) for one audio
    sample; ``damp`` = (2ζω, ω²)."""
    t = wp.tid()
    w = t / modes
    m = t - w * modes
    f1 = float(acc_f[t]) / FIX_F
    kick = float(acc_j[t]) / FIX_J
    acc_f[t] = wp.int64(0)
    acc_j[t] = wp.int64(0)
    im = inv_mass[m]
    a = trans[m]
    g = drive[m]
    c = damp[m]
    qq = q[t]
    vv = v[t] + kick * im
    f0 = f_prev[t]
    k = n_sub(frac[w], spf)
    for s in range(k):
        u = (f0 + (f1 - f0) * (float(s) + 0.5) / float(k)) * im
        qn = a[0] * qq + a[1] * vv + g[0] * u
        vv = a[2] * qq + a[3] * vv + g[1] * u
        qq = qn
        out[t * ns + s] = (u - c[0] * vv - c[1] * qq) * rad[m]
    q[t] = qq
    v[t] = vv
    f_prev[t] = f1
    kicks[t] = kicks[t] + kick
    sustained[t] = sustained[t] + 0.5 * (f0 + f1) * dt


@wp.kernel
def source(out: wp.array(dtype=float), frac: wp.array(dtype=wp.float64), samples: wp.array(dtype=wp.int64),
           spf: wp.float64, mode_start: wp.array(dtype=wp.int32), objects: int, modes: int, ns: int, ring: int,
           src: wp.array(dtype=float)):
    """One thread per (world, object, sample)."""
    t = wp.tid()
    s = t % ns
    g = t / ns
    w = g / objects
    o = g - w * objects
    if s >= n_sub(frac[w], spf):
        return
    total = float(0.0)
    for m in range(mode_start[o], mode_start[o + 1]):
        total += out[(w * modes + m) * ns + s]
    src[g * ring + int((samples[w] + wp.int64(s)) % wp.int64(ring))] = total


@wp.kernel
def mix(src: wp.array(dtype=float), frac: wp.array(dtype=wp.float64), samples: wp.array(dtype=wp.int64),
        spf: wp.float64, obj_body: wp.array(dtype=wp.int32), body_q: wp.array(dtype=wp.transform),
        body_com: wp.array(dtype=wp.vec3), mics: wp.array(dtype=wp.vec3), objects: int, ns: int, ring: int,
        length: int, delay_per_m: float, max_delay: int, r_min: float, audio: wp.array(dtype=float)):
    """One thread per (world, microphone, sample): p = Σ_o src_o(t - r/c) / r."""
    t = wp.tid()
    s = t % ns
    wm = t / ns
    nm = mics.shape[0]
    w = wm / nm
    mic = wm - w * nm
    if s >= n_sub(frac[w], spf):
        return
    idx = samples[w] + wp.int64(s)
    p = float(0.0)
    for o in range(objects):
        b = obj_body[w * objects + o]
        r = wp.max(wp.length(mics[mic] - wp.transform_point(body_q[b], body_com[b])), r_min)
        d = wp.min(int(wp.round(r * delay_per_m)), max_delay)
        j = (idx - wp.int64(d)) % wp.int64(ring)
        if j < wp.int64(0):
            j = j + wp.int64(ring)
        p += src[(w * objects + o) * ring + int(j)] / r
    audio[wm * length + int(idx % wp.int64(length))] = p


@wp.kernel
def advance_pairs(active: wp.array(dtype=wp.int32), fast: wp.array(dtype=wp.int32), settle: int,
                  state: wp.array(dtype=wp.int32), new_count: wp.array(dtype=wp.int32)):
    i = wp.tid()
    st = state[i]
    nxt = int(0)
    if active[i] != 0:
        if st == 0:
            nxt = 1
            if fast[i] != 0:       # any fast first contact opens an episode (WP13), kicked or not
                nxt = 2
        elif st == 1:
            nxt = 1
        elif fast[i] != 0:
            nxt = 2
        else:
            nxt = st + 1
            if nxt >= 2 + settle:
                nxt = 1
    state[i] = nxt
    active[i] = 0
    fast[i] = 0
    new_count[i] = 0


@wp.kernel
def advance_clock(frac: wp.array(dtype=wp.float64), samples: wp.array(dtype=wp.int64), spf: wp.float64):
    w = wp.tid()
    k = n_sub(frac[w], spf)
    samples[w] = samples[w] + wp.int64(k)
    frac[w] = frac[w] + spf - wp.float64(k)
