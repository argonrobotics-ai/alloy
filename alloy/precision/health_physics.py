# SPDX-License-Identifier: Apache-2.0
"""Full-level health kernels: joint and weld violation from ``body_q``, and per-world energy (SPEC §3.7, D20).

Joint error is evaluated from body poses, so it is the real residual for maximal-coordinate solvers (XPBD) and
roundoff for reduced-coordinate ones (MuJoCo/Featherstone write ``body_q`` by FK of ``joint_q``,
newton@ac7f7830:newton/_src/solvers/mujoco/solver_mujoco.py:6260-6324). Welds and connects come from the MuJoCo
equality attributes (newton/_src/solvers/mujoco/equality.py:62-219). Energy is a float64 sum per world in a fixed
order (no float atomics): KE + gravity PE above the world's lowest body/particle height at reset, summed in chunks of
the world's items and then over its chunks in order (``chunk_sums``: bitwise deterministic and batch-invariant).
"""

from __future__ import annotations

import warp as wp

from alloy._vendor import newton
from alloy.precision.chunk_sums import Chunks, chunk_count, chunk_sum
from alloy.precision.health_kernels import Accum, EnergyRef, PARTICLE_ACTIVE, body_xform
from alloy.precision.types import StepHealth

J_PRISMATIC = wp.constant(int(newton.JointType.PRISMATIC))
J_REVOLUTE = wp.constant(int(newton.JointType.REVOLUTE))
J_BALL = wp.constant(int(newton.JointType.BALL))
J_FIXED = wp.constant(int(newton.JointType.FIXED))
EQ_CONNECT = wp.constant(1)
EQ_WELD = wp.constant(2)


@wp.struct
class WorldList:
    """World ``w`` visits ``items[begin[w]:end[w]]`` in order (built on the host, digest.world_items)."""

    begin: wp.array[wp.int32]
    end: wp.array[wp.int32]
    items: wp.array[wp.int32]


@wp.struct
class WeldTable:
    kind: wp.array[wp.int32]          # 0 skip, EQ_CONNECT, EQ_WELD
    body1: wp.array[wp.int32]
    body2: wp.array[wp.int32]
    anchor1: wp.array[wp.vec3]        # connect: point in body1 frame
    anchor2: wp.array[wp.vec3]        # the same point in body2 frame (connect) / weld point in body2 frame
    rel: wp.array[wp.transform]       # weld: target pose of body2 in body1's frame
    lim_m: wp.array[wp.float32]
    lim_rad: wp.array[wp.float32]
    world: wp.array[wp.int32]


@wp.func
def quat_angle(q: wp.quat) -> float:
    """Rotation angle of a unit quaternion, in [0, pi]."""
    v = wp.vec3(q[0], q[1], q[2])
    return 2.0 * wp.atan2(wp.length(v), wp.abs(q[3]))


@wp.func
def vec_angle(a: wp.vec3, b: wp.vec3) -> float:
    """Angle between two vectors via atan2 (acos near 1 turns float32 roundoff into ~5e-4 rad)."""
    return wp.atan2(wp.length(wp.cross(a, b)), wp.dot(a, b))


@wp.func
def note_joint(acc: Accum, w: int, err_m: float, err_rad: float, lim_m: float, lim_rad: float):
    if not (wp.isfinite(err_m) and wp.isfinite(err_rad)):
        return  # non-finite poses are NONFINITE_Q already
    wp.atomic_max(acc.joint_m, w, err_m)
    wp.atomic_max(acc.joint_rad, w, err_rad)
    if err_m > lim_m or err_rad > lim_rad:
        wp.atomic_add(acc.joint_fail, w, 1)


@wp.kernel
def check_joints(joint_type: wp.array[wp.int32], joint_parent: wp.array[wp.int32],
                 joint_child: wp.array[wp.int32], joint_X_p: wp.array[wp.transform],
                 joint_X_c: wp.array[wp.transform], joint_axis: wp.array[wp.vec3],
                 joint_qd_start: wp.array[wp.int32], joint_world: wp.array[wp.int32],
                 body_q: wp.array[wp.transform], lim_m: wp.float32, lim_rad: wp.float32, acc: Accum):
    j = wp.tid()
    w = joint_world[j]
    jt = joint_type[j]
    if w < 0 or not (jt == J_PRISMATIC or jt == J_REVOLUTE or jt == J_BALL or jt == J_FIXED):
        return
    xp = body_xform(body_q, joint_parent[j]) * joint_X_p[j]
    xc = body_q[joint_child[j]] * joint_X_c[j]
    dp = wp.transform_get_translation(xc) - wp.transform_get_translation(xp)
    qp = wp.transform_get_rotation(xp)
    qc = wp.transform_get_rotation(xc)
    rel_angle = quat_angle(wp.quat_inverse(qp) * qc)
    err_m = wp.length(dp)
    err_rad = float(0.0)
    if jt == J_FIXED:
        err_rad = rel_angle
    elif jt == J_REVOLUTE or jt == J_PRISMATIC:
        axis = wp.normalize(joint_axis[joint_qd_start[j]])
        ap = wp.quat_rotate(qp, axis)
        if jt == J_REVOLUTE:
            err_rad = vec_angle(ap, wp.quat_rotate(qc, axis))
        else:
            err_m = wp.length(dp - wp.dot(dp, ap) * ap)
            err_rad = rel_angle
    note_joint(acc, w, err_m, err_rad, lim_m, lim_rad)


@wp.kernel
def check_welds(t: WeldTable, enabled: wp.array[wp.bool], body_q: wp.array[wp.transform], acc: Accum):
    e = wp.tid()
    w = t.world[e]
    if t.kind[e] == 0 or w < 0 or not enabled[e]:
        return
    x1 = body_xform(body_q, t.body1[e])
    x2 = body_xform(body_q, t.body2[e])
    p2 = wp.transform_point(x2, t.anchor2[e])
    if t.kind[e] == EQ_CONNECT:
        note_joint(acc, w, wp.length(wp.transform_point(x1, t.anchor1[e]) - p2), 0.0, t.lim_m[e], t.lim_rad[e])
        return
    target = x1 * t.rel[e]
    err_m = wp.length(p2 - wp.transform_point(target, t.anchor2[e]))
    err_rad = quat_angle(wp.quat_inverse(wp.transform_get_rotation(target)) * wp.transform_get_rotation(x2))
    note_joint(acc, w, err_m, err_rad, t.lim_m[e], t.lim_rad[e])


@wp.struct
class EnergyModel:
    bodies: WorldList
    particles: WorldList
    body_mass: wp.array[wp.float32]
    body_com: wp.array[wp.vec3]
    body_inertia: wp.array[wp.mat33]
    particle_mass: wp.array[wp.float32]
    particle_flags: wp.array[wp.int32]
    gravity: wp.array[wp.vec3]
    body_active: wp.array[wp.int8]      # 0 = retired by crumble (topology, SCR-WP10-2): left out of KE and PE
    # SCR-DC3-4: passive spring energy 0.5 k (q - q_ref)^2. Joint springs: 1-dof joints with dof_passive_stiffness
    springs: WorldList                  # items index spring_q / spring_k / spring_ref
    spring_q: wp.array[wp.int32]        # joint coordinate
    spring_k: wp.array[wp.float32]
    spring_ref: wp.array[wp.float32]
    # fixed tendons with stiffness: L = sum coef * q over tendon_q[adr : adr + num]; deadband [lo, hi]
    tendons: WorldList                  # items index tendon_adr / tendon_num / tendon_k / tendon_lo / tendon_hi
    tendon_adr: wp.array[wp.int32]
    tendon_num: wp.array[wp.int32]
    tendon_k: wp.array[wp.float32]
    tendon_lo: wp.array[wp.float32]
    tendon_hi: wp.array[wp.float32]
    tendon_q: wp.array[wp.int32]
    tendon_coef: wp.array[wp.float32]
    chunks: Chunks                      # the world's bodies-then-particles sequence in chunks (chunk_sums)
    partial: wp.array[wp.float64]       # [chunks] scratch: one chunk's sum (one stream, chunk_sums)


@wp.func
def world_gravity(m: EnergyModel, w: int) -> wp.vec3d:
    if m.gravity.shape[0] > w:
        return wp.vec3d(m.gravity[w])
    return wp.vec3d(m.gravity[0])


@wp.func
def height(g: wp.vec3d, x: wp.vec3d) -> wp.float64:
    """Height along -g (0 when gravity is zero)."""
    gn = wp.length(g)
    if gn == wp.float64(0.0):
        return wp.float64(0.0)
    return -wp.dot(g, x) / gn


@wp.func
def spring_energy(m: EnergyModel, joint_q: wp.array[wp.float32], w: int) -> wp.float64:
    """Σ 0.5 k (q - q_ref)^2 over the world's joint springs plus its fixed-tendon springs (deadband [lo, hi])."""
    e = wp.float64(0.0)
    if joint_q.shape[0] == 0:
        return e
    for k in range(m.springs.begin[w], m.springs.end[w]):
        i = m.springs.items[k]
        dq = wp.float64(joint_q[m.spring_q[i]]) - wp.float64(m.spring_ref[i])
        e += wp.float64(0.5) * wp.float64(m.spring_k[i]) * dq * dq
    for k in range(m.tendons.begin[w], m.tendons.end[w]):
        t = m.tendons.items[k]
        length = wp.float64(0.0)
        for j in range(m.tendon_adr[t], m.tendon_adr[t] + m.tendon_num[t]):
            length += wp.float64(m.tendon_coef[j]) * wp.float64(joint_q[m.tendon_q[j]])
        x = wp.float64(0.0)
        if length < wp.float64(m.tendon_lo[t]):
            x = length - wp.float64(m.tendon_lo[t])
        elif length > wp.float64(m.tendon_hi[t]):
            x = length - wp.float64(m.tendon_hi[t])
        e += wp.float64(0.5) * wp.float64(m.tendon_k[t]) * x * x
    return e


@wp.kernel
def world_energy_chunks(m: EnergyModel, body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector],
                        particle_q: wp.array[wp.vec3], particle_qd: wp.array[wp.vec3], z_ref: wp.array[wp.float64],
                        partial: wp.array[wp.float64]):
    """One chunk of a world's KE + Σ m·|g|·(h - z_ref[w]): its bodies, then its particles, float64, in item order
    (``chunk_sums``)."""
    c = wp.tid()
    w = m.chunks.world[c]
    g = world_gravity(m, w)
    gn = wp.length(g)
    nb = m.bodies.end[w] - m.bodies.begin[w]
    lo = m.chunks.lo[c]
    hi = m.chunks.hi[c]
    e = wp.float64(0.0)
    for p in range(lo, wp.min(hi, nb)):
        b = m.bodies.items[m.bodies.begin[w] + p]
        mass = wp.float64(m.body_mass[b])
        if mass > wp.float64(0.0) and m.body_active[b] != wp.int8(0):
            X = body_q[b]
            v = wp.vec3d(wp.spatial_top(body_qd[b]))
            om = wp.spatial_bottom(body_qd[b])
            om_body = wp.quat_rotate_inv(wp.transform_get_rotation(X), om)
            rot_ke = wp.float64(0.5) * wp.float64(wp.dot(om_body, m.body_inertia[b] * om_body))
            com = wp.vec3d(wp.transform_point(X, m.body_com[b]))
            e += wp.float64(0.5) * mass * wp.dot(v, v) + rot_ke + mass * gn * (height(g, com) - z_ref[w])
    for p in range(wp.max(lo, nb), hi):
        i = m.particles.items[m.particles.begin[w] + p - nb]
        mass = wp.float64(m.particle_mass[i])
        if mass > wp.float64(0.0) and (m.particle_flags[i] & PARTICLE_ACTIVE) != 0:
            v = wp.vec3d(particle_qd[i])
            e += wp.float64(0.5) * mass * wp.dot(v, v) + mass * gn * (height(g, wp.vec3d(particle_q[i])) - z_ref[w])
    partial[c] = e


@wp.kernel
def world_energy_sum(m: EnergyModel, partial: wp.array[wp.float64], joint_q: wp.array[wp.float32],
                     out: wp.array[wp.float64]):
    """E per world (D20): its chunks in order, plus passive spring energy."""
    w = wp.tid()
    out[w] = chunk_sum(m.chunks, partial, w) + spring_energy(m, joint_q, w)


def launch_world_energy(m: EnergyModel, body_q: wp.array, body_qd: wp.array, particle_q: wp.array,
                        particle_qd: wp.array, joint_q: wp.array, z_ref: wp.array, out: wp.array, device) -> None:
    """E = KE + Σ m·|g|·(h - z_ref[w]) + passive spring energy per world into ``out`` [W] (float64, fixed order)."""
    n = chunk_count(m.chunks)
    if n:
        wp.launch(world_energy_chunks, dim=n, inputs=[m, body_q, body_qd, particle_q, particle_qd, z_ref, m.partial],
                  device=device)
    wp.launch(world_energy_sum, dim=out.shape[0], inputs=[m, m.partial, joint_q, out], device=device)


@wp.kernel
def lowest_height(m: EnergyModel, body_q: wp.array[wp.transform], particle_q: wp.array[wp.vec3],
                  out: wp.array[wp.float64]):
    """z_ref per world: the lowest body COM / active particle height (D20); 0 for an empty world."""
    w = wp.tid()
    g = world_gravity(m, w)
    lo = wp.float64(1.0e300)
    for k in range(m.bodies.begin[w], m.bodies.end[w]):
        b = m.bodies.items[k]
        if m.body_mass[b] > 0.0 and m.body_active[b] != wp.int8(0):
            lo = wp.min(lo, height(g, wp.vec3d(wp.transform_point(body_q[b], m.body_com[b]))))
    for k in range(m.particles.begin[w], m.particles.end[w]):
        i = m.particles.items[k]
        if m.particle_mass[i] > 0.0 and (m.particle_flags[i] & PARTICLE_ACTIVE) != 0:
            lo = wp.min(lo, height(g, wp.vec3d(particle_q[i])))
    if lo > wp.float64(1.0e299):
        lo = wp.float64(0.0)
    out[w] = lo


@wp.kernel
def init_energy_ref(e_now: wp.array[wp.float64], scale_floor: wp.array[wp.float64], mask: wp.array[wp.int32],
                    ref: EnergyRef):
    """E0 = E(reset state); E_scale = max(E0 - PE_min, m_tot·g·L_char, 1e-3 J) with PE_min = 0 (D20)."""
    w = wp.tid()
    if mask[w] == 0:
        return
    ref.e0[w] = e_now[w]
    ref.e_max[w] = e_now[w]
    ref.w_cum[w] = wp.float64(0.0)
    ref.e_scale[w] = wp.max(wp.max(e_now[w], scale_floor[w]), wp.float64(1.0e-3))


@wp.kernel
def bump_step(counter: wp.array[wp.int64]):
    """Device step counter: one ``update`` call = one step (graph replays see the right step, unlike a host int)."""
    counter[0] = counter[0] + wp.int64(1)


@wp.kernel
def add_work(w_ext: wp.array[wp.float64], ref: EnergyRef):
    w = wp.tid()
    ref.w_cum[w] = ref.w_cum[w] + w_ext[w]


@wp.kernel
def forget_seen(record: wp.array[StepHealth]):
    """``seen_flags`` back to the flags that are on now and ``peak_pen_m`` back to the depth now
    (``HealthMonitor.forget_seen``: the end of a settle)."""
    w = wp.tid()
    h = record[w]
    h.seen_flags = h.flags
    h.peak_pen_m = 0.0
    if h.max_pen_m > 0.0:                    # NaN (not measured) compares false
        h.peak_pen_m = h.max_pen_m
    record[w] = h
