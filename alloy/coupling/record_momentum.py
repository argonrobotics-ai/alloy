# SPDX-License-Identifier: Apache-2.0
"""Momentum-consistent proxy feedback from a VBD destination (gaps-now #10a slice A; known gap coupled-record-damping).

**What was wrong.** VBD feeds a MuJoCo body back the force of its particle records evaluated at VBD's last iterate
(``coupling_harvest_proxy_wrenches``). VBD is not converged at its iteration count: the top layer of a pad meets the
body and balances it against elastic forces that have not yet moved the layers below, so the particles take far less
momentum than the evaluated force says. Mid-air on main (CPU, 2026-11-08, [internal note] notes/wp10-runs/recmom):
- C3: a 0.515 kg platen at 0.1 m/s onto a free 16 g pad. At first contact the pad took 0.030 mN·s in one exchange
  and the platen was fed 0.587 (20x), then 0.41 against 1.87; the pair gained +2.3 mN·s, 4.5 % of the platen's m V0.
- A 2 kg ball at 0.31 m/s onto a free 11 x 11 cloth: +27.5 mN·s in the first contact exchange, 4.4 % of m V0.

**What this does.** After VBD's own harvest, per world:
- P = the momentum the world's particles took over the coupled step, less gravity and their external forces
  (``particle_f``): what every contact on them gave them. Self-contact and elastic forces are internal.
- E = the same records' force at the feedback state, summed over the contributors the particles touch: each proxy
  body, and the static or kinematic shapes as one contributor.
- The mismatch P / dt - E goes to the contributors in proportion to |E_c|, so a lone body gets back exactly what the
  particles took. When none holds a force at the feedback state (a contact made and broken inside one exchange: the
  ball below on the Warp route, 0.5 ms exchanges), it goes by their share of the world's records. A proxy body's share
  is applied at its records' force-weighted centroid. The static share has no dynamics to feed.

A world is left as VBD harvests it (open) when something outside the records moves its particles: a pinned, inactive
or proxy particle, or a record against a dynamic body VBD itself integrates. At rest P / dt equals E (the contacts
carry the weights), so resting loads keep today's feedback.

Capture-safe: fixed-size launches and zero fills only; the sums are fixed-point integers (deterministic on GPU); the step-start velocities are copied when the coupler
distributes the input state (not on iteration restarts, which restore the same input).
"""

from __future__ import annotations

import logging
from typing import Any

import warp as wp

from alloy._vendor import newton
from alloy._vendor.newton._src.geometry.soft_contacts_mesh import (_make_soft_contact_filter_data,
                                                                   _soft_contact_force_enabled,
                                                                   _SoftContactFilterData)
from alloy._vendor.newton._src.solvers.vbd.rigid_vbd_kernels import _eval_soft_ef_contact

log = logging.getLogger(__name__)

ENABLED = True                    # A/B switch for probes and the law's main-behaviour check; read at install
_P_ACTIVE = wp.constant(int(newton.ParticleFlags.ACTIVE))
_P_PROXY = wp.constant(int(newton.ParticleFlags.PROXY))
_B_PROXY = wp.constant(int(newton.BodyFlags.PROXY))
_B_KINEMATIC = wp.constant(int(newton.BodyFlags.KINEMATIC))
_PARTICLE_Q = int(newton.StateFlags.PARTICLE_Q)


@wp.func
def _world(w: int, worlds: int) -> int:
    if w < 0 or w >= worlds:
        return -1
    return w


# Fixed-point sums (SPEC D9 rule D3: no float atomics): integer atomics are order-independent, so GPU runs stay
# bit-for-bit. Units per quantum: momentum 1e-15 N·s, forces and |f|-weighted points 1 nN (N·m), points 1 nm; int64
# holds 9e3 N·s, 9e9 N and 9e9 m.
Q_P = wp.constant(wp.float64(1.0e15))
Q_F = wp.constant(wp.float64(1.0e9))
Q_X = wp.constant(wp.float64(1.0e9))


@wp.struct
class _WorldSums:
    p: wp.array2d(dtype=wp.int64)     # the particles' momentum change less gravity and particle_f [Q_P]
    e: wp.array2d(dtype=wp.int64)     # every contributor's record force on them at the feedback state [Q_F]
    e_static: wp.array2d(dtype=wp.int64)   # ... of static and kinematic shapes
    w_sum: wp.array(dtype=wp.int64)   # sum over proxy bodies of |E_b| [Q_F]
    n: wp.array(dtype=wp.int32)       # records of every contributor
    open: wp.array(dtype=wp.int32)    # 1: something outside the records moves the particles (no correction)


@wp.struct
class _BodySums:
    e: wp.array2d(dtype=wp.int64)     # E_b: its records' force on the particles [Q_F]
    a: wp.array(dtype=wp.int64)       # sum of |f| over its records [Q_F]
    c: wp.array2d(dtype=wp.int64)     # sum of |f| x contact point [Q_F]
    n: wp.array(dtype=wp.int32)       # its records
    k: wp.array2d(dtype=wp.int64)     # sum of contact points [Q_X]


@wp.func
def _q(x: float, scale: wp.float64) -> wp.int64:
    return wp.int64(wp.round(wp.float64(x) * scale))


@wp.func
def _add3(sums: wp.array2d(dtype=wp.int64), i: int, v: wp.vec3, scale: wp.float64):
    for k in range(3):
        wp.atomic_add(sums, i, k, _q(v[k], scale))


@wp.func
def _get3(sums: wp.array2d(dtype=wp.int64), i: int, scale: wp.float64) -> wp.vec3:
    return wp.vec3(float(wp.float64(sums[i, 0]) / scale), float(wp.float64(sums[i, 1]) / scale),
                   float(wp.float64(sums[i, 2]) / scale))


@wp.func
def _get(v: wp.int64, scale: wp.float64) -> float:
    return float(wp.float64(v) / scale)


@wp.kernel(enable_backward=False)
def _particle_momentum(qd0: wp.array[wp.vec3], qd1: wp.array[wp.vec3], f_ext: wp.array[wp.vec3],
                       mass: wp.array[float], inv_mass: wp.array[float], flags: wp.array[wp.int32],
                       pworld: wp.array[wp.int32], gravity: wp.array[wp.vec3], dt: float, worlds: int,
                       ws: _WorldSums):
    """P per world: m (v1 - v0) - (m g + f_ext) dt over dynamic particles; anything else opens the world."""
    i = wp.tid()
    w = _world(pworld[i], worlds)
    if w < 0:
        return
    f = flags[i]
    if (f & _P_ACTIVE) == 0 or (f & _P_PROXY) != 0 or inv_mass[i] == 0.0:
        ws.open[w] = 1
        return
    m = mass[i]
    ext = wp.vec3(0.0)
    if f_ext.shape[0] > i:
        ext = f_ext[i]
    _add3(ws.p, w, m * (qd1[i] - qd0[i]) - (m * gravity[w] + ext) * dt, Q_P)


@wp.struct
class _Records:
    count: wp.array[int]
    indices: wp.array[wp.vec3i]
    bary: wp.array[wp.vec3]
    shape: wp.array[int]
    body_pos: wp.array[wp.vec3]
    body_vel: wp.array[wp.vec3]
    normal: wp.array[wp.vec3]
    ke: wp.array[float]
    kd: wp.array[float]
    mu: wp.array[float]


@wp.struct
class _Bodies:
    q: wp.array[wp.transform]
    q_prev: wp.array[wp.transform]
    qd: wp.array[wp.spatial_vector]
    com: wp.array[wp.vec3]
    flags: wp.array[wp.int32]
    inv_mass: wp.array[float]
    world: wp.array[wp.int32]
    to_proxy: wp.array[int]


@wp.func
def _is_proxy(b: int, bodies: _Bodies) -> bool:
    if b >= bodies.to_proxy.shape[0] or bodies.to_proxy[b] < 0:
        return False
    return (bodies.flags[b] & _B_PROXY) != 0


@wp.func
def _is_fixed(b: int, bodies: _Bodies) -> bool:
    return (bodies.flags[b] & _B_KINEMATIC) != 0 or bodies.inv_mass[b] == 0.0


@wp.kernel(enable_backward=False)
def _record_forces(dt: float, rec: _Records, bodies: _Bodies, q: wp.array[wp.vec3], q_prev: wp.array[wp.vec3],
                   radius: wp.array[float], pworld: wp.array[wp.int32], shape_body: wp.array[wp.int32],
                   margin: wp.array[float], friction_epsilon: float, log_barrier: bool, worlds: int,
                   filt: _SoftContactFilterData, ws: _WorldSums, bs: _BodySums):
    """E per world and per proxy body: each record's force on its particles at the feedback state (VBD's law)."""
    c = wp.tid()
    if c >= rec.count[0] or not _soft_contact_force_enabled(c, filt):
        return
    corners = rec.indices[c]
    s = rec.shape[c]
    if corners[0] < 0 or corners[0] >= q.shape[0] or s < 0:
        return
    w = _world(pworld[corners[0]], worlds)
    if w < 0:
        return
    b = shape_body[s]
    if b >= 0 and not _is_proxy(b, bodies) and not _is_fixed(b, bodies):
        ws.open[w] = 1                  # a body VBD integrates: its momentum stays inside the destination
        return
    f, _h, cp = _eval_soft_ef_contact(c, corners, rec.bary[c], q, q_prev, radius, rec.ke[c], rec.kd[c], rec.mu[c],
                                      friction_epsilon, shape_body, bodies.q, bodies.q_prev, bodies.qd, bodies.com,
                                      rec.shape, rec.body_pos, rec.body_vel, rec.normal, margin, dt, log_barrier)
    _add3(ws.e, w, f, Q_F)
    wp.atomic_add(ws.n, w, 1)
    if b >= 0 and _is_proxy(b, bodies):
        _add3(bs.e, b, f, Q_F)
        wp.atomic_add(bs.a, b, _q(wp.length(f), Q_F))
        _add3(bs.c, b, wp.length(f) * cp, Q_F)
        wp.atomic_add(bs.n, b, 1)
        _add3(bs.k, b, cp, Q_X)
    else:
        _add3(ws.e_static, w, f, Q_F)


@wp.kernel(enable_backward=False)
def _world_weights(bodies: _Bodies, worlds: int, bs: _BodySums, ws: _WorldSums):
    b = wp.tid()
    if not _is_proxy(b, bodies):
        return
    w = _world(bodies.world[b], worlds)
    if w >= 0:
        wp.atomic_add(ws.w_sum, w, _q(wp.length(_get3(bs.e, b, Q_F)), Q_F))


@wp.func
def _share(b: int, w: int, ws: _WorldSums, bs: _BodySums) -> float:
    """|E_b| / sum |E_c|; when no contributor holds a force at the feedback state (a contact made and broken within
    the step), the body's share of the world's records."""
    total = _get(ws.w_sum[w], Q_F) + wp.length(_get3(ws.e_static, w, Q_F))
    if total > 0.0:
        return wp.length(_get3(bs.e, b, Q_F)) / total
    if ws.n[w] > 0:
        return float(bs.n[b]) / float(ws.n[w])
    return 0.0


@wp.kernel(enable_backward=False)
def _correct(inv_dt: float, bodies: _Bodies, worlds: int, ws: _WorldSums, bs: _BodySums,
             out_body_f: wp.array[wp.spatial_vector]):
    """Each proxy body's share of the world's mismatch P / dt - E, as a reaction on the body at its records' centre."""
    b = wp.tid()
    if not _is_proxy(b, bodies):
        return
    w = _world(bodies.world[b], worlds)
    g = bodies.to_proxy[b]
    if w < 0 or ws.open[w] != 0 or bs.n[b] <= 0 or g >= out_body_f.shape[0]:
        return
    share = _share(b, w, ws, bs)
    if share <= 0.0:
        return
    d_body = -share * (_get3(ws.p, w, Q_P) * inv_dt - _get3(ws.e, w, Q_F))
    centre = _get3(bs.k, b, Q_X) / float(bs.n[b])
    if bs.a[b] > wp.int64(0):
        centre = _get3(bs.c, b, Q_F) / _get(bs.a[b], Q_F)
    arm = centre - wp.transform_point(bodies.q[b], bodies.com[b])
    out_body_f[g] = out_body_f[g] + wp.spatial_vector(d_body, wp.cross(arm, d_body))   # one thread per proxy slot


class RecordMomentum:
    """Wraps one VBD destination's input notification (step-start velocities) and body harvest (the correction)."""

    def __init__(self, solver: Any) -> None:
        m, dev = solver.model, solver.device
        self.solver, self.worlds = solver, max(int(getattr(m, "world_count", 1) or 1), 1)
        W, B = self.worlds, max(int(m.body_count), 1)
        self.qd0 = wp.zeros(max(int(m.particle_count), 1), dtype=wp.vec3, device=dev)
        self.ws, self.bs = _WorldSums(), _BodySums()
        self.ws.p, self.ws.e, self.ws.e_static = (wp.zeros((W, 3), dtype=wp.int64, device=dev) for _ in range(3))
        self.ws.w_sum = wp.zeros(W, dtype=wp.int64, device=dev)
        self.ws.n, self.ws.open = (wp.zeros(W, dtype=wp.int32, device=dev) for _ in range(2))
        self.bs.e, self.bs.c, self.bs.k = (wp.zeros((B, 3), dtype=wp.int64, device=dev) for _ in range(3))
        self.bs.a = wp.zeros(B, dtype=wp.int64, device=dev)
        self.bs.n = wp.zeros(B, dtype=wp.int32, device=dev)
        self.no_f = wp.zeros(0, dtype=wp.vec3, device=dev)
        self.inner_notify = solver.coupling_notify_input_state_update
        self.inner_harvest = solver.coupling_harvest_proxy_wrenches
        solver.coupling_notify_input_state_update = self.notify
        solver.coupling_harvest_proxy_wrenches = self.harvest

    def notify(self, state: Any, flags: Any, *, iteration_restart: bool = False, dt: float = 0.0) -> None:
        if (int(flags) & _PARTICLE_Q and dt > 0.0 and not iteration_restart and state.particle_qd is not None
                and self.solver.model.particle_count > 0):
            wp.copy(dest=self.qd0, src=state.particle_qd)
        self.inner_notify(state, flags, iteration_restart=iteration_restart, dt=dt)

    def harvest(self, body_local_to_proxy_global: Any, out_body_f: Any, *, body_qd_before: Any, state: Any,
                state_out: Any, contacts: Any, dt: float) -> None:
        self.inner_harvest(body_local_to_proxy_global, out_body_f, body_qd_before=body_qd_before, state=state,
                           state_out=state_out, contacts=contacts, dt=dt)
        s = self.solver
        if (contacts is None or dt <= 0.0 or not s._integrates_rigid_bodies or s.model.particle_count == 0
                or contacts.soft_contact_max == 0
                or s.body_particle_contact_penalty_k.shape[0] < contacts.soft_contact_max):
            return
        bodies = self._bodies(body_local_to_proxy_global, state_out)
        for a in (self.ws.p, self.ws.e, self.ws.e_static, self.ws.w_sum, self.ws.n, self.ws.open, self.bs.e,
                  self.bs.a, self.bs.c, self.bs.n, self.bs.k):
            a.zero_()
        self._momentum(state, state_out, dt)
        self._records(bodies, contacts, state_out, dt)
        wp.launch(_world_weights, dim=s.model.body_count, inputs=[bodies, self.worlds, self.bs, self.ws],
                  device=s.device)
        wp.launch(_correct, dim=s.model.body_count, device=s.device,
                  inputs=[1.0 / float(dt), bodies, self.worlds, self.ws, self.bs, out_body_f])

    def _bodies(self, to_proxy: Any, state_out: Any) -> _Bodies:
        m, b = self.solver.model, _Bodies()
        b.q, b.q_prev, b.qd, b.com = (state_out.body_q, self.solver._coupling_body_q_prev_snapshot,
                                      state_out.body_qd, m.body_com)
        b.flags, b.inv_mass, b.world, b.to_proxy = m.body_flags, m.body_inv_mass, m.body_world, to_proxy
        return b

    def _momentum(self, state: Any, state_out: Any, dt: float) -> None:
        m = self.solver.model
        f_ext = state.particle_f if state.particle_f is not None else self.no_f
        wp.launch(_particle_momentum, dim=m.particle_count, device=self.solver.device,
                  inputs=[self.qd0, state_out.particle_qd, f_ext, m.particle_mass, m.particle_inv_mass,
                          m.particle_flags, m.particle_world, m.gravity, float(dt), self.worlds, self.ws])

    def _records(self, bodies: _Bodies, contacts: Any, state_out: Any, dt: float) -> None:
        s, m, r = self.solver, self.solver.model, _Records()
        r.count, r.indices, r.bary = contacts.soft_contact_count, contacts.soft_contact_indices, \
            contacts.soft_contact_barycentric
        r.shape, r.body_pos, r.body_vel = contacts.soft_contact_shape, contacts.soft_contact_body_pos, \
            contacts.soft_contact_body_vel
        r.normal, r.ke = contacts.soft_contact_normal, s.body_particle_contact_penalty_k
        r.kd, r.mu = s.body_particle_contact_material_kd, s.body_particle_contact_material_mu
        filt = _make_soft_contact_filter_data(m, contacts, state_out.particle_q, state_out.body_q)
        wp.launch(_record_forces, dim=contacts.soft_contact_max, device=s.device,
                  inputs=[float(dt), r, bodies, state_out.particle_q, s._coupling_particle_q_prev_snapshot,
                          m.particle_radius, m.particle_world, m.shape_body, m.shape_margin,
                          float(s.friction_epsilon), bool(s.rigid_soft_contact_use_log_barrier), self.worlds, filt,
                          self.ws, self.bs])


def install_record_momentum(sched: Any) -> list[RecordMomentum]:
    """One per VBD destination of a coupled (not one-way) body-proxy mapping (as ``install_stiffness_caps``)."""
    if not ENABLED:
        return []
    names = sorted({mp.dst_name for mp in sched._proxy_mappings
                    if (mp.src_name, mp.dst_name) not in sched._one_way_routes and mp.proxy_ids_local is not None
                    and isinstance(sched._entries[mp.dst_name].solver, newton.solvers.SolverVBD)})
    out = [RecordMomentum(sched._entries[n].solver) for n in names]
    for n in names:
        log.info("record momentum feedback on %s", n)
    return out
