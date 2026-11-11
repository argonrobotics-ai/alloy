# SPDX-License-Identifier: Apache-2.0
"""Shell inertia inside SolverDFSPH (engine item 3, SPH-VBD coupler P2: liquid on light shells and cloth; design
[internal note], handoff notes/wp9-p2-handoff.md, WP9 review 2026-11-06).

Why. P1 prescribes the soft surface over an exchange (soft_walls.py), so the liquid sees a wall that does not yield and
the staggered pass loop has to find the shell's response alone. Its error grows like the added-mass ratio m_a/m_s, and
under water a cloth runs at 10-100: P1's cloth probe spent the whole pass cap (8) on every exchange.

What. Inside one exchange each soft vertex k is a particle of VBD's mass m_k (Newton installs VBD's effective masses
in the SPH view, solver_coupled.py:2871-2879) under the liquid's impulse and the non-fluid acceleration VBD's pass
implies, spread evenly over the exchange:
    v_k(t) = v_begin_k + a_k t + J_k(t) / m_k,        a_k = (v_end_k - v_begin_k) / dt_x - F_used_k / m_k
- v_begin_k, v_end_k: the vertex's velocity at the exchange start and VBD's end velocity. Newton notifies SPH twice
  per pass: when it copies the coupled state in (the begin state, solver_coupled.py:3246-3257) and after the sync
  (the end state, solver_coupled_proxy.py:1489-1494); both are gathered at the notify, so nothing is kept across
  exchanges (rollback and freeze need no adapter). The wall starts from end - v_end dt_x (coupling_hooks.py).
- J_k(t): the liquid's impulse on k so far in the exchange (soft_walls' int64 window plus the step's slot impulses),
  read at the end of the current SPH step (t = its end time).
- F_used_k: the fluid force VBD used in this pass. Newton adds ``proxy.coupling_forces`` to VBD's particle_f
  (solver_coupled_proxy.py:1423-1432), skips the particle rewind for staggered pairs (:1498-1499) and hands the same
  array to the harvest (:1582-1590), which overwrites it after SPH has stepped. It is copied at the first SPH step of
  every pass (``pass_start``); before the first harvest it is zero, the coupler's initial value.
- At convergence (the harvest J(dt_x)/dt_x = F_used) SPH's end velocity is VBD's. Where VBD's response to the force
  is its mass (a free sheet), a_k does not depend on F_used, so the path SPH sees is the same in every pass and the
  interface converges at once (L6). Adding the shell's response J/m on top of v_end instead counts the force twice: a
  plate of mass m on a spring k under a column of added mass m_a would get omega^2 = k (m + m_a) / (m (m + 2 m_a))
  instead of k / (m + m_a).
- Spread, not front-loaded: v_end - F_used dt_x / m + J(t) / m (the non-fluid impulse all at t = 0) gave a light taut
  cloth a ~1 m/s jump inside every exchange (its load is held by tension), and v_end + (J(t) - F_used t) / m made the
  mid-exchange path depend on F_used (L6 at m_a / m = 50: 5.9 passes per exchange).

Inside each pressure solve the wall also yields to the current iterate (Gissler et al. 2019's interlinked solve, on
vertices). The CG product (krylov.py) gains
    (A k)_i += s sum_b gpsi_b . a_b,   a_b = sum_v w_bv a_v,   a_v = (1/m_v) sum_{(j,c) at v} w_cv m_j k_j gpsi_c
with s = dt (divergence) or dt^2 (density): A = A_fluid + s B M^-1 B^T, symmetric positive semidefinite for equal
particle masses, so CG stays valid. That is the condition the fluid CG already has (krylov.py: one particle mass and
rest density per solver; a liquid with unequal masses runs Jacobi, where the soft walls stay P1). The slot reaction
m_j k_j gpsi_c is the one ``apply_pressure_kernel`` gives the wall. After each solve and the non-pressure stage the
vertex and slot wall velocities are refreshed from the step's slot impulses. Vertices of VBD mass 0 (pinned) are
infinitely heavy. Elasticity stays in VBD; the harvest stays the exchange impulse / dt_x.

Scope: open (two-sided) components only, i.e. cloth, membranes and shells. Closed tet bodies keep P1's prescribed
walls bit for bit. A surface vertex of a solid moves with the whole body through its stiffness, so the free-vertex
model mispredicts it: on L3's free cube (m_a/m ~ 0.5) shell inertia raised the passes per exchange from 1.97 to 2.52
and the coupler's momentum drift from 2.8e-4 to 7.5e-4 of P0, the same base.

Two-sided cloth (``soft_project``, called by integrate.project_kernel): a liquid particle that ends a step on the
other side of a cloth face, or within half its thickness, goes back to its start side; the start side is the slot's
start-of-step normal, so nothing is kept across steps. At the cloth's open border the side is not enforced.

Newton lines cited are at the pin 4f3b41ef (alloy.core.newton_compat.NEWTON_PIN); P1 cited 8f3466f9, where the
proxy loop is the same. Forces in the product are summed per vertex in int64 fixed point (2^44: 5.7e-14 N resolution,
|F| < 5.2e5 N per vertex), so the product is deterministic and capture-safe.
"""

from __future__ import annotations

import logging

import numpy as np
import warp as wp

from .bodies import IMP_SCALE, _fx, _ufx
from .pressure import MODE_DIVERGENCE, Particles, is_fluid
from .soft_walls import SoftArrays, SoftWalls, _accumulate_kernel, _face_at, _unit_normal, soft_closest

log = logging.getLogger("alloy.sph")
FORCE_SCALE = wp.constant(17592186044416.0)  # 2^44 [1/N]


@wp.func
def _row(a: wp.array2d[wp.int64], k: int, scale: float) -> wp.vec3:
    return wp.vec3(_ufx(a[k, 0], scale), _ufx(a[k, 1], scale), _ufx(a[k, 2], scale))


@wp.func
def _shell_slot(soft: SoftArrays, s: int, face: int) -> bool:
    """A slot on a face of an open (two-sided) component: the faces shell inertia acts on."""
    if s < soft.base:
        return False
    return soft.two_sided[soft.face_comp[face]] != 0


@wp.func
def _slot_mix(soft: SoftArrays, face: int, w: wp.vec3, a: wp.array[wp.vec3]) -> wp.vec3:
    t = soft.tri[face]
    return w[0] * a[t[0]] + w[1] * a[t[1]] + w[2] * a[t[2]]


@wp.func
def _on_border(soft: SoftArrays, f: int, w: wp.vec3) -> bool:
    """Whether the closest point with barycentric weights w lies on the open border of face f's surface."""
    nz = int(0)
    e = int(0)  # zero weight at index 2: edge (t0, t1) = adj 0; at 0: (t1, t2) = adj 1; at 1: (t2, t0) = adj 2
    k = int(0)  # the vertex when two weights are zero
    for j in range(3):
        if w[j] == 0.0:
            nz += 1
        else:
            k = j
    if nz == 0:
        return False
    if nz == 1:
        if w[0] == 0.0:
            e = 1
        elif w[1] == 0.0:
            e = 2
        return soft.adj[f][e] < 0
    return soft.vopen[soft.tri[f][k]] != 0


@wp.func
def soft_project(soft: SoftArrays, f: int, x: wp.vec3, dt: float, n0: wp.vec3, eps: float):
    """(x, n, v_b, pushed): point x projected out of soft face f at its end-of-step place (the face advanced by dt).

    Closed surfaces: out along the outward normal to phi = eps (P1). Two-sided (cloth, SPH-VBD P2): back onto the
    start side, the side of the slot's start-of-step normal n0, whenever x ended on the other side of the face or
    within half + eps of it, so liquid never crosses a cloth within a step. Fresh every step (n0 comes from this
    step's boundary pass), so no state crosses steps. Where the closest point lies on the cloth's open border the
    side is not enforced: liquid may pass around a free edge."""
    phi, n, _xb, vb, w, dist = soft_closest(soft, f, x, dt)
    pushed = int(0)
    if soft.two_sided[soft.face_comp[f]] == 0 or _on_border(soft, f, w):
        if phi < eps:
            x = x + (eps - phi) * n
            pushed = 1
        return x, n, vb, pushed
    a, b, c = _face_at(soft, f, dt)
    nf = _unit_normal(a, b, c)
    if wp.dot(n0, nf) < 0.0:
        nf = -nf
    xc = w[0] * a + w[1] * b + w[2] * c
    hgt = wp.dot(x - xc, nf)  # signed height above the face on the start side
    if hgt < soft.half + eps:
        x = x + (soft.half + eps - hgt) * nf
        n = nf
        pushed = 1
    return x, n, vb, pushed


@wp.kernel(enable_backward=False)
def gather_velocity_kernel(vid: wp.array[wp.int32], qd: wp.array[wp.vec3], last: wp.array[wp.vec3],
                           prev: wp.array[wp.vec3]):
    """At a notify: the previous notify's velocities move to ``prev``, this state's to ``last``."""
    k = wp.tid()
    prev[k] = last[k]
    last[k] = qd[vid[k]]


@wp.kernel(enable_backward=False)
def pass_start_kernel(vid: wp.array[wp.int32], shell: wp.array[wp.int32], mass: wp.array[wp.float32],
                      l2g: wp.array[int], f_global: wp.array[wp.vec3], has_f: int, inv_m: wp.array[wp.float32],
                      f_used: wp.array[wp.vec3]):
    """Pass start: 1/m (0 for mass 0: pinned, and off the shells) and F_used copied from the coupler's array."""
    k = wp.tid()
    i = vid[k]
    m = mass[i]
    im = float(0.0)
    if m > 0.0 and shell[k] != 0:
        im = 1.0 / m
    f = wp.vec3(0.0, 0.0, 0.0)
    if has_f != 0:
        g = l2g[i]
        if g >= 0:
            f = f_global[g]
    inv_m[k] = im
    f_used[k] = f


@wp.kernel(enable_backward=False)
def set_velocity_kernel(vid: wp.array[wp.int32], shell: wp.array[wp.int32], v_begin: wp.array[wp.vec3],
                        v_end: wp.array[wp.vec3], inv_m: wp.array[wp.float32], f_used: wp.array[wp.vec3],
                        dt_x: float, tau: float, exch: wp.array2d[wp.int64], step_j: wp.array2d[wp.int64],
                        qd: wp.array[wp.vec3], v: wp.array[wp.vec3]):
    """v_k = v_begin + a tau + (window + step impulse) / m at exchange time tau (module doc), into the state and the
    soft faces' velocities; vertices off the shells are left at VBD's end velocity (P1)."""
    k = wp.tid()
    if shell[k] == 0:
        return
    J = wp.vec3(_ufx(exch[k, 0] + step_j[k, 0], IMP_SCALE), _ufx(exch[k, 1] + step_j[k, 1], IMP_SCALE),
                _ufx(exch[k, 2] + step_j[k, 2], IMP_SCALE))
    im = inv_m[k]
    a = (v_end[k] - v_begin[k]) / dt_x - im * f_used[k]
    vk = v_begin[k] + tau * a + im * J
    qd[vid[k]] = vk
    v[k] = vk


@wp.kernel(enable_backward=False)
def refresh_vb_kernel(slot_shape: wp.array2d[wp.int32], slot_face: wp.array2d[wp.int32],
                      slot_bary: wp.array2d[wp.vec3], soft: SoftArrays, slot_vb: wp.array2d[wp.vec3]):
    """Soft slots' wall velocity: the barycentric mix of the refreshed vertex velocities."""
    i, b = wp.tid()
    if _shell_slot(soft, slot_shape[i, b], slot_face[i, b]):
        slot_vb[i, b] = _slot_mix(soft, slot_face[i, b], slot_bary[i, b], soft.v)


@wp.kernel(enable_backward=False)
def scatter_force_kernel(p: Particles, slot_shape: wp.array2d[wp.int32], slot_face: wp.array2d[wp.int32],
                         slot_bary: wp.array2d[wp.vec3], slot_gpsi: wp.array2d[wp.vec3], soft: SoftArrays,
                         vec: wp.array[wp.float32], out: wp.array2d[wp.int64]):
    """B^T vec: the wall reaction m_i vec_i gpsi_b of every soft slot, split to its face's vertices (fixed point)."""
    i, b = wp.tid()
    if not _shell_slot(soft, slot_shape[i, b], slot_face[i, b]) or not is_fluid(p, i):
        return
    f = (p.mass[i] * vec[i]) * slot_gpsi[i, b]
    t = soft.tri[slot_face[i, b]]
    w = slot_bary[i, b]
    for a in range(3):
        for d in range(3):
            wp.atomic_add(out, t[a], d, _fx(f[d] * w[a], FORCE_SCALE))


@wp.kernel(enable_backward=False)
def vertex_accel_kernel(fx: wp.array2d[wp.int64], inv_m: wp.array[wp.float32], a_v: wp.array[wp.vec3]):
    k = wp.tid()
    a_v[k] = inv_m[k] * _row(fx, k, FORCE_SCALE)


@wp.kernel(enable_backward=False)
def product_kernel(p: Particles, slot_shape: wp.array2d[wp.int32], slot_face: wp.array2d[wp.int32],
                   slot_bary: wp.array2d[wp.vec3], slot_gpsi: wp.array2d[wp.vec3], soft: SoftArrays,
                   a_v: wp.array[wp.vec3], mode: int, dt: float, vec: wp.array[wp.float32],
                   converged: wp.array[wp.int32], restart: wp.array[wp.int32], gated: int,
                   out: wp.array[wp.float32], pair: wp.array[wp.vec2]):
    """out_i += s sum_b gpsi_b . a_b and pair_i.x += vec_i times that (krylov.cg_apply_kernel's skip rules)."""
    i = wp.tid()
    if not is_fluid(p, i):
        return
    wi = p.world[i]
    if converged[wi] != 0 or (gated != 0 and restart[wi] == 0):
        return
    c = float(0.0)
    hit = int(0)
    for b in range(4):
        if _shell_slot(soft, slot_shape[i, b], slot_face[i, b]):
            c += wp.dot(_slot_mix(soft, slot_face[i, b], slot_bary[i, b], a_v), slot_gpsi[i, b])
            hit = 1
    if hit == 0:
        return
    if mode == MODE_DIVERGENCE:
        c *= dt
    else:
        c *= dt * dt
    out[i] = out[i] + c
    q = pair[i]
    pair[i] = wp.vec2(q[0] + vec[i] * c, q[1])


class SoftInertia:
    """Per-vertex inertia of the soft faces of one SolverDFSPH (module doc). Active once a coupler has synced the
    proxies (a standalone soft run keeps P1's prescribed walls)."""

    def __init__(self, soft: SoftWalls, slots, device):
        self.soft, self.slots, self.device = soft, slots, device  # slots: the solver's BoundarySlots
        V = int(soft.vid.shape[0]) if soft.count else 1
        vec = lambda: wp.zeros(V, dtype=wp.vec3, device=device)  # noqa: E731
        self.inv_m = wp.zeros(V, dtype=wp.float32, device=device)
        self.v_begin, self.v_end, self.f_used, self.a_v = vec(), vec(), vec(), vec()
        self.shell = wp.array(shell_vertices(soft) if soft.count else [0], dtype=wp.int32, device=device)
        self.step_j = wp.zeros((V, 3), dtype=wp.int64, device=device)  # the step's slot impulses per vertex
        self.force = wp.zeros((V, 3), dtype=wp.int64, device=device)  # the CG product's wall forces per vertex
        self._work = wp.zeros(max(soft.components, 1), dtype=wp.int64, device=device)  # scratch for the scatter
        self._empty_f = wp.zeros(1, dtype=wp.vec3, device=device)
        self._empty_map = wp.full(1, -1, dtype=int, device=device)
        self.feedback = None  # (local_to_global, coupler force array), from the first harvest
        self.active = False
        self._t, self._tau, self._dt_x = 0.0, 0.0, 1.0  # exchange time at the step's start and end; dt_x [s]
        self._notifies = 0  # velocity notifies since the last pass start (Newton: copy-in, then sync)
        self._warned = False

    def _vlaunch(self, kernel, inputs: list) -> None:
        wp.launch(kernel, dim=self.soft.vid.shape[0], inputs=inputs, device=self.device)

    def on_notify(self, state) -> None:
        """Each coupler notify that writes particle velocities: the begin state (copy-in), then the end (sync)."""
        self._vlaunch(gather_velocity_kernel, [self.soft.vid, state.particle_qd, self.v_end, self.v_begin])
        self._notifies += 1

    def pass_start(self, model, dt_x: float) -> None:
        """First SPH step after a proxy sync: 1/m and F_used; exchange time 0 (v_begin, v_end from the notifies)."""
        self.active = True
        l2g, f = self.feedback if self.feedback is not None else (self._empty_map, self._empty_f)
        self._vlaunch(pass_start_kernel, [self.soft.vid, self.shell, model.particle_mass, l2g, f,
                                          int(self.feedback is not None), self.inv_m, self.f_used])
        self._t, self._dt_x = 0.0, float(dt_x)
        if self._notifies < 2:  # no copy-in before the sync (a caller other than Newton's coupler): constant v_end
            wp.copy(self.v_begin, self.v_end)
            if self.feedback is not None and not self._warned:
                log.warning("SolverDFSPH soft inertia: %d velocity notify(s) before this pass, expected Newton's two "
                            "(copy-in, then sync); v_begin = v_end, so the shell path loses its spread (a change in "
                            "the coupler's notify order)", self._notifies)
                self._warned = True
        self._notifies = 0

    def begin_step(self, state, dt: float) -> None:
        """Step start (before the gather): the velocities the step would end with without its own liquid impulse
        (host floats, like P1's rewind: a captured exchange replays them in order)."""
        if not self.active:
            return
        self._tau = self._t + float(dt)
        self.step_j.zero_()
        self.set_velocity(state)

    def set_velocity(self, state) -> None:
        self._vlaunch(set_velocity_kernel, [self.soft.vid, self.shell, self.v_begin, self.v_end, self.inv_m,
                                            self.f_used, self._dt_x, self._tau, self.soft.exch, self.step_j,
                                            state.particle_qd, self.soft.arrays.v])

    def refresh(self, state, slot_impulse) -> None:
        """After a solve or the non-pressure stage: vertex velocities from the step's slot impulses so far, then the
        soft slots' wall velocities."""
        if not self.active:
            return
        sl = self.slots
        self.step_j.zero_()
        wp.launch(_accumulate_kernel, dim=sl.shape.shape, device=self.device,
                  inputs=[sl.shape, sl.face, sl.bary, sl.vb, slot_impulse, self.soft.arrays, 0, self.step_j,
                          self._work])
        self.set_velocity(state)
        wp.launch(refresh_vb_kernel, dim=sl.shape.shape, device=self.device,
                  inputs=[sl.shape, sl.face, sl.bary, self.soft.arrays, sl.vb])

    def end_step(self, state) -> None:
        """After soft_walls summed the step into the exchange window: end-of-step velocities from the window alone."""
        if not self.active:
            return
        self.step_j.zero_()
        self.set_velocity(state)
        self._t = self._tau

    def product(self, p: Particles, mode: int, dt: float, vec, out, pair, converged, restart, gated: int) -> None:
        """The interlinked term of one CG product (module doc), added to ``out`` and ``pair``."""
        if not self.active:
            return
        sl = self.slots
        self.force.zero_()
        wp.launch(scatter_force_kernel, dim=sl.shape.shape, device=self.device,
                  inputs=[p, sl.shape, sl.face, sl.bary, sl.gpsi, self.soft.arrays, vec, self.force])
        self._vlaunch(vertex_accel_kernel, [self.force, self.inv_m, self.a_v])
        wp.launch(product_kernel, dim=sl.shape.shape[0], device=self.device,
                  inputs=[p, sl.shape, sl.face, sl.bary, sl.gpsi, self.soft.arrays, self.a_v, int(mode), float(dt),
                          vec, converged, restart, int(gated), out, pair])

    def capture_feedback(self, local_to_global, out) -> None:
        """Harvest hook: keep the coupler's force array (VBD reads it as particle_f next pass); it must stay the
        same array from pass to pass (Newton 4f3b41ef hands ``proxy.coupling_forces`` every time)."""
        if self.feedback is None:
            self.feedback = (local_to_global, out)
            return
        if out.ptr != self.feedback[1].ptr or local_to_global.ptr != self.feedback[0].ptr:
            raise RuntimeError("SolverDFSPH soft inertia: the harvest got a different proxy force array than before "
                               "(a coupler or Scheduler rebuilt on the same solver?); F_used would be read from a "
                               "stale array. Build a fresh SolverDFSPH for the new coupler (soft_inertia.py doc)")


def shell_vertices(soft: SoftWalls):
    """Host: 1 per compact soft vertex of an open (two-sided) component, the vertices shell inertia moves."""
    tri, comp, two = soft.arrays.tri.numpy(), soft.arrays.face_comp.numpy(), soft.arrays.two_sided.numpy()
    flag = np.zeros(int(soft.vid.shape[0]), np.int32)
    on = two[comp] != 0
    for c in range(3):
        flag[tri[on, c]] = 1
    return flag
