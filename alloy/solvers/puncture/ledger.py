# SPDX-License-Identifier: Apache-2.0
"""Per-world energy ledger of the puncture plug-in (design note §6; float64, fixed summation order).

net = E − W_needle + D_fric + R, which must never rise (beyond solver round-off):
- E: KE + gravity PE (about z = 0) + stable Neo-Hookean tet energy + spring energy + anchor energy + thread stretch
  and bending (chain.py);
- W_needle: work of the needles on the tissue/thread, midpoint rule Σ ½(F_start + F_end)·Δc over each anchor's target
  motion (F_start = 0 for an anchor created this step) (Δc: the needle's rigid motion at the anchor's closest point; the thread is part of the system);
- D_fric: Coulomb + viscous slip work, Σ −f_t·(u t) ≥ 0;
- R: anchor energy removed by events (puncture conversion, release, handover), E_a(before) − E_a(after).
VBD's implicit Euler, tet damping and contact are losses and only lower net. A rise in net is an injection.
Per-item energies are written in parallel, per-world sums loop items in index order (one thread per world).
"""

from __future__ import annotations

import warp as wp

from .events import Tracks
from .needle import Needles
from .terms import FREE, STOPPER, SWAGE, TIP, Anchors, Params, Tissue, anchor_eval, anchor_target, tip_pressing


@wp.struct
class Ledger:
    e: wp.array[wp.float64]          # per world, end of the last step
    w: wp.array[wp.float64]          # per world, cumulative needle work
    d: wp.array[wp.float64]          # per world, cumulative friction dissipation
    r: wp.array[wp.float64]          # per world, cumulative event release
    net0: wp.array[wp.float64]       # per world, net after the first step
    net_prev: wp.array[wp.float64]
    step_gain_max: wp.array[wp.float64]
    started: wp.array[wp.int32]


@wp.func
def anchor_energy(j: int, an: Anchors, ts: Tissue, nd: Needles, pr: Params, x: wp.array[wp.vec3],
                  body_q: wp.array[wp.transform]) -> wp.float64:
    st = an.state[j]
    if st == FREE:
        return wp.float64(0.0)
    p, p0, c, dc, t, k, a = anchor_target(j, an, ts, nd, x, x, body_q, body_q)
    d = p - c
    if st == TIP:
        if not tip_pressing(an, nd, j, body_q, p, c):
            return wp.float64(0.0)
        return wp.float64(0.5 * pr.k_tip * wp.dot(d, d))
    if st == STOPPER:
        return wp.float64(0.5 * pr.k_n * wp.dot(d, d))
    if st == SWAGE:
        return wp.float64(0.5 * pr.k_tip * wp.dot(d, d))
    dn = d - wp.dot(d, t) * t
    return wp.float64(0.5 * pr.k_n * wp.dot(dn, dn))


@wp.kernel(enable_backward=False)
def anchor_energy_sum(an: Anchors, ts: Tissue, nd: Needles, pr: Params, tr: Tracks, x: wp.array[wp.vec3],
                      body_q: wp.array[wp.transform], out: wp.array[wp.float64]):
    w = wp.tid()
    e = wp.float64(0.0)
    for j in range(w * tr.capacity, w * tr.capacity + tr.used[w]):
        e = e + anchor_energy(j, an, ts, nd, pr, x, body_q)
    out[w] = e


@wp.kernel(enable_backward=False)
def anchor_work(dt: float, an: Anchors, ts: Tissue, nd: Needles, pr: Params, x: wp.array[wp.vec3],
                x0: wp.array[wp.vec3], body_q: wp.array[wp.transform], body_q0: wp.array[wp.transform],
                f_prev: wp.array[wp.vec3], w_out: wp.array[wp.float64], d_out: wp.array[wp.float64],
                e_out: wp.array[wp.float64], f_now: wp.array[wp.vec3]):
    """Per anchor at the end of the step: needle work (midpoint), slip dissipation, stored energy."""
    j = wp.tid()
    st = an.state[j]
    if st == FREE:
        w_out[j] = wp.float64(0.0)
        d_out[j] = wp.float64(0.0)
        e_out[j] = wp.float64(0.0)
        f_now[j] = wp.vec3(0.0)
        return
    f, h, k, a = anchor_eval(j, an, ts, nd, pr, dt, x, x0, body_q, body_q0)
    p, p0, c, dc, t, k2, a2 = anchor_target(j, an, ts, nd, x, x0, body_q, body_q0)
    fs = f_prev[j]                                   # zero for an anchor created this step
    wk = float(0.0)
    if k2 < 0:                                       # needle-borne target (TIP, NEEDLE): the needle does the work
        wk = wp.dot(0.5 * (fs + f), dc)
    ft = wp.dot(f, t) * t
    u = wp.dot(t, (p - p0) - dc)
    w_out[j] = wp.float64(wk)
    d_out[j] = wp.float64(-wp.dot(ft, u * t))
    e_out[j] = anchor_energy(j, an, ts, nd, pr, x, body_q)
    f_now[j] = f


@wp.struct
class Ranges:
    """Per-world [start, end) of particles, tets and springs (contiguous per world; checked on the host)."""

    particle: wp.array[wp.int32]
    tet: wp.array[wp.int32]
    spring: wp.array[wp.int32]


@wp.kernel(enable_backward=False)
def tet_energy(x: wp.array[wp.vec3], tet_indices: wp.array2d[wp.int32], tet_poses: wp.array[wp.mat33],
               tet_materials: wp.array2d[float], out: wp.array[wp.float64]):
    """Newton's stable Neo-Hookean (particle_vbd_kernels: P = mu F + lambda' (J - alpha) cof F) as an energy:
    V0 [mu/2 (tr F^T F - 3) + lambda'/2 (J - alpha)^2 - mu^2/(2 lambda')], lambda' = lambda + mu, alpha = 1 + mu/lambda'.
    Zero at rest."""
    t = wp.tid()
    mu = tet_materials[t, 0]
    lam = tet_materials[t, 1] + mu
    dm_inv = tet_poses[t]
    v0 = 1.0 / (wp.determinant(dm_inv) * 6.0)
    a = x[tet_indices[t, 0]]
    ds = wp.matrix_from_cols(x[tet_indices[t, 1]] - a, x[tet_indices[t, 2]] - a, x[tet_indices[t, 3]] - a)
    f = ds * dm_inv
    if lam <= 0.0:
        out[t] = wp.float64(0.0)
        return
    alpha = 1.0 + mu / lam
    ic = wp.ddot(f, f)
    jd = wp.determinant(f) - alpha
    psi = 0.5 * mu * (ic - 3.0) + 0.5 * lam * jd * jd - 0.5 * mu * mu / lam
    out[t] = wp.float64(v0) * wp.float64(psi)


@wp.kernel(enable_backward=False)
def world_energy(rg: Ranges, x: wp.array[wp.vec3], v: wp.array[wp.vec3], mass: wp.array[float],
                 gravity: wp.array[wp.vec3], e_tet: wp.array[wp.float64], spring_indices: wp.array[wp.int32],
                 spring_rest: wp.array[float], spring_k: wp.array[float], e_anchor: wp.array[wp.float64],
                 out: wp.array[wp.float64]):
    """E per world: KE + m |g| z (z about 0, along -g) + tets + springs + anchors, summed in index order."""
    w = wp.tid()
    g = gravity[0]
    if gravity.shape[0] > 1:
        g = gravity[w]
    e = wp.float64(0.0)
    for i in range(rg.particle[w], rg.particle[w + 1]):
        m = mass[i]
        e = e + wp.float64(0.5 * m * wp.dot(v[i], v[i]) - m * wp.dot(g, x[i]))
    for t in range(rg.tet[w], rg.tet[w + 1]):
        e = e + e_tet[t]
    for s in range(rg.spring[w], rg.spring[w + 1]):
        d = wp.length(x[spring_indices[2 * s]] - x[spring_indices[2 * s + 1]]) - spring_rest[s]
        e = e + wp.float64(0.5 * spring_k[s] * d * d)
    out[w] = e + e_anchor[w]


@wp.kernel(enable_backward=False)
def book(tr: Tracks, w_j: wp.array[wp.float64], d_j: wp.array[wp.float64], e_new: wp.array[wp.float64],
         r_step: wp.array[wp.float64], lg: Ledger):
    """Add this step's anchor work/dissipation and event release, then the per-step gain of net."""
    w = wp.tid()
    ws = wp.float64(0.0)
    ds = wp.float64(0.0)
    for j in range(w * tr.capacity, w * tr.capacity + tr.used[w]):
        ws = ws + w_j[j]
        ds = ds + d_j[j]
    lg.w[w] = lg.w[w] + ws
    lg.d[w] = lg.d[w] + ds
    lg.r[w] = lg.r[w] + r_step[w]
    lg.e[w] = e_new[w]
    net = e_new[w] - lg.w[w] + lg.d[w] + lg.r[w]
    if lg.started[w] == 0:
        lg.started[w] = 1
        lg.net0[w] = net
    else:
        lg.step_gain_max[w] = wp.max(lg.step_gain_max[w], net - lg.net_prev[w])
    lg.net_prev[w] = net


@wp.kernel(enable_backward=False)
def release(e_pre: wp.array[wp.float64], e_post: wp.array[wp.float64], r_step: wp.array[wp.float64]):
    w = wp.tid()
    r_step[w] = e_pre[w] - e_post[w]


def _ranges(model, W: int, device) -> Ranges:
    import numpy as np

    def starts(world_of: np.ndarray, what: str) -> np.ndarray:
        if W <= 1:
            return np.asarray([0, world_of.size], np.int32)
        if world_of.size and (np.any(world_of < 0) or np.any(np.diff(world_of) < 0)):
            raise ValueError(f"puncture ledger: {what} are not grouped by world (global items or unsorted)")
        return np.searchsorted(world_of, np.arange(W + 1)).astype(np.int32)

    pw = model.particle_world.numpy() if model.particle_count else np.zeros(0, np.int32)
    tw = pw[model.tet_indices.numpy()[:, 0]] if model.tet_count else np.zeros(0, np.int32)
    sw = pw[model.spring_indices.numpy().reshape(-1, 2)[:, 0]] if model.spring_count else np.zeros(0, np.int32)
    rg = Ranges()
    rg.particle, rg.tet, rg.spring = (wp.array(starts(a, n), dtype=wp.int32, device=device)
                                      for a, n in ((pw, "particles"), (tw, "tets"), (sw, "springs")))
    return rg


class PunctureLedger:
    """Owns the ledger arrays and the launches around one SolverPuncture step (module doc)."""

    def __init__(self, solver) -> None:
        m, dev = solver.model, solver.model.device
        W, A = solver.topology.world_count, solver.puncture.anchors_per_world
        self.s, self.W = solver, W
        z64 = lambda n: wp.zeros(n, dtype=wp.float64, device=dev)   # noqa: E731
        lg = Ledger()
        lg.e, lg.w, lg.d, lg.r, lg.net0, lg.net_prev, lg.step_gain_max = (z64(W) for _ in range(7))
        lg.started = wp.zeros(W, dtype=wp.int32, device=dev)
        self.lg = lg
        self.ranges = _ranges(m, W, dev)
        self._e_pre, self._e_post, self._r, self._e_anchor, self._e_world = (z64(W) for _ in range(5))
        self._w_j, self._d_j, self._e_j = (z64(W * A) for _ in range(3))
        self._f_prev = wp.zeros(W * A, dtype=wp.vec3, device=dev)
        self._f_now = wp.zeros(W * A, dtype=wp.vec3, device=dev)
        self._e_tet = z64(max(int(m.tet_count), 1))
        self._dummy_i = wp.zeros(2, dtype=wp.int32, device=dev)
        self._dummy_f = wp.zeros(1, dtype=float, device=dev)

    def _anchor_sum(self, x, q, out) -> None:
        s = self.s
        wp.launch(anchor_energy_sum, dim=self.W, inputs=[s.anchors, s.tissue, s.needles.dev, s.params, s.tracks, x, q,
                                                         out], device=s.model.device)

    def before_events(self, x0, q0) -> None:
        self._anchor_sum(x0, q0, self._e_pre)

    def after_events(self, x0, q0) -> None:
        self._anchor_sum(x0, q0, self._e_post)
        wp.launch(release, dim=self.W, inputs=[self._e_pre, self._e_post, self._r], device=self.s.model.device)

    def after_step(self, state_out, q1, q0, dt: float) -> None:
        s, m, dev = self.s, self.s.model, self.s.model.device
        wp.launch(anchor_work, dim=self._w_j.shape[0],
                  inputs=[dt, s.anchors, s.tissue, s.needles.dev, s.params, state_out.particle_q, s.particle_q_prev,
                          q1, q0, self._f_prev, self._w_j, self._d_j, self._e_j, self._f_now], device=dev)
        wp.copy(self._f_prev, self._f_now)
        self._anchor_sum(state_out.particle_q, q1, self._e_anchor)
        if s._n_threads:
            from .chain import chain_energy
            wp.launch(chain_energy, dim=self.W, inputs=[s.chain, s.chain_world, state_out.particle_q, self._e_anchor],
                      device=dev)
        if m.tet_count:
            wp.launch(tet_energy, dim=m.tet_count, inputs=[state_out.particle_q, m.tet_indices, m.tet_poses,
                                                          m.tet_materials, self._e_tet], device=dev)
        has_s = m.spring_count > 0
        wp.launch(world_energy, dim=self.W,
                  inputs=[self.ranges, state_out.particle_q, state_out.particle_qd, m.particle_mass, m.gravity,
                          self._e_tet, m.spring_indices if has_s else self._dummy_i,
                          m.spring_rest_length if has_s else self._dummy_f,
                          m.spring_stiffness if has_s else self._dummy_f, self._e_anchor, self._e_world], device=dev)
        wp.launch(book, dim=self.W, inputs=[s.tracks, self._w_j, self._d_j, self._e_world, self._r, self.lg],
                  device=dev)

    def report(self) -> dict[str, list[float]]:
        """Host read-back (tests and benches only): per-world E, W, D, R, net - net0 and the largest step gain."""
        e, w, d, r = (a.numpy() for a in (self.lg.e, self.lg.w, self.lg.d, self.lg.r))
        net = e - w + d + r
        return dict(E=e.tolist(), W=w.tolist(), D=d.tolist(), R=r.tolist(), net_rise=(net - self.lg.net0.numpy()).tolist(),
                    step_gain_max=self.lg.step_gain_max.numpy().tolist())
