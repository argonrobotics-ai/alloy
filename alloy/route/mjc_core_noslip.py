# SPDX-License-Identifier: Apache-2.0
"""Scoped noslip on MuJoCo C data in numpy: the CPU route's fallback when WP2's C extension is missing (SCR-WP8-6).

The algorithm is :mod:`alloy.contact.noslip` (``ScopedNoslip``) and its C port (``alloy/route/_mjc/noslip.c``):
MuJoCo's noslip post-solve on the frictional rows the scope selects, in efc order. A slot is one dof friction row or
one elliptic contact's ``dim - 1`` friction rows (normal force fixed). Same row selection (``keep_row``), slot cap
(``rows_per_world``; extra slots keep the main solve's forces), first-sweep regularizer cost, dry-friction clamp,
elliptic-cone QCQP (with the two Alloy differences: a result outside the cone is put on it, a non-finite one becomes
zero), cost-change rule (a contact change above 1e-10 keeps the old forces) and stopping rule.

It sweeps in force space: ``A = J_s M^-1 J_s'`` over the scoped rows once, then the residual ``J_s qacc - aref``
is kept as a vector ``v`` updated by ``A[:, rows] delta``. That is the same arithmetic as the C port's qacc-space sweep
up to summation order (parity within rounding, not bitwise). At the end ``efc_force``, ``qfrc_constraint`` (+ J' delta)
and ``qacc`` (+ M^-1 J' delta) are written; ``restart`` first sets the touched trees' ``qacc = qacc_smooth + M^-1
qfrc_constraint`` as MuJoCo's noslip does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from alloy.contact.noslip import ALL, NoslipScope, NoslipSpec

FRICTION_DOF = int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)
ELLIPTIC = int(mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC)
MINVAL = float(mujoco.mjMINVAL)
ROBOT_BIT, PAIR_BIT, SURF_BIT = 1, 2, 4
QCQP_EPS = 1.0e-10
QCQP_ITERS = 20


@dataclass
class NoslipCounters:
    slots: int = 0
    sweeps: int = 0
    overflow: int = 0


@dataclass(frozen=True)
class _Slots:
    r0: np.ndarray       # [n] dof friction row or contact normal row
    nf: np.ndarray       # [n] friction rows (0 = dof row)
    con: np.ndarray      # [n] contact id (-1 = dof row)
    rows: np.ndarray     # [k] friction efc rows of every slot, slot after slot
    start: np.ndarray    # [n + 1] each slot's first position in ``rows``


@dataclass
class ScopedNoslipNumpy:
    """One model's scoped noslip (``spec`` and its ``scope`` from :func:`alloy.contact.noslip.build_scope`)."""

    spec: NoslipSpec
    scope: NoslipScope
    max_slots: int
    counters: NoslipCounters = field(default_factory=NoslipCounters)

    def __post_init__(self) -> None:
        self._all = int(self.spec.scope == ALL)
        self._pairs = {(int(a), int(b)) for a, b in np.asarray(self.scope.pairs).reshape(-1, 2)}

    def run(self, m: mujoco.MjModel, d: mujoco.MjData) -> None:
        """The noslip on ``d`` right after ``mj_fwdConstraint`` (writes efc_force, qfrc_constraint, qacc)."""
        if d.nefc == 0 or m.nv == 0:
            return
        slots = self._select(m, d)
        if slots is None:
            return
        J = dense_rows(m, d, slots.rows)
        Y = np.empty_like(J)
        mujoco.mj_solveM(m, d, Y, J)
        if self.spec.restart:
            _restart_trees(m, d, slots)
        f0 = np.array(d.efc_force[slots.rows], dtype=np.float64)
        f, iters = self._sweeps(m, d, slots, J, Y, f0)
        delta = f - f0
        d.efc_force[slots.rows] = f
        d.qfrc_constraint[:] += J.T @ delta
        d.qacc[:] += Y.T @ delta
        d.solver_niter[0] += iters
        self.counters.sweeps += iters

    # ---- slots ------------------------------------------------------------------------------------------------------
    def _select(self, m: mujoco.MjModel, d: mujoco.MjData) -> _Slots | None:
        nefc = int(d.nefc)
        typ, ids = d.efc_type[:nefc], d.efc_id[:nefc]
        keep = np.zeros(nefc, dtype=bool)
        dof = np.flatnonzero(typ == FRICTION_DOF)
        keep[dof] = self.scope.dof[ids[dof]] != 0
        ell = np.flatnonzero(typ == ELLIPTIC)
        if ell.size:
            keep[ell] = self._keep_contacts(d, ell, ids[ell])
        first = np.flatnonzero(keep)
        over = first.size > self.max_slots
        first = first[: self.max_slots]
        self.counters.slots += int(first.size)
        self.counters.overflow += int(over)
        if first.size == 0:
            return None
        is_dof = typ[first] == FRICTION_DOF
        con = np.where(is_dof, -1, ids[first])
        dim = np.asarray(d.contact.dim)[np.maximum(con, 0)] if d.ncon else np.ones(first.size, dtype=np.int64)
        nf = np.where(is_dof, 0, dim - 1)
        nrow = np.maximum(nf, 1)
        start = np.concatenate(([0], np.cumsum(nrow)))
        base = np.where(is_dof, first, first + 1)
        rows = np.repeat(base - start[:-1], nrow) + np.arange(int(start[-1]))
        return _Slots(first, nf.astype(np.int64), con, rows.astype(np.int64), start)

    def _keep_contacts(self, d: mujoco.MjData, rows: np.ndarray, cids: np.ndarray) -> np.ndarray:
        c = d.contact
        ok = (c.efc_address[cids] == rows) & (c.dim[cids] > 1)
        g = c.geom[cids]
        gs = self.scope.geom
        sa = np.where(g[:, 0] >= 0, gs[np.maximum(g[:, 0], 0)], 0)
        sb = np.where(g[:, 1] >= 0, gs[np.maximum(g[:, 1], 0)], 0)
        x = sa ^ sb
        keep = (self._all + (x & ROBOT_BIT) + (x & SURF_BIT)) > 0
        maybe = np.flatnonzero(ok & ~keep & ((sa & sb & PAIR_BIT) != 0))
        for i in maybe:
            lo, hi = sorted((int(g[i, 0]), int(g[i, 1])))
            keep[i] = (lo, hi) in self._pairs
        return ok & keep

    # ---- sweeps -----------------------------------------------------------------------------------------------------
    def _sweeps(self, m: mujoco.MjModel, d: mujoco.MjData, s: _Slots, J: np.ndarray, Y: np.ndarray,
                f0: np.ndarray) -> tuple[np.ndarray, int]:
        A = J @ Y.T
        v = J @ np.asarray(d.qacc) - np.asarray(d.efc_aref)[s.rows]
        f = f0.copy()
        slots = _slot_data(d, s, A)
        lam = [0.0] * len(slots)
        scale = inertia_scale(m, d)
        improvement, it = _regularizer_cost(d, s), 0
        while it < self.spec.iterations:
            warm = it > 0 and self.spec.warm_qcqp
            for i, sl in enumerate(slots):
                if sl[0] == 0:
                    improvement -= _sweep_dof(sl, v, f)
                else:
                    improvement -= _sweep_contact(sl, v, f, lam, i, warm)
            it += 1
            if improvement * scale < self.spec.tolerance:
                break
            improvement = 0.0
        return f, it


def inertia_scale(m: mujoco.MjModel, d: mujoco.MjData) -> float:
    """The stop test's scale, 1 / trace(M) at the current configuration, as MuJoCo 3.17's own noslip (and its main
    solver, MuJoCo Warp 3.17 and alloy.contact.noslip) scale it. MuJoCo 3.16 used trace(M(qpos0)) =
    ``stat.meaninertia * nv``. ``d.M`` is the reduced sparse inertia; each row's diagonal is its last entry."""
    if m.nv == 0:
        return 1.0
    diag = np.asarray(d.M)[np.asarray(m.M_rowadr) + np.asarray(m.M_rownnz) - 1]
    return 1.0 / max(float(diag.sum()), MINVAL)


def _slot_data(d: mujoco.MjData, s: _Slots, A: np.ndarray) -> list[tuple]:
    """Per slot, what its sweeps reuse: (0, p, adiag, frictionloss, A column) for a dof row, (nf, p0, p1, block with
    the diagonal floored, mu, normal force, A columns) for a contact."""
    out: list[tuple] = []
    starts, nfs, r0s = s.start.tolist(), s.nf.tolist(), s.r0.tolist()
    for i, nf in enumerate(nfs):
        p0, p1, r0 = starts[i], starts[i + 1], r0s[i]
        if nf == 0:
            out.append((0, p0, max(MINVAL, float(A[p0, p0])), float(d.efc_frictionloss[r0]), A[:, p0].copy()))
            continue
        Ac = np.array(A[p0:p1, p0:p1])
        Ac[np.diag_indices(nf)] = np.maximum(1.0e-10, np.diag(Ac))
        mu = np.asarray(d.contact.friction[int(s.con[i])][:nf], dtype=np.float64)
        out.append((nf, p0, p1, Ac, mu, float(d.efc_force[r0]), np.ascontiguousarray(A[:, p0:p1])))
    return out


def _sweep_dof(sl: tuple, v: np.ndarray, f: np.ndarray) -> float:
    _, p, adiag, fl, col = sl
    old = float(f[p])
    new, change = dof_force(adiag, fl, float(v[p]), old)
    if new != old:
        f[p] = new
        v += col * (new - old)
    return change


def _sweep_contact(sl: tuple, v: np.ndarray, f: np.ndarray, lam: list[float], i: int, warm: bool) -> float:
    nf, p0, p1, Ac, mu, normal, cols = sl
    old = f[p0:p1].copy()
    new, change, la = contact_block(Ac, v[p0:p1].copy(), old, mu, normal, nf, lam[i] if warm else 0.0)
    lam[i] = la
    delta = new - old
    if np.any(delta != 0.0):
        f[p0:p1] = new
        v += cols @ delta
    return change


def dense_rows(m: mujoco.MjModel, d: mujoco.MjData, rows: np.ndarray) -> np.ndarray:
    """efc_J rows as a dense [len(rows), nv] array (sparse or dense Jacobian)."""
    nv = int(m.nv)
    if not mujoco.mj_isSparse(m):
        return np.array(np.asarray(d.efc_J)[: int(d.nefc) * nv].reshape(-1, nv)[rows])
    out = np.zeros((rows.size, nv))
    nnz, adr = np.asarray(d.efc_J_rownnz)[rows], np.asarray(d.efc_J_rowadr)[rows]
    start = np.cumsum(nnz) - nnz
    flat = np.repeat(adr - start, nnz) + np.arange(int(nnz.sum()))
    out[np.repeat(np.arange(rows.size), nnz), np.asarray(d.efc_J_colind)[flat]] = np.asarray(d.efc_J)[flat]
    return out


def _regularizer_cost(d: mujoco.MjData, s: _Slots) -> float:
    """0.5 f^2 / D over every slot's rows, the contact's normal row included (MuJoCo's first-sweep correction)."""
    rows = np.concatenate([np.arange(int(r), int(r) + int(n) + 1) for r, n in zip(s.r0, s.nf)])
    D, force = np.asarray(d.efc_D)[rows], np.asarray(d.efc_force)[rows]
    c = 0.0
    for Dk, fk in zip(D.tolist(), force.tolist()):
        if Dk > 0.0:
            c += 0.5 * fk * fk / Dk
    return c


def _restart_trees(m: mujoco.MjModel, d: mujoco.MjData, s: _Slots) -> None:
    """MuJoCo's start on the trees the slots touch: qacc = qacc_smooth + M^-1 qfrc_constraint."""
    need = np.zeros(max(int(m.ntree), 1), dtype=bool)
    for r0, con in zip(s.r0.tolist(), s.con.tolist()):
        if con < 0:
            need[m.dof_treeid[d.efc_id[r0]]] = True
            continue
        for g in d.contact.geom[con]:
            t = int(m.body_treeid[m.geom_bodyid[g]]) if g >= 0 else -1
            if t >= 0:
                need[t] = True
    tmp = np.zeros(m.nv)
    mujoco.mj_solveM(m, d, tmp.reshape(1, -1), np.asarray(d.qfrc_constraint).reshape(1, -1))
    sel = need[np.asarray(m.dof_treeid)]
    d.qacc[sel] = np.asarray(d.qacc_smooth)[sel] + tmp[sel]


# ---- projections (alloy.contact.noslip_cone; MuJoCo's mju_QCQP2 / QCQP3 / QCQP) ----------------------------------
def dof_force(adiag: float, frictionloss: float, r: float, old: float) -> tuple[float, float]:
    """Dry friction row: (new force clamped to +-frictionloss, cost change); ``r`` = J qacc - aref."""
    force = min(max(old - r / adiag, -frictionloss), frictionloss)
    delta = force - old
    return force, 0.5 * delta * delta * adiag + delta * r


def contact_block(Ac: np.ndarray, res: np.ndarray, old: np.ndarray, mu: np.ndarray, normal: float, nf: int,
                  la0: float) -> tuple[np.ndarray, float, float]:
    """One elliptic contact's friction rows: (new forces, cost change, QCQP la). A change above 1e-10 keeps ``old``."""
    bc = res - Ac @ old
    force, la = cone_force(Ac, bc, mu, normal, nf, la0)
    delta = force - old
    change = float(0.5 * delta @ (Ac @ delta) + delta @ res)
    if change > 1.0e-10:
        return old, 0.0, la
    return force, change, la


def cone_force(Ac: np.ndarray, bc: np.ndarray, mu: np.ndarray, normal: float, nf: int,
               la0: float) -> tuple[np.ndarray, float]:
    """Friction minimising 0.5 f'Af + f'b inside the elliptic cone of the fixed normal force; (force, la)."""
    if normal < MINVAL:
        return np.zeros(nf), 0.0
    la = 0.0
    if nf == 2:
        force, active = qcqp2(Ac, bc, mu, normal)
    elif nf == 3:
        force, active = qcqp3(Ac, bc, mu, normal)
    else:
        force, active, la = qcqpn(Ac, bc, mu, normal, nf, la0 if nf == 5 else 0.0)
        la = la if nf == 5 else 0.0
    fk = force / np.where(mu != 0.0, mu, MINVAL)
    ss = float(fk @ fk)
    if not math.isfinite(ss):
        return np.zeros(nf), 0.0
    if active or ss > normal * normal:
        force = force * math.sqrt(normal * normal / max(MINVAL, ss))
    return force, la


def qcqp2(Ain: np.ndarray, bin_: np.ndarray, d: np.ndarray, r: float) -> tuple[np.ndarray, bool]:
    """min 0.5 x'Ax + x'b s.t. sum (x_i / d_i)^2 <= r^2 in 2 dimensions (mju_QCQP2's operations)."""
    d0, d1 = float(d[0]), float(d[1])
    b1, b2 = float(bin_[0]) * d0, float(bin_[1]) * d1
    A11, A22 = float(Ain[0, 0]) * d0 * d0, float(Ain[1, 1]) * d1 * d1
    A12 = float(Ain[0, 1]) * d0 * d1
    la = v1 = v2 = 0.0
    for _ in range(QCQP_ITERS):
        det = (A11 + la) * (A22 + la) - A12 * A12
        if det < QCQP_EPS:
            return np.zeros(2), False
        detinv = 1.0 / det
        P11, P22, P12 = (A22 + la) * detinv, (A11 + la) * detinv, -A12 * detinv
        v1 = -P11 * b1 - P12 * b2
        v2 = -P12 * b1 - P22 * b2
        val = v1 * v1 + v2 * v2 - r * r
        if val < QCQP_EPS:
            break
        deriv = -2.0 * (P11 * v1 * v1 + 2.0 * P12 * v1 * v2 + P22 * v2 * v2)
        delta = -val / deriv
        if delta < QCQP_EPS:
            break
        la += delta
    return np.array([v1 * d0, v2 * d1]), la != 0.0


def qcqp3(Ain: np.ndarray, bin_: np.ndarray, d: np.ndarray, r: float) -> tuple[np.ndarray, bool]:
    """The same in 3 dimensions (mju_QCQP3's operations)."""
    d0, d1, d2 = float(d[0]), float(d[1]), float(d[2])
    b1, b2, b3 = float(bin_[0]) * d0, float(bin_[1]) * d1, float(bin_[2]) * d2
    A11, A22, A33 = float(Ain[0, 0]) * d0 * d0, float(Ain[1, 1]) * d1 * d1, float(Ain[2, 2]) * d2 * d2
    A12, A13, A23 = float(Ain[0, 1]) * d0 * d1, float(Ain[0, 2]) * d0 * d2, float(Ain[1, 2]) * d1 * d2
    la = v1 = v2 = v3 = 0.0
    for _ in range(QCQP_ITERS):
        P11 = (A22 + la) * (A33 + la) - A23 * A23
        P22 = (A11 + la) * (A33 + la) - A13 * A13
        P33 = (A11 + la) * (A22 + la) - A12 * A12
        P12 = A13 * A23 - A12 * (A33 + la)
        P13 = A12 * A23 - A13 * (A22 + la)
        P23 = A12 * A13 - A23 * (A11 + la)
        det = (A11 + la) * P11 + A12 * P12 + A13 * P13
        if det < QCQP_EPS:
            return np.zeros(3), False
        detinv = 1.0 / det
        P11, P22, P33, P12, P13, P23 = (x * detinv for x in (P11, P22, P33, P12, P13, P23))
        v1 = -P11 * b1 - P12 * b2 - P13 * b3
        v2 = -P12 * b1 - P22 * b2 - P23 * b3
        v3 = -P13 * b1 - P23 * b2 - P33 * b3
        val = v1 * v1 + v2 * v2 + v3 * v3 - r * r
        if val < QCQP_EPS:
            break
        deriv = -2.0 * (P11 * v1 * v1 + P22 * v2 * v2 + P33 * v3 * v3) - 4.0 * (
            P12 * v1 * v2 + P13 * v1 * v3 + P23 * v2 * v3)
        delta = -val / deriv
        if delta < QCQP_EPS:
            break
        la += delta
    return np.array([v1 * d0, v2 * d1, v3 * d2]), la != 0.0


def _chol_shift(A: list[list[float]], n: int, la: float) -> list[list[float]] | None:
    """Lower Cholesky factor of A + la I (MuJoCo's loop and 1e-10 rank threshold, Python floats); None: not SPD."""
    L = [row[:] for row in A]
    for i in range(n):
        L[i][i] = A[i][i] + la
    for j in range(n):
        Lj = L[j]
        diag = Lj[j]
        for k in range(j):
            diag -= Lj[k] * Lj[k]
        if diag < QCQP_EPS:
            return None
        diag = math.sqrt(diag)
        Lj[j] = diag
        for i in range(j + 1, n):
            Li = L[i]
            val = Li[j]
            for k in range(j):
                val -= Li[k] * Lj[k]
            Li[j] = val / diag
    return L


def _chol_solve(L: list[list[float]], n: int, b: list[float]) -> list[float]:
    """Solve L L' x = b (forward then back substitution, MuJoCo's order)."""
    x = [0.0] * n
    for i in range(n):
        val = b[i]
        for k in range(i):
            val -= L[i][k] * x[k]
        x[i] = val / L[i][i]
    for i in range(n - 1, -1, -1):
        val = x[i]
        for k in range(i + 1, n):
            val -= L[k][i] * x[k]
        x[i] = val / L[i][i]
    return x


def _dot(a: list[float], b: list[float]) -> float:
    s = 0.0
    for x, y in zip(a, b):
        s += x * y
    return s


def qcqpn(Ain: np.ndarray, bin_: np.ndarray, d: np.ndarray, r: float, n: int,
          la0: float = 0.0) -> tuple[np.ndarray, bool, float]:
    """The generic QCQP (n <= 5): Newton on phi(la) = |x(la)|^2 - r^2 from ``la0`` (alloy's qcqp5_warm; 0 is MuJoCo's
    start); (x, active, la)."""
    dd = [float(v) for v in d[:n]]
    Ar = np.asarray(Ain).tolist()
    A = [[Ar[i][j] * dd[i] * dd[j] for j in range(n)] for i in range(n)]
    b = [float(bin_[i]) * dd[i] for i in range(n)]
    res = [0.0] * n
    la = la0
    for _ in range(QCQP_ITERS):
        L = _chol_shift(A, n, la)
        if L is None:
            return np.zeros(n), False, 0.0
        res = [-v for v in _chol_solve(L, n, b)]
        val = _dot(res, res) - r * r
        if val < QCQP_EPS:
            if la == la0 and la0 > 0.0 and val < 0.0:   # warm start right of the root: one tangent step left
                la = max(la + val / (2.0 * _dot(res, _chol_solve(L, n, res))), 0.0)
                continue
            break
        delta = -val / (-2.0 * _dot(res, _chol_solve(L, n, res)))
        if delta < QCQP_EPS:
            break
        la += delta
    return np.array([res[i] * dd[i] for i in range(n)]), la != 0.0, la


__all__ = ["NoslipCounters", "ScopedNoslipNumpy", "cone_force", "contact_block", "dense_rows", "dof_force", "qcqp2",
           "qcqp3", "qcqpn"]
