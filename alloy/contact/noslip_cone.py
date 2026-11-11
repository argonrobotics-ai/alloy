# SPDX-License-Identifier: Apache-2.0
"""Friction-row projections of the scoped noslip (:mod:`alloy.contact.noslip`): the dry-friction clamp of a dof
friction row and the elliptic-cone QCQP of a contact's friction rows with its normal force fixed, as MuJoCo's noslip
does them (mujoco_warp@86e2fae0:mujoco_warp/_src/solver.py:5008-5112). ``qcqp5`` is MuJoCo Warp's ``_qcqp`` at
n = 5 (condim-6 contacts) with static loops: bit-identical results on CPU, but the 5 x 5 factor stays in registers
(the generic version indexes a local-memory matrix dynamically: ~10 µs per call at N = 1 on an L4).

One deliberate difference: a QCQP result outside the cone is put on it even when the QCQP reports "inactive". Its
Newton loop stops at the first step when that step is below 1e-10 (a friction block with a nearly uncontrolled
direction), returning the unconstrained solution outside the cone; on T12 that unbounded friction was a 1,570 m/s
to 1e17 m/s kick (SB-7, both MuJoCo Warp's noslip and this one).
"""

import warp as wp
from alloy._vendor.mujoco_warp._src import types as mjt
from alloy._vendor.mujoco_warp._src.solver import _qcqp, _qcqp2, _qcqp3

from alloy.contact.noslip_kernels import MINVAL, Rows, Scratch, aidx


@wp.func
def dof_force(sc: Scratch, s: int, w: int, r0: int, res: float, old: float, rows: Rows):
    """Dry friction row: clamp to [-frictionloss, frictionloss]; (new force, cost change)."""
    adiag = wp.max(MINVAL, sc.A[w, aidx(sc, s, 0, 0)])
    fl = rows.frictionloss[w, r0]
    force = wp.clamp(old - (res - rows.aref[w, r0]) / adiag, -fl, fl)
    delta = force - old
    return force, 0.5 * delta * delta * adiag + delta * (res - rows.aref[w, r0])


@wp.func
def chol_solve5(L: mjt.mat55, b: mjt.vec5) -> mjt.vec5:
    """mujoco_warp ``_chol_solve5`` at n = 5 with static loops (registers, same operation order)."""
    x = mjt.vec5(0.0)
    for i in range(5):
        val = b[i]
        for k in range(5):
            if k < i:
                val -= L[i, k] * x[k]
        x[i] = val / L[i, i]
    for ii in range(5):
        i = 4 - ii
        val = x[i]
        for k in range(5):
            if k > i:
                val -= L[k, i] * x[k]
        x[i] = val / L[i, i]
    return x


@wp.func
def chol5(A: mjt.mat55, la: float):
    """Lower Cholesky factor of A + la I, static loops (mju_cholFactor, 1e-10 rank threshold); (L, ok)."""
    L = A
    for i in range(5):
        L[i, i] = A[i, i] + la
    for j in range(5):
        diag = L[j, j]
        for k in range(5):
            if k < j:
                diag -= L[j, k] * L[j, k]
        if diag < 1.0e-10:
            return L, False
        diag = wp.sqrt(diag)
        L[j, j] = diag
        for i in range(5):
            if i > j:
                val = L[i, j]
                for k in range(5):
                    if k < j:
                        val -= L[i, k] * L[j, k]
                L[i, j] = val / diag
    return L, True


@wp.func
def qcqp5_warm(Ain: mjt.mat55, bin: mjt.vec5, d: mjt.vec5, r: float, la0: float):
    """mujoco_warp ``_qcqp`` at n = 5 (condim-6 contacts) with static loops, its Newton iteration on la started at
    ``la0`` (the contact's la of the previous sweep; 0 = MuJoCo's start, bit-identical); returns (x, active, la).

    Newton on phi(la) = |x(la)|^2 - r^2 (convex, decreasing) from la0 > 0: a start left of the root climbs to it as
    from 0 (monotone), a start right of it (phi < 0) takes one tangent step, which lands left of the root by
    convexity (clamped at 0, where an inside-the-cone solution then stops as MuJoCo's does). Same root, so the same
    forces up to the stopping tolerance; a converged la takes 1-2 factorizations instead of MuJoCo's 5-20."""
    A = mjt.mat55(0.0)
    b = mjt.vec5(0.0)
    for i in range(5):
        b[i] = bin[i] * d[i]
        for j in range(5):
            A[i, j] = Ain[i, j] * d[i] * d[j]
    res = mjt.vec5(0.0)
    la = la0
    for _ in range(20):
        L, ok = chol5(A, la)
        if not ok:
            return mjt.vec5(0.0), False, 0.0
        res = -chol_solve5(L, b)
        val = wp.dot(res, res) - r * r
        if val < 1.0e-10:
            if la == la0 and la0 > 0.0 and val < 0.0:  # warm start right of the root: one tangent step left
                la = wp.max(la + val / (2.0 * wp.dot(res, chol_solve5(L, res))), 0.0)
                continue
            break
        deriv = -2.0 * wp.dot(res, chol_solve5(L, res))
        delta = -val / deriv
        if delta < 1.0e-10:
            break
        la += delta
    for i in range(5):
        res[i] = res[i] * d[i]
    return res, la != 0.0, la


@wp.func
def qcqp5(Ain: mjt.mat55, bin: mjt.vec5, d: mjt.vec5, r: float):
    """mujoco_warp ``_qcqp`` at n = 5 (condim-6 contacts) with static loops: same Newton iteration and operation
    order, but the 5 x 5 factor stays in registers (the generic version indexes local memory dynamically)."""
    x, active, la = qcqp5_warm(Ain, bin, d, r, 0.0)
    return x, active


@wp.func
def qcqp5_l0(Ain: mjt.mat55, bin: mjt.vec5, d: mjt.vec5, r: float, la0: float, L0: mjt.mat55, l0: int):
    """:func:`qcqp5_warm` given the la = 0 factor ``L0`` of the scaled block (``l0`` 1; 2: that factor failed; 0: not
    given): a solve that starts at la = 0 and lands inside the cone (a sticking contact) returns without factoring,
    bit-identical to qcqp5_warm; every other case runs qcqp5_warm."""
    if la0 == 0.0 and l0 != 0:
        if l0 == 2:
            return mjt.vec5(0.0), False, 0.0
        b = mjt.vec5(0.0)
        for i in range(5):
            b[i] = bin[i] * d[i]
        res = -chol_solve5(L0, b)
        if wp.dot(res, res) - r * r < 1.0e-10:
            for i in range(5):
                res[i] = res[i] * d[i]
            return res, False, 0.0
    return qcqp5_warm(Ain, bin, d, r, la0)


@wp.func
def scaled_factor(Ain: mjt.mat55, d: mjt.vec5):
    """(L0, l0) for :func:`qcqp5_l0`: chol5 of qcqp5_warm's scaled block at la = 0 (same operations)."""
    A = mjt.mat55(0.0)
    for i in range(5):
        for j in range(5):
            A[i, j] = Ain[i, j] * d[i] * d[j]
    L, ok = chol5(A, 0.0)
    return L, wp.where(ok, 1, 2)


@wp.func
def cone_force_warm(Ac: mjt.mat55, bc: mjt.vec5, friction: mjt.vec5, normal: float, nf: int, la0: float,
                    L0: mjt.mat55, l0: int):
    """Friction force minimising 0.5 f'Af + f'b inside the elliptic cone of the fixed normal force; (force, la): la is
    the condim-6 QCQP's multiplier (its warm start next sweep, see :func:`qcqp5_warm`), 0 for the other sizes.
    ``L0``, ``l0``: the condim-6 block's la = 0 factor when known (:func:`qcqp5_l0`; l0 = 0: not known)."""
    force = mjt.vec5(0.0)
    if normal < MINVAL:
        return force, 0.0
    active = bool(False)
    la = float(0.0)
    if nf == 2:
        force, active = _qcqp2(Ac, bc, friction, normal)
    elif nf == 3:
        force, active = _qcqp3(Ac, bc, friction, normal)
    elif nf == 5:
        force, active, la = qcqp5_l0(Ac, bc, friction, normal, la0, L0, l0)
    else:
        force, active = _qcqp(Ac, bc, friction, normal, nf)
    ss = float(0.0)
    for k in range(nf):
        fk = force[k] / wp.where(friction[k] != 0.0, friction[k], MINVAL)
        ss += fk * fk
    if not wp.isfinite(ss):
        return mjt.vec5(0.0), 0.0
    if active or ss > normal * normal:  # on or outside the cone: put the force on the ellipsoid
        scl = wp.sqrt(normal * normal / wp.max(MINVAL, ss))
        for k in range(nf):
            force[k] = force[k] * scl
    return force, la


@wp.func
def cone_force(Ac: mjt.mat55, bc: mjt.vec5, friction: mjt.vec5, normal: float, nf: int) -> mjt.vec5:
    """Friction force minimising 0.5 f'Af + f'b inside the elliptic cone of the fixed normal force."""
    force, la = cone_force_warm(Ac, bc, friction, normal, nf, 0.0, mjt.mat55(0.0), 0)
    return force


@wp.func
def contact_block(Ac: mjt.mat55, res: mjt.vec5, old: mjt.vec5, friction: mjt.vec5, normal: float, nf: int, la0: float,
                  L0: mjt.mat55, l0: int):
    """One elliptic contact's friction rows with the normal force fixed (mujoco_warp solver.py:5008-5112): the block
    ``Ac`` (J M^-1 J' on its nf friction rows, diagonal already floored at 1e-10) and the residual ``res`` = J qacc -
    aref at the current forces ``old``; (new forces, cost change, QCQP la). A change above 1e-10 keeps the old forces
    (change 0). Static loops: the block stays in registers. ``L0``, ``l0``: see :func:`qcqp5_l0`."""
    bc = mjt.vec5(0.0)
    for k in range(5):
        if k < nf:
            bc[k] = res[k]
            for l in range(5):
                if l < nf:
                    bc[k] = bc[k] - Ac[k, l] * old[l]
    force, la = cone_force_warm(Ac, bc, friction, normal, nf, la0, L0, l0)
    delta = force - old
    change = float(0.0)
    for k in range(5):
        if k < nf:
            ad = float(0.0)
            for l in range(5):
                if l < nf:
                    ad += Ac[k, l] * delta[l]
            change += 0.5 * delta[k] * ad + delta[k] * res[k]
    if change > 1.0e-10:
        return old, 0.0, la
    return force, change, la


@wp.func
def contact_force(sc: Scratch, s: int, w: int, r0: int, nf: int, jq: mjt.vec5, old: mjt.vec5, friction: mjt.vec5,
                  rows: Rows, la0: float):
    """:func:`contact_block` for slot s with its block from ``sc.A`` and aref from efc, from the residual ``jq`` = J
    qacc; (new forces, cost change, QCQP la). ``la0``: the QCQP's warm start (:func:`cone_force_warm`)."""
    Ac = mjt.mat55(0.0)
    res = mjt.vec5(0.0)
    for k in range(5):
        if k < nf:
            for l in range(5):
                if l < nf:
                    Ac[k, l] = sc.A[w, aidx(sc, s, k, l)]
            res[k] = jq[k] - rows.aref[w, r0 + 1 + k]
            Ac[k, k] = wp.max(1.0e-10, Ac[k, k])
    return contact_block(Ac, res, old, friction, rows.force[w, r0], nf, la0, mjt.mat55(0.0), 0)
