# SPDX-License-Identifier: Apache-2.0
"""Block-per-world solve of the scoped noslip (:mod:`alloy.contact.noslip`).

One warp (``LANES`` = 32 threads) runs a world's whole noslip in one launch: scoped row selection in efc order (warp
ballot, no atomics, no sort), each slot's local J, the factor of each tree a slot touches (trees with no scoped row
keep the main solver's qacc, e.g. T09's 93-dof cable under scope "robot"; :mod:`alloy.contact.noslip_warp`),
``Y = M^-1 J'`` and friction blocks, then MuJoCo's projected Gauss-Seidel sweeps.

Sweeps: lane ``l`` holds qacc of dofs ``l, l + 32, ...`` in registers. A contact's residual ``J qacc`` is a warp
shuffle sum (the bitwise-same value in every lane, so every lane computes the same projection and control flow stays
uniform); a dof friction row's J is one entry, so its residual is one shuffle from the lane holding that dof. The
update ``qacc += Y delta`` touches registers only, ``J' delta`` accumulates in registers for the final
``qfrc_constraint`` / ``Ma`` write, and slot s's forces stay in lane ``s % 32``'s registers (two slots per lane;
later slots in efc.force) until the end.

Why: on T12 at N = 1 one warp issues every instruction with no other warp to hide latency (Nsight Compute: 5.3 cycles
per instruction), so a row update costs its dependent chain: the one-thread-per-world sweep took ~1.5 µs per row,
efc.force round trips go to L2 after each store, and a dof row's 5-step warp sum added zeros. On CPU the block is
one lane, which then holds every dof (same code; the shuffles are identities).
"""

# no `from __future__ import annotations`: Warp evaluates the closure types (vecM) in kernel signatures
import functools

import warp as wp
from alloy._vendor.mujoco_warp._src import types as mjt

from alloy.contact import noslip_cone as cone
from alloy.contact import noslip_kernels as nk
from alloy.contact.noslip_dl import delassus
from alloy.contact.noslip_warp import (LANES, block_sync, lane_bcast, lane_prefix, refresh_trees, warp_sum,
                                      warp_sum5)


@wp.func
def slot_forces(w: int, lane: int, nb: int, n: int, j: int, rows: nk.Rows, sc: nk.Scratch) -> mjt.vec5:
    """Forces of slot lane + j nb (zero past n): each lane keeps two slots' forces in registers between updates."""
    f = mjt.vec5(0.0)
    s = lane + j * nb
    if s < n:
        mt = sc.meta[w, s]
        for k in range(5):
            if k < wp.max(mt[nk.NF], 1):
                f[k] = rows.force[w, mt[nk.FIRST] + k]
    return f


@wp.func
def old_forces(w: int, nb: int, s: int, nr: int, first: int, fa: mjt.vec5, fb: mjt.vec5, rows: nk.Rows) -> mjt.vec5:
    """Slot s's current forces (nr rows): from the lane keeping them (s < 2 nb), else from efc.force."""
    old = mjt.vec5(0.0)
    for k in range(5):
        if k < nr:
            if s < nb:
                old[k] = lane_bcast(fa[k], s)
            elif s < 2 * nb:
                old[k] = lane_bcast(fb[k], s - nb)
            else:
                old[k] = rows.force[w, first + k]
    return old


@wp.func
def keep_forces(w: int, lane: int, nb: int, s: int, nr: int, first: int, fa: mjt.vec5, fb: mjt.vec5,
                new: mjt.vec5, rows: nk.Rows):
    """Store slot s's new forces where :func:`old_forces` reads them; returns (fa, fb)."""
    if s < nb:
        if lane == s:
            fa = new
    elif s < 2 * nb:
        if lane == s - nb:
            fb = new
    else:
        for k in range(5):
            if k < nr:
                rows.force[w, first + k] = new[k]
    return fa, fb


@wp.func
def flush_forces(w: int, lane: int, nb: int, n: int, j: int, f: mjt.vec5, rows: nk.Rows, sc: nk.Scratch):
    """Write the register forces of slot lane + j nb back to efc.force."""
    s = lane + j * nb
    if s < n:
        mt = sc.meta[w, s]
        for k in range(5):
            if k < wp.max(mt[nk.NF], 1):
                rows.force[w, mt[nk.FIRST] + k] = f[k]


@wp.func
def keep_at(w: int, r: int, nrow: int, rows: nk.Rows, con: nk.Contacts) -> int:
    if r >= nrow:
        return 0
    return nk.keep_row(w, r, rows, con)


@wp.func
def take(w: int, lane: int, r: int, keep: int, base: int, sc: nk.Scratch, overflow: wp.array[int]) -> int:
    """Slot of a kept row (ballot prefix); returns the next base."""
    pre = lane_prefix(keep)
    if keep == 1:
        slot = base + pre[0]
        if slot < sc.maxsel:
            sc.rows[w, slot] = r
        else:
            overflow[w] = 1
    return base + pre[1]


@wp.func
def select_slots(w: int, lane: int, nb: int, nrow: int, rows: nk.Rows, con: nk.Contacts, sc: nk.Scratch,
                 overflow: wp.array[int]) -> int:
    """Scoped rows into sc.rows[w, :] in efc order; returns the scoped count (may exceed sc.maxsel). Four rounds of nb
    rows per iteration: their keep tests are independent, so their dependent load chains overlap."""
    base = int(0)
    for c in range(0, (nrow + nb - 1) // nb, 4):
        r = c * nb + lane
        k0 = keep_at(w, r, nrow, rows, con)
        k1 = keep_at(w, r + nb, nrow, rows, con)
        k2 = keep_at(w, r + 2 * nb, nrow, rows, con)
        k3 = keep_at(w, r + 3 * nb, nrow, rows, con)
        base = take(w, lane, r, k0, base, sc, overflow)
        base = take(w, lane, r + nb, k1, base, sc, overflow)
        base = take(w, lane, r + 2 * nb, k2, base, sc, overflow)
        base = take(w, lane, r + 3 * nb, k3, base, sc, overflow)
    return base


@wp.func
def gather_slots(w: int, lane: int, nb: int, n: int, rows: nk.Rows, con: nk.Contacts, tr: nk.Trees,
                 sc: nk.Scratch):
    """Per (slot, friction row), one lane each: packed meta and the trees it needs (k = 0), local J."""
    for p in range(lane, n * sc.maxf, nb):
        s = p // sc.maxf
        k = p - s * sc.maxf
        r0 = sc.rows[w, s]
        nf, ta, tb = nk.slot_trees(rows, w, r0, con, tr)
        first = wp.where(nf == 0, r0, r0 + 1)
        if k == 0:
            adra = wp.where(ta >= 0, tr.dofadr[wp.max(ta, 0)], 0)
            adrb = wp.where(tb >= 0, tr.dofadr[wp.max(tb, 0)], wp.where(nf == 0, rows.id[w, r0], 0))  # nk.DOF
            sc.meta[w, s] = nk.vec8i(r0, nf, ta, tb, adra, nk.tree_size(tr, ta), adrb, first)
            if ta >= 0:
                sc.need[w, ta] = 1
            if tb >= 0:
                sc.need[w, tb] = 1
        if k < wp.max(nf, 1):
            nk.copy_row(sc, s, k, first + k, ta, tb, w, rows, tr)


@wp.func
def solve_slots(w: int, lane: int, nb: int, n: int, tr: nk.Trees, sc: nk.Scratch):
    """Y = M^-1 J' per (slot, friction row), one lane each."""
    for p in range(lane, n * sc.maxf, nb):
        s = p // sc.maxf
        k = p - s * sc.maxf
        mt = sc.meta[w, s]
        if k < wp.max(mt[nk.NF], 1):
            nk.solve_row(sc, s, k, mt[nk.TA], mt[nk.TB], w, tr)


@wp.func
def friction_blocks(w: int, lane: int, nb: int, n: int, tr: nk.Trees, sc: nk.Scratch):
    """A[s] = J M^-1 J' over each slot's friction rows, one entry per lane."""
    mf2 = sc.maxf * sc.maxf
    for p in range(lane, n * mf2, nb):
        s = p // mf2
        k = (p - s * mf2) // sc.maxf
        l = p - s * mf2 - k * sc.maxf
        mt = sc.meta[w, s]
        nr = wp.max(mt[nk.NF], 1)
        if k < nr and l < nr:
            nk.friction_block(sc, s, k, l, nk.tree_size(tr, mt[nk.TA]) + nk.tree_size(tr, mt[nk.TB]), w)


@wp.func
def prepare(w: int, lane: int, nb: int, ntree: int, nrow: int, rows: nk.Rows, con: nk.Contacts, tr: nk.Trees,
            ms: nk.Mass, sc: nk.Scratch, out: nk.Dofs, overflow: wp.array[int]) -> int:
    """World w's slots, local J, tree factors (restarting qacc on those trees) and Y; returns the slot count to sweep
    (0: nothing to do). Every return is warp-uniform (counts come from a ballot). The qacc-space sweep also needs
    :func:`friction_blocks`; the force-space one reads the blocks from its Delassus matrix."""
    count = select_slots(w, lane, nb, nrow, rows, con, sc, overflow)
    if lane == 0:
        sc.count[w] = count
    n = wp.min(count, sc.maxsel)
    if nrow == 0 or (n == 0 and con.all_contacts == 0):
        return 0
    for t in range(lane, ntree, nb):
        sc.need[w, t] = con.all_contacts  # scope "all" restarts every tree, as MuJoCo's noslip
    block_sync()
    gather_slots(w, lane, nb, n, rows, con, tr, sc)
    block_sync()
    refresh_trees(w, lane, nb, ntree, tr, ms, sc, out)
    block_sync()
    if n == 0:
        return 0
    solve_slots(w, lane, nb, n, tr, sc)
    block_sync()
    return n


@functools.cache
def solve_kernel(lane_dofs: int, lanes: int = LANES, warm_qcqp: bool = True, force_space: bool = True):
    """The block kernel for ``lane_dofs`` dofs per lane (ceil(nv / lanes)) and a block of ``lanes`` threads (32 on
    CUDA, 1 on CPU: a compile-time constant, so lane index arithmetic needs no runtime division); ``warm_qcqp``:
    condim-6 cone QCQPs after the first sweep start from the slot's previous la (:func:`noslip_cone.qcqp5_warm`; False
    = MuJoCo's start every sweep); ``force_space``: sweep on the Delassus matrix of the slots' rows when they fit
    (:mod:`alloy.contact.noslip_dl`), else in qacc space. Compiled once per combination."""
    M = lane_dofs
    NB = lanes
    WARM = int(warm_qcqp)
    FS = int(force_space)
    force_space_run = delassus(lane_dofs, lanes)
    vecM = wp.types.vector(length=M, dtype=float)
    vecMi = wp.types.vector(length=M, dtype=int)
    matM5 = wp.types.matrix(shape=(M, 5), dtype=float)

    @wp.func
    def coefficients(w: int, lane: int, nb: int, nv: int, s: int, mt: nk.vec8i, tree: vecMi, sc: nk.Scratch):
        """This lane's J and Y entries of slot s (zero off its trees); static indices keep them in registers."""
        jm = matM5(0.0)
        ym = matM5(0.0)
        nr = wp.max(mt[nk.NF], 1)
        for m in range(M):
            d = lane + nb * m
            i = int(-1)
            if d < nv:
                i = nk.lane_local(d, tree[m], mt)
            if i >= 0:
                for k in range(5):
                    if k < nr:
                        jm[m, k] = sc.J[w, nk.jidx(sc, s, k, i)]
                        ym[m, k] = sc.Y[w, nk.jidx(sc, s, k, i)]
        return jm, ym

    @wp.func
    def sweep_dof(w: int, lane: int, nb: int, nv: int, s: int, mt: nk.vec8i, old: float, q: vecM, acc: vecM,
                  tree: vecMi, rows: nk.Rows, sc: nk.Scratch):
        """Dof friction row: MuJoCo Warp writes its J as one entry (1 on dof mt[DOF]), so J qacc comes from the lane
        holding that dof (the warp sum of the generic row would add zeros); returns (q, acc, force, cost decrease)."""
        d = mt[nk.DOF]
        jv = sc.J[w, nk.jidx(sc, s, 0, d - mt[nk.ADRA])]
        part = float(0.0)
        for m in range(M):
            if lane + nb * m == d:
                part = jv * q[m]
        f0, change = cone.dof_force(sc, s, w, mt[nk.FIRST], lane_bcast(part, d % nb), old, rows)
        delta = f0 - old
        for m in range(M):
            dd = lane + nb * m
            if dd < nv and tree[m] == mt[nk.TA]:
                q[m] = q[m] + sc.Y[w, nk.jidx(sc, s, 0, dd - mt[nk.ADRA])] * delta
            if dd == d:
                acc[m] = acc[m] + jv * delta
        return q, acc, f0, change

    @wp.func
    def sweep_contact(w: int, lane: int, nb: int, nv: int, s: int, mt: nk.vec8i, old: mjt.vec5, q: vecM, acc: vecM,
                      tree: vecMi, rows: nk.Rows, con: nk.Contacts, sc: nk.Scratch, warm: int):
        """Elliptic contact: J qacc as a warp sum, then the cone QCQP (warm-started from the slot's previous la when
        ``warm``; every lane stores the same la, so each reads back its own write); returns (q, acc, forces, cost
        decrease)."""
        nf = mt[nk.NF]
        jm, ym = coefficients(w, lane, nb, nv, s, mt, tree, sc)
        part = mjt.vec5(0.0)
        for k in range(5):
            if k < nf:
                for m in range(M):
                    part[k] = part[k] + jm[m, k] * q[m]
        la0 = float(0.0)
        if warm != 0:
            la0 = sc.lam[w, s]
        new, change, la = cone.contact_force(sc, s, w, mt[nk.R0], nf, warp_sum5(part), old,
                                             con.friction[rows.id[w, mt[nk.R0]]], rows, la0)
        sc.lam[w, s] = la
        delta = new - old
        for m in range(M):
            for k in range(5):
                q[m] = q[m] + ym[m, k] * delta[k]
                acc[m] = acc[m] + jm[m, k] * delta[k]
        return q, acc, new, change

    @wp.func
    def sweep(w: int, lane: int, nb: int, nv: int, n: int, improvement: float, q: vecM, acc: vecM, tree: vecMi,
              fa: mjt.vec5, fb: mjt.vec5, rows: nk.Rows, con: nk.Contacts, sc: nk.Scratch, warm: int):
        """One Gauss-Seidel sweep in efc order; returns (q, acc, fa, fb, improvement minus each row's cost change, in
        MuJoCo's order). Every lane computes the same forces; slot s's forces live in lane s % nb (fa: s < nb, fb:
        s < 2 nb) or in efc.force."""
        for s in range(n):
            mt = sc.meta[w, s]
            nr = wp.max(mt[nk.NF], 1)
            old = old_forces(w, nb, s, nr, mt[nk.FIRST], fa, fb, rows)
            new = mjt.vec5(0.0)
            change = float(0.0)
            if mt[nk.NF] == 0:
                q, acc, f0, change = sweep_dof(w, lane, nb, nv, s, mt, old[0], q, acc, tree, rows, sc)
                new = mjt.vec5(f0, 0.0, 0.0, 0.0, 0.0)
            else:
                q, acc, new, change = sweep_contact(w, lane, nb, nv, s, mt, old, q, acc, tree, rows, con, sc, warm)
            fa, fb = keep_forces(w, lane, nb, s, nr, mt[nk.FIRST], fa, fb, new, rows)
            improvement -= change
        return q, acc, fa, fb, improvement

    @wp.kernel(enable_backward=False, module="unique")
    def solve(njmax: int, iterations: int, nv: int, ntree: int, nefc: wp.array[int], tolerance: wp.array[float],
              stall: float,
              rows: nk.Rows, con: nk.Contacts, tr: nk.Trees, ms: nk.Mass,
              sc: nk.Scratch, out: nk.Dofs, overflow: wp.array[int], solver_niter: wp.array[int]):
        w, lane = wp.tid()
        nb = int(NB)
        n = prepare(w, lane, nb, ntree, wp.min(nefc[w], njmax), rows, con, tr, ms, sc, out, overflow)
        if n == 0:
            return
        q = vecM(0.0)
        acc = vecM(0.0)
        tree = vecMi(0)
        for m in range(M):
            d = lane + nb * m
            if d < nv:
                q[m] = out.qacc[w, d]
                tree[m] = tr.of_dof[d]
        reg = float(0.0)  # MuJoCo adds the rows' 0.5 f^2 R to the first sweep's improvement (lanes split the slots)
        for s in range(lane, n, nb):
            reg += nk.regularizer_cost(sc, s, w, rows)
        reg = warp_sum(reg)
        trace = float(0.0)  # MuJoCo Warp 3.17's stop-test scale: the inverse trace of this world's M (solver.py:145)
        for i in range(lane, nv, nb):
            trace += ms.M[w, ms.rowadr[i] + ms.rownnz[i] - 1]
        scale = 1.0 / wp.max(1.0e-15, warp_sum(trace))
        tol = tolerance[w % tolerance.shape[0]]
        if wp.static(FS != 0):
            if force_space_run(w, lane, nv, n, iterations, reg, scale, tol, stall, WARM, rows, con, tr, sc, out,
                               solver_niter) == 1:
                return
        friction_blocks(w, lane, nb, n, tr, sc)
        block_sync()
        fa = slot_forces(w, lane, nb, n, 0, rows, sc)
        fb = slot_forces(w, lane, nb, n, 1, rows, sc)
        it = int(0)
        total = float(0.0)  # the cost decrease of every sweep so far (the stall test's yardstick)
        while it < iterations:
            q, acc, fa, fb, improvement = sweep(w, lane, nb, nv, n, wp.where(it == 0, reg, 0.0), q, acc, tree, fa, fb,
                                                rows, con, sc, wp.where(it == 0, 0, WARM))
            it += 1
            total += improvement
            # Alloy's stall stop (opt-in, stall > 0): a sweep that lowers the cost by no more than `stall` of all
            # the sweeps so far has hit the float32 floor; more sweeps only cost the batch time (TRAIN-FIX item 2)
            if improvement * scale < tol or (stall > 0.0 and it > 1 and improvement <= stall * total):
                break
        flush_forces(w, lane, nb, n, 0, fa, rows, sc)
        flush_forces(w, lane, nb, n, 1, fb, rows, sc)
        for m in range(M):
            d = lane + nb * m
            if d < nv:
                out.qacc[w, d] = q[m]
                out.qfrc_constraint[w, d] = out.qfrc_constraint[w, d] + acc[m]
                out.Ma[w, d] = out.Ma[w, d] + acc[m]
        if lane == 0:
            solver_niter[w] += it

    return solve


__all__ = ["LANES", "solve_kernel"]
