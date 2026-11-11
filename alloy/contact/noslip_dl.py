# SPDX-License-Identifier: Apache-2.0
"""Force-space sweeps of the scoped noslip (:mod:`alloy.contact.noslip_block`): MuJoCo's projected Gauss-Seidel on
the slots' friction rows with the Delassus matrix ``C = J M^-1 J'`` of those rows formed once per solve.

Why: the qacc-space sweep re-reads a row's J and ``Y = M^-1 J'`` and sums ``J qacc`` over the warp at every row
update. At N = 1 one warp issues every instruction with nothing to hide latency, and Warp's array indexing
(negative-index wrap, strides) was most of those instructions (Nsight Compute, T12 in the grasp: 0.41 ms per solve
for 16 sweeps over 16 dof rows and two condim-6 contacts).

Here friction row g (slot order, a slot's rows consecutive) lives in lane ``g % nb``: its ``J qacc`` (v), force, aref
and dof-row constants are registers, so a row reads its residual and old force with one shuffle each; after a slot's
update every lane adds ``C[r, g] delta`` to its own rows (C column-major in scratch: one coalesced load per column).
A contact's friction block is C's diagonal block, gathered by shuffles from the lanes that loaded it. Same rows,
order, projections (:mod:`alloy.contact.noslip_cone`), cost changes and stopping rule as the qacc-space sweep;
``qacc += Y (f - f_start)`` and ``J' (f - f_start)`` are applied once at the end. Used when the slots fit (``SMAX``
slots, ``RMAX`` friction rows); otherwise the kernel keeps the qacc-space sweep.
"""

import warp as wp
from alloy._vendor.mujoco_warp._src import types as mjt

from alloy.contact import noslip_cone as cone
from alloy.contact import noslip_kernels as nk
from alloy.contact.noslip_warp import block_sync, lane_bcast

RMAX = 64  # friction rows of the force-space sweep (two registers per lane on CUDA)
SMAX = 64  # slots
KC = 48  # per-slot contact constants in sc.K: block Ac (25), its scaled la = 0 factor L0 (15, packed lower), friction
# (5), normal force, factor state
K_L0, K_FR, K_N, K_OK = 25, 40, 45, 46


@wp.func_native(snippet="""
#if defined(__CUDA_ARCH__)
return __shfl_sync(0xffffffffu, v, src);
#else
return v;
#endif
""")
def bcast_i(v: int, src: int) -> int:
    """Lane ``src``'s v in every lane (int)."""
    ...


@wp.func_native(snippet="""
#if defined(__CUDA_ARCH__)
int r = v;
for (int o = 1; o < 32; o <<= 1) {
  int t = __shfl_up_sync(0xffffffffu, r, o);
  if ((threadIdx.x & 31) >= o) r += t;
}
return r;
#else
return v;
#endif
""")
def scan_incl(v: int) -> int:
    """Inclusive prefix sum over the warp's lanes (identity on CPU's one lane)."""
    ...


@wp.func
def c_entry(w: int, sr: int, kr: int, sg: int, kg: int, tr: nk.Trees, sc: nk.Scratch) -> float:
    """C between friction row kr of slot sr and row kg of slot sg: J[sr, kr] . Y[sg, kg] over the trees they share
    (for sr = sg the same sum, in the same order, as ``nk.friction_block``). A dof friction row's J has one entry, so
    a pair with one reads it and the other row's Y there (C is symmetric: J_r M^-1 J_g' = J_g M^-1 J_r')."""
    a = sc.meta[w, sr]
    b = sc.meta[w, sg]
    if sr != sg and (a[nk.NF] == 0 or b[nk.NF] == 0):
        if a[nk.NF] != 0:  # the dof row is g: read it from the other side
            return c_dof(w, sg, b, sr, kr, a, sc)
        return c_dof(w, sr, a, sg, kg, b, sc)
    acc = float(0.0)
    for side in range(2):
        t = wp.where(side == 0, a[nk.TA], a[nk.TB])
        if t >= 0:
            ia = wp.where(side == 0, 0, a[nk.NA])
            ib = int(-1)
            if t == b[nk.TA]:
                ib = 0
            elif t == b[nk.TB]:
                ib = b[nk.NA]
            if ib >= 0:
                for i in range(tr.dofnum[t]):
                    acc += sc.J[w, nk.jidx(sc, sr, kr, ia + i)] * sc.Y[w, nk.jidx(sc, sg, kg, ib + i)]
    return acc


@wp.func
def c_dof(w: int, sd: int, md: nk.vec8i, so: int, ko: int, mo: nk.vec8i, sc: nk.Scratch) -> float:
    """C between dof friction slot sd (meta md) and row ko of slot so (meta mo): J_d[d] Y_o[d] at the row's dof d."""
    d = md[nk.DOF]
    i = nk.lane_local(d, md[nk.TA], mo)
    if i < 0:
        return 0.0
    return sc.J[w, nk.jidx(sc, sd, 0, d - md[nk.ADRA])] * sc.Y[w, nk.jidx(sc, so, ko, i)]


@wp.func
def row_jq(w: int, s: int, k: int, tr: nk.Trees, sc: nk.Scratch, qacc: wp.array2d[float]) -> float:
    """J qacc of friction row k of slot s (local J over the slot's trees; qacc after the tree phase)."""
    mt = sc.meta[w, s]
    acc = float(0.0)
    for i in range(nk.tree_size(tr, mt[nk.TA]) + nk.tree_size(tr, mt[nk.TB])):
        acc += sc.J[w, nk.jidx(sc, s, k, i)] * qacc[w, nk.gdof(tr, i, mt[nk.TA], mt[nk.TB])]
    return acc


@wp.func
def slot_rows(w: int, s: int, n: int, sc: nk.Scratch) -> int:
    """Friction rows of slot s (1 for a dof row), 0 past the slots."""
    if s >= n:
        return 0
    return wp.max(sc.meta[w, s][nk.NF], 1)


@wp.func
def tri(i: int, j: int) -> int:
    return i * (i + 1) // 2 + j


@wp.func
def cache_contact(w: int, s: int, g: int, nf: int, rows: nk.Rows, con: nk.Contacts, sc: nk.Scratch):
    """Contact slot s (friction rows g ..): its block Ac from C (diagonal floored at 1e-10, as ``contact_force``),
    friction, normal force and, at condim 6, the la = 0 factor of the QCQP's scaled block (``cone.scaled_factor``),
    constant over the sweeps."""
    r0 = sc.meta[w, s][nk.R0]
    fr = con.friction[rows.id[w, r0]]
    Ac = mjt.mat55(0.0)
    for k in range(5):
        if k < nf:
            for l in range(5):
                if l < nf:
                    Ac[k, l] = sc.C[w, (g + l) * RMAX + g + k]
            Ac[k, k] = wp.max(1.0e-10, Ac[k, k])
    base = s * KC
    for k in range(5):
        for l in range(5):
            sc.K[w, base + k * 5 + l] = Ac[k, l]
        sc.K[w, base + K_FR + k] = fr[k]
    sc.K[w, base + K_N] = rows.force[w, r0]
    l0 = int(0)
    if nf == 5:
        L0, l0 = cone.scaled_factor(Ac, fr)
        for k in range(5):
            for l in range(5):
                if l <= k:
                    sc.K[w, base + K_L0 + tri(k, l)] = L0[k, l]
    sc.K[w, base + K_OK] = float(l0)


@wp.func
def cached_block(w: int, s: int, sc: nk.Scratch):
    """(Ac, L0, l0, friction, normal) of contact slot s from :func:`cache_contact`."""
    base = s * KC
    Ac = mjt.mat55(0.0)
    L0 = mjt.mat55(0.0)
    fr = mjt.vec5(0.0)
    for k in range(5):
        for l in range(5):
            Ac[k, l] = sc.K[w, base + k * 5 + l]
            if l <= k:
                L0[k, l] = sc.K[w, base + K_L0 + tri(k, l)]
        fr[k] = sc.K[w, base + K_FR + k]
    return Ac, L0, int(sc.K[w, base + K_OK]), fr, sc.K[w, base + K_N]


def delassus(lane_dofs: int, lanes: int):
    """The force-space solve for a block of ``lanes`` threads holding ``lane_dofs`` dofs each (the kernel's layout):
    a ``wp.func`` run(...) -> 1 when it solved, 0 when the slots do not fit (the caller sweeps in qacc space)."""
    M = lane_dofs
    NB = lanes
    KR = -(-RMAX // lanes)  # row registers per lane
    KS = -(-SMAX // lanes)  # slot registers per lane
    vecM = wp.types.vector(length=M, dtype=float)
    vecR = wp.types.vector(length=KR, dtype=float)
    vecRi = wp.types.vector(length=KR, dtype=int)
    vecSi = wp.types.vector(length=KS, dtype=int)
    matR5 = wp.types.matrix(shape=(KR, 5), dtype=float)

    @wp.func
    def sel(v: vecR, j: int) -> float:
        x = float(0.0)
        for jj in range(KR):
            if jj == j:
                x = v[jj]
        return x

    @wp.func
    def put(v: vecR, j: int, x: float) -> vecR:
        for jj in range(KR):
            if jj == j:
                v[jj] = x
        return v

    @wp.func
    def sel_i(v: vecSi, j: int) -> int:
        x = int(0)
        for jj in range(KS):
            if jj == j:
                x = v[jj]
        return x

    @wp.func
    def row_of(v: vecR, g: int) -> float:
        """Row g's entry of a lane-distributed vector, in every lane."""
        j = g // NB
        return lane_bcast(sel(v, j), g - j * NB)

    @wp.func
    def first_rows(w: int, lane: int, n: int, sc: nk.Scratch):
        """(first friction row of each of this lane's slots, total friction rows R)."""
        g0 = vecSi(0)
        carry = int(0)
        for j in range(KS):
            nr = slot_rows(w, lane + j * NB, n, sc)
            inc = scan_incl(nr)
            g0[j] = carry + inc - nr
            carry += bcast_i(inc, NB - 1)
        return g0, carry

    @wp.func
    def setup(w: int, lane: int, n: int, R: int, g0: vecSi, rows: nk.Rows, con: nk.Contacts, tr: nk.Trees,
              sc: nk.Scratch, qacc: wp.array2d[float]):
        """Row map (slot, row) by friction row, C, the contacts' constants (:func:`cache_contact`), and this lane's
        registers: rows (v, f, aref, adiag, fl, efc row) and slots (friction row count, 0 = dof row)."""
        for j in range(KS):
            s = lane + j * NB
            for k in range(slot_rows(w, s, n, sc)):
                sc.rowmap[w, g0[j] + k] = s * 8 + k
        block_sync()
        for p in range(lane, R * R, NB):  # the upper triangle, mirrored (C is symmetric)
            r = p // R
            g = p - r * R
            if g >= r:
                cr = sc.rowmap[w, r]
                cg = sc.rowmap[w, g]
                c = c_entry(w, cr // 8, cr % 8, cg // 8, cg % 8, tr, sc)
                sc.C[w, g * RMAX + r] = c
                sc.C[w, r * RMAX + g] = c
        block_sync()
        v = vecR(0.0)
        f = vecR(0.0)
        aref = vecR(0.0)
        adiag = vecR(0.0)
        fl = vecR(0.0)
        erow = vecRi(0)
        for j in range(KR):
            g = lane + j * NB
            if g < R:
                code = sc.rowmap[w, g]
                s = code // 8
                k = code % 8
                mt = sc.meta[w, s]
                e = mt[nk.FIRST] + k
                erow[j] = e
                f[j] = rows.force[w, e]
                aref[j] = rows.aref[w, e]
                v[j] = row_jq(w, s, k, tr, sc, qacc)
                if mt[nk.NF] == 0:  # C[g, g] = nk.friction_block's value, bit for bit
                    adiag[j] = wp.max(nk.MINVAL, sc.C[w, g * RMAX + g])
                    fl[j] = rows.frictionloss[w, e]
        snf = vecSi(0)
        for j in range(KS):
            s = lane + j * NB
            if s < n:
                snf[j] = sc.meta[w, s][nk.NF]
                if snf[j] > 0:
                    cache_contact(w, s, g0[j], snf[j], rows, con, sc)
        block_sync()
        return v, f, aref, adiag, fl, erow, snf

    @wp.func
    def dof_row(w: int, lane: int, R: int, g: int, v: vecR, f: vecR, aref: vecR, adiag: vecR, fl: vecR,
                sc: nk.Scratch):
        """Dry-friction row g (``cone.dof_force``'s clamp and cost change); returns (v, f, cost change)."""
        c = vecR(0.0)
        for j in range(KR):
            if lane + j * NB < R:
                c[j] = sc.C[w, g * RMAX + lane + j * NB]
        old = row_of(f, g)
        res = row_of(v, g) - row_of(aref, g)
        ad = row_of(adiag, g)
        lim = row_of(fl, g)
        force = wp.clamp(old - res / ad, -lim, lim)
        delta = force - old
        for j in range(KR):
            v[j] = v[j] + c[j] * delta
        jg = g // NB
        if lane == g - jg * NB:
            f = put(f, jg, force)
        return v, f, 0.5 * delta * delta * ad + delta * res

    @wp.func
    def contact_rows(w: int, lane: int, R: int, s: int, g: int, nf: int, v: vecR, f: vecR, aref: vecR,
                     sc: nk.Scratch, warm: int):
        """Elliptic contact on rows g .. g + nf - 1 (``cone.contact_block`` on the cached block); returns (v, f, cost
        change)."""
        c = matR5(0.0)
        for j in range(KR):
            if lane + j * NB < R:
                for k in range(5):
                    if k < nf:
                        c[j, k] = sc.C[w, (g + k) * RMAX + lane + j * NB]
        Ac, L0, l0, fr, normal = cached_block(w, s, sc)
        old = mjt.vec5(0.0)
        res = mjt.vec5(0.0)
        for k in range(5):
            if k < nf:
                old[k] = row_of(f, g + k)
                res[k] = row_of(v, g + k) - row_of(aref, g + k)
        la0 = float(0.0)
        if warm != 0:
            la0 = sc.lam[w, s]
        force, change, la = cone.contact_block(Ac, res, old, fr, normal, nf, la0, L0, l0)
        sc.lam[w, s] = la  # every lane stores the same la and reads back its own write
        delta = force - old
        for j in range(KR):
            for k in range(5):
                if k < nf:
                    v[j] = v[j] + c[j, k] * delta[k]
        for k in range(5):
            if k < nf:
                jk = (g + k) // NB
                if lane == g + k - jk * NB:
                    f = put(f, jk, force[k])
        return v, f, change

    @wp.func
    def sweep(w: int, lane: int, n: int, R: int, improvement: float, v: vecR, f: vecR, aref: vecR, adiag: vecR,
              fl: vecR, snf: vecSi, sc: nk.Scratch, warm: int):
        """One Gauss-Seidel sweep over the slots in efc order; returns (v, f, improvement minus each cost change)."""
        g = int(0)
        for s in range(n):
            js = s // NB
            nf = bcast_i(sel_i(snf, js), s - js * NB)
            change = float(0.0)
            if nf == 0:
                v, f, change = dof_row(w, lane, R, g, v, f, aref, adiag, fl, sc)
                g += 1
            else:
                v, f, change = contact_rows(w, lane, R, s, g, nf, v, f, aref, sc, warm)
                g += nf
            improvement -= change
        return v, f, improvement

    @wp.func
    def finish(w: int, lane: int, nv: int, n: int, R: int, f: vecR, f0: vecR, erow: vecRi, rows: nk.Rows,
               tr: nk.Trees, sc: nk.Scratch, out: nk.Dofs):
        """efc.force of the rows; qacc += Y (f - f_start), qfrc_constraint and Ma += J' (f - f_start)."""
        for j in range(KR):
            if lane + j * NB < R:
                rows.force[w, erow[j]] = f[j]
        q = vecM(0.0)
        acc = vecM(0.0)
        for m in range(M):
            if lane + NB * m < nv:
                q[m] = out.qacc[w, lane + NB * m]
        g = int(0)
        for s in range(n):
            mt = sc.meta[w, s]
            for k in range(wp.max(mt[nk.NF], 1)):
                df = row_of(f, g) - row_of(f0, g)
                g += 1
                for m in range(M):
                    d = lane + NB * m
                    if d < nv:
                        i = nk.lane_local(d, tr.of_dof[d], mt)
                        if i >= 0:
                            q[m] = q[m] + sc.Y[w, nk.jidx(sc, s, k, i)] * df
                            acc[m] = acc[m] + sc.J[w, nk.jidx(sc, s, k, i)] * df
        for m in range(M):
            d = lane + NB * m
            if d < nv:
                out.qacc[w, d] = q[m]
                out.qfrc_constraint[w, d] = out.qfrc_constraint[w, d] + acc[m]
                out.Ma[w, d] = out.Ma[w, d] + acc[m]

    @wp.func
    def run(w: int, lane: int, nv: int, n: int, iterations: int, reg: float, scale: float, tol: float, stall: float,
            warm: int,
            rows: nk.Rows, con: nk.Contacts, tr: nk.Trees, sc: nk.Scratch, out: nk.Dofs,
            solver_niter: wp.array[int]) -> int:
        """The whole sweep phase of one world; 0 (nothing done) when the slots exceed SMAX or their rows RMAX."""
        if n > SMAX:
            return 0
        g0, R = first_rows(w, lane, n, sc)
        if R > RMAX:
            return 0
        v, f, aref, adiag, fl, erow, snf = setup(w, lane, n, R, g0, rows, con, tr, sc, out.qacc)
        f0 = f
        it = int(0)
        total = float(0.0)
        while it < iterations:
            v, f, improvement = sweep(w, lane, n, R, wp.where(it == 0, reg, 0.0), v, f, aref, adiag, fl, snf, sc,
                                      wp.where(it == 0, 0, warm))
            it += 1
            total += improvement
            # Alloy's stall stop (opt-in, stall > 0): a sweep that lowers the cost by no more than `stall` of all
            # the sweeps so far has hit the float32 floor; more sweeps only cost the batch time (TRAIN-FIX item 2)
            if improvement * scale < tol or (stall > 0.0 and it > 1 and improvement <= stall * total):
                break
        finish(w, lane, nv, n, R, f, f0, erow, rows, tr, sc, out)
        if lane == 0:
            solver_niter[w] += it
        return 1

    return run


__all__ = ["RMAX", "SMAX", "delassus"]
