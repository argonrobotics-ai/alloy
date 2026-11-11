# SPDX-License-Identifier: Apache-2.0
"""Warp structs and functions of the scoped noslip (:mod:`alloy.contact.noslip`).

MuJoCo's noslip (mujoco_warp@86e2fae0:mujoco_warp/_src/solver.py:5008-5112, 5221-5370) on a subset of the frictional
rows, with the inverse inertia applied per kinematic tree instead of densely over all dofs. Per world, a slot is
either one dof friction row (``FRICTION_DOF``) or one elliptic contact (its ``dim - 1`` friction rows). M is block
diagonal over trees, so ``M^-1 J'`` of a slot lives on the dofs of the (at most two) trees it touches: local index
i < nA is dof ``adrA + i``, else ``adrB + i - nA``.

Scratch arrays are [world, index]: one block of threads per world (:mod:`alloy.contact.noslip_block`) reads a world's
scratch contiguously and factors only the trees its slots touch (:func:`refresh_tree8` in registers up to ``MAXT8``
dofs, the block's warp-wide factor above); the other trees keep the main solver's qacc. Row projections:
:mod:`alloy.contact.noslip_cone`.
"""

from __future__ import annotations

import warp as wp
from alloy._vendor.mujoco_warp._src import types as mjt

FRICTION_DOF = int(mjt.ConstraintType.FRICTION_DOF)
ELLIPTIC = int(mjt.ConstraintType.CONTACT_ELLIPTIC)
MINVAL = float(mjt.MJ_MINVAL)
vec8i = wp.types.vector(length=8, dtype=wp.int32)
# slot meta (vec8i): normal/dof efc row r0, friction rows nf (0 = dof row), tree A, tree B, dofadr A, dofnum A,
# dofadr B (dof rows: the dof), first friction efc row
R0, NF, TA, TB, ADRA, NA, ADRB, FIRST = 0, 1, 2, 3, 4, 5, 6, 7
DOF = ADRB  # dof friction slots (NF = 0, no tree B): the row's dof
ROBOT_BIT, PAIR_BIT, SURF_BIT = 1, 2, 4  # geom_scope bits: geom on a scoped tree; geom in a scoped geom pair;
# surface geom (its contacts with geoms that are not surface geoms are scoped)


@wp.struct
class Trees:
    dofadr: wp.array[int]
    dofnum: wp.array[int]
    of_dof: wp.array[int]  # dof -> tree


@wp.struct
class Rows:  # MuJoCo Warp efc views; J is sparse [w, 0, nnz] or dense [w, row, dof]
    sparse: int
    type: wp.array2d[int]
    id: wp.array2d[int]
    J_rownnz: wp.array2d[int]
    J_rowadr: wp.array2d[int]
    J_colind: wp.array3d[int]
    J: wp.array3d[float]
    D: wp.array2d[float]
    aref: wp.array2d[float]
    frictionloss: wp.array2d[float]
    force: wp.array2d[float]


@wp.struct
class Scratch:  # all [world, index]
    maxsel: int
    maxt: int
    maxf: int
    restart: int  # 1: needed trees restart qacc from qfrc_constraint (MuJoCo); 0: keep the main solve's qacc
    count: wp.array[int]
    rows: wp.array2d[int]
    meta: wp.array2d[vec8i]
    J: wp.array2d[float]
    Y: wp.array2d[float]
    A: wp.array2d[float]
    L: wp.array2d[float]
    V: wp.array2d[float]
    need: wp.array2d[int]  # [world, tree] 1 = a slot touches the tree (factor it; restart its qacc if restart)
    lam: wp.array2d[float]  # [world, slot] the slot's last cone-QCQP multiplier (warm start of the next sweep)
    rowmap: wp.array2d[int]  # [world, friction row] slot * 8 + row (force-space sweep, noslip_dl)
    C: wp.array2d[float]  # [world, g * RMAX + r] Delassus matrix of the slots' friction rows (column-major)
    K: wp.array2d[float]  # [world, slot * KC + i] per-contact constants of the force-space sweep (noslip_dl)


@wp.struct
class Dofs:  # MuJoCo Warp dof-space outputs the noslip updates
    qacc: wp.array2d[float]
    qfrc_constraint: wp.array2d[float]
    Ma: wp.array2d[float]


@wp.struct
class Mass:  # MuJoCo Warp's sparse M (rows hold columns <= row) and the smooth terms the noslip restarts from
    rownnz: wp.array[int]
    rowadr: wp.array[int]
    colind: wp.array[int]
    M: wp.array2d[float]
    qacc_smooth: wp.array2d[float]
    qfrc_smooth: wp.array2d[float]


@wp.struct
class Contacts:  # MuJoCo Warp contact views and the scope masks
    dim: wp.array[int]
    efc_address: wp.array2d[int]
    geom: wp.array[wp.vec2i]
    friction: wp.array[mjt.vec5]
    geom_bodyid: wp.array[int]
    body_treeid: wp.array[int]
    geom_scope: wp.array[int]  # ROBOT_BIT | PAIR_BIT | SURF_BIT
    dof_scope: wp.array[int]
    pairs: wp.array[wp.vec2i]  # scoped geom pairs, (min, max)
    all_contacts: int


@wp.func
def tree_size(tr: Trees, t: int) -> int:
    if t < 0:
        return 0
    return tr.dofnum[t]


@wp.func
def gdof(tr: Trees, i: int, ta: int, tb: int) -> int:
    na = tree_size(tr, ta)
    if i < na:
        return tr.dofadr[ta] + i
    return tr.dofadr[tb] + i - na


@wp.func
def local_of(tree: int, dof: int, ta: int, tb: int, tr: Trees) -> int:
    """Local index of ``dof`` (on kinematic tree ``tree``) in a slot on trees (ta, tb); -1 if not on them."""
    if tree == ta and ta >= 0:
        return dof - tr.dofadr[ta]
    if tree == tb and tb >= 0:
        return tree_size(tr, ta) + dof - tr.dofadr[tb]
    return -1


@wp.func
def lane_local(d: int, tree: int, mt: vec8i) -> int:
    """Local index of dof d (on kinematic tree ``tree``) in the slot described by ``mt``; -1 if not on its trees."""
    if tree == mt[TA] and mt[TA] >= 0:
        return d - mt[ADRA]
    if tree == mt[TB] and mt[TB] >= 0:
        return mt[NA] + d - mt[ADRB]
    return -1


@wp.func
def jidx(sc: Scratch, s: int, k: int, i: int) -> int:
    return (s * sc.maxf + k) * 2 * sc.maxt + i


@wp.func
def aidx(sc: Scratch, s: int, k: int, l: int) -> int:
    return (s * sc.maxf + k) * sc.maxf + l


@wp.func
def lidx(sc: Scratch, t: int, i: int, j: int) -> int:
    return (t * sc.maxt + i) * sc.maxt + j


@wp.func
def keep_row(w: int, r: int, rows: Rows, con: Contacts) -> int:
    """1 if efc row ``r`` starts a scoped slot: a scoped dof friction row, or a scoped elliptic contact's normal row."""
    t = rows.type[w, r]
    if t == FRICTION_DOF:
        return con.dof_scope[rows.id[w, r]]
    if t != ELLIPTIC:
        return 0
    c = rows.id[w, r]
    if con.efc_address[c, 0] != r or con.dim[c] <= 1:
        return 0
    g = con.geom[c]
    sa = int(0)
    sb = int(0)
    if g[0] >= 0:
        sa = con.geom_scope[g[0]]
    if g[1] >= 0:
        sb = con.geom_scope[g[1]]
    keep = con.all_contacts + ((sa ^ sb) & ROBOT_BIT) + ((sa ^ sb) & SURF_BIT)  # robot-robot contacts stay out
    if keep == 0 and (sa & sb & PAIR_BIT) != 0:
        keep = in_pairs(con, wp.min(g[0], g[1]), wp.max(g[0], g[1]))
    return wp.min(keep, 1)


@wp.func
def in_pairs(con: Contacts, lo: int, hi: int) -> int:
    for p in range(con.pairs.shape[0]):
        if con.pairs[p][0] == lo and con.pairs[p][1] == hi:
            return 1
    return 0


@wp.func
def solve_tree(sc: Scratch, t: int, n: int, w: int, buf: wp.array2d[float], base: int):
    """buf[w, base : base + n] <- M_t^-1 buf[w, ...] by forward and back substitution with the tree factor."""
    for i in range(n):
        v = buf[w, base + i]
        for k in range(i):
            v -= sc.L[w, lidx(sc, t, i, k)] * buf[w, base + k]
        buf[w, base + i] = v / sc.L[w, lidx(sc, t, i, i)]
    for ii in range(n):
        i = n - 1 - ii
        v = buf[w, base + i]
        for k in range(i + 1, n):
            v -= sc.L[w, lidx(sc, t, k, i)] * buf[w, base + k]
        buf[w, base + i] = v / sc.L[w, lidx(sc, t, i, i)]


MAXT8 = 8  # trees up to 8 dofs factor in registers (refresh_tree8)
mat88 = wp.types.matrix(shape=(8, 8), dtype=float)
vec8 = wp.types.vector(length=8, dtype=float)


@wp.func
def chol8(A: mat88, n: int) -> mat88:
    """Lower Cholesky factor of the leading n x n block, static loops (registers)."""
    L = A
    for j in range(8):
        if j < n:
            d = L[j, j]
            for k in range(8):
                if k < j:
                    d -= L[j, k] * L[j, k]
            d = wp.sqrt(wp.max(d, MINVAL))
            L[j, j] = d
            for i in range(8):
                if i > j and i < n:
                    v = L[i, j]
                    for k in range(8):
                        if k < j:
                            v -= L[i, k] * L[j, k]
                    L[i, j] = v / d
    return L


@wp.func
def chol8_solve(L: mat88, b: vec8, n: int) -> vec8:
    """L L' x = b on the leading n entries, static loops (the operations of ``solve_tree``)."""
    x = b
    for i in range(8):
        if i < n:
            v = x[i]
            for k in range(8):
                if k < i:
                    v -= L[i, k] * x[k]
            x[i] = v / L[i, i]
    for ii in range(8):
        i = 7 - ii
        if i < n:
            v = x[i]
            for k in range(8):
                if k > i and k < n:
                    v -= L[k, i] * x[k]
            x[i] = v / L[i, i]
    return x


@wp.func
def refresh_tree8(tr: Trees, t: int, w: int, ms: Mass, sc: Scratch, out: Dofs):
    """Tree t (at most 8 dofs): its factor; with ``sc.restart`` also qacc = qacc_smooth + M^-1 qfrc_constraint and
    Ma = qfrc_smooth + qfrc_constraint (= M qacc) on its dofs, as MuJoCo's noslip starts (registers)."""
    adr = tr.dofadr[t]
    n = tr.dofnum[t]
    A = mat88(0.0)
    for i in range(8):
        if i < n:
            for e in range(ms.rownnz[adr + i]):
                c = ms.colind[ms.rowadr[adr + i] + e] - adr
                for j in range(8):
                    if j == c:
                        A[i, j] = ms.M[w, ms.rowadr[adr + i] + e]
    L = chol8(A, n)
    f = vec8(0.0)
    for i in range(8):
        if i < n:
            f[i] = out.qfrc_constraint[w, adr + i]
            for j in range(8):
                if j <= i:
                    sc.L[w, lidx(sc, t, i, j)] = L[i, j]
    if sc.restart == 0:
        return
    v = chol8_solve(L, f, n)
    for i in range(8):
        if i < n:
            out.qacc[w, adr + i] = ms.qacc_smooth[w, adr + i] + v[i]
            out.Ma[w, adr + i] = ms.qfrc_smooth[w, adr + i] + out.qfrc_constraint[w, adr + i]


@wp.func
def slot_trees(rows: Rows, w: int, r0: int, con: Contacts, tr: Trees):
    """(friction rows, tree A, tree B) of a slot; 0 friction rows = a dof friction row; tree B -1 = one tree."""
    if rows.type[w, r0] == FRICTION_DOF:
        return 0, tr.of_dof[rows.id[w, r0]], -1
    c = rows.id[w, r0]
    g = con.geom[c]
    ta = int(-1)
    tb = int(-1)
    if g[0] >= 0:
        ta = con.body_treeid[con.geom_bodyid[g[0]]]
    if g[1] >= 0:
        tb = con.body_treeid[con.geom_bodyid[g[1]]]
    if tb == ta:
        tb = -1
    if ta < 0:
        ta = tb
        tb = -1
    return con.dim[c] - 1, ta, tb


@wp.func
def copy_row(sc: Scratch, s: int, k: int, row: int, ta: int, tb: int, w: int, rows: Rows, tr: Trees):
    """Local copy of efc row ``row`` into J[s, k] (dofs of trees ta, tb)."""
    n = tree_size(tr, ta) + tree_size(tr, tb)
    for i in range(n):
        sc.J[w, jidx(sc, s, k, i)] = 0.0
    if rows.sparse != 0:
        for e in range(rows.J_rownnz[w, row]):
            sp = rows.J_rowadr[w, row] + e
            dof = rows.J_colind[w, 0, sp]
            i = local_of(tr.of_dof[dof], dof, ta, tb, tr)
            if i >= 0:
                sc.J[w, jidx(sc, s, k, i)] = rows.J[w, 0, sp]
    else:
        for i in range(n):
            sc.J[w, jidx(sc, s, k, i)] = rows.J[w, row, gdof(tr, i, ta, tb)]


@wp.func
def solve_tree8(sc: Scratch, t: int, n: int, w: int, base: int):
    """sc.Y[w, base : base + n] <- M_t^-1 of it for a tree of at most ``MAXT8`` dofs: the factor and the vector in
    registers (independent loads, static loops; ``solve_tree`` chains each load on the previous result)."""
    L = mat88(0.0)
    b = vec8(0.0)
    for i in range(8):
        if i < n:
            b[i] = sc.Y[w, base + i]
            for j in range(8):
                if j <= i:
                    L[i, j] = sc.L[w, lidx(sc, t, i, j)]
    x = chol8_solve(L, b, n)
    for i in range(8):
        if i < n:
            sc.Y[w, base + i] = x[i]


@wp.func
def solve_row(sc: Scratch, s: int, k: int, ta: int, tb: int, w: int, tr: Trees):
    """Y[s, k] = M^-1 J[s, k]' with the factors of trees ta and tb."""
    for i in range(tree_size(tr, ta) + tree_size(tr, tb)):
        sc.Y[w, jidx(sc, s, k, i)] = sc.J[w, jidx(sc, s, k, i)]
    if ta >= 0:
        if tree_size(tr, ta) <= MAXT8:
            solve_tree8(sc, ta, tree_size(tr, ta), w, jidx(sc, s, k, 0))
        else:
            solve_tree(sc, ta, tree_size(tr, ta), w, sc.Y, jidx(sc, s, k, 0))
    if tb >= 0:
        if tree_size(tr, tb) <= MAXT8:
            solve_tree8(sc, tb, tree_size(tr, tb), w, jidx(sc, s, k, tree_size(tr, ta)))
        else:
            solve_tree(sc, tb, tree_size(tr, tb), w, sc.Y, jidx(sc, s, k, tree_size(tr, ta)))


@wp.func
def friction_block(sc: Scratch, s: int, k: int, l: int, n: int, w: int):
    """A[s, k, l] = J[s, k] . Y[s, l] (J M^-1 J' on the slot's friction rows)."""
    a = float(0.0)
    for i in range(n):
        a += sc.J[w, jidx(sc, s, k, i)] * sc.Y[w, jidx(sc, s, l, i)]
    sc.A[w, aidx(sc, s, k, l)] = a


@wp.func
def regularizer_cost(sc: Scratch, s: int, w: int, rows: Rows) -> float:
    """0.5 f^2 R over the slot's rows (MuJoCo adds it to the improvement of the first sweep)."""
    mt = sc.meta[w, s]
    r0 = mt[R0]
    c = float(0.0)
    for k in range(mt[NF] + 1):
        d = rows.D[w, r0 + k]
        if d > 0.0:
            c += 0.5 * rows.force[w, r0 + k] * rows.force[w, r0 + k] / d
    return c
