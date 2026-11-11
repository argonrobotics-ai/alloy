# SPDX-License-Identifier: Apache-2.0
"""Block-sparse Newton factorization for MuJoCo Warp (SCR-WP13-SPARSE-1): the device kernels.

MuJoCo Warp factors the Newton Hessian H = M + J'DJ (+ the elliptic cone term) densely, nv x nv per world. For a
model whose kinematic trees all have at most 6 dofs (free bodies, small articulations) H is block-sparse: one 6x6
block per tree on the diagonal, and an off-diagonal block for every pair of trees a constraint row touches.

Where ``solver_niter == 0`` (the first factorizations of each solve, while the constraint rows are fixed),
``mjw_block`` builds that tree graph from the sparse J and hands it to ``adj`` where it differs from the last
one ordered. There ``symbolic`` (one thread per world) orders it by greedy minimum degree (ties by tree id), records
the fill, the elimination tree's levels and the CSR lists below; elsewhere it keeps them. A world that overflows a cap
is flagged and takes MuJoCo Warp's dense path.

``factor_kernel(dtype)`` (one thread block per world) gathers the blocks from the dense ``ctx.h`` (upper triangle, as
the dense kernel reads it), factors left-looking level by level (diagonal blocks, a barrier, the off-diagonal blocks
of those columns, a barrier) and solves L L' x = grad, writing search = -x, search_dot = x'x and the Newton
decrement grad'x as the dense kernel does. Every block and vector is written by one thread (no float atomics);
pivots below mjMINVAL are raised to it, as mju_cholFactor does, and counted per world (``clamps``, read as
FALLBACK_USED).
"""

import warp as wp                                      # no postponed annotations: the factories' types are locals

B = 6
MINVAL = wp.constant(1.0e-15)                          # mjMINVAL


@wp.func_native(snippet="WP_TILE_SYNC();")
def _sync():
    pass


@wp.func
def _mark(adj: wp.array3d[wp.int32], deg: wp.array2d[wp.int32], w: int, a: int, b: int):
    if a != b and adj[w, a, b] == 0:
        adj[w, a, b] = 1
        adj[w, b, a] = 1
        deg[w, a] = deg[w, a] + 1
        deg[w, b] = deg[w, b] + 1


@wp.func
def _degrees(T: int, w: int, adj: wp.array3d[wp.int32], deg: wp.array2d[wp.int32], left: wp.array2d[wp.int32]):
    for a in range(T):
        n = int(0)
        for b in range(T):
            if adj[w, a, b] != 0:
                n += 1
        deg[w, a] = n
        left[w, a] = 1


@wp.func
def _min_degree(T: int, w: int, deg: wp.array2d[wp.int32], left: wp.array2d[wp.int32]) -> int:
    best = int(-1)
    bdeg = int(2147483647)
    for t in range(T):                                 # ties by tree id
        if left[w, t] != 0 and deg[w, t] < bdeg:
            best = t
            bdeg = deg[w, t]
    return best


@wp.func
def _neighbours(T: int, w: int, v: int, adj: wp.array3d[wp.int32], deg: wp.array2d[wp.int32],
                left: wp.array2d[wp.int32], nbuf: wp.array2d[wp.int32]) -> int:
    """The trees still left that ``v`` (just eliminated) touches, into ``nbuf``; their degrees drop by one."""
    nn = int(0)
    for u in range(T):
        if left[w, u] != 0 and adj[w, v, u] != 0:
            nbuf[w, nn] = u
            nn += 1
            deg[w, u] = deg[w, u] - 1
    return nn


@wp.kernel
def symbolic(ntree: int, off_cap: int, pair_cap: int, solver_niter: wp.array[int], changed: wp.array[wp.int32],
             adj: wp.array3d[wp.int32],
             deg: wp.array2d[wp.int32], left: wp.array2d[wp.int32], order: wp.array2d[wp.int32],
             pos: wp.array2d[wp.int32], nbuf: wp.array2d[wp.int32],
             col_ptr: wp.array2d[wp.int32], col_row: wp.array2d[wp.int32], row_ptr: wp.array2d[wp.int32],
             row_blk: wp.array2d[wp.int32], off_row: wp.array2d[wp.int32], off_col: wp.array2d[wp.int32],
             level: wp.array2d[wp.int32], nlev: wp.array[wp.int32], lvl_node: wp.array2d[wp.int32],
             node_item: wp.array2d[wp.int32], lvl_off: wp.array2d[wp.int32], off_item: wp.array2d[wp.int32],
             opair_ptr: wp.array2d[wp.int32], opair_a: wp.array2d[wp.int32], opair_b: wp.array2d[wp.int32],
             overflow: wp.array[wp.int32]):
    w = wp.tid()
    if solver_niter[w] != 0 or changed[w] == 0:        # the same graph as last ordered: every list still holds
        return
    T = ntree
    overflow[w] = 0
    _degrees(T, w, adj, deg, left)                     # the graph: mjw_block's graph kernels (before)
    n_off = int(0)
    for s in range(T):                                 # greedy minimum degree; fill joins the neighbours
        best = _min_degree(T, w, deg, left)
        order[w, s] = best
        pos[w, best] = s
        left[w, best] = 0
        nn = _neighbours(T, w, best, adj, deg, left, nbuf)
        col_ptr[w, s] = n_off
        if n_off + nn > off_cap:
            overflow[w] = 1
            return
        for k in range(nn):
            col_row[w, n_off + k] = nbuf[w, k]          # tree ids for now; nodes after the order is complete
            for k2 in range(k + 1, nn):
                _mark(adj, deg, w, nbuf[w, k], nbuf[w, k2])
        n_off += nn
    col_ptr[w, T] = n_off
    _finish(T, pair_cap, w, adj, pos, order, nbuf, col_ptr, col_row, row_ptr, row_blk, off_row, off_col, level,
            nlev, lvl_node, node_item, lvl_off, off_item, opair_ptr, opair_a, opair_b, overflow)


@wp.func
def _finish(T: int, pair_cap: int, w: int, adj: wp.array3d[wp.int32], pos: wp.array2d[wp.int32],
            order: wp.array2d[wp.int32], cursor: wp.array2d[wp.int32], col_ptr: wp.array2d[wp.int32],
            col_row: wp.array2d[wp.int32],
            row_ptr: wp.array2d[wp.int32], row_blk: wp.array2d[wp.int32], off_row: wp.array2d[wp.int32],
            off_col: wp.array2d[wp.int32], level: wp.array2d[wp.int32], nlev: wp.array[wp.int32],
            lvl_node: wp.array2d[wp.int32], node_item: wp.array2d[wp.int32], lvl_off: wp.array2d[wp.int32],
            off_item: wp.array2d[wp.int32], opair_ptr: wp.array2d[wp.int32], opair_a: wp.array2d[wp.int32],
            opair_b: wp.array2d[wp.int32], overflow: wp.array[wp.int32]):
    """Columns in node numbering (sorted), block ids in adj, levels, level-sorted items, row CSR and update pairs."""
    _columns(T, w, adj, pos, order, col_ptr, col_row, row_ptr, off_row, off_col, level)
    _row_csr(T, w, cursor, col_ptr, col_row, row_ptr, row_blk)
    _level_lists(T, w, col_ptr, level, nlev, lvl_node, node_item, lvl_off, off_item)
    _update_pairs(w, pair_cap, col_ptr[w, T], adj, order, off_row, off_col, row_ptr, row_blk, opair_ptr, opair_a,
                  opair_b, overflow)


@wp.func
def _columns(T: int, w: int, adj: wp.array3d[wp.int32], pos: wp.array2d[wp.int32], order: wp.array2d[wp.int32],
             col_ptr: wp.array2d[wp.int32], col_row: wp.array2d[wp.int32], row_ptr: wp.array2d[wp.int32],
             off_row: wp.array2d[wp.int32], off_col: wp.array2d[wp.int32], level: wp.array2d[wp.int32]):
    """Each column's rows as sorted node ids; block e = (i, j) noted in adj as e + 2; row counts; etree levels."""
    for j in range(T):
        level[w, j] = 0
        row_ptr[w, j] = 0
    for j in range(T):
        a0 = col_ptr[w, j]
        a1 = col_ptr[w, j + 1]
        for e in range(a0, a1):
            col_row[w, e] = pos[w, col_row[w, e]]
        for e in range(a0 + 1, a1):                    # insertion sort: short lists
            v = col_row[w, e]
            f = e - 1
            while f >= a0 and col_row[w, f] > v:
                col_row[w, f + 1] = col_row[w, f]
                f -= 1
            col_row[w, f + 1] = v
        for e in range(a0, a1):
            i = col_row[w, e]
            off_row[w, e] = i
            off_col[w, e] = j
            adj[w, order[w, i], order[w, j]] = e + 2   # 0 none, 1 structure, e + 2 block e
            adj[w, order[w, j], order[w, i]] = e + 2
            row_ptr[w, i] = row_ptr[w, i] + 1
        if a1 > a0:                                    # the parent is the first row: one level above
            p = col_row[w, a0]
            level[w, p] = wp.max(level[w, p], level[w, j] + 1)


@wp.func
def _row_csr(T: int, w: int, cursor: wp.array2d[wp.int32], col_ptr: wp.array2d[wp.int32],
             col_row: wp.array2d[wp.int32], row_ptr: wp.array2d[wp.int32], row_blk: wp.array2d[wp.int32]):
    """The transpose of the columns: each row's blocks, sorted by column."""
    acc = int(0)
    for i in range(T):
        c = row_ptr[w, i]
        row_ptr[w, i] = acc
        acc += c
    row_ptr[w, T] = acc
    for i in range(T):
        cursor[w, i] = row_ptr[w, i]
    for j in range(T):
        for e in range(col_ptr[w, j], col_ptr[w, j + 1]):
            i = col_row[w, e]
            row_blk[w, cursor[w, i]] = e
            cursor[w, i] = cursor[w, i] + 1


@wp.func
def _level_lists(T: int, w: int, col_ptr: wp.array2d[wp.int32], level: wp.array2d[wp.int32],
                 nlev: wp.array[wp.int32], lvl_node: wp.array2d[wp.int32], node_item: wp.array2d[wp.int32],
                 lvl_off: wp.array2d[wp.int32], off_item: wp.array2d[wp.int32]):
    """Level-sorted nodes and off-diagonal blocks (a block belongs to its column's level)."""
    nl = int(0)
    for j in range(T):
        nl = wp.max(nl, level[w, j] + 1)
    nlev[w] = nl
    k = int(0)
    ko = int(0)
    for lv in range(nl):
        lvl_node[w, lv] = k
        lvl_off[w, lv] = ko
        for j in range(T):
            if level[w, j] == lv:
                node_item[w, k] = j
                k += 1
                for e in range(col_ptr[w, j], col_ptr[w, j + 1]):
                    off_item[w, ko] = e
                    ko += 1
    lvl_node[w, nl] = k
    lvl_off[w, nl] = ko


@wp.func
def _update_pairs(w: int, pair_cap: int, n_off: int, adj: wp.array3d[wp.int32], order: wp.array2d[wp.int32],
                  off_row: wp.array2d[wp.int32], off_col: wp.array2d[wp.int32], row_ptr: wp.array2d[wp.int32],
                  row_blk: wp.array2d[wp.int32], opair_ptr: wp.array2d[wp.int32], opair_a: wp.array2d[wp.int32],
                  opair_b: wp.array2d[wp.int32], overflow: wp.array[wp.int32]):
    """For block e = (i, j): the pairs (L_ik, L_jk) of its update S_ij = A_ij - sum_k L_ik L_jk'."""
    npair = int(0)
    for e in range(n_off):
        opair_ptr[w, e] = npair
        i = off_row[w, e]
        j = off_col[w, e]
        for q in range(row_ptr[w, j], row_ptr[w, j + 1]):
            kb = row_blk[w, q]
            ik = adj[w, order[w, i], order[w, off_col[w, kb]]] - 2
            if ik >= 0:
                if npair >= pair_cap:
                    overflow[w] = 1
                    return
                opair_a[w, npair] = ik
                opair_b[w, npair] = kb
                npair += 1
    opair_ptr[w, n_off] = npair


def _block_funcs(dtype):
    """6x6 Cholesky with MuJoCo C's pivot clamp, the inverse of a lower factor, and H's upper-triangle read."""
    F64 = dtype == wp.float64
    M = wp.types.matrix(shape=(B, B), dtype=dtype)

    @wp.func
    def chol(a: M, w: int, bad: wp.array[wp.int32]) -> M:
        lo = M()
        for j in range(B):
            s = a[j, j]
            for k in range(j):
                s -= lo[j, k] * lo[j, k]
            if not (s >= dtype(MINVAL)):               # also NaN
                if wp.static(not F64):
                    bad[w] = 1                          # float32: flag the world for the float64 redo
                s = dtype(MINVAL)                       # mju_cholFactor: the pivot is raised to mjMINVAL
            dj = wp.sqrt(s)
            lo[j, j] = dj
            for i in range(j + 1, B):
                t = a[i, j]
                for k in range(j):
                    t -= lo[i, k] * lo[j, k]
                lo[i, j] = t / dj
        return lo

    @wp.func
    def inv_lower(lo: M) -> M:
        out = M()
        for j in range(B):
            out[j, j] = dtype(1.0) / lo[j, j]
            for i in range(j + 1, B):
                t = dtype(0.0)
                for k in range(j, i):
                    t -= lo[i, k] * out[k, j]
                out[i, j] = t / lo[i, i]
        return out

    @wp.func
    def h_at(h: wp.array3d[float], w: int, r: int, c: int) -> dtype:
        if r <= c:                                     # MuJoCo Warp fills and reads the upper triangle
            return dtype(h[w, r, c])
        return dtype(h[w, c, r])

    return M, wp.types.vector(B, dtype), chol, inv_lower, h_at


def _gather_func(dtype, M, V, h_at):
    @wp.func
    def gather(w: int, lane: int, bd: int, T: int, h: wp.array3d[float], grad: wp.array2d[float],
               order: wp.array2d[wp.int32], tree_dofadr: wp.array[int], tree_dofnum: wp.array[int],
               col_ptr: wp.array2d[wp.int32], off_row: wp.array2d[wp.int32], off_col: wp.array2d[wp.int32],
               a_diag: wp.array2d[M], lo: wp.array2d[M], g: wp.array2d[V]):
        """Diagonal blocks and the gradient (a tree under 6 dofs padded with identity), then the off-diagonal
        blocks (fill blocks gather zero)."""
        for j in range(lane, T, bd):
            t = order[w, j]
            a0 = tree_dofadr[t]
            n = tree_dofnum[t]
            blk = M()
            gv = V()
            for r in range(B):
                for c in range(B):
                    if r < n and c < n:
                        blk[r, c] = h_at(h, w, a0 + r, a0 + c)
                    elif r == c:
                        blk[r, c] = dtype(1.0)
                if r < n:
                    gv[r] = dtype(grad[w, a0 + r])
            a_diag[w, j] = blk
            g[w, j] = gv
        for e in range(lane, col_ptr[w, T], bd):
            ti = order[w, off_row[w, e]]
            tj = order[w, off_col[w, e]]
            ai = tree_dofadr[ti]
            aj = tree_dofadr[tj]
            blk = M()
            for r in range(tree_dofnum[ti]):
                for c in range(tree_dofnum[tj]):
                    blk[r, c] = h_at(h, w, ai + r, aj + c)
            lo[w, e] = blk

    return gather


def _factor_func(M, chol, inv_lower):
    @wp.func
    def factor(w: int, lane: int, bd: int, bad: wp.array[wp.int32], nlev: wp.array[wp.int32],
               lvl_node: wp.array2d[wp.int32], node_item: wp.array2d[wp.int32], lvl_off: wp.array2d[wp.int32],
               off_item: wp.array2d[wp.int32], off_col: wp.array2d[wp.int32], row_ptr: wp.array2d[wp.int32],
               row_blk: wp.array2d[wp.int32], opair_ptr: wp.array2d[wp.int32], opair_a: wp.array2d[wp.int32],
               opair_b: wp.array2d[wp.int32], a_diag: wp.array2d[M], lo: wp.array2d[M], dinv: wp.array2d[M]):
        """Left-looking, level by level: S_jj = A_jj - sum_k L_jk L_jk', Dinv_j = chol(S_jj)^-1; a barrier;
        L_ij = (A_ij - sum_k L_ik L_jk') Dinv_j'; a barrier. Every input of a block is on a lower level."""
        for lv in range(nlev[w]):
            for it in range(lvl_node[w, lv] + lane, lvl_node[w, lv + 1], bd):
                j = node_item[w, it]
                s = a_diag[w, j]
                for q in range(row_ptr[w, j], row_ptr[w, j + 1]):
                    ljk = lo[w, row_blk[w, q]]
                    s = s - ljk * wp.transpose(ljk)
                dinv[w, j] = inv_lower(chol(s, w, bad))
            _sync()
            for it in range(lvl_off[w, lv] + lane, lvl_off[w, lv + 1], bd):
                e = off_item[w, it]
                s = lo[w, e]
                for q in range(opair_ptr[w, e], opair_ptr[w, e + 1]):
                    s = s - lo[w, opair_a[w, q]] * wp.transpose(lo[w, opair_b[w, q]])
                lo[w, e] = s * wp.transpose(dinv[w, off_col[w, e]])
            _sync()

    return factor


def _solve_funcs(M, V):
    @wp.func
    def forward(w: int, lane: int, bd: int, nlev: wp.array[wp.int32], lvl_node: wp.array2d[wp.int32],
                node_item: wp.array2d[wp.int32], row_ptr: wp.array2d[wp.int32], row_blk: wp.array2d[wp.int32],
                off_col: wp.array2d[wp.int32], lo: wp.array2d[M], dinv: wp.array2d[M], g: wp.array2d[V],
                y: wp.array2d[V]):
        """L y = g by ascending level."""
        for lv in range(nlev[w]):
            for it in range(lvl_node[w, lv] + lane, lvl_node[w, lv + 1], bd):
                j = node_item[w, it]
                rv = g[w, j]
                for q in range(row_ptr[w, j], row_ptr[w, j + 1]):
                    e = row_blk[w, q]
                    rv = rv - lo[w, e] * y[w, off_col[w, e]]
                y[w, j] = dinv[w, j] * rv
            _sync()

    @wp.func
    def backward(w: int, lane: int, bd: int, nlev: wp.array[wp.int32], lvl_node: wp.array2d[wp.int32],
                 node_item: wp.array2d[wp.int32], col_ptr: wp.array2d[wp.int32], off_row: wp.array2d[wp.int32],
                 lo: wp.array2d[M], dinv: wp.array2d[M], y: wp.array2d[V], x: wp.array2d[V]):
        """L' x = y by descending level."""
        for q2 in range(nlev[w]):
            lv = nlev[w] - 1 - q2
            for it in range(lvl_node[w, lv] + lane, lvl_node[w, lv + 1], bd):
                j = node_item[w, it]
                rv = y[w, j]
                for e in range(col_ptr[w, j], col_ptr[w, j + 1]):
                    rv = rv - wp.transpose(lo[w, e]) * x[w, off_row[w, e]]
                x[w, j] = wp.transpose(dinv[w, j]) * rv
            _sync()

    return forward, backward


def _output_func(dtype, V):
    F64 = dtype == wp.float64

    @wp.func
    def output(w: int, lane: int, bd: int, T: int, order: wp.array2d[wp.int32], tree_dofadr: wp.array[int],
               tree_dofnum: wp.array[int], g: wp.array2d[V], x: wp.array2d[V], bad: wp.array[wp.int32],
               clamps: wp.array[wp.int32], search: wp.array2d[float], search_dot: wp.array[float],
               decrement: wp.array[float]):
        """search = -x on each tree's real dofs; lane 0: x'x, g'x, and the clamp count (float64) or the non-finite
        flag (float32)."""
        for j in range(lane, T, bd):
            t = order[w, j]
            xv = x[w, j]
            for r in range(tree_dofnum[t]):
                search[w, tree_dofadr[t] + r] = -float(xv[r])
        _sync()
        if lane == 0:
            sxx = dtype(0.0)
            sgx = dtype(0.0)
            for j in range(T):
                sxx += wp.dot(x[w, j], x[w, j])
                sgx += wp.dot(g[w, j], x[w, j])
            search_dot[w] = float(sxx)
            decrement[w] = float(sgx)
            if wp.static(F64):
                clamps[w] = clamps[w] + 1              # one per clamped factorization (FALLBACK_USED)
            if wp.static(not F64):
                if not (wp.isfinite(sxx) and wp.isfinite(sgx)):
                    bad[w] = 1

    return output


def factor_kernel(dtype):
    """The factor + solve kernel in ``dtype`` (float32 first; float64 redoes the worlds float32 flagged)."""
    F64 = dtype == wp.float64
    M, V, chol, inv_lower, h_at = _block_funcs(dtype)
    gather = _gather_func(dtype, M, V, h_at)
    factor = _factor_func(M, chol, inv_lower)
    forward, backward = _solve_funcs(M, V)
    output = _output_func(dtype, V)

    @wp.kernel(module="unique", enable_backward=False)
    def kernel(ntree: int, done: wp.array[bool], overflow: wp.array[wp.int32], bad: wp.array[wp.int32],
               clamps: wp.array[wp.int32], h: wp.array3d[float], grad: wp.array2d[float],
               order: wp.array2d[wp.int32], tree_dofadr: wp.array[int], tree_dofnum: wp.array[int],
               nlev: wp.array[wp.int32], lvl_node: wp.array2d[wp.int32], node_item: wp.array2d[wp.int32],
               lvl_off: wp.array2d[wp.int32], off_item: wp.array2d[wp.int32], col_ptr: wp.array2d[wp.int32],
               off_row: wp.array2d[wp.int32], off_col: wp.array2d[wp.int32], row_ptr: wp.array2d[wp.int32],
               row_blk: wp.array2d[wp.int32], opair_ptr: wp.array2d[wp.int32], opair_a: wp.array2d[wp.int32],
               opair_b: wp.array2d[wp.int32], a_diag: wp.array2d[M], lo: wp.array2d[M], dinv: wp.array2d[M],
               g: wp.array2d[V], y: wp.array2d[V], x: wp.array2d[V],
               search: wp.array2d[float], search_dot: wp.array[float], decrement: wp.array[float]):
        w, lane = wp.tid()
        if done[w] or overflow[w] != 0:
            return
        if wp.static(F64):
            if bad[w] == 0:
                return
        else:
            if lane == 0:
                bad[w] = 0                              # before the first barrier: every flag comes after it
        bd = wp.block_dim()
        gather(w, lane, bd, ntree, h, grad, order, tree_dofadr, tree_dofnum, col_ptr, off_row, off_col, a_diag, lo,
               g)
        _sync()
        factor(w, lane, bd, bad, nlev, lvl_node, node_item, lvl_off, off_item, off_col, row_ptr, row_blk,
               opair_ptr, opair_a, opair_b, a_diag, lo, dinv)
        forward(w, lane, bd, nlev, lvl_node, node_item, row_ptr, row_blk, off_col, lo, dinv, g, y)
        backward(w, lane, bd, nlev, lvl_node, node_item, col_ptr, off_row, lo, dinv, y, x)
        output(w, lane, bd, ntree, order, tree_dofadr, tree_dofnum, g, x, bad, clamps, search, search_dot,
               decrement)

    return kernel, M, V
