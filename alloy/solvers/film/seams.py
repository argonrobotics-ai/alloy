# SPDX-License-Identifier: Apache-2.0
"""Seams: tear lines of a film, held closed by its strain projection, with the force each one carries.

A sheet that can tear is pre-split by the topology package (``alloy.topology.emit.add_presplit_cloth``): the
vertices on its tear lines are duplicated, each twin pair a ``CLOTH_EDGE`` bond with a stiff zero-length VBD spring.
The film's in-plane stiffness is its strain projection (design note §3), so a twin pair also joins the projection as
a long-range pair at rest length 0, and the long-range limits are built on the welded sheet (seam copies mapped to one
copy), each acting while the seams it crosses are intact (``pair_gates``; ``topology.build_topology``): an intact
seam is as inextensible as the sheet. VBD's side of it (colours, pieces) is in seam_vbd.py.

The force a seam carries, on its a side: its twin spring's (ke·gap left by VBD's solve) plus the projection's
impulses across it: its own pair's and its share of every long-range pair crossing it (``set_crossings``; momentum
Jacobi applies equal and opposite corrections m·δ to a pair's two ends; the hard-constraint force of position based
dynamics is λ / dt²). A hanging strip split across its middle: 5.68 mN read against 5.47 mN of weight below the
seam (test_paper_tear.py).

Each step the solver publishes, per topology bond, that force and the force the seam breaks at: the sheet's strength
over the seam's length (``topology:bond_break_force``, failure.fracture_stress·h·length) for an interior seam, the
tear force ``film:tear_force`` (Γ·h/2, Elmendorf's trouser tear) at a crack tip (a neighbouring seam along its line
is broken) or for the out-of-plane part of the force at a free edge. The topology controller decides and commits the
breaks (``alloy.topology``; its strain test uses config.SEAM_OVERLOAD_STRAIN); a broken seam leaves the projection.
The projection resolves these forces only where it converges: a light sheet pulled by about a newton is not (notes in
docs/notes/paper-pack.md); there the strain test tears it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

CLOTH_EDGE = 3                       # alloy.topology.types.BondKind.CLOTH_EDGE (read-only)


@dataclass
class Seams:
    """Host-side seam table (one row per CLOTH_EDGE twin pair between film particles)."""

    a: np.ndarray                    # [S] int32 particles
    b: np.ndarray
    bond: np.ndarray                 # [S] int32 topology bond ids
    spring: np.ndarray               # [S] int32 the twin's VBD spring (zeroed on a break)
    pair: np.ndarray                 # [S] int32 the seam's long-range pair id (set by build_topology)
    edge: np.ndarray                 # [S] int32 1 = at a free edge of the sheet
    tri: np.ndarray                  # [S] int32 a triangle at the seam's a end (the sheet's normal there)
    nbr_start: np.ndarray            # [S+1] int32 CSR of seams along the same line (sharing a mesh edge)
    nbr: np.ndarray
    break_force: np.ndarray          # [S] float32 N, interior
    tear_force: np.ndarray           # [S] float32 N, at a crack tip or free edge
    n_bonds: int                     # rows of the topology bond table
    rep: np.ndarray                  # [particles] int64: each seam copy -> its vertex's first copy (identity elsewhere)
    across: np.ndarray               # [S, 3] at rest, from the b copy's triangles towards the a copy's (side test)
    world: np.ndarray                # [S] int32 the seam's world (-1: global)
    world_start: np.ndarray          # [K+1] int32 CSR over ``by_world``: the seams of each world (global ones last)
    by_world: np.ndarray             # [S] int32 seam ids grouped by world
    cross_start: np.ndarray | None = None    # [S+1] int32 CSR: the long-range pairs carrying force across the seam
    cross_pair: np.ndarray | None = None     # int32 pair ids (the seam's own rest-0 pair first)
    cross_side: np.ndarray | None = None     # int32 which end of the pair (0, 1) is on the seam's a side
    cross_weight: np.ndarray | None = None   # float32 share of the pair's force (1 / seams it crosses)


def _topology(model):
    return getattr(model, "topology", None)


def find_seams(model, particles: np.ndarray) -> Seams | None:
    """The CLOTH_EDGE twins of ``model`` between film ``particles``; None when the model has none."""
    tp = _topology(model)
    if tp is None or getattr(tp, "bond_kind", None) is None or tp.bond_kind.shape[0] == 0:
        return None
    kind, a, b = tp.bond_kind.numpy(), tp.bond_particle_a.numpy(), tp.bond_particle_b.numpy()
    film = np.zeros(int(model.particle_count), bool)
    film[particles] = True
    sel = np.flatnonzero((kind == CLOTH_EDGE) & (a >= 0) & (b >= 0))
    sel = sel[film[a[sel]] & film[b[sel]]]
    if not len(sel):
        return None
    A, B = a[sel].astype(np.int32), b[sel].astype(np.int32)
    root = _union(A, B)
    rep = np.arange(int(model.particle_count), dtype=np.int64)
    for v in set(A.tolist()) | set(B.tolist()):
        rep[v] = root(v)
    tri = rep[model.tri_indices.numpy().astype(np.int64)]                  # the sheet with its seams welded shut
    edges = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    uniq, count = np.unique(edges, axis=0, return_counts=True)
    free = np.zeros(int(model.particle_count), bool)
    free[uniq[count == 1].reshape(-1)] = True                              # the welded sheet's free edges
    by_rep: dict[int, list[int]] = {}
    for s, u in enumerate(A):
        by_rep.setdefault(int(rep[u]), []).append(s)
    adj: dict[int, set[int]] = {}
    for u, v in uniq:
        if int(u) in by_rep and int(v) in by_rep:
            adj.setdefault(int(u), set()).add(int(v))
            adj.setdefault(int(v), set()).add(int(u))
    nbrs = []
    for s, u in enumerate(A):
        r = int(rep[u])
        near = [t for t in by_rep[r] if t != s] + [t for q in sorted(adj.get(r, ())) for t in by_rep[q]]
        nbrs.append(near)
    nbr_start = np.zeros(len(sel) + 1, np.int32)
    nbr_start[1:] = np.cumsum([len(n) for n in nbrs])
    bf = tp.bond_break_force.numpy()[sel].astype(np.float32)
    tf = getattr(getattr(model, "film", None), "tear_force", None)
    tear = np.minimum(tf.numpy()[A], tf.numpy()[B]) if tf is not None else np.full(len(sel), np.inf)
    tear = tear.astype(np.float32)
    across = _across(model.particle_q.numpy().astype(np.float64), ind := model.tri_indices.numpy().astype(np.int64),
                     A, B)
    tri_of = np.full(int(model.particle_count), -1, np.int64)
    tri_of[ind[::-1].reshape(-1)] = np.repeat(np.arange(len(ind))[::-1], 3)      # the first triangle of each vertex
    return Seams(a=A, b=B, bond=sel.astype(np.int32), spring=tp.bond_spring.numpy()[sel].astype(np.int32),
                 pair=np.full(len(sel), -1, np.int32),
                 edge=free[rep[A]].astype(np.int32), tri=tri_of[A].astype(np.int32), nbr_start=nbr_start,
                 nbr=np.asarray([t for n in nbrs for t in n], np.int32), break_force=bf, tear_force=tear,
                 n_bonds=int(kind.shape[0]), rep=rep, across=across, **_world(model, A))


def _world(model, A: np.ndarray) -> dict[str, np.ndarray]:
    """Each seam's world and the seams grouped by world (global seams, world -1, in a last group)."""
    pw = getattr(model, "particle_world", None)
    world = (pw.numpy()[A] if pw is not None else np.zeros(len(A))).astype(np.int32)
    slot = np.where(world < 0, int(world.max(initial=0)) + 1, world)
    order = np.argsort(slot, kind="stable")
    start = np.searchsorted(slot[order], np.arange(int(slot.max(initial=0)) + 2)).astype(np.int32)
    return {"world": world, "world_start": start, "by_world": order.astype(np.int32)}


def _across(x: np.ndarray, ind: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Per seam, the unit direction from its b copy's triangles to its a copy's (centroid means at rest)."""
    cen = x[ind].mean(axis=1)
    flat = ind.reshape(-1)
    acc = np.zeros((len(x), 3))
    np.add.at(acc, flat, np.repeat(cen, 3, axis=0))
    cnt = np.bincount(flat, minlength=len(x))
    out = np.zeros((len(A), 3))
    ok = (cnt[A] > 0) & (cnt[B] > 0)
    out[ok] = acc[A[ok]] / cnt[A[ok], None] - acc[B[ok]] / cnt[B[ok], None]
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return np.divide(out, n, out=np.zeros_like(out), where=n > 0.0)


def pair_gates(pairs: np.ndarray, x: np.ndarray, seams: Seams | None, h: float,
               world: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Long-range pairs (built on the welded sheet) that cross a seam line: a seam within 0.6·h of the pair's rest
    segment (twins are an edge apart along a line, so a crossing passes within h/2 of one) with the pair's ends on
    its two sides (``across``; an end on the seam itself is on its a side, where the welded sheet's copy is). A pair
    acts only while all the seams it crosses are intact (longrange.pair_corrections), and its force counts towards
    theirs, shared equally (``set_crossings``). Returns the gate CSR [C+1], [G] (bond ids) and the crossings as
    arrays (pair, seam, side = the pair's end on the seam's a side, weight), ordered by pair then seam. Per world
    (``world``: particle world ids), once per distinct world geometry (replicas share one table)."""
    start = np.zeros(len(pairs) + 1, np.int32)
    none = {"pair": np.zeros(0, np.int64), "seam": np.zeros(0, np.int64), "side": np.zeros(0, np.int32),
            "weight": np.zeros(0, np.float32)}
    if seams is None or not len(pairs):
        return start, np.zeros(0, np.int32), none
    P = np.asarray(pairs, np.int64).reshape(-1, 2)
    pw = world[P[:, 0]] if world is not None else np.zeros(len(P), np.int64)
    out: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    cache: dict[bytes, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for w in np.unique(pw):
        cs, ks = np.flatnonzero(pw == w), np.flatnonzero(seams.world == w)
        if not len(ks):
            continue
        o = int(min(P[cs].min(), seams.a[ks].min()))
        key = ((P[cs] - o).tobytes() + (seams.a[ks] - o).tobytes() + np.round(x[P[cs]] - x[o], 7).tobytes()
               + np.round(x[seams.a[ks]] - x[o], 7).tobytes() + np.round(seams.across[ks], 7).tobytes())
        if key not in cache:                     # identical worlds (replicas) share one crossing table
            cache[key] = _world_crossings(P[cs], x, seams.a[ks], seams.across[ks], h)
        c, k, side = cache[key]
        out.append((cs[c], ks[k], side))
    ci = np.concatenate([np.zeros(0, np.int64)] + [r[0] for r in out])
    ki = np.concatenate([np.zeros(0, np.int64)] + [r[1] for r in out])
    side = np.concatenate([np.zeros(0, np.int32)] + [r[2] for r in out])
    order = np.lexsort((ki, ci))
    ci, ki, side = ci[order], ki[order], side[order]
    count = np.bincount(ci, minlength=len(P))
    start[1:] = np.cumsum(count)
    cross = {"pair": ci, "seam": ki, "side": side, "weight": (1.0 / count[ci]).astype(np.float32)}
    return start, seams.bond[ki].astype(np.int32), cross


def _world_crossings(P: np.ndarray, x: np.ndarray, sa: np.ndarray, across: np.ndarray, h: float):
    """One world's crossings: (pair row, seam row, side) for pairs ``P`` and seams at ``sa`` (k-d tree candidates
    within the bounding ball of each pair's segment, then the exact distance and side tests)."""
    from scipy.spatial import cKDTree

    a, b = x[P[:, 0]], x[P[:, 1]]
    hits = cKDTree(x[sa]).query_ball_point(0.5 * (a + b), 0.5 * np.linalg.norm(b - a, axis=1) + 0.6 * h)
    n = np.fromiter((len(c) for c in hits), np.int64, len(hits))
    ci = np.repeat(np.arange(len(P)), n)
    ki = np.concatenate([np.zeros(0, np.int64)] + [np.asarray(c, np.int64) for c in hits if len(c)])
    a, b, p, d_s = a[ci], b[ci], x[sa[ki]], across[ki]
    d = b - a
    dd = np.einsum("kj,kj->k", d, d)
    t = np.clip(np.einsum("kj,kj->k", p - a, d) / np.where(dd > 0.0, dd, 1.0), 0.0, 1.0) * (dd > 0.0)
    near = np.linalg.norm(a + t[:, None] * d - p, axis=1) <= 0.6 * h
    si, sj = np.einsum("kj,kj->k", a - p, d_s), np.einsum("kj,kj->k", b - p, d_s)
    tol = 1e-6 * h
    hit = near & (np.minimum(si, sj) < -tol) & (np.maximum(si, sj) >= -tol)
    return ci[hit], ki[hit], (si[hit] <= sj[hit]).astype(np.int32)   # side 0: end 0 further on the a side


def set_crossings(s: Seams, cross: dict[str, np.ndarray]) -> None:
    """The seam-major CSR of the pairs carrying force across each seam: its own rest-0 pair (a = end 0), then the
    long-range pairs crossing it in pair order (deterministic sums)."""
    S = len(s.a)
    seam = np.concatenate([np.arange(S), cross["seam"]])
    group = np.concatenate([np.zeros(S, np.int64), np.ones(len(cross["seam"]), np.int64)])
    pair = np.concatenate([s.pair.astype(np.int64), cross["pair"]])
    order = np.lexsort((pair, group, seam))
    s.cross_start = np.searchsorted(seam[order], np.arange(S + 1)).astype(np.int32)
    s.cross_pair = pair[order].astype(np.int32)
    s.cross_side = np.concatenate([np.zeros(S, np.int32), cross["side"]])[order].astype(np.int32)
    s.cross_weight = np.concatenate([np.ones(S, np.float32), cross["weight"]])[order].astype(np.float32)


def _union(A: np.ndarray, B: np.ndarray):
    """Union-find over twin pairs: root(v) = the smallest copy of v's vertex."""
    parent: dict[int, int] = {}

    def root(x: int) -> int:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in zip(A.tolist(), B.tolist()):
        ru, rv = root(u), root(v)
        if ru != rv:
            parent[max(ru, rv)] = min(ru, rv)
    return root


@wp.struct
class SeamArrays:
    a: wp.array[wp.int32]
    b: wp.array[wp.int32]
    bond: wp.array[wp.int32]
    spring: wp.array[wp.int32]
    pair: wp.array[wp.int32]
    edge: wp.array[wp.int32]
    tri: wp.array[wp.int32]
    nbr_start: wp.array[wp.int32]
    nbr: wp.array[wp.int32]
    break_force: wp.array[float]
    tear_force: wp.array[float]
    cross_start: wp.array[wp.int32]
    cross_pair: wp.array[wp.int32]
    cross_side: wp.array[wp.int32]
    cross_weight: wp.array[float]
    world: wp.array[wp.int32]
    world_start: wp.array[wp.int32]
    by_world: wp.array[wp.int32]
    impulse: wp.array[wp.vec3]       # Σ m_a·shift_a this step (kg·m)
    f_avg: wp.array[wp.vec3]         # N, the force averaged over FilmConfig.seam_force_tau


def upload(s: Seams, device) -> SeamArrays:
    out = SeamArrays()
    i32 = lambda x: wp.array(np.ascontiguousarray(x, np.int32), dtype=wp.int32, device=device)  # noqa: E731
    f32 = lambda x: wp.array(np.ascontiguousarray(x, np.float32), dtype=float, device=device)  # noqa: E731
    out.a, out.b, out.bond, out.pair, out.edge = i32(s.a), i32(s.b), i32(s.bond), i32(s.pair), i32(s.edge)
    out.tri = i32(s.tri)
    out.spring = i32(s.spring)
    out.nbr_start, out.nbr = i32(s.nbr_start), i32(s.nbr if len(s.nbr) else np.zeros(1))
    out.break_force, out.tear_force = f32(s.break_force), f32(s.tear_force)
    out.cross_start, out.cross_pair = i32(s.cross_start), i32(s.cross_pair)
    out.cross_side, out.cross_weight = i32(s.cross_side), f32(s.cross_weight)
    out.world, out.world_start, out.by_world = i32(s.world), i32(s.world_start), i32(s.by_world)
    out.impulse = wp.zeros(len(s.a), dtype=wp.vec3, device=device)
    out.f_avg = wp.zeros(len(s.a), dtype=wp.vec3, device=device)
    return out


@wp.func
def _intact(intact: wp.array[wp.int8], bond: int) -> bool:
    return intact[bond] != wp.int8(0)


@wp.func
def _impulse(ma: float, mb: float, wa: float, shift_a: wp.vec3, shift_b: wp.vec3) -> wp.vec3:
    """The seam's impulse from its two end shifts (equal and opposite): from a's end unless a is pinned."""
    if wa > 0.0 and ma > 0.0:
        return shift_a * ma
    return -(shift_b * mb)


@wp.kernel(enable_backward=False)
def seam_spring(sa: SeamArrays, intact: wp.array[wp.int8], x: wp.array[wp.vec3], ke: wp.array[float], dt: float):
    """Before the projection: the impulse the twin's VBD spring carried this step, ke·(x_b - x_a)·dt (on a's end).
    The springs keep the seam closed inside VBD's solve, as the membrane does elsewhere; without them the far side
    falls away every step and the triangles at the seam sit at their strain cap."""
    s = wp.tid()
    sa.impulse[s] = wp.vec3(0.0)
    sp = sa.spring[s]
    if not _intact(intact, sa.bond[s]) or sp < 0:
        return
    sa.impulse[s] = (x[sa.b[s]] - x[sa.a[s]]) * (ke[sp] * dt * dt)


@wp.kernel(enable_backward=False)
def seam_impulse(sa: SeamArrays, intact: wp.array[wp.int8], mass: wp.array[float], inv_mass: wp.array[float],
                 pairs: wp.array2d[wp.int32], pair_delta: wp.array2d[wp.vec3], pair_active: wp.array[wp.int32],
                 count: wp.array[wp.int32], relaxation: float):
    """One sweep's corrections across each intact seam, on its a side: its own rest-0 pair's and its share of every
    long-range pair crossing it (momentum mode: scale = relaxation / busiest end's count)."""
    s = wp.tid()
    if not _intact(intact, sa.bond[s]):
        return
    for e in range(sa.cross_start[s], sa.cross_start[s + 1]):
        c = sa.cross_pair[e]
        if pair_active[c] == 0:
            continue
        i = pairs[c, 0]
        j = pairs[c, 1]
        scale = relaxation / float(wp.max(wp.max(count[i], count[j]), 1))
        di = pair_delta[c, 0] * scale
        dj = pair_delta[c, 1] * scale
        J = wp.vec3(0.0)
        if sa.cross_side[e] == 0:
            J = _impulse(mass[i], mass[j], inv_mass[i], di, dj)
        else:
            J = _impulse(mass[j], mass[i], inv_mass[j], dj, di)
        sa.impulse[s] = sa.impulse[s] + J * sa.cross_weight[e]


@wp.kernel(enable_backward=False)
def seam_force(sa: SeamArrays, intact: wp.array[wp.int8], x: wp.array[wp.vec3], tri_indices: wp.array2d[wp.int32],
               inv_dt2: float, alpha: float, force: wp.array[float], threshold: wp.array[float]):
    """Per bond: the force the seam carries, averaged over FilmConfig.seam_force_tau (``alpha`` = dt / tau, at most
    1), and the force it breaks at.
    - interior seam: its total force against the strength over its length (``break_force``);
    - crack tip (a neighbouring seam broken): its total force against the tear force;
    - free edge of the sheet: the out-of-plane part of its force against the tear force (paper tears from an edge
      when pulled out of plane, Elmendorf's mode), else the total against the strength (in plane it needs its
      strength). The larger of the two ratios decides and is reported."""
    s = wp.tid()
    bond = sa.bond[s]
    if not _intact(intact, bond):
        sa.f_avg[s] = wp.vec3(0.0)
        force[bond] = 0.0
        threshold[bond] = sa.tear_force[s]
        return
    f = sa.f_avg[s] + (sa.impulse[s] * inv_dt2 - sa.f_avg[s]) * alpha
    sa.f_avg[s] = f
    tip = int(0)
    for e in range(sa.nbr_start[s], sa.nbr_start[s + 1]):
        if not _intact(intact, sa.bond[sa.nbr[e]]):
            tip = 1
    total = wp.length(f)
    force[bond] = total
    if tip != 0:
        threshold[bond] = wp.min(sa.tear_force[s], sa.break_force[s])
        return
    threshold[bond] = sa.break_force[s]
    if sa.edge[s] != 0 and sa.tri[s] >= 0:
        t = sa.tri[s]
        n = wp.cross(x[tri_indices[t, 1]] - x[tri_indices[t, 0]], x[tri_indices[t, 2]] - x[tri_indices[t, 0]])
        ln = wp.length(n)
        if ln > 0.0:
            oop = wp.abs(wp.dot(f, n / ln))
            if oop * sa.break_force[s] > total * sa.tear_force[s]:      # oop / tear > total / strength
                force[bond] = oop
                threshold[bond] = sa.tear_force[s]


@wp.kernel(enable_backward=False)
def zero_average(sa: SeamArrays, mask: wp.array[wp.bool]):
    """Reset: the averaged force of the seams in the masked worlds."""
    s = wp.tid()
    w = sa.world[s]
    if w >= 0 and w < mask.shape[0]:
        if mask[w]:
            sa.f_avg[s] = wp.vec3(0.0)


__all__ = ["CLOTH_EDGE", "SeamArrays", "Seams", "find_seams", "pair_gates", "seam_force", "seam_impulse", "seam_spring",
           "set_crossings", "upload", "zero_average"]
