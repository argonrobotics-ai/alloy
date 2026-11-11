# SPDX-License-Identifier: Apache-2.0
"""Seam-twin self-contact exclusion that lasts only while the twins are joined (SCR-SC5-2).

Pre-split twins sit at one point. With VBD self-contact on, an intact seam pushes itself open unless each seam vertex
ignores the faces around its twin. Newton's ``particle_rest_shape_contact_exclusion_radius`` does that on rest
positions, so the exclusion never ends and cut faces pass through each other when pressed back together.

``SeamContactFilter`` replaces that radius (the solver's must be 0). At attach it adds entries to the VBD
detector's own filter lists:
- vertex v -> every triangle incident to a twin u of v;
- edge incident to v -> every edge incident to u.
Twins are the copies of one seam point: a connected group of CLOTH_EDGE / TET_CUT bonds. Each added entry is gated
by its pair (v, u) and filters only while v and u are still joined. Duplicate entries act as an OR: an edge pair
filters while any of its twin pairs is joined.

A bond joins its twins while it is intact and, after it breaks, until its twins first move further apart than the
solver's contact distance (``particle_self_contact_margin``); this ``released`` latch is per bond. Twins sit at one
point when their bond breaks, and VBD's contact barrier works on unsigned distance, so it cannot tell which side a
vertex at distance zero belongs to: measured on a split block, un-excluding coincident twins at the break shot one
piece 2.7 mm into the other. A blade cut releases its twins while the blade is between them (blade thickness >
contact distance). Twins that never separate stay excluded, as before this change.

``update`` recomputes each group's joined components and rewrites the lists in place: one thread per group, then
one per row holding a gated entry. The arrays keep their size and address, so the update can be captured in a
graph. A gated-off entry takes the value of the nearest active entry before it in its row (or -1). Every row
therefore stays non-decreasing, and Newton's lookup (binary search for the first entry > key, then compare the one
before) finds exactly the active entries.

Newton contract (newton@ac7f7830): the detector reads the lists at every detection launch
(geometry/tri_mesh_collision.py:961, :1005; kernels.py:1625-1636, :1780-1789). VBD passes the rest-shape radius
as the minimum query distance (solvers/vbd/solver_vbd.py:5303, :5308). Only public attributes are used.
tests/topology/test_seam_contact.py is the tripwire.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.core.errors import AssemblyError
from alloy.topology.types import BondKind, BondSet

log = logging.getLogger("alloy.topology")

TWIN_KINDS = (int(BondKind.CLOTH_EDGE), int(BondKind.TET_CUT))


@wp.kernel
def release_twins(bond: wp.array[wp.int32], a: wp.array[wp.int32], b: wp.array[wp.int32], intact: wp.array[wp.int8],
                  q: wp.array[wp.vec3], release: float, released: wp.array[wp.int8]):
    """One thread per twin bond: latch ``released`` once a broken bond's twins are more than ``release`` apart."""
    i = bond[wp.tid()]
    if intact[i] != wp.int8(0):
        released[i] = wp.int8(0)
    elif released[i] == wp.int8(0) and wp.length(q[a[i]] - q[b[i]]) > release:
        released[i] = wp.int8(1)


@wp.kernel
def join_components(start: wp.array[wp.int32], member: wp.array[wp.int32], bond_start: wp.array[wp.int32],
                    bond: wp.array[wp.int32], a: wp.array[wp.int32], b: wp.array[wp.int32],
                    intact: wp.array[wp.int8], released: wp.array[wp.int8], comp: wp.array[wp.int32]):
    """One thread per twin group: comp[p] = smallest member joined to p (bond intact, or broken but not released)."""
    g = wp.tid()
    for k in range(start[g], start[g + 1]):
        comp[member[k]] = member[k]
    for _ in range(start[g + 1] - start[g]):           # label propagation; size passes >= the group's diameter
        for j in range(bond_start[g], bond_start[g + 1]):
            i = bond[j]
            if intact[i] != wp.int8(0) or released[i] == wp.int8(0):
                lo = wp.min(comp[a[i]], comp[b[i]])
                comp[a[i]] = lo
                comp[b[i]] = lo


@wp.kernel
def gate_rows(rows: wp.array[wp.int32], offsets: wp.array[wp.int32], orig: wp.array[wp.int32],
              pa: wp.array[wp.int32], pb: wp.array[wp.int32], comp: wp.array[wp.int32], out: wp.array[wp.int32]):
    """One thread per row holding a gated entry: active entries keep their value, the rest repeat the last one."""
    r = rows[wp.tid()]
    last = int(-1)
    for k in range(offsets[r], offsets[r + 1]):
        u = pa[k]
        if u < 0 or comp[u] == comp[pb[k]]:
            last = orig[k]
        out[k] = last


@dataclass(frozen=True)
class TwinGroups:
    """Copies of one seam point, as CSR lists of particles and of the twin bonds joining them."""

    start: np.ndarray       # [G+1]
    member: np.ndarray      # particles
    bond_start: np.ndarray  # [G+1]
    bond: np.ndarray        # bond ids

    @property
    def count(self) -> int:
        return int(self.start.shape[0] - 1)

    def ordered_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """Every (v, u), v != u, of members of one group."""
        vs, us = [], []
        for g in range(self.count):
            m = self.member[self.start[g]:self.start[g + 1]]
            v, u = np.meshgrid(m, m, indexing="ij")
            off = v != u
            vs.append(v[off])
            us.append(u[off])
        return np.concatenate(vs), np.concatenate(us)


def twin_groups(kind: np.ndarray, a: np.ndarray, b: np.ndarray, n_particles: int) -> TwinGroups:
    """Connected components of the CLOTH_EDGE / TET_CUT bond graph (all bonds, intact or not)."""
    ids = np.flatnonzero(np.isin(kind, TWIN_KINDS) & (a >= 0) & (b >= 0))
    if ids.size == 0:
        z = np.zeros(1, np.int64)
        return TwinGroups(z, z[:0], z, z[:0])
    ba, bb = a[ids].astype(np.int64), b[ids].astype(np.int64)
    label = np.arange(n_particles, dtype=np.int64)
    while True:
        lo = np.minimum(label[ba], label[bb])
        new = label.copy()
        np.minimum.at(new, ba, lo)
        np.minimum.at(new, bb, lo)
        new = new[new]                                   # pointer jump
        if np.array_equal(new, label):
            break
        label = new
    member = np.unique(np.concatenate([ba, bb]))
    roots, member_group = np.unique(label[member], return_inverse=True)
    order = np.argsort(member_group, kind="stable")
    start = np.concatenate([[0], np.cumsum(np.bincount(member_group, minlength=roots.size))])
    bond_group = np.searchsorted(roots, label[ba])
    border = np.argsort(bond_group, kind="stable")
    bond_start = np.concatenate([[0], np.cumsum(np.bincount(bond_group, minlength=roots.size))])
    return TwinGroups(start, member[order], bond_start, ids[border].astype(np.int64))


def _incident(n: int, elems: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """vertex -> element CSR (start [n+1], items)."""
    v = elems.reshape(-1).astype(np.int64)
    e = np.repeat(np.arange(elems.shape[0], dtype=np.int64), elems.shape[1])
    order = np.argsort(v, kind="stable")
    return np.concatenate([[0], np.cumsum(np.bincount(v, minlength=n))]), e[order]


def _expand(start: np.ndarray, count: np.ndarray) -> np.ndarray:
    """Concatenated ranges [start[i], start[i] + count[i])."""
    total = int(count.sum())
    if total == 0:
        return np.zeros(0, np.int64)
    shift = np.repeat(start - np.concatenate([[0], np.cumsum(count)[:-1]]), count)
    return shift + np.arange(total, dtype=np.int64)


def _vertex_tri_entries(v, u, tris, n):
    """(row vertex v, triangle incident to u, v, u)."""
    start, items = _incident(n, tris)
    cnt = start[u + 1] - start[u]
    return np.repeat(v, cnt), items[_expand(start[u], cnt)], np.repeat(v, cnt), np.repeat(u, cnt)


def _edge_edge_entries(v, u, edges, n):
    """(edge incident to v, edge incident to u, v, u)."""
    start, items = _incident(n, edges)
    cv = start[v + 1] - start[v]
    pair = np.repeat(np.arange(v.size), cv)
    ev = items[_expand(start[v], cv)]
    uu = u[pair]
    cu = start[uu + 1] - start[uu]
    rep = lambda x: np.repeat(x, cu)  # noqa: E731
    return rep(ev), items[_expand(start[uu], cu)], rep(v[pair]), rep(uu)


@dataclass
class GatedList:
    """One filter family (vertex-triangle or edge-edge): the list Newton reads plus the gating it needs."""

    live: wp.array      # values the detector reads (rewritten by ``update``)
    offsets: wp.array   # [rows+1]
    orig: wp.array      # values with every entry active
    pa: wp.array        # gating pair (pa, pb) per entry; pa = -1 for Newton's own entries
    pb: wp.array
    rows: wp.array      # rows that hold a gated entry

    @staticmethod
    def build(n_rows: int, base: tuple[wp.array | None, wp.array | None], extra, device) -> GatedList:
        """Merge Newton's list ``base`` (values, offsets) with gated ``extra`` (rows, cols, pa, pb)."""
        rows, cols, pa, pb = (np.asarray(x, np.int64) for x in extra)
        bv, bo = (None if x is None else x.numpy().astype(np.int64) for x in base)
        if bv is None or bo is None:
            bv, bo = np.zeros(0, np.int64), np.zeros(n_rows + 1, np.int64)
        brow = np.repeat(np.arange(n_rows, dtype=np.int64), np.diff(bo))
        width = max(int(cols.max(initial=0)), int(bv.max(initial=0))) + 1
        keep = ~np.isin(rows * width + cols, brow * width + bv)        # Newton's own entries stay permanent
        r = np.concatenate([brow, rows[keep]])
        c = np.concatenate([bv, cols[keep]])
        a = np.concatenate([np.full(bv.size, -1), pa[keep]])
        b = np.concatenate([np.full(bv.size, -1), pb[keep]])
        order = np.lexsort((b, a, c, r))
        r, c, a, b = r[order], c[order], a[order], b[order]
        uniq = np.ones(r.size, bool)
        uniq[1:] = (np.diff(r) != 0) | (np.diff(c) != 0) | (np.diff(a) != 0) | (np.diff(b) != 0)
        r, c, a, b = r[uniq], c[uniq], a[uniq], b[uniq]
        off = np.concatenate([[0], np.cumsum(np.bincount(r, minlength=n_rows))])
        arr = lambda x: wp.array(x.astype(np.int32), dtype=wp.int32, device=device)  # noqa: E731
        return GatedList(arr(c), arr(off), arr(c), arr(a), arr(b), arr(np.unique(r[a >= 0])))

    @property
    def gated(self) -> int:
        return int(self.rows.shape[0])

    def update(self, comp: wp.array) -> None:
        if self.gated:
            wp.launch(gate_rows, dim=self.gated, device=comp.device,
                      inputs=[self.rows, self.offsets, self.orig, self.pa, self.pb, comp, self.live])


def vbd_detector(model: newton.Model, solver) -> object:
    """The solver-owned self-contact detector of ``solver``, checked for use with ``SeamContactFilter``."""
    if not isinstance(solver, newton.solvers.SolverVBD):
        raise AssemblyError(f"seam contact: {type(solver).__name__} is not SolverVBD")
    if not getattr(solver, "particle_enable_self_contact", False):
        raise AssemblyError("seam contact: SolverVBD self-contact is off (particle_enable_self_contact=False)")
    if getattr(solver, "collision_pipeline", None) is not None:
        raise AssemblyError("seam contact: self-contact owned by a CollisionPipeline is not supported; build "
                            "SolverVBD without collision_pipeline")
    det = getattr(solver, "trimesh_collision_detector", None)
    if det is None or det.model is not model:
        raise AssemblyError("seam contact: the solver was built on a different model (or view); particle and "
                            "triangle ids would not match the bonds")
    radius = float(getattr(solver, "particle_rest_shape_contact_exclusion_radius", 0.0))
    if radius > 0.0:
        raise AssemblyError(f"seam contact: particle_rest_shape_contact_exclusion_radius={radius} keeps seam twins "
                            "excluded for good; build SolverVBD with 0 (the controller excludes twins while joined)")
    return det


class SeamContactFilter:
    """Gates a SolverVBD's self-contact filter lists by twin-bond state (see the module docstring)."""

    def __init__(self, model: newton.Model, bonds: BondSet, solver) -> None:
        det = vbd_detector(model, solver)
        dev, n = model.device, model.particle_count
        g = twin_groups(bonds.kind.numpy().astype(np.int64), bonds.a.numpy(), bonds.b.numpy(), n)
        if g.count == 0 or not model.tri_count:
            raise AssemblyError("seam contact: no seam twins (CLOTH_EDGE / TET_CUT bonds) on triangles to gate")
        v, u = g.ordered_pairs()
        tris = model.tri_indices.numpy()
        self.vt = GatedList.build(n, (det.vertex_triangle_filtering_list, det.vertex_triangle_filtering_list_offsets),
                                  _vertex_tri_entries(v, u, tris, n), dev)
        self.ee = None
        if model.edge_count:
            edges = model.edge_indices.numpy()[:, 2:4]
            self.ee = GatedList.build(model.edge_count, (det.edge_filtering_list, det.edge_filtering_list_offsets),
                                      _edge_edge_entries(v, u, edges, n), dev)
        ee = (self.ee.live, self.ee.offsets) if self.ee else (det.edge_filtering_list, det.edge_filtering_list_offsets)
        det.set_collision_filter_list(self.vt.live, self.vt.offsets, *ee)
        arr = lambda x: wp.array(np.asarray(x).astype(np.int32), dtype=wp.int32, device=dev)  # noqa: E731
        self._groups = (arr(g.start), arr(g.member), arr(g.bond_start), arr(g.bond))
        self._a, self._b = bonds.a, bonds.b
        self.release = float(solver.particle_self_contact_margin)
        self.released = wp.zeros(max(bonds.count, 1), dtype=wp.int8, device=dev)     # per bond (state)
        self.comp = wp.array(np.arange(max(n, 1), dtype=np.int32), dtype=wp.int32, device=dev)
        self.group_count, self.twin_bonds = g.count, int(g.bond.size)
        self.update(bonds.intact, model.particle_q)
        log.info("seam contact: %d twin groups, %d vertex-triangle + %d edge-edge gated rows, release %.3g m",
                 g.count, self.vt.gated, self.ee.gated if self.ee else 0, self.release)

    def update(self, intact: wp.array, particle_q: wp.array) -> None:
        """Latch released twins, recompute joined twins and rewrite the filter lists (kernels only; capturable)."""
        dev = self.comp.device
        wp.launch(release_twins, dim=self.twin_bonds, device=dev,
                  inputs=[self._groups[3], self._a, self._b, intact, particle_q, self.release, self.released])
        wp.launch(join_components, dim=self.group_count, device=dev,
                  inputs=[*self._groups, self._a, self._b, intact, self.released, self.comp])
        self.vt.update(self.comp)
        if self.ee is not None:
            self.ee.update(self.comp)
