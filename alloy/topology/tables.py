# SPDX-License-Identifier: Apache-2.0
"""Host-built device tables the topology kernels read (built once at controller construction).

Native elements are found from the Model: particle bonds by spring endpoints, body bonds by MuJoCo equality
bodies. This makes a BondSet built by any producer usable. Cross-seam bending tags (``topology:edge_bond*``) are
used only when the BondSet rows line up with the Model's ``topology:bond`` rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mujoco
import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.core.errors import AssemblyError
from alloy.topology.bonds import NS, bond_column, has_bonds
from alloy.topology.types import BondKind, BondSet

log = logging.getLogger("alloy.topology")

PARTICLE_KINDS = (int(BondKind.SPRING), int(BondKind.CLOTH_EDGE), int(BondKind.TET_CUT))
BODY_KINDS = (int(BondKind.RIGID_WELD), int(BondKind.JOINT))
EQ_CONNECT, EQ_WELD = int(mujoco.mjtEq.mjEQ_CONNECT), int(mujoco.mjtEq.mjEQ_WELD)   # = Newton EqType values


def item_worlds(world: np.ndarray, world_count: int, what: str) -> np.ndarray:
    """World of each item; in a single-world model global items (−1) belong to world 0."""
    w = np.asarray(world, dtype=np.int64).copy()
    if world_count <= 1:
        w[w < 0] = 0
    elif np.any(w < 0):
        raise AssemblyError(f"{what}: {int((w < 0).sum())} items belong to no world in a {world_count}-world model")
    return w


def csr_union(vert_start: np.ndarray, vert_elems: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Per bond, the elements incident to a or b (CSR start, flat list)."""
    def span(v: np.ndarray) -> list[np.ndarray]:
        return [vert_elems[vert_start[x]:vert_start[x + 1]] if x >= 0 else np.zeros(0, np.int64) for x in v]
    lists = [np.unique(np.concatenate([ea, eb])) for ea, eb in zip(span(a), span(b))]
    start = np.concatenate([[0], np.cumsum([len(x) for x in lists])]).astype(np.int32)
    flat = np.concatenate(lists).astype(np.int32) if lists else np.zeros(0, np.int32)
    return start, flat


def vertex_incidence(elements: np.ndarray, n_vert: int) -> tuple[np.ndarray, np.ndarray]:
    """CSR vertex -> elements containing it."""
    if elements.size == 0:
        return np.zeros(n_vert + 1, np.int64), np.zeros(0, np.int64)
    k = elements.shape[1]
    v = elements.reshape(-1)
    e = np.repeat(np.arange(len(elements)), k)
    order = np.argsort(v, kind="stable")
    counts = np.bincount(v, minlength=n_vert)
    return np.concatenate([[0], np.cumsum(counts)]), e[order]


@dataclass
class BondTables:
    """Device arrays indexed by bond (length B) plus per-world CSR (length W+1)."""

    count: int
    world: wp.array                # [B] int32 bond world (−1 mapped to 0 in single-world models)
    world_start: wp.array          # [W+1] int32
    world_order: wp.array          # [B] int32 bond ids grouped by world, index order
    spring: wp.array               # [B] int32 native spring (−1)
    eq: wp.array                   # [B] int32 Newton equality row (−1)
    elem_start: wp.array           # [B+1] int32 incident triangles (CLOTH_EDGE) or tets (TET_CUT)
    elem: wp.array                 # int32
    cut_strain: wp.array           # [B] float32 blade gate (inf = none)
    cut_force: wp.array            # [B] float32
    rest: wp.array                 # [B] float32 spring rest length or body distance at construction
    rel0: wp.array                 # [B] transform: body b in body a's frame at construction (body bonds)
    edge_ids: wp.array             # [E_tag] int32 cross-seam bending elements
    edge_bonds: wp.array           # [E_tag, 2] int32 their bonds
    has_eq: bool
    has_particle: bool


def _spring_map(model: newton.Model) -> dict[tuple[int, int], int]:
    if not model.spring_count:
        return {}
    si = model.spring_indices.numpy().reshape(-1, 2)
    return {(int(min(i, j)), int(max(i, j))): s for s, (i, j) in enumerate(si)}


def _eq_map(model: newton.Model) -> dict[tuple[int, int, int], int]:
    mj = getattr(model, "mujoco", None)
    if mj is None or getattr(mj, "equality_constraint_count", 0) == 0:
        return {}
    b1, b2 = mj.equality_constraint_body1.numpy(), mj.equality_constraint_body2.numpy()
    typ = mj.equality_constraint_type.numpy()
    return {(int(t), int(x), int(y)): r for r, (t, x, y) in enumerate(zip(typ, b1, b2))}


def _native_ids(model: newton.Model, kind: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    springs, eqs = _spring_map(model), _eq_map(model)
    spring = np.full(len(kind), -1, np.int32)
    eq = np.full(len(kind), -1, np.int32)
    for i, (k, x, y) in enumerate(zip(kind, a, b)):
        if k in PARTICLE_KINDS:
            spring[i] = springs.get((int(min(x, y)), int(max(x, y))), -1)
        elif k in BODY_KINDS:
            t = EQ_WELD if k == int(BondKind.RIGID_WELD) else EQ_CONNECT
            eq[i] = eqs.get((t, int(x), int(y)), eqs.get((t, int(y), int(x)), -1))
    missing = np.flatnonzero(((spring < 0) & np.isin(kind, PARTICLE_KINDS)) | ((eq < 0) & np.isin(kind, BODY_KINDS)))
    if len(missing):
        raise AssemblyError(f"{len(missing)} bonds have no native element (first: bond {int(missing[0])}, "
                            f"kind {BondKind(int(kind[missing[0]])).name}, a={int(a[missing[0]])}, b={int(b[missing[0]])})")
    return spring, eq


def _rows_aligned(model: newton.Model, bonds: BondSet, kind: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
    if not has_bonds(model) or bond_column(model, "kind").shape[0] != len(kind):
        return False
    return bool(np.array_equal(bond_column(model, "kind").numpy(), kind))


def _elements(model: newton.Model, kind: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(kind)
    start = np.zeros(n + 1, np.int32)
    flat_parts, counts = [], np.zeros(n, np.int64)
    for k, arr, cnt in ((int(BondKind.CLOTH_EDGE), model.tri_indices, model.tri_count),
                        (int(BondKind.TET_CUT), model.tet_indices, model.tet_count)):
        sel = np.flatnonzero(kind == k)
        if len(sel) == 0 or not cnt:
            continue
        vs, ve = vertex_incidence(arr.numpy().reshape(cnt, -1).astype(np.int64), model.particle_count)
        s, f = csr_union(vs, ve, a[sel], b[sel])
        counts[sel] = np.diff(s)
        flat_parts.append((sel, s, f))
    start[1:] = np.cumsum(counts)
    flat = np.zeros(int(start[-1]), np.int32)
    for sel, s, f in flat_parts:
        for j, bi in enumerate(sel):
            flat[start[bi]:start[bi + 1]] = f[s[j]:s[j + 1]]
    return start, flat


def _rel0(model: newton.Model, kind: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rest distance and pose of b in a's frame for body bonds (world = identity for −1)."""
    q = model.body_q.numpy() if model.body_count else np.zeros((0, 7))
    rel = np.tile(np.array([0, 0, 0, 0, 0, 0, 1.0], np.float32), (len(kind), 1))
    rest = np.zeros(len(kind), np.float32)
    for i in np.flatnonzero(np.isin(kind, BODY_KINDS)):
        xa = wp.transform(*q[a[i]]) if a[i] >= 0 else wp.transform_identity()
        xb = wp.transform(*q[b[i]]) if b[i] >= 0 else wp.transform_identity()
        r = wp.transform_multiply(wp.transform_inverse(xa), xb)
        rel[i] = np.array([*r.p, *r.q], np.float32)
        rest[i] = float(np.linalg.norm(np.array(xb.p) - np.array(xa.p)))
    return rest, rel


def build_bond_tables(model: newton.Model, bonds: BondSet, world_count: int) -> BondTables:
    dev = model.device
    kind = bonds.kind.numpy().astype(np.int64)
    a, b = bonds.a.numpy().astype(np.int64), bonds.b.numpy().astype(np.int64)
    world = item_worlds(bonds.world.numpy(), world_count, "BondSet")
    order = np.argsort(world, kind="stable").astype(np.int32)
    wstart = np.concatenate([[0], np.cumsum(np.bincount(world, minlength=world_count)[:world_count])]).astype(np.int32)
    spring, eq = _native_ids(model, kind, a, b)
    est, eflat = _elements(model, kind, a, b)
    aligned = _rows_aligned(model, bonds, kind, a, b)
    inf = np.full(len(kind), np.inf, np.float32)
    cut_s = bond_column(model, "cut_strain").numpy() if aligned else inf
    cut_f = bond_column(model, "cut_force").numpy() if aligned else inf
    rest, rel = _rel0(model, kind, a, b)
    rest = np.where(spring >= 0, model.spring_rest_length.numpy()[np.maximum(spring, 0)] if model.spring_count else 0.0,
                    rest).astype(np.float32)
    eids, ebonds = _edge_tags(model, aligned)
    i32 = lambda x: wp.array(np.asarray(x, np.int32), dtype=wp.int32, device=dev)  # noqa: E731
    f32 = lambda x: wp.array(np.asarray(x, np.float32), dtype=wp.float32, device=dev)  # noqa: E731
    return BondTables(
        count=len(kind), world=i32(world if len(world) else np.zeros(1)), world_start=i32(wstart), world_order=i32(order), spring=i32(spring), eq=i32(eq),
        elem_start=i32(est), elem=i32(eflat if len(eflat) else np.zeros(1)), cut_strain=f32(cut_s),
        cut_force=f32(cut_f), rest=f32(rest), rel0=wp.array(rel, dtype=wp.transform, device=dev),
        edge_ids=i32(eids if len(eids) else np.zeros(0)),
        edge_bonds=wp.array(ebonds.reshape(-1, 2).astype(np.int32), dtype=wp.int32, device=dev),
        has_eq=bool((eq >= 0).any()), has_particle=bool((spring >= 0).any()))


def _edge_tags(model: newton.Model, aligned: bool) -> tuple[np.ndarray, np.ndarray]:
    if not aligned or not model.edge_count:
        return np.zeros(0, np.int32), np.zeros((0, 2), np.int32)
    ns = getattr(model, NS)
    b0, b1 = ns.edge_bond0.numpy(), ns.edge_bond1.numpy()
    ids = np.flatnonzero((b0 >= 0) | (b1 >= 0)).astype(np.int32)
    return ids, np.stack([b0[ids], b1[ids]], axis=1).astype(np.int32)
