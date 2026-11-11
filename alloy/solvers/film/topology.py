# SPDX-License-Identifier: Apache-2.0
"""Host-side film topology, built once at solver construction (no per-step host work).

- ``FilmTopology``: the triangles that are limited or feel air, their particles, and the particle→triangle CSR the
  gather kernels walk (fixed order: deterministic); particles grouped by world for the per-world ledger.
- ``bending_ke``: per-hinge VBD bending stiffness from the plate rigidity D (discrete shells weighting).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import warp as wp

from .seams import Seams, find_seams, pair_gates, set_crossings

log = logging.getLogger("alloy.film")


def _attr(model, name: str, count: int, default: float) -> np.ndarray:
    ns = getattr(model, "film", None)
    arr = getattr(ns, name, None) if ns is not None else None
    if arr is None:
        return np.full(count, default, dtype=np.float32)
    return np.asarray(arr.numpy(), dtype=np.float32)


@dataclass
class FilmTopology:
    tri_mask: np.ndarray          # [T] bool: film triangle (limited or airborne)
    particles: np.ndarray         # [P] int32 film particles, world-major, ascending within a world
    vtx_start: np.ndarray         # [P+1] int32 CSR into vtx_tri / vtx_slot
    vtx_tri: np.ndarray           # int32 triangle ids
    vtx_slot: np.ndarray          # int32 corner (0, 1, 2) of the particle in that triangle
    world_start: np.ndarray       # [W+1] int32 ranges of ``particles`` per world
    limited: int                  # triangles with strain_cap > 0
    airborne: int                 # triangles with air > 0
    plastic: int                  # hinge edges with yield_angle > 0
    pairs: np.ndarray             # [C, 2] int32 long-range limit particle pairs (longrange.py)
    pair_rest: np.ndarray         # [C] float32
    pair_cap: np.ndarray          # [C] float32
    pair_start: np.ndarray        # [P+1] int32 CSR of each film particle's pair ends
    pair_id: np.ndarray           # int32 pair ids
    pair_end: np.ndarray          # int32 end (0, 1)
    gate_start: np.ndarray        # [C+1] int32 CSR: the seam bonds each pair spans (seams.pair_gates)
    gate: np.ndarray              # int32 topology bond ids; a pair acts while all of them are intact
    limited_tris: np.ndarray      # int32 limited triangle ids, world-major (health residual per world)
    tri_world_start: np.ndarray   # [W+1] int32 ranges of ``limited_tris``
    seams: Seams | None = None    # pre-split tear lines (seams.py), None without any
    j2: int = 0                   # hinge edges with a J2 section (j2_yield > 0, sheet_j2.py)

    @property
    def any(self) -> bool:
        return bool(self.limited or self.airborne or self.plastic or self.j2)


def _csr(parts: np.ndarray, items: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CSR over film particles of ``items`` [K, m] (particle ids per item): (start, item id, slot), ascending item
    id per particle (fixed gather order)."""
    local = np.full(int(parts.max()) + 1 if len(parts) else 0, -1, np.int64)
    local[parts] = np.arange(len(parts))
    K, m = items.shape
    owner = local[items.reshape(-1)]
    item = np.repeat(np.arange(K, dtype=np.int64), m)
    slot = np.tile(np.arange(m, dtype=np.int64), K)
    order = np.lexsort((item, owner))
    counts = np.bincount(owner, minlength=len(parts))
    start = np.zeros(len(parts) + 1, np.int32)
    start[1:] = np.cumsum(counts)
    return start, item[order].astype(np.int32), slot[order].astype(np.int32)


def _mean_edge(model, tris: np.ndarray) -> float:
    if not len(tris):
        return 0.0
    x, ind = model.particle_q.numpy().astype(np.float64), model.tri_indices.numpy()[tris]
    return float(np.linalg.norm(x[ind] - x[np.roll(ind, 1, axis=1)], axis=2).mean())


def _pair_limits(model, tris: np.ndarray, cap: np.ndarray, world: np.ndarray, rep: np.ndarray | None = None):
    """Long-range limits per world (worlds overlap in space, so each is built alone); identical worlds reuse the
    first one's pairs shifted by their particle offset. ``rep`` maps seam copies to one copy (seams.py): the limits
    are then built on the welded sheet and span its seams."""
    from .longrange import build_pair_limits

    lim = cap[tris] > 0.0
    ind = model.tri_indices.numpy()[tris[lim]]
    if rep is not None:
        ind = rep[ind].astype(ind.dtype)
    tri_cap = cap[tris[lim]]
    if not len(ind):
        return np.zeros((0, 2), np.int32), np.zeros(0, np.float32), np.zeros(0, np.float32)
    x = model.particle_q.numpy().astype(np.float64)
    tw = world[ind[:, 0]]
    cache: dict[bytes, tuple] = {}
    P, R, C = [], [], []
    for w in np.unique(tw):
        wi = ind[tw == w]
        verts = np.unique(wi)
        loc = np.searchsorted(verts, wi)
        pc = np.zeros(len(verts), np.float32)
        np.maximum.at(pc, loc.reshape(-1), np.repeat(tri_cap[tw == w], 3))
        key = loc.tobytes() + np.round(x[verts] - x[verts[0]], 7).tobytes() + pc.tobytes()
        if key not in cache:
            cache[key] = build_pair_limits(x[verts], loc, pc)
        lim_w = cache[key]
        P.append(verts[lim_w.pairs] if len(lim_w.pairs) else np.zeros((0, 2), np.int64))
        R.append(lim_w.rest)
        C.append(lim_w.cap)
    return (np.concatenate(P).astype(np.int32), np.concatenate(R).astype(np.float32),
            np.concatenate(C).astype(np.float32))


def build_topology(model, *, long_range: bool = True) -> FilmTopology:
    """Film arrays of ``model`` (attributes in config.py); an all-default model gives an empty topology."""
    T = int(model.tri_count)
    cap = _attr(model, "strain_cap", T, 0.0)
    air = _attr(model, "air", T, 0.0)
    E = int(model.edge_count)
    ya = _attr(model, "yield_angle", E, 0.0)
    mask = (cap > 0.0) | (air > 0.0)
    tris = np.flatnonzero(mask).astype(np.int32)
    ind = model.tri_indices.numpy()[tris] if len(tris) else np.zeros((0, 3), np.int32)
    flat = ind.reshape(-1)
    parts = np.unique(flat).astype(np.int32)
    world = model.particle_world.numpy() if model.particle_world is not None else np.zeros(model.particle_count, np.int32)
    W = max(int(model.world_count), 1)
    pw = world[parts] if len(parts) else np.zeros(0, np.int32)
    order = np.lexsort((parts, pw))
    parts, pw = parts[order], pw[order]
    if len(pw) and pw.min() < 0:
        log.warning("film: %d film particles in the global world (-1) are left out of the per-world ledger",
                    int((pw < 0).sum()))
    world_start = np.searchsorted(pw, np.arange(W + 1)).astype(np.int32)
    vtx_start, vtx_tri, vtx_slot = _csr(parts, ind) if len(parts) else (np.zeros(1, np.int32),) + (np.zeros(0, np.int32),) * 2
    if len(ind):
        vtx_tri = tris[vtx_tri].astype(np.int32)         # item ids -> model triangle ids
    seams = find_seams(model, parts) if len(parts) else None
    if long_range and len(parts) and (cap > 0.0).any():         # on the welded sheet: limits span intact seams
        pairs, prest, pcap = _pair_limits(model, tris, cap, world, seams.rep if seams is not None else None)
    else:
        pairs, prest, pcap = np.zeros((0, 2), np.int32), np.zeros(0, np.float32), np.zeros(0, np.float32)
    gate_start, gate, cross = pair_gates(pairs, model.particle_q.numpy().astype(np.float64), seams,
                                         _mean_edge(model, tris), world)
    if seams is not None:                       # twins at rest length 0: an intact seam is as inextensible as the sheet
        seams.pair = (len(pairs) + np.arange(len(seams.a))).astype(np.int32)
        pairs = np.concatenate([pairs.reshape(-1, 2), np.stack([seams.a, seams.b], 1)]).astype(np.int32)
        prest = np.concatenate([prest, np.zeros(len(seams.a), np.float32)])
        pcap = np.concatenate([pcap, np.zeros(len(seams.a), np.float32)])
        gate_start = np.concatenate([gate_start, gate_start[-1] + 1 + np.arange(len(seams.a))]).astype(np.int32)
        gate = np.concatenate([gate, seams.bond]).astype(np.int32)                # each seam pair: its own bond
        set_crossings(seams, cross)
    pair_start, pair_id, pair_end = _csr(parts, pairs) if len(pairs) else (np.zeros(len(parts) + 1, np.int32),) + (np.zeros(0, np.int32),) * 2
    lt = np.flatnonzero(cap > 0.0).astype(np.int32)
    tw = world[model.tri_indices.numpy()[lt, 0]] if len(lt) else np.zeros(0, np.int32)
    order = np.lexsort((lt, tw))
    lt, tw = lt[order], tw[order]
    tri_world_start = np.searchsorted(tw, np.arange(W + 1)).astype(np.int32)
    return FilmTopology(tri_mask=mask, particles=parts, vtx_start=vtx_start, vtx_tri=vtx_tri, vtx_slot=vtx_slot,
                        world_start=world_start, limited=int((cap > 0.0).sum()), airborne=int((air > 0.0).sum()),
                        plastic=int((ya > 0.0).sum()), j2=int((_attr(model, "j2_yield", E, 0.0) > 0.0).sum()),
                        pairs=pairs, pair_rest=prest, pair_cap=pcap,
                        pair_start=pair_start, pair_id=pair_id, pair_end=pair_end, gate_start=gate_start, gate=gate,
                        limited_tris=lt, tri_world_start=tri_world_start, seams=seams)


@dataclass
class DeviceTopology:
    particles: wp.array
    vtx_start: wp.array
    vtx_tri: wp.array
    vtx_slot: wp.array
    world_start: wp.array
    pairs: wp.array
    pair_rest: wp.array
    pair_cap: wp.array
    pair_start: wp.array
    pair_id: wp.array
    pair_end: wp.array
    gate_start: wp.array
    gate: wp.array
    limited_tris: wp.array
    tri_world_start: wp.array

    @staticmethod
    def upload(top: FilmTopology, device) -> DeviceTopology:
        def a(x):
            return wp.array(np.ascontiguousarray(x, dtype=np.int32), dtype=wp.int32, device=device)

        def f(x):
            return wp.array(np.ascontiguousarray(x, dtype=np.float32), dtype=float, device=device)

        pairs = wp.array(np.ascontiguousarray(top.pairs.reshape(-1, 2), dtype=np.int32), dtype=wp.int32,
                         device=device)
        return DeviceTopology(a(top.particles), a(top.vtx_start), a(top.vtx_tri), a(top.vtx_slot),
                              a(top.world_start), pairs, f(top.pair_rest), f(top.pair_cap), a(top.pair_start),
                              a(top.pair_id), a(top.pair_end), a(top.gate_start), a(top.gate if len(top.gate) else [0]),
                              a(top.limited_tris), a(top.tri_world_start))


def bending_ke(vertices: np.ndarray, edges: np.ndarray, D: float) -> np.ndarray:
    """Per-hinge ``edge_ke`` [N] so VBD's ½·ke·L·(θ-θ̄)² is the plate energy ½·D·κ²·area of a uniform bend.

    A hinge between triangles of areas A1, A2 on edge e: the centroids are d = 2(A1+A2)/(3|e|) apart across the
    edge, θ = κ·d for a uniform curvature κ normal to e, and the strip |e|·d holds ½·D·κ²·|e|·d. Matching
    ½·ke·|e|·θ² gives ke = D/d = 3·D·|e|/(2(A1+A2)), Grinspun et al. 2003's |e|/h̄_e weight (h̄_e = (A1+A2)/(3|e|))
    halved by the ½ of VBD's energy. ``edges`` rows are Newton's (o0, o1, v0, v1); boundary rows (o < 0) get D/d
    with the single triangle. ``vertices`` are rest positions.
    """
    ke = np.zeros(len(edges), dtype=np.float64)
    for n, (o0, o1, a, b) in enumerate(edges):
        e = vertices[b] - vertices[a]
        le = float(np.linalg.norm(e))
        if le <= 0.0:
            continue
        area = 0.0
        for o in (o0, o1):
            if o >= 0:
                area += 0.5 * float(np.linalg.norm(np.cross(e, vertices[o] - vertices[a])))
        if area > 0.0:
            ke[n] = 3.0 * D * le / (2.0 * area)
    return ke
