# SPDX-License-Identifier: Apache-2.0
"""Virtual-node copies for cutting a tet mesh along any path: assembly and split geometry (WP10 item 9 step 3).

After the virtual node algorithm (Molino, Bao and Fedkiw, SIGGRAPH 2004, §4-5), with all incidence fixed at assembly
([internal note] scr/WP10.md, step 3). One cut may pass through each node's neighbourhood; its two sides are + and -,
oriented by the blade face (``vnode_cut``).

Per mesh node j of the cuttable region (a *hub*), two copy particles: v+_j and v-_j. Each has a zero-length weld
spring to j. Per cuttable tet T, two fragment tets: T+ on its corners' v+ copies and T- on their v- copies. Copies
are shared between neighbouring tets, so fragments on one side of the cut stay joined through them, as in Molino's
scheme. Until a tet next to j splits, v+_j and v-_j are dormant: no mass, no ACTIVE flag, weld stiffness 0, and
``vnode_cut`` keeps them on j. Fragment tets carry no material until their tet splits.

When the blade splits T (``vnode_cut``):
- T's material goes to 0; T+ and T- take it scaled by their volume fractions f+ and f- = 1 - f+. An affine map keeps
  volume ratios, so the current shape gives the rest fraction.
- Per corner j, the hub hands ``1 - HUB_KEEP`` of T's corner mass to v+_j and v-_j in the ratio f+ : f-, keeping
  momentum.
- At j's first split, the weld to the copy on j's own side turns on, making that copy j's material node in the
  fragments on that side. The other copy is the virtual node shared by the fragments on the far side.

``positive_fraction`` is the volume fraction of a tet on the + side of a plane, from its corners' signed distances.
One corner alone on a side cuts off a corner tet; a two-two split cuts off a triangular prism (three tets).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import warp as wp

from alloy._vendor import newton

LOCAL_EDGES = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])  # a tet's edges, corner pairs
WELD_KE_RATIO = 1.0     # weld ke = ratio (k_mu + k_lambda) V^(1/3), as emit.SOFT_SEAM_KE_RATIO for tet twins
#                         (coordinator ruling 2026-11-06T18:59Z). 100 converged ~9x too slowly at 10 VBD iterations
#                         and a welded node held a pressing knife like a rock; TOP-2 on copies read peak 0.78 N at 1
#                         against 1.72 N at 100 (seam twin 0.96 N). The oblique stroke's rest-vs-world deviation is
#                         1.13 mm at 1 (diagnostic guard 1.2 mm).
HUB_KEEP = 0.5          # share of a split tet's corner mass its hub keeps, so a hub never reaches mass 0


@dataclass(frozen=True)
class TetCopies:
    """Host record, model indices. Rows: cuttable tets (C) and hubs (H)."""

    tet: np.ndarray        # [C] the cuttable tet T
    hub: np.ndarray        # [C, 4] T's corner particles
    hub_row: np.ndarray    # [C, 4] their rows in the hub table
    copy_tet: np.ndarray   # [C, 2] T+, T-
    material: np.ndarray   # [C, 3] T's (k_mu, k_lambda, k_damp)
    mass: np.ndarray       # [C] T's mass [kg] (density x rest volume)
    node_hub: np.ndarray   # [H] hub particle
    node: np.ndarray       # [H, 2] copies v+, v-
    weld: np.ndarray       # [H, 2] spring index hub -> copy
    weld_ke: np.ndarray    # [H] weld stiffness when on [N/m]
    edge: np.ndarray       # [E, 2] mesh edges of the cuttable tets, hub rows (low, high)
    tet_edge: np.ndarray   # [C, 6] each tet's edges (``LOCAL_EDGES`` order) in the edge table
    tet_edge_flip: np.ndarray  # [C, 6] the local pair runs high -> low in hub rows

    @property
    def count(self) -> int:
        return int(self.tet.shape[0])


def add_tet_copies(builder: newton.ModelBuilder, tets: Sequence[int], density: float) -> TetCopies:
    """Add dormant copies for builder tets ``tets`` (call after the soft mesh, before ``color``)."""
    tets = np.asarray(tets, np.int64)
    hub = np.asarray(builder.tet_indices, np.int64)[tets]
    mat = np.asarray(builder.tet_materials, np.float64)[tets]
    node_hub, hub_row = np.unique(hub, return_inverse=True)
    hub_row = hub_row.reshape(hub.shape)
    x = np.asarray(builder.particle_q, np.float64)[hub]
    rest_vol = np.abs(np.linalg.det(x[:, 1:] - x[:, :1])) / 6.0
    check_hub_mass(np.asarray(builder.particle_mass, np.float64)[node_hub], hub_row, density * rest_vol)
    node = np.empty((len(node_hub), 2), np.int64)
    weld = np.empty((len(node_hub), 2), np.int64)
    for r, h in enumerate(node_hub):
        for k in range(2):
            node[r, k] = builder.add_particle(builder.particle_q[h], (0.0, 0.0, 0.0), 0.0, builder.particle_radius[h],
                                              flags=0)
            builder.add_spring(int(h), int(node[r, k]), 0.0, 0.0, 0.0)
            weld[r, k] = len(builder.spring_stiffness) - 1
    vol = np.empty(len(tets))
    copy_tet = np.empty((len(tets), 2), np.int64)
    for c in range(len(tets)):
        for k in range(2):
            vol[c] = builder.add_tetrahedron(*(int(n) for n in node[hub_row[c], k]), k_mu=0.0, k_lambda=0.0,
                                             k_damp=0.0)
            copy_tet[c, k] = len(builder.tet_indices) - 1
    if np.any(vol <= 0.0):
        raise ValueError(f"tet copies: {int((vol <= 0).sum())} cuttable tets are inverted or degenerate")
    tet_ke = WELD_KE_RATIO * (mat[:, 0] + mat[:, 1]) * np.cbrt(vol)
    weld_ke = np.zeros(len(node_hub))
    np.maximum.at(weld_ke, hub_row.reshape(-1), np.repeat(tet_ke, 4))
    edge, tet_edge, flip = edge_table(hub_row, len(node_hub))
    return TetCopies(tets, hub, hub_row, copy_tet, mat, density * vol, node_hub, node, weld, weld_ke, edge, tet_edge,
                     flip)


def check_hub_mass(hub_mass: np.ndarray, hub_row: np.ndarray, tet_mass: np.ndarray) -> None:
    """A hub hands (1 - HUB_KEEP) of each split tet's corner share to its copies (``vnode_split.hand_over``), so a
    dynamic hub must hold at least its cuttable tets' corner shares (density x volume / 4) or it could reach mass <= 0.
    That fails when ``density`` exceeds the mesh's, or the mesh lumps mass another way. Mass-0 (kinematic) hubs book
    their copies' mass as added instead (WP4 review 2026-11-06)."""
    need = np.zeros(len(hub_mass))
    np.add.at(need, hub_row.reshape(-1), np.repeat(tet_mass / 4.0, 4))
    short = (hub_mass > 0.0) & (hub_mass < need * (1.0 - 1.0e-5))
    if np.any(short):
        raise ValueError(f"tet copies: {int(short.sum())} hubs hold less mass than their cuttable tets' corner shares "
                         f"(worst {float((hub_mass / need)[short].min()):.3f} of it): pass the mesh's density")


def edge_table(hub_row: np.ndarray, hubs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(edges [E, 2] hub rows low -> high, tet edges [C, 6], flipped [C, 6]): the mesh edges of the cuttable tets."""
    pairs = hub_row[:, LOCAL_EDGES]                                   # [C, 6, 2]
    key = pairs.min(axis=-1) * hubs + pairs.max(axis=-1)
    uniq, inv = np.unique(key.reshape(-1), return_inverse=True)
    return np.stack([uniq // hubs, uniq % hubs], axis=1), inv.reshape(-1, 6), pairs[..., 0] > pairs[..., 1]


def recolor_copies(builder: newton.ModelBuilder, copies: TetCopies) -> None:
    """Give copies VBD colours; call after ``builder.color()``.

    Newton's colouring graph leaves out springs and zero-material tets. A copy is adjacent to its hub (weld) and to
    the copies of the hub's mesh neighbours (fragment tets), which mirror the hub's own adjacency. So v+_j and v-_j
    take colour (c_j + 1) mod C, where c_j is hub j's colour. A bijection keeps neighbouring copies apart, a shift
    with C >= 2 keeps a copy off its hub's colour, and no colour is added. (Without this, welded or fragment-sharing
    nodes solved in parallel diverged within 5 substeps of a split.)"""
    builder.particle_color_groups = recolored(builder.particle_color_groups, builder.particle_count, copies)


def recolor_model(model: newton.Model, copies: TetCopies) -> None:
    """``recolor_copies`` on a finalized model, before its SolverVBD is built (when the builder is not at hand).

    ``finalize`` also wrote each particle's colour into ``model.particle_colors``, and SolverVBD's spring kernel gives
    a spring's force to the end whose entry there is the colour being solved. So that array is rewritten with the
    groups. Left as built, 73 of 90 copies of a 96-tet block held another colour than their group: a copy then took
    its weld's force at an older iterate or not at all while its hub took all of it, and the two pieces of a cut
    block slid apart on the ground for good (6.5 and 0.5 mm/s; a net 5 mN on a 27 mN piece once converged)."""
    groups = recolored([g.numpy() for g in model.particle_color_groups], model.particle_count, copies)
    model.particle_color_groups = [wp.array(g.astype(np.int32), dtype=wp.int32, device=model.device) for g in groups]
    colors = getattr(model, "particle_colors", None)
    if colors is not None:
        host = colors.numpy().copy()
        for color, ids in enumerate(groups):
            host[ids] = color
        colors.assign(host)


def recolored(groups: Sequence, particle_count: int, copies: TetCopies) -> list[np.ndarray]:
    """The colour groups with every copy moved to its hub's colour + 1 (see ``recolor_copies``)."""
    if len(groups) < 2:
        raise ValueError("recolor_copies: needs at least 2 colours (call color() first)")
    color = np.full(particle_count, -1, np.int64)
    for g, ids in enumerate(groups):
        color[np.asarray(ids, np.int64)] = g
    hub = color[copies.node_hub]
    if (hub < 0).any():
        raise ValueError("recolor_copies: a hub is uncoloured (call color() first)")
    color[copies.node] = ((hub + 1) % len(groups))[:, None]
    dtype = np.asarray(groups[0]).dtype
    return [np.flatnonzero(color == g).astype(dtype) for g in range(len(groups))]


def concat_copies(parts: Sequence[TetCopies]) -> TetCopies:
    """One record from copies added by several ``add_tet_copies`` calls on one builder (global particle, tet and
    spring indices already; hub rows and edge rows are offset)."""
    hubs = np.cumsum([0] + [len(c.node_hub) for c in parts])
    edges = np.cumsum([0] + [len(c.edge) for c in parts])
    cat = lambda f, off=None: np.concatenate([getattr(c, f) + (0 if off is None else off[i])  # noqa: E731
                                              for i, c in enumerate(parts)])
    return TetCopies(cat("tet"), cat("hub"), cat("hub_row", hubs), cat("copy_tet"), cat("material"), cat("mass"),
                     cat("node_hub"), cat("node"), cat("weld"), cat("weld_ke"), cat("edge", hubs),
                     cat("tet_edge", edges), cat("tet_edge_flip"))


def positive_fraction_np(s: np.ndarray) -> np.ndarray:
    """[n] volume fraction of tets on the + side (s >= 0) of a plane, from corner signed distances s [n, 4]."""
    return np.array([_fraction_row(row) for row in np.asarray(s, np.float64)])


def _corner_cut(s: np.ndarray, v: int) -> float:
    """Volume fraction of the corner tet cut off around corner v (alone on its side)."""
    f = 1.0
    for j in range(4):
        if j != v:
            f *= s[v] / (s[v] - s[j])
    return f


def _fraction_row(s: np.ndarray) -> float:
    pos = np.flatnonzero(s >= 0.0)
    if pos.size in (0, 4):
        return float(pos.size == 4)
    if pos.size == 1:
        return _corner_cut(s, int(pos[0]))
    if pos.size == 3:
        return 1.0 - _corner_cut(s, int(np.flatnonzero(s < 0.0)[0]))
    return _prism_fraction(s, int(pos[0]), int(pos[1]))


def _prism_fraction(s: np.ndarray, a: int, b: int) -> float:
    """Two-two split: the + part is the prism (a, p_ac, p_ad) - (b, p_bc, p_bd) in barycentric space."""
    c, d = (j for j in range(4) if j not in (a, b))
    e = np.eye(4)

    def cut(i: int, j: int) -> np.ndarray:
        t = s[i] / (s[i] - s[j])
        return (1.0 - t) * e[i] + t * e[j]

    prism = (e[a], cut(a, c), cut(a, d), e[b], cut(b, c), cut(b, d))
    return sum(_bary_volume(*(prism[i] for i in tet)) for tet in ((0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5)))


def _bary_volume(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Volume of a tet with barycentric corners, as a fraction of the reference tet's."""
    m = np.stack([p1 - p0, p2 - p0, p3 - p0])[:, 1:]
    return abs(float(np.linalg.det(m)))


@wp.func
def corner_cut(s0: float, s1: float, s2: float, s3: float) -> float:
    """Fraction of the corner tet around corner 0 when corner 0 is alone on its side."""
    return (s0 / (s0 - s1)) * (s0 / (s0 - s2)) * (s0 / (s0 - s3))


@wp.func
def bary_det(a: wp.vec4, b: wp.vec4, c: wp.vec4, d: wp.vec4) -> float:
    u = b - a
    v = c - a
    w = d - a
    return wp.abs(wp.determinant(wp.mat33(u[1], u[2], u[3], v[1], v[2], v[3], w[1], w[2], w[3])))


@wp.func
def unit4(i: int) -> wp.vec4:
    v = wp.vec4(0.0)
    v[i] = 1.0
    return v


@wp.func
def cut_point(s: wp.vec4, i: int, j: int) -> wp.vec4:
    t = s[i] / (s[i] - s[j])
    return (1.0 - t) * unit4(i) + t * unit4(j)


@wp.func
def prism_fraction(s: wp.vec4, a: int, b: int, c: int, d: int) -> float:
    p0 = unit4(a)
    p1 = cut_point(s, a, c)
    p2 = cut_point(s, a, d)
    p3 = unit4(b)
    p4 = cut_point(s, b, c)
    p5 = cut_point(s, b, d)
    return bary_det(p0, p1, p2, p3) + bary_det(p1, p2, p3, p4) + bary_det(p2, p3, p4, p5)


@wp.func
def positive_fraction(s: wp.vec4) -> float:
    """Volume fraction of a tet on the + side (s >= 0) of a plane, from its corners' signed distances."""
    npos = int(0)
    for i in range(4):
        if s[i] >= 0.0:
            npos += 1
    if npos == 0:
        return 0.0
    if npos == 4:
        return 1.0
    if npos == 1 or npos == 3:                 # lone corner v on its side
        v = int(0)
        for i in range(4):
            if (npos == 1 and s[i] >= 0.0) or (npos == 3 and s[i] < 0.0):
                v = i
        f = corner_cut(s[v], s[(v + 1) % 4], s[(v + 2) % 4], s[(v + 3) % 4])
        if npos == 1:
            return f
        return 1.0 - f
    a = int(-1)
    b = int(-1)
    c = int(-1)
    d = int(-1)
    for i in range(4):
        if s[i] >= 0.0:
            if a < 0:
                a = i
            else:
                b = i
        else:
            if c < 0:
                c = i
            else:
                d = i
    return prism_fraction(s, a, b, c, d)


vec6 = wp.types.vector(length=6, dtype=wp.float32)


@wp.func
def local_edge(a: int, b: int) -> int:
    """Index in ``LOCAL_EDGES`` of corner pair a < b."""
    if a == 0:
        return b - 1
    if a == 1:
        return b + 1
    return 5


@wp.func
def edge_param(t6: vec6, v: int, j: int) -> float:
    """Crossing parameter along corner v -> corner j, from t6 (each local edge low corner -> high corner)."""
    k = local_edge(wp.min(v, j), wp.max(v, j))
    if v < j:
        return t6[k]
    return 1.0 - t6[k]


@wp.func
def param_point(t6: vec6, i: int, j: int) -> wp.vec4:
    t = edge_param(t6, i, j)
    return (1.0 - t) * unit4(i) + t * unit4(j)


@wp.func
def fraction_from_params(mask: int, t6: vec6) -> float:
    """Volume fraction on the + side (corners in ``mask``) of a tet cut through its edges at t6: a lone corner cuts
    off the product of its three edge parameters; a two-two split, a prism (three tets)."""
    npos = int(0)
    for i in range(4):
        npos += (mask >> i) & 1
    if npos == 0:
        return 0.0
    if npos == 4:
        return 1.0
    if npos == 1 or npos == 3:
        v = int(0)
        for i in range(4):
            if ((mask >> i) & 1) == wp.where(npos == 1, 1, 0):
                v = i
        f = float(1.0)
        for j in range(4):
            if j != v:
                f = f * edge_param(t6, v, j)
        return wp.where(npos == 1, f, 1.0 - f)
    a = int(-1)
    b = int(-1)
    c = int(-1)
    d = int(-1)
    for i in range(4):
        if (mask >> i) & 1 != 0:
            if a < 0:
                a = i
            else:
                b = i
        else:
            if c < 0:
                c = i
            else:
                d = i
    p0 = unit4(a)
    p1 = param_point(t6, a, c)
    p2 = param_point(t6, a, d)
    p3 = unit4(b)
    p4 = param_point(t6, b, c)
    p5 = param_point(t6, b, d)
    return bary_det(p0, p1, p2, p3) + bary_det(p1, p2, p3, p4) + bary_det(p2, p3, p4, p5)
