# SPDX-License-Identifier: Apache-2.0
"""Soft walls: VBD surfaces as SolverDFSPH boundaries (engine item 3, SPH-VBD coupler P1; design note
[internal note]).

Which faces. In a coupled scene VBD is the source and SPH the destination of a particle proxy mapping
(``SolverCoupledProxy.Proxy(source="vbd", destination="sph", particles=<surface vertices>)``). The SPH entry's view
keeps every particle index, flags the mapped vertices ``ParticleFlags.PROXY`` and keeps the model's triangles (a tet
body from ``add_soft_grid`` carries its surface triangles). Every triangle whose three vertices are soft particles (the
proxies, or ``soft_particles`` given to the solver for a standalone run) is a soft face. Faces joined through shared
vertices form a component; component c is the virtual shape ``S + c`` (S = ``model.shape_count``) in the boundary
slots, so the pressure solves, csf, the wall viscosity and the ambient seal read it like any wall. Per-shape tables stop
at S: a virtual shape has a free contact line and no seal (csf.py, ambient.py read past their end as free / unsealed).
A closed component (every edge on two faces) is signed by its outward face normals, its winding fixed by its signed
volume; an open one (cloth) is two-sided with half thickness ``half_thickness`` (P1 judges closed bodies only).

Geometry per step. Face bounds, inflated by the slot reach, go into one ``wp.Bvh`` grouped by world and are refit at
every step (P1 vertices move every substep). A particle's candidates come from ``bvh_query_aabb`` from its world's group
root, the closest point from Newton's ``triangle_closest_point_barycentric``; per component the nearest face wins. The
slot keeps that face and its barycentric weights; the wall velocity is their mix of the vertex velocities.

Motion (P1). The surface is prescribed over the exchange: the solver advances every soft vertex with its own velocity
(lagged sync = begin pose and VBD's end velocity, so the last substep lands on VBD's end pose; implicit Euler); the
rewind hook leaves the velocities alone. The projection pass pushes particles out of a face at its end-of-step place.

Reaction. Each slot's boundary impulse on the wall is split to its face's vertices by barycentric weight and summed per
vertex in int64 fixed point (bodies.py's scale): deterministic, no float atomics. Its sum over an exchange divided by
dt_x is the proxy force the harvest returns. The wall's work on the liquid, -J . v_b, is summed per component.
"""

from __future__ import annotations

import logging

import numpy as np
import warp as wp
from alloy._vendor.newton import ParticleFlags
from alloy._vendor.newton._src.geometry.kernels import triangle_closest_point_barycentric

from .bodies import IMP_SCALE, IMP_SCALE_HOST, WORK_SCALE, _fx, _ufx

log = logging.getLogger("alloy.sph")
WORK_SCALE_HOST = float(2**50)  # bodies.WORK_SCALE


@wp.struct
class SoftArrays:
    base: int  # virtual shape id of component 0 (model.shape_count)
    nface: int  # 0: no soft walls, every soft branch is skipped
    bvh: wp.uint64
    tri: wp.array[wp.vec3i]  # compact vertex ids per face
    face_comp: wp.array[wp.int32]
    face_world: wp.array[wp.int32]
    world_has: wp.array[wp.int32]  # per world: 1 if it has soft faces (a missing group root reads the global root)
    two_sided: wp.array[wp.int32]  # per component
    half: float  # half thickness of two-sided components [m]
    x: wp.array[wp.vec3]  # compact vertex positions at the step start
    v: wp.array[wp.vec3]  # compact vertex velocities (prescribed over the step)
    adj: wp.array[wp.vec3i]  # per face: the face across edge (t0,t1), (t1,t2), (t2,t0); -1 on an open edge
    vstart: wp.array[wp.int32]  # per compact vertex: CSR start into vface
    vface: wp.array[wp.int32]  # incident faces per vertex
    vopen: wp.array[wp.int32]  # per compact vertex: 1 on an open (one-face) edge, the free border of a cloth


@wp.func
def _face_at(soft: SoftArrays, f: int, dt: float):
    """Vertices of face f advanced by dt."""
    t = soft.tri[f]
    return soft.x[t[0]] + dt * soft.v[t[0]], soft.x[t[1]] + dt * soft.v[t[1]], soft.x[t[2]] + dt * soft.v[t[2]]


@wp.func
def _unit_normal(a: wp.vec3, b: wp.vec3, c: wp.vec3) -> wp.vec3:
    nf = wp.cross(b - a, c - a)
    ln = wp.length(nf)
    if ln > 0.0:
        return nf / ln
    return wp.vec3(0.0, 0.0, 1.0)


@wp.func
def _vertex_pseudo_normal(soft: SoftArrays, k: int, dt: float) -> wp.vec3:
    """Angle-weighted normal of compact vertex k (Baerentzen and Aanaes 2005), over its faces in CSR order."""
    nsum = wp.vec3(0.0, 0.0, 0.0)
    for j in range(soft.vstart[k], soft.vstart[k + 1]):
        g = soft.vface[j]
        t = soft.tri[g]
        a, b, c = _face_at(soft, g, dt)
        p, q, r = a, b, c
        if t[1] == k:
            p, q, r = b, c, a
        elif t[2] == k:
            p, q, r = c, a, b
        e1 = q - p
        e2 = r - p
        l1 = wp.length(e1)
        l2 = wp.length(e2)
        if l1 > 0.0 and l2 > 0.0:
            ang = wp.acos(wp.clamp(wp.dot(e1, e2) / (l1 * l2), -1.0, 1.0))
            nsum += ang * _unit_normal(a, b, c)
    return nsum


@wp.func
def _pseudo_normal(soft: SoftArrays, f: int, w: wp.vec3, nf: wp.vec3, dt: float) -> wp.vec3:
    """The normal whose sign is exact for the closest feature: the face's own (interior), the sum of the two faces
    of an edge, or the angle-weighted normal of a vertex. A single face's normal is not: near an edge, on the plane
    of one face, its sign is rounding noise and a particle outside reads inside."""
    nz = int(0)
    for j in range(3):
        if w[j] == 0.0:
            nz += 1
    if nz == 0:
        return nf
    t = soft.tri[f]
    if nz == 1:
        e = int(0)  # zero weight at index 2: edge (t0, t1) = adj 0; at 0: (t1, t2) = adj 1; at 1: (t2, t0) = adj 2
        if w[0] == 0.0:
            e = 1
        elif w[1] == 0.0:
            e = 2
        g = soft.adj[f][e]
        if g < 0:
            return nf
        a, b, c = _face_at(soft, g, dt)
        return nf + _unit_normal(a, b, c)
    k = t[0]
    if w[1] > w[0] and w[1] >= w[2]:
        k = t[1]
    elif w[2] > w[0] and w[2] > w[1]:
        k = t[2]
    return _vertex_pseudo_normal(soft, k, dt)


@wp.func
def soft_closest(soft: SoftArrays, f: int, x: wp.vec3, dt: float):
    """(phi, n, x_b, v_b, bary, dist) of face f for point x, the face advanced by dt along its vertex velocities:
    signed distance (two-sided: |d| - half), outward normal, surface point x - phi n, its velocity, the barycentric
    weights of the closest point on the face and the unsigned distance to it."""
    t = soft.tri[f]
    va, vb, vc = soft.v[t[0]], soft.v[t[1]], soft.v[t[2]]
    a, b, c = _face_at(soft, f, dt)
    w = triangle_closest_point_barycentric(a, b, c, x)
    xc = w[0] * a + w[1] * b + w[2] * c
    nf = _unit_normal(a, b, c)
    d = x - xc
    dist = wp.length(d)
    closed = soft.two_sided[soft.face_comp[f]] == 0
    ns = nf
    if closed:
        ns = _pseudo_normal(soft, f, w, nf, dt)
    side = float(1.0)
    if wp.dot(d, ns) < 0.0:
        side = -1.0
    n = side * nf
    if dist > 1.0e-12:
        n = d / dist
    phi = dist - soft.half
    if closed:
        phi = side * dist
        n = side * n  # outward
    return phi, n, x - phi * n, w[0] * va + w[1] * vb + w[2] * vc, w, dist


@wp.func
def soft_candidates(soft: SoftArrays, x: wp.vec3, w: int):
    """Up to four nearest soft components of world w around x: (component, face, phi) per entry, component -1 empty."""
    cb = wp.vec4i(-1, -1, -1, -1)
    fb = wp.vec4i(-1, -1, -1, -1)
    db = wp.vec4(1.0e30, 1.0e30, 1.0e30, 1.0e30)
    pb = wp.vec4(0.0, 0.0, 0.0, 0.0)
    if soft.nface == 0:
        return cb, fb, pb
    if soft.world_has[w] == 0:
        return cb, fb, pb
    query = wp.bvh_query_aabb(soft.bvh, x, x, wp.bvh_get_group_root(soft.bvh, w))
    f = int(0)
    while wp.bvh_query_next(query, f):
        if soft.face_world[f] != w:
            continue
        phi, _n, _xb, _vb, _bw, dist = soft_closest(soft, f, x, 0.0)
        c = soft.face_comp[f]
        j = int(-1)
        for k in range(4):
            if j < 0 and (cb[k] == c or cb[k] < 0):
                j = k
        if j >= 0 and dist < db[j]:
            cb[j] = c
            fb[j] = f
            db[j] = dist
            pb[j] = phi
    return cb, fb, pb


@wp.kernel(enable_backward=False)
def _gather_kernel(vid: wp.array[wp.int32], q: wp.array[wp.vec3], qd: wp.array[wp.vec3], x: wp.array[wp.vec3],
                   v: wp.array[wp.vec3]):
    k = wp.tid()
    x[k] = q[vid[k]]
    v[k] = qd[vid[k]]


@wp.kernel(enable_backward=False)
def _bounds_kernel(soft: SoftArrays, pad: float, lo: wp.array[wp.vec3], hi: wp.array[wp.vec3]):
    f = wp.tid()
    t = soft.tri[f]
    a, b, c = soft.x[t[0]], soft.x[t[1]], soft.x[t[2]]
    e = wp.vec3(pad, pad, pad)
    lo[f] = wp.min(wp.min(a, b), c) - e
    hi[f] = wp.max(wp.max(a, b), c) + e


@wp.kernel(enable_backward=False)
def _advance_kernel(vid: wp.array[wp.int32], dt: float, q: wp.array[wp.vec3], qd: wp.array[wp.vec3]):
    k = wp.tid()
    i = vid[k]
    q[i] = q[i] + dt * qd[i]


@wp.kernel(enable_backward=False)
def _accumulate_kernel(
    slot_shape: wp.array2d[wp.int32], slot_face: wp.array2d[wp.int32], slot_bary: wp.array2d[wp.vec3],
    slot_vb: wp.array2d[wp.vec3], slot_impulse: wp.array2d[wp.vec3], soft: SoftArrays, one_way: int,
    exch: wp.array2d[wp.int64], work: wp.array[wp.int64],
):
    i, k = wp.tid()
    s = slot_shape[i, k]
    if s < soft.base:
        return
    J = slot_impulse[i, k]
    if J[0] == 0.0 and J[1] == 0.0 and J[2] == 0.0:
        return
    wp.atomic_add(work, s - soft.base, _fx(-wp.dot(J, slot_vb[i, k]), WORK_SCALE))
    if one_way != 0:
        return
    t = soft.tri[slot_face[i, k]]
    bw = slot_bary[i, k]
    for a in range(3):
        for d in range(3):
            wp.atomic_add(exch, t[a], d, _fx(J[d] * bw[a], IMP_SCALE))


@wp.kernel(enable_backward=False)
def _harvest_kernel(vid: wp.array[wp.int32], local_to_global: wp.array[int], exch: wp.array2d[wp.int64], dt: float,
                    out: wp.array[wp.vec3]):
    """The exchange-averaged fluid force on each proxy vertex, written directly (bodies.harvest_kernel's MPM pattern)."""
    k = wp.tid()
    g = local_to_global[vid[k]]
    if g < 0:
        return
    out[g] = wp.vec3(_ufx(exch[k, 0], IMP_SCALE), _ufx(exch[k, 1], IMP_SCALE), _ufx(exch[k, 2], IMP_SCALE)) / dt


def soft_topology(model, soft_particles=None) -> dict | None:
    """Host: soft faces (compact vertex ids, winding fixed), components, closedness, face worlds; None if none."""
    P = int(model.particle_count)
    if P == 0 or int(getattr(model, "tri_count", 0) or 0) == 0:
        return None
    mask = np.zeros(P, bool)
    if soft_particles is None:
        mask = (model.particle_flags.numpy() & int(ParticleFlags.PROXY)) != 0
    else:
        mask[np.asarray(list(soft_particles), dtype=np.int64)] = True
    tri = model.tri_indices.numpy().astype(np.int64)
    tri = tri[mask[tri].all(axis=1)]
    if len(tri) == 0:
        return None
    vid, tri_c = np.unique(tri, return_inverse=True)
    tri_c = tri_c.reshape(-1, 3)
    comp = _components(tri_c, len(vid))
    x = model.particle_q.numpy().astype(np.float64)[vid]
    C = int(comp.max()) + 1
    two_sided = np.zeros(C, np.int32)
    for c in range(C):
        fc = np.nonzero(comp == c)[0]
        if not _closed(tri_c[fc]):
            two_sided[c] = 1
            continue
        a, b, d = x[tri_c[fc, 0]], x[tri_c[fc, 1]], x[tri_c[fc, 2]]
        if np.einsum("ij,ij->i", a, np.cross(b, d)).sum() < 0.0:  # signed volume: wound inward
            tri_c[fc] = tri_c[fc][:, [0, 2, 1]]
    world = model.particle_world.numpy()[vid[tri_c[:, 0]]].astype(np.int32)
    if (world < 0).any() and int(model.world_count) > 1:
        raise ValueError("SolverDFSPH: soft faces on global particles (world -1) in a multi-world model; build each "
                         "soft body inside its world (begin_world / end_world)")
    adj, vstart, vface = _adjacency(tri_c, len(vid))
    vopen = np.zeros(len(vid), np.int32)  # vertices on an open edge (the free border of a cloth)
    for e in range(3):
        o = adj[:, e] < 0
        vopen[tri_c[o, e]] = 1
        vopen[tri_c[o, (e + 1) % 3]] = 1
    return {"adj": adj, "vstart": vstart, "vface": vface, "vopen": vopen, "vid": vid.astype(np.int32),
            "tri": tri_c.astype(np.int32), "comp": comp.astype(np.int32), "two_sided": two_sided,
            "world": np.maximum(world, 0)}


def _components(tri: np.ndarray, nv: int) -> np.ndarray:
    """Component id per face (faces joined through shared vertices), numbered by first face."""
    parent = np.arange(nv)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for t in tri:
        r0 = find(int(t[0]))
        for v in t[1:]:
            r = find(int(v))
            if r != r0:
                parent[r] = r0
    roots = np.array([find(int(t[0])) for t in tri])
    _u, first = np.unique(roots, return_index=True)
    order = {int(roots[i]): n for n, i in enumerate(sorted(first))}
    return np.array([order[int(r)] for r in roots], np.int64)


def _adjacency(tri: np.ndarray, nv: int):
    """Per face the face across each edge (-1: open or non-manifold) and the vertex -> faces CSR."""
    owner: dict[tuple[int, int], list[int]] = {}
    for f, t in enumerate(tri):
        for e in range(3):
            key = tuple(sorted((int(t[e]), int(t[(e + 1) % 3]))))
            owner.setdefault(key, []).append(f)
    adj = np.full((len(tri), 3), -1, np.int32)
    for f, t in enumerate(tri):
        for e in range(3):
            fs = owner[tuple(sorted((int(t[e]), int(t[(e + 1) % 3]))))]
            if len(fs) == 2:
                adj[f, e] = fs[0] if fs[1] == f else fs[1]
    inc = [[] for _ in range(nv)]
    for f, t in enumerate(tri):
        for v in t:
            inc[int(v)].append(f)
    vstart = np.zeros(nv + 1, np.int32)
    vstart[1:] = np.cumsum([len(x) for x in inc])
    vface = np.array([f for x in inc for f in x], np.int32)
    return adj, vstart, vface


def _closed(tri: np.ndarray) -> bool:
    e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    _u, cnt = np.unique(e, axis=0, return_counts=True)
    return bool((cnt == 2).all())


def _dummy(device, dtype, value=0):
    return wp.full(1, value, dtype=dtype, device=device)


def empty_soft(base: int, device) -> SoftArrays:
    """SoftArrays with no faces (every soft branch skipped)."""
    a = SoftArrays()
    a.base, a.nface, a.half, a.bvh = int(base), 0, 0.0, wp.uint64(0)
    a.tri, a.x, a.v = _dummy(device, wp.vec3i), _dummy(device, wp.vec3), _dummy(device, wp.vec3)
    a.face_comp, a.face_world = _dummy(device, wp.int32), _dummy(device, wp.int32)
    a.world_has, a.two_sided = _dummy(device, wp.int32), _dummy(device, wp.int32)
    a.adj, a.vstart, a.vface = _dummy(device, wp.vec3i, -1), _dummy(device, wp.int32), _dummy(device, wp.int32)
    a.vopen = _dummy(device, wp.int32)
    return a


class SoftWalls:
    """Soft faces of one SolverDFSPH: topology, the grouped BVH, per-step gather / refit / advance, reactions."""

    def __init__(self, model, reach: float, device, soft_particles=None, half_thickness: float = 0.0):
        self.device = device
        self.base = int(model.shape_count)
        top = soft_topology(model, soft_particles)
        a = empty_soft(self.base, device)
        a.half = float(half_thickness)
        self.count = 0 if top is None else len(top["tri"])
        self.components = 0 if top is None else int(top["comp"].max()) + 1
        self.pad = float(reach) + float(half_thickness)
        if top is None:
            self.arrays = a
            self.work = wp.zeros(1, dtype=wp.int64, device=device)
            return
        V, W = len(top["vid"]), max(int(model.world_count), 1)
        self.vid = wp.array(top["vid"], dtype=wp.int32, device=device)
        a.nface = self.count
        a.tri = wp.array(top["tri"], dtype=wp.vec3i, device=device)
        a.face_comp = wp.array(top["comp"], dtype=wp.int32, device=device)
        a.face_world = wp.array(top["world"], dtype=wp.int32, device=device)
        has = np.zeros(W, np.int32)
        has[np.unique(top["world"])] = 1
        a.world_has = wp.array(has, dtype=wp.int32, device=device)
        a.two_sided = wp.array(top["two_sided"], dtype=wp.int32, device=device)
        a.adj = wp.array(top["adj"], dtype=wp.vec3i, device=device)
        a.vstart = wp.array(top["vstart"], dtype=wp.int32, device=device)
        a.vface = wp.array(top["vface"], dtype=wp.int32, device=device)
        a.vopen = wp.array(top["vopen"], dtype=wp.int32, device=device)
        a.x = wp.array(model.particle_q.numpy()[top["vid"]], dtype=wp.vec3, device=device)
        a.v = wp.zeros(V, dtype=wp.vec3, device=device)
        self._lo = wp.zeros(self.count, dtype=wp.vec3, device=device)
        self._hi = wp.zeros(self.count, dtype=wp.vec3, device=device)
        self.arrays = a
        wp.launch(_bounds_kernel, dim=self.count, inputs=[a, self.pad, self._lo, self._hi], device=device)
        self.bvh = wp.Bvh(self._lo, self._hi, groups=a.face_world)
        a.bvh = self.bvh.id
        self.exch = wp.zeros((V, 3), dtype=wp.int64, device=device)
        self.work = wp.zeros(self.components, dtype=wp.int64, device=device)
        log.info("SolverDFSPH: %d soft faces on %d vertices in %d components (%d two-sided)", self.count, V,
                 self.components, int(top["two_sided"].sum()))

    def begin(self, state) -> None:
        """Step start: gather the soft vertices' positions and velocities, refit the face bounds."""
        if self.count == 0:
            return
        a = self.arrays
        wp.launch(_gather_kernel, dim=self.vid.shape[0], inputs=[self.vid, state.particle_q, state.particle_qd, a.x, a.v],
                  device=self.device)
        wp.launch(_bounds_kernel, dim=self.count, inputs=[a, self.pad, self._lo, self._hi], device=self.device)
        self.bvh.refit()
        self.work.zero_()

    def advance(self, state, dt: float) -> None:
        """Move the soft vertices with their own velocities (P1: prescribed over the exchange)."""
        if self.count:
            wp.launch(_advance_kernel, dim=self.vid.shape[0], inputs=[self.vid, float(dt), state.particle_q,
                                                                      state.particle_qd], device=self.device)

    def accumulate(self, slots, slot_impulse, n: int, one_way: bool) -> None:
        if self.count and n:
            wp.launch(_accumulate_kernel, dim=(n, slots.shape.shape[1]), device=self.device,
                      inputs=[slots.shape, slots.face, slots.bary, slots.vb, slot_impulse, self.arrays, int(one_way),
                              self.exch, self.work])

    def new_window(self) -> None:
        """A new exchange impulse window (every coupler notification)."""
        if self.count:
            self.exch.zero_()

    def harvest(self, local_to_global, out, dt: float) -> None:
        if self.count:
            wp.launch(_harvest_kernel, dim=self.vid.shape[0], inputs=[self.vid, local_to_global, self.exch, float(dt),
                                                                      out], device=self.device)

    def vertex_impulse(self) -> np.ndarray:
        """Host: the exchange window's impulse [N s] per soft vertex (compact order, ``vid``)."""
        return self.exch.numpy().astype(np.float64) / IMP_SCALE_HOST if self.count else np.zeros((0, 3))

    def component_work(self) -> np.ndarray:
        """Host: the last step's work [J] of each soft component on the liquid."""
        return self.work.numpy().astype(np.float64) / WORK_SCALE_HOST if self.count else np.zeros(0)
