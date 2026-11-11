# SPDX-License-Identifier: Apache-2.0
"""Device side of the camera's deformables (``alloy.camera.deform``): per-world vertex buffers that the scene's
``wp.Mesh`` objects (and OptiX GAS) are built on, refreshed from ``state.particle_q`` before every render.

One render: a gather kernel per object copies its particles of every world into the camera's buffers (grains: 12
icosahedron vertices of the particle's radius each; inactive grains and strays collapse to a point), a bounds kernel
sets each world mesh's box, then every world mesh's BVH is refitted (``wp.Mesh.refit``, OptiX GAS update), and the
full-material tracer's smooth corner normals are recomputed in a fixed order (no atomics: same bits every run). The
caller refits the top level afterwards (``GpuScene.update``).

Example::

    ds = DeformSet(layout, first_mesh=len(sd.meshes), device="cuda:0", model=sim.engine.model)
    sd = ds.scene_data(sd)                       # one mesh per deformable and world, instanced in that world
    gs = GpuScene(sd, device="cuda:0", points=ds.points())
    ds.attach(gs)
    ds.update(sim.engine.state)                  # before gs.update(body_q)
"""
from __future__ import annotations

import logging

import numpy as np
import warp as wp

from .deform import ICO_V, DeformLayout, rest_bounds
from .scene import IDENTITY_XF, MeshGeom, SceneData

log = logging.getLogger(__name__)
ACTIVE = wp.constant(1)  # newton.ParticleFlags.ACTIVE (model.particle_flags is int32)


@wp.kernel(enable_backward=False)
def gather_surface(pq: wp.array(dtype=wp.vec3), start: int, stride: int, out: wp.array2d(dtype=wp.vec3)):
    w, i = wp.tid()
    out[w, i] = pq[start + w * stride + i]


@wp.kernel(enable_backward=False)
def gather_grains(pq: wp.array(dtype=wp.vec3), radius: wp.array(dtype=float), flags: wp.array(dtype=wp.int32),
                  start: int, stride: int, ico: wp.array(dtype=wp.vec3), lo: wp.vec3, hi: wp.vec3,
                  out: wp.array2d(dtype=wp.vec3)):
    w, i = wp.tid()
    p = start + w * stride + i // 12
    x = pq[p]
    r = radius[p]
    inside = x[0] >= lo[0] and x[1] >= lo[1] and x[2] >= lo[2] and x[0] <= hi[0] and x[1] <= hi[1] and x[2] <= hi[2]
    if (flags[p] & ACTIVE) == 0 or not inside:
        r = 0.0
        x = wp.min(wp.max(x, lo), hi)
    out[w, i] = x + ico[i % 12] * r


@wp.kernel(enable_backward=False)
def reset_bounds(first: int, lo: wp.array(dtype=wp.vec3), hi: wp.array(dtype=wp.vec3)):
    w = wp.tid()
    lo[first + w] = wp.vec3(1.0e30)
    hi[first + w] = wp.vec3(-1.0e30)


@wp.kernel(enable_backward=False)
def mesh_bounds(pts: wp.array2d(dtype=wp.vec3), first: int, lo: wp.array(dtype=wp.vec3),
                hi: wp.array(dtype=wp.vec3)):
    w, i = wp.tid()
    wp.atomic_min(lo, first + w, pts[w, i])  # min / max: exact in any order
    wp.atomic_max(hi, first + w, pts[w, i])


@wp.kernel(enable_backward=False)
def vertex_normals(pts: wp.array2d(dtype=wp.vec3), tris: wp.array2d(dtype=wp.int32), vf_off: wp.array(dtype=wp.int32),
                   vf: wp.array(dtype=wp.int32), out: wp.array2d(dtype=wp.vec3)):
    """Area-weighted mean face normal at each vertex, faces summed in a fixed order (CSR ``vf_off`` / ``vf``)."""
    w, v = wp.tid()
    n = wp.vec3(0.0)
    for k in range(vf_off[v], vf_off[v + 1]):
        t = vf[k]
        a = pts[w, tris[t, 0]]
        n += wp.cross(pts[w, tris[t, 1]] - a, pts[w, tris[t, 2]] - a)
    out[w, v] = n


@wp.kernel(enable_backward=False)
def corner_normals(vn: wp.array2d(dtype=wp.vec3), pts: wp.array2d(dtype=wp.vec3), tris: wp.array2d(dtype=wp.int32),
                   tri_off: wp.array(dtype=wp.int32), first: int, out: wp.array(dtype=wp.vec3)):
    """Corner shading normals (ShadeData.tri_n) of world w's copy: the unit vertex normal, the face normal where the
    vertex normal vanishes."""
    w, t, c = wp.tid()
    n = vn[w, tris[t, c]]
    if wp.length(n) <= 1.0e-30:
        a = pts[w, tris[t, 0]]
        n = wp.cross(pts[w, tris[t, 1]] - a, pts[w, tris[t, 2]] - a)
    if wp.length(n) <= 1.0e-30:
        n = wp.vec3(0.0, 0.0, 1.0)
    out[(tri_off[first + w] + t) * 3 + c] = wp.normalize(n)


def vertex_faces(tris: np.ndarray, num_verts: int) -> tuple[np.ndarray, np.ndarray]:
    """CSR (offsets (V + 1,), faces) of the faces around each vertex, in face order."""
    corners = np.asarray(tris, np.int64).reshape(-1)
    order = np.argsort(corners, kind="stable")
    off = np.zeros(num_verts + 1, np.int64)
    np.add.at(off, corners + 1, 1)
    return np.cumsum(off).astype(np.int32), (order // 3).astype(np.int32)


class DeformSet:
    """The deformables of one camera scene: world w's copy of object o is scene mesh ``first[o] + w``."""

    def __init__(self, layout: DeformLayout, first_mesh: int, device: str, model):
        self.layout, self.device, self.model = layout, str(device), model
        w, dev = layout.worlds, self.device
        self.first, k = [], int(first_mesh)
        for o in layout.objects:
            self.first.append(k)
            k += w
        self.buffers = [wp.array(np.ascontiguousarray(np.broadcast_to(o.rest, (w, o.num_verts, 3))), dtype=wp.vec3,
                                 device=dev) for o in layout.objects]
        self.tris = [wp.array(o.tris, dtype=wp.int32, device=dev) for o in layout.objects]
        self.ico = wp.array(ICO_V, dtype=wp.vec3, device=dev)
        lo, hi = rest_bounds(layout)
        self.cull = (wp.vec3(*lo), wp.vec3(*hi))
        self.normals = {}  # object -> (vf_off, vf, vertex normal buffer): surfaces in a full-material scene
        self.gs = self.shade = None
        self.topology = self._topology()  # cutting / tearing plug-ins change it: the scene is then rebuilt

    def _topology(self) -> tuple:
        m = self.model
        tri = getattr(m, "tri_indices", None)
        return (int(m.particle_q.shape[0]), 0 if tri is None else int(tri.shape[0]), 0 if tri is None else tri.ptr)

    def stale(self) -> bool:
        """True when the model's particles or triangles changed since bind (a topology plug-in cut or tore)."""
        return self._topology() != self.topology

    def mesh_ids(self) -> list[int]:
        return [f + w for f in self.first for w in range(self.layout.worlds)]

    def points(self) -> dict[int, wp.array]:
        """{scene mesh index: its vertex buffer (a row of the object's (worlds, V) buffer)} for GpuScene."""
        return {f + w: buf[w] for f, buf in zip(self.first, self.buffers) for w in range(self.layout.worlds)}

    def scene_data(self, sd: SceneData) -> SceneData:
        """``sd`` plus one mesh per deformable and world (rest pose), instanced once in that world, fixed."""
        meshes, rows, textures = list(sd.meshes), [], list(sd.textures)
        for o in self.layout.objects:
            tex = -1 if o.image is None else len(textures)
            textures += [] if o.image is None else [o.image]  # a scanned soft body's photo (scan_soft)
            for w in range(self.layout.worlds):
                rows.append((len(meshes), w, o.color, f"{o.name}#{w}" if w else o.name))
                meshes.append(MeshGeom(f"{o.name}@{w}", o.rest, o.tris, o.uv if tex >= 0 else None, tex))
        k, grp, col, names = zip(*rows)
        n = len(rows)
        return SceneData(meshes, np.r_[sd.inst_mesh, k], np.r_[sd.inst_group, grp], np.r_[sd.inst_body, np.full(n, -1)],
                         np.r_[sd.inst_local, np.tile(IDENTITY_XF, (n, 1))], np.r_[sd.inst_color, np.asarray(col)],
                         sd.num_worlds, list(sd.inst_name) + list(names), sd.lighting, sd.camera,
                         textures, np.r_[sd.inst_textured, np.ones(n, bool)])  # scan textures kept

    def attach(self, gs, shade=None) -> None:
        """Bind to the built GpuScene (and a full-material scene's ShadeData, whose corner normals then follow)."""
        self.gs, self.shade = gs, shade
        gs.deform = self
        if shade is None:
            return
        for i, o in enumerate(self.layout.objects):
            if o.kind == "surface":
                off, vf = vertex_faces(o.tris, o.num_verts)
                self.normals[i] = (wp.array(off, dtype=wp.int32, device=self.device),
                                   wp.array(vf, dtype=wp.int32, device=self.device),
                                   wp.zeros((self.layout.worlds, o.num_verts), dtype=wp.vec3, device=self.device))

    def update(self, state) -> None:
        """Copy every deformable of ``state`` (Newton state: ``particle_q``) into the scene and refit its meshes."""
        pq = getattr(state, "particle_q", None)
        if pq is None or self.gs is None:
            return
        lay, dev, m = self.layout, self.device, self.model
        for i, (o, buf) in enumerate(zip(lay.objects, self.buffers)):
            dim = (lay.worlds, o.num_verts)
            if o.kind == "surface":
                wp.launch(gather_surface, dim=dim, inputs=[pq, o.start, lay.stride], outputs=[buf], device=dev)
            else:
                wp.launch(gather_grains, dim=dim, inputs=[pq, m.particle_radius, m.particle_flags, o.start, lay.stride,
                                                          self.ico, *self.cull], outputs=[buf], device=dev)
            wp.launch(reset_bounds, dim=lay.worlds, inputs=[self.first[i], self.gs.mesh_lo, self.gs.mesh_hi], device=dev)
            wp.launch(mesh_bounds, dim=dim, inputs=[buf, self.first[i], self.gs.mesh_lo, self.gs.mesh_hi], device=dev)
            if i in self.normals:
                off, vf, vn = self.normals[i]
                wp.launch(vertex_normals, dim=dim, inputs=[buf, self.tris[i], off, vf], outputs=[vn], device=dev)
                wp.launch(corner_normals, dim=(lay.worlds, len(o.tris), 3),
                          inputs=[vn, buf, self.tris[i], self.shade.mesh_tri_off, self.first[i]],
                          outputs=[self.shade.tri_n], device=dev)
        for k in self.mesh_ids():
            self.gs.meshes[k].refit()
        if self.gs.optix is not None:
            for f in self.first:
                self.gs.optix.refit_gas(f, f + lay.worlds)
