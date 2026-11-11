# SPDX-License-Identifier: Apache-2.0
"""SPH liquids of an ``alloy.Sim`` in the camera's device scenes: one mesh per liquid and world, rebuilt from the sim's
state before every render (:mod:`alloy.camera.liquid_gpu`), drawn with the liquid's look
(:mod:`alloy.camera.liquid_looks`).

The meshes come last in the scene (after the deformables). Their vertex and index buffers belong to the
:class:`LiquidSet`; the ``wp.Mesh`` of each world is rebuilt over them every render on the Warp backend, and the OptiX
GAS in place (``OptixScene.rebuild_gas``). A full-material scene reserves each mesh's capacity in its shading arrays
and refreshes the corner normals every render.

Example::

    layout = liquid_objects(sim)                       # None without SPH liquids
    ls = LiquidSet(layout, first_mesh=len(sd.meshes), device="cuda:0", engine=sim.engine)
    gs = GpuScene(ls.scene_data(sd), device="cuda:0", liquid=ls.buffers())
    ls.attach(gs)
    ls.update(sim.engine.state)                        # before gs.update(body_q)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import warp as wp

from . import liquid_kernels as lk
from .errors import CameraError
from .liquid_gpu import LiquidObject, LiquidSurface, capacity
from .liquid_looks import fast_color, liquid_key, liquid_material
from .scene import IDENTITY_XF, MeshGeom, SceneData

log = logging.getLogger(__name__)
SPH_ROUTE = "liquid_sph"
_EYE = np.eye(4).tolist()


@dataclass(frozen=True)
class LiquidLayout:
    """Where the liquids live: liquid o of world w is ``state.particle_q[o.start + w * stride :][: o.count]``."""

    worlds: int
    stride: int
    objects: tuple


def _routes(sim) -> dict[str, str]:
    try:
        return {n: str(r["route"]) for n, r in sim.routes().items()}
    except (AttributeError, KeyError, TypeError):
        return {}


def has_liquids(sim) -> bool:
    """True for an ``alloy.Sim`` with an SPH liquid (routes only; cheap)."""
    if getattr(sim, "index", None) is None or not getattr(sim.index, "particles", None):
        return False
    return SPH_ROUTE in _routes(sim).values()


def _material(sim, name: str) -> str:
    from .item_looks import scene_of
    scene = scene_of(sim)
    item = scene.items.get(name) if scene is not None else None
    return liquid_key(item.get("material") if isinstance(item, dict) else None)


def liquid_objects(sim) -> LiquidLayout | None:
    """The SPH liquids of an ``alloy.Sim`` to draw (None for other sources or without any). Raises CameraError when
    the particles are not one block per world."""
    index, engine = getattr(sim, "index", None), getattr(sim, "engine", None)
    if index is None or engine is None or not getattr(index, "particles", None):
        return None
    routes = _routes(sim)
    objs = tuple(LiquidObject(n, r.start, len(r), float(sim.particle_volume(n)) ** (1.0 / 3.0), _material(sim, n))
                 for n, r in index.particles.items() if routes.get(n) == SPH_ROUTE and len(r))
    if not objs:
        return None
    worlds, stride = int(index.worlds), int(index.stride["particle"])
    if int(engine.model.particle_q.shape[0]) != worlds * stride:
        raise CameraError(f"{engine.model.particle_q.shape[0]} particles are not {worlds} worlds of {stride}",
                          "the camera draws liquids only when every world holds its own particles")
    log.info("camera: %d liquid(s) drawn live (%s)", len(objs), ", ".join(f"{o.name}: {o.material}" for o in objs))
    return LiquidLayout(worlds, stride, objs)


@wp.kernel(enable_backward=False)
def _scatter_flags(ids: wp.array(dtype=wp.int32), src: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32)):
    i = wp.tid()
    out[ids[i]] = src[i]


class DeviceFlags:
    """The model's particle flags with each SPH solver's live flags (poured, retired) on the device: a solver that
    steps on a compact view of the model keeps its own copy (``alloy.api._liquid``)."""

    def __init__(self, engine):
        from alloy.api import _liquid
        self.model, self.views = engine.model, []
        for s in _liquid.liquid_solvers(engine):
            ids = _liquid._local_to_global(engine, s)
            if ids is not None and len(ids):
                self.views.append((wp.array(np.asarray(ids, np.int32), dtype=wp.int32, device=engine.model.device), s))
        self.buf = wp.clone(self.model.particle_flags) if self.views else None

    def current(self) -> wp.array:
        if not self.views:
            return self.model.particle_flags
        wp.copy(self.buf, self.model.particle_flags)
        for ids, s in self.views:
            wp.launch(_scatter_flags, dim=ids.shape[0], inputs=[ids, s.model.particle_flags], outputs=[self.buf],
                      device=self.buf.device)
        return self.buf


class LiquidSet:
    """The liquids of one camera scene: world w's mesh of liquid o is scene mesh ``first[o] + w``."""

    def __init__(self, layout: LiquidLayout, first_mesh: int, device: str, engine):
        self.layout, self.device = layout, str(device)
        self.surfaces = [LiquidSurface(o, layout.worlds, layout.stride, self.device) for o in layout.objects]
        self.first = [int(first_mesh) + i * layout.worlds for i in range(len(layout.objects))]
        self.flags = DeviceFlags(engine)
        self.n_particles = int(engine.model.particle_q.shape[0])
        self.gs = self.shade = None

    def stale(self) -> bool:
        """True when the model's particle count changed since bind (the scene is then rebuilt)."""
        return int(self.flags.model.particle_q.shape[0]) != self.n_particles

    def mesh_ids(self) -> list[int]:
        return [f + w for f in self.first for w in range(self.layout.worlds)]

    def buffers(self) -> dict[int, tuple]:
        """{scene mesh: (vertex buffer, index buffer, triangle capacity)} for GpuScene / OptiX."""
        return {f + w: (s.pts[w], s.tris[w], s.tcap) for f, s in zip(self.first, self.surfaces)
                for w in range(self.layout.worlds)}

    def scene_data(self, sd: SceneData) -> SceneData:
        """``sd`` plus one placeholder mesh per liquid and world, instanced once in that world (fast tracer tint)."""
        meshes, rows = list(sd.meshes), []
        for o in self.layout.objects:
            for w in range(self.layout.worlds):
                rows.append((len(meshes), w, fast_color(o.material), f"{o.name}#{w}" if w else o.name))
                meshes.append(MeshGeom(f"{o.name}@{w}", np.zeros((1, 3)), np.zeros((1, 3))))
        k, grp, col, names = zip(*rows)
        n = len(rows)
        return SceneData(meshes, np.r_[sd.inst_mesh, k], np.r_[sd.inst_group, grp], np.r_[sd.inst_body, np.full(n, -1)],
                         np.r_[sd.inst_local, np.tile(IDENTITY_XF, (n, 1))], np.r_[sd.inst_color, np.asarray(col)],
                         sd.num_worlds, list(sd.inst_name) + list(names), sd.lighting, sd.camera,
                         sd.textures, np.r_[sd.inst_textured, np.ones(n, bool)])  # scan textures kept

    def attach(self, gs, shade=None) -> None:
        """Bind to the built GpuScene (and a full-material scene's ShadeData, whose corner normals then follow)."""
        self.gs, self.shade = gs, shade
        gs.liquid = self
        mesh_of = gs.sd.inst_mesh[gs.order]
        self.slots = {k: int(np.flatnonzero(mesh_of == k)[0]) for k in self.mesh_ids()}  # GPU instance of each mesh
        self._ids = gs.geo.inst_mesh.numpy()

    def update(self, state) -> None:
        """Rebuild every liquid surface of ``state`` (Newton state: ``particle_q``) and its meshes' BVHs."""
        pq = getattr(state, "particle_q", None)
        if pq is None or self.gs is None:
            return
        flags, dev, nw = self.flags.current(), self.device, self.layout.worlds
        for f, s in zip(self.first, self.surfaces):
            _, nt = s.build(pq, flags, getattr(state, "particle_qd", None))
            wp.launch(lk.mesh_bounds, dim=nw, inputs=[s.g, s.nt, s.pts, f], outputs=[self.gs.mesh_lo, self.gs.mesh_hi],
                      device=dev)
            if self.shade is not None:
                wp.launch(lk.corner_normals, dim=(nw, int(nt.max()), 3),
                          inputs=[s.vn, s.pts, s.tris, s.nt, s.tcap, self.shade.mesh_tri_off, f],
                          outputs=[self.shade.tri_n], device=dev)
            if self.gs.optix is not None:
                self.gs.optix.rebuild_gas(f, f + nw, nt)
            else:
                self._rebuild_meshes(f, s, nt)
        if self.gs.optix is None:
            self.gs.geo.inst_mesh.assign(self._ids)

    def _rebuild_meshes(self, first: int, s: LiquidSurface, nt: np.ndarray) -> None:
        """New ``wp.Mesh`` objects (BVH built on the device) over world w's first ``nt[w]`` triangles."""
        for w in range(self.layout.worlds):
            m = wp.Mesh(points=s.pts[w], indices=s.tris[w][: 3 * int(nt[w])], bvh_constructor=None)
            self.gs.meshes[first + w] = m
            self._ids[self.slots[first + w]] = m.id

    def surface(self, name: str, world: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Host (verts, tris, vertex normals) of liquid ``name``'s last surface in ``world``."""
        for o, s in zip(self.layout.objects, self.surfaces):
            if o.name == name:
                return s.mesh(world)
        raise CameraError(f"no liquid {name!r} in this scene", f"use one of {[o.name for o in self.layout.objects]}")


def liquid_pack_meshes(layout: LiquidLayout):
    """(meshes, {material: pack material}, {}, present (len(meshes), worlds) bool) of the full-material scene: world w's
    mesh of liquid o is ``meshes[o * worlds + w]`` (a placeholder holding the mesh's capacity), present in world w
    only."""
    from .pbr.pack_io import Mesh
    meshes, mats = [], {}
    for o in layout.objects:
        name, cap = f"liquid_{o.material}", capacity(o)
        mats.setdefault(name, liquid_material(o.material))
        tri, ids = np.zeros((cap, 3), np.int32), np.zeros(cap, np.int32)
        n = np.zeros((cap, 3, 3), np.float32)
        n[..., 2] = 1.0
        for w in range(layout.worlds):
            meta = {"name": f"{o.name}@{w}", "object_name": o.name, "materials": [name], "object_to_world": _EYE,
                    "dynamic": True, "kind": "liquid"}
            meshes.append(Mesh(meta, np.zeros((1, 3), np.float32), tri, n, ids))
    present = np.zeros((len(meshes), layout.worlds), bool)
    present[np.arange(len(meshes)), np.tile(np.arange(layout.worlds), len(layout.objects))] = True
    for m in mats.values():
        m.pop("textures", None)
    return meshes, mats, {}, present


def liquid_surfaces(sim, layout: LiquidLayout, world: int = 0) -> dict[str, tuple]:
    """{liquid: (verts, tris, vertex normals)} of world ``world`` at the sim's current state, built as the camera
    builds them (for scene packs: ``pack_write.write_sim_pack``)."""
    dev = str(sim.engine.model.device)
    flags = DeviceFlags(sim.engine).current()
    out = {}
    for o in layout.objects:
        s = LiquidSurface(o, layout.worlds, layout.stride, dev)
        s.build(sim.engine.state.particle_q, flags, sim.engine.state.particle_qd)
        out[o.name] = s.mesh(world)
    return out
