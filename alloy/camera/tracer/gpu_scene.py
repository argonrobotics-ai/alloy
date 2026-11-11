# SPDX-License-Identifier: Apache-2.0
"""Device copy of a :class:`~alloy.camera.scene.SceneData`: one ``wp.Mesh`` per unique mesh, instances sorted by world
group, and a grouped top-level ``wp.Bvh`` (group w = world w, group num_worlds = shapes shared by every world).

Body-bound instances move with ``state.body_q``: :meth:`GpuScene.update` writes every instance transform in one
kernel launch and refits (never rebuilds) the top-level BVH.

Example::

    sc = GpuScene(scene_data, device="cuda:0")
    sc.update(state.body_q)        # after each physics step
    sc.enable_optix()              # RT-core tracing (backend="optix"); update() then also refreshes its IAS
"""
from __future__ import annotations

import logging

import numpy as np
import warp as wp

from ..scene import SceneData
from .geometry import Geo, group_roots_kernel, instance_bounds

log = logging.getLogger(__name__)
TIE_TOUCH = 1.0e-3  # metres: bounds this close count as touching (set_transmissive)


@wp.kernel
def pose_instances(body_q: wp.array(dtype=wp.transform), inst_body: wp.array(dtype=wp.int32),
                   inst_local: wp.array(dtype=wp.transform), out: wp.array(dtype=wp.transform)):
    """World-from-mesh transform of every instance: body pose x local pose (fixed instances keep the local pose)."""
    i = wp.tid()
    b = inst_body[i]
    if b >= 0:
        out[i] = wp.transform_multiply(body_q[b], inst_local[i])
    else:
        out[i] = inst_local[i]


class GpuScene:
    """N worlds of one scene on one device; ``geo`` is the struct the kernels trace against."""

    def __init__(self, sd: SceneData, device: str = "cuda:0", bvh_constructor: str = "sah", points: dict | None = None,
                 liquid: dict | None = None):
        """``points``: {mesh index: device vec3 array} vertex buffers to build those meshes on (deformables, refitted
        in place every render: see ``alloy.camera.deform_gpu``); ``liquid``: {mesh index: (vertex buffer, index buffer,
        triangle capacity)} of meshes whose topology changes every render (``alloy.camera.liquid_scene``: rebuilt, never
        refitted); other meshes get their own copy."""
        self.sd, self.device, self.num_worlds = sd, str(device), int(sd.num_worlds)
        self.dynamic = sorted(points or {})  # meshes whose points change (OptiX builds their GAS updatable)
        self.rebuilt = dict(liquid or {})  # meshes rebuilt every render (OptiX: a GAS buffer of their capacity)
        self.deform = None  # DeformSet, set by its attach()
        self.liquid = None  # LiquidSet, set by its attach()
        groups = np.where(sd.inst_group < 0, self.num_worlds, sd.inst_group)
        self.order = np.argsort(groups, kind="stable").astype(np.int32)  # GPU instance i = SceneData instance order[i]
        self.meshes = [self._mesh(m, (points or {}).get(k), self.rebuilt.get(k)) for k, m in enumerate(sd.meshes)]
        lo = np.stack([m.points.min(0) if len(m.points) else np.zeros(3) for m in sd.meshes]).astype(np.float32)
        hi = np.stack([m.points.max(0) if len(m.points) else np.zeros(3) for m in sd.meshes]).astype(np.float32)
        o, dev = self.order, self.device
        n = len(o)
        self.mesh_lo = wp.array(lo, dtype=wp.vec3, device=dev)
        self.mesh_hi = wp.array(hi, dtype=wp.vec3, device=dev)
        self.inst_group = groups[o].astype(np.int32)  # host copy: world of each GPU instance (num_worlds = shared)
        self.inst_src = wp.array(sd.inst_mesh[o], dtype=wp.int32, device=dev)
        self.inst_body = wp.array(sd.inst_body[o], dtype=wp.int32, device=dev)
        self.inst_local = wp.array(sd.inst_local[o], dtype=wp.transform, device=dev)
        self.has_bodies = bool((sd.inst_body >= 0).any())
        self.xf = wp.array(sd.inst_local[o], dtype=wp.transform, device=dev)
        self.inv = wp.zeros(n, dtype=wp.transform, device=dev)
        lowers, uppers = wp.zeros(n, dtype=wp.vec3, device=dev), wp.zeros(n, dtype=wp.vec3, device=dev)
        self._bounds(lowers, uppers)
        self.bvh = wp.Bvh(lowers, uppers, constructor=bvh_constructor, leaf_size=1,
                          groups=wp.array(groups[o].astype(np.int32), dtype=wp.int32, device=dev))
        self.group_roots = wp.zeros(self.num_worlds + 1, dtype=wp.int32, device=dev)
        wp.launch(group_roots_kernel, dim=self.num_worlds + 1, inputs=[self.bvh.id, self.group_roots], device=dev)
        self.optix = None  # OptixScene once enable_optix() ran
        self.geo = self._geo(np.array([m.id for m in self.meshes], np.uint64), bool((groups == self.num_worlds).any()))
        log.info("camera scene: %d meshes (%d tris), %d worlds, %d instances", len(self.meshes),
                 sum(len(m.tris) for m in sd.meshes), self.num_worlds, n)

    def _mesh(self, m, pts: wp.array | None = None, rebuilt: tuple | None = None) -> wp.Mesh:
        if rebuilt is not None:  # a placeholder over the live buffers (one triangle) until the first rebuild
            return wp.Mesh(points=rebuilt[0], indices=rebuilt[1][:3], bvh_constructor=None)
        pts = wp.array(m.points, dtype=wp.vec3, device=self.device) if pts is None else pts
        return wp.Mesh(points=pts,
                       indices=wp.array(m.tris.reshape(-1), dtype=wp.int32, device=self.device), bvh_constructor="sah")

    def _geo(self, mesh_ids: np.ndarray, use_global: bool) -> Geo:
        geo = Geo()
        geo.bvh = self.bvh.id
        geo.group_roots = self.group_roots
        geo.inst_mesh = wp.array(mesh_ids[self.sd.inst_mesh[self.order]], dtype=wp.uint64, device=self.device)
        geo.inst_xf = self.xf
        geo.inst_inv = self.inv
        geo.inst_mat = wp.array(np.arange(self.num_instances, dtype=np.int32), dtype=wp.int32, device=self.device)
        geo.inst_tie = wp.zeros(self.num_instances, dtype=wp.int32, device=self.device)  # set_transmissive
        geo.num_worlds = self.num_worlds
        geo.use_global = 1 if use_global else 0
        return geo

    def set_transmissive(self, mesh_flags) -> None:
        """Mark the instances of meshes with ``mesh_flags[mesh]`` true (glass, water) as transmissive (1) for the
        distance-tie rule (``geometry.wins_tie``), and opaque instances whose bounds touch one of them (a table under a
        glass) with 2: the OptiX trace checks ties only on those. The array is ``geo.inst_tie``.

        Example: ``scene.set_transmissive([False, True])`` for SceneData meshes (table, glass)."""
        flags = np.asarray(mesh_flags, bool)[self.sd.inst_mesh[self.order]]
        tie = flags.astype(np.int32)
        if flags.any() and not flags.all():
            lo, hi = self.bvh.lowers.numpy(), self.bvh.uppers.numpy()
            touch = np.zeros(len(flags), bool)
            for j in np.flatnonzero(flags):
                touch |= ((lo <= hi[j] + TIE_TOUCH) & (hi >= lo[j] - TIE_TOUCH)).all(axis=1)
            tie[touch & ~flags] = 2
        self.geo.inst_tie = wp.array(tie, dtype=wp.int32, device=self.device)

    def enable_optix(self):
        """Attach (once) the OptiX mirror of this scene; ``update`` / ``refit`` / ``set_transforms`` then refresh it.
        Raises CameraError when the OptiX helper is not built (see :mod:`alloy.camera.optix`)."""
        if self.optix is None:
            from ..optix.scene import OptixScene
            self.optix = OptixScene(self)
        return self.optix

    @property
    def num_instances(self) -> int:
        return len(self.order)

    def _bounds(self, lowers: wp.array, uppers: wp.array) -> None:
        wp.launch(instance_bounds, dim=self.num_instances,
                  inputs=[self.mesh_lo, self.mesh_hi, self.inst_src, self.xf, self.inv, lowers, uppers],
                  device=self.device)

    def update(self, body_q: wp.array | None) -> None:
        """Move body-bound instances to ``body_q`` (Newton ``state.body_q``) and refit the top-level BVH (and the
        OptiX IAS when attached)."""
        if body_q is not None and self.has_bodies:
            wp.launch(pose_instances, dim=self.num_instances, inputs=[body_q, self.inst_body, self.inst_local],
                      outputs=[self.xf], device=self.device)
        self.refit()

    def set_transforms(self, xf: np.ndarray | wp.array) -> None:
        """Set every instance's world-from-mesh transform directly ((I, 7) in GPU instance order), then refit."""
        if isinstance(xf, wp.array):
            wp.copy(self.xf, xf)
        else:
            self.xf.assign(np.asarray(xf, np.float32))
        self.refit()

    def world_bounds(self, world: int = 0, max_size: float = 15.0) -> tuple[np.ndarray, np.ndarray]:
        """World-space (lo, hi) of world ``world``'s instances (and shared ones) at their current poses; instances
        larger than ``max_size`` metres (ground planes) are left out."""
        lo, hi = self.bvh.lowers.numpy(), self.bvh.uppers.numpy()
        groups = np.where(self.sd.inst_group < 0, self.num_worlds, self.sd.inst_group)[self.order]
        keep = ((groups == world) | (groups == self.num_worlds)) & ((hi - lo).max(axis=1) <= max_size)
        if not keep.any():
            return np.zeros(3), np.ones(3)
        return lo[keep].min(0), hi[keep].max(0)

    def refit(self) -> None:
        self._bounds(self.bvh.lowers, self.bvh.uppers)
        self.bvh.refit()
        if self.optix is not None:
            self.optix.refit()
