# SPDX-License-Identifier: Apache-2.0
"""OptiX mirror of a :class:`~alloy.camera.tracer.gpu_scene.GpuScene`: one acceleration structure (GAS) per unique mesh
and one top-level structure (IAS) per group of up to 8 worlds. The instances shared by every world (group
``num_worlds``) are listed first in every IAS with visibility mask 0xFF, then each world's own instances with mask bit
``w % group``; a ray of world ``w`` traces ``handles[w // group]`` with mask ``1 << (w % group)``. Instance ids are the
GpuScene instance indices, so hit records index the scene's arrays directly; barycentrics follow
``wp.mesh_query_ray`` (``P = u p0 + v p1 + (1 - u - v) p2``).

Knobs (environment, defaults in brackets): ``ALLOY_CAMERA_IAS_GROUP`` [8] worlds per IAS (1..8; 1 = one IAS per
world), ``ALLOY_CAMERA_IAS_THREADS`` [8] host threads that submit the per-step IAS refresh,
``ALLOY_CAMERA_IAS_REBUILD_EVERY`` [16] full IAS rebuild every N refreshes (0 = refit only).

Example::

    ox = OptixScene(gpu_scene)       # or gpu_scene.enable_optix()
    ox.refit()                       # after gpu_scene.xf changed
    ox.trace_closest(n, o, d, tmax, world, hit_t, hit_u, hit_v, hit_prim, hit_inst)
"""
from __future__ import annotations

import ctypes
import logging

import numpy as np
import warp as wp

from ..envvars import env_int
from ..errors import CameraError
from .lib import (INSTANCE_WORDS, KIND_BEVEL, KIND_BEVEL_ANY, KIND_CLOSEST, KIND_LOCAL, KIND_SHADOW, OptixContext,
                  check, lib)

log = logging.getLogger(__name__)
_FLAG_DISABLE_ANYHIT = 1 << 2


@wp.kernel(enable_backward=False)
def fill_instances(xf: wp.array(dtype=wp.transform), slot_inst: wp.array(dtype=wp.int32),
                   slot_mask: wp.array(dtype=wp.uint32), gas_lo: wp.array(dtype=wp.uint32),
                   gas_hi: wp.array(dtype=wp.uint32), flags: wp.uint32, out_f: wp.array(dtype=float),
                   out_u: wp.array(dtype=wp.uint32)):
    """OptixInstance[i] for slot i (IAS-major) from the scene's (p, q) world-from-local transforms."""
    i = wp.tid()
    g = slot_inst[i]
    t = xf[g]
    r = wp.quat_to_matrix(wp.transform_get_rotation(t))
    p = wp.transform_get_translation(t)
    b = i * 20
    for row in range(3):
        out_f[b + row * 4 + 0] = r[row, 0]
        out_f[b + row * 4 + 1] = r[row, 1]
        out_f[b + row * 4 + 2] = r[row, 2]
        out_f[b + row * 4 + 3] = p[row]
    out_u[b + 12] = wp.uint32(g)
    out_u[b + 13] = wp.uint32(0)
    out_u[b + 14] = slot_mask[i]
    out_u[b + 15] = flags
    out_u[b + 16] = gas_lo[g]
    out_u[b + 17] = gas_hi[g]
    out_u[b + 18] = wp.uint32(0)
    out_u[b + 19] = wp.uint32(0)


def ias_slots(inst_group: np.ndarray, num_worlds: int, group: int) -> tuple[np.ndarray, np.ndarray, list, list]:
    """Instance slots of every IAS: (instance index, visibility mask, first slot, slot count per IAS).
    ``inst_group``: world of each GpuScene instance, ``num_worlds`` for the shared ones."""
    shared = np.flatnonzero(inst_group == num_worlds)
    slots, masks, off, cnt = [], [], [], []
    for j in range((num_worlds + group - 1) // group):
        off.append(sum(len(s) for s in slots))
        slots.append(shared)
        masks.append(np.full(len(shared), 0xFF, np.uint32))
        for w in range(j * group, min((j + 1) * group, num_worlds)):
            own = np.flatnonzero(inst_group == w)
            slots.append(own)
            masks.append(np.full(len(own), 0xFF if group == 1 else 1 << (w - j * group), np.uint32))
        cnt.append(sum(len(s) for s in slots) - off[-1])
    return np.concatenate(slots).astype(np.int32), np.concatenate(masks).astype(np.uint32), off, cnt


class OptixScene:
    """GAS per unique mesh and one IAS per group of worlds (see the module docstring)."""

    def __init__(self, scene, group: int | None = None, n_threads: int | None = None, allow_update: bool = True,
                 ctx: OptixContext | None = None):
        group = env_int("ALLOY_CAMERA_IAS_GROUP", 8) if group is None else group
        n_threads = env_int("ALLOY_CAMERA_IAS_THREADS", 8) if n_threads is None else n_threads
        if not 1 <= group <= 8:
            raise CameraError(f"IAS group of {group} worlds", "use 1 to 8 worlds per IAS (8 visibility-mask bits)")
        self.sc, self.group, self.allow_update = scene, group, allow_update
        self.ctx = ctx or OptixContext(scene.device)
        dyn, reb = set(getattr(scene, "dynamic", ())), getattr(scene, "rebuilt", {})
        self.gas = [self.ctx.add_rebuildable(*reb[k]) if k in reb else self.ctx.add_mesh(m, k in dyn)
                    for k, m in enumerate(scene.meshes)]
        self._upload_gas_handles(scene)
        self._build_slots(scene, n_threads)
        self.rebuild_every = env_int("ALLOY_CAMERA_IAS_REBUILD_EVERY", 16)
        self.n_refresh = 0
        log.info("optix: %d GAS (%.1f MB), %d IAS for %d worlds (%d per IAS, %.1f MB)", len(self.gas),
                 sum(lib().owf_gas_bytes(self.ctx.h, k) for k in self.gas) / 1e6, self.n_ias, scene.num_worlds, group,
                 lib().owf_world_bytes(self.ctx.h) / 1e6)

    def _upload_gas_handles(self, scene) -> None:
        dev = scene.device
        handles = np.asarray([self.ctx.gas_handle(k) for k in self.gas], np.uint64)[scene.sd.inst_mesh[scene.order]]
        self.gas_lo = wp.array((handles & np.uint64(0xFFFFFFFF)).astype(np.uint32), dtype=wp.uint32, device=dev)
        self.gas_hi = wp.array((handles >> np.uint64(32)).astype(np.uint32), dtype=wp.uint32, device=dev)

    def _build_slots(self, scene, n_threads: int) -> None:
        dev, nw, group = scene.device, scene.num_worlds, self.group
        slots, masks, off, cnt = ias_slots(scene.inst_group, nw, group)
        self.n_ias = len(off)
        self.slot_inst = wp.array(slots, dtype=wp.int32, device=dev)
        self.slot_mask = wp.array(masks, dtype=wp.uint32, device=dev)
        self.inst_buf = wp.zeros(len(slots) * INSTANCE_WORDS, dtype=wp.uint32, device=dev)
        self.inst_f = self.inst_buf.view(float)
        self._fill()
        c_off, c_cnt = (ctypes.c_int * self.n_ias)(*off), (ctypes.c_int * self.n_ias)(*cnt)
        check(lib().owf_setup_worlds(self.ctx.h, self.n_ias, c_off, c_cnt, int(self.inst_buf.ptr),
                                     int(self.allow_update), n_threads, self.ctx.stream), "IAS setup")
        ias = np.asarray([lib().owf_ias_handle(self.ctx.h, j) for j in range(self.n_ias)], np.uint64)
        self.world_handles_wp = wp.array(ias[np.arange(nw) // group], dtype=wp.uint64, device=dev)
        self.world_handles = int(self.world_handles_wp.ptr)
        self.world_mask = None if group == 1 else wp.array((1 << (np.arange(nw) % group)).astype(np.uint32),
                                                           dtype=wp.uint32, device=dev)
        self.gas_handles = int(lib().owf_gas_handles_ptr(self.ctx.h))

    def _fill(self) -> None:
        wp.launch(fill_instances, dim=self.slot_inst.shape[0],
                  inputs=[self.sc.xf, self.slot_inst, self.slot_mask, self.gas_lo, self.gas_hi,
                          wp.uint32(_FLAG_DISABLE_ANYHIT)], outputs=[self.inst_f, self.inst_buf], device=self.sc.device)

    def refit_gas(self, k0: int, k1: int) -> None:
        """Refit the GAS of scene meshes k0..k1-1 (deformables, built updatable) after their points changed."""
        self.ctx.refit(self.gas[k0], self.gas[k1 - 1] + 1)  # built in mesh order: a contiguous GAS range

    def rebuild_gas(self, k0: int, k1: int, counts) -> None:
        """Rebuild the GAS of scene meshes k0..k1-1 (liquids) over their first ``counts[k - k0]`` triangles."""
        if self.gas[k1 - 1] - self.gas[k0] != k1 - 1 - k0:
            raise CameraError("liquid GAS are not contiguous", "build the liquid meshes last (alloy.camera.liquid_scene)")
        self.ctx.rebuild(self.gas[k0], counts)

    def refit(self, rebuild: bool | None = None) -> None:
        """Instances from ``scene.xf``, then refresh every IAS. ``rebuild=None``: update (refit), with a full rebuild
        every ``ALLOY_CAMERA_IAS_REBUILD_EVERY`` refreshes so the BVHs stay good when objects travel far; True / False
        force a rebuild / an update."""
        self._fill()
        self.n_refresh += 1
        if rebuild is None:
            rebuild = self.rebuild_every > 0 and self.n_refresh % self.rebuild_every == 0
        check(lib().owf_update_worlds(self.ctx.h, int(rebuild or not self.allow_update), self.ctx.stream), "IAS update")

    def trace_closest(self, n, ray_o, ray_d, ray_tmax, ray_world, hit_t, hit_u, hit_v, hit_prim, hit_inst,
                      queue=None) -> None:
        """Closest hit of rays 0..n-1 (or of the paths in ``queue``) in each ray's world; ``hit_inst`` = -1 on a miss."""
        hits = {"hit_t": hit_t, "hit_u": hit_u, "hit_v": hit_v, "hit_prim": hit_prim, "hit_inst": hit_inst,
                "inst_tie": self.sc.geo.inst_tie}
        self.ctx.trace(KIND_CLOSEST, n, ray_o, ray_d, ray_tmax, ray_world, hits=hits, queue=queue,
                       handles=self.world_handles, world_mask=self.world_mask)

    def trace_shadow(self, n, ray_o, ray_d, ray_tmax, ray_world, occluded, queue=None) -> None:
        """Any hit before tmax: ``occluded[path]`` = 1 (blocked) or 0."""
        self.ctx.trace(KIND_SHADOW, n, ray_o, ray_d, ray_tmax, ray_world, hits={"occluded": occluded}, queue=queue,
                       handles=self.world_handles, world_mask=self.world_mask)

    def trace_bevel(self, n, queue, bb, n_dev=None, anyhit: bool = True) -> None:
        """Bevel samples (one thread per queued path, csrc/owf_programs.cu ``__raygen__bevel``): ray slots
        ``path * 8 + sample`` of the BevelBuf ``bb``, kept hits per slot at ``slot * 4 + j`` (mesh-local face, Warp
        barycentrics). ``n_dev``: device int with the queue size (``n`` is then an upper bound). ``anyhit``: one
        any-hit traversal per sample ray instead of the closest-hit march (Cycles' OptiX local intersection)."""
        bevel = {"bv_n": bb.n, "bv_state": bb.state, "bv_eps": bb.eps, "bv_nhits": bb.nhits, "bv_face": bb.face,
                 "bv_u": bb.u, "bv_v": bb.v}
        self.ctx.trace(KIND_BEVEL_ANY if anyhit else KIND_BEVEL, n, bb.ray_o, bb.ray_d, bb.ray_tmax, bb.mesh,
                       queue=queue, handles=self.gas_handles, n_dev=n_dev, bevel=bevel)

    def trace_local(self, n, ray_o, ray_d, ray_tmax, ray_mesh, hit_t, hit_u, hit_v, hit_prim, hit_inst,
                    queue=None) -> None:
        """Rays against one mesh's GAS alone (mesh local frame): ``ray_mesh`` = scene mesh index per path."""
        hits = {"hit_t": hit_t, "hit_u": hit_u, "hit_v": hit_v, "hit_prim": hit_prim, "hit_inst": hit_inst}
        self.ctx.trace(KIND_LOCAL, n, ray_o, ray_d, ray_tmax, ray_mesh, hits=hits, queue=queue,
                       handles=self.gas_handles)
