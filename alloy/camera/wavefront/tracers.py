# SPDX-License-Identifier: Apache-2.0
"""Ray tracers of the wavefront: OptiX (``backend="optix"``, see :mod:`alloy.camera.optix`) or Warp's software BVH
(``backend="wavefront-warp"``, the validation mode: the same wavefront reproduces the megakernel to float rounding).

Both offer ``trace_closest``, ``trace_shadow``, ``trace_bevel`` and ``refit``.

Example::

    tracer = make_tracer(gpu_scene, "wavefront-warp")
"""
from __future__ import annotations

import numpy as np
import warp as wp

from ..errors import CameraError
from ..tracer.geometry import Geo, closest_hit_uv, occluded
from .state import path_of

BACKENDS = ("warp", "optix", "wavefront-warp", "auto")


@wp.kernel(enable_backward=False)
def warp_closest(geo: Geo, q: wp.array(dtype=wp.int32), identity: wp.int32, o: wp.array(dtype=wp.vec3),
                 d: wp.array(dtype=wp.vec3), tmax: wp.array(dtype=float), world: wp.array(dtype=wp.int32),
                 hit_t: wp.array(dtype=float), hit_u: wp.array(dtype=float), hit_v: wp.array(dtype=float),
                 hit_prim: wp.array(dtype=wp.int32), hit_inst: wp.array(dtype=wp.int32)):
    p = path_of(q, identity, wp.tid())
    inst, t, n, f, u, v = closest_hit_uv(geo, world[p], o[p], d[p], tmax[p])
    hit_inst[p] = inst
    if inst >= 0:
        hit_t[p] = t
        hit_u[p] = u
        hit_v[p] = v
        hit_prim[p] = f


@wp.kernel(enable_backward=False)
def warp_shadow(geo: Geo, q: wp.array(dtype=wp.int32), o: wp.array(dtype=wp.vec3), d: wp.array(dtype=wp.vec3),
                tmax: wp.array(dtype=float), world: wp.array(dtype=wp.int32), occ: wp.array(dtype=wp.int32)):
    p = q[wp.tid()]
    occ[p] = wp.where(occluded(geo, world[p], o[p], d[p], tmax[p]), 1, 0)


class WarpTracer:
    """Same interface as :class:`~alloy.camera.optix.scene.OptixScene`, traced by Warp's software BVH."""

    def __init__(self, scene):
        self.sc = scene
        self.mesh_ids = wp.array(np.asarray([m.id for m in scene.meshes], np.uint64), dtype=wp.uint64,
                                 device=scene.device)

    def trace_bevel(self, n, queue, bb, n_dev=None, anyhit: bool = True) -> None:
        """Bevel march in Warp (the closest-hit march of the OptiX ``trace`` mode, whatever ``anyhit`` says)."""
        from .bevel import warp_bevel_march
        wp.launch(warp_bevel_march, dim=n, inputs=[self.mesh_ids, queue, n_dev, bb], device=self.sc.device)

    def refit(self, rebuild: bool = False) -> None:
        """Nothing to refresh: the scene's own BVH is refit by the scene."""

    def trace_closest(self, n, ray_o, ray_d, ray_tmax, ray_world, hit_t, hit_u, hit_v, hit_prim, hit_inst,
                      queue=None) -> None:
        wp.launch(warp_closest, dim=n, inputs=[self.sc.geo, queue if queue is not None else ray_world,
                                               int(queue is None), ray_o, ray_d, ray_tmax, ray_world],
                  outputs=[hit_t, hit_u, hit_v, hit_prim, hit_inst], device=self.sc.device)

    def trace_shadow(self, n, ray_o, ray_d, ray_tmax, ray_world, occluded, queue=None) -> None:
        wp.launch(warp_shadow, dim=n, inputs=[self.sc.geo, queue, ray_o, ray_d, ray_tmax, ray_world],
                  outputs=[occluded], device=self.sc.device)


def resolve_backend(backend: str, device: str = "cuda:0") -> str:
    """The concrete backend of a ``RenderSettings.backend`` value: "auto" -> "optix" when it is available, else
    "warp"; anything unknown raises a CameraError.

    Example::

        resolve_backend("auto")        # "warp" on a machine without the OptiX helper
    """
    if backend not in BACKENDS:
        raise CameraError(f"unknown render backend {backend!r}", f"use one of {', '.join(BACKENDS)}")
    if backend != "auto":
        return backend
    from ..optix import available
    return "optix" if available(device)[0] else "warp"


def make_tracer(scene, backend: str):
    """"optix" -> the scene's OptiX mirror (built on first use); "wavefront-warp" -> :class:`WarpTracer`."""
    if backend == "optix":
        from ..optix import available
        ok, why = available(scene.device)
        if not ok:
            raise CameraError(f"backend='optix' is not available: {why}", "use backend='warp', or fix the above")
        return scene.enable_optix()
    if backend == "wavefront-warp":
        return WarpTracer(scene)
    raise CameraError(f"backend {backend!r} has no wavefront tracer", "use 'optix' or 'wavefront-warp'")
