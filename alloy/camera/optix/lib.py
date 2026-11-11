# SPDX-License-Identifier: Apache-2.0
"""ctypes glue over ``libowf.so`` (built by :mod:`alloy.camera.optix.build`): the OptiX context, one acceleration
structure per unique mesh and the trace launches. Everything runs on Warp's CUDA context and current stream; every
array is a Warp array passed by pointer.

Example::

    from alloy.camera.optix.lib import OptixContext
    ctx = OptixContext("cuda:0")
    k = ctx.add_mesh(mesh)               # wp.Mesh -> GAS index
    h = ctx.gas_handle(k)                # traversable handle (uint64)
"""
from __future__ import annotations

import ctypes
import logging
import threading
from pathlib import Path

import warp as wp

from ..errors import CameraError
from .build import built_files

log = logging.getLogger(__name__)
KIND_CLOSEST, KIND_SHADOW, KIND_LOCAL, KIND_BEVEL, KIND_BEVEL_ANY = 0, 1, 2, 3, 4
BEVEL_MAX_SAMPLES, BEVEL_MAX_HITS = 8, 4  # csrc/owf_params.h
INSTANCE_WORDS = 20  # sizeof(OptixInstance) / 4
_LIB: ctypes.CDLL | None = None
_LOCK = threading.Lock()
_POINTER_FIELDS = ("handles", "queue", "ray_o", "ray_d", "ray_tmax", "ray_world", "world_mask", "hit_t", "hit_u",
                   "hit_v", "hit_prim", "hit_inst", "occluded", "bv_n", "bv_state", "bv_eps", "bv_nhits", "bv_face",
                   "bv_u", "bv_v", "n_dev", "inst_tie")


class LaunchParams(ctypes.Structure):
    """Mirror of csrc/owf_params.h ``OwfParams``: keep the field order and types in sync."""

    _fields_ = [(k, ctypes.c_uint64) for k in _POINTER_FIELDS] + [("n", ctypes.c_int), ("tmin", ctypes.c_float)]


def _declare(lib: ctypes.CDLL) -> None:
    """Argument and result types of the C API."""
    vp, u64, i32 = ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int
    ip = ctypes.POINTER(ctypes.c_int)
    sig = {"owf_last_error": (ctypes.c_char_p, []), "owf_log": (ctypes.c_char_p, [vp]),
           "owf_create": (vp, [vp, ctypes.c_char_p, ctypes.c_size_t, i32]),
           "owf_add_gas": (i32, [vp, u64, i32, u64, i32, vp]), "owf_gas_handle": (u64, [vp, i32]),
           "owf_add_gas_dynamic": (i32, [vp, u64, i32, u64, i32, vp]), "owf_refit_gas": (i32, [vp, i32, i32, vp]),
           "owf_add_gas_rebuild": (i32, [vp, u64, i32, u64, i32, vp]), "owf_rebuild_gas": (i32, [vp, i32, i32, ip, vp]),
           "owf_gas_bytes": (u64, [vp, i32]), "owf_gas_handles_ptr": (u64, [vp]),
           "owf_setup_worlds": (i32, [vp, i32, ip, ip, u64, i32, i32, vp]), "owf_ias_handle": (u64, [vp, i32]),
           "owf_world_handles_ptr": (u64, [vp]), "owf_world_bytes": (u64, [vp]),
           "owf_update_worlds": (i32, [vp, i32, vp]), "owf_trace": (i32, [vp, i32, ctypes.POINTER(LaunchParams), vp])}
    for name, (res, args) in sig.items():
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = res, args
    lib.owf_destroy.argtypes = [vp]


def lib() -> ctypes.CDLL:
    """The loaded ``libowf.so`` (cached). Raises CameraError when it is not built or does not match this module."""
    global _LIB
    with _LOCK:
        if _LIB is None:
            so = built_files()[0]
            if not so.is_file():
                raise CameraError(f"the OptiX helper is not built ({so} is missing)",
                                  "run: python -m alloy.camera.optix build --accept-nvidia-optix-license")
            lb = ctypes.CDLL(str(so))
            _declare(lb)
            if lb.owf_params_size() != ctypes.sizeof(LaunchParams) or lb.owf_instance_size() != 4 * INSTANCE_WORDS:
                raise CameraError("libowf.so does not match alloy.camera.optix.lib (struct sizes differ)",
                                  "rebuild with: python -m alloy.camera.optix build --force")
            _LIB = lb
        return _LIB


def check(rc: int, what: str) -> None:
    """Raise a CameraError with the library's last error when a C call returned non-zero."""
    if rc != 0:
        raise CameraError(f"OptiX {what} failed: {lib().owf_last_error().decode()}",
                          "check that the GPU driver provides OptiX 9 (libnvoptix.so.1) and that the GPU has RT cores")


def ptr(a: wp.array | None) -> int:
    """Device address of a Warp array (0 for None)."""
    return 0 if a is None else int(a.ptr)


class OptixContext:
    """OptiX device context + pipeline on Warp's CUDA context."""

    def __init__(self, device: str | None = None, program: str | Path | None = None, log_level: int = 2):
        self.device = wp.get_device(device)
        lb = lib()  # raises a CameraError that says how to build when the helper is missing
        data = Path(program or built_files()[1]).read_bytes()
        self.h = lb.owf_create(ctypes.c_void_p(int(self.device.context)), data, len(data), log_level)
        if not self.h:
            raise CameraError(f"creating the OptiX context failed: {lib().owf_last_error().decode()}",
                              "check the NVIDIA driver (libnvoptix.so.1) and that the device is an RTX-class GPU")

    @property
    def stream(self) -> ctypes.c_void_p:
        return ctypes.c_void_p(int(wp.get_stream(self.device).cuda_stream))

    def add_mesh(self, mesh: wp.Mesh, dynamic: bool = False) -> int:
        """Build a GAS over a wp.Mesh's point and index arrays (compacted; ``dynamic``: updatable in place, see
        :meth:`refit`); returns its index."""
        n_v, n_t = int(mesh.points.shape[0]), int(mesh.indices.shape[0]) // 3
        build = lib().owf_add_gas_dynamic if dynamic else lib().owf_add_gas
        k = build(self.h, int(mesh.points.ptr), n_v, int(mesh.indices.ptr), n_t, self.stream)
        if k < 0:
            check(-1, "acceleration structure build")
        return k

    def add_rebuildable(self, points: wp.array, indices: wp.array, n_tris: int) -> int:
        """A GAS over vertex / index buffers of ``n_tris`` triangles capacity, rebuilt in place by :meth:`rebuild`."""
        k = lib().owf_add_gas_rebuild(self.h, int(points.ptr), int(points.shape[0]), int(indices.ptr), int(n_tris),
                                      self.stream)
        if k < 0:
            check(-1, "acceleration structure build")
        return k

    def rebuild(self, k0: int, counts) -> None:
        """Rebuild GAS k0, k0 + 1, ... over their first ``counts[i]`` triangles (queued on the device's stream)."""
        n = len(counts)
        arr = (ctypes.c_int * n)(*[int(c) for c in counts])
        check(lib().owf_rebuild_gas(self.h, int(k0), int(k0) + n, arr, self.stream), "GAS rebuild")

    def refit(self, k0: int, k1: int) -> None:
        """Refit the dynamic GAS k0..k1-1 to their meshes' current points (queued on the device's stream)."""
        check(lib().owf_refit_gas(self.h, int(k0), int(k1), self.stream), "GAS refit")

    def gas_handle(self, k: int) -> int:
        return int(lib().owf_gas_handle(self.h, k))

    def trace(self, kind: int, n: int, ray_o: wp.array, ray_d: wp.array, ray_tmax: wp.array, ray_world=None, *,
              handles: int = 0, queue=None, world_mask=None, tmin: float = 0.0, n_dev=None, hits: dict | None = None,
              bevel: dict | None = None) -> None:
        """One launch of ``kind`` over ``n`` rays; ``hits`` maps hit_t/hit_u/hit_v/hit_prim/hit_inst/occluded to
        arrays, ``bevel`` maps bv_* names to arrays (see csrc/owf_params.h)."""
        arrays = {**(hits or {}), **(bevel or {})}
        p = LaunchParams(handles=int(handles), queue=ptr(queue), ray_o=ptr(ray_o), ray_d=ptr(ray_d),
                         ray_tmax=ptr(ray_tmax), ray_world=ptr(ray_world), world_mask=ptr(world_mask),
                         n=int(n), tmin=float(tmin), n_dev=ptr(n_dev), **{k: ptr(v) for k, v in arrays.items()})
        check(lib().owf_trace(self.h, kind, ctypes.byref(p), self.stream), "trace")

    def close(self) -> None:
        if getattr(self, "h", None):
            wp.synchronize_device(self.device)
            lib().owf_destroy(self.h)
            self.h = None

    def __del__(self):
        try:
            self.close()
        except Exception:  # interpreter shutdown
            pass
