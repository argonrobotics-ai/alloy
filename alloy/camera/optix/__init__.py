# SPDX-License-Identifier: Apache-2.0
"""RT-core ray tracing for the camera's wavefront renderer: OptiX traces, Warp shades.

``RenderSettings(backend="optix")`` needs a one-time build (the NVIDIA OptiX headers are under NVIDIA's SDK licence
and are never shipped)::

    python -m alloy.camera.optix build --accept-nvidia-optix-license

The build function lives in :mod:`alloy.camera.optix.build` (``build.build(accept_licence=True)``).

Example::

    from alloy.camera.optix import available
    ok, reason = available()         # (False, "the OptiX helper is not built; run ...") before the build
"""
from __future__ import annotations

import ctypes

from .build import LICENCE_NOTICE, build_dir, built_files, is_built

__all__ = ["LICENCE_NOTICE", "available", "build_dir", "built_files", "is_built"]
BUILD_HINT = "python -m alloy.camera.optix build --accept-nvidia-optix-license"


def _driver_has_optix() -> bool:
    try:
        ctypes.CDLL("libnvoptix.so.1")
    except OSError:
        return False
    return True


def available(device: str | None = None) -> tuple[bool, str]:
    """(usable, reason). Checks, in order: the helper is built, the driver provides ``libnvoptix.so.1``, a CUDA
    device exists. ``reason`` is "" when usable, else what is missing and what to do.

    Example::

        ok, why = available()
        backend = "optix" if ok else "warp"
    """
    if not is_built():
        return False, f"the OptiX helper is not built; run: {BUILD_HINT}"
    if not _driver_has_optix():
        return False, ("libnvoptix.so.1 was not found: install the NVIDIA driver's GL/OptiX package (on Ubuntu "
                       "servers: libnvidia-gl-<driver version>-server)")
    import warp as wp
    wp.init()
    dev = str(device or "cuda:0")
    if dev.startswith("cpu") or wp.get_cuda_device_count() == 0:
        return False, "no CUDA device: the OptiX backend needs an NVIDIA RTX-class GPU"
    return True, ""
