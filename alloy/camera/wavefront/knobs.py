# SPDX-License-Identifier: Apache-2.0
"""Tuning knobs of the wavefront renderer, read from the environment (``ALLOY_CAMERA_WF_*``; defaults = the
fastest settings found in speed tests) or given explicitly as a :class:`WavefrontOptions`.

| variable | default | meaning |
|---|---|---|
| ``ALLOY_CAMERA_WF_BEVEL`` | ``anyhit`` | Bevel rays: ``anyhit`` = one OptiX any-hit traversal per sample ray (Cycles' OptiX local intersection); ``trace`` = the closest-hit march (restart by eps, at most 24 steps) |
| ``ALLOY_CAMERA_WF_BEVEL_FLAT`` | 1 | cast no Bevel rays where the output is provably the input normal (flat regions) |
| ``ALLOY_CAMERA_WF_SPEC`` | 1 | one shade kernel per material, launched over its range of the sorted queue (material constants folded in) |
| ``ALLOY_CAMERA_WF_SPEC_MIN`` | 4096 | neighbouring material ranges smaller than this share one generic-kernel launch |
| ``ALLOY_CAMERA_WF_SHADE_WARPS`` | 12 | shade kernels get ``__launch_bounds__`` for this many resident warps per SM (<= 168 registers); 0 = no bound |
| ``ALLOY_CAMERA_WF_SORT`` | 1 | sort the active paths by material before shading |
| ``ALLOY_CAMERA_WF_SORT_MIN`` | 65536 | queues smaller than this are not sorted |
| ``ALLOY_CAMERA_WF_MAX_PATHS`` | 2^22 below 32 GB, 2^24, 2^25 from 64 GB | paths per wave; lower it to fit a small GPU |
| ``ALLOY_CAMERA_IAS_GROUP`` / ``_THREADS`` / ``_REBUILD_EVERY`` | 8 / 8 / 16 | see :mod:`alloy.camera.optix.scene` |

Example::

    opts = WavefrontOptions.from_env()                # ALLOY_CAMERA_WF_* or the defaults
    opts = WavefrontOptions(max_paths=1 << 22, spec=False)
    renderer = PbrRenderer(scene, 128, 128, settings, wavefront=opts)
"""
from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from ..envvars import env_flag, env_int, env_str
from ..errors import CameraError

BEVEL_MODES = ("anyhit", "trace")


@dataclass(frozen=True)
class WavefrontOptions:
    """Speed knobs of the wavefront (results do not depend on them except through float rounding)."""

    bevel: str = "anyhit"
    bevel_flat: bool = True
    spec: bool = True
    spec_min: int = 4096
    shade_warps: int = 12
    sort: bool = True
    sort_min: int = 1 << 16
    max_paths: int | None = None  # None: ALLOY_CAMERA_WF_MAX_PATHS, else the device default

    def __post_init__(self) -> None:
        if self.bevel not in BEVEL_MODES:
            raise CameraError(f"wavefront Bevel mode {self.bevel!r}", f"use one of {BEVEL_MODES}")

    @staticmethod
    def from_env() -> "WavefrontOptions":
        """Options from ``ALLOY_CAMERA_WF_*`` (unset variables keep the defaults)."""
        d = WavefrontOptions
        mp = env_int("ALLOY_CAMERA_WF_MAX_PATHS", 0)
        return WavefrontOptions(
            bevel=env_str("ALLOY_CAMERA_WF_BEVEL", d.bevel), bevel_flat=env_flag("ALLOY_CAMERA_WF_BEVEL_FLAT", d.bevel_flat),
            spec=env_flag("ALLOY_CAMERA_WF_SPEC", d.spec), spec_min=env_int("ALLOY_CAMERA_WF_SPEC_MIN", d.spec_min),
            shade_warps=env_int("ALLOY_CAMERA_WF_SHADE_WARPS", d.shade_warps),
            sort=env_flag("ALLOY_CAMERA_WF_SORT", d.sort), sort_min=env_int("ALLOY_CAMERA_WF_SORT_MIN", d.sort_min),
            max_paths=mp or None)

    def paths_per_wave(self, device, large_default: bool) -> int:
        """Paths per wave: the option, else by GPU memory: 2^22 below 32 GB (room for physics next to the camera;
        a wave with Bevel buffers takes about 2.5 GB), 2^24 below 64 GB, 2^25 from 64 GB when ``large_default``."""
        if self.max_paths:
            return int(self.max_paths)
        mem = wp.get_device(device).total_memory
        if mem < 32 * (1 << 30):
            return 1 << 22
        return 1 << 25 if large_default and mem >= 64 * (1 << 30) else 1 << 24
