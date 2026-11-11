# SPDX-License-Identifier: Apache-2.0
"""Texture SDFs for precision-pair meshes (contact §4.4, §4.7; CT-10 budget).

SDFs are built on the builder side, before ``finalize`` (Newton bakes them into the model there), with
``Mesh.build_sdf(target_voxel_size, narrow_band_range, margin, texture_format, cache_dir)`` (newton
geometry/types.py:860-1000). Newton's on-disk cache is keyed by mesh content and build parameters; the Alloy key
(mesh sha256 + spec digest) is logged and returned for reports. The grid padding is the band edge, never Newton's
0.05 m default (a 30 mm part at 0.1 mm with 5 cm padding needs a 3e9-cell launch, spike addendum). Texture SDFs need
CUDA (newton builder.py:1092-1106).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.contact.types import PrecisionPairSpec
from alloy.contact.validate import SDF_PART_BUDGET_B, mesh_info, sdf_bytes_estimate
from alloy.core.errors import PrecisionSpecError
from alloy.core.ids import digest_of
from alloy.core.log import get_logger

log = get_logger("contact.sdf_cache")

SDF_SCENE_BUDGET_B = 2**30          # CT-10: 1 GB per scene
_MESH_TYPES = (int(newton.GeoType.MESH), int(newton.GeoType.CONVEX_MESH))


@dataclass(frozen=True)
class SdfInfo:
    """One baked SDF."""

    key: str                         # sha256(mesh) + spec digest (first 16 hex each)
    nbytes: int                      # device bytes of the texture data
    estimate_bytes: float            # contact §4.7 formula, for comparison
    voxel: float                     # m
    extent: tuple[float, float, float]


def sdf_key(mesh: newton.Mesh, spec: PrecisionPairSpec) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(mesh.vertices, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(mesh.indices, dtype=np.int32).tobytes())
    params = (spec.voxel_size, tuple(spec.narrow_band), spec.texture_format, spec.gap, spec.margin, spec.mode)
    return f"{h.hexdigest()[:16]}-{digest_of(params)[:16]}"


def _nbytes(obj, seen: set[int] | None = None, depth: int = 0) -> int:
    """Device bytes of the arrays / textures / volumes reachable from ``obj`` (a newton SDF)."""
    seen = set() if seen is None else seen
    if obj is None or id(obj) in seen or depth > 4:
        return 0
    seen.add(id(obj))
    if isinstance(obj, wp.array):
        return int(obj.capacity)
    for attr in ("nbytes", "size_in_bytes"):
        val = getattr(obj, attr, None)
        if isinstance(val, int) and not isinstance(obj, (np.ndarray,)):
            return val
    if isinstance(obj, (int, float, str, bytes, np.ndarray)):
        return 0
    fields = vars(obj).values() if hasattr(obj, "__dict__") else ()
    return sum(_nbytes(v, seen, depth + 1) for v in fields)


def build_feature_sdf(mesh: newton.Mesh, spec: PrecisionPairSpec, *, device: str,
                      cache_dir: Path | None = None) -> SdfInfo:
    """Bake ``mesh``'s SDF in place for ``spec`` and return its size; raises ``PrecisionSpecError`` over 64 MB."""
    if not wp.get_device(device).is_cuda:
        raise PrecisionSpecError("texture SDFs need a CUDA device (newton builder.py:1092-1106)")
    pad = max(spec.narrow_band[1], spec.gap + spec.margin) + 2.0 * spec.voxel_size
    key = sdf_key(mesh, spec)
    mesh.build_sdf(device=device, target_voxel_size=spec.voxel_size, narrow_band_range=tuple(spec.narrow_band),
                   margin=pad, texture_format=spec.texture_format,
                   cache_dir=None if cache_dir is None else str(Path(cache_dir)))
    info_mesh = mesh_info(np.asarray(mesh.vertices), np.asarray(mesh.indices).reshape(-1, 3))
    info = SdfInfo(key=key, nbytes=_nbytes(mesh.sdf), voxel=spec.voxel_size, extent=info_mesh.extent,
                   estimate_bytes=sdf_bytes_estimate(info_mesh.area, spec.narrow_band[1] - spec.narrow_band[0],
                                                     spec.voxel_size))
    log.info("SDF %s: %.2f MB (estimate %.2f MB), voxel %g m, extent %s", key, info.nbytes / 2**20,
             info.estimate_bytes / 2**20, spec.voxel_size, tuple(round(e, 5) for e in info.extent))
    if info.nbytes > SDF_PART_BUDGET_B:
        raise PrecisionSpecError(f"SDF {key}: {info.nbytes / 2**20:.1f} MB > 64 MB per part (CT-10); crop the feature")
    return info


def attach_pair_sdfs(builder: newton.ModelBuilder, spec: PrecisionPairSpec, *, device: str,
                     cache_dir: Path | None = None) -> list[SdfInfo]:
    """Bake SDFs on the mesh shapes of ``spec`` (call after ``add_shadow_shapes`` with the shadow ids, before
    ``finalize``). Primitive shapes keep Newton's own texture SDF (hydroelastic) or analytic narrow phase."""
    out = []
    for shape in (spec.shape_a, spec.shape_b):
        if int(builder.shape_type[shape]) in _MESH_TYPES and spec.analytic_id is None:
            out.append(build_feature_sdf(builder.shape_source[shape], spec, device=device, cache_dir=cache_dir))
    return out


def check_sdf_budget(infos: Iterable[SdfInfo]) -> int:
    """Total bytes of distinct SDFs; ``PrecisionSpecError`` above 1 GB per scene (CT-10)."""
    total = sum({i.key: i.nbytes for i in infos}.values())
    if total > SDF_SCENE_BUDGET_B:
        raise PrecisionSpecError(f"scene SDFs {total / 2**20:.0f} MB > 1 GB (CT-10)")
    return total
