# SPDX-License-Identifier: Apache-2.0
"""The SB-7 scorer's model collides mesh geoms as their convex hulls, as MuJoCo does (Alloy venv).

MuJoCo has no concave mesh collision: every mesh geom collides as its convex hull (qhull, capped at the mesh's
``maxhullvert``), so that is the geometry every engine's S1 rows are judged by. Newton's MJCF importer keeps them as
``GeoType.MESH`` (triangle-mesh BVH collision). On T12 episode (100000, 381), frame 2118, that mesh-mesh path reported
-14.07 mm between ``left_hull_linear_tip_left_4`` and ``right_hull_model2__10_2``; MuJoCo's ``mj_geomDistance`` and a
Frank-Wolfe distance on the same vertices put them +1.08 mm apart (WP3: the path composes body and shape frames wrongly
when both are non-identity; the Newton bug goes to WP0). As ``CONVEX_MESH`` the pair gives +1.08 mm.

``meshes_as_hulls`` turns every colliding ``MESH`` shape into ``CONVEX_MESH`` holding its hull
(``Mesh.compute_convex_hull``, the rule of Newton's ``approximate_meshes("convex_hull")``), one hull per shared mesh so
replicated worlds still share it. It runs on the builder just before ``finalize``, which precomputes per-type collision
data (convex support tables and hull adjacency, mesh BVHs, open-boundary flags); flipping ``shape_type`` on a finalized
model would leave those stale. The target's builder is internal to its ``make_sim``, hence the scoped hook on
``ModelBuilder.finalize``. Shapes with SDF or hydroelastic state (Alloy's precision pairs) keep ``MESH``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from alloy._vendor import newton
from alloy._vendor.newton import GeoType, ShapeFlags


@dataclass
class HullCount:
    """One ``finalize``'s conversion."""

    shapes: int = 0      # colliding MESH shapes now CONVEX_MESH (all worlds)
    meshes: int = 0      # distinct meshes hulled
    kept_sdf: int = 0    # colliding MESH shapes left as MESH: SDF / hydroelastic state


def _has_sdf(b: newton.ModelBuilder, i: int) -> bool:
    """SDF / hydroelastic state a convex hull would drop (``approximate_meshes``' own guard)."""
    src = b.shape_source[i]
    return (b.shape_sdf_max_resolution[i] is not None or b.shape_sdf_target_voxel_size[i] is not None
            or b.shape_sdf_padding[i] is not None or bool(b.shape_flags[i] & ShapeFlags.HYDROELASTIC)
            or getattr(src, "sdf", None) is not None)


def hull_builder_meshes(b: newton.ModelBuilder) -> HullCount:
    """Turn ``b``'s colliding MESH shapes (SDF / hydroelastic ones excepted) into CONVEX_MESH with their hulls."""
    hulls: dict[int, tuple[newton.Mesh, newton.Mesh]] = {}   # id(source) -> (source kept alive, its hull)
    n = HullCount()
    for i, kind in enumerate(b.shape_type):
        if kind != GeoType.MESH or not b.shape_flags[i] & ShapeFlags.COLLIDE_SHAPES:
            continue
        if _has_sdf(b, i):
            n.kept_sdf += 1
            continue
        src = b.shape_source[i]
        if id(src) not in hulls:
            hulls[id(src)] = (src, src.compute_convex_hull())
        b.shape_source[i] = hulls[id(src)][1]
        b.shape_type[i] = GeoType.CONVEX_MESH
        n.shapes += 1
    n.meshes = len(hulls)
    return n


@contextlib.contextmanager
def meshes_as_hulls() -> Iterator[list[HullCount]]:
    """Within the block every ``ModelBuilder.finalize`` first runs ``hull_builder_meshes``; yields their counts."""
    counts: list[HullCount] = []
    finalize = newton.ModelBuilder.finalize

    def hulled(self, *args, **kwargs):
        counts.append(hull_builder_meshes(self))
        return finalize(self, *args, **kwargs)

    newton.ModelBuilder.finalize = hulled
    try:
        yield counts
    finally:
        newton.ModelBuilder.finalize = finalize


def colliding_mesh_shapes(model: newton.Model) -> np.ndarray:
    """Indices of the model's shapes that collide through Newton's triangle-mesh (MESH) path."""
    kind, flags = model.shape_type.numpy(), model.shape_flags.numpy()
    return np.flatnonzero((kind == int(GeoType.MESH)) & ((flags & int(ShapeFlags.COLLIDE_SHAPES)) != 0))


__all__ = ["HullCount", "colliding_mesh_shapes", "hull_builder_meshes", "meshes_as_hulls"]
