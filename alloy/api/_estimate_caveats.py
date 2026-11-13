# SPDX-License-Identifier: Apache-2.0
"""What ``alloy.estimate`` knows it leaves out for a scene (``caveats``), found in its one-world template.

Mesh-on-mesh contact: the step terms see shapes, not the contacts between them, and meshes resting on
meshes step slower than estimated (bench/results/estimate_calib/wp7_s1_rows.json), so the estimate says so.
"""

from __future__ import annotations

from typing import Any

import numpy as np

TOUCHING_MESH = ("{n} pair(s) of mesh / convex-hull / cylinder shapes on different bodies touch at the start: the step "
                 "time does not count mesh-on-mesh contact. On one L4 (Alloy main e65d95254) 16 bowls stacked 4 high "
                 "stepped 1.9-2.5x the estimate (decomposed) and 1.6-2.1x (one hull each) at 1-256 worlds, 2 towers of "
                 "8 decomposed bowls 2.9x (1 world) and 28x (64 worlds); spaced apart the same bowls stepped 0.5-1.0x "
                 "it. measure=True times a real run")
CCD_GEO = ("MESH", "CONVEX_MESH", "CYLINDER", "CONE", "ELLIPSOID")      # MuJoCo's convex (CCD) path


def caveats(load: dict[str, Any]) -> list[str]:
    """What the estimate knows it does not model for this scene (``TOUCHING_MESH``), as sentences."""
    n = int(load.get("touching_mesh_pairs", 0))
    return [TOUCHING_MESH.format(n=n)] if n else []


def touching_mesh_pairs(model: Any) -> int:
    """Pairs of colliding CCD-path shapes (``CCD_GEO``) on different bodies, one of them moving, whose bounding
    spheres (``shape_collision_radius``, plus both margins) overlap at the start pose: resting mesh-on-mesh contact.
    One shape against all later ones in numpy. Example: ``touching_mesh_pairs(built.model)``."""
    from alloy._vendor import newton  # noqa: PLC0415

    if not model.shape_count:
        return 0
    kinds, flags = model.shape_type.numpy(), model.shape_flags.numpy()
    ccd = np.isin(kinds, [int(getattr(newton.GeoType, g)) for g in CCD_GEO])
    ccd &= (flags & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0
    idx = np.flatnonzero(ccd)
    if idx.size < 2:
        return 0
    body = model.shape_body.numpy()[idx]
    xf = model.shape_transform.numpy()[idx]
    pos = xf[:, :3].astype(np.float64)
    if model.body_count:
        bq = model.body_q.numpy().astype(np.float64)
        moving = body >= 0
        pos[moving] = bq[body[moving], :3] + _rotate(bq[body[moving], 3:7], pos[moving])
    mass = np.where(body >= 0, model.body_mass.numpy()[np.maximum(body, 0)] if model.body_count else 0.0, 0.0)
    reach = model.shape_collision_radius.numpy()[idx].astype(np.float64) + model.shape_margin.numpy()[idx]
    n = 0
    for i in range(idx.size - 1):
        j = np.arange(i + 1, idx.size)
        near = np.linalg.norm(pos[j] - pos[i], axis=1) <= reach[j] + reach[i]
        n += int(np.count_nonzero(near & (body[j] != body[i]) & ((mass[j] > 0) | (mass[i] > 0))))
    return n


def _rotate(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rows of ``v`` rotated by the rows of the unit quaternions ``q_xyzw``."""
    u, w = q_xyzw[:, :3], q_xyzw[:, 3:4]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)
