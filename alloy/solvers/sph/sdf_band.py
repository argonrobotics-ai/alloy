# SPDX-License-Identifier: Apache-2.0
"""Which mesh walls SolverDFSPH reads through the mesh BVH instead of their texture SDF (SCR-WP9-9, SCR-JOB2-2).

A mesh shape without a volume SDF (``sdf_index`` -1: every mesh on CPU, and on CUDA every mesh nobody baked) has no
texture to read. A mesh whose texture SDF was baked for something else, e.g. a precision pair's +-0.5 mm band, has one,
but past its narrow band the texture holds only the coarse far field, while the liquid needs exact distances out to its
reach (``ShapeArrays.reach``: 1.5 spacings, or h with wall viscosity). Both read the BVH (``boundary.shape_sdf``).

Band test: the texture's narrow band is the set of coarse cells that own a subgrid (Newton's ``_locate_cell``:
``start_slot != SLOT_EMPTY``; demoted ``SLOT_LINEAR`` cells are exact under coarse interpolation). A mesh's SDF covers
the reach when every probe point at +reach outside its faces (each face's centroid and corners, moved along the face
normal) lies in such a cell inside the SDF box. The texture itself stays as baked (WP3's pair SDF is untouched).
"""

from __future__ import annotations

import numpy as np
import warp as wp
from alloy._vendor.newton import GeoType, Model
from alloy._vendor.newton._src.geometry.sdf_texture import SLOT_EMPTY, TextureSDFData, _locate_cell

MESH_GEOS = (int(GeoType.MESH), int(GeoType.CONVEX_MESH))


@wp.kernel(enable_backward=False)
def _band_miss_kernel(tex: wp.array[TextureSDFData], k: int, pts: wp.array[wp.vec3], scale: wp.vec3,
                      miss: wp.array[wp.int32]):
    """miss[0] += 1 for every probe point (shape-local, scaled) outside the texture's narrow band or SDF box."""
    i = wp.tid()
    t = tex[k]
    q = pts[i]
    if not t.scale_baked:
        q = wp.cw_div(q, scale)
    lo = t.sdf_box_lower
    hi = t.sdf_box_upper
    if q[0] < lo[0] or q[1] < lo[1] or q[2] < lo[2] or q[0] > hi[0] or q[1] > hi[1] or q[2] > hi[2]:
        wp.atomic_add(miss, 0, 1)
        return
    loc = _locate_cell(t, wp.cw_mul(q - lo, t.inv_sdf_dx))
    if loc.start_slot == SLOT_EMPTY:
        wp.atomic_add(miss, 0, 1)


def probe_points(mesh, scale: np.ndarray, reach: float) -> np.ndarray:
    """Shape-local (scaled) points at +reach outside each face: centroid and corners along the face normal."""
    v = np.asarray(mesh.vertices, np.float64) * scale
    f = np.asarray(mesh.indices, np.int64).reshape(-1, 3)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n, axis=1)
    ok = ln > 0.0
    n = n[ok] / ln[ok, None]
    pts = np.concatenate([(a[ok] + b[ok] + c[ok]) / 3.0, a[ok], b[ok], c[ok]]) + reach * np.tile(n, (4, 1))
    return pts.astype(np.float32)


def band_covers(model: Model, s: int, reach: float, device: wp.context.Device) -> bool:
    """Whether shape s's texture SDF holds its narrow band out to ``reach`` (module doc)."""
    mesh = model.shape_source[s]
    pts = probe_points(mesh, np.asarray(model.shape_scale.numpy()[s], np.float64), reach)
    if len(pts) == 0:
        return False
    miss = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(_band_miss_kernel, dim=len(pts), inputs=[model._texture_sdf_data, int(model._shape_sdf_index.numpy()[s]),
              wp.array(pts, dtype=wp.vec3, device=device), wp.vec3(*model.shape_scale.numpy()[s]), miss],
              device=device)
    return int(miss.numpy()[0]) == 0


def bvh_flags(model: Model, ids: np.ndarray, reach: float, device: wp.context.Device) -> tuple[np.ndarray, dict]:
    """Per model shape 1 = read through the mesh BVH: a table mesh without an SDF, or whose SDF band misses the
    reach (one band test per distinct texture and scale). Returns the flags and the shape ids by reason."""
    flags = np.zeros(int(model.shape_count), np.int32)
    geo, sdf = model.shape_type.numpy(), model._shape_sdf_index.numpy()
    mesh = np.unique(ids[np.isin(geo[ids], MESH_GEOS)])
    no_sdf = mesh[sdf[mesh] < 0]
    narrow, tested = [], {}
    for s in mesh[sdf[mesh] >= 0]:
        key = (int(sdf[s]), tuple(np.round(model.shape_scale.numpy()[s], 9)))
        if key not in tested:
            tested[key] = band_covers(model, int(s), reach, device)
        if not tested[key]:
            narrow.append(int(s))
    flags[no_sdf] = 1
    flags[narrow] = 1
    return flags, {"no_sdf": no_sdf.tolist(), "narrow_band": narrow}
