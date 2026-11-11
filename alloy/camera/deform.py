# SPDX-License-Identifier: Apache-2.0
"""Deformables in the camera: cloth, soft-body surfaces and MPM grains of an ``alloy.Sim``, redrawn from the device
state at every render (no host copy; interface: SCR-OSS4-DEFORM).

At bind time :func:`deform_objects` lists what to draw from the sim's names and routes: a cloth or soft body
(routes ``cloth_vbd``, ``cloth_style3d``, ``cloth_film``, ``soft_vbd``) is drawn as its triangles (cloth sheets, soft
bodies' boundary faces), MPM grains, paste and MPM liquid (``grain_mpm``, ``paste_mpm``, ``liquid_mpm``) as one small
sphere (an icosahedron with smooth normals) per particle of its radius. Each world gets its own copy of each mesh;
:class:`alloy.camera.deform_gpu.DeformSet` refits them from ``state.particle_q`` before every render. SPH liquids get
a smoothed surface per world rebuilt every render (``alloy.camera.liquid_scene``).

Example::

    from alloy.camera.deform import deform_objects
    objs = deform_objects(sim)            # [DeformObject("towel", "surface", ...), ...]
    imgs = alloy.Camera("c", width=128, height=128).render(sim)    # the towel follows the sim
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .errors import CameraError

log = logging.getLogger(__name__)

SURFACE_ROUTES = ("cloth_vbd", "cloth_style3d", "cloth_film", "soft_vbd")
GRAIN_ROUTES = ("grain_mpm", "paste_mpm", "liquid_mpm")
MAX_TRIS = 1 << 27  # drawn triangles over all deformables and worlds of one camera scene
COLORS = {"surface": (0.62, 0.66, 0.72), "grains": (0.55, 0.42, 0.28)}  # without a look: light cloth, brown soil
CULL_M = 50.0  # grains farther than this from the scene's rest bounds are not drawn (MPM never retires strays)

_T = (1.0 + 5.0 ** 0.5) / 2.0
_ICO_V = np.array([[-1, _T, 0], [1, _T, 0], [-1, -_T, 0], [1, -_T, 0], [0, -1, _T], [0, 1, _T], [0, -1, -_T],
                   [0, 1, -_T], [_T, 0, -1], [_T, 0, 1], [-_T, 0, -1], [-_T, 0, 1]], np.float64)
ICO_V = (_ICO_V / np.linalg.norm(_ICO_V, axis=1, keepdims=True)).astype(np.float32)  # 12 unit vertices
ICO_F = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2],
                  [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5],
                  [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]], np.int32)  # 20 faces, counter-clockwise outside


@dataclass(frozen=True)
class DeformObject:
    """One deformable to draw: ``start`` is its first particle in world 0, ``count`` its particles per world;
    ``tris`` index ``rest`` (the drawn vertices in world 0's build pose: the particles of a surface, 12 per grain)."""

    name: str
    kind: str  # "surface" | "grains"
    start: int
    count: int
    tris: np.ndarray  # (m, 3) int32
    rest: np.ndarray  # (V, 3) float32
    color: tuple = (0.6, 0.6, 0.6)
    uv: np.ndarray | None = None  # (V, 2) glTF UVs of a scanned soft body (scan_soft), fixed at the build pose
    image: np.ndarray | None = None  # its uint8 (H, W, 3) sRGB photo texture

    @property
    def num_verts(self) -> int:
        return len(self.rest)


@dataclass(frozen=True)
class DeformLayout:
    """Where the particles live: object o of world w is ``state.particle_q[o.start + w * stride :][: o.count]``."""

    worlds: int
    stride: int
    objects: tuple


def grain_mesh(centers: np.ndarray, radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(verts (12 n, 3), tris (20 n, 3)): an icosahedron of ``radius[i]`` around each of the n ``centers``."""
    c = np.asarray(centers, np.float32).reshape(-1, 3)
    r = np.broadcast_to(np.asarray(radius, np.float32), (len(c),))
    verts = (c[:, None, :] + r[:, None, None] * ICO_V[None]).reshape(-1, 3)
    tris = (ICO_F[None] + 12 * np.arange(len(c), dtype=np.int32)[:, None, None]).reshape(-1, 3)
    return verts.astype(np.float32), tris.astype(np.int32)


def _routes(sim) -> dict[str, str]:
    try:
        return {n: str(r["route"]) for n, r in sim.routes().items()}
    except (AttributeError, KeyError, TypeError):
        return {}


def _surface(name: str, r: range, model, tri_range: range) -> DeformObject | None:
    tris = model.tri_indices.numpy()[tri_range.start:tri_range.stop].astype(np.int64) - r.start
    if not len(tris):
        return None
    if tris.min() < 0 or tris.max() >= len(r):
        raise CameraError(f"{name!r}: its triangles index particles outside its own range",
                          "the camera needs each object's triangles over its own particles (SCR-OSS4-DEFORM)")
    rest = model.particle_q.numpy()[r.start:r.stop].astype(np.float32)
    return DeformObject(name, "surface", r.start, len(r), tris.astype(np.int32), rest, COLORS["surface"])


def _grains(name: str, r: range, model) -> DeformObject:
    rest_c = model.particle_q.numpy()[r.start:r.stop]
    rad = model.particle_radius.numpy()[r.start:r.stop]
    verts, tris = grain_mesh(rest_c, rad)
    return DeformObject(name, "grains", r.start, len(r), tris, verts, COLORS["grains"])


def has_deformables(sim) -> bool:
    """True for an ``alloy.Sim`` with a cloth, soft body or MPM material (routes only; cheap)."""
    if getattr(sim, "index", None) is None or not getattr(sim.index, "particles", None):
        return False
    return any(r in SURFACE_ROUTES + GRAIN_ROUTES for r in _routes(sim).values())


def deform_objects(sim) -> DeformLayout | None:
    """The deformables of an ``alloy.Sim`` to draw (None for other sources or a sim without any).
    Raises CameraError when the particle layout is not one block per world or the triangles would not fit."""
    index, engine = getattr(sim, "index", None), getattr(sim, "engine", None)
    if index is None or engine is None or not getattr(index, "particles", None):
        return None
    routes, model = _routes(sim), engine.model
    objs = []
    for name, r in index.particles.items():
        route = routes.get(name, "")
        if route in SURFACE_ROUTES and name in index.tris:
            o = _surface(name, r, model, index.tris[name])
            if o is not None:
                objs.append(o)
        elif route in GRAIN_ROUTES:
            objs.append(_grains(name, r, model))
    if not objs:
        return None
    from .scan_soft import soft_scan_uvs
    objs = soft_scan_uvs(sim, objs)  # scanned soft bodies: their photo texture
    worlds, stride = int(index.worlds), int(index.stride["particle"])
    if int(model.particle_q.shape[0]) != worlds * stride:
        raise CameraError(f"{model.particle_q.shape[0]} particles are not {worlds} worlds of {stride}",
                          "the camera draws deformables only when every world holds its own particles")
    tris = worlds * sum(len(o.tris) for o in objs)
    if tris > MAX_TRIS:
        raise CameraError(f"the deformables need {tris:,} triangles over {worlds} worlds (limit {MAX_TRIS:,})",
                          "render fewer worlds at once, or use fewer / larger MPM particles")
    log.info("camera: %d deformables (%s), %d drawn triangles per world", len(objs),
             ", ".join(f"{o.name}: {o.kind}" for o in objs), tris // worlds)
    return DeformLayout(worlds, stride, tuple(objs))


def rest_bounds(layout: DeformLayout) -> tuple[np.ndarray, np.ndarray]:
    """(lo, hi) of every deformable's rest vertices, padded by :data:`CULL_M` (grains outside are not drawn)."""
    pts = np.concatenate([o.rest for o in layout.objects])
    return (pts.min(0) - CULL_M).astype(np.float32), (pts.max(0) + CULL_M).astype(np.float32)
