# SPDX-License-Identifier: Apache-2.0
"""Liquid surfaces: SPH particles drawn as water.

:func:`surface_mesh` turns one liquid's particles into a closed triangle mesh: every grid node sums a smooth kernel
(cubic spline, support 4 particle radii) over the particles near it, normalised so the inside of the liquid reads 1,
and Warp's marching cubes extracts the half-density surface, wound outward. :func:`liquid_meshes` does this for every
liquid of an ``alloy.Sim`` world and returns :class:`~alloy.camera.meshes.TexturedMesh` objects with a clear water
material (transmission 1, IOR 1.33), which :func:`~alloy.camera.meshes.model_scene` and
:func:`alloy.camera.render` draw with Cycles' glass closure.

Example::

    from alloy.camera.liquid import surface_mesh
    verts, tris = surface_mesh(points, radius=0.003)       # (V, 3) float32, (T, 3) int32

    imgs = alloy.camera.render(sim, quality="full")        # liquids of the sim drawn as water
"""
from __future__ import annotations

import logging
import math

import numpy as np
import warp as wp

from .errors import CameraError
from .meshes import TexturedMesh

log = logging.getLogger(__name__)

WATER = {"color": (1.0, 1.0, 1.0), "roughness": 0.0, "transmission": 1.0, "ior": 1.33, "absorption": None}
SUPPORT = 4.0   # kernel support, in particle radii (2 particle spacings)
CELL = 1.0      # grid spacing, in particle radii
MAX_NODES = 1 << 26


@wp.func
def _cubic(q: float) -> float:
    """Cubic spline kernel shape on q = r / h in [0, 1] (Monaghan; normalised by the caller)."""
    if q >= 1.0:
        return 0.0
    if q <= 0.5:
        return 6.0 * (q * q * q - q * q) + 1.0
    s = 1.0 - q
    return 2.0 * s * s * s


@wp.kernel(enable_backward=False)
def _density(grid: wp.uint64, pts: wp.array(dtype=wp.vec3), lo: wp.vec3, cell: float, h: float, scale: float,
             ny: int, nz: int, out: wp.array(dtype=float)):
    i = wp.tid()
    ix = i // (ny * nz)
    iy = (i // nz) % ny
    iz = i % nz
    x = lo + wp.vec3(float(ix), float(iy), float(iz)) * cell
    q = wp.hash_grid_query(grid, x, h)
    j = int(0)
    s = float(0.0)
    while wp.hash_grid_query_next(q, j):
        s += _cubic(wp.length(x - pts[j]) / h)
    out[i] = s * scale


def _grid_shape(pts: np.ndarray, radius: float) -> tuple[np.ndarray, float, tuple[int, int, int]]:
    """(lower corner, cell, (nx, ny, nz)) of a grid that covers the particles' kernels with a free border."""
    cell = CELL * radius
    pad = SUPPORT * radius + 2.0 * cell
    lo = pts.min(0) - pad
    n = np.ceil((pts.max(0) + pad - lo) / cell).astype(int) + 1
    if int(np.prod(n)) > MAX_NODES:
        raise CameraError(f"a liquid spans {n.tolist()} grid cells of {cell:.3g} m (more than {MAX_NODES})",
                          "draw fewer particles (drop the ones that left the scene) or pass a larger radius")
    return lo.astype(np.float32), float(cell), (int(n[0]), int(n[1]), int(n[2]))


def _outward(verts: np.ndarray, tris: np.ndarray, field: np.ndarray, lo: np.ndarray, cell: float) -> np.ndarray:
    """``tris`` wound so their normals point down the density (out of the liquid)."""
    if not len(tris):
        return tris
    p = verts[tris]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    c = (p.mean(1) - lo) / cell
    g = np.stack(np.gradient(field), -1)
    idx = np.clip(np.rint(c).astype(int), 0, np.array(field.shape) - 1)
    flip = np.einsum("ij,ij->i", n, g[idx[:, 0], idx[:, 1], idx[:, 2]]) > 0.0
    out = tris.copy()
    out[flip] = out[flip][:, [0, 2, 1]]
    return out


def surface_mesh(points: np.ndarray, radius: float, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """Closed surface of a liquid given as particles: ``points`` (N, 3) metres, ``radius`` the particle radius
    (half the particle spacing). Returns (verts (V, 3) float32, tris (T, 3) int32), triangles wound outward; empty
    arrays without particles.

    Example: ``verts, tris = surface_mesh(sim.state(world=0)["water.points"], radius=0.003)``."""
    pts = np.asarray(points, np.float32).reshape(-1, 3)
    if radius <= 0.0 or not math.isfinite(radius):
        raise CameraError(f"liquid particle radius {radius}", "pass the particle radius in metres (> 0)")
    if not len(pts):
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
    lo, cell, (nx, ny, nz) = _grid_shape(pts, radius)
    h = SUPPORT * radius
    # each particle stands for (2 r)^3; the 3D cubic spline integrates to pi h^3 / 10 (before 8 / (pi h^3))
    scale = (2.0 * radius) ** 3 * 8.0 / (math.pi * h ** 3)
    pw = wp.array(pts, dtype=wp.vec3, device=device)
    grid = wp.HashGrid(128, 128, 128, device=device)
    grid.build(pw, h)
    field = wp.zeros(nx * ny * nz, dtype=float, device=device)
    wp.launch(_density, dim=nx * ny * nz, inputs=[grid.id, pw, wp.vec3(*lo), cell, h, float(scale), ny, nz],
              outputs=[field], device=device)
    mc = wp.MarchingCubes(nx, ny, nz, device=device)
    mc.surface(field.reshape((nx, ny, nz)), 0.5)
    verts = lo + mc.verts.numpy().astype(np.float32) * cell
    tris = mc.indices.numpy().astype(np.int32).reshape(-1, 3)
    tris = _outward(verts, tris, field.numpy().reshape(nx, ny, nz), lo, cell)
    log.debug("liquid surface: %d particles -> %d triangles (grid %dx%dx%d)", len(pts), len(tris), nx, ny, nz)
    return verts.astype(np.float32), tris


def liquid_names(sim) -> list[str]:
    """Names of the SPH liquids of an ``alloy.Sim`` (``sim.routes()``)."""
    return [n for n, r in sim.routes().items() if "liquid" in str(r["route"]).lower()
            and n in sim.names().get("particles", [])]


def liquid_meshes(sim, world: int = 0, material: dict | None = None, device: str = "cpu") -> list[TexturedMesh]:
    """One water :class:`TexturedMesh` per liquid of ``sim`` in ``world`` (poured particles only). ``material``
    overrides fields of :data:`WATER` (e.g. ``{"absorption": (0.4, 0.06, 0.02)}``).

    Example: ``model_scene(sim.engine.model, liquid_meshes(sim), quality="full")``."""
    mat = {**WATER, **(material or {})}
    out = []
    for name in liquid_names(sim):
        st = sim.state([name], world=world)
        pts = np.asarray(st[f"{name}.points"])
        active = np.asarray(st.get(f"{name}.active", np.ones(len(pts), bool)), bool)
        radius = 0.5 * sim.particle_volume(name) ** (1.0 / 3.0)
        verts, tris = surface_mesh(pts[active], radius, device=device)
        if len(tris):
            out.append(TexturedMesh(name, verts, tris, **mat))
    return out
