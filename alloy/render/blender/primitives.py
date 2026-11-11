# SPDX-License-Identifier: Apache-2.0
"""Tessellated primitive and grain meshes as numpy polygons (no bpy).

A mesh is (verts [V,3] f32, corners [L] i32, sizes [F] i32): polygon f uses corners[start_f : start_f + sizes[f]].
Shapes follow Newton: capsule/cylinder/cone are [r, half_height] along local Z (cone apex at +Z), the plane is
[width, length] full extents with 0 meaning infinite.
"""

from __future__ import annotations

import math

import numpy as np

import presets

MeshArrays = tuple[np.ndarray, np.ndarray, np.ndarray]
INFINITE_PLANE_M = 40.0


def revolve(profile: list[tuple[float, float]], segments: int) -> MeshArrays:
    """Surface of revolution about Z of a (radius, z) profile; entries with radius 0 become single pole vertices."""
    verts: list[np.ndarray] = []
    rings: list[list[int]] = []
    ang = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    for r, z in profile:
        base = sum(len(v) for v in verts)
        if r <= 0.0:
            verts.append(np.array([[0.0, 0.0, z]]))
            rings.append([base])
        else:
            verts.append(np.stack([r * np.cos(ang), r * np.sin(ang), np.full(segments, z)], axis=1))
            rings.append(list(range(base, base + segments)))
    corners: list[int] = []
    sizes: list[int] = []
    for a, b in zip(rings, rings[1:]):
        _bridge(a, b, segments, corners, sizes)
    return np.concatenate(verts).astype(np.float32), np.asarray(corners, np.int32), np.asarray(sizes, np.int32)


def _bridge(a: list[int], b: list[int], n: int, corners: list[int], sizes: list[int]) -> None:
    """Faces between two rings (or a pole and a ring), wound outward for a top-to-bottom profile."""
    for i in range(n):
        j = (i + 1) % n
        if len(a) == 1 and len(b) == 1:
            return
        if len(a) == 1:
            corners += [a[0], b[i], b[j]]
            sizes.append(3)
        elif len(b) == 1:
            corners += [a[i], b[0], a[j]]
            sizes.append(3)
        else:
            corners += [a[i], b[i], b[j], a[j]]
            sizes.append(4)


def box(hx: float, hy: float, hz: float) -> MeshArrays:
    v = np.array([[x, y, z] for z in (-hz, hz) for y in (-hy, hy) for x in (-hx, hx)], np.float32)
    quads = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]]
    return v, np.asarray(quads, np.int32).ravel(), np.full(6, 4, np.int32)


def sphere(rx: float, ry: float, rz: float, segments: int = 64, rings: int = 32) -> MeshArrays:
    th = np.linspace(0.0, math.pi, rings + 1)
    v, c, s = revolve([(math.sin(t), math.cos(t)) for t in th], segments)
    return v * np.array([rx, ry, rz], np.float32), c, s


def capsule(r: float, hh: float, segments: int = 64, rings: int = 16) -> MeshArrays:
    th = np.linspace(0.0, math.pi / 2, rings + 1)
    top = [(r * math.sin(t), hh + r * math.cos(t)) for t in th]
    bottom = [(r * math.cos(t), -hh - r * math.sin(t)) for t in th]
    return revolve(top + bottom, segments)


def cylinder(r: float, hh: float, segments: int = 64) -> MeshArrays:
    return revolve([(0.0, hh), (r, hh), (r, -hh), (0.0, -hh)], segments)


def cone(r: float, hh: float, segments: int = 64) -> MeshArrays:
    steps = np.linspace(0.0, 1.0, 9)[1:]
    side = [(r * s, hh - 2.0 * hh * s) for s in steps]
    return revolve([(0.0, hh)] + side + [(0.0, -hh)], segments)


def plane(width: float, length: float) -> MeshArrays:
    hx = 0.5 * (width if width > 0 else INFINITE_PLANE_M)
    hy = 0.5 * (length if length > 0 else INFINITE_PLANE_M)
    v = np.array([[-hx, -hy, 0], [hx, -hy, 0], [hx, hy, 0], [-hx, hy, 0]], np.float32)
    return v, np.arange(4, dtype=np.int32), np.array([4], np.int32)


def shape_mesh(typ: str, size: list[float]) -> MeshArrays:
    """Mesh of a primitive shape type with Newton's size convention."""
    s = list(size) + [0.0, 0.0, 0.0]
    if typ == "box":
        return box(s[0], s[1], s[2])
    if typ == "sphere":
        return sphere(s[0], s[0], s[0])
    if typ == "ellipsoid":
        return sphere(s[0], s[1], s[2])
    if typ == "capsule":
        return capsule(s[0], s[1])
    if typ == "cylinder":
        return cylinder(s[0], s[1])
    if typ == "cone":
        return cone(s[0], s[1])
    if typ == "plane":
        return plane(s[0], s[1])
    raise ValueError(f"no primitive for shape type {typ!r}")


def grain_mesh(name: str, radius: float = 1.0, scale: float = 1.0) -> MeshArrays:
    """Preset grain (presets.GRAINS) with its long axis on X; `sphere` is scaled by the point radius."""
    g = presets.GRAINS[name]
    th = np.linspace(0.0, math.pi, g["rings"] + 1)
    taper = g.get("taper", 0.0)
    prof = [(math.sin(t) * (1.0 - taper * math.cos(t) ** 2), math.cos(t)) for t in th]
    v, c, s = revolve(prof, g["segments"])
    ax = np.asarray(g["axes"], np.float32)
    v = v[:, [2, 0, 1]] * ax  # revolve axis Z -> long axis X
    if g.get("lumpy"):
        rng = np.random.default_rng(7)
        v *= (1.0 + g["lumpy"] * (rng.random((len(v), 1)) - 0.5)).astype(np.float32)
    k = (radius if g.get("unit") else 1.0) * scale
    return (v * np.float32(k)).astype(np.float32), c, s


def tris(faces: np.ndarray) -> MeshArrays:
    """Triangle faces [F,3] as polygon arrays (vertices filled in by the caller)."""
    faces = np.asarray(faces, np.int32)
    return np.zeros((0, 3), np.float32), faces.ravel(), np.full(len(faces), 3, np.int32)


def weld_exact(verts: np.ndarray, faces: np.ndarray, rel_tol: float = 1e-7) -> tuple[np.ndarray, np.ndarray]:
    """Merge coincident vertices (STL triangle soups) so smooth shading has shared normals."""
    if len(verts) == 0:
        return verts, faces
    span = float(np.ptp(verts, axis=0).max()) or 1.0
    key = np.round(verts / (span * rel_tol)).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return verts[first], inverse.reshape(-1)[faces]


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals of a triangle mesh."""
    v = np.asarray(verts, np.float64)
    fn = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    n = np.linalg.norm(vn, axis=1, keepdims=True)
    return (vn / np.where(n > 0, n, 1.0)).astype(np.float32)
