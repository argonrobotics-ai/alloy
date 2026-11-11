# SPDX-License-Identifier: Apache-2.0
"""Per-body facts the rigid-step caps read (SCR-WP0-RIGIDDT, ``alloy.scene.rigid_step``).

Sizes come from the emitted template's collision shapes: each shape's box in its parent frame and its thinnest member
(a primitive's smallest width; a mesh's smallest width over its face normals and, for a closed mesh, its thinnest
wall: ``alloy.cards.geometry.wall_thickness``, the card validator's probe). Start drops and column loads come from
every world's start positions: the API's per-world rows where given, else the template pose, with a randomized pose
axis taken over its whole [lo, hi] range (conservatively: highest for a body's own drop, never counted as a support
it might miss, always counted as a load it might carry).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from alloy._vendor import newton
from alloy.cards import geometry as geo
from alloy.contact.inertia_floor import SLENDER
from alloy.core.defaults import GRAVITY
from alloy.scene.randomize import POSE_AXES

GEO = newton.GeoType
COLLIDE = int(newton.ShapeFlags.COLLIDE_SHAPES)
CELLS_PER_PASS = 4_000_000      # worlds x bodies x bodies booleans per numpy pass (start drops and loads)
RESTING_TOL = 1.0e-3            # m: a body this far into what is under it still rests on it (start contact)


@dataclass(frozen=True)
class BodyFacts:
    """One dynamic free body, as the caps see it."""

    name: str
    body: int               # template body id
    mass: float             # kg
    thin: float             # m: thinnest collision member
    lever: float            # m: the body's middle half extent, the spin lever of cap 3 (a rod's radius, a plate's
    #                         in-plane size: spin about a thin axis needs contacts around it)
    r_gyr: float            # m: smallest principal radius of gyration
    slender: bool           # largest / smallest principal inertia >= SLENDER (the inertia floor's gate)
    contact_speed: float    # m/s: the card's loads.contact_speed
    drop: float = 0.0       # m: largest start drop onto what is under it, over worlds
    load: float = 1.0       # column load factor: 1 + (mass resting on it) / its mass, largest over worlds

    @property
    def speed(self) -> float:
        """The speed the caps assume [m/s]: the card's contact speed or the start drop's impact, whichever is larger."""
        return max(self.contact_speed, math.sqrt(2.0 * GRAVITY * self.drop))


@dataclass(frozen=True)
class ShapeBox:
    """A collision shape's axis-aligned box in its parent frame (a body, or the world for a static shape)."""

    centre: np.ndarray      # (3,)
    half: np.ndarray        # (3,)
    thin: float             # m: thinnest member (inf for a plane)
    mesh: geo.MeshData | None = None    # a non-convex mesh, for the wall probe


def _rotated(xform: Any, centre: np.ndarray, half: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centre and half extents of a box (``centre``, ``half`` in the shape frame) in the parent frame."""
    p = np.asarray(xform[:3], np.float64)
    qx, qy, qz, qw = (float(v) for v in xform[3:7])
    R = np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                  [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                  [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
    return p + R @ centre, np.abs(R) @ half


def _mesh_box(src: Any, scale: np.ndarray, convex: bool) -> tuple[np.ndarray, np.ndarray, float, geo.MeshData | None]:
    v = np.asarray(src.vertices, np.float64) * scale
    f = np.asarray(src.indices, np.int64).reshape(-1, 3)
    lo, hi = v.min(axis=0), v.max(axis=0)
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = n[np.linalg.norm(n, axis=1) > 1e-12]
    dirs = np.vstack([np.eye(3), n / np.linalg.norm(n, axis=1, keepdims=True)]) if len(n) else np.eye(3)
    proj = v @ dirs.T
    width = float((proj.max(axis=0) - proj.min(axis=0)).min())
    mesh = None if convex else geo.MeshData(v, f)
    return 0.5 * (lo + hi), 0.5 * (hi - lo), width, mesh


def shape_box(b: newton.ModelBuilder, s: int) -> ShapeBox | None:
    """``ShapeBox`` of collision shape ``s``; None for shapes that do not collide, planes and unknown kinds."""
    if not int(b.shape_flags[s]) & COLLIDE:
        return None
    kind, sc = int(b.shape_type[s]), np.asarray(b.shape_scale[s], np.float64)
    centre, mesh = np.zeros(3), None
    if kind in (GEO.BOX, GEO.ELLIPSOID):
        half, thin = sc.copy(), 2.0 * float(sc.min())
    elif kind == GEO.SPHERE:
        half, thin = np.full(3, sc[0]), 2.0 * float(sc[0])
    elif kind == GEO.CAPSULE:
        half, thin = np.array([sc[0], sc[0], sc[1] + sc[0]]), 2.0 * float(sc[0])
    elif kind in (GEO.CYLINDER, GEO.CONE):
        half, thin = np.array([sc[0], sc[0], sc[1]]), 2.0 * float(min(sc[0], sc[1]))
    elif kind in (GEO.MESH, GEO.CONVEX_MESH) and b.shape_source[s] is not None:
        centre, half, thin, mesh = _mesh_box(b.shape_source[s], sc, kind == GEO.CONVEX_MESH)
    else:
        return None
    c, h = _rotated(b.shape_transform[s], centre, half)
    return ShapeBox(c, h, thin, mesh)


def body_boxes(b: newton.ModelBuilder, body: int) -> list[ShapeBox]:
    return [x for s in range(len(b.shape_body)) if int(b.shape_body[s]) == body for x in [shape_box(b, s)] if x]


def union(boxes: Sequence[ShapeBox]) -> tuple[np.ndarray, np.ndarray]:
    """Centre and half extents of the union of ``boxes`` (in their shared frame)."""
    lo = np.min([x.centre - x.half for x in boxes], axis=0)
    hi = np.max([x.centre + x.half for x in boxes], axis=0)
    return 0.5 * (lo + hi), 0.5 * (hi - lo)


def inertia_facts(b: newton.ModelBuilder, body: int, mass: float) -> tuple[float, bool]:
    """(smallest principal radius of gyration [m], slender) from the template inertia, as the inertia floor reads it."""
    principal = np.linalg.eigvalsh(np.asarray(b.body_inertia[body], np.float64).reshape(3, 3))
    smallest = max(float(principal[0]), 0.0)
    r_gyr = math.sqrt(smallest / mass) if mass > 0.0 else 0.0
    return r_gyr, bool(float(principal[-1]) >= SLENDER * smallest)


def mesh_thin(boxes: Sequence[ShapeBox], within: float) -> float:
    """Thinnest wall of the body's closed non-convex meshes, measured below ``within`` (inf above it or none)."""
    walls = [geo.wall_thickness(x.mesh, within=within) for x in boxes if x.mesh is not None and geo.is_closed(x.mesh)]
    return min(walls, default=math.inf)


def statics(b: newton.ModelBuilder) -> tuple[np.ndarray, np.ndarray, float]:
    """World boxes of the static collision shapes ((k, 3) centres, (k, 3) half extents) and the highest horizontal
    plane's height (-inf without one)."""
    cs, hs, plane = [], [], -math.inf
    for s in range(len(b.shape_body)):
        if int(b.shape_body[s]) >= 0 or not int(b.shape_flags[s]) & COLLIDE:
            continue
        if int(b.shape_type[s]) == GEO.PLANE:
            x = b.shape_transform[s]
            _, normal = _rotated(x, np.zeros(3), np.array([0.0, 0.0, 1.0]))
            plane = max(plane, float(x[2])) if normal[2] > 0.999 else plane
            continue
        box = shape_box(b, s)
        if box is not None:
            cs.append(box.centre)
            hs.append(box.half)
    return np.reshape(cs, (-1, 3)), np.reshape(hs, (-1, 3)), plane


def start_positions(names: Sequence[str], template: np.ndarray, rows: Mapping[str, Any] | None,
                    placements: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Per-world body positions (W, n, 3) and per-body half ranges (n, 3) of randomized pose axes (centred on their
    range). ``template``: (n, 3) template positions; ``rows``: {name: (W, 3)} per-world positions."""
    given = {k: np.asarray(v, np.float64) for k, v in (rows or {}).items() if k in names}
    W = max((len(v) for v in given.values()), default=1)
    P = np.broadcast_to(template, (W, *template.shape)).copy()
    U = np.zeros_like(template)
    for i, name in enumerate(names):
        if name in given:
            P[:, i] = given[name] if len(given[name]) == W else given[name][:1]
        for r in getattr(placements.get(name), "randomize", ()):
            axis = POSE_AXES.get(r.field)
            if axis is not None:
                P[:, i, axis] = 0.5 * (r.lo + r.hi)
                U[i, axis] = 0.5 * (r.hi - r.lo)
    return P, U


def drops_and_loads(P: np.ndarray, U: np.ndarray, half: np.ndarray, mass: np.ndarray,
                    fixed: tuple[np.ndarray, np.ndarray, float]) -> tuple[np.ndarray, np.ndarray]:
    """Largest start drop [m] and column load factor of each body over worlds (module docstring)."""
    W, n = P.shape[:2]
    drop, load = np.zeros(n), np.ones(n)
    step = max(1, CELLS_PER_PASS // max(n * n, 1))
    for w0 in range(0, W, step):
        d, ld = _drops_loads_chunk(P[w0:w0 + step], U, half, mass, fixed)
        drop, load = np.maximum(drop, d), np.maximum(load, ld)
    return drop, load


def _drops_loads_chunk(P: np.ndarray, U: np.ndarray, half: np.ndarray, mass: np.ndarray,
                       fixed: tuple[np.ndarray, np.ndarray, float]) -> tuple[np.ndarray, np.ndarray]:
    sc, sh, plane = fixed
    gap = np.abs(P[:, :, None, :2] - P[:, None, :, :2])                      # (w, i, j, 2)
    reach = half[:, None, :2] + half[None, :, :2]
    may = np.all(gap < reach + U[:, None, :2] + U[None, :, :2], axis=-1)       # j's footprint can meet i's
    must = np.all(gap < reach - U[:, None, :2] - U[None, :, :2], axis=-1)      # ... and always does
    below = P[:, None, :, 2] < P[:, :, None, 2]                               # j's centre under i's
    bottom = P[:, :, 2] + U[:, 2] - half[:, 2]                                # i's highest start bottom
    tops = np.where(must & below, (P[:, :, 2] - U[:, 2] + half[:, 2])[:, None, :], -np.inf)
    support = np.maximum(tops.max(axis=2), plane)
    if len(sc):
        sgap = np.abs(P[:, :, None, :2] - sc[None, None, :, :2])
        under = np.all(sgap < half[:, None, :2] + sh[None, :, :2] - U[:, None, :2], axis=-1)
        stop = sc[:, 2] + sh[:, 2]
        under &= stop[None, None, :] <= bottom[:, :, None] + RESTING_TOL
        support = np.maximum(support, np.where(under, stop[None, None, :], -np.inf).max(axis=2))
    drop = np.where(np.isfinite(support), np.maximum(bottom - support, 0.0), 0.0).max(axis=0)
    above = may & (P[:, None, :, 2] > P[:, :, None, 2])                       # j can rest on i's column
    load = 1.0 + (above * mass[None, None, :]).sum(axis=2) / np.maximum(mass, 1e-12)[None, :]
    return drop, load.max(axis=0)


def with_starts(facts: Sequence[BodyFacts], drop: np.ndarray, load: np.ndarray) -> list[BodyFacts]:
    return [replace(f, drop=float(drop[i]), load=float(load[i])) for i, f in enumerate(facts)]


__all__ = ["BodyFacts", "ShapeBox", "body_boxes", "drops_and_loads", "inertia_facts", "mesh_thin", "shape_box",
           "start_positions", "statics", "union", "with_starts"]
