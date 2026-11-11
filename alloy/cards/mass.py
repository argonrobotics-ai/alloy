# SPDX-License-Identifier: Apache-2.0
"""Mass of an object from its card: geometry × sourced density, computed by Newton's own inertia routine.

``newton.geometry.compute_inertia_shape`` (newton@ac7f7830:newton/_src/geometry/inertia.py:608) is what
``ModelBuilder.add_shape_*`` uses, so CARD-10 compares the mass the compiled Model will hold, not a re-derivation.
Meshes with ``part_materials`` are split by OBJ group; each group must be a closed sub-mesh.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np

from alloy.cards import geometry as geo
from alloy.cards import types as ct
from alloy.cards.paths import CardView
from alloy.core.errors import CardError

log = logging.getLogger("alloy.cards")

AssetResolver = Callable[[str], Path]

_GEO_NAMES = {"box": "BOX", "sphere": "SPHERE", "capsule": "CAPSULE", "cylinder": "CYLINDER", "cone": "CONE",
              "ellipsoid": "ELLIPSOID"}


def _newton():
    import warp as wp  # noqa: PLC0415 - heavy import, only when a mass is asked for

    from alloy._vendor import newton  # noqa: PLC0415
    from alloy._vendor.newton.geometry import compute_inertia_shape  # noqa: PLC0415

    return wp, newton, compute_inertia_shape


def primitive_mass(p: ct.Primitive, density: float) -> float:
    """Mass [kg] of a solid primitive at ``density`` via Newton (plane → 0)."""
    if p.kind == "plane":
        return 0.0
    wp, newton, inertia = _newton()
    s = tuple(p.size) + (0.0,) * (3 - len(p.size))
    scale = (s[0], s[0], s[0]) if p.kind == "sphere" else s[:3]
    kind = getattr(newton.GeoType, _GEO_NAMES[p.kind])
    mass, _, _ = inertia(int(kind), wp.vec3(*scale), None, float(density))
    return float(mass)


def mesh_mass(mesh: geo.MeshData, density: float) -> float:
    """Mass [kg] of a closed triangle mesh at ``density`` via Newton's mesh inertia."""
    if len(mesh.faces) == 0:
        return 0.0
    wp, newton, inertia = _newton()
    nm = newton.Mesh(mesh.vertices.astype(np.float32), mesh.faces.reshape(-1).astype(np.int32), compute_inertia=True)
    mass, _, _ = inertia(int(newton.GeoType.MESH), wp.vec3(1.0, 1.0, 1.0), nm, float(density))
    return float(mass)


def _mesh_object_mass(view: CardView, shape: ct.MeshSource, assets: AssetResolver) -> float:
    mesh = geo.read_mesh(assets(shape.path)).scaled(shape.scale)
    if not view.parts:
        return mesh_mass(mesh, view.material.density.value)
    total, used = 0.0, np.zeros(len(mesh.faces), dtype=bool)
    for name, idx in mesh.groups.items():
        mat = view.parts.get(name, view.material)
        total += mesh_mass(mesh.subset(name), mat.density.value)
        used[idx] = True
    rest = geo.MeshData(mesh.vertices, mesh.faces[~used])
    return total + mesh_mass(rest, view.material.density.value)


def card_mass(view: CardView, assets: AssetResolver) -> float | None:
    """Mass [kg] the card implies, or None where the card alone does not fix it (USD, MJCF, analytic SDF)."""
    shape, rho = view.card.shape, view.material.density.value
    if isinstance(shape, ct.Primitive):
        return primitive_mass(shape, rho)
    if isinstance(shape, ct.MeshSource):
        return _mesh_object_mass(view, shape, assets)
    if isinstance(shape, ct.CurveSource):
        return rho * math.pi * shape.radius**2 * geo.curve_length(shape)
    if isinstance(shape, ct.ParticleSource):
        lo, hi = np.asarray(shape.region[0]), np.asarray(shape.region[1])
        return rho * float(np.prod(np.abs(hi - lo)))
    if isinstance(shape, ct.TetSource) and shape.from_mesh:
        return mesh_mass(geo.read_mesh(assets(shape.from_mesh)), rho)
    if isinstance(shape, ct.SDFSource) and shape.path and Path(shape.path).suffix.lower() in (".obj", ".stl"):
        return mesh_mass(geo.read_mesh(assets(shape.path)), rho)
    log.info("card_mass: %s has a %s source; mass is not fixed by the card", view.card.name, type(shape).__name__)
    return None


def require_mass(view: CardView, assets: AssetResolver) -> float:
    m = card_mass(view, assets)
    if m is None:
        raise CardError(view.card.name, "shape", f"mass is not defined by a {type(view.card.shape).__name__} card")
    return m
