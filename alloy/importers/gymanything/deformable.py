# SPDX-License-Identifier: Apache-2.0
"""Cloth sheets and cables: their own shapes (a thin box or sheet mesh, a curve) and densities from areal/line mass.

- Cloth: a flat rectangular sheet becomes the library's cloth form (box primitive, half-thickness z, as
  ``napkin_cotton_aniso``); any other sheet keeps its mesh. Density = mass / (area x thickness), else the cited
  bulk density. ``features.thickness`` and ``min_wall`` carry the cited sheet thickness.
- Cable: a ``curve`` shape (centre-line points + radius); density = mass / (pi r^2 L); a cited EI goes to
  ``features.bend_stiffness``.

Example::

    >>> rec = convert_deformable(load_spec("specs/wiping/microfibre_cloth.json"), Path("out"))   # doctest: +SKIP
    >>> rec.collision                                                                             # doctest: +SKIP
    'cloth-box'
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from alloy.cards import geometry as geo
from alloy.cards import types as ct
from alloy.cards.loader import dump_card
from alloy.core.quantity import Q
from alloy.importers.gymanything import convert as cv
from alloy.importers.gymanything import meshes
from alloy.importers.gymanything.provenance import note_for, q_from
from alloy.importers.gymanything.spec import GarSpec

FLAT_TOL = 0.25            # sheet: out-of-plane extent <= this x the cited thickness + 1 mm counts as flat
RECT_TOL = 0.02            # sheet area within 2 % of its bounding rectangle -> a rectangle
DEFAULT_POISSON = 0.3      # woven/film sheets when their sim gives no Poisson ratio (source default)


def _density(spec: GarSpec, volume: float, what: str, rec: cv.ImportRecord) -> Q:
    if spec.mass is not None:
        rec.mass_basis = spec.mass.basis
        note = note_for(spec.mass, spec.sim, f"effective: {spec.mass.value:g} kg over {what} = {volume:.6g} m^3")
        unc = (spec.mass.unc[0] / volume, spec.mass.unc[1] / volume) if spec.mass.unc else None
        return Q(spec.mass.value / volume, "fit", unc, note)
    if spec.material.density is not None:
        rec.mass_basis = f"material density ({spec.material.density.basis})"
        return q_from(spec.material.density, spec.sim, f"{spec.material.name} density")
    rec.defaults.append("density")
    return Q(cv.DEFAULT_DENSITY, "default")


def _material(spec: GarSpec, volume: float, what: str, rec: cv.ImportRecord) -> ct.MaterialCard:
    rho = _density(spec, volume, what, rec)
    if not cv.DENSITY_RANGE[0] <= rho.value <= cv.DENSITY_RANGE[1]:
        raise cv.ConvertError(f"{spec.card_name}: density {rho.value:.4g} kg/m^3 outside {cv.DENSITY_RANGE} "
                              f"({what} = {volume:.4g} m^3): check the cited thickness/radius and the mass")
    return ct.MaterialCard(name=spec.card_name, density=rho, surface=cv._surface(spec, rec), elastic=cv._elastic(spec))


def _sheet_area(mesh: geo.MeshData) -> float:
    v = mesh.vertices[mesh.faces]
    return float(0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1).sum())


def _xy_hull_area(mesh: geo.MeshData) -> float:
    from scipy.spatial import ConvexHull, QhullError  # noqa: PLC0415

    try:
        return float(ConvexHull(mesh.vertices[:, :2]).volume)        # 2-D hull "volume" is its area
    except (QhullError, ValueError):
        return 0.0


def top_layer(mesh: geo.MeshData) -> geo.MeshData:
    """The upward faces of a closed thin slab (normal z > 0.5), re-indexed: the single sheet Alloy's cloth emitter
    expects (a closed slab would become two cloth layers)."""
    v = mesh.vertices[mesh.faces]
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    up = n[:, 2] > 0.5 * np.linalg.norm(n, axis=1)
    used, remap = np.unique(mesh.faces[up], return_inverse=True)
    return geo.MeshData(mesh.vertices[used], remap.reshape(-1, 3).astype(np.int64))


def _cloth_shape(spec: GarSpec, mesh: geo.MeshData, t: float, out: Path) -> tuple[ct.ShapeSource, float, str]:
    """(shape, sheet area, collision label). A flat rectangle (sheet or thin slab) -> box primitive; a closed thin
    slab of another outline -> its top layer as a sheet mesh; any other surface (a bag) -> the surface itself."""
    lo, hi = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
    span = hi - lo
    slab = span[2] <= 4.0 * t + 1e-3
    rect = span[0] * span[1]
    if slab and abs(_xy_hull_area(mesh) - rect) <= RECT_TOL * rect and _is_rectangle(mesh, rect):
        half = (float(0.5 * span[0]), float(0.5 * span[1]), float(0.5 * t))
        return ct.Primitive("box", half), float(rect), "cloth-box"
    sheet = top_layer(mesh) if slab and geo.is_closed(mesh) else mesh
    sha = meshes.write_canonical(out / cv.mesh_rel(spec), sheet, cv.header(spec))
    return ct.MeshSource(path=cv.mesh_rel(spec), sha256=sha, collision="convex"), _sheet_area(sheet), "cloth-mesh"


def _is_rectangle(mesh: geo.MeshData, rect: float) -> bool:
    """The upward area equals the bounding rectangle (no cut-outs between panels)."""
    top = top_layer(mesh) if geo.is_closed(mesh) else mesh
    return abs(_sheet_area(top) - rect) <= RECT_TOL * rect


def _cloth_elastic(spec: GarSpec, rec: cv.ImportRecord) -> ct.Elastic | None:
    """Cited moduli, else E from the membrane stretch stiffness (E = k / t, model) and a default Poisson ratio."""
    e = cv._elastic(spec)
    c = spec.cloth
    if e is not None or c is None:
        return e
    k = 0.5 * (c.stretch_warp.value + c.stretch_weft.value)
    youngs = Q(k / c.thickness.value, "model", None,
               f"E = mean warp/weft stretch stiffness {k:.4g} N/m / thickness {c.thickness.value:g} m "
               f"(stretch: {note_for(c.stretch_warp, spec.sim)})")
    rec.defaults.append("elastic.poisson")
    return ct.Elastic(youngs=youngs, poisson=Q(DEFAULT_POISSON, "default"), cloth=cloth_aniso(spec))


def cloth_aniso(spec: GarSpec) -> ct.ClothAniso | None:
    """The cited warp/weft/shear/bend constants as Alloy's ``ClothAniso``."""
    c = spec.cloth
    if c is None:
        return None
    names = ("stretch_warp", "stretch_weft", "shear", "bend_warp", "bend_weft")
    return ct.ClothAniso(*(q_from(getattr(c, n), spec.sim, n.replace("_", " ")) for n in names))


def convert_cloth(spec: GarSpec, out: Path) -> cv.ImportRecord:
    """A cloth spec -> material (with ``elastic.cloth``) + object (thin box or sheet mesh), written under ``out``."""
    if spec.cloth is None or spec.mesh is None:
        raise cv.ConvertError(f"{spec.card_name}: a cloth needs its cloth block and a sheet mesh")
    rec = cv.ImportRecord(spec.card_name, spec.sim, spec.key, spec.name, spec.category, spec.kind)
    mesh = meshes.weld(geo.read_mesh(spec.mesh))
    t = spec.cloth.thickness.value
    shape, area, rec.collision = _cloth_shape(spec, mesh, t, out)
    mat = _material(spec, area * t, f"sheet area {area:.4g} m^2 x thickness {t:g} m", rec)
    mat = ct.MaterialCard(name=mat.name, density=mat.density, surface=mat.surface, elastic=_cloth_elastic(spec, rec))
    thick = q_from(spec.cloth.thickness, spec.sim, "sheet thickness")
    obj = ct.ObjectCard(name=spec.card_name, shape=shape, material=spec.card_name, motion=spec.motion,  # type: ignore[arg-type]
                        features=ct.Features(thickness=thick, min_wall=thick))
    rec.volume_m3, rec.mass_kg = area * t, mat.density.value * area * t
    if isinstance(shape, ct.MeshSource):
        rec.mesh, rec.mesh_sha256 = shape.path, shape.sha256
    _write_cards(spec, mat, obj, out)
    return rec


def convert_cable(spec: GarSpec, out: Path) -> cv.ImportRecord:
    """A cable spec -> material + ``curve`` object (EI in ``features.bend_stiffness`` when cited)."""
    c = spec.cable
    if c is None:
        raise cv.ConvertError(f"{spec.card_name}: kind 'cable' needs the cable block")
    rec = cv.ImportRecord(spec.card_name, spec.sim, spec.key, spec.name, spec.category, spec.kind, collision="curve")
    pts = np.asarray(c.points, float)
    length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    r = c.radius.value
    mat = _material(spec, math.pi * r * r * length, f"pi r^2 L (r {r:g} m, L {length:.4g} m)", rec)
    mat = ct.MaterialCard(name=mat.name, density=mat.density, surface=mat.surface, elastic=_cable_elastic(spec, rec))
    ei = q_from(c.bend_stiffness, spec.sim, "bending stiffness EI") if c.bend_stiffness else None
    obj = ct.ObjectCard(name=spec.card_name, shape=ct.CurveSource(points=tuple(map(tuple, pts.tolist())), radius=r),
                        material=spec.card_name, motion=spec.motion,  # type: ignore[arg-type]
                        features=ct.Features(bend_stiffness=ei))
    rec.volume_m3 = math.pi * r * r * length
    rec.mass_kg = mat.density.value * rec.volume_m3
    _write_cards(spec, mat, obj, out)
    return rec


def _cable_elastic(spec: GarSpec, rec: cv.ImportRecord) -> ct.Elastic | None:
    """Cited moduli, else E from the cited EI over the solid section (E = 4 EI / (pi r^4), model)."""
    e = cv._elastic(spec)
    c = spec.cable
    if e is not None or c is None or c.bend_stiffness is None:
        if e is None:
            rec.warnings.append("no modulus and no EI cited: the cable card has no elastic block")
        return e
    r = c.radius.value
    youngs = Q(4.0 * c.bend_stiffness.value / (math.pi * r**4), "model", None,
               f"E = 4 EI / (pi r^4) from EI {c.bend_stiffness.value:g} N m^2 and r {r:g} m "
               f"(EI: {note_for(c.bend_stiffness, spec.sim)})")
    rec.defaults.append("elastic.poisson")
    return ct.Elastic(youngs=youngs, poisson=Q(DEFAULT_POISSON, "default"))


def _write_cards(spec: GarSpec, mat: ct.MaterialCard, obj: ct.ObjectCard, out: Path) -> None:
    cv._write(out / cv.material_rel(spec), dump_card(mat, header=cv.header(spec)))
    cv._write(out / cv.object_rel(spec), dump_card(obj, header=cv.header(spec)))


def convert_deformable(spec: GarSpec, out: Path) -> cv.ImportRecord:
    """Dispatch on ``spec.kind`` (cloth or cable)."""
    if spec.kind == "cloth":
        return convert_cloth(spec, out)
    if spec.kind == "cable":
        return convert_cable(spec, out)
    raise cv.ConvertError(f"{spec.card_name}: kind {spec.kind!r} is not a cloth or cable")
