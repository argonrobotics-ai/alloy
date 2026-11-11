# SPDX-License-Identifier: Apache-2.0
"""``gar.spec/1`` -> Alloy cards (``alloy.material/1`` + ``alloy.object/1``) and a canonical mesh, in a library root.

Output layout under ``out`` (a library root, usable as ``CardLibrary(roots=[default_root(), out])``; the shared
importer layout of ``alloy.importers.common``): ``materials/<sim>/<card>.yaml``, ``objects/<sim>/<card>.yaml``,
``meshes/<card>/physics.obj`` (+ ``physics.hulls.obj``, see ``hullgen``), with ``<card>`` = ``gar_<sim>_<key>``.

Mass: a cited object mass becomes an *effective* density (mass / mesh volume, source ``fit``), the library's rule
for whole-object catalogue masses (library/SOURCES.md), so the compiled body has the datasheet mass.

Example::

    >>> rec = convert_spec(load_spec("specs/wine_pouring/wine_glass.json"), Path("out"))   # doctest: +SKIP
    >>> rec.card, rec.collision, round(rec.mass_kg, 3)                                       # doctest: +SKIP
    ('gar_wine_pouring_wine_glass', 'sdf', 0.19)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from alloy.cards import geometry as geo
from alloy.cards import types as ct
from alloy.cards.loader import dump_card
from alloy.core.quantity import Q
from alloy.importers.common.cards import min_wall_note
from alloy.importers.gymanything import meshes
from alloy.importers.gymanything.provenance import (UPSTREAM, UPSTREAM_COMMIT, UPSTREAM_LICENCE, cite_ref, default_q,
                                                   note_for, q_from)
from alloy.importers.gymanything.spec import GarSpec

log = logging.getLogger("alloy.importers.gymanything")

RIGID_DT = 1.25e-3            # s, the rigid step the thin-member rule uses (alloy.cards.validate_object default)
CONTACT_SPEED = 0.5           # m/s, LoadEnvelope default
THIN_LIMIT = 2.0 * CONTACT_SPEED * RIGID_DT
DENSITY_RANGE = (10.0, 25_000.0)
DEFAULT_DENSITY, DEFAULT_MU, DEFAULT_RESTITUTION = 1000.0, 0.5, 0.3


class ConvertError(ValueError):
    """A spec cannot become a valid card; the message names the object and what to change."""


@dataclass
class ImportRecord:
    """One row of the import index (JSON-serialisable via ``as_dict``)."""

    card: str
    sim: str
    key: str
    name: str
    category: str
    kind: str
    collision: str = ""
    mass_kg: float | None = None
    volume_m3: float | None = None
    convex_ratio: float | None = None
    mass_basis: str = "none"
    friction_basis: str = "none"
    defaults: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    mesh: str = ""
    mesh_sha256: str = ""
    n_hulls: int | None = None
    hull_volume_ratio: float | None = None
    sourced: list[str] = field(default_factory=list)      # fields filled from sourcing records (``sourced``)
    visual_parts: int = 0                                  # render parts written to visual.glb
    visual_bbox_ok: bool | None = None                     # the parts span the card mesh's bounding box

    def as_dict(self) -> dict:
        return asdict(self)


def mesh_rel(spec: GarSpec) -> str:
    """Mesh path inside the library root (``meshes/<card>/physics.obj``)."""
    return f"meshes/{spec.card_name}/physics.obj"


def material_rel(spec: GarSpec) -> str:
    return f"materials/{spec.sim}/{spec.card_name}.yaml"


def object_rel(spec: GarSpec) -> str:
    return f"objects/{spec.sim}/{spec.card_name}.yaml"


def header(spec: GarSpec) -> str:
    """Card/mesh header: what, where from, licence."""
    ref = cite_ref(spec.sim, spec.source_ref) if spec.source_ref else f"{UPSTREAM}@{UPSTREAM_COMMIT}"
    return (f"{spec.card_name}: {spec.name} (Gym-Anything for Robotics, sim '{spec.sim}')\n"
            f"geometry and values exported from {ref}\n"
            f"licence: {spec.licence} ({UPSTREAM_LICENCE}); see LICENSE-gym-anything in this folder")


def _density(spec: GarSpec, volume: float, rec: ImportRecord) -> Q:
    if spec.mass is not None:
        rec.mass_basis = spec.mass.basis
        what = (f"effective: {spec.mass.value:g} kg object mass over the {volume:.6g} m^3 imported mesh volume "
                f"(material {spec.material.name})")
        return Q(spec.mass.value / volume, "fit", _scaled_unc(spec, volume), note_for(spec.mass, spec.sim, what))
    if spec.material.density is not None:
        rec.mass_basis = f"material density ({spec.material.density.basis})"
        rec.warnings.append("no object mass cited: mass = material density x solid mesh volume")
        return q_from(spec.material.density, spec.sim, f"{spec.material.name} bulk density (solid-object assumption)")
    rec.defaults.append("density")
    return default_q(DEFAULT_DENSITY)


def _scaled_unc(spec: GarSpec, volume: float) -> tuple[float, float] | None:
    u = spec.mass.unc if spec.mass else None
    return (u[0] / volume, u[1] / volume) if u else None


def _surface(spec: GarSpec, rec: ImportRecord) -> ct.Surface:
    if spec.friction is not None:
        rec.friction_basis = spec.friction.basis
        what = "Coulomb coefficient used for both static and dynamic in their sim"
        if spec.friction_partner:
            what += f", against {spec.friction_partner}"
        mu = q_from(spec.friction, spec.sim, what)
        mu_s = mu_d = mu
    else:
        rec.defaults.append("surface.mu")
        mu_s = mu_d = default_q(DEFAULT_MU)
    if spec.restitution is not None:
        e = q_from(spec.restitution, spec.sim, "coefficient of restitution")
    else:
        rec.defaults.append("surface.restitution")
        e = default_q(DEFAULT_RESTITUTION)
    return ct.Surface(mu_static=mu_s, mu_dynamic=mu_d, restitution=e)


def _elastic(spec: GarSpec) -> ct.Elastic | None:
    m = spec.material
    if spec.kind == "rigid" or m.youngs is None or m.poisson is None:
        return None
    cloth = None
    if spec.cloth is not None:
        c = spec.cloth
        cloth = ct.ClothAniso(*(q_from(getattr(c, n), spec.sim, n.replace("_", " "))
                                for n in ("stretch_warp", "stretch_weft", "shear", "bend_warp", "bend_weft")))
    return ct.Elastic(youngs=q_from(m.youngs, spec.sim, f"{m.name} Young's modulus"),
                      poisson=q_from(m.poisson, spec.sim, f"{m.name} Poisson ratio"), cloth=cloth)


def material_card(spec: GarSpec, volume: float, rec: ImportRecord) -> ct.MaterialCard:
    """The object's own material card (effective density, cited friction/restitution, moduli for deformables)."""
    if volume <= 0:
        raise ConvertError(f"{spec.card_name}: mesh volume {volume} m^3; export a closed or solid mesh")
    rho = _density(spec, volume, rec)
    if not DENSITY_RANGE[0] <= rho.value <= DENSITY_RANGE[1]:
        raise ConvertError(f"{spec.card_name}: density {rho.value:.4g} kg/m^3 outside {DENSITY_RANGE} "
                           f"(mass {spec.mass.value if spec.mass else None} kg over {volume:.4g} m^3): the mesh "
                           "volume does not match the object (export the walls, not the closed outline, for a "
                           "hollow object)")
    return ct.MaterialCard(name=spec.card_name, density=rho, surface=_surface(spec, rec), elastic=_elastic(spec))


def _features(spec: GarSpec, facts: meshes.MeshFacts, rec: ImportRecord) -> ct.Features:
    thickness = q_from(spec.cloth.thickness, spec.sim, "sheet thickness") if spec.cloth else None
    wall = facts.wall_probe if spec.thickness else facts.min_wall    # a cited wall confirms the probe (needles)
    if wall >= THIN_LIMIT and thickness is None:
        return ct.Features()
    min_wall = None
    if wall < THIN_LIMIT:
        cited = f"; cited wall {spec.thickness.value:g} m ({cite_ref(spec.sim, spec.thickness.cite)})" \
            if spec.thickness else ""
        note = f"thinnest wall ray-cast on the imported mesh{cited}" if wall == facts.wall_probe else \
            min_wall_note(wall, facts.wall_probe)                   # re-measured: the probe read a mesh defect
        min_wall = Q(wall, "model", None, note)
        rec.warnings.append(f"thin wall {wall * 1e3:.2f} mm < {THIN_LIMIT * 1e3:.2f} mm: min_wall set")
    return ct.Features(thickness=thickness, min_wall=min_wall)


def object_card(spec: GarSpec, facts: meshes.MeshFacts, sha: str, rec: ImportRecord) -> ct.ObjectCard:
    """The object card: mesh shape (convex hull or SDF collision), its own material, cited features."""
    shape = ct.MeshSource(path=mesh_rel(spec), sha256=sha, collision=facts.collision)  # type: ignore[arg-type]
    return ct.ObjectCard(name=spec.card_name, shape=shape, material=spec.card_name,
                         motion=spec.motion, features=_features(spec, facts, rec))  # type: ignore[arg-type]


def convert_spec(spec: GarSpec, out: Path) -> ImportRecord:
    """Write the mesh, material and object card of one rigid spec under ``out``; returns its index row.

    Raises ``ConvertError`` / ``meshes.MeshError`` with what to change in the exporter.
    """
    if spec.kind not in ("rigid", "soft") or spec.mesh is None:
        raise ConvertError(f"{spec.card_name}: kind {spec.kind!r} is not converted by this importer yet")
    rec = ImportRecord(spec.card_name, spec.sim, spec.key, spec.name, spec.category, spec.kind)
    mesh = meshes.prepare(geo.read_mesh(spec.mesh))
    if not geo.is_watertight(mesh):
        mesh = meshes.convex_hull_mesh(mesh)          # a closed shape, so the card's mass is the cited mass
        rec.warnings.append("mesh is not watertight: replaced by its convex hull (export a closed mesh to keep "
                            "the shape)")
    facts = meshes.measure(mesh, THIN_LIMIT)
    rec.collision, rec.volume_m3, rec.convex_ratio = facts.collision, facts.volume, facts.convex_ratio
    mat = material_card(spec, facts.volume, rec)
    sha = meshes.write_canonical(out / mesh_rel(spec), mesh, header(spec))
    obj = object_card(spec, facts, sha, rec)
    rec.mesh, rec.mesh_sha256 = mesh_rel(spec), sha
    rec.mass_kg = mat.density.value * facts.volume
    _write(out / material_rel(spec), dump_card(mat, header=header(spec)))
    _write(out / object_rel(spec), dump_card(obj, header=header(spec)))
    log.info("imported %s: %s collision, %.4g kg, %s", spec.card_name, facts.collision, rec.mass_kg, rec.defaults)
    return rec


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
